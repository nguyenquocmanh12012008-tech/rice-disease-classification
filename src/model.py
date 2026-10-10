"""Custom DenseNet-B121-style model used by the original notebook."""

import torch
from torch import nn


class BottleneckLayer(nn.Module):
    """Append growth_rate new feature maps to the existing feature maps."""

    def __init__(self, in_channels: int, filters: int, strides: int = 1) -> None:
        super().__init__()
        self.skip_connection = None

        # Preserve the notebook's bottleneck widths: 4 * filters, then filters.
        self.conv1 = nn.Conv2d(
            in_channels, filters * 4, kernel_size=1, stride=strides, padding=0
        )
        self.bn1 = nn.BatchNorm2d(filters * 4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            filters * 4, filters, kernel_size=3, stride=strides, padding=1
        )
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.skip_connection = x
        new_features = self.relu1(self.bn1(self.conv1(x)))
        new_features = self.relu2(self.bn2(self.conv2(new_features)))

        # Keep the notebook's concatenation order: new features, then input.
        return torch.cat((new_features, self.skip_connection), dim=1)


class TransitionLayer(nn.Module):
    """Normalize, halve channels, and downsample between dense blocks."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = in_channels // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.pool(x)


class DenseBlock(nn.Module):
    """Apply bottleneck layers, concatenating each layer's features."""

    def __init__(
        self, in_channels: int, repetition: int = 1, growth_rate: int = 32
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels

        for _ in range(repetition):
            self.layers.append(BottleneckLayer(channels, growth_rate))
            channels += growth_rate

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNetB121(nn.Module):
    """Custom DenseNet variant defined by the project's original notebook."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        growth_rate: int = 32,
    ) -> None:
        super().__init__()
        in_channels = input_shape[0]

        # Preserve the notebook's initial BatchNorm -> ReLU -> convolution order.
        self.bn_init = nn.BatchNorm2d(in_channels)
        self.relu_init = nn.ReLU()
        self.conv_init = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3
        )
        self.pool_init = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        channels = 64

        self.dense1 = DenseBlock(channels, 6, growth_rate)
        channels = self.dense1.out_channels
        self.trans1 = TransitionLayer(channels)
        channels = self.trans1.out_channels

        self.dense2 = DenseBlock(channels, 12, growth_rate)
        channels = self.dense2.out_channels
        self.trans2 = TransitionLayer(channels)
        channels = self.trans2.out_channels

        self.dense3 = DenseBlock(channels, 24, growth_rate)
        channels = self.dense3.out_channels
        self.trans3 = TransitionLayer(channels)
        channels = self.trans3.out_channels

        self.dense4 = DenseBlock(channels, 16, growth_rate)
        channels = self.dense4.out_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_init(x)
        x = self.relu_init(x)
        x = self.conv_init(x)
        x = self.pool_init(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)  # Raw logits; CrossEntropyLoss applies log-softmax.
