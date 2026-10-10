"""ImageFolder dataset and deterministic train/validation split."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 123
NUM_WORKERS = 4


class EncodedDataset(Dataset):
    """Wrap a dataset and retain the notebook's one-hot float labels."""

    def __init__(self, dataset: Dataset, num_classes: int = NUM_CLASSES) -> None:
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        features, label = self.dataset[idx]
        one_hot = F.one_hot(
            torch.tensor(label), num_classes=self.num_classes
        ).float()
        return features, one_hot


def resolve_train_images(data_root: str | Path) -> Path:
    """Accept either the competition root or its train_images directory."""
    root = Path(data_root)
    train_images = root if root.name == "train_images" else root / "train_images"
    if not train_images.is_dir():
        raise FileNotFoundError(
            f"Expected the labeled ImageFolder data at '{train_images}'. "
            "Pass the competition data root containing train_images/."
        )
    return train_images


def build_data_loaders(
    data_root: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    validation_fraction: float = VALIDATION_FRACTION,
    split_seed: int = SPLIT_SEED,
):
    """Create the notebook-equivalent ImageFolder split and data loaders."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    train_images = resolve_train_images(data_root)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ]
    )
    full_dataset = datasets.ImageFolder(str(train_images), transform=transform)

    if len(full_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class folders, found "
            f"{len(full_dataset.classes)}: {full_dataset.classes}"
        )

    validation_size = int(validation_fraction * len(full_dataset))
    training_size = len(full_dataset) - validation_size
    if training_size == 0 or validation_size == 0:
        raise ValueError(
            "The dataset must contain enough images to create both "
            "training and validation subsets."
        )

    generator = torch.Generator().manual_seed(split_seed)
    training_subset, validation_subset = random_split(
        full_dataset, [training_size, validation_size], generator=generator
    )

    training_dataset = EncodedDataset(training_subset, len(full_dataset.classes))
    validation_dataset = EncodedDataset(validation_subset, len(full_dataset.classes))

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return full_dataset, training_loader, validation_loader
