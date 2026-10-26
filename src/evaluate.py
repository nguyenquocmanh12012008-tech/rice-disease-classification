"""Evaluate the custom DenseNet on the validation split."""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.dataset import (
    BATCH_SIZE,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_WORKERS,
    build_data_loaders,
)
from src.model import DenseNetB121


def evaluate_model(
    model: nn.Module,
    data_loader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Compute sample-weighted loss and accuracy without tracking gradients."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(features)
            loss = loss_function(outputs, labels)
            batch_size = features.size(0)
            predictions = outputs.argmax(dim=1)
            true_labels = labels

            running_loss += loss.item() * batch_size
            running_correct += (predictions == true_labels).sum().item()
            running_total += batch_size

    if running_total == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return {
        "loss": running_loss / running_total,
        "accuracy": running_correct / running_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the rice leaf model.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/paddy-disease-classification"),
        help="Competition data root or its train_images directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/densenetb121_paddy.pth"),
        help="Path to the trained state-dict checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    full_dataset, _, validation_loader = build_data_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetB121(
        input_shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        num_classes=len(full_dataset.classes),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = evaluate_model(model, validation_loader, nn.CrossEntropyLoss(), device)
    print(
        f"Validation loss: {metrics['loss']:.4f} | "
        f"Validation accuracy: {metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
