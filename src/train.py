"""Training entry point for the custom DenseNet model."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim

from src.dataset import (
    BATCH_SIZE,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_WORKERS,
    build_data_loaders,
)
from src.evaluate import evaluate_model
from src.model import DenseNetB121

EPOCHS = 10
ADAM_EPSILON = 0.05


def train(
    data_root: Path,
    output_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict[str, list[float]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    full_dataset, training_loader, validation_loader = build_data_loaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetB121(
        input_shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        num_classes=len(full_dataset.classes),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), eps=ADAM_EPSILON)
    loss_function = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started...")
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for features, labels in training_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_now = features.size(0)
            predictions = outputs.argmax(dim=1)
            true_labels = labels.argmax(dim=1)
            running_loss += loss.item() * batch_size_now
            running_correct += (predictions == true_labels).sum().item()
            running_total += batch_size_now

        if running_total == 0:
            raise ValueError("The training data loader produced no batches.")

        train_loss = running_loss / running_total
        train_accuracy = running_correct / running_total
        validation_metrics = evaluate_model(
            model, validation_loader, loss_function, device
        )

        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(validation_metrics["loss"])
        history["val_accuracy"].append(validation_metrics["accuracy"])

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{epochs} finished in {elapsed:.1f}s | "
            f"loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - "
            f"val_loss: {validation_metrics['loss']:.4f} - "
            f"val_accuracy: {validation_metrics['accuracy']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "densenetb121_paddy.pth")
    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(full_dataset.class_to_idx, file, indent=2)

    print(f"Model weights and training metadata saved to {output_dir}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the custom DenseNet rice leaf classifier."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/paddy-disease-classification"),
        help="Competition data root or its train_images directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
