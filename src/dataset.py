"""Load ImageFolder data, create a class-stratified validation split, and apply image transforms."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 123
NUM_WORKERS = 4

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


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


def _stratified_split_indices(
    targets: list[int],
    num_classes: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a seeded, class-stratified split with exactly the requested number of validation images."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not targets:
        raise ValueError("Cannot split an empty dataset.")

    validation_size = int(validation_fraction * len(targets))
    if validation_size == 0 or validation_size == len(targets):
        raise ValueError("The dataset must contain enough images for both splits.")

    class_indices = [
        [index for index, target in enumerate(targets) if target == class_index]
        for class_index in range(num_classes)
    ]
    quotas = [validation_fraction * len(indices) for indices in class_indices]
    validation_counts = [int(quota) for quota in quotas]
    remainder = validation_size - sum(validation_counts)

    # Largest-remainder allocation preserves the requested total while keeping
    # each class's validation share as close as possible to the target fraction.
    remainder_order = sorted(
        range(num_classes),
        key=lambda index: (-(quotas[index] - validation_counts[index]), index),
    )
    for class_index in remainder_order[:remainder]:
        validation_counts[class_index] += 1

    generator = torch.Generator().manual_seed(seed)
    training_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index, indices in enumerate(class_indices):
        if not indices:
            raise ValueError(f"Class index {class_index} has no images.")
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[position] for position in permutation]
        count = validation_counts[class_index]
        validation_indices.extend(shuffled_indices[:count])
        training_indices.extend(shuffled_indices[count:])

    return training_indices, validation_indices


def build_data_loaders(
    data_root: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    validation_fraction: float = VALIDATION_FRACTION,
    split_seed: int = SPLIT_SEED,
):
    """Build training and validation loaders, augmenting training images only."""
    train_images = resolve_train_images(data_root)
    training_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    metadata_dataset = datasets.ImageFolder(str(train_images))
    if len(metadata_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class folders, found "
            f"{len(metadata_dataset.classes)}: {metadata_dataset.classes}"
        )

    training_dataset = datasets.ImageFolder(
        str(train_images), transform=training_transform
    )
    validation_dataset = datasets.ImageFolder(
        str(train_images), transform=validation_transform
    )
    if training_dataset.class_to_idx != metadata_dataset.class_to_idx:
        raise ValueError("Training class mapping does not match the dataset mapping.")
    if validation_dataset.class_to_idx != metadata_dataset.class_to_idx:
        raise ValueError("Validation class mapping does not match the dataset mapping.")

    training_indices, validation_indices = _stratified_split_indices(
        metadata_dataset.targets,
        len(metadata_dataset.classes),
        validation_fraction,
        split_seed,
    )
    training_subset = Subset(training_dataset, training_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    training_loader = DataLoader(
        training_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return metadata_dataset, training_loader, validation_loader
