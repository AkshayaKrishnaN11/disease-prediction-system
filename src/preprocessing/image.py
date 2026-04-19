"""
Image preprocessing and augmentation pipeline for chest X-ray classification.
Uses OpenCV + Albumentations for augmentation, creates PyTorch DataLoaders.
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_DIR, CNN_PARAMS


# ──────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────
def get_train_transforms(image_size: int = None):
    """Training augmentation pipeline: flip, rotate, brightness, noise."""
    size = image_size or CNN_PARAMS["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = None):
    """Validation/test pipeline: resize + normalize only."""
    size = image_size or CNN_PARAMS["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Chest X-Ray dataset for pneumonia classification.
    Expects folder structure:
        chest_xray/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    """

    CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []

        for class_name, label in self.CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"  ⚠ Directory not found: {class_dir}")
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    self.images.append(str(img_path))
                    self.labels.append(label)

        print(f"  Loaded {len(self.images)} images from {split} " +
              f"(NORMAL: {self.labels.count(0)}, PNEUMONIA: {self.labels.count(1)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image with OpenCV (BGR → RGB)
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if reading fails
            image = np.zeros(
                (CNN_PARAMS["image_size"], CNN_PARAMS["image_size"], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────
def get_xray_dataloaders(
    data_dir: str = None,
    batch_size: int = None,
    image_size: int = None,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for chest X-ray data.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = data_dir or str(DATA_DIR / "chest_xray")
    batch_size = batch_size or CNN_PARAMS["batch_size"]
    image_size = image_size or CNN_PARAMS["image_size"]

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = ChestXRayDataset(data_dir, split="train", transform=train_transform)
    val_dataset = ChestXRayDataset(data_dir, split="val", transform=val_transform)
    test_dataset = ChestXRayDataset(data_dir, split="test", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
