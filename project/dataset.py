import os
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np


def find_data_dir() -> Path:
    """Locate the data directory regardless of execution working directory."""
    candidates = [
        Path("data"),
        Path("../data"),
        Path("../../data"),
        Path("/home/user/Puente/vision/data")
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()
    raise FileNotFoundError("Could not find the 'data' directory containing 'train' and 'test'.")


def load_split(
    split_name: str,
    target_size: Optional[Tuple[int, int]] = None,
    to_gray: bool = False,
    max_samples_per_class: Optional[int] = None,
    allowed_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> Tuple[List[np.ndarray], List[str], List[str], List[Path]]:
    """
    Load images and labels from a split folder ('train' or 'test').
    Dynamically discovers all class subfolders and correctly handles .jpg, .jpeg, and .png.

    Args:
        split_name: 'train' or 'test'
        target_size: Optional (width, height) to resize images
        to_gray: If True, converts images to grayscale
        max_samples_per_class: Optional cap on images per class
        allowed_extensions: Tuple of lower-cased valid image extensions

    Returns:
        images: List of loaded images (numpy ndarrays)
        labels: List of string class names corresponding to each image
        classes: Sorted list of unique class names discovered
        file_paths: List of Path objects for each loaded image
    """
    data_dir = find_data_dir()
    split_dir = data_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    images = []
    labels = []
    file_paths = []

    for cls_name in classes:
        cls_folder = split_dir / cls_name
        files = sorted([
            f for f in cls_folder.iterdir()
            if f.is_file() and f.suffix.lower() in allowed_extensions
        ])

        if max_samples_per_class is not None:
            files = files[:max_samples_per_class]

        for file_path in files:
            img = cv2.imread(str(file_path))
            if img is None:
                continue

            if to_gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if target_size is not None:
                img = cv2.resize(img, target_size)

            images.append(img)
            labels.append(cls_name)
            file_paths.append(file_path)

    return images, labels, classes, file_paths


if __name__ == "__main__":
    train_imgs, train_lbls, classes, _ = load_split("train")
    test_imgs, test_lbls, _, _ = load_split("test")
    print(f"Discovered Classes: {classes}")
    print(f"Loaded Train: {len(train_imgs)} images")
    print(f"Loaded Test:  {len(test_imgs)} images")
    for cls in classes:
        print(f"  - {cls}: {train_lbls.count(cls)} train, {test_lbls.count(cls)} test")
