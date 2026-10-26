"""
Dataset Loader and Split Generator for the AT&T / ORL Face Database.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


def natural_sort_key(s: str):
    """Sort strings containing numbers naturally (e.g. s1, s2, ..., s10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_orl_dataset(data_dir: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load all 400 grayscale PGM images from the AT&T / ORL dataset.

    Args:
        data_dir: Directory containing subject folders s1...s40.
        normalize: If True, scale pixel values to [0.0, 1.0].

    Returns:
        X: Feature matrix of shape (400, 10304).
        y: Identity labels array of shape (400,) in range [0, 39].
        metadata: Dictionary containing dataset dimensions and metadata.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Discover subject folders
    subject_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("s") and d[1:].isdigit()
    ]
    subject_dirs.sort(key=natural_sort_key)

    if len(subject_dirs) != 40:
        raise ValueError(f"Expected 40 subject directories, found {len(subject_dirs)} in {data_dir}")

    images = []
    labels = []
    file_paths = []

    for subject_idx, sub_dir in enumerate(subject_dirs):
        sub_path = os.path.join(data_dir, sub_dir)
        img_files = [f for f in os.listdir(sub_path) if f.lower().endswith(".pgm")]
        img_files.sort(key=natural_sort_key)

        if len(img_files) != 10:
            raise ValueError(f"Subject {sub_dir} contains {len(img_files)} images, expected 10.")

        for img_file in img_files:
            file_path = os.path.join(sub_path, img_file)
            with Image.open(file_path) as img:
                w, h = img.size
                if (w, h) != (92, 112):
                    raise ValueError(f"Image {file_path} has size {(w, h)}, expected (92, 112).")
                arr = np.array(img, dtype=np.float32)
                if normalize:
                    arr = arr / 255.0
                images.append(arr.flatten())
                labels.append(subject_idx)
                file_paths.append(file_path)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    metadata = {
        "dataset_name": "AT&T / ORL Database of Faces",
        "num_identities": 40,
        "images_per_identity": 10,
        "total_images": 400,
        "image_width": 92,
        "image_height": 112,
        "flattened_features": 10304,
        "pixel_min": float(X.min()),
        "pixel_max": float(X.max()),
        "file_paths": file_paths,
        "subject_names": subject_dirs,
    }

    return X, y, metadata


def generate_stratified_splits(
    y: np.ndarray,
    n_splits: int = 10,
    train_per_class: int = 5,
    test_per_class: int = 5,
    seeds: Optional[List[int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate repeated stratified train/test splits.
    For each identity, exactly `train_per_class` images are placed in train
    and `test_per_class` in test.

    Args:
        y: Identity labels array of shape (N,).
        n_splits: Number of random split repetitions.
        train_per_class: Number of train images per identity (default 5).
        test_per_class: Number of test images per identity (default 5).
        seeds: List of random seeds for each split repetition.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    if seeds is None:
        seeds = [42 + i * 37 for i in range(n_splits)]
    elif len(seeds) != n_splits:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_splits ({n_splits}).")

    unique_classes = np.unique(y)
    splits = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        train_idx = []
        test_idx = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) < (train_per_class + test_per_class):
                raise ValueError(
                    f"Class {cls} has only {len(cls_indices)} samples, "
                    f"cannot split into {train_per_class} train and {test_per_class} test."
                )
            shuffled = rng.permutation(cls_indices)
            train_idx.extend(shuffled[:train_per_class])
            test_idx.extend(shuffled[train_per_class:train_per_class + test_per_class])

        splits.append((np.array(train_idx, dtype=np.int32), np.array(test_idx, dtype=np.int32)))

    return splits
