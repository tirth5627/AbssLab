"""
Dataset Loader — Custom Scraped Dataset Reader.

Loads face images from the organized directory structure,
pairs them with demographic metadata, and provides structured
access for the evaluation pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and organizes the scraped facial dataset.

    Reads images from data/raw/{variant}/ directories and
    pairs them with demographic metadata from metadata.json.

    Attributes:
        data_dir: Base data directory path.
        metadata: List of subject metadata dictionaries.
    """

    VARIANTS = ["REAL", "GAUSSIAN", "SALT_PEPPER"]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata: List[Dict] = []
        self._metadata_lookup: Dict[str, Dict] = {}

    def load_metadata(self) -> List[Dict]:
        """
        Load demographic metadata from metadata.json.

        Returns:
            List of subject metadata dictionaries.

        Raises:
            FileNotFoundError: If metadata.json does not exist.
        """
        metadata_file = self.metadata_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build lookup by subject_id
        self._metadata_lookup = {
            m["subject_id"]: m for m in self.metadata
        }

        logger.info(f"Loaded metadata for {len(self.metadata)} subjects")
        return self.metadata

    def get_subject_metadata(self, subject_id: str) -> Optional[Dict]:
        """
        Get demographic metadata for a specific subject.

        Args:
            subject_id: Subject identifier (e.g., 'S0001').

        Returns:
            Metadata dictionary or None if not found.
        """
        return self._metadata_lookup.get(subject_id)

    def load_image(
        self, subject_id: str, image_idx: int = 1, variant: str = "REAL"
    ) -> Optional[np.ndarray]:
        """
        Load a single image by subject ID, image index, and variant.

        Args:
            subject_id: Subject identifier (e.g., 'S0001').
            image_idx: Image index (1 or 2).
            variant: Image variant ('REAL', 'GAUSSIAN', 'SALT_PEPPER').

        Returns:
            Image as numpy array (H, W, C) in RGB, or None on error.
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Use: {self.VARIANTS}")

        filename = f"{subject_id}_{image_idx:03d}.jpg"
        filepath = self.raw_dir / variant / filename

        if not filepath.exists():
            logger.warning(f"Image not found: {filepath}")
            return None

        img = cv2.imread(str(filepath))
        if img is None:
            logger.warning(f"Could not read image: {filepath}")
            return None

        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_subject_images(
        self, subject_id: str, variant: str = "REAL"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load both images for a subject.

        Args:
            subject_id: Subject identifier.
            variant: Image variant.

        Returns:
            Tuple of (image_1, image_2) as numpy arrays, None for missing.
        """
        img1 = self.load_image(subject_id, 1, variant)
        img2 = self.load_image(subject_id, 2, variant)
        return img1, img2

    def get_all_subject_ids(self) -> List[str]:
        """
        Get a sorted list of all subject IDs from metadata.

        Returns:
            List of subject ID strings.
        """
        return sorted(self._metadata_lookup.keys())

    def get_variant_dir(self, variant: str) -> Path:
        """
        Get the directory path for a specific image variant.

        Args:
            variant: Image variant name.

        Returns:
            Path to the variant directory.
        """
        return self.raw_dir / variant

    def count_images(self, variant: str = "REAL") -> int:
        """
        Count the number of images in a variant directory.

        Args:
            variant: Image variant name.

        Returns:
            Number of JPEG images found.
        """
        variant_dir = self.raw_dir / variant
        if not variant_dir.exists():
            return 0
        return len(list(variant_dir.glob("*.jpg")))
