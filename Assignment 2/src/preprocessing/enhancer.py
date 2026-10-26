"""
Image Enhancer — CLAHE, Resizing, and Normalization.

Provides preprocessing utilities for standardizing face images
before feature extraction.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Image preprocessing and enhancement pipeline.

    Applies CLAHE contrast enhancement, resizing, color space conversion,
    and normalization to prepare images for feature extraction.

    Attributes:
        target_size: Target dimensions (H, W) for resizing.
        clip_limit: CLAHE clip limit for contrast enhancement.
        tile_grid_size: CLAHE tile grid dimensions.
        normalize: Whether to normalize pixel values to [0, 1].
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.normalize = normalize

        # Create CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.

        Args:
            image: Input image as numpy array.

        Returns:
            Resized image.
        """
        return cv2.resize(
            image,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Converts to LAB color space, applies CLAHE to the L channel,
        and converts back to RGB. This enhances contrast while
        preserving color information.

        Args:
            image: Input RGB image as uint8 numpy array.

        Returns:
            Contrast-enhanced RGB image.
        """
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        l_channel = lab[:, :, 0]
        lab[:, :, 0] = self._clahe.apply(l_channel)

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale.

        Args:
            image: Input RGB image.

        Returns:
            Grayscale image with single channel.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] float range.

        Args:
            image: Input uint8 image.

        Returns:
            Normalized float64 image in [0, 1].
        """
        return image.astype(np.float64) / 255.0

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline.

        Steps:
        1. Resize to target dimensions
        2. Apply CLAHE contrast enhancement
        3. Optionally normalize to [0, 1]

        Args:
            image: Input RGB image as uint8 numpy array.

        Returns:
            Preprocessed image (float64 if normalized, uint8 otherwise).
        """
        # Step 1: Resize
        processed = self.resize(image)

        # Step 2: CLAHE
        processed = self.apply_clahe(processed)

        # Step 3: Normalize
        if self.normalize:
            processed = self.normalize_image(processed)

        return processed
