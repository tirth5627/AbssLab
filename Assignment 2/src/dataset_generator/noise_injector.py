"""
Noise Injector — Gaussian & Salt-and-Pepper Noise Applier.

Generates corrupted image variants from clean (REAL) images
to simulate sensor noise and transmission errors.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NoiseInjector:
    """
    Applies synthetic noise corruption to clean face images.

    Generates GAUSSIAN and SALT_PEPPER noise variants from the
    REAL (clean) image directory, saving them in parallel directories.

    Attributes:
        gaussian_mean: Mean of Gaussian noise distribution.
        gaussian_sigma: Standard deviation of Gaussian noise.
        sp_amount: Fraction of pixels affected by salt-and-pepper noise.
        sp_ratio: Ratio of salt (white) to pepper (black) pixels.
    """

    def __init__(
        self,
        gaussian_mean: float = 0,
        gaussian_sigma: float = 25,
        sp_amount: float = 0.05,
        sp_ratio: float = 0.5,
    ):
        self.gaussian_mean = gaussian_mean
        self.gaussian_sigma = gaussian_sigma
        self.sp_amount = sp_amount
        self.sp_ratio = sp_ratio

    def apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply additive Gaussian noise to an image.

        Simulates sensor noise typically encountered in low-light
        or high-ISO imaging conditions.

        Args:
            image: Input image as uint8 numpy array (H, W, C).

        Returns:
            Noisy image as uint8 numpy array, clipped to [0, 255].
        """
        noise = np.random.normal(
            self.gaussian_mean, self.gaussian_sigma, image.shape
        ).astype(np.float64)

        noisy = image.astype(np.float64) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def apply_salt_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply salt-and-pepper noise to an image.

        Simulates transmission errors or dead/hot pixels in
        imaging sensors.

        Args:
            image: Input image as uint8 numpy array (H, W, C).

        Returns:
            Noisy image as uint8 numpy array with salt (255) and
            pepper (0) pixel corruptions.
        """
        noisy = image.copy()
        h, w = image.shape[:2]
        num_pixels = h * w

        # Salt (white) pixels
        num_salt = int(num_pixels * self.sp_amount * self.sp_ratio)
        salt_coords = (
            np.random.randint(0, h, num_salt),
            np.random.randint(0, w, num_salt),
        )
        noisy[salt_coords[0], salt_coords[1]] = 255

        # Pepper (black) pixels
        num_pepper = int(num_pixels * self.sp_amount * (1 - self.sp_ratio))
        pepper_coords = (
            np.random.randint(0, h, num_pepper),
            np.random.randint(0, w, num_pepper),
        )
        noisy[pepper_coords[0], pepper_coords[1]] = 0

        return noisy

    def generate_variants(
        self,
        real_dir: str = "data/raw/REAL",
        output_base: str = "data/raw",
    ) -> dict:
        """
        Generate noise-corrupted variants for all images in the REAL directory.

        Reads each image from real_dir, applies Gaussian and Salt-and-Pepper
        noise, and saves the results in parallel GAUSSIAN/ and SALT_PEPPER/
        directories.

        Args:
            real_dir: Path to clean (REAL) images directory.
            output_base: Base path for noise variant directories.

        Returns:
            Dictionary with generation statistics.
        """
        real_path = Path(real_dir)
        gaussian_path = Path(output_base) / "GAUSSIAN"
        sp_path = Path(output_base) / "SALT_PEPPER"

        gaussian_path.mkdir(parents=True, exist_ok=True)
        sp_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(real_path.glob("*.jpg"))
        if not image_files:
            logger.warning(f"No JPEG images found in {real_dir}")
            return {"gaussian_generated": 0, "salt_pepper_generated": 0}

        logger.info(
            f"Generating noise variants for {len(image_files)} images..."
        )
        logger.info(
            f"  Gaussian: mean={self.gaussian_mean}, sigma={self.gaussian_sigma}"
        )
        logger.info(
            f"  Salt & Pepper: amount={self.sp_amount}, ratio={self.sp_ratio}"
        )

        stats = {"gaussian_generated": 0, "salt_pepper_generated": 0}

        for img_file in tqdm(image_files, desc="Generating noise variants"):
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Could not read: {img_file}")
                continue

            # Convert BGR (OpenCV) to RGB for processing, then back
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Gaussian variant
            gaussian_img = self.apply_gaussian_noise(img_rgb)
            gaussian_bgr = cv2.cvtColor(gaussian_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(gaussian_path / img_file.name), gaussian_bgr)
            stats["gaussian_generated"] += 1

            # Salt & Pepper variant
            sp_img = self.apply_salt_pepper_noise(img_rgb)
            sp_bgr = cv2.cvtColor(sp_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(sp_path / img_file.name), sp_bgr)
            stats["salt_pepper_generated"] += 1

        logger.info(
            f"Noise generation complete: "
            f"{stats['gaussian_generated']} Gaussian, "
            f"{stats['salt_pepper_generated']} Salt & Pepper"
        )

        return stats
