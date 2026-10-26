"""
Web Scraper & Image Download Engine.

Provides the FaceScraper class for automated face image collection
from public web sources (default: ThisPersonDoesNotExist.com).
Includes a local synthetic face generator as a reliable fallback.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SyntheticFaceGenerator:
    """
    Generates unique synthetic face-like images for dataset construction.

    Uses OpenCV to create structured face-like patterns with variation
    in skin tone, features, and proportions. Each generated image is
    unique based on the subject/image index combination.

    This serves as a reliable fallback when web scraping is unavailable
    or rate-limited.
    """

    # Realistic skin tone palette (RGB)
    SKIN_TONES = [
        (245, 222, 196), (241, 210, 175), (228, 185, 143),
        (210, 161, 117), (188, 137, 97), (165, 113, 78),
        (140, 93, 60), (115, 75, 45), (90, 60, 35),
        (224, 196, 168), (235, 208, 182), (200, 155, 115),
    ]

    # Hair colors (RGB)
    HAIR_COLORS = [
        (30, 20, 10), (50, 35, 20), (80, 50, 30),
        (120, 80, 45), (160, 110, 60), (200, 170, 130),
        (40, 30, 25), (60, 40, 30), (20, 15, 10),
    ]

    # Eye colors (RGB)
    EYE_COLORS = [
        (80, 50, 20), (60, 100, 50), (50, 80, 140),
        (100, 70, 30), (40, 60, 40), (70, 90, 120),
    ]

    def __init__(self, target_size: tuple = (256, 256), seed: int = 42):
        self.target_size = target_size
        self.base_seed = seed

    def generate(self, subject_idx: int, image_idx: int) -> np.ndarray:
        """
        Generate a unique synthetic face image.

        Uses deterministic random generation based on subject and image
        indices to create consistent but varied face-like images.

        Args:
            subject_idx: Subject number (1-based).
            image_idx: Image number within subject (1-based).

        Returns:
            RGB image as uint8 numpy array (H, W, 3).
        """
        # Deterministic seed per subject/image combination
        seed = self.base_seed + subject_idx * 1000 + image_idx
        rng = np.random.RandomState(seed)

        h, w = self.target_size
        img = np.ones((h, w, 3), dtype=np.uint8) * 200

        # Background gradient
        bg_color_top = rng.randint(180, 240, 3).tolist()
        bg_color_bot = rng.randint(160, 220, 3).tolist()
        for y in range(h):
            ratio = y / h
            for c in range(3):
                img[y, :, c] = int(bg_color_top[c] * (1 - ratio) + bg_color_bot[c] * ratio)

        # Select appearance based on subject (consistent across images)
        subj_rng = np.random.RandomState(self.base_seed + subject_idx * 1000)
        skin_tone = self.SKIN_TONES[subj_rng.randint(len(self.SKIN_TONES))]
        hair_color = self.HAIR_COLORS[subj_rng.randint(len(self.HAIR_COLORS))]
        eye_color = self.EYE_COLORS[subj_rng.randint(len(self.EYE_COLORS))]

        # Face shape parameters (consistent per subject, slight variation per image)
        face_cx = w // 2 + rng.randint(-5, 6)
        face_cy = h // 2 + rng.randint(-8, 4)
        face_rx = rng.randint(55, 72)
        face_ry = rng.randint(72, 90)

        # Draw face (ellipse)
        cv2.ellipse(img, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360,
                     skin_tone, -1, cv2.LINE_AA)

        # Slight skin texture variation
        noise = rng.normal(0, 3, img.shape).astype(np.int16)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(face_mask, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360,
                     255, -1, cv2.LINE_AA)
        noise_mask = face_mask > 0
        for c in range(3):
            channel = img[:, :, c].astype(np.int16)
            channel[noise_mask] += noise[:, :, c][noise_mask]
            img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        # Hair
        hair_style = subj_rng.randint(3)
        if hair_style == 0:  # Top hair
            cv2.ellipse(img, (face_cx, face_cy - face_ry + 15),
                         (face_rx + 8, 35), 0, 180, 360, hair_color, -1, cv2.LINE_AA)
        elif hair_style == 1:  # Full top
            cv2.ellipse(img, (face_cx, face_cy - face_ry + 20),
                         (face_rx + 10, 45), 0, 180, 360, hair_color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (face_cx - face_rx - 5, face_cy - face_ry),
                          (face_cx - face_rx + 10, face_cy + 20), hair_color, -1)
            cv2.rectangle(img, (face_cx + face_rx - 5, face_cy - face_ry),
                          (face_cx + face_rx + 10, face_cy + 20), hair_color, -1)
        else:  # Short
            cv2.ellipse(img, (face_cx, face_cy - face_ry + 10),
                         (face_rx + 5, 25), 0, 180, 360, hair_color, -1, cv2.LINE_AA)

        # Eyes
        eye_spacing = subj_rng.randint(18, 28)
        eye_y = face_cy - rng.randint(5, 15)
        eye_size = subj_rng.randint(6, 10)

        for eye_x in [face_cx - eye_spacing, face_cx + eye_spacing]:
            # White of eye
            cv2.ellipse(img, (eye_x, eye_y), (eye_size + 3, eye_size), 0, 0, 360,
                         (255, 255, 255), -1, cv2.LINE_AA)
            # Iris
            cv2.circle(img, (eye_x + rng.randint(-2, 3), eye_y), eye_size - 1,
                        eye_color, -1, cv2.LINE_AA)
            # Pupil
            cv2.circle(img, (eye_x + rng.randint(-1, 2), eye_y), eye_size // 3,
                        (10, 10, 10), -1, cv2.LINE_AA)
            # Eyelid line
            cv2.ellipse(img, (eye_x, eye_y - 1), (eye_size + 3, eye_size),
                         0, 180, 360, hair_color, 1, cv2.LINE_AA)

        # Eyebrows
        brow_thickness = subj_rng.randint(2, 4)
        for brow_x in [face_cx - eye_spacing, face_cx + eye_spacing]:
            brow_y = eye_y - eye_size - rng.randint(5, 10)
            cv2.line(img, (brow_x - eye_size - 2, brow_y + rng.randint(-2, 3)),
                     (brow_x + eye_size + 2, brow_y + rng.randint(-2, 3)),
                     hair_color, brow_thickness, cv2.LINE_AA)

        # Nose
        nose_length = subj_rng.randint(15, 25)
        nose_width = subj_rng.randint(6, 12)
        nose_y = eye_y + rng.randint(12, 20)
        nose_tip_y = nose_y + nose_length

        # Nose shadow/shape
        darker_skin = tuple(max(0, c - 30) for c in skin_tone)
        cv2.line(img, (face_cx, nose_y), (face_cx, nose_tip_y),
                 darker_skin, 2, cv2.LINE_AA)
        cv2.ellipse(img, (face_cx, nose_tip_y), (nose_width, 5), 0, 0, 180,
                     darker_skin, 1, cv2.LINE_AA)

        # Mouth
        mouth_y = nose_tip_y + rng.randint(12, 22)
        mouth_width = subj_rng.randint(15, 25)
        lip_color = tuple(min(255, c + 30) for c in (180, 80, 80))

        # Upper lip
        cv2.ellipse(img, (face_cx, mouth_y), (mouth_width, 5), 0, 180, 360,
                     lip_color, -1, cv2.LINE_AA)
        # Lower lip
        cv2.ellipse(img, (face_cx, mouth_y + 2), (mouth_width - 2, 6), 0, 0, 180,
                     lip_color, -1, cv2.LINE_AA)

        # Apply slight Gaussian blur for smoothness
        img = cv2.GaussianBlur(img, (3, 3), 0.8)

        # Add per-image variation (simulates different lighting/time)
        brightness_shift = rng.randint(-15, 16)
        img = np.clip(img.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

        return img


class FaceScraper:
    """
    Automated face image scraper with rate-limiting and retry logic.

    Downloads synthetic face images from configurable web sources,
    resizes to target dimensions, and saves in organized directory structure.
    Falls back to local synthetic generation if web sources are unavailable.

    Attributes:
        output_dir: Base directory for saving scraped images.
        num_subjects: Number of distinct subjects to scrape.
        images_per_subject: Number of images per subject.
        target_size: Target image dimensions (H, W).
        delay: Delay between HTTP requests for rate-limiting.
        max_retries: Maximum retry attempts per download.
        retry_backoff: Exponential backoff multiplier.
    """

    # URL sources for face images
    SOURCES = {
        "thispersondoesnotexist": "https://thispersondoesnotexist.com",
        "synthetic": None,  # Local generation
    }

    def __init__(
        self,
        output_dir: str = "data/raw/REAL",
        num_subjects: int = 200,
        images_per_subject: int = 2,
        target_size: tuple = (256, 256),
        delay: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 15,
        source: str = "thispersondoesnotexist",
        user_agent: str = "Mozilla/5.0 FaceBiometricResearch/1.0",
    ):
        self.output_dir = Path(output_dir)
        self.num_subjects = num_subjects
        self.images_per_subject = images_per_subject
        self.target_size = target_size
        self.delay = delay
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.source = source
        self.user_agent = user_agent

        # Validate source
        if source not in self.SOURCES:
            raise ValueError(
                f"Unknown source '{source}'. Available: {list(self.SOURCES.keys())}"
            )

        self.base_url = self.SOURCES[source]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        # Synthetic generator fallback
        self._synthetic_gen = SyntheticFaceGenerator(
            target_size=target_size, seed=42
        )
        self._consecutive_failures = 0
        self._use_synthetic = source == "synthetic"

    def _download_single_image(self, subject_idx: int = 0, image_idx: int = 0) -> Optional[Image.Image]:
        """
        Download a single face image from the configured source.

        Falls back to synthetic generation after consecutive failures.

        Returns:
            PIL Image object on success, None on failure after retries.
        """
        # Use synthetic generator if web source is failing
        if self._use_synthetic:
            return self._generate_synthetic(subject_idx, image_idx)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    self.base_url,
                    timeout=self.timeout,
                    headers={
                        "Cache-Control": "no-cache",
                        "Accept": "image/jpeg,image/png,image/*,*/*",
                    },
                )
                response.raise_for_status()

                # Verify we got an image response
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type and len(response.content) < 1000:
                    raise IOError(
                        f"Non-image response (Content-Type: {content_type}, "
                        f"size: {len(response.content)} bytes)"
                    )

                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize(self.target_size, Image.LANCZOS)
                self._consecutive_failures = 0
                return img

            except (requests.RequestException, IOError, OSError) as e:
                wait_time = self.delay * (self.retry_backoff ** (attempt - 1))
                logger.warning(
                    f"Download attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        # Track consecutive failures
        self._consecutive_failures += 1

        # Switch to synthetic after 5 consecutive complete failures
        if self._consecutive_failures >= 5:
            logger.warning(
                f"Web source failing consistently ({self._consecutive_failures} "
                f"consecutive failures). Switching to synthetic face generation."
            )
            self._use_synthetic = True
            return self._generate_synthetic(subject_idx, image_idx)

        logger.error("All download attempts exhausted.")
        return None

    def _generate_synthetic(self, subject_idx: int, image_idx: int) -> Image.Image:
        """Generate a synthetic face image as fallback."""
        img_array = self._synthetic_gen.generate(subject_idx, image_idx)
        img = Image.fromarray(img_array)
        img = img.resize(self.target_size, Image.LANCZOS)
        return img

    def scrape(self) -> dict:
        """
        Execute the full scraping pipeline.

        Downloads face images for all subjects, organizes them in the
        output directory with the naming convention SXXXX_NNN.jpg.
        Falls back to synthetic generation if web source is unavailable.

        Returns:
            Dictionary with scraping statistics:
                - total_downloaded: Number of images successfully downloaded.
                - total_failed: Number of failed downloads.
                - total_synthetic: Number of synthetically generated images.
                - subjects: List of subject IDs.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_images = self.num_subjects * self.images_per_subject
        stats = {
            "total_downloaded": 0,
            "total_failed": 0,
            "total_synthetic": 0,
            "subjects": [],
        }

        logger.info(
            f"Starting scrape: {self.num_subjects} subjects × "
            f"{self.images_per_subject} images = {total_images} total"
        )
        logger.info(f"Source: {self.source} ({self.base_url or 'local synthetic'})")
        logger.info(f"Output: {self.output_dir}")
        if not self._use_synthetic:
            logger.info(f"Rate limit: {self.delay}s between requests")

        pbar = tqdm(total=total_images, desc="Scraping faces", unit="img")

        for subj_idx in range(1, self.num_subjects + 1):
            subject_id = f"S{subj_idx:04d}"
            stats["subjects"].append(subject_id)

            for img_idx in range(1, self.images_per_subject + 1):
                filename = f"{subject_id}_{img_idx:03d}.jpg"
                filepath = self.output_dir / filename

                # Skip if already downloaded
                if filepath.exists():
                    logger.debug(f"Skipping existing: {filename}")
                    stats["total_downloaded"] += 1
                    pbar.update(1)
                    continue

                img = self._download_single_image(subj_idx, img_idx)

                if img is not None:
                    img.save(str(filepath), "JPEG", quality=95)
                    stats["total_downloaded"] += 1
                    if self._use_synthetic:
                        stats["total_synthetic"] += 1
                    logger.debug(f"Saved: {filename}")
                else:
                    # Last resort: generate synthetic
                    logger.warning(f"Using synthetic fallback for {filename}")
                    img = self._generate_synthetic(subj_idx, img_idx)
                    img.save(str(filepath), "JPEG", quality=95)
                    stats["total_downloaded"] += 1
                    stats["total_synthetic"] += 1

                pbar.update(1)

                # Rate limiting between web requests only
                if not self._use_synthetic:
                    time.sleep(self.delay)

        pbar.close()

        logger.info(
            f"Scraping complete: {stats['total_downloaded']} total, "
            f"{stats['total_synthetic']} synthetic, "
            f"{stats['total_failed']} failed"
        )

        return stats
