"""
Tests for Image Preprocessing Pipeline.

Validates resizing, CLAHE enhancement, and normalization functions.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.enhancer import ImageEnhancer


class TestImageEnhancer:
    """Tests for ImageEnhancer class."""

    @pytest.fixture
    def enhancer(self):
        """Create a default ImageEnhancer instance."""
        return ImageEnhancer(target_size=(256, 256), normalize=True)

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    def test_resize_output_shape(self, enhancer, sample_image):
        """Resized image should match target dimensions."""
        resized = enhancer.resize(sample_image)
        assert resized.shape == (256, 256, 3)

    def test_resize_non_square(self):
        """Resize should handle non-square input."""
        enhancer = ImageEnhancer(target_size=(256, 256))
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        resized = enhancer.resize(img)
        assert resized.shape == (256, 256, 3)

    def test_clahe_output_shape(self, enhancer, sample_image):
        """CLAHE should preserve image dimensions."""
        enhanced = enhancer.apply_clahe(sample_image)
        assert enhanced.shape == sample_image.shape

    def test_clahe_output_dtype(self, enhancer, sample_image):
        """CLAHE output should be uint8."""
        enhanced = enhancer.apply_clahe(sample_image)
        assert enhanced.dtype == np.uint8

    def test_clahe_output_range(self, enhancer, sample_image):
        """CLAHE output should be in [0, 255]."""
        enhanced = enhancer.apply_clahe(sample_image)
        assert enhanced.min() >= 0
        assert enhanced.max() <= 255

    def test_normalize_range(self, enhancer, sample_image):
        """Normalized image should be in [0, 1]."""
        normalized = enhancer.normalize_image(sample_image)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_dtype(self, enhancer, sample_image):
        """Normalized image should be float64."""
        normalized = enhancer.normalize_image(sample_image)
        assert normalized.dtype == np.float64

    def test_grayscale_conversion(self, enhancer, sample_image):
        """Grayscale conversion should produce single-channel output."""
        gray = enhancer.to_grayscale(sample_image)
        assert len(gray.shape) == 2
        assert gray.shape[:2] == sample_image.shape[:2]

    def test_full_pipeline(self, enhancer, sample_image):
        """Full enhance pipeline should produce correct output."""
        enhanced = enhancer.enhance(sample_image)
        assert enhanced.shape == (256, 256, 3)
        assert enhanced.dtype == np.float64
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0

    def test_full_pipeline_no_normalize(self, sample_image):
        """Pipeline without normalization should return uint8."""
        enhancer = ImageEnhancer(target_size=(256, 256), normalize=False)
        enhanced = enhancer.enhance(sample_image)
        assert enhanced.shape == (256, 256, 3)
        assert enhanced.dtype == np.uint8
