"""
Tests for Feature Extraction Modules.

Validates vector dimensions, L2 normalization, and feature bounds
for both handcrafted and deep extractors.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction.base import FeatureExtractor
from src.feature_extraction.handcrafted import HandcraftedExtractor


class TestL2Normalization:
    """Tests for L2 normalization utility."""

    def test_unit_norm(self):
        """L2-normalized vector should have unit norm."""
        v = np.array([3.0, 4.0])
        normalized = FeatureExtractor.l2_normalize(v)
        assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-6)

    def test_zero_vector(self):
        """Zero vector should not cause division error."""
        v = np.zeros(10)
        normalized = FeatureExtractor.l2_normalize(v)
        assert np.all(np.isfinite(normalized))

    def test_preserves_direction(self):
        """Normalization should preserve vector direction."""
        v = np.array([1.0, 2.0, 3.0])
        normalized = FeatureExtractor.l2_normalize(v)
        # Direction: all ratios should be equal
        ratios = v / (normalized + 1e-10)
        assert np.allclose(ratios, ratios[0], atol=1e-4)

    def test_already_normalized(self):
        """Already-normalized vector should remain unchanged."""
        v = np.array([0.6, 0.8])
        normalized = FeatureExtractor.l2_normalize(v)
        assert np.allclose(v, normalized, atol=1e-6)


class TestHandcraftedExtractor:
    """Tests for HandcraftedExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a default HandcraftedExtractor."""
        return HandcraftedExtractor(
            histogram_bins=32,
            lbp_radius=1,
            lbp_n_points=8,
        )

    @pytest.fixture
    def sample_image_uint8(self):
        """Sample uint8 RGB image."""
        return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_image_float(self):
        """Sample float64 RGB image in [0, 1]."""
        return np.random.rand(256, 256, 3).astype(np.float64)

    def test_feature_vector_1d(self, extractor, sample_image_uint8):
        """Feature vector should be 1-dimensional."""
        features = extractor.extract(sample_image_uint8)
        assert features.ndim == 1

    def test_feature_vector_dimension(self, extractor, sample_image_uint8):
        """Feature vector should have expected dimensionality."""
        features = extractor.extract(sample_image_uint8)
        # 3 * 32 (color hist) + 10 (LBP: 8+2) + 7 (Hu moments) = 113
        expected_dim = 3 * 32 + (8 + 2) + 7
        assert len(features) == expected_dim

    def test_feature_dtype(self, extractor, sample_image_uint8):
        """Feature vector should be float64."""
        features = extractor.extract(sample_image_uint8)
        assert features.dtype == np.float64

    def test_float_input(self, extractor, sample_image_float):
        """Extractor should handle float64 input."""
        features = extractor.extract(sample_image_float)
        assert features.ndim == 1
        assert np.all(np.isfinite(features))

    def test_normalized_output(self, extractor, sample_image_uint8):
        """extract_normalized should return unit-norm vector."""
        features = extractor.extract_normalized(sample_image_uint8)
        norm = np.linalg.norm(features)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_deterministic(self, extractor, sample_image_uint8):
        """Same input should produce same output."""
        f1 = extractor.extract(sample_image_uint8)
        f2 = extractor.extract(sample_image_uint8)
        assert np.array_equal(f1, f2)

    def test_different_images_different_features(self, extractor):
        """Different images should produce different features."""
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
        f1 = extractor.extract(img1)
        f2 = extractor.extract(img2)
        assert not np.array_equal(f1, f2)

    def test_custom_bins(self):
        """Custom histogram bins should change feature dimension."""
        extractor = HandcraftedExtractor(histogram_bins=16, lbp_n_points=8)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        features = extractor.extract(img)
        expected_dim = 3 * 16 + (8 + 2) + 7
        assert len(features) == expected_dim
