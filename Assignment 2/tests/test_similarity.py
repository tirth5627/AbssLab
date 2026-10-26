"""
Tests for Cosine Similarity Functions.

Validates similarity score bounds, edge cases, and
batch computation correctness.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matching.similarity import cosine_similarity, compute_scores


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity = 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        assert np.isclose(sim, 1.0, atol=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity = 0.0 (remapped)."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert np.isclose(sim, 0.0, atol=1e-6)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity = 0.5 (remapped)."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = cosine_similarity(v1, v2)
        assert np.isclose(sim, 0.5, atol=1e-6)

    def test_bounds(self):
        """Similarity should be in [0, 1] for any input."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            v1 = rng.randn(128)
            v2 = rng.randn(128)
            sim = cosine_similarity(v1, v2)
            assert 0.0 <= sim <= 1.0

    def test_zero_vector(self):
        """Zero vector should return similarity = 0.0."""
        v1 = np.zeros(10)
        v2 = np.array([1.0] * 10)
        sim = cosine_similarity(v1, v2)
        assert sim == 0.0

    def test_symmetric(self):
        """Cosine similarity should be symmetric."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        assert cosine_similarity(v1, v2) == cosine_similarity(v2, v1)

    def test_scale_invariant(self):
        """Similarity should be scale-invariant."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([2.0, 4.0, 6.0])  # 2 * v1
        sim = cosine_similarity(v1, v2)
        assert np.isclose(sim, 1.0, atol=1e-6)


class TestComputeScores:
    """Tests for batch score computation."""

    def test_genuine_pair_scoring(self):
        """Genuine pair should compute score from correct features."""
        pairs = [{"subject_id_a": "S0001", "subject_id_b": "S0001",
                   "image_idx_a": 1, "image_idx_b": 2, "pair_type": "genuine"}]
        features = {
            "S0001": {
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([1.0, 0.0, 0.0]),
            }
        }
        scores = compute_scores(pairs, features)
        assert len(scores) == 1
        assert np.isclose(scores[0], 1.0, atol=1e-6)

    def test_impostor_pair_scoring(self):
        """Impostor pair should compute cross-subject score."""
        pairs = [{"subject_id_a": "S0001", "subject_id_b": "S0002",
                   "image_idx_a": 1, "image_idx_b": 1, "pair_type": "impostor"}]
        features = {
            "S0001": {1: np.array([1.0, 0.0])},
            "S0002": {1: np.array([0.0, 1.0])},
        }
        scores = compute_scores(pairs, features)
        assert len(scores) == 1
        assert np.isclose(scores[0], 0.5, atol=1e-6)  # Orthogonal → 0.5

    def test_missing_features(self):
        """Missing features should return score of 0.0."""
        pairs = [{"subject_id_a": "S0001", "subject_id_b": "S9999",
                   "image_idx_a": 1, "image_idx_b": 1, "pair_type": "impostor"}]
        features = {"S0001": {1: np.array([1.0, 0.0])}}
        scores = compute_scores(pairs, features)
        assert scores[0] == 0.0

    def test_multiple_pairs(self):
        """Should handle multiple pairs correctly."""
        pairs = [
            {"subject_id_a": "S0001", "subject_id_b": "S0001",
             "image_idx_a": 1, "image_idx_b": 2, "pair_type": "genuine"},
            {"subject_id_a": "S0001", "subject_id_b": "S0002",
             "image_idx_a": 1, "image_idx_b": 1, "pair_type": "impostor"},
        ]
        features = {
            "S0001": {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
            "S0002": {1: np.array([0.0, 1.0])},
        }
        scores = compute_scores(pairs, features)
        assert len(scores) == 2
