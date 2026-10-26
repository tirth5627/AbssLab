"""
Tests for Biometric Evaluation Metrics.

Validates FMR, FNMR, EER computation, and AUC-ROC bounds
including edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import BiometricMetrics


class TestFMR:
    """Tests for False Match Rate computation."""

    def test_all_below_threshold(self):
        """All impostor scores below threshold → FMR = 0."""
        scores = np.array([0.1, 0.2, 0.3])
        fmr = BiometricMetrics.compute_fmr(scores, 0.5)
        assert fmr == 0.0

    def test_all_above_threshold(self):
        """All impostor scores above threshold → FMR = 1."""
        scores = np.array([0.6, 0.7, 0.8])
        fmr = BiometricMetrics.compute_fmr(scores, 0.5)
        assert fmr == 1.0

    def test_partial_match(self):
        """Some impostor scores above threshold."""
        scores = np.array([0.1, 0.6, 0.7, 0.8])
        fmr = BiometricMetrics.compute_fmr(scores, 0.5)
        assert np.isclose(fmr, 3.0 / 4.0)

    def test_empty_scores(self):
        """Empty score array → FMR = 0."""
        scores = np.array([])
        fmr = BiometricMetrics.compute_fmr(scores, 0.5)
        assert fmr == 0.0

    def test_fmr_at_exact_threshold(self):
        """Scores equal to threshold should be counted as matches."""
        scores = np.array([0.5, 0.5, 0.5])
        fmr = BiometricMetrics.compute_fmr(scores, 0.5)
        assert fmr == 1.0


class TestFNMR:
    """Tests for False Non-Match Rate computation."""

    def test_all_above_threshold(self):
        """All genuine scores above threshold → FNMR = 0."""
        scores = np.array([0.6, 0.7, 0.8])
        fnmr = BiometricMetrics.compute_fnmr(scores, 0.5)
        assert fnmr == 0.0

    def test_all_below_threshold(self):
        """All genuine scores below threshold → FNMR = 1."""
        scores = np.array([0.1, 0.2, 0.3])
        fnmr = BiometricMetrics.compute_fnmr(scores, 0.5)
        assert fnmr == 1.0

    def test_partial_non_match(self):
        """Some genuine scores below threshold."""
        scores = np.array([0.1, 0.2, 0.7, 0.8])
        fnmr = BiometricMetrics.compute_fnmr(scores, 0.5)
        assert np.isclose(fnmr, 2.0 / 4.0)

    def test_empty_scores(self):
        """Empty score array → FNMR = 0."""
        scores = np.array([])
        fnmr = BiometricMetrics.compute_fnmr(scores, 0.5)
        assert fnmr == 0.0


class TestEER:
    """Tests for Equal Error Rate computation."""

    def test_perfect_separation(self):
        """Perfectly separated scores should give EER ≈ 0."""
        genuine = np.array([0.9, 0.95, 0.85, 0.88])
        impostor = np.array([0.1, 0.15, 0.05, 0.12])
        metrics = BiometricMetrics(threshold_step=0.001)
        eer, threshold = metrics.compute_eer(genuine, impostor)
        assert eer < 0.1  # Should be very low

    def test_complete_overlap(self):
        """Completely overlapping scores should give EER ≈ 0.5."""
        rng = np.random.RandomState(42)
        genuine = rng.rand(1000)
        impostor = rng.rand(1000)
        metrics = BiometricMetrics(threshold_step=0.001)
        eer, threshold = metrics.compute_eer(genuine, impostor)
        assert 0.3 < eer < 0.7  # Should be near 0.5

    def test_eer_range(self):
        """EER should be in [0, 1]."""
        genuine = np.array([0.7, 0.8, 0.6])
        impostor = np.array([0.3, 0.4, 0.5])
        metrics = BiometricMetrics(threshold_step=0.001)
        eer, threshold = metrics.compute_eer(genuine, impostor)
        assert 0.0 <= eer <= 1.0

    def test_eer_threshold_range(self):
        """EER threshold should be in [0, 1]."""
        genuine = np.array([0.7, 0.8, 0.6])
        impostor = np.array([0.3, 0.4, 0.5])
        metrics = BiometricMetrics(threshold_step=0.001)
        eer, threshold = metrics.compute_eer(genuine, impostor)
        assert 0.0 <= threshold <= 1.0


class TestAUCROC:
    """Tests for AUC-ROC computation."""

    def test_perfect_discrimination(self):
        """Perfect separation should give AUC ≈ 1."""
        genuine = np.array([0.9, 0.95, 0.85, 0.88])
        impostor = np.array([0.1, 0.15, 0.05, 0.12])
        metrics = BiometricMetrics()
        auc = metrics.compute_roc_auc(genuine, impostor)
        assert auc > 0.95

    def test_random_discrimination(self):
        """Random scores should give AUC ≈ 0.5."""
        rng = np.random.RandomState(42)
        genuine = rng.rand(1000)
        impostor = rng.rand(1000)
        metrics = BiometricMetrics()
        auc = metrics.compute_roc_auc(genuine, impostor)
        assert 0.4 < auc < 0.6

    def test_auc_bounds(self):
        """AUC should be in [0, 1]."""
        genuine = np.array([0.7, 0.8, 0.6])
        impostor = np.array([0.3, 0.4, 0.5])
        metrics = BiometricMetrics()
        auc = metrics.compute_roc_auc(genuine, impostor)
        assert 0.0 <= auc <= 1.0


class TestEvaluateAll:
    """Tests for the complete evaluation suite."""

    def test_evaluate_all_keys(self):
        """evaluate_all should return all expected keys."""
        genuine = np.array([0.7, 0.8, 0.6, 0.75])
        impostor = np.array([0.3, 0.4, 0.2, 0.35])
        metrics = BiometricMetrics(threshold_step=0.01)
        results = metrics.evaluate_all(genuine, impostor)

        expected_keys = {
            "eer", "eer_threshold", "auc_roc",
            "operational_thresholds",
            "genuine_scores_stats", "impostor_scores_stats",
            "num_genuine_pairs", "num_impostor_pairs",
        }
        assert expected_keys.issubset(set(results.keys()))

    def test_evaluate_all_pair_counts(self):
        """Pair counts should match input sizes."""
        genuine = np.array([0.7, 0.8, 0.6])
        impostor = np.array([0.3, 0.4])
        metrics = BiometricMetrics(threshold_step=0.01)
        results = metrics.evaluate_all(genuine, impostor)
        assert results["num_genuine_pairs"] == 3
        assert results["num_impostor_pairs"] == 2
