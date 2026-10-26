"""
Biometric Performance Metrics — FMR, FNMR, EER, and AUC-ROC.

Implements standard biometric evaluation metrics for verification
system performance assessment.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


class BiometricMetrics:
    """
    Computes biometric verification performance metrics.

    Performs threshold sweep analysis to calculate FMR, FNMR,
    EER, AUC-ROC, and operational thresholds.

    Attributes:
        threshold_step: Step size for threshold sweep.
        threshold_range: Tuple of (min, max) threshold values.
        fmr_targets: List of target FMR values for operational thresholds.
    """

    def __init__(
        self,
        threshold_step: float = 0.001,
        threshold_range: Tuple[float, float] = (0.0, 1.0),
        fmr_targets: List[float] = None,
    ):
        self.threshold_step = threshold_step
        self.threshold_range = threshold_range
        self.fmr_targets = fmr_targets or [0.001, 0.0001]

    @staticmethod
    def compute_fmr(impostor_scores: np.ndarray, threshold: float) -> float:
        """
        Compute False Match Rate (FMR) at a given threshold.

        FMR(T) = |{impostor scores >= T}| / |total impostor pairs|

        Args:
            impostor_scores: Array of impostor comparison scores.
            threshold: Decision threshold.

        Returns:
            FMR value in [0, 1].
        """
        if len(impostor_scores) == 0:
            return 0.0
        return float(np.sum(impostor_scores >= threshold) / len(impostor_scores))

    @staticmethod
    def compute_fnmr(genuine_scores: np.ndarray, threshold: float) -> float:
        """
        Compute False Non-Match Rate (FNMR) at a given threshold.

        FNMR(T) = |{genuine scores < T}| / |total genuine pairs|

        Args:
            genuine_scores: Array of genuine comparison scores.
            threshold: Decision threshold.

        Returns:
            FNMR value in [0, 1].
        """
        if len(genuine_scores) == 0:
            return 0.0
        return float(np.sum(genuine_scores < threshold) / len(genuine_scores))

    def threshold_sweep(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Perform threshold sweep to compute FMR and FNMR at each threshold.

        Args:
            genuine_scores: Array of genuine comparison scores.
            impostor_scores: Array of impostor comparison scores.

        Returns:
            Dictionary with keys:
                - thresholds: Array of threshold values
                - fmr: FMR at each threshold
                - fnmr: FNMR at each threshold
        """
        thresholds = np.arange(
            self.threshold_range[0],
            self.threshold_range[1] + self.threshold_step,
            self.threshold_step,
        )

        fmr_values = np.array([
            self.compute_fmr(impostor_scores, t) for t in thresholds
        ])
        fnmr_values = np.array([
            self.compute_fnmr(genuine_scores, t) for t in thresholds
        ])

        return {
            "thresholds": thresholds,
            "fmr": fmr_values,
            "fnmr": fnmr_values,
        }

    def compute_eer(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER).

        Finds the threshold where |FMR(T) - FNMR(T)| is minimized.

        Args:
            genuine_scores: Array of genuine comparison scores.
            impostor_scores: Array of impostor comparison scores.

        Returns:
            Tuple of (eer_value, eer_threshold).
        """
        sweep = self.threshold_sweep(genuine_scores, impostor_scores)
        thresholds = sweep["thresholds"]
        fmr = sweep["fmr"]
        fnmr = sweep["fnmr"]

        # Find where |FMR - FNMR| is minimized
        diff = np.abs(fmr - fnmr)
        eer_idx = np.argmin(diff)

        eer_value = float((fmr[eer_idx] + fnmr[eer_idx]) / 2.0)
        eer_threshold = float(thresholds[eer_idx])

        logger.info(f"EER = {eer_value:.4f} at threshold = {eer_threshold:.4f}")
        return eer_value, eer_threshold

    def compute_roc_auc(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> float:
        """
        Compute Area Under the ROC Curve (AUC-ROC).

        Args:
            genuine_scores: Array of genuine comparison scores.
            impostor_scores: Array of impostor comparison scores.

        Returns:
            AUC-ROC value in [0, 1].
        """
        # Labels: 1 = genuine, 0 = impostor
        labels = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores)),
        ])
        scores = np.concatenate([genuine_scores, impostor_scores])

        auc = roc_auc_score(labels, scores)
        logger.info(f"AUC-ROC = {auc:.4f}")
        return float(auc)

    def compute_roc_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve data points.

        Args:
            genuine_scores: Array of genuine comparison scores.
            impostor_scores: Array of impostor comparison scores.

        Returns:
            Tuple of (fpr, tpr, thresholds) arrays.
        """
        labels = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores)),
        ])
        scores = np.concatenate([genuine_scores, impostor_scores])

        fpr, tpr, thresholds = roc_curve(labels, scores)
        return fpr, tpr, thresholds

    def find_operational_thresholds(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Find operational thresholds at target FMR values.

        For security-critical applications, identifies the threshold
        and corresponding FNMR at specific FMR operating points.

        Args:
            genuine_scores: Array of genuine comparison scores.
            impostor_scores: Array of impostor comparison scores.

        Returns:
            Dictionary mapping FMR target to threshold and FNMR values.
        """
        sweep = self.threshold_sweep(genuine_scores, impostor_scores)
        thresholds = sweep["thresholds"]
        fmr = sweep["fmr"]
        fnmr = sweep["fnmr"]

        results = {}
        for target_fmr in self.fmr_targets:
            # Find the first threshold where FMR <= target
            mask = fmr <= target_fmr
            if np.any(mask):
                idx = np.argmax(mask)  # First True index
                results[f"FMR_{target_fmr}"] = {
                    "threshold": float(thresholds[idx]),
                    "achieved_fmr": float(fmr[idx]),
                    "fnmr_at_threshold": float(fnmr[idx]),
                }
            else:
                results[f"FMR_{target_fmr}"] = {
                    "threshold": float(thresholds[-1]),
                    "achieved_fmr": float(fmr[-1]),
                    "fnmr_at_threshold": float(fnmr[-1]),
                }

            logger.info(
                f"At FMR={target_fmr}: threshold={results[f'FMR_{target_fmr}']['threshold']:.4f}, "
                f"FNMR={results[f'FMR_{target_fmr}']['fnmr_at_threshold']:.4f}"
            )

        return results

    def evaluate_all(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Dict:
        """
        Run the complete evaluation metrics suite.

        Args:
            genuine_scores: Array of genuine comparison scores.
            impostor_scores: Array of impostor comparison scores.

        Returns:
            Dictionary containing all computed metrics.
        """
        eer, eer_threshold = self.compute_eer(genuine_scores, impostor_scores)
        auc = self.compute_roc_auc(genuine_scores, impostor_scores)
        operational = self.find_operational_thresholds(
            genuine_scores, impostor_scores
        )

        return {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "auc_roc": auc,
            "operational_thresholds": operational,
            "genuine_scores_stats": {
                "mean": float(np.mean(genuine_scores)),
                "std": float(np.std(genuine_scores)),
                "min": float(np.min(genuine_scores)),
                "max": float(np.max(genuine_scores)),
                "median": float(np.median(genuine_scores)),
            },
            "impostor_scores_stats": {
                "mean": float(np.mean(impostor_scores)),
                "std": float(np.std(impostor_scores)),
                "min": float(np.min(impostor_scores)),
                "max": float(np.max(impostor_scores)),
                "median": float(np.median(impostor_scores)),
            },
            "num_genuine_pairs": len(genuine_scores),
            "num_impostor_pairs": len(impostor_scores),
        }
