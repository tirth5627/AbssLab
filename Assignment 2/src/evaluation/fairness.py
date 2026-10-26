"""
Fairness Analyzer — Disaggregated Demographic Fairness Metrics.

Evaluates biometric system performance across demographic subgroups
to detect and quantify algorithmic bias.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from .metrics import BiometricMetrics

logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """
    Analyzes demographic fairness of biometric matching performance.

    Disaggregates matching metrics across gender and ethnicity subgroups,
    computing per-group FMR, FNMR, and Demographic Disparity Ratios (DDR).

    Attributes:
        metrics: BiometricMetrics instance for metric computation.
    """

    def __init__(self, metrics: BiometricMetrics = None):
        self.metrics = metrics or BiometricMetrics()

    def _group_scores_by_attribute(
        self,
        pairs: List[Dict],
        scores: np.ndarray,
        metadata_lookup: Dict[str, Dict],
        attribute: str,
    ) -> Dict[str, List[float]]:
        """
        Group comparison scores by a demographic attribute.

        For genuine pairs, uses the subject's attribute.
        For impostor pairs, groups by subject A's attribute.

        Args:
            pairs: List of pair dictionaries.
            scores: Corresponding similarity scores.
            metadata_lookup: Dictionary mapping subject_id to metadata.
            attribute: Demographic attribute key ('gender' or 'ethnicity').

        Returns:
            Dictionary mapping attribute values to lists of scores.
        """
        grouped = {}

        for pair, score in zip(pairs, scores):
            sid_a = pair["subject_id_a"]
            meta = metadata_lookup.get(sid_a, {})
            group = meta.get(attribute, "Unknown")

            if group not in grouped:
                grouped[group] = []
            grouped[group].append(score)

        return grouped

    def compute_group_metrics(
        self,
        genuine_pairs: List[Dict],
        impostor_pairs: List[Dict],
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        metadata_lookup: Dict[str, Dict],
        attribute: str,
        global_threshold: float,
    ) -> Dict[str, Dict]:
        """
        Compute per-group FMR and FNMR at a global threshold.

        Args:
            genuine_pairs: List of genuine pair dictionaries.
            impostor_pairs: List of impostor pair dictionaries.
            genuine_scores: Genuine comparison scores.
            impostor_scores: Impostor comparison scores.
            metadata_lookup: Subject metadata lookup.
            attribute: Demographic attribute to disaggregate by.
            global_threshold: Decision threshold (typically EER threshold).

        Returns:
            Dictionary with per-group metrics including FMR, FNMR, and count.
        """
        # Group genuine scores
        genuine_grouped = self._group_scores_by_attribute(
            genuine_pairs, genuine_scores, metadata_lookup, attribute
        )

        # Group impostor scores
        impostor_grouped = self._group_scores_by_attribute(
            impostor_pairs, impostor_scores, metadata_lookup, attribute
        )

        results = {}
        for group in set(list(genuine_grouped.keys()) + list(impostor_grouped.keys())):
            gen_scores = np.array(genuine_grouped.get(group, []))
            imp_scores = np.array(impostor_grouped.get(group, []))

            fnmr = (
                self.metrics.compute_fnmr(gen_scores, global_threshold)
                if len(gen_scores) > 0
                else float("nan")
            )
            fmr = (
                self.metrics.compute_fmr(imp_scores, global_threshold)
                if len(imp_scores) > 0
                else float("nan")
            )

            results[group] = {
                "fmr": fmr,
                "fnmr": fnmr,
                "genuine_count": len(gen_scores),
                "impostor_count": len(imp_scores),
            }

            logger.info(
                f"  {attribute}={group}: FMR={fmr:.4f}, FNMR={fnmr:.4f}, "
                f"genuine={len(gen_scores)}, impostor={len(imp_scores)}"
            )

        return results

    def compute_ddr(
        self,
        group_metrics: Dict[str, Dict],
        overall_fnmr: float,
    ) -> Dict[str, float]:
        """
        Compute Demographic Disparity Ratio (DDR) for each group.

        DDR_group = FNMR_group / FNMR_overall

        A DDR > 1 indicates the group experiences higher FNMR than average
        (potentially disadvantaged). DDR < 1 indicates lower FNMR.

        Args:
            group_metrics: Per-group metrics from compute_group_metrics.
            overall_fnmr: Overall FNMR at the global threshold.

        Returns:
            Dictionary mapping group names to DDR values.
        """
        ddr = {}
        for group, metrics in group_metrics.items():
            if overall_fnmr > 0 and not np.isnan(metrics["fnmr"]):
                ddr[group] = float(metrics["fnmr"] / overall_fnmr)
            else:
                ddr[group] = float("nan")

        logger.info(f"DDR values: {ddr}")
        return ddr

    def full_fairness_analysis(
        self,
        genuine_pairs: List[Dict],
        impostor_pairs: List[Dict],
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        metadata_lookup: Dict[str, Dict],
        global_threshold: float,
        overall_fnmr: float,
    ) -> Dict:
        """
        Run complete demographic fairness analysis.

        Evaluates performance disparities across both gender and
        ethnicity dimensions.

        Args:
            genuine_pairs: List of genuine pair dictionaries.
            impostor_pairs: List of impostor pair dictionaries.
            genuine_scores: Genuine comparison scores.
            impostor_scores: Impostor comparison scores.
            metadata_lookup: Subject metadata lookup.
            global_threshold: Decision threshold.
            overall_fnmr: Overall FNMR value.

        Returns:
            Dictionary with complete fairness analysis results.
        """
        results = {}

        for attribute in ["gender", "ethnicity"]:
            logger.info(f"Analyzing fairness for: {attribute}")

            group_metrics = self.compute_group_metrics(
                genuine_pairs, impostor_pairs,
                genuine_scores, impostor_scores,
                metadata_lookup, attribute, global_threshold,
            )

            ddr = self.compute_ddr(group_metrics, overall_fnmr)

            results[attribute] = {
                "group_metrics": group_metrics,
                "ddr": ddr,
            }

        return results
