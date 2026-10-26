"""
ROC & DET Curve Plotter — Biometric Performance Visualization.

Generates publication-quality ROC curves, DET curves, and
score distribution histograms.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")  # Non-interactive backend
logger = logging.getLogger(__name__)

# Publication-quality plot styling
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette for variants
VARIANT_COLORS = {
    "REAL": "#2196F3",       # Blue
    "GAUSSIAN": "#FF9800",   # Orange
    "SALT_PEPPER": "#F44336", # Red
}

VARIANT_STYLES = {
    "REAL": "-",
    "GAUSSIAN": "--",
    "SALT_PEPPER": "-.",
}


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str = "outputs/figures/roc_curves.png",
    title: str = "ROC Curves — Biometric Verification Performance",
) -> str:
    """
    Plot ROC curves for multiple dataset variants.

    Args:
        roc_data: Dictionary mapping variant names to (fpr, tpr, auc) tuples.
        output_path: Path to save the figure.
        title: Plot title.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for variant, (fpr, tpr, auc) in roc_data.items():
        color = VARIANT_COLORS.get(variant, "#9E9E9E")
        style = VARIANT_STYLES.get(variant, "-")
        ax.plot(
            fpr, tpr,
            color=color, linestyle=style,
            label=f"{variant} (AUC = {auc:.4f})",
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate (FMR)")
    ax.set_ylabel("True Positive Rate (1 - FNMR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved ROC curves to {output_path}")
    return output_path


def plot_det_curves(
    det_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = "outputs/figures/det_curves.png",
    title: str = "DET Curves — Detection Error Tradeoff",
) -> str:
    """
    Plot DET (Detection Error Tradeoff) curves.

    Uses probit (normal deviate) scale for both axes,
    which is the standard DET curve representation.

    Args:
        det_data: Dictionary mapping variant names to (fmr, fnmr) arrays.
        output_path: Path to save the figure.
        title: Plot title.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for variant, (fmr, fnmr) in det_data.items():
        color = VARIANT_COLORS.get(variant, "#9E9E9E")
        style = VARIANT_STYLES.get(variant, "-")

        # Filter out zero values for log scale
        mask = (fmr > 0) & (fnmr > 0) & (fmr < 1) & (fnmr < 1)
        if np.any(mask):
            ax.plot(
                fmr[mask], fnmr[mask],
                color=color, linestyle=style,
                label=variant,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Match Rate (FMR)")
    ax.set_ylabel("False Non-Match Rate (FNMR)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved DET curves to {output_path}")
    return output_path


def plot_score_distributions(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    variant: str = "REAL",
    eer_threshold: Optional[float] = None,
    output_path: str = "outputs/figures/score_distributions.png",
    title: Optional[str] = None,
) -> str:
    """
    Plot genuine vs impostor score distribution histograms.

    Args:
        genuine_scores: Array of genuine comparison scores.
        impostor_scores: Array of impostor comparison scores.
        variant: Dataset variant name for labeling.
        eer_threshold: Optional EER threshold to mark.
        output_path: Path to save the figure.
        title: Optional custom title.

    Returns:
        Path to the saved figure.
    """
    if title is None:
        title = f"Score Distributions — {variant}"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histograms
    bins = np.linspace(0, 1, 80)
    ax.hist(
        genuine_scores, bins=bins, alpha=0.6,
        color="#4CAF50", label=f"Genuine (n={len(genuine_scores)})",
        density=True, edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        impostor_scores, bins=bins, alpha=0.6,
        color="#F44336", label=f"Impostor (n={len(impostor_scores)})",
        density=True, edgecolor="white", linewidth=0.5,
    )

    # EER threshold line
    if eer_threshold is not None:
        ax.axvline(
            eer_threshold, color="#FF9800", linestyle="--",
            linewidth=2, label=f"EER Threshold = {eer_threshold:.4f}",
        )

    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved score distributions to {output_path}")
    return output_path


def plot_fmr_fnmr_curves(
    thresholds: np.ndarray,
    fmr: np.ndarray,
    fnmr: np.ndarray,
    eer_threshold: Optional[float] = None,
    variant: str = "REAL",
    output_path: str = "outputs/figures/fmr_fnmr_curves.png",
) -> str:
    """
    Plot FMR and FNMR as functions of threshold.

    Args:
        thresholds: Threshold values array.
        fmr: FMR values array.
        fnmr: FNMR values array.
        eer_threshold: Optional EER threshold to mark.
        variant: Dataset variant name.
        output_path: Output file path.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, fmr, color="#F44336", label="FMR (False Match Rate)")
    ax.plot(thresholds, fnmr, color="#2196F3", label="FNMR (False Non-Match Rate)")

    if eer_threshold is not None:
        ax.axvline(
            eer_threshold, color="#FF9800", linestyle="--",
            label=f"EER Threshold = {eer_threshold:.4f}",
        )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"FMR & FNMR vs Threshold — {variant}")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved FMR/FNMR curves to {output_path}")
    return output_path
