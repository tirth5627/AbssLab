"""
Demographic Fairness Plots — FMR/FNMR Bar Charts and DDR Visualization.

Generates publication-quality charts for demographic bias analysis
and noise degradation comparison.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Seaborn styling
sns.set_style("whitegrid")

# Demographic color palettes
GENDER_COLORS = {"Male": "#42A5F5", "Female": "#EF5350"}
ETHNICITY_COLORS = {
    "Asian": "#66BB6A",
    "Caucasian": "#42A5F5",
    "African": "#FFA726",
    "Hispanic": "#AB47BC",
}


def plot_demographic_fmr_fnmr(
    fairness_results: Dict,
    output_path: str = "outputs/figures/demographic_fairness.png",
    title: str = "Demographic Fairness — FMR & FNMR by Subgroup",
) -> str:
    """
    Plot FMR and FNMR bar charts disaggregated by gender and ethnicity.

    Args:
        fairness_results: Results from FairnessAnalyzer.full_fairness_analysis().
        output_path: Path to save the figure.
        title: Plot title.

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for row, attribute in enumerate(["gender", "ethnicity"]):
        if attribute not in fairness_results:
            continue

        group_metrics = fairness_results[attribute]["group_metrics"]
        groups = sorted(group_metrics.keys())

        fmr_vals = [group_metrics[g]["fmr"] for g in groups]
        fnmr_vals = [group_metrics[g]["fnmr"] for g in groups]

        color_map = GENDER_COLORS if attribute == "gender" else ETHNICITY_COLORS
        colors = [color_map.get(g, "#9E9E9E") for g in groups]

        x = np.arange(len(groups))
        width = 0.35

        # FMR subplot
        ax_fmr = axes[row, 0]
        bars = ax_fmr.bar(x, fmr_vals, width, color=colors, edgecolor="white")
        ax_fmr.set_xlabel(attribute.capitalize())
        ax_fmr.set_ylabel("False Match Rate (FMR)")
        ax_fmr.set_title(f"FMR by {attribute.capitalize()}")
        ax_fmr.set_xticks(x)
        ax_fmr.set_xticklabels(groups, rotation=0)
        ax_fmr.set_ylim(bottom=0)

        # Add value labels
        for bar, val in zip(bars, fmr_vals):
            ax_fmr.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
            )

        # FNMR subplot
        ax_fnmr = axes[row, 1]
        bars = ax_fnmr.bar(x, fnmr_vals, width, color=colors, edgecolor="white")
        ax_fnmr.set_xlabel(attribute.capitalize())
        ax_fnmr.set_ylabel("False Non-Match Rate (FNMR)")
        ax_fnmr.set_title(f"FNMR by {attribute.capitalize()}")
        ax_fnmr.set_xticks(x)
        ax_fnmr.set_xticklabels(groups, rotation=0)
        ax_fnmr.set_ylim(bottom=0)

        for bar, val in zip(bars, fnmr_vals):
            ax_fnmr.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
            )

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved demographic fairness chart to {output_path}")
    return output_path


def plot_ddr_chart(
    fairness_results: Dict,
    output_path: str = "outputs/figures/ddr_chart.png",
    title: str = "Demographic Disparity Ratio (DDR)",
) -> str:
    """
    Plot DDR bar chart showing disparity ratios across subgroups.

    DDR = 1.0 means equal to overall; >1.0 means disadvantaged.

    Args:
        fairness_results: Results from FairnessAnalyzer.
        output_path: Path to save the figure.
        title: Plot title.

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, attribute in enumerate(["gender", "ethnicity"]):
        if attribute not in fairness_results:
            continue

        ddr = fairness_results[attribute]["ddr"]
        groups = sorted(ddr.keys())
        values = [ddr[g] for g in groups]

        color_map = GENDER_COLORS if attribute == "gender" else ETHNICITY_COLORS
        colors = [color_map.get(g, "#9E9E9E") for g in groups]

        ax = axes[idx]
        x = np.arange(len(groups))
        bars = ax.bar(x, values, color=colors, edgecolor="white")

        # Reference line at DDR = 1.0
        ax.axhline(y=1.0, color="#FF9800", linestyle="--", linewidth=1.5,
                    label="Parity (DDR = 1.0)")

        ax.set_xlabel(attribute.capitalize())
        ax.set_ylabel("DDR (FNMR_group / FNMR_overall)")
        ax.set_title(f"DDR by {attribute.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=0)
        ax.legend()
        ax.set_ylim(bottom=0)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
            )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved DDR chart to {output_path}")
    return output_path


def plot_noise_degradation(
    variant_metrics: Dict[str, Dict],
    output_path: str = "outputs/figures/noise_degradation.png",
    title: str = "Noise Degradation — Performance Impact Analysis",
) -> str:
    """
    Plot EER and AUC comparison across noise variants.

    Args:
        variant_metrics: Dictionary mapping variant names to metric results.
        output_path: Path to save the figure.
        title: Plot title.

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    variants = sorted(variant_metrics.keys())
    eers = [variant_metrics[v]["eer"] for v in variants]
    aucs = [variant_metrics[v]["auc_roc"] for v in variants]

    colors = [
        {"REAL": "#4CAF50", "GAUSSIAN": "#FF9800", "SALT_PEPPER": "#F44336"}.get(v, "#9E9E9E")
        for v in variants
    ]

    # EER comparison
    ax1 = axes[0]
    bars = ax1.bar(variants, eers, color=colors, edgecolor="white")
    ax1.set_ylabel("Equal Error Rate (EER)")
    ax1.set_title("EER by Variant")
    ax1.set_ylim(bottom=0)

    for bar, val in zip(bars, eers):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11,
        )

    # AUC comparison
    ax2 = axes[1]
    bars = ax2.bar(variants, aucs, color=colors, edgecolor="white")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("AUC-ROC by Variant")
    ax2.set_ylim(0, 1.05)

    for bar, val in zip(bars, aucs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11,
        )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved noise degradation chart to {output_path}")
    return output_path
