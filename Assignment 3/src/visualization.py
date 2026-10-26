"""
Visualization module for face recognition benchmarks.
Generates all 7 required publication-grade plots.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FaceVisualizer:
    """Handles generation and saving of all benchmark plots."""

    def __init__(self, output_dir: str = "outputs/figures", style: str = "seaborn-v0_8-whitegrid"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply clean publication style
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")
        plt.rcParams.update({
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "font.family": "sans-serif",
            "figure.autolayout": True,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 1.0,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        })

    def plot_accuracy_vs_components(
        self,
        component_counts: List[int],
        pca_acc_mean: List[float],
        pca_acc_std: List[float],
        svd_acc_mean: List[float],
        svd_acc_std: List[float],
        lda_acc_mean: List[float],
        lda_acc_std: List[float],
        raw_pixel_acc_mean: float,
        raw_pixel_acc_std: float,
        save_name: str = "plot1_accuracy_vs_components.png"
    ) -> str:
        """Plot 1: Recognition accuracy vs number of components with std bands."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        comps = np.array(component_counts)

        # Baseline Raw Pixels
        ax.axhline(
            y=raw_pixel_acc_mean * 100,
            color="#2c3e50",
            linestyle="--",
            linewidth=2,
            label=f"Raw Pixels (10,304 dims): {raw_pixel_acc_mean*100:.2f}%"
        )
        ax.axhspan(
            (raw_pixel_acc_mean - raw_pixel_acc_std) * 100,
            (raw_pixel_acc_mean + raw_pixel_acc_std) * 100,
            color="#2c3e50",
            alpha=0.10
        )

        # PCA
        pca_m = np.array(pca_acc_mean) * 100
        pca_s = np.array(pca_acc_std) * 100
        ax.plot(comps, pca_m, marker="o", linewidth=2.5, color="#1f77b4", label="PCA + 1-NN")
        ax.fill_between(comps, pca_m - pca_s, pca_m + pca_s, color="#1f77b4", alpha=0.18)

        # Truncated SVD
        svd_m = np.array(svd_acc_mean) * 100
        svd_s = np.array(svd_acc_std) * 100
        ax.plot(comps, svd_m, marker="s", linewidth=2.5, linestyle="-.", color="#2ca02c", label="Truncated SVD + 1-NN")
        ax.fill_between(comps, svd_m - svd_s, svd_m + svd_s, color="#2ca02c", alpha=0.18)

        # LDA
        lda_m = np.array(lda_acc_mean) * 100
        lda_s = np.array(lda_acc_std) * 100
        ax.plot(comps, lda_m, marker="^", linewidth=2.5, color="#d62728", label="LDA + 1-NN (Fisherfaces)")
        ax.fill_between(comps, lda_m - lda_s, lda_m + lda_s, color="#d62728", alpha=0.18)

        ax.set_title("Plot 1: Identification Accuracy vs. Number of Retained Components (ORL Faces)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Subspace Components (k)", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Recognition Accuracy (%) [Mean ± 1 Std Dev]", fontsize=11, fontweight="semibold")
        ax.set_xticks(comps)
        ax.set_ylim([max(0, min(pca_m.min(), svd_m.min(), lda_m.min()) - 10), 102])
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_face_reconstructions(
        self,
        original_faces: List[np.ndarray],
        pca_reconstructions: Dict[int, List[np.ndarray]],
        svd_reconstructions: Dict[int, List[np.ndarray]],
        component_keys: List[int] = [5, 10, 20, 30],
        subject_labels: Optional[List[str]] = None,
        save_name: str = "plot2_face_reconstructions.png"
    ) -> str:
        """Plot 2: Original face beside low-rank reconstructions for PCA and SVD."""
        num_faces = len(original_faces)
        num_cols = 1 + len(component_keys) * 2  # Original + (PCA + SVD for each k)
        # Better layout: 2 rows per face (Row 1: Original + PCA at k; Row 2: Same Original placeholder + SVD at k)
        fig, axes = plt.subplots(
            nrows=num_faces * 2,
            ncols=1 + len(component_keys),
            figsize=(2.2 * (1 + len(component_keys)), 2.6 * num_faces * 2),
            dpi=300
        )

        for f_idx in range(num_faces):
            orig_img = original_faces[f_idx].reshape(112, 92)
            subj_title = subject_labels[f_idx] if subject_labels else f"Subject #{f_idx+1}"

            # Row for PCA
            row_pca = f_idx * 2
            axes[row_pca, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
            axes[row_pca, 0].set_title(f"{subj_title}\nOriginal", fontsize=9, fontweight="bold")
            axes[row_pca, 0].axis("off")

            for col_idx, k in enumerate(component_keys, start=1):
                pca_rec = pca_reconstructions[k][f_idx].reshape(112, 92)
                axes[row_pca, col_idx].imshow(pca_rec, cmap="gray", vmin=0, vmax=1)
                axes[row_pca, col_idx].set_title(f"PCA\nk = {k}", fontsize=9)
                axes[row_pca, col_idx].axis("off")

            # Row for SVD
            row_svd = f_idx * 2 + 1
            axes[row_svd, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
            axes[row_svd, 0].set_title(f"{subj_title}\nOriginal", fontsize=9, fontweight="bold")
            axes[row_svd, 0].axis("off")

            for col_idx, k in enumerate(component_keys, start=1):
                svd_rec = svd_reconstructions[k][f_idx].reshape(112, 92)
                axes[row_svd, col_idx].imshow(svd_rec, cmap="gray", vmin=0, vmax=1)
                axes[row_svd, col_idx].set_title(f"SVD\nk = {k}", fontsize=9)
                axes[row_svd, col_idx].axis("off")

        fig.suptitle("Plot 2: Original Test Faces vs. Low-Rank Reconstructions (PCA and SVD)", fontsize=13, fontweight="bold", y=0.995)
        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_reconstruction_error(
        self,
        component_counts: List[int],
        pca_mse: List[float],
        svd_mse: List[float],
        pca_rel_err: List[float],
        svd_rel_err: List[float],
        save_name: str = "plot3_reconstruction_error.png"
    ) -> str:
        """Plot 3: Reconstruction error (MSE and Relative Frobenius Error) vs component count."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        comps = np.array(component_counts)

        # Left panel: MSE
        ax1.plot(comps, pca_mse, marker="o", linewidth=2.2, color="#1f77b4", label="PCA (Centered)")
        ax1.plot(comps, svd_mse, marker="s", linewidth=2.2, linestyle="--", color="#2ca02c", label="Truncated SVD")
        ax1.set_title("Reconstruction Mean Squared Error (MSE)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Number of Components (k)", fontsize=10, fontweight="semibold")
        ax1.set_ylabel("Pixel MSE (Scale [0, 1])", fontsize=10, fontweight="semibold")
        ax1.set_xticks(comps)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        # Right panel: Relative Frobenius Error
        ax2.plot(comps, np.array(pca_rel_err) * 100, marker="o", linewidth=2.2, color="#1f77b4", label="PCA (Centered)")
        ax2.plot(comps, np.array(svd_rel_err) * 100, marker="s", linewidth=2.2, linestyle="--", color="#2ca02c", label="Truncated SVD")
        ax2.set_title("Relative Frobenius Reconstruction Error (%)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Number of Components (k)", fontsize=10, fontweight="semibold")
        ax2.set_ylabel("Relative Error (%) [||X - X̂||_F / ||X||_F]", fontsize=10, fontweight="semibold")
        ax2.set_xticks(comps)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        fig.suptitle("Plot 3: Test Image Reconstruction Error vs. Number of Components", fontsize=13, fontweight="bold")
        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_explained_variance(
        self,
        component_counts: List[int],
        cumulative_variance_ratio: np.ndarray,
        thresholds: List[float] = [0.90, 0.95],
        svd_cumulative_variance: Optional[np.ndarray] = None,
        save_name: str = "plot4_explained_variance.png"
    ) -> str:
        """Plot 4: Cumulative explained variance with 90% and 95% threshold markers (supports PCA alone or SVD vs PCA)."""
        if svd_cumulative_variance is not None:
            return self.plot_cumulative_variance_svd_vs_pca(
                pca_cumulative_variance=cumulative_variance_ratio,
                svd_cumulative_variance=svd_cumulative_variance,
                thresholds=thresholds,
                save_name=save_name
            )

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        k_indices = np.arange(1, len(cumulative_variance_ratio) + 1)

        ax.plot(k_indices, cumulative_variance_ratio * 100, color="#1f77b4", linewidth=2.5, label="Cumulative Explained Variance")
        ax.fill_between(k_indices, 0, cumulative_variance_ratio * 100, color="#1f77b4", alpha=0.12)

        colors = ["#ff7f0e", "#d62728"]
        for th, col in zip(thresholds, colors):
            th_pct = th * 100
            # find first component reaching threshold
            reach_idx = np.where(cumulative_variance_ratio >= th)[0]
            if len(reach_idx) > 0:
                k_reach = reach_idx[0] + 1
                ax.axhline(y=th_pct, color=col, linestyle="--", linewidth=1.8, label=f"{th_pct:.0f}% Energy Threshold (k={k_reach})")
                ax.axvline(x=k_reach, color=col, linestyle=":", linewidth=1.5)
                ax.scatter([k_reach], [th_pct], color=col, s=70, zorder=5)

        ax.set_title("Plot 4: PCA Cumulative Explained Variance Ratio (ORL Dataset)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Principal Components (k)", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Cumulative Variance Retained (%)", fontsize=11, fontweight="semibold")
        ax.set_xlim([1, len(cumulative_variance_ratio)])
        ax.set_ylim([0, 105])
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_cumulative_variance_svd_vs_pca(
        self,
        pca_cumulative_variance: np.ndarray,
        svd_cumulative_variance: np.ndarray,
        thresholds: List[float] = [0.90, 0.95],
        lda_max_k: int = 39,
        save_name: str = "plot4_explained_variance.png"
    ) -> str:
        """
        Plot: Cumulative Explained Variance Comparison (Truncated SVD vs. PCA)
        Includes 90% and 95% threshold cross points, LDA capacity limit (k=39),
        maximum possible rank limits (k=199 for PCA, k=200 for SVD),
        and lower panel showing the variance discrepancy Delta = PCA - SVD.
        """
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(11, 7.5),
            dpi=300,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        n_pca = len(pca_cumulative_variance)
        n_svd = len(svd_cumulative_variance)
        k_pca = np.arange(1, n_pca + 1)
        k_svd = np.arange(1, n_svd + 1)

        pca_pct = pca_cumulative_variance * 100
        svd_pct = svd_cumulative_variance * 100

        # Top Panel: Cumulative Curves
        ax1.plot(k_pca, pca_pct, color="#1f77b4", linewidth=2.5, label="PCA (Centered Covariance)")
        ax1.fill_between(k_pca, 0, pca_pct, color="#1f77b4", alpha=0.10)

        ax1.plot(k_svd, svd_pct, color="#2ca02c", linewidth=2.5, linestyle="--", label="Truncated SVD (Uncentered Data Matrix)")

        # Thresholds
        colors_th = ["#f59e0b", "#ef4444"]
        for th, col in zip(thresholds, colors_th):
            th_pct = th * 100
            ax1.axhline(y=th_pct, color=col, linestyle=":", linewidth=1.4, alpha=0.85)

            # PCA reach
            r_pca = np.where(pca_cumulative_variance >= th)[0]
            if len(r_pca) > 0:
                k_p = r_pca[0] + 1
                ax1.scatter([k_p], [pca_pct[k_p-1]], color="#1f77b4", edgecolor=col, s=60, zorder=5)
                ax1.annotate(
                    f"PCA {th_pct:.0f}% (k={k_p})",
                    xy=(k_p, pca_pct[k_p-1]),
                    xytext=(k_p + 4, pca_pct[k_p-1] - 4),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
                    fontsize=8.5,
                    fontweight="bold",
                    color="#1f77b4"
                )

            # SVD reach
            r_svd = np.where(svd_cumulative_variance >= th)[0]
            if len(r_svd) > 0:
                k_s = r_svd[0] + 1
                ax1.scatter([k_s], [svd_pct[k_s-1]], color="#2ca02c", edgecolor=col, s=60, zorder=5)
                ax1.annotate(
                    f"SVD {th_pct:.0f}% (k={k_s})",
                    xy=(k_s, svd_pct[k_s-1]),
                    xytext=(k_s + 4, svd_pct[k_s-1] + 2),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0),
                    fontsize=8.5,
                    fontweight="bold",
                    color="#2ca02c"
                )

        # Highlight k=39 (LDA Upper Bound)
        if lda_max_k <= n_pca and lda_max_k <= n_svd:
            ax1.axvline(x=lda_max_k, color="#8b5cf6", linestyle="-.", linewidth=1.5, alpha=0.8)
            ax1.annotate(
                f"LDA Max Capacity (k=39)\nPCA: {pca_pct[lda_max_k-1]:.1f}% | SVD: {svd_pct[lda_max_k-1]:.1f}%",
                xy=(lda_max_k, pca_pct[lda_max_k-1]),
                xytext=(lda_max_k + 8, 45),
                arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=1.2),
                fontsize=8.5,
                fontweight="semibold",
                color="#6d28d9",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ede9fe", edgecolor="#8b5cf6", alpha=0.85)
            )

        # Highlight max rank limits
        ax1.axvline(x=n_pca, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.7)
        ax1.text(n_pca - 2, 12, f"PCA Max Rank\nmin(N-1, D)={n_pca}", color="#1f77b4", fontsize=7.5, ha="right")

        ax1.axvline(x=n_svd, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.7)
        ax1.text(n_svd + 2, 12, f"SVD Max Rank\nmin(N, D)={n_svd}", color="#2ca02c", fontsize=7.5, ha="left")

        ax1.set_title("Plot 4: Cumulative Explained Variance Ratio — Truncated SVD vs. PCA (ORL Faces)", fontsize=12.5, fontweight="bold", pad=10)
        ax1.set_ylabel("Cumulative Variance Retained (%)", fontsize=10.5, fontweight="semibold")
        ax1.set_xlim([0, max(n_pca, n_svd) + 5])
        ax1.set_ylim([0, 105])
        ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Bottom Panel: Delta (PCA - SVD)
        min_len = min(n_pca, n_svd)
        delta_pct = (pca_cumulative_variance[:min_len] - svd_cumulative_variance[:min_len]) * 100
        k_common = np.arange(1, min_len + 1)

        ax2.plot(k_common, delta_pct, color="#4338ca", linewidth=1.8, label="Variance Advantage: PCA - SVD (%)")
        ax2.fill_between(k_common, 0, delta_pct, color="#6366f1", alpha=0.20)
        ax2.axhline(y=0, color="#64748b", linestyle="-", linewidth=1.0)
        ax2.set_xlabel("Number of Retained Subspace Components (k)", fontsize=10.5, fontweight="semibold")
        ax2.set_ylabel("Δ Variance (%)", fontsize=10, fontweight="semibold")
        ax2.set_xlim([0, max(n_pca, n_svd) + 5])
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

        max_delta_idx = int(np.argmax(delta_pct))
        ax2.annotate(
            f"Max Gap: +{delta_pct[max_delta_idx]:.2f}% at k={max_delta_idx+1}\n(Uncentered SVD spends component 1 on DC mean)",
            xy=(max_delta_idx + 1, delta_pct[max_delta_idx]),
            xytext=(max_delta_idx + 15, delta_pct[max_delta_idx] * 0.75),
            arrowprops=dict(arrowstyle="->", color="#4338ca", lw=1.0),
            fontsize=8,
            color="#312e81"
        )

        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        # Also ensure plot4_cumulative_variance_svd_vs_pca.png is written
        alt_path = os.path.join(self.output_dir, "plot4_cumulative_variance_svd_vs_pca.png")
        plt.savefig(alt_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_spectra_pca_svd_lda(
        self,
        pca_eigenvalues: np.ndarray,
        svd_singular_values: np.ndarray,
        lda_discriminant_ratios: np.ndarray,
        save_name: str = "plot8_spectra_pca_svd_lda.png"
    ) -> str:
        """
        Plot 8: Sorted Spectra across PCA, SVD, and LDA
        - Panel 1: PCA Sorted Eigenvalues (lambda_i) with max components min(N-1, D)=199
        - Panel 2: Truncated SVD Singular Values (sigma_i) with max components min(N, D)=200
        - Panel 3: LDA Discriminant Ratio (eta_i) with max components min(C-1, D)=39
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)

        # -------------------------------------------------------------
        # Panel 1: PCA Eigenvalues
        # -------------------------------------------------------------
        n_pca = len(pca_eigenvalues)
        k_pca = np.arange(1, n_pca + 1)

        ax1.plot(k_pca, pca_eigenvalues, color="#1f77b4", linewidth=2.2, label=r"PCA Eigenvalues ($\lambda_i$)")
        ax1.fill_between(k_pca, 0, pca_eigenvalues, color="#1f77b4", alpha=0.15)
        ax1.scatter(k_pca[:5], pca_eigenvalues[:5], color="#1f77b4", s=40, zorder=5)

        ax1.annotate(
            f"λ₁ = {pca_eigenvalues[0]:.1f}\nλ₂ = {pca_eigenvalues[1]:.1f}",
            xy=(1, pca_eigenvalues[0]),
            xytext=(18, pca_eigenvalues[0] - 6),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
            fontsize=8.5,
            fontweight="bold",
            color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff", edgecolor="#93c5fd", alpha=0.9)
        )

        ax1.axvline(x=n_pca, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Rank Limit: min(N-1, D) = {n_pca}")
        ax1.text(
            0.96, 0.28,
            f"Total Possible Components: {n_pca}\n• N = 200 training faces\n• Centering removes 1 d.o.f.\n• λ_{n_pca+1} ... λ_D ≡ 0",
            transform=ax1.transAxes,
            ha="right",
            fontsize=8,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
        )

        # Semi-log inset for scree decay
        ax1_ins = ax1.inset_axes([0.48, 0.48, 0.48, 0.45])
        ax1_ins.semilogy(k_pca, pca_eigenvalues, color="#1f77b4", linewidth=1.5)
        ax1_ins.set_title("Log₁₀ Scree Plot", fontsize=7.5, pad=3)
        ax1_ins.tick_params(axis="both", labelsize=6.5)
        ax1_ins.grid(True, linestyle=":", alpha=0.5)

        ax1.set_title(rf"PCA: Sorted Eigenvalues ($\lambda_i$)" + f"\n[Total Possible Components: {n_pca}]", fontsize=11, fontweight="bold", pad=10)
        ax1.set_xlabel("Principal Component Index (i)", fontsize=9.5, fontweight="semibold")
        ax1.set_ylabel(r"Eigenvalue / Sample Variance ($\lambda_i$)", fontsize=9.5, fontweight="semibold")
        ax1.set_xlim([0, n_pca + 5])
        ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # Panel 2: SVD Singular Values
        # -------------------------------------------------------------
        n_svd = len(svd_singular_values)
        k_svd = np.arange(1, n_svd + 1)

        ax2.plot(k_svd, svd_singular_values, color="#2ca02c", linewidth=2.2, label=r"SVD Singular Values ($\sigma_i$)")
        ax2.fill_between(k_svd, 0, svd_singular_values, color="#2ca02c", alpha=0.15)
        ax2.scatter(k_svd[:5], svd_singular_values[:5], color="#2ca02c", s=40, zorder=5)

        ax2.annotate(
            f"σ₁ = {svd_singular_values[0]:.1f}\n(Mean Illumination DC Offset)\nσ₂ = {svd_singular_values[1]:.1f}",
            xy=(1, svd_singular_values[0]),
            xytext=(22, svd_singular_values[0] - 130),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
            fontsize=8.5,
            fontweight="bold",
            color="#15803d",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4", edgecolor="#86efac", alpha=0.9)
        )

        ax2.axvline(x=n_svd, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Rank Limit: min(N, D) = {n_svd}")
        ax2.text(
            0.96, 0.28,
            f"Total Possible Components: {n_svd}\n• N = 200 training faces\n• Uncentered rank = min(200, 10304)\n• Retains full N dimensions",
            transform=ax2.transAxes,
            ha="right",
            fontsize=8,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
        )

        # Semi-log inset for singular decay
        ax2_ins = ax2.inset_axes([0.48, 0.48, 0.48, 0.45])
        ax2_ins.semilogy(k_svd, svd_singular_values, color="#2ca02c", linewidth=1.5)
        ax2_ins.set_title("Log₁₀ Singular Values", fontsize=7.5, pad=3)
        ax2_ins.tick_params(axis="both", labelsize=6.5)
        ax2_ins.grid(True, linestyle=":", alpha=0.5)

        ax2.set_title(rf"Truncated SVD: Singular Values ($\sigma_i$)" + f"\n[Total Possible Components: {n_svd}]", fontsize=11, fontweight="bold", pad=10)
        ax2.set_xlabel("Singular Vector Index (i)", fontsize=9.5, fontweight="semibold")
        ax2.set_ylabel(r"Singular Value Magnitude ($\sigma_i$)", fontsize=9.5, fontweight="semibold")
        ax2.set_xlim([0, n_svd + 5])
        ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # Panel 3: LDA Discriminant Ratio
        # -------------------------------------------------------------
        n_lda = len(lda_discriminant_ratios)
        k_lda = np.arange(1, n_lda + 1)
        lda_pct = lda_discriminant_ratios * 100

        ax3.bar(k_lda, lda_pct, color="#e11d48", alpha=0.75, edgecolor="#be123c", width=0.68, label="Discriminant Ratio (%)")
        ax3.plot(k_lda, lda_pct, color="#9f1239", linewidth=1.8, marker="o", markersize=3.5)

        ax3.annotate(
            f"η₁ = {lda_pct[0]:.1f}%\nη₂ = {lda_pct[1]:.1f}%\nη₃ = {lda_pct[2]:.1f}%",
            xy=(1, lda_pct[0]),
            xytext=(7, lda_pct[0] - 2.5),
            arrowprops=dict(arrowstyle="->", color="#be123c", lw=1.2),
            fontsize=8.5,
            fontweight="bold",
            color="#be123c",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2", edgecolor="#fecdd3", alpha=0.9)
        )

        ax3.axvline(x=n_lda, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Rank Limit: min(C-1, D) = {n_lda}")
        ax3.text(
            0.96, 0.28,
            f"Total Possible Components: {n_lda}\n• C = 40 classes (identities)\n• S_B rank = C - 1 = 39\n• Hard mathematical ceiling",
            transform=ax3.transAxes,
            ha="right",
            fontsize=8,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
        )

        # Secondary y-axis for cumulative discriminant ratio
        ax3_cum = ax3.twinx()
        ax3_cum.plot(k_lda, np.cumsum(lda_pct), color="#d97706", linewidth=1.8, linestyle="--", label="Cumulative Discriminant Ratio (%)")
        ax3_cum.set_ylabel("Cumulative Between-Class Variance (%)", fontsize=8.5, color="#b45309", fontweight="semibold")
        ax3_cum.set_ylim([0, 105])
        ax3_cum.tick_params(axis="y", labelsize=7.5, colors="#b45309")

        ax3.set_title(rf"LDA: Discriminant Ratio ($\eta_i$)" + f"\n[Total Possible Components: {n_lda}]", fontsize=11, fontweight="bold", pad=10)
        ax3.set_xlabel("Discriminant Dimension Index (i)", fontsize=9.5, fontweight="semibold")
        ax3.set_ylabel("Discriminant Ratio / Between-Class Variance (%)", fontsize=9.5, fontweight="semibold")
        ax3.set_xlim([0, n_lda + 2])
        ax3.set_ylim([0, max(lda_pct) * 1.15])
        ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(
            "Plot 8: Eigenspectrum, Singular Value Spectrum, and Discriminant Ratios with Theoretical Rank Bounds",
            fontsize=13,
            fontweight="bold",
            y=0.99
        )

        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_2d_embeddings(
        self,
        pca_2d: np.ndarray,
        svd_2d: np.ndarray,
        lda_2d: np.ndarray,
        labels: np.ndarray,
        selected_subjects: List[int],
        save_name: str = "plot5_2d_embeddings.png"
    ) -> str:
        """Plot 5: 2D subspace embeddings for PCA, SVD, and LDA highlighting class separability."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

        # Filter points belonging to selected subjects
        mask = np.isin(labels, selected_subjects)
        filtered_labels = labels[mask]

        palette = sns.color_palette("tab10", n_colors=len(selected_subjects))
        color_map = {subj: palette[i] for i, subj in enumerate(selected_subjects)}

        # Subplot 1: PCA
        for subj in selected_subjects:
            s_mask = labels == subj
            ax1.scatter(
                pca_2d[s_mask, 0], pca_2d[s_mask, 1],
                color=color_map[subj], label=f"Subj {subj+1}",
                s=45, alpha=0.85, edgecolors="none"
            )
        ax1.set_title("PCA (Components 1 & 2)\n[Unsupervised Variance]", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Principal Component 1", fontsize=10)
        ax1.set_ylabel("Principal Component 2", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Truncated SVD
        for subj in selected_subjects:
            s_mask = labels == subj
            ax2.scatter(
                svd_2d[s_mask, 0], svd_2d[s_mask, 1],
                color=color_map[subj], label=f"Subj {subj+1}",
                s=45, alpha=0.85, edgecolors="none"
            )
        ax2.set_title("Truncated SVD (Components 1 & 2)\n[Unsupervised Low-Rank]", fontsize=11, fontweight="bold")
        ax2.set_xlabel("SVD Component 1", fontsize=10)
        ax2.set_ylabel("SVD Component 2", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Subplot 3: LDA
        for subj in selected_subjects:
            s_mask = labels == subj
            ax3.scatter(
                lda_2d[s_mask, 0], lda_2d[s_mask, 1],
                color=color_map[subj], label=f"Subj {subj+1}",
                s=45, alpha=0.85, edgecolors="none"
            )
        ax3.set_title("LDA (Components 1 & 2)\n[Supervised Class-Discriminative]", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Discriminant Component 1", fontsize=10)
        ax3.set_ylabel("Discriminant Component 2", fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Place single common legend
        handles, leg_labels = ax3.get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="lower center", ncol=len(selected_subjects), bbox_to_anchor=(0.5, -0.05), frameon=True, facecolor="white", edgecolor="#cccccc")

        fig.suptitle("Plot 5: Two-Dimensional Feature Embeddings across 8 Subject Identities", fontsize=13, fontweight="bold")
        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_confusion_matrices(
        self,
        conf_mat_pca: np.ndarray,
        conf_mat_svd: np.ndarray,
        conf_mat_lda: np.ndarray,
        save_name: str = "plot6_confusion_matrices.png"
    ) -> str:
        """Plot 6: 40x40 Confusion matrices for best PCA, SVD, and LDA configurations."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)

        cmap = "Blues"

        # PCA
        im1 = ax1.imshow(conf_mat_pca, cmap=cmap, interpolation="nearest", aspect="auto")
        ax1.set_title("PCA Best Configuration\n(40 Identities)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Predicted Identity", fontsize=10)
        ax1.set_ylabel("True Identity", fontsize=10)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # SVD
        im2 = ax2.imshow(conf_mat_svd, cmap=cmap, interpolation="nearest", aspect="auto")
        ax2.set_title("Truncated SVD Best Configuration\n(40 Identities)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Predicted Identity", fontsize=10)
        ax2.set_ylabel("True Identity", fontsize=10)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # LDA
        im3 = ax3.imshow(conf_mat_lda, cmap="Reds", interpolation="nearest", aspect="auto")
        ax3.set_title("LDA Best Configuration\n(40 Identities)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Predicted Identity", fontsize=10)
        ax3.set_ylabel("True Identity", fontsize=10)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        fig.suptitle("Plot 6: 40 × 40 Identity Confusion Matrices on Held-Out Test Set", fontsize=13, fontweight="bold")
        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_computational_cost(
        self,
        methods: List[str],
        fit_times_ms: List[float],
        transform_times_us: List[float],
        save_name: str = "plot7_computational_cost.png"
    ) -> str:
        """Plot 7: Computational cost comparison (Fit time and Transform time per image)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        x = np.arange(len(methods))
        colors = ["#1f77b4", "#2ca02c", "#d62728"]

        # Left: Fit time
        bars1 = ax1.bar(x, fit_times_ms, color=colors, width=0.55, edgecolor="#555555", alpha=0.85)
        ax1.set_title("Training Fit Time (N = 200 Images)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Time (milliseconds)", fontsize=10, fontweight="semibold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=10, fontweight="semibold")
        ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02 * max(fit_times_ms), f"{h:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Right: Transform time per image
        bars2 = ax2.bar(x, transform_times_us, color=colors, width=0.55, edgecolor="#555555", alpha=0.85)
        ax2.set_title("Projection Time per Single Image", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Time (microseconds)", fontsize=10, fontweight="semibold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontsize=10, fontweight="semibold")
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02 * max(transform_times_us), f"{h:.1f} µs", ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.suptitle("Plot 7: Computational Efficiency Benchmarks across Subspace Methods", fontsize=13, fontweight="bold")
        out_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
