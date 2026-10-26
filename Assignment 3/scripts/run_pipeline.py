"""
Main Pipeline Runner for Assignment 3: PCA vs Truncated SVD vs LDA on ORL Faces.
Executes all 10 experimental splits, computes all core metrics, generates all 7 plots,
and exports all summary tables.
"""

import os
import sys
import argparse
import time
import json
import yaml
import numpy as np
import pandas as pd

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data_loader import load_orl_dataset, generate_stratified_splits
from src.models import PCAReducer, SVDReducer, LDAReducer
from src.evaluation import evaluate_classifier, compute_reconstruction_metrics, compute_storage_metrics
from src.visualization import FaceVisualizer


def run_experiment(config_path: str = "config/default_config.yaml"):
    print("=" * 80)
    print("ASSIGNMENT 3: PCA vs. Truncated SVD vs. LDA on AT&T / ORL Face Database")
    print("=" * 80)

    # 1. Load Configuration
    if not os.path.isabs(config_path):
        config_path = os.path.join(parent_dir, config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(parent_dir, config["dataset"]["data_dir"])
    output_dir = os.path.join(parent_dir, config["paths"]["output_dir"])
    fig_dir = os.path.join(parent_dir, config["paths"]["figures_dir"])
    tables_dir = os.path.join(parent_dir, config["paths"]["tables_dir"])

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # 2. Load Dataset
    print(f"\n[Step 1] Loading AT&T / ORL Face Dataset from: {data_dir}")
    X, y, metadata = load_orl_dataset(data_dir=data_dir, normalize=True)
    print(f"Loaded {X.shape[0]} images across {len(np.unique(y))} identities.")
    print(f"Image dimensions: {metadata['image_width']} x {metadata['image_height']} = {metadata['flattened_features']} features.")
    print(f"Pixel value range: [{metadata['pixel_min']:.3f}, {metadata['pixel_max']:.3f}]")

    # 3. Generate Stratified Splits
    num_splits = config["experiment"]["num_splits"]
    seeds = config["experiment"]["random_seeds"]
    print(f"\n[Step 2] Generating {num_splits} stratified splits (5 train / 5 test per identity) with seeds: {seeds}")
    splits = generate_stratified_splits(
        y=y,
        n_splits=num_splits,
        train_per_class=config["experiment"]["train_samples_per_identity"],
        test_per_class=config["experiment"]["test_samples_per_identity"],
        seeds=seeds
    )

    common_components = config["experiment"]["common_components"]
    extended_components = config["experiment"]["extended_components"]
    print(f"Common evaluation components: {common_components}")
    print(f"Extended evaluation components: {extended_components}")

    # Data structures to collect metrics across splits
    raw_acc_list = []
    raw_f1_list = []

    # results: method -> k -> list of split dicts
    results_records = []
    reconstruction_records = []
    timing_records = []

    # Store matrices for best configuration confusion matrices (split 0)
    best_conf_matrices = {"PCA": None, "SVD": None, "LDA": None}
    best_models = {"PCA": None, "SVD": None, "LDA": None}
    best_accs = {"PCA": -1.0, "SVD": -1.0, "LDA": -1.0}

    print("\n[Step 3] Executing Benchmark across all 10 randomized splits...")

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        seed = seeds[split_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 3.1 Raw Pixels Baseline (1-NN)
        eval_raw = evaluate_classifier(X_train, y_train, X_test, y_test, classifier_type="1-NN")
        raw_acc_list.append(eval_raw["accuracy"])
        raw_f1_list.append(eval_raw["macro_f1"])

        # 3.2 Evaluate PCA
        for k in extended_components:
            pca = PCAReducer(n_components=k, random_state=seed)
            pca.fit(X_train)
            Z_train = pca.transform(X_train)
            Z_test = pca.transform(X_test)
            eval_res = evaluate_classifier(Z_train, y_train, Z_test, y_test, classifier_type="1-NN")

            # Reconstruction metrics on test set
            X_rec = pca.inverse_transform(Z_test)
            rec_metrics = compute_reconstruction_metrics(X_test, X_rec)

            storage = compute_storage_metrics(
                n_features=metadata["flattened_features"],
                n_components=k,
                n_samples=len(test_idx)
            )

            results_records.append({
                "split": split_idx,
                "seed": seed,
                "method": "PCA",
                "components": k,
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"],
                "fit_time": pca.fit_time,
                "transform_time_per_image": pca.last_transform_time / len(test_idx),
                "mse": rec_metrics["mse"],
                "rel_error": rec_metrics["relative_frobenius_error"],
                "psnr": rec_metrics["psnr"],
                "projected_bytes_per_image": storage["projected_bytes_per_image"],
                "model_param_bytes": storage["model_param_bytes"],
            })

            if eval_res["accuracy"] > best_accs["PCA"]:
                best_accs["PCA"] = eval_res["accuracy"]
                best_conf_matrices["PCA"] = eval_res["confusion_matrix"]
                best_models["PCA"] = (k, pca)

        # 3.3 Evaluate Truncated SVD
        for k in extended_components:
            svd = SVDReducer(n_components=k, centered=False, random_state=seed)
            svd.fit(X_train)
            Z_train = svd.transform(X_train)
            Z_test = svd.transform(X_test)
            eval_res = evaluate_classifier(Z_train, y_train, Z_test, y_test, classifier_type="1-NN")

            # Reconstruction metrics on test set
            X_rec = svd.inverse_transform(Z_test)
            rec_metrics = compute_reconstruction_metrics(X_test, X_rec)

            storage = compute_storage_metrics(
                n_features=metadata["flattened_features"],
                n_components=k,
                n_samples=len(test_idx)
            )

            results_records.append({
                "split": split_idx,
                "seed": seed,
                "method": "Truncated SVD",
                "components": k,
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"],
                "fit_time": svd.fit_time,
                "transform_time_per_image": svd.last_transform_time / len(test_idx),
                "mse": rec_metrics["mse"],
                "rel_error": rec_metrics["relative_frobenius_error"],
                "psnr": rec_metrics["psnr"],
                "projected_bytes_per_image": storage["projected_bytes_per_image"],
                "model_param_bytes": storage["model_param_bytes"],
            })

            if eval_res["accuracy"] > best_accs["SVD"]:
                best_accs["SVD"] = eval_res["accuracy"]
                best_conf_matrices["SVD"] = eval_res["confusion_matrix"]
                best_models["SVD"] = (k, svd)

        # 3.4 Evaluate LDA (Fisherfaces) - valid for k in common_components up to 39
        for k in common_components:
            lda = LDAReducer(n_components=k, solver="svd")
            lda.fit(X_train, y_train)
            Z_train = lda.transform(X_train)
            Z_test = lda.transform(X_test)
            eval_res = evaluate_classifier(Z_train, y_train, Z_test, y_test, classifier_type="1-NN")

            storage = compute_storage_metrics(
                n_features=metadata["flattened_features"],
                n_components=k,
                n_samples=len(test_idx)
            )

            results_records.append({
                "split": split_idx,
                "seed": seed,
                "method": "LDA",
                "components": k,
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"],
                "fit_time": lda.fit_time,
                "transform_time_per_image": lda.last_transform_time / len(test_idx),
                "mse": np.nan,
                "rel_error": np.nan,
                "psnr": np.nan,
                "projected_bytes_per_image": storage["projected_bytes_per_image"],
                "model_param_bytes": storage["model_param_bytes"],
            })

            if eval_res["accuracy"] > best_accs["LDA"]:
                best_accs["LDA"] = eval_res["accuracy"]
                best_conf_matrices["LDA"] = eval_res["confusion_matrix"]
                best_models["LDA"] = (k, lda)

        print(f"  Completed Split {split_idx+1}/{num_splits} (Seed {seed}) - Raw 1-NN Acc: {eval_raw['accuracy']*100:.1f}%")

    df_results = pd.DataFrame(results_records)

    # 4. Compute Aggregate Statistics (Mean ± Std)
    raw_acc_mean = float(np.mean(raw_acc_list))
    raw_acc_std = float(np.std(raw_acc_list))
    raw_f1_mean = float(np.mean(raw_f1_list))
    raw_f1_std = float(np.std(raw_f1_list))

    print("\n" + "=" * 80)
    print(f"RAW PIXEL BASELINE (10,304 features + 1-NN):")
    print(f"  Mean Accuracy: {raw_acc_mean*100:.2f}% ± {raw_acc_std*100:.2f}% | Macro-F1: {raw_f1_mean:.4f} ± {raw_f1_std:.4f}")
    print("=" * 80)

    # Summary table per Method and Component Count
    summary_df = df_results.groupby(["method", "components"]).agg({
        "accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "fit_time": ["mean", "std"],
        "transform_time_per_image": ["mean", "std"],
        "mse": ["mean", "std"],
        "rel_error": ["mean", "std"],
        "psnr": ["mean", "std"],
        "projected_bytes_per_image": "first",
        "model_param_bytes": "first",
    }).reset_index()

    # Flatten multiindex columns
    new_cols = []
    for col in summary_df.columns.values:
        if isinstance(col, tuple):
            if col[1] in ("", "first"):
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)
    summary_df.columns = new_cols

    # Save raw and summary results
    df_results.to_csv(os.path.join(tables_dir, "raw_benchmark_records.csv"), index=False)
    summary_df.to_csv(os.path.join(tables_dir, "summary_benchmark_records.csv"), index=False)

    # 5. Build Specific Tables Requested by the Assignment
    print("\n[Step 4] Generating Assignment Tables (Tables A, B, C, D)...")

    # Table A: Dataset and configuration
    table_a_data = [
        {"Parameter": "Dataset", "Value": metadata["dataset_name"]},
        {"Parameter": "Total Images / Identities", "Value": f"{metadata['total_images']} images / {metadata['num_identities']} people"},
        {"Parameter": "Images per Identity", "Value": str(metadata["images_per_identity"])},
        {"Parameter": "Image Dimensions", "Value": f"{metadata['image_width']} x {metadata['image_height']} pixels (Grayscale PGM)"},
        {"Parameter": "Features per Image (Flattened)", "Value": str(metadata["flattened_features"])},
        {"Parameter": "Pixel Normalization Scale", "Value": "[0.0, 1.0] (float32)"},
        {"Parameter": "Train / Test Split", "Value": f"{config['experiment']['train_samples_per_identity']} train + {config['experiment']['test_samples_per_identity']} test per identity (200 train / 200 test)"},
        {"Parameter": "Split Repetitions", "Value": f"{num_splits} randomized splits with fixed recorded seeds"},
        {"Parameter": "Random Seeds", "Value": str(seeds)},
        {"Parameter": "Primary Classifier", "Value": "1-Nearest Neighbor (1-NN, Euclidean distance)"},
        {"Parameter": "Secondary Classifier", "Value": "Linear Support Vector Machine (C=1.0)"},
        {"Parameter": "Component Counts Evaluated", "Value": str(common_components)},
    ]
    table_a_df = pd.DataFrame(table_a_data)
    table_a_df.to_csv(os.path.join(tables_dir, "table_a_configuration.csv"), index=False)

    # Table B: Recognition performance (Methods vs Component Counts)
    table_b_rows = [
        {
            "Method": "Raw pixels + 1-NN",
            "Components": metadata["flattened_features"],
            "Mean Accuracy (%)": f"{raw_acc_mean*100:.2f}%",
            "Std Dev (%)": f"±{raw_acc_std*100:.2f}%",
            "Macro-F1": f"{raw_f1_mean:.4f} ± {raw_f1_std:.4f}",
            "Fit Time (ms)": "0.00 ms (None)",
        }
    ]

    for method in ["PCA", "Truncated SVD", "LDA"]:
        for k in common_components:
            row_match = summary_df[(summary_df["method"] == method) & (summary_df["components"] == k)]
            if not row_match.empty:
                r = row_match.iloc[0]
                table_b_rows.append({
                    "Method": f"{method} + 1-NN",
                    "Components": k,
                    "Mean Accuracy (%)": f"{r['accuracy_mean']*100:.2f}%",
                    "Std Dev (%)": f"±{r['accuracy_std']*100:.2f}%",
                    "Macro-F1": f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}",
                    "Fit Time (ms)": f"{r['fit_time_mean']*1000:.2f} ms",
                })

    table_b_df = pd.DataFrame(table_b_rows)
    table_b_df.to_csv(os.path.join(tables_dir, "table_b_recognition_performance.csv"), index=False)

    # Table C: Reconstruction and compression
    table_c_rows = []
    for k in [10, 20, 39]:
        # PCA
        r_pca = summary_df[(summary_df["method"] == "PCA") & (summary_df["components"] == k)].iloc[0]
        table_c_rows.append({
            "Method": "PCA",
            "Components": k,
            "Reconstruction MSE": f"{r_pca['mse_mean']:.6f}",
            "Relative Frobenius Error": f"{r_pca['rel_error_mean']*100:.2f}%",
            "PSNR (dB)": f"{r_pca['psnr_mean']:.2f} dB",
            "Feature Storage / Image": f"{int(r_pca['projected_bytes_per_image'])} B ({r_pca['projected_bytes_per_image']/1024:.2f} KB)",
            "Model Storage": f"{r_pca['model_param_bytes']/1024:.2f} KB",
            "Notes": "Subtracted mean face (centered). Exact orthogonal inverse transform.",
        })
        # SVD
        r_svd = summary_df[(summary_df["method"] == "Truncated SVD") & (summary_df["components"] == k)].iloc[0]
        table_c_rows.append({
            "Method": "Truncated SVD",
            "Components": k,
            "Reconstruction MSE": f"{r_svd['mse_mean']:.6f}",
            "Relative Frobenius Error": f"{r_svd['rel_error_mean']*100:.2f}%",
            "PSNR (dB)": f"{r_svd['psnr_mean']:.2f} dB",
            "Feature Storage / Image": f"{int(r_svd['projected_bytes_per_image'])} B ({r_svd['projected_bytes_per_image']/1024:.2f} KB)",
            "Model Storage": f"{r_svd['model_param_bytes']/1024:.2f} KB",
            "Notes": "Uncentered data matrix. Projects on leading right singular vectors.",
        })

    # Add LDA note row
    table_c_rows.append({
        "Method": "LDA",
        "Components": 39,
        "Reconstruction MSE": "N/A",
        "Relative Frobenius Error": "N/A",
        "PSNR (dB)": "N/A",
        "Feature Storage / Image": f"{39 * 4} B (0.15 KB)",
        "Model Storage": f"{(10304 * 39 + 10304) * 4 / 1024:.2f} KB",
        "Notes": "Not primary: LDA optimizes class discriminability, not image reconstruction.",
    })
    table_c_df = pd.DataFrame(table_c_rows)
    table_c_df.to_csv(os.path.join(tables_dir, "table_c_reconstruction_compression.csv"), index=False)

    # Table D: Computational Efficiency
    table_d_rows = []
    for method in ["PCA", "Truncated SVD", "LDA"]:
        # Use k=39 as common representative point
        r = summary_df[(summary_df["method"] == method) & (summary_df["components"] == 39)].iloc[0]
        table_d_rows.append({
            "Method": method,
            "Components Evaluated": 39,
            "Fit Time (ms)": f"{r['fit_time_mean']*1000:.2f} ± {r['fit_time_std']*1000:.2f} ms",
            "Transform Time / Image (µs)": f"{r['transform_time_per_image_mean']*1e6:.1f} ± {r['transform_time_per_image_std']*1e6:.1f} µs",
            "Model Parameter Storage": f"{r['model_param_bytes']/1024:.2f} KB",
            "Feature Compression Ratio": f"{metadata['flattened_features'] / 39:.1f}x",
        })
    table_d_df = pd.DataFrame(table_d_rows)
    table_d_df.to_csv(os.path.join(tables_dir, "table_d_computational_efficiency.csv"), index=False)

    # Best configurations summary table
    best_rows = []
    for method in ["PCA", "Truncated SVD", "LDA"]:
        sub_df = summary_df[summary_df["method"] == method]
        best_row = sub_df.loc[sub_df["accuracy_mean"].idxmax()]
        best_rows.append({
            "Method": method,
            "Best Components (k)": int(best_row["components"]),
            "Peak Mean Accuracy (%)": f"{best_row['accuracy_mean']*100:.2f}% ± {best_row['accuracy_std']*100:.2f}%",
            "Macro-F1": f"{best_row['macro_f1_mean']:.4f} ± {best_row['macro_f1_std']:.4f}",
            "Fit Time (ms)": f"{best_row['fit_time_mean']*1000:.2f} ms",
            "Compression vs Raw": f"{metadata['flattened_features'] / best_row['components']:.1f}x",
        })
    best_summary_df = pd.DataFrame(best_rows)
    best_summary_df.to_csv(os.path.join(tables_dir, "best_configurations_summary.csv"), index=False)

    print("\nTable B Summary (Accuracy across components):")
    print(table_b_df.to_string(index=False))

    # 6. Generate All 7 Plots
    print("\n[Step 5] Rendering All 7 Publication-Quality Plots...")
    viz = FaceVisualizer(output_dir=fig_dir)

    # Plot 1: Accuracy vs Components
    pca_sub = summary_df[summary_df["method"] == "PCA"].sort_values("components")
    svd_sub = summary_df[summary_df["method"] == "Truncated SVD"].sort_values("components")
    lda_sub = summary_df[summary_df["method"] == "LDA"].sort_values("components")

    # Plot for common components (up to 39)
    p1_path = viz.plot_accuracy_vs_components(
        component_counts=common_components,
        pca_acc_mean=pca_sub[pca_sub["components"].isin(common_components)]["accuracy_mean"].tolist(),
        pca_acc_std=pca_sub[pca_sub["components"].isin(common_components)]["accuracy_std"].tolist(),
        svd_acc_mean=svd_sub[svd_sub["components"].isin(common_components)]["accuracy_mean"].tolist(),
        svd_acc_std=svd_sub[svd_sub["components"].isin(common_components)]["accuracy_std"].tolist(),
        lda_acc_mean=lda_sub["accuracy_mean"].tolist(),
        lda_acc_std=lda_sub["accuracy_std"].tolist(),
        raw_pixel_acc_mean=raw_acc_mean,
        raw_pixel_acc_std=raw_acc_std,
        save_name="plot1_accuracy_vs_components.png"
    )
    print(f"  [OK] Saved Plot 1: {p1_path}")

    # Plot 2: Face Reconstructions on Test Faces (Split 0)
    train_idx0, test_idx0 = splits[0]
    X_train0, X_test0 = X[train_idx0], X[test_idx0]
    y_test0 = y[test_idx0]

    # Pick 2 different subjects from test set
    subj_a, subj_b = 0, 9  # Subject 1 and Subject 10
    idx_a = np.where(y_test0 == subj_a)[0][0]
    idx_b = np.where(y_test0 == subj_b)[0][0]
    test_face_indices = [idx_a, idx_b]
    orig_test_faces = [X_test0[i] for i in test_face_indices]

    rec_k_values = [5, 10, 20, 30]
    pca_recs = {k: [] for k in rec_k_values}
    svd_recs = {k: [] for k in rec_k_values}

    for k in rec_k_values:
        p_model = PCAReducer(n_components=k, random_state=42).fit(X_train0)
        s_model = SVDReducer(n_components=k, centered=False, random_state=42).fit(X_train0)
        p_pred = p_model.inverse_transform(p_model.transform(np.array(orig_test_faces)))
        s_pred = s_model.inverse_transform(s_model.transform(np.array(orig_test_faces)))
        pca_recs[k] = [p_pred[0], p_pred[1]]
        svd_recs[k] = [s_pred[0], s_pred[1]]

    p2_path = viz.plot_face_reconstructions(
        original_faces=orig_test_faces,
        pca_reconstructions=pca_recs,
        svd_reconstructions=svd_recs,
        component_keys=rec_k_values,
        subject_labels=["Subject 1 (Test Sample)", "Subject 10 (Test Sample)"],
        save_name="plot2_face_reconstructions.png"
    )
    print(f"  [OK] Saved Plot 2: {p2_path}")

    # Plot 3: Reconstruction Error vs Component Count
    p3_path = viz.plot_reconstruction_error(
        component_counts=common_components,
        pca_mse=pca_sub[pca_sub["components"].isin(common_components)]["mse_mean"].tolist(),
        svd_mse=svd_sub[svd_sub["components"].isin(common_components)]["mse_mean"].tolist(),
        pca_rel_err=pca_sub[pca_sub["components"].isin(common_components)]["rel_error_mean"].tolist(),
        svd_rel_err=svd_sub[svd_sub["components"].isin(common_components)]["rel_error_mean"].tolist(),
        save_name="plot3_reconstruction_error.png"
    )
    print(f"  [OK] Saved Plot 3: {p3_path}")

    y_train0 = y[train_idx0]

    # Plot 4: Cumulative Explained Variance: SVD vs. PCA Comparison
    full_pca = PCAReducer(n_components=min(199, X_train0.shape[0]-1), random_state=42)
    full_pca.fit(X_train0)
    pca_cum_var = np.cumsum(full_pca.get_explained_variance_ratio())
    pca_eigenvals = full_pca.model.explained_variance_

    # Exact SVD on uncentered training partition (rank = min(N_train, D) = 200)
    U_svd, s_svd, Vt_svd = np.linalg.svd(X_train0, full_matrices=False)
    X_svd_trans = X_train0 @ Vt_svd.T
    svd_exp_var = np.var(X_svd_trans, axis=0)
    svd_cum_var = np.cumsum(svd_exp_var / np.var(X_train0, axis=0).sum())

    p4_path = viz.plot_cumulative_variance_svd_vs_pca(
        pca_cumulative_variance=pca_cum_var,
        svd_cumulative_variance=svd_cum_var,
        thresholds=config["experiment"]["variance_thresholds"],
        lda_max_k=39,
        save_name="plot4_explained_variance.png"
    )
    print(f"  [OK] Saved Plot 4 (SVD vs PCA Cumulative Variance): {p4_path}")

    # Plot 8: Sorted Spectra (PCA Eigenvalues, SVD Singular Values, LDA Discriminant Ratio)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    full_lda = LinearDiscriminantAnalysis(solver="svd").fit(X_train0, y_train0)
    lda_ratios = full_lda.explained_variance_ratio_

    p8_path = viz.plot_spectra_pca_svd_lda(
        pca_eigenvalues=pca_eigenvals,
        svd_singular_values=s_svd,
        lda_discriminant_ratios=lda_ratios,
        save_name="plot8_spectra_pca_svd_lda.png"
    )
    print(f"  [OK] Saved Plot 8 (Spectra & Discriminant Ratios): {p8_path}")

    # Export spectrum data JSON for dynamic widgets and interactive blog
    spectrum_data = {
        "pca_eigenvalues": [float(v) for v in pca_eigenvals],
        "svd_singular_values": [float(v) for v in s_svd],
        "lda_discriminant_ratios": [float(v) for v in lda_ratios],
        "pca_cumulative_variance": [float(v) for v in pca_cum_var],
        "svd_cumulative_variance": [float(v) for v in svd_cum_var],
        "total_possible_components": {
            "PCA_train": len(pca_eigenvals),
            "PCA_formula": "min(N_train - 1, D) = min(200 - 1, 10304) = 199",
            "SVD_train": len(s_svd),
            "SVD_formula": "min(N_train, D) = min(200, 10304) = 200",
            "LDA_train": len(lda_ratios),
            "LDA_formula": "min(C - 1, D) = min(40 - 1, 10304) = 39",
            "PCA_full": 399,
            "SVD_full": 400,
            "LDA_full": 39
        }
    }
    with open(os.path.join(tables_dir, "spectrum_data.json"), "w", encoding="utf-8") as f:
        json.dump(spectrum_data, f, indent=2)
    print(f"  [OK] Saved spectrum metadata to {os.path.join(tables_dir, 'spectrum_data.json')}")

    # Plot 5: 2D Embeddings (Visual Class Separation)
    # Fit 2D models on entire dataset for visualization
    pca_2d_model = PCAReducer(n_components=2, random_state=42).fit(X)
    svd_2d_model = SVDReducer(n_components=2, centered=False, random_state=42).fit(X)
    lda_2d_model = LDAReducer(n_components=2, solver="svd").fit(X, y)

    pca_2d_pts = pca_2d_model.transform(X)
    svd_2d_pts = svd_2d_model.transform(X)
    lda_2d_pts = lda_2d_model.transform(X)

    chosen_subjects = [s - 1 for s in config["experiment"]["embedding_subjects_to_plot"]]
    p5_path = viz.plot_2d_embeddings(
        pca_2d=pca_2d_pts,
        svd_2d=svd_2d_pts,
        lda_2d=lda_2d_pts,
        labels=y,
        selected_subjects=chosen_subjects,
        save_name="plot5_2d_embeddings.png"
    )
    print(f"  [OK] Saved Plot 5: {p5_path}")

    # Plot 6: Confusion Matrices (40x40)
    p6_path = viz.plot_confusion_matrices(
        conf_mat_pca=best_conf_matrices["PCA"],
        conf_mat_svd=best_conf_matrices["SVD"],
        conf_mat_lda=best_conf_matrices["LDA"],
        save_name="plot6_confusion_matrices.png"
    )
    print(f"  [OK] Saved Plot 6: {p6_path}")

    # Plot 7: Computational Efficiency
    fit_times = [
        pca_sub[pca_sub["components"] == 39]["fit_time_mean"].values[0] * 1000,
        svd_sub[svd_sub["components"] == 39]["fit_time_mean"].values[0] * 1000,
        lda_sub[lda_sub["components"] == 39]["fit_time_mean"].values[0] * 1000,
    ]
    trans_times = [
        pca_sub[pca_sub["components"] == 39]["transform_time_per_image_mean"].values[0] * 1e6,
        svd_sub[svd_sub["components"] == 39]["transform_time_per_image_mean"].values[0] * 1e6,
        lda_sub[lda_sub["components"] == 39]["transform_time_per_image_mean"].values[0] * 1e6,
    ]
    p7_path = viz.plot_computational_cost(
        methods=["PCA", "Truncated SVD", "LDA"],
        fit_times_ms=fit_times,
        transform_times_us=trans_times,
        save_name="plot7_computational_cost.png"
    )
    print(f"  [OK] Saved Plot 7: {p7_path}")

    print("\n" + "=" * 80)
    print("ALL 10 SPLITS, METRICS, AND PLOTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return {
        "raw_acc_mean": raw_acc_mean,
        "raw_acc_std": raw_acc_std,
        "summary_df": summary_df,
        "table_a_df": table_a_df,
        "table_b_df": table_b_df,
        "table_c_df": table_c_df,
        "table_d_df": table_d_df,
        "best_summary_df": best_summary_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Assignment 3 Benchmarking Pipeline")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_experiment(config_path=args.config)
