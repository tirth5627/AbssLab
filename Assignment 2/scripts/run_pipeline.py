#!/usr/bin/env python3
"""
Main Evaluation Pipeline Entrypoint.

Orchestrates the complete biometric evaluation workflow:
1. Load dataset and metadata
2. Preprocess images
3. Extract features
4. Generate pairs and compute scores
5. Evaluate performance metrics
6. Run fairness analysis
7. Generate visualizations and export report

Usage:
    python scripts/run_pipeline.py --variant REAL --feature-type deep --export-report
    python scripts/run_pipeline.py --variant REAL --feature-type handcrafted
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.loader import DatasetLoader
from src.preprocessing.enhancer import ImageEnhancer
from src.feature_extraction.handcrafted import HandcraftedExtractor
from src.feature_extraction.embeddings import DeepExtractor
from src.matching.pair_generator import PairGenerator
from src.matching.similarity import compute_scores
from src.evaluation.metrics import BiometricMetrics
from src.evaluation.fairness import FairnessAnalyzer
from src.visualization.roc_plots import (
    plot_roc_curves,
    plot_det_curves,
    plot_score_distributions,
    plot_fmr_fnmr_curves,
)
from src.visualization.demographic_plots import (
    plot_demographic_fmr_fnmr,
    plot_ddr_chart,
)


def setup_logging() -> None:
    """Configure logging."""
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log", mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run biometric evaluation pipeline on scraped facial dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant", type=str, default="REAL",
        choices=["REAL", "GAUSSIAN", "SALT_PEPPER"],
        help="Dataset variant to evaluate",
    )
    parser.add_argument(
        "--feature-type", type=str, default="handcrafted",
        choices=["handcrafted", "deep", "both"],
        help="Feature extraction method",
    )
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--export-report", action="store_true",
        help="Export evaluation results to JSON",
    )
    parser.add_argument(
        "--skip-fairness", action="store_true",
        help="Skip demographic fairness analysis",
    )

    args = parser.parse_args()

    os.chdir(project_root)
    setup_logging()
    logger = logging.getLogger("run_pipeline")

    # Load config
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    feature_cfg = config.get("features", {})
    matching_cfg = config.get("matching", {})
    eval_cfg = config.get("evaluation", {})

    start_time = time.time()

    # ──────────────────────────────────────────
    # Step 1: Load Dataset & Metadata
    # ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STEP 1: Loading dataset (variant={args.variant})")
    logger.info("=" * 60)

    loader = DatasetLoader(data_dir=args.data_dir)
    metadata = loader.load_metadata()
    subject_ids = loader.get_all_subject_ids()
    logger.info(f"Loaded {len(subject_ids)} subjects, {loader.count_images(args.variant)} images")

    # Build metadata lookup
    metadata_lookup = {m["subject_id"]: m for m in metadata}

    # ──────────────────────────────────────────
    # Step 2: Preprocessing
    # ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing images")
    logger.info("=" * 60)

    preproc_cfg = config.get("preprocessing", {})
    clahe_cfg = preproc_cfg.get("clahe", {})

    enhancer = ImageEnhancer(
        target_size=tuple(config.get("image", {}).get("target_size", [256, 256])),
        clip_limit=clahe_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        normalize=preproc_cfg.get("normalize", True),
    )

    # ──────────────────────────────────────────
    # Step 3: Feature Extraction
    # ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STEP 3: Extracting features (type={args.feature_type})")
    logger.info("=" * 60)

    feature_types_to_run = (
        ["handcrafted", "deep"] if args.feature_type == "both"
        else [args.feature_type]
    )

    all_results = {}

    for ft in feature_types_to_run:
        logger.info(f"\n--- Feature type: {ft} ---")

        # Create extractor
        if ft == "handcrafted":
            hc_cfg = feature_cfg.get("handcrafted", {})
            extractor = HandcraftedExtractor(
                histogram_bins=hc_cfg.get("histogram_bins", 32),
                lbp_radius=hc_cfg.get("lbp_radius", 1),
                lbp_n_points=hc_cfg.get("lbp_n_points", 8),
            )
        else:
            deep_cfg = feature_cfg.get("deep", {})
            extractor = DeepExtractor(
                model_name=deep_cfg.get("model", "facenet"),
                pretrained=deep_cfg.get("pretrained", "vggface2"),
                device=deep_cfg.get("device", "auto"),
                embedding_dim=deep_cfg.get("embedding_dim", 512),
            )

        # Extract features for all subjects
        features = {}  # subject_id -> {1: vector, 2: vector}

        for subject_id in tqdm(subject_ids, desc=f"Extracting {ft} features"):
            img1 = loader.load_image(subject_id, 1, args.variant)
            img2 = loader.load_image(subject_id, 2, args.variant)

            features[subject_id] = {}

            if img1 is not None:
                enhanced1 = enhancer.enhance(img1)
                features[subject_id][1] = extractor.extract_normalized(enhanced1)

            if img2 is not None:
                enhanced2 = enhancer.enhance(img2)
                features[subject_id][2] = extractor.extract_normalized(enhanced2)

        logger.info(f"Extracted features for {len(features)} subjects")

        # ──────────────────────────────────────────
        # Step 4: Pair Generation & Scoring
        # ──────────────────────────────────────────
        logger.info("STEP 4: Generating pairs and computing scores")

        pair_gen = PairGenerator(
            subject_ids=subject_ids,
            num_impostor_pairs=matching_cfg.get("num_impostor_pairs", 1000),
            seed=matching_cfg.get("random_seed", 42),
        )

        genuine_pairs, impostor_pairs = pair_gen.generate_all_pairs()

        genuine_scores = compute_scores(genuine_pairs, features)
        impostor_scores = compute_scores(impostor_pairs, features)

        logger.info(
            f"Scores — Genuine: mean={genuine_scores.mean():.4f} (n={len(genuine_scores)}), "
            f"Impostor: mean={impostor_scores.mean():.4f} (n={len(impostor_scores)})"
        )

        # ──────────────────────────────────────────
        # Step 5: Performance Metrics
        # ──────────────────────────────────────────
        logger.info("STEP 5: Computing performance metrics")

        metrics = BiometricMetrics(
            threshold_step=eval_cfg.get("threshold_step", 0.001),
            threshold_range=tuple(eval_cfg.get("threshold_range", [0.0, 1.0])),
            fmr_targets=eval_cfg.get("fmr_targets", [0.001, 0.0001]),
        )

        eval_results = metrics.evaluate_all(genuine_scores, impostor_scores)
        logger.info(f"EER = {eval_results['eer']:.4f}, AUC = {eval_results['auc_roc']:.4f}")

        # ──────────────────────────────────────────
        # Step 6: Fairness Analysis
        # ──────────────────────────────────────────
        fairness_results = None
        if not args.skip_fairness:
            logger.info("STEP 6: Demographic fairness analysis")

            analyzer = FairnessAnalyzer(metrics=metrics)
            overall_fnmr = metrics.compute_fnmr(
                genuine_scores, eval_results["eer_threshold"]
            )

            fairness_results = analyzer.full_fairness_analysis(
                genuine_pairs=genuine_pairs,
                impostor_pairs=impostor_pairs,
                genuine_scores=genuine_scores,
                impostor_scores=impostor_scores,
                metadata_lookup=metadata_lookup,
                global_threshold=eval_results["eer_threshold"],
                overall_fnmr=overall_fnmr,
            )

        # ──────────────────────────────────────────
        # Step 7: Visualization
        # ──────────────────────────────────────────
        logger.info("STEP 7: Generating visualizations")

        fig_dir = Path("outputs/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"{args.variant}_{ft}"

        # Score distributions
        plot_score_distributions(
            genuine_scores, impostor_scores,
            variant=f"{args.variant} ({ft})",
            eer_threshold=eval_results["eer_threshold"],
            output_path=str(fig_dir / f"score_distributions_{suffix}.png"),
        )

        # ROC curve
        fpr, tpr, _ = metrics.compute_roc_curve(genuine_scores, impostor_scores)
        plot_roc_curves(
            {f"{args.variant} ({ft})": (fpr, tpr, eval_results["auc_roc"])},
            output_path=str(fig_dir / f"roc_curve_{suffix}.png"),
        )

        # FMR/FNMR curves
        sweep = metrics.threshold_sweep(genuine_scores, impostor_scores)
        plot_fmr_fnmr_curves(
            sweep["thresholds"], sweep["fmr"], sweep["fnmr"],
            eer_threshold=eval_results["eer_threshold"],
            variant=f"{args.variant} ({ft})",
            output_path=str(fig_dir / f"fmr_fnmr_{suffix}.png"),
        )

        # DET curve
        plot_det_curves(
            {f"{args.variant} ({ft})": (sweep["fmr"], sweep["fnmr"])},
            output_path=str(fig_dir / f"det_curve_{suffix}.png"),
        )

        # Fairness plots
        if fairness_results:
            plot_demographic_fmr_fnmr(
                fairness_results,
                output_path=str(fig_dir / f"demographic_fairness_{suffix}.png"),
            )
            plot_ddr_chart(
                fairness_results,
                output_path=str(fig_dir / f"ddr_chart_{suffix}.png"),
            )

        # Store results
        result_key = f"{args.variant}_{ft}"
        all_results[result_key] = {
            "variant": args.variant,
            "feature_type": ft,
            "metrics": eval_results,
            "fairness": _serialize_fairness(fairness_results) if fairness_results else None,
        }

    # ──────────────────────────────────────────
    # Export Report
    # ──────────────────────────────────────────
    if args.export_report:
        report_dir = Path("outputs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation_results.json"

        # Merge with existing results if present
        existing = {}
        if report_path.exists():
            with open(report_path, "r") as f:
                existing = json.load(f)

        existing.update(all_results)

        with open(report_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        logger.info(f"Exported results to {report_path}")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Pipeline complete! Total time: {elapsed:.1f}s")
    logger.info("=" * 60)


def _serialize_fairness(fairness_results: dict) -> dict:
    """Convert fairness results to JSON-serializable format."""
    serialized = {}
    for attr, data in fairness_results.items():
        serialized[attr] = {
            "group_metrics": {},
            "ddr": {},
        }
        for group, metrics in data.get("group_metrics", {}).items():
            serialized[attr]["group_metrics"][group] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
            }
        for group, val in data.get("ddr", {}).items():
            serialized[attr]["ddr"][group] = (
                float(val) if isinstance(val, (np.floating, float)) else val
            )
    return serialized


if __name__ == "__main__":
    main()
