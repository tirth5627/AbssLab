#!/usr/bin/env python3
"""
Robustness Evaluation Script.

Runs the evaluation pipeline across all dataset variants (REAL, GAUSSIAN,
SALT_PEPPER) and computes performance degradation metrics.

Usage:
    python scripts/evaluate_robustness.py --all-variants
    python scripts/evaluate_robustness.py --all-variants --feature-type deep
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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.loader import DatasetLoader
from src.preprocessing.enhancer import ImageEnhancer
from src.feature_extraction.handcrafted import HandcraftedExtractor
from src.feature_extraction.embeddings import DeepExtractor
from src.matching.pair_generator import PairGenerator
from src.matching.similarity import compute_scores
from src.evaluation.metrics import BiometricMetrics
from src.visualization.roc_plots import (
    plot_roc_curves,
    plot_score_distributions,
    plot_det_curves,
)
from src.visualization.demographic_plots import plot_noise_degradation


def setup_logging():
    """Configure logging."""
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "robustness.log", mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_variant(
    variant: str,
    loader: DatasetLoader,
    enhancer: ImageEnhancer,
    extractor,
    subject_ids: list,
    matching_cfg: dict,
    eval_cfg: dict,
    logger,
) -> dict:
    """
    Run full evaluation for a single dataset variant.

    Returns dictionary with all computed metrics.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating variant: {variant}")
    logger.info(f"{'='*50}")

    # Extract features
    features = {}
    for subject_id in tqdm(subject_ids, desc=f"Features [{variant}]"):
        img1 = loader.load_image(subject_id, 1, variant)
        img2 = loader.load_image(subject_id, 2, variant)

        features[subject_id] = {}
        if img1 is not None:
            enhanced1 = enhancer.enhance(img1)
            features[subject_id][1] = extractor.extract_normalized(enhanced1)
        if img2 is not None:
            enhanced2 = enhancer.enhance(img2)
            features[subject_id][2] = extractor.extract_normalized(enhanced2)

    # Generate pairs
    pair_gen = PairGenerator(
        subject_ids=subject_ids,
        num_impostor_pairs=matching_cfg.get("num_impostor_pairs", 1000),
        seed=matching_cfg.get("random_seed", 42),
    )
    genuine_pairs, impostor_pairs = pair_gen.generate_all_pairs()

    # Compute scores
    genuine_scores = compute_scores(genuine_pairs, features)
    impostor_scores = compute_scores(impostor_pairs, features)

    # Metrics
    metrics = BiometricMetrics(
        threshold_step=eval_cfg.get("threshold_step", 0.001),
        threshold_range=tuple(eval_cfg.get("threshold_range", [0.0, 1.0])),
        fmr_targets=eval_cfg.get("fmr_targets", [0.001, 0.0001]),
    )

    eval_results = metrics.evaluate_all(genuine_scores, impostor_scores)

    # ROC data
    fpr, tpr, _ = metrics.compute_roc_curve(genuine_scores, impostor_scores)
    sweep = metrics.threshold_sweep(genuine_scores, impostor_scores)

    return {
        "eval_results": eval_results,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "roc": (fpr, tpr, eval_results["auc_roc"]),
        "sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate noise robustness across all dataset variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all-variants", action="store_true",
        help="Evaluate all variants (REAL, GAUSSIAN, SALT_PEPPER)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["REAL", "GAUSSIAN", "SALT_PEPPER"],
        help="Specific variants to evaluate",
    )
    parser.add_argument(
        "--feature-type", type=str, default="handcrafted",
        choices=["handcrafted", "deep"],
        help="Feature extraction method",
    )
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base data directory",
    )

    args = parser.parse_args()

    os.chdir(project_root)
    setup_logging()
    logger = logging.getLogger("evaluate_robustness")

    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)

    feature_cfg = config.get("features", {})
    matching_cfg = config.get("matching", {})
    eval_cfg = config.get("evaluation", {})

    variants = args.variants if not args.all_variants else ["REAL", "GAUSSIAN", "SALT_PEPPER"]

    start_time = time.time()

    # Load dataset
    loader = DatasetLoader(data_dir=args.data_dir)
    loader.load_metadata()
    subject_ids = loader.get_all_subject_ids()

    # Create enhancer
    preproc_cfg = config.get("preprocessing", {})
    clahe_cfg = preproc_cfg.get("clahe", {})
    enhancer = ImageEnhancer(
        target_size=tuple(config.get("image", {}).get("target_size", [256, 256])),
        clip_limit=clahe_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        normalize=preproc_cfg.get("normalize", True),
    )

    # Create extractor
    if args.feature_type == "handcrafted":
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

    # Evaluate each variant
    all_variant_results = {}
    roc_data = {}
    det_data = {}

    for variant in variants:
        result = evaluate_variant(
            variant, loader, enhancer, extractor,
            subject_ids, matching_cfg, eval_cfg, logger,
        )
        all_variant_results[variant] = result["eval_results"]
        roc_data[variant] = result["roc"]
        det_data[variant] = (result["sweep"]["fmr"], result["sweep"]["fnmr"])

        # Per-variant score distribution
        plot_score_distributions(
            result["genuine_scores"],
            result["impostor_scores"],
            variant=variant,
            eer_threshold=result["eval_results"]["eer_threshold"],
            output_path=f"outputs/figures/score_dist_{variant}_{args.feature_type}.png",
        )

    # ──────────────────────────────────────────
    # Cross-variant analysis
    # ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VARIANT ROBUSTNESS ANALYSIS")
    logger.info("=" * 60)

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Combined ROC curves
    plot_roc_curves(
        roc_data,
        output_path=str(fig_dir / f"roc_all_variants_{args.feature_type}.png"),
        title=f"ROC Curves — All Variants ({args.feature_type})",
    )

    # Combined DET curves
    plot_det_curves(
        det_data,
        output_path=str(fig_dir / f"det_all_variants_{args.feature_type}.png"),
        title=f"DET Curves — All Variants ({args.feature_type})",
    )

    # Noise degradation chart
    plot_noise_degradation(
        all_variant_results,
        output_path=str(fig_dir / f"noise_degradation_{args.feature_type}.png"),
    )

    # Compute degradation scores
    if "REAL" in all_variant_results:
        real_eer = all_variant_results["REAL"]["eer"]
        logger.info(f"\nBaseline REAL EER: {real_eer:.4f}")

        for variant in variants:
            if variant != "REAL":
                delta = all_variant_results[variant]["eer"] - real_eer
                logger.info(f"  DeltaEER ({variant}): {delta:+.4f}")
                all_variant_results[variant]["delta_eer"] = float(delta)

    # Save robustness report
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"robustness_results_{args.feature_type}.json"

    with open(report_path, "w") as f:
        json.dump(all_variant_results, f, indent=2, default=str)

    logger.info(f"\nRobustness report saved to {report_path}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal robustness evaluation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
