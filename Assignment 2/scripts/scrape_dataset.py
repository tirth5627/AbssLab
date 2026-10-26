#!/usr/bin/env python3
"""
Dataset Web Scraper CLI Entrypoint.

Orchestrates the full dataset construction pipeline:
1. Scrapes face images from web sources
2. Generates demographic metadata
3. Applies noise corruption variants

Usage:
    python scripts/scrape_dataset.py --subjects 200 --images-per-subject 2 --output-dir data/
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_generator.scraper import FaceScraper
from src.dataset_generator.metadata_builder import MetadataBuilder
from src.dataset_generator.noise_injector import NoiseInjector


def setup_logging(output_dir: str) -> None:
    """Configure logging with console and file output."""
    log_dir = Path(output_dir).parent / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "scrape.log", mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape face images, build metadata, and generate noise variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subjects", type=int, default=200,
        help="Number of distinct subjects to scrape",
    )
    parser.add_argument(
        "--images-per-subject", type=int, default=2,
        help="Number of images per subject",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Base output directory for dataset",
    )
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="Delay between HTTP requests (overrides config)",
    )
    parser.add_argument(
        "--skip-scraping", action="store_true",
        help="Skip scraping step (use existing REAL images)",
    )
    parser.add_argument(
        "--skip-noise", action="store_true",
        help="Skip noise generation step",
    )

    args = parser.parse_args()

    # Change to project root
    os.chdir(project_root)

    setup_logging(args.output_dir)
    logger = logging.getLogger("scrape_dataset")

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")

    scraping_cfg = config.get("scraping", {})
    noise_cfg = config.get("noise", {})
    metadata_cfg = config.get("metadata", {})

    real_dir = Path(args.output_dir) / "raw" / "REAL"
    metadata_dir = Path(args.output_dir) / "metadata"

    start_time = time.time()

    # ──────────────────────────────────────────
    # Step 1: Web Scraping
    # ──────────────────────────────────────────
    if not args.skip_scraping:
        logger.info("=" * 60)
        logger.info("STEP 1: Web Scraping — Downloading face images")
        logger.info("=" * 60)

        scraper = FaceScraper(
            output_dir=str(real_dir),
            num_subjects=args.subjects,
            images_per_subject=args.images_per_subject,
            target_size=tuple(config.get("image", {}).get("target_size", [256, 256])),
            delay=args.delay or scraping_cfg.get("request_delay_sec", 1.0),
            max_retries=scraping_cfg.get("max_retries", 3),
            retry_backoff=scraping_cfg.get("retry_backoff", 2.0),
            timeout=scraping_cfg.get("timeout_sec", 15),
            source=scraping_cfg.get("source", "thispersondoesnotexist"),
            user_agent=scraping_cfg.get("user_agent", "Mozilla/5.0 FaceBiometricResearch/1.0"),
        )

        scrape_stats = scraper.scrape()
        logger.info(f"Scraping results: {scrape_stats['total_downloaded']} downloaded, "
                     f"{scrape_stats['total_failed']} failed")
    else:
        logger.info("Skipping scraping step (--skip-scraping)")

    # ──────────────────────────────────────────
    # Step 2: Metadata Generation
    # ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Generating demographic metadata")
    logger.info("=" * 60)

    builder = MetadataBuilder(
        num_subjects=args.subjects,
        age_range=tuple(metadata_cfg.get("age_range", [18, 80])),
        age_gap_range=tuple(metadata_cfg.get("age_gap_range", [1, 5])),
        genders=metadata_cfg.get("genders", ["Male", "Female"]),
        ethnicities=metadata_cfg.get("ethnicities",
                                      ["Asian", "Caucasian", "African", "Hispanic"]),
        seed=metadata_cfg.get("random_seed", 42),
    )

    metadata = builder.build_metadata()
    stats = builder.compute_statistics(metadata)
    builder.save(metadata, stats, str(metadata_dir))

    # ──────────────────────────────────────────
    # Step 3: Noise Variant Generation
    # ──────────────────────────────────────────
    if not args.skip_noise:
        logger.info("=" * 60)
        logger.info("STEP 3: Generating noise-corrupted variants")
        logger.info("=" * 60)

        gaussian_cfg = noise_cfg.get("gaussian", {})
        sp_cfg = noise_cfg.get("salt_pepper", {})

        injector = NoiseInjector(
            gaussian_mean=gaussian_cfg.get("mean", 0),
            gaussian_sigma=gaussian_cfg.get("sigma", 25),
            sp_amount=sp_cfg.get("amount", 0.05),
            sp_ratio=sp_cfg.get("salt_vs_pepper", 0.5),
        )

        noise_stats = injector.generate_variants(
            real_dir=str(real_dir),
            output_base=str(Path(args.output_dir) / "raw"),
        )
        logger.info(f"Noise generation results: {noise_stats}")
    else:
        logger.info("Skipping noise generation (--skip-noise)")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Dataset construction complete! Total time: {elapsed:.1f}s")
    logger.info(f"Dataset location: {args.output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
