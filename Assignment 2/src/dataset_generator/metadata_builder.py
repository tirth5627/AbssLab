"""
Metadata Builder — Demographic Parser & Age Gap Tagger.

Generates structured demographic metadata for scraped subjects,
including age, gender, ethnicity, and simulated age gap information.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """
    Generates and manages demographic metadata for facial dataset subjects.

    Uses deterministic hashing from subject IDs for reproducible
    demographic attribute assignment, ensuring balanced demographic
    distributions across the dataset.

    Attributes:
        num_subjects: Number of subjects in the dataset.
        age_range: Tuple of (min_age, max_age).
        age_gap_range: Tuple of (min_gap, max_gap) in years.
        genders: List of gender categories.
        ethnicities: List of ethnicity categories.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_subjects: int = 200,
        age_range: tuple = (18, 80),
        age_gap_range: tuple = (1, 5),
        genders: list = None,
        ethnicities: list = None,
        seed: int = 42,
    ):
        self.num_subjects = num_subjects
        self.age_range = age_range
        self.age_gap_range = age_gap_range
        self.genders = genders or ["Male", "Female"]
        self.ethnicities = ethnicities or ["Asian", "Caucasian", "African", "Hispanic"]
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _assign_demographic(self, subject_id: str) -> dict:
        """
        Assign demographic attributes to a subject deterministically.

        Uses the subject ID to seed random generation, ensuring
        reproducibility while maintaining balanced distributions.

        Args:
            subject_id: Subject identifier string (e.g., 'S0001').

        Returns:
            Dictionary with demographic attributes.
        """
        # Hash subject ID for deterministic but varied assignment
        hash_val = int(hashlib.md5(subject_id.encode()).hexdigest(), 16)

        # Balanced assignment using modulo for even distribution
        gender = self.genders[hash_val % len(self.genders)]
        ethnicity = self.ethnicities[
            (hash_val // len(self.genders)) % len(self.ethnicities)
        ]

        # Age: pseudo-random within range using hash
        age_span = self.age_range[1] - self.age_range[0]
        age = self.age_range[0] + (hash_val % age_span)

        # Age gap: pseudo-random within range
        gap_span = self.age_gap_range[1] - self.age_gap_range[0] + 1
        age_gap = self.age_gap_range[0] + ((hash_val // 100) % gap_span)

        return {
            "subject_id": subject_id,
            "age": int(age),
            "gender": gender,
            "ethnicity": ethnicity,
            "age_gap_years": int(age_gap),
            "images": [
                f"{subject_id}_001.jpg",
                f"{subject_id}_002.jpg",
            ],
        }

    def build_metadata(self) -> list:
        """
        Generate demographic metadata for all subjects.

        Returns:
            List of dictionaries, one per subject, containing
            demographic attributes and image filenames.
        """
        metadata = []
        for i in range(1, self.num_subjects + 1):
            subject_id = f"S{i:04d}"
            entry = self._assign_demographic(subject_id)
            metadata.append(entry)

        logger.info(f"Generated metadata for {len(metadata)} subjects")
        return metadata

    def compute_statistics(self, metadata: list) -> dict:
        """
        Compute summary statistics for the dataset.

        Args:
            metadata: List of subject metadata dictionaries.

        Returns:
            Dictionary with dataset statistics including
            demographic distributions and age statistics.
        """
        ages = [m["age"] for m in metadata]
        genders = [m["gender"] for m in metadata]
        ethnicities = [m["ethnicity"] for m in metadata]
        age_gaps = [m["age_gap_years"] for m in metadata]

        stats = {
            "total_subjects": len(metadata),
            "total_images": len(metadata) * 2,
            "age_statistics": {
                "mean": float(np.mean(ages)),
                "std": float(np.std(ages)),
                "min": int(np.min(ages)),
                "max": int(np.max(ages)),
                "median": float(np.median(ages)),
            },
            "age_gap_statistics": {
                "mean": float(np.mean(age_gaps)),
                "std": float(np.std(age_gaps)),
                "min": int(np.min(age_gaps)),
                "max": int(np.max(age_gaps)),
            },
            "gender_distribution": {
                g: genders.count(g) for g in set(genders)
            },
            "ethnicity_distribution": {
                e: ethnicities.count(e) for e in set(ethnicities)
            },
        }

        logger.info(f"Dataset statistics: {stats['total_subjects']} subjects, "
                     f"{stats['total_images']} images")
        return stats

    def save(self, metadata: list, stats: dict, output_dir: str = "data/metadata") -> None:
        """
        Save metadata and statistics to JSON files.

        Args:
            metadata: List of subject metadata dictionaries.
            stats: Dataset statistics dictionary.
            output_dir: Directory to save JSON files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        metadata_file = out_path / "metadata.json"
        stats_file = out_path / "statistics.json"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_file}")

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics to {stats_file}")
