"""
Pair Generator — Genuine & Impostor Pair Sampling.

Constructs biometric comparison pairs for verification evaluation,
including same-subject genuine pairs and cross-subject impostor pairs.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PairGenerator:
    """
    Generates genuine and impostor comparison pairs for biometric evaluation.

    Genuine pairs compare two images of the same subject (same identity).
    Impostor pairs compare images from different subjects (different identities).

    Attributes:
        subject_ids: List of all subject identifiers.
        num_impostor_pairs: Target number of impostor pairs to generate.
        seed: Random seed for reproducible pair sampling.
    """

    def __init__(
        self,
        subject_ids: List[str],
        num_impostor_pairs: int = 1000,
        seed: int = 42,
    ):
        self.subject_ids = sorted(subject_ids)
        self.num_impostor_pairs = num_impostor_pairs
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_genuine_pairs(self) -> List[Dict]:
        """
        Generate genuine (same-subject) comparison pairs.

        Creates one pair per subject: image_001 vs image_002.

        Returns:
            List of pair dictionaries with keys:
                - subject_id_a, subject_id_b: Same subject ID
                - image_idx_a, image_idx_b: Image indices (1, 2)
                - pair_type: 'genuine'
        """
        pairs = []
        for subject_id in self.subject_ids:
            pairs.append({
                "subject_id_a": subject_id,
                "subject_id_b": subject_id,
                "image_idx_a": 1,
                "image_idx_b": 2,
                "pair_type": "genuine",
            })

        logger.info(f"Generated {len(pairs)} genuine pairs")
        return pairs

    def generate_impostor_pairs(self) -> List[Dict]:
        """
        Generate impostor (cross-subject) comparison pairs.

        Randomly samples pairs of different subjects, comparing
        image_001 of subject A with image_001 of subject B.

        Returns:
            List of pair dictionaries with keys:
                - subject_id_a, subject_id_b: Different subject IDs
                - image_idx_a, image_idx_b: Both 1
                - pair_type: 'impostor'
        """
        n = len(self.subject_ids)
        if n < 2:
            logger.error("Need at least 2 subjects for impostor pairs")
            return []

        # Maximum possible impostor pairs: n*(n-1)/2
        max_pairs = n * (n - 1) // 2
        target = min(self.num_impostor_pairs, max_pairs)

        # Generate random unique pairs
        pair_set = set()
        attempts = 0
        max_attempts = target * 10  # Safety limit

        while len(pair_set) < target and attempts < max_attempts:
            idx_a = self.rng.randint(0, n)
            idx_b = self.rng.randint(0, n)

            if idx_a != idx_b:
                # Canonical ordering to avoid duplicates
                pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                pair_set.add(pair_key)

            attempts += 1

        pairs = []
        for idx_a, idx_b in pair_set:
            pairs.append({
                "subject_id_a": self.subject_ids[idx_a],
                "subject_id_b": self.subject_ids[idx_b],
                "image_idx_a": 1,
                "image_idx_b": 1,
                "pair_type": "impostor",
            })

        logger.info(f"Generated {len(pairs)} impostor pairs")
        return pairs

    def generate_all_pairs(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate both genuine and impostor pairs.

        Returns:
            Tuple of (genuine_pairs, impostor_pairs).
        """
        genuine = self.generate_genuine_pairs()
        impostor = self.generate_impostor_pairs()
        return genuine, impostor
