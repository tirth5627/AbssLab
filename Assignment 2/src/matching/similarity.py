"""
Similarity Scoring — Cosine & Euclidean Distance Functions.

Computes comparison scores between feature vectors for
biometric verification matching.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    S(v1, v2) = (v1 · v2) / (||v1||_2 * ||v2||_2)

    Returns a value in [-1, 1], remapped to [0, 1] for threshold comparison.

    Args:
        v1: First feature vector.
        v2: Second feature vector.

    Returns:
        Cosine similarity score in [0, 1].
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = dot / (norm1 * norm2)

    # Remap from [-1, 1] to [0, 1]
    return float((sim + 1.0) / 2.0)


def compute_scores(
    pairs: List[Dict],
    features: Dict[str, Dict[int, np.ndarray]],
) -> np.ndarray:
    """
    Compute similarity scores for a list of comparison pairs.

    Args:
        pairs: List of pair dictionaries from PairGenerator.
        features: Nested dictionary mapping subject_id → image_idx → feature_vector.
            Example: features["S0001"][1] = np.array([...])

    Returns:
        1-D array of similarity scores, one per pair.
    """
    scores = []

    for pair in pairs:
        sid_a = pair["subject_id_a"]
        sid_b = pair["subject_id_b"]
        idx_a = pair["image_idx_a"]
        idx_b = pair["image_idx_b"]

        v1 = features.get(sid_a, {}).get(idx_a)
        v2 = features.get(sid_b, {}).get(idx_b)

        if v1 is None or v2 is None:
            logger.warning(
                f"Missing features for pair: {sid_a}_{idx_a} vs {sid_b}_{idx_b}"
            )
            scores.append(0.0)
            continue

        score = cosine_similarity(v1, v2)
        scores.append(score)

    return np.array(scores, dtype=np.float64)
