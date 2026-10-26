"""
Assignment 3: Comparative Study of PCA, Truncated SVD, and LDA on the AT&T / ORL Face Database.
"""

from .data_loader import load_orl_dataset, generate_stratified_splits
from .models import DimensionReducer, PCAReducer, SVDReducer, LDAReducer
from .evaluation import evaluate_classifier, compute_reconstruction_metrics, compute_storage_metrics
from .visualization import FaceVisualizer

__all__ = [
    "load_orl_dataset",
    "generate_stratified_splits",
    "DimensionReducer",
    "PCAReducer",
    "SVDReducer",
    "LDAReducer",
    "evaluate_classifier",
    "compute_reconstruction_metrics",
    "compute_storage_metrics",
    "FaceVisualizer",
]
