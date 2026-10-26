"""
Evaluation metrics, classifier training, and benchmark pipelines.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_reconstruction_metrics(X_true: np.ndarray, X_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Reconstruction Mean Squared Error (MSE), Relative Frobenius Error,
    and Peak Signal-to-Noise Ratio (PSNR).

    Args:
        X_true: Original image feature matrix of shape (N, D).
        X_pred: Reconstructed feature matrix of shape (N, D).

    Returns:
        Dictionary with 'mse', 'relative_frobenius_error', and 'psnr'.
    """
    diff = X_true - X_pred
    mse = float(np.mean(diff ** 2))
    norm_diff = np.linalg.norm(diff, ord='fro')
    norm_true = np.linalg.norm(X_true, ord='fro')
    rel_error = float(norm_diff / (norm_true + 1e-12))

    # PSNR calculation (for pixel values in [0, 1])
    max_val = 1.0
    if mse > 1e-12:
        psnr = float(10.0 * np.log10((max_val ** 2) / mse))
    else:
        psnr = float(100.0)

    return {
        "mse": mse,
        "relative_frobenius_error": rel_error,
        "psnr": psnr,
    }


def compute_storage_metrics(
    n_features: int,
    n_components: int,
    n_samples: int,
    precision_bytes: int = 4
) -> Dict[str, Any]:
    """
    Calculate memory and storage footprints for projected representations and model projection matrices.

    Args:
        n_features: Original feature dimension (e.g. 10,304).
        n_components: Reduced component dimension k.
        n_samples: Number of samples (e.g. 200 test images).
        precision_bytes: Bytes per float (4 for float32, 8 for float64).

    Returns:
        Dictionary containing storage metrics in bytes and formatted KB/MB.
    """
    raw_feature_bytes_per_image = n_features * precision_bytes
    projected_bytes_per_image = n_components * precision_bytes
    total_projected_bytes = projected_bytes_per_image * n_samples

    # Model parameters: projection matrix (D x k) + mean vector (D)
    model_param_bytes = (n_features * n_components + n_features) * precision_bytes
    compression_ratio = float(n_features / max(1, n_components))

    return {
        "n_components": n_components,
        "raw_bytes_per_image": raw_feature_bytes_per_image,
        "projected_bytes_per_image": projected_bytes_per_image,
        "total_projected_bytes": total_projected_bytes,
        "projected_kb_per_image": projected_bytes_per_image / 1024.0,
        "model_param_bytes": model_param_bytes,
        "model_param_kb": model_param_bytes / 1024.0,
        "compression_ratio": compression_ratio,
    }


def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = "1-NN",
    **kwargs
) -> Dict[str, Any]:
    """
    Train classifier on projected training features and evaluate on test features.

    Args:
        X_train: Training features (N_train, k).
        y_train: Training identity labels (N_train,).
        X_test: Test features (N_test, k).
        y_test: Test identity labels (N_test,).
        classifier_type: '1-NN' or 'LinearSVM'.

    Returns:
        Dictionary with accuracy, macro_f1, predictions, and timing.
    """
    t0 = time.perf_counter()
    if classifier_type == "1-NN":
        metric = kwargs.get("metric", "euclidean")
        clf = KNeighborsClassifier(n_neighbors=1, metric=metric)
    elif classifier_type in ("LinearSVM", "SVM"):
        C = kwargs.get("C", 1.0)
        random_state = kwargs.get("random_state", 42)
        clf = LinearSVC(C=C, random_state=random_state, dual="auto", max_iter=2000)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    inference_time = time.perf_counter() - t1

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_test, y_pred, labels=np.arange(len(np.unique(y_test))))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred,
        "confusion_matrix": conf_mat,
        "clf_train_time": train_time,
        "clf_inference_time": inference_time,
    }
