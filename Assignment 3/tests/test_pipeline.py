"""
Unit test suite for Assignment 3: PCA, Truncated SVD, and LDA benchmarking.
"""

import os
import sys
import numpy as np
import pytest

# Ensure parent directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data_loader import load_orl_dataset, generate_stratified_splits
from src.models import PCAReducer, SVDReducer, LDAReducer
from src.evaluation import evaluate_classifier, compute_reconstruction_metrics, compute_storage_metrics


@pytest.fixture(scope="module")
def dataset_path():
    data_dir = os.path.join(parent_dir, "data")
    assert os.path.exists(data_dir), f"Dataset directory missing: {data_dir}"
    return data_dir


def test_data_loader(dataset_path):
    X, y, meta = load_orl_dataset(dataset_path, normalize=True)
    assert X.shape == (400, 10304), f"Expected shape (400, 10304), got {X.shape}"
    assert len(y) == 400
    assert len(np.unique(y)) == 40
    assert 0.0 <= X.min() <= X.max() <= 1.0
    assert meta["image_width"] == 92
    assert meta["image_height"] == 112
    assert meta["num_identities"] == 40
    assert meta["images_per_identity"] == 10


def test_stratified_splits(dataset_path):
    X, y, _ = load_orl_dataset(dataset_path, normalize=True)
    n_splits = 10
    seeds = [42 + i for i in range(n_splits)]
    splits = generate_stratified_splits(y, n_splits=n_splits, train_per_class=5, test_per_class=5, seeds=seeds)

    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert len(train_idx) == 200
        assert len(test_idx) == 200
        # Assert disjoint
        assert len(set(train_idx).intersection(set(test_idx))) == 0

        # Assert exactly 5 samples per class in train and test
        train_labels = y[train_idx]
        test_labels = y[test_idx]
        for cls in range(40):
            assert np.sum(train_labels == cls) == 5
            assert np.sum(test_labels == cls) == 5


def test_pca_reducer(dataset_path):
    X, y, _ = load_orl_dataset(dataset_path, normalize=True)
    X_train, X_test = X[:200], X[200:]
    k = 20

    pca = PCAReducer(n_components=k, random_state=42)
    pca.fit(X_train)
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)

    assert Z_train.shape == (200, k)
    assert Z_test.shape == (200, k)

    X_rec = pca.inverse_transform(Z_test)
    assert X_rec.shape == X_test.shape

    metrics = compute_reconstruction_metrics(X_test, X_rec)
    assert 0.0 < metrics["mse"] < 0.1
    assert 0.0 < metrics["relative_frobenius_error"] < 0.5
    assert metrics["psnr"] > 15.0


def test_svd_reducer(dataset_path):
    X, y, _ = load_orl_dataset(dataset_path, normalize=True)
    X_train, X_test = X[:200], X[200:]
    k = 20

    svd = SVDReducer(n_components=k, centered=False, random_state=42)
    svd.fit(X_train)
    Z_train = svd.transform(X_train)
    Z_test = svd.transform(X_test)

    assert Z_train.shape == (200, k)
    assert Z_test.shape == (200, k)

    X_rec = svd.inverse_transform(Z_test)
    assert X_rec.shape == X_test.shape

    metrics = compute_reconstruction_metrics(X_test, X_rec)
    assert 0.0 < metrics["mse"] < 0.1
    assert metrics["psnr"] > 15.0


def test_lda_reducer(dataset_path):
    X, y, _ = load_orl_dataset(dataset_path, normalize=True)
    splits = generate_stratified_splits(y, n_splits=1, train_per_class=5, test_per_class=5, seeds=[42])
    train_idx, test_idx = splits[0]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    k = 39

    lda = LDAReducer(n_components=k, solver="svd")
    lda.fit(X_train, y_train)
    Z_train = lda.transform(X_train)
    Z_test = lda.transform(X_test)

    assert Z_train.shape == (200, k)
    assert Z_test.shape == (200, k)

    eval_res = evaluate_classifier(Z_train, y_train, Z_test, y_test, classifier_type="1-NN")
    assert eval_res["accuracy"] > 0.80
    assert eval_res["macro_f1"] > 0.80
    assert eval_res["confusion_matrix"].shape == (40, 40)


def test_storage_metrics():
    res = compute_storage_metrics(n_features=10304, n_components=39, n_samples=200, precision_bytes=4)
    assert res["projected_bytes_per_image"] == 39 * 4
    assert res["total_projected_bytes"] == 39 * 4 * 200
    assert res["compression_ratio"] == 10304 / 39


def test_spectrum_and_variance_plots(dataset_path, tmp_path):
    from src.visualization import FaceVisualizer
    X, y, _ = load_orl_dataset(dataset_path, normalize=True)
    splits = generate_stratified_splits(y, n_splits=1, train_per_class=5, test_per_class=5, seeds=[42])
    train_idx, _ = splits[0]
    X_train, y_train = X[train_idx], y[train_idx]

    # PCA
    pca = PCAReducer(n_components=199, random_state=42).fit(X_train)
    pca_eigenvals = pca.model.explained_variance_
    pca_cum_var = np.cumsum(pca.get_explained_variance_ratio())
    assert len(pca_eigenvals) == 199, "PCA must have min(N-1, D) = 199 non-zero components"

    # SVD
    U, s_svd, Vt = np.linalg.svd(X_train, full_matrices=False)
    X_trans = X_train @ Vt.T
    svd_exp_var = np.var(X_trans, axis=0)
    svd_cum_var = np.cumsum(svd_exp_var / np.var(X_train, axis=0).sum())
    assert len(s_svd) == 200, "SVD must have min(N, D) = 200 singular values"

    # LDA
    lda = LDAReducer(n_components=39, solver="svd").fit(X_train, y_train)
    lda_ratios = getattr(lda.model, "explained_variance_ratio_", None)
    assert len(lda_ratios) == 39, "LDA must have min(C-1, D) = 39 discriminant components"

    viz = FaceVisualizer(output_dir=str(tmp_path))
    p4 = viz.plot_cumulative_variance_svd_vs_pca(pca_cum_var, svd_cum_var, save_name="test_p4.png")
    p8 = viz.plot_spectra_pca_svd_lda(pca_eigenvals, s_svd, lda_ratios, save_name="test_p8.png")

    assert os.path.exists(p4)
    assert os.path.exists(p8)
    assert os.path.getsize(p4) > 10000
    assert os.path.getsize(p8) > 10000
