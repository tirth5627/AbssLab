"""
Dimensionality Reduction Models: PCA, Truncated SVD, and LDA.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DimensionReducer(ABC):
    """Abstract base class for dimensionality reduction methods."""

    def __init__(self, n_components: int, name: str):
        self.n_components = n_components
        self.name = name
        self.fit_time: float = 0.0
        self.last_transform_time: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DimensionReducer":
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct original high-dimensional features from projected representation."""
        pass


class PCAReducer(DimensionReducer):
    """
    Principal Component Analysis (Eigenfaces).
    Unsupervised, variance-maximizing orthogonal projection.
    Data is centered by subtracting the training mean face.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        super().__init__(n_components=n_components, name="PCA")
        self.random_state = random_state
        self.model = PCA(n_components=n_components, random_state=random_state)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAReducer":
        t0 = time.perf_counter()
        self.model.fit(X)
        self.fit_time = time.perf_counter() - t0
        self.mean_ = self.model.mean_
        self.components_ = self.model.components_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        Z = self.model.transform(X)
        self.last_transform_time = time.perf_counter() - t0
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.inverse_transform(Z)

    def get_explained_variance_ratio(self) -> np.ndarray:
        return self.model.explained_variance_ratio_

    def get_cumulative_explained_variance(self) -> float:
        return float(np.sum(self.model.explained_variance_ratio_))


class SVDReducer(DimensionReducer):
    """
    Truncated Singular Value Decomposition (SVD).
    Factorizes data matrix into singular vectors and singular values.
    By default operates on uncentered data as standard TruncatedSVD.
    Optionally supports mean-centering if centered=True.
    """

    def __init__(self, n_components: int, centered: bool = False, random_state: int = 42):
        name = "Truncated SVD (Centered)" if centered else "Truncated SVD"
        super().__init__(n_components=n_components, name=name)
        self.centered = centered
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SVDReducer":
        t0 = time.perf_counter()
        if self.centered:
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            X_proc = X - self.mean_
        else:
            self.mean_ = None
            X_proc = X

        self.model.fit(X_proc)
        self.fit_time = time.perf_counter() - t0
        self.components_ = self.model.components_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.centered and self.mean_ is not None:
            X_proc = X - self.mean_
        else:
            X_proc = X
        Z = self.model.transform(X_proc)
        self.last_transform_time = time.perf_counter() - t0
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        X_rec = self.model.inverse_transform(Z)
        if self.centered and self.mean_ is not None:
            X_rec = X_rec + self.mean_
        return X_rec

    def get_explained_variance_ratio(self) -> np.ndarray:
        return self.model.explained_variance_ratio_


class LDAReducer(DimensionReducer):
    """
    Linear Discriminant Analysis (Fisherfaces).
    Supervised projection maximizing between-class scatter over within-class scatter.
    Maximum valid number of components is min(n_classes - 1, n_features) = 39 for 40 classes.
    Uses 'svd' solver which is numerically stable and avoids inverting high-dimensional covariance.
    """

    def __init__(self, n_components: int, solver: str = "svd"):
        super().__init__(n_components=n_components, name="LDA")
        self.solver = solver
        self.model = LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
        self.scalings_: Optional[np.ndarray] = None
        self.xbar_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAReducer":
        if y is None:
            raise ValueError("LDA requires identity labels 'y' for supervised training.")
        t0 = time.perf_counter()
        self.model.fit(X, y)
        self.fit_time = time.perf_counter() - t0
        self.scalings_ = getattr(self.model, "scalings_", None)
        self.xbar_ = getattr(self.model, "xbar_", None)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        Z = self.model.transform(X)
        self.last_transform_time = time.perf_counter() - t0
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Approximate pseudo-inverse reconstruction for LDA.
        Note: LDA is designed for discrimination rather than reconstruction.
        """
        if self.scalings_ is not None and self.xbar_ is not None:
            # scalings_ shape: (D, k)
            # Z shape: (N, k)
            # pinv reconstruction: Z @ pinv(scalings_.T) + xbar_
            W = self.scalings_[:, :self.n_components] # (D, k)
            W_pinv = np.linalg.pinv(W.T) # (k, D)
            return Z @ W_pinv + self.xbar_
        else:
            raise NotImplementedError("Reconstruction not supported for this LDA configuration.")
