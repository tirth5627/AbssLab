"""
Feature Extractor Base Class — Abstract Interface.

Defines the contract for all feature extraction implementations,
including L2 normalization utility.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """
    Abstract base class for biometric feature extractors.

    All feature extractors must implement the `extract` method
    that converts a preprocessed face image into a fixed-length
    feature vector.
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a feature vector from a preprocessed face image.

        Args:
            image: Preprocessed face image as numpy array.
                   Shape depends on implementation (RGB or grayscale).

        Returns:
            1-D feature vector as numpy float64 array.
        """
        pass

    @staticmethod
    def l2_normalize(vector: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        L2-normalize a feature vector.

        Computes: v_hat = v / (||v||_2 + epsilon)

        Args:
            vector: Input feature vector.
            epsilon: Small constant to avoid division by zero.

        Returns:
            L2-normalized feature vector with unit norm.
        """
        norm = np.linalg.norm(vector)
        return vector / (norm + epsilon)

    def extract_normalized(self, image: np.ndarray) -> np.ndarray:
        """
        Extract and L2-normalize a feature vector.

        Convenience method that combines extraction and normalization.

        Args:
            image: Preprocessed face image.

        Returns:
            L2-normalized feature vector.
        """
        features = self.extract(image)
        return self.l2_normalize(features)
