"""
Handcrafted Feature Extractor — Color Histograms, LBP, and Hu Moments.

Provides a baseline feature representation using traditional
computer vision descriptors without deep learning.
"""

import logging

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .base import FeatureExtractor

logger = logging.getLogger(__name__)


class HandcraftedExtractor(FeatureExtractor):
    """
    Handcrafted feature extractor combining color histograms,
    Local Binary Patterns (LBP), and Hu spatial moments.

    Feature vector composition:
    - Color Histogram: H, S, V channels × histogram_bins each
    - LBP: Uniform LBP histogram (n_points + 2 bins)
    - Hu Moments: 7-dimensional log-scale moment invariants

    Total dimensionality: 3×histogram_bins + (lbp_n_points + 2) + 7

    Attributes:
        histogram_bins: Number of bins per color channel histogram.
        lbp_radius: Radius for LBP computation.
        lbp_n_points: Number of sampling points for LBP.
    """

    def __init__(
        self,
        histogram_bins: int = 32,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
    ):
        self.histogram_bins = histogram_bins
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points

    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from HSV channels.

        Args:
            image: RGB image as float64 [0,1] or uint8 [0,255].

        Returns:
            Concatenated H, S, V channel histograms.
        """
        # Ensure uint8 for OpenCV
        if image.dtype == np.float64 or image.dtype == np.float32:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

        histograms = []
        # H channel: range [0, 180]
        hist_h = cv2.calcHist([hsv], [0], None, [self.histogram_bins], [0, 180])
        histograms.append(hist_h.flatten())

        # S channel: range [0, 256]
        hist_s = cv2.calcHist([hsv], [1], None, [self.histogram_bins], [0, 256])
        histograms.append(hist_s.flatten())

        # V channel: range [0, 256]
        hist_v = cv2.calcHist([hsv], [2], None, [self.histogram_bins], [0, 256])
        histograms.append(hist_v.flatten())

        # Normalize each histogram independently
        features = np.concatenate(histograms)
        total = features.sum()
        if total > 0:
            features = features / total

        return features

    def _extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) histogram features.

        Uses uniform LBP pattern, which captures local texture
        information robust to monotonic illumination changes.

        Args:
            image: RGB or grayscale image.

        Returns:
            Normalized LBP histogram.
        """
        # Convert to grayscale if needed
        if image.dtype == np.float64 or image.dtype == np.float32:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        if len(img_uint8.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8

        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method="uniform",
        )

        # Compute histogram of LBP codes
        n_bins = self.lbp_n_points + 2  # uniform patterns + 1 non-uniform bin
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True,
        )

        return hist

    def _extract_hu_moments(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Hu moment invariants for spatial shape description.

        Hu moments are translation, scale, and rotation invariant,
        providing global shape information.

        Args:
            image: RGB or grayscale image.

        Returns:
            7-dimensional log-scale Hu moment vector.
        """
        # Convert to grayscale uint8
        if image.dtype == np.float64 or image.dtype == np.float32:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        if len(img_uint8.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log-transform for better numerical properties
        # Use sign-preserving log: sign(h) * log10(|h| + epsilon)
        epsilon = 1e-10
        hu_log = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)

        return hu_log

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the full handcrafted feature vector.

        Concatenates color histogram, LBP, and Hu moment features
        into a single descriptor.

        Args:
            image: Preprocessed face image (RGB, float64 or uint8).

        Returns:
            Concatenated, L2-normalized feature vector.
        """
        color_hist = self._extract_color_histogram(image)
        lbp_hist = self._extract_lbp(image)
        hu_moments = self._extract_hu_moments(image)

        # Concatenate all features
        features = np.concatenate([color_hist, lbp_hist, hu_moments])

        return features.astype(np.float64)
