"""
Deep Feature Extractor — FaceNet / ResNet Embeddings.

Provides deep neural network-based facial embeddings using
pre-trained models for high-accuracy biometric matching.
"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import FeatureExtractor

logger = logging.getLogger(__name__)


class DeepExtractor(FeatureExtractor):
    """
    Deep feature extractor using pre-trained face recognition models.

    Primary: FaceNet (InceptionResnetV1) via facenet-pytorch
    Fallback: torchvision ResNet18 with adaptive pooling

    Outputs a 512-dimensional L2-normalized embedding vector suitable
    for cosine similarity matching.

    Attributes:
        model_name: Model architecture ('facenet' or 'resnet').
        pretrained: Pre-trained weights identifier.
        device: Computation device ('auto', 'cpu', 'cuda').
        embedding_dim: Output embedding dimensionality.
    """

    def __init__(
        self,
        model_name: str = "facenet",
        pretrained: str = "vggface2",
        device: str = "auto",
        embedding_dim: int = 512,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim

        # Resolve device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"DeepExtractor using device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        """
        Load the pre-trained deep model.

        Attempts FaceNet first, falls back to ResNet18 if unavailable.

        Returns:
            Pre-trained PyTorch model in eval mode.
        """
        if self.model_name == "facenet":
            try:
                from facenet_pytorch import InceptionResnetV1

                model = InceptionResnetV1(
                    pretrained=self.pretrained
                ).to(self.device)
                logger.info(
                    f"Loaded FaceNet (InceptionResnetV1) with "
                    f"'{self.pretrained}' weights"
                )
                return model
            except ImportError:
                logger.warning(
                    "facenet-pytorch not installed. Falling back to ResNet18."
                )
                self.model_name = "resnet"

        # Fallback: ResNet18 with modified final layer
        import torchvision.models as models

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace classifier with embedding layer
        in_features = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(in_features, self.embedding_dim)
        resnet = resnet.to(self.device)

        logger.info(
            f"Loaded ResNet18 fallback with {self.embedding_dim}-dim output"
        )
        return resnet

    def _preprocess_for_model(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for the deep model.

        Handles both FaceNet (160×160, specific normalization) and
        ResNet (224×224, ImageNet normalization) requirements.

        Args:
            image: Input image as numpy array (H, W, C), RGB,
                   either uint8 [0,255] or float64 [0,1].

        Returns:
            Preprocessed tensor with batch dimension (1, C, H, W).
        """
        import cv2

        # Ensure uint8 for resizing
        if image.dtype in (np.float64, np.float32):
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()

        if self.model_name == "facenet":
            # FaceNet expects 160×160 RGB, normalized to [-1, 1]
            img_resized = cv2.resize(img_uint8, (160, 160),
                                     interpolation=cv2.INTER_LANCZOS4)
            img_float = img_resized.astype(np.float32) / 255.0
            # Normalize to [-1, 1]
            img_float = (img_float - 0.5) / 0.5
        else:
            # ResNet expects 224×224 RGB, ImageNet normalization
            img_resized = cv2.resize(img_uint8, (224, 224),
                                     interpolation=cv2.INTER_LANCZOS4)
            img_float = img_resized.astype(np.float32) / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_float = (img_float - mean) / std

        # HWC -> CHW, add batch dimension
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a deep embedding vector from a face image.

        Args:
            image: Preprocessed face image (RGB, float64 or uint8).

        Returns:
            512-dimensional feature vector as float64 numpy array.
        """
        input_tensor = self._preprocess_for_model(image)

        with torch.no_grad():
            embedding = self.model(input_tensor)

        # Convert to numpy
        features = embedding.cpu().numpy().flatten()

        # Ensure correct dimensionality
        if len(features) != self.embedding_dim:
            logger.warning(
                f"Expected {self.embedding_dim}-dim, got {len(features)}-dim. "
                f"Padding/truncating."
            )
            if len(features) < self.embedding_dim:
                features = np.pad(
                    features,
                    (0, self.embedding_dim - len(features)),
                )
            else:
                features = features[: self.embedding_dim]

        return features.astype(np.float64)

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Processes images in a memory-efficient manner suitable
        for GPUs with limited VRAM (e.g., 4GB RTX 3050).

        Args:
            images: List of preprocessed face images.

        Returns:
            Array of shape (N, embedding_dim) with L2-normalized embeddings.
        """
        embeddings = []

        for image in images:
            emb = self.extract_normalized(image)
            embeddings.append(emb)

        return np.array(embeddings)
