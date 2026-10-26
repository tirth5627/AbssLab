"""
Tests for Web Scraper & Dataset Generator.

Validates scraper output structure, metadata schema,
and image format/dimensions.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_generator.scraper import FaceScraper
from src.dataset_generator.metadata_builder import MetadataBuilder
from src.dataset_generator.noise_injector import NoiseInjector


class TestMetadataBuilder:
    """Tests for MetadataBuilder class."""

    def test_build_metadata_count(self):
        """Metadata should generate entries for all subjects."""
        builder = MetadataBuilder(num_subjects=10, seed=42)
        metadata = builder.build_metadata()
        assert len(metadata) == 10

    def test_metadata_fields(self):
        """Each metadata entry should have required fields."""
        builder = MetadataBuilder(num_subjects=5, seed=42)
        metadata = builder.build_metadata()

        required_fields = {"subject_id", "age", "gender", "ethnicity",
                           "age_gap_years", "images"}
        for entry in metadata:
            assert required_fields.issubset(set(entry.keys()))

    def test_subject_id_format(self):
        """Subject IDs should follow SXXXX format."""
        builder = MetadataBuilder(num_subjects=5, seed=42)
        metadata = builder.build_metadata()

        for entry in metadata:
            sid = entry["subject_id"]
            assert sid.startswith("S")
            assert len(sid) == 5
            assert sid[1:].isdigit()

    def test_age_in_range(self):
        """Ages should fall within configured range."""
        builder = MetadataBuilder(num_subjects=50, age_range=(18, 80), seed=42)
        metadata = builder.build_metadata()

        for entry in metadata:
            assert 18 <= entry["age"] <= 80

    def test_gender_values(self):
        """Gender should be Male or Female."""
        builder = MetadataBuilder(num_subjects=50, seed=42)
        metadata = builder.build_metadata()

        valid_genders = {"Male", "Female"}
        for entry in metadata:
            assert entry["gender"] in valid_genders

    def test_ethnicity_values(self):
        """Ethnicity should be from the configured set."""
        builder = MetadataBuilder(num_subjects=50, seed=42)
        metadata = builder.build_metadata()

        valid = {"Asian", "Caucasian", "African", "Hispanic"}
        for entry in metadata:
            assert entry["ethnicity"] in valid

    def test_images_list(self):
        """Each entry should list exactly 2 images."""
        builder = MetadataBuilder(num_subjects=5, seed=42)
        metadata = builder.build_metadata()

        for entry in metadata:
            assert len(entry["images"]) == 2
            assert entry["images"][0].endswith("_001.jpg")
            assert entry["images"][1].endswith("_002.jpg")

    def test_reproducibility(self):
        """Same seed should produce identical metadata."""
        builder1 = MetadataBuilder(num_subjects=10, seed=42)
        builder2 = MetadataBuilder(num_subjects=10, seed=42)
        m1 = builder1.build_metadata()
        m2 = builder2.build_metadata()
        assert m1 == m2

    def test_statistics_computation(self):
        """Statistics should include all required fields."""
        builder = MetadataBuilder(num_subjects=50, seed=42)
        metadata = builder.build_metadata()
        stats = builder.compute_statistics(metadata)

        assert stats["total_subjects"] == 50
        assert stats["total_images"] == 100
        assert "age_statistics" in stats
        assert "gender_distribution" in stats
        assert "ethnicity_distribution" in stats

    def test_save_and_load(self, tmp_path):
        """Metadata should round-trip through save/load."""
        builder = MetadataBuilder(num_subjects=5, seed=42)
        metadata = builder.build_metadata()
        stats = builder.compute_statistics(metadata)
        builder.save(metadata, stats, str(tmp_path))

        # Verify files exist
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "statistics.json").exists()

        # Verify content
        with open(tmp_path / "metadata.json") as f:
            loaded = json.load(f)
        assert len(loaded) == 5


class TestNoiseInjector:
    """Tests for NoiseInjector class."""

    def test_gaussian_noise_shape(self):
        """Gaussian noise should preserve image shape."""
        injector = NoiseInjector(gaussian_sigma=25)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        noisy = injector.apply_gaussian_noise(img)
        assert noisy.shape == img.shape

    def test_gaussian_noise_dtype(self):
        """Gaussian noise output should be uint8."""
        injector = NoiseInjector(gaussian_sigma=25)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        noisy = injector.apply_gaussian_noise(img)
        assert noisy.dtype == np.uint8

    def test_gaussian_noise_range(self):
        """Gaussian noise output should be in [0, 255]."""
        injector = NoiseInjector(gaussian_sigma=50)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        noisy = injector.apply_gaussian_noise(img)
        assert noisy.min() >= 0
        assert noisy.max() <= 255

    def test_gaussian_noise_changes_image(self):
        """Gaussian noise should modify the image."""
        injector = NoiseInjector(gaussian_sigma=25)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        noisy = injector.apply_gaussian_noise(img)
        assert not np.array_equal(img, noisy)

    def test_salt_pepper_shape(self):
        """Salt-and-pepper noise should preserve image shape."""
        injector = NoiseInjector(sp_amount=0.05)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        noisy = injector.apply_salt_pepper_noise(img)
        assert noisy.shape == img.shape

    def test_salt_pepper_dtype(self):
        """Salt-and-pepper noise output should be uint8."""
        injector = NoiseInjector(sp_amount=0.05)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        noisy = injector.apply_salt_pepper_noise(img)
        assert noisy.dtype == np.uint8

    def test_salt_pepper_has_extremes(self):
        """Salt-and-pepper noise should introduce 0 and 255 values."""
        injector = NoiseInjector(sp_amount=0.1)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        noisy = injector.apply_salt_pepper_noise(img)
        assert 0 in noisy
        assert 255 in noisy


class TestFaceScraper:
    """Tests for FaceScraper class."""

    def test_invalid_source_raises(self):
        """Invalid source should raise ValueError."""
        with pytest.raises(ValueError):
            FaceScraper(source="nonexistent_source")

    def test_output_dir_created(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        output = tmp_path / "test_output"
        scraper = FaceScraper(output_dir=str(output), num_subjects=0)
        # The directory should be created during scrape
        scraper.num_subjects = 0  # Don't actually scrape
        stats = scraper.scrape()
        assert output.exists()
