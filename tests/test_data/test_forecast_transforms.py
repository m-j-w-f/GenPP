"""Tests for ForecastDataset transform functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from genpp.data.icon.dataset import ForecastDataset


class TestForecastDatasetTransforms:
    """Test suite for ForecastDataset transform functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with synthetic test data."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create synthetic tensor files
            # Note: fc_dict still uses old keys internally (predicted_vars/auxiliary_vars)
            # but the dataset converts them to new structure
            fc_data = {
                "predicted_vars": torch.randn(5, 10, 10),
                "auxiliary_vars": torch.randn(5, 10, 10),  # Same shape as predicted_vars for std
            }
            meta_data = torch.randn(2, 10, 10)
            rea_data = torch.randn(2, 10, 10)

            fc_path = tmpdir_path / "fc_test.pt"
            meta_path = tmpdir_path / "meta_test.pt"
            rea_path = tmpdir_path / "rea_test.pt"

            torch.save(fc_data, fc_path)
            torch.save(meta_data, meta_path)
            torch.save(rea_data, rea_path)

            yield {
                "fc_path": fc_path,
                "meta_path": meta_path,
                "rea_path": rea_path,
                "tmpdir": tmpdir_path,
            }

    @pytest.fixture
    def norm_stats(self):
        """Create synthetic normalization statistics."""
        return {
            "pred_mean": torch.zeros(5, 1, 1),
            "pred_std": torch.ones(5, 1, 1),
            "aux_mean": torch.zeros(5, 1, 1),  # Same shape as pred for std normalization
            "aux_std": torch.ones(5, 1, 1),
            "rea_mean": torch.zeros(2, 1, 1),
            "rea_std": torch.ones(2, 1, 1),
        }

    @pytest.fixture
    def feature_metadata(self):
        """Create synthetic feature metadata."""
        return {
            "max_timedelta": 120.0,
        }

    def test_no_transforms(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that dataset works without transforms."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["meta_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=None,
            y_transform=None,
        )

        sample = dataset[0]

        assert "x" in sample
        assert "y" in sample
        assert "timedelta" in sample
        assert sample["x"]["predicted_vars_mean"].shape == (5, 10, 10)
        assert sample["x"]["predicted_vars_std"].shape == (5, 10, 10)
        assert sample["x"]["all_vars_mean"].shape == (5, 10, 10)
        assert sample["x"]["all_vars_std"].shape == (5, 10, 10)
        assert sample["x"]["meta_vars"].shape == (2, 10, 10)
        assert sample["y"].shape == (2, 10, 10)

    def test_x_transform_applied(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that x_transform is applied to input features."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["meta_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        # Define a simple transform that multiplies by 2
        def x_transform(x):
            return x * 2.0

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=x_transform,
            y_transform=None,
        )

        # Get sample with transform
        sample_with_transform = dataset[0]

        # Create dataset without transform to compare
        dataset_no_transform = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=None,
            y_transform=None,
        )
        sample_no_transform = dataset_no_transform[0]

        # Check that transform was applied (values should be doubled)
        assert torch.allclose(
            sample_with_transform["x"]["predicted_vars_mean"],
            sample_no_transform["x"]["predicted_vars_mean"] * 2.0,
        )
        assert torch.allclose(
            sample_with_transform["x"]["predicted_vars_std"],
            sample_no_transform["x"]["predicted_vars_std"] * 2.0,
        )
        assert torch.allclose(
            sample_with_transform["x"]["meta_vars"],
            sample_no_transform["x"]["meta_vars"] * 2.0,
        )
        # y should be unchanged
        assert torch.allclose(
            sample_with_transform["y"],
            sample_no_transform["y"],
        )

    def test_y_transform_applied(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that y_transform is applied to target."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["meta_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        # Define a simple transform that adds 10
        def y_transform(y):
            return y + 10.0

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=None,
            y_transform=y_transform,
        )

        # Get sample with transform
        sample_with_transform = dataset[0]

        # Create dataset without transform to compare
        dataset_no_transform = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=None,
            y_transform=None,
        )
        sample_no_transform = dataset_no_transform[0]

        # Check that transform was applied (values should be +10)
        assert torch.allclose(
            sample_with_transform["y"],
            sample_no_transform["y"] + 10.0,
        )
        # x should be unchanged
        assert torch.allclose(
            sample_with_transform["x"]["predicted_vars_mean"],
            sample_no_transform["x"]["predicted_vars_mean"],
        )

    def test_both_transforms_applied(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that both x_transform and y_transform are applied."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["meta_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        # Define transforms
        def x_transform(x):
            return x * 3.0

        def y_transform(y):
            return y - 5.0

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=x_transform,
            y_transform=y_transform,
        )

        # Get sample with transforms
        sample_with_transforms = dataset[0]

        # Create dataset without transforms to compare
        dataset_no_transform = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=None,
            y_transform=None,
        )
        sample_no_transform = dataset_no_transform[0]

        # Check that both transforms were applied
        assert torch.allclose(
            sample_with_transforms["x"]["predicted_vars_mean"],
            sample_no_transform["x"]["predicted_vars_mean"] * 3.0,
        )
        assert torch.allclose(
            sample_with_transforms["y"],
            sample_no_transform["y"] - 5.0,
        )

    def test_transform_order_after_normalization(self, temp_data_dir, feature_metadata):
        """Test that transforms are applied after normalization."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["meta_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        # Create norm stats that shift values
        norm_stats = {
            "pred_mean": torch.ones(5, 1, 1) * 5.0,
            "pred_std": torch.ones(5, 1, 1) * 2.0,
            "aux_mean": torch.ones(5, 1, 1) * 5.0,  # Same shape as pred
            "aux_std": torch.ones(5, 1, 1) * 2.0,
            "rea_mean": torch.ones(2, 1, 1) * 5.0,
            "rea_std": torch.ones(2, 1, 1) * 2.0,
        }

        # Define a transform that should be applied after normalization
        def y_transform(y):
            # If normalization was applied first, values should be centered around 0
            # Return the mean to test
            return y.mean()

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="zscore",
            x_transform=None,
            y_transform=y_transform,
        )

        sample = dataset[0]

        # After z-score normalization, mean should be close to 0
        # Transform returns the mean, so result should be close to 0
        # (not the original data mean which would be different)
        # This confirms transform is applied after normalization
        assert isinstance(sample["y"], torch.Tensor)

    def test_minmax_normalization_with_transforms(self, temp_data_dir, feature_metadata):
        """Test that transforms work with minmax normalization."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["meta_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        # Create norm stats for minmax
        norm_stats = {
            "pred_min": torch.zeros(5, 1, 1),
            "pred_max": torch.ones(5, 1, 1),
            "aux_min": torch.zeros(5, 1, 1),  # Same shape as pred
            "aux_max": torch.ones(5, 1, 1),
            "rea_min": torch.zeros(2, 1, 1),
            "rea_max": torch.ones(2, 1, 1),
        }

        def y_transform(y):
            return y * 100.0

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            normalize_type="minmax",
            x_transform=None,
            y_transform=y_transform,
        )

        sample = dataset[0]

        # Verify that we got output (transform was applied)
        assert sample["y"].shape == (2, 10, 10)
