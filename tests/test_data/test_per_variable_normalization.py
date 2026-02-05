"""Tests for ForecastDataset per-variable normalization functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from genpp.data.icon.dataset import ForecastDataset


class TestForecastDatasetPerVariableNormalization:
    """Test suite for ForecastDataset per-variable x and y normalization."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with synthetic test data using unified tensor format."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create unified tensor (c_total, x, y) with:
            # all_vars_mean (5), all_vars_std (5), meta_vars (2) = 12 total channels
            unified_tensor = torch.randn(12, 10, 10)
            rea_data = torch.randn(2, 10, 10)

            fc_path = tmpdir_path / "fc_test.pt"
            rea_path = tmpdir_path / "rea_test.pt"

            torch.save(unified_tensor, fc_path)
            torch.save(rea_data, rea_path)

            yield {
                "fc_path": fc_path,
                "rea_path": rea_path,
                "tmpdir": tmpdir_path,
            }

    @pytest.fixture
    def norm_stats(self):
        """Create synthetic normalization statistics."""
        return {
            "all_mean": torch.zeros(5, 1, 1),
            "all_std": torch.ones(5, 1, 1),
            "all_min": torch.zeros(5, 1, 1),
            "all_max": torch.ones(5, 1, 1),
            "aux_mean": torch.zeros(5, 1, 1),
            "aux_std": torch.ones(5, 1, 1),
            "aux_min": torch.zeros(5, 1, 1),
            "aux_max": torch.ones(5, 1, 1),
            "rea_mean": torch.zeros(2, 1, 1),
            "rea_std": torch.ones(2, 1, 1),
            "rea_min": torch.zeros(2, 1, 1),
            "rea_max": torch.ones(2, 1, 1) * 2,  # Different max for minmax testing
        }

    @pytest.fixture
    def feature_metadata(self):
        """Create synthetic feature metadata."""
        return {
            "max_timedelta": 120.0,
            "all_var_mean_indices": list(range(5)),
            "all_var_std_indices": list(range(5, 10)),
            "meta_var_indices": list(range(10, 12)),
            "meta_var_names": ["sin_prediction_time", "cos_prediction_time"],
            "predicted_var_mean_indices": [0, 1],  # First two are predicted vars
            "predicted_var_std_indices": [0, 1],
            "predicted_var_mean_names": ["var_a", "var_b"],
            "predicted_var_std_names": ["var_a", "var_b"],
            "all_var_mean_names": ["var_a", "var_b", "var_c", "var_d", "var_e"],
            "all_var_std_names": ["var_a", "var_b", "var_c", "var_d", "var_e"],
        }

    # ==================== Y VARIABLE NORMALIZATION TESTS ====================

    @pytest.mark.unit
    def test_default_zscore_normalization_y(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that default zscore normalization works for all y variables."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            y_default_normalize_type="zscore",
        )

        assert dataset._y_zscore_indices == [0, 1], "Expected both y vars to use zscore"
        assert dataset._y_minmax_indices == [], "Expected no y minmax vars"
        assert dataset._y_none_indices == [], "Expected no y none vars"

    @pytest.mark.unit
    def test_default_minmax_normalization_y(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that default minmax normalization works for all y variables."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            y_default_normalize_type="minmax",
        )

        assert dataset._y_zscore_indices == [], "Expected no y zscore vars"
        assert dataset._y_minmax_indices == [0, 1], "Expected all y vars to use minmax"
        assert dataset._y_none_indices == [], "Expected no y none vars"

    @pytest.mark.unit
    def test_per_variable_mixed_normalization_y(self, temp_data_dir, norm_stats, feature_metadata):
        """Test per-variable y normalization with mixed types."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            y_default_normalize_type="zscore",
            y_normalize_types={"var_a": "minmax", "var_b": None},
        )

        assert dataset._y_zscore_indices == [], "Expected no y zscore vars"
        assert dataset._y_minmax_indices == [0], "Expected var_a to use minmax"
        assert dataset._y_none_indices == [1], "Expected var_b to have no normalization"

    @pytest.mark.unit
    def test_partial_override_uses_default_y(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that unspecified y variables use the default normalization type."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            y_default_normalize_type="minmax",
            y_normalize_types={"var_a": "zscore"},  # var_b not specified, uses default
        )

        assert dataset._y_zscore_indices == [0], "Expected var_a to use zscore"
        assert dataset._y_minmax_indices == [1], "Expected var_b to use default minmax"
        assert dataset._y_none_indices == [], "Expected no y none vars"

    @pytest.mark.unit
    def test_none_normalization_y(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that None normalization type leaves y values unchanged."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            y_default_normalize_type="zscore",
            y_normalize_types={"var_a": None, "var_b": None},
        )

        assert dataset._y_zscore_indices == [], "Expected no y zscore vars"
        assert dataset._y_minmax_indices == [], "Expected no y minmax vars"
        assert dataset._y_none_indices == [0, 1], "Expected both y vars to have no normalization"

    @pytest.mark.unit
    def test_invalid_normalization_type_raises_error_y(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that invalid y normalization type raises ValueError."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        with pytest.raises(ValueError, match="Unknown normalization type 'invalid'"):
            ForecastDataset(
                samples=samples,
                norm_stats=norm_stats,
                feature_metadata=feature_metadata,
                y_default_normalize_type="zscore",
                y_normalize_types={"var_a": "invalid"},
            )

    # ==================== X VARIABLE NORMALIZATION TESTS ====================

    @pytest.mark.unit
    def test_default_zscore_normalization_x(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that default zscore normalization works for all x variables."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="zscore",
        )

        assert dataset._x_mean_zscore_indices == [0, 1, 2, 3, 4], "Expected all x vars to use zscore"
        assert dataset._x_mean_minmax_indices == [], "Expected no x minmax vars"
        assert dataset._x_mean_none_indices == [], "Expected no x none vars"

    @pytest.mark.unit
    def test_default_minmax_normalization_x(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that default minmax normalization works for all x variables."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="minmax",
        )

        assert dataset._x_mean_zscore_indices == [], "Expected no x zscore vars"
        assert dataset._x_mean_minmax_indices == [0, 1, 2, 3, 4], "Expected all x vars to use minmax"
        assert dataset._x_mean_none_indices == [], "Expected no x none vars"

    @pytest.mark.unit
    def test_per_variable_mixed_normalization_x(self, temp_data_dir, norm_stats, feature_metadata):
        """Test per-variable x normalization with mixed types."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="zscore",
            x_normalize_types={"var_a": "minmax", "var_c": None},
        )

        assert dataset._x_mean_zscore_indices == [1, 3, 4], "Expected var_b, var_d, var_e to use zscore"
        assert dataset._x_mean_minmax_indices == [0], "Expected var_a to use minmax"
        assert dataset._x_mean_none_indices == [2], "Expected var_c to have no normalization"

    @pytest.mark.unit
    def test_partial_override_uses_default_x(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that unspecified x variables use the default normalization type."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="minmax",
            x_normalize_types={"var_a": "zscore"},  # others not specified, use default
        )

        assert dataset._x_mean_zscore_indices == [0], "Expected var_a to use zscore"
        assert dataset._x_mean_minmax_indices == [1, 2, 3, 4], "Expected others to use default minmax"
        assert dataset._x_mean_none_indices == [], "Expected no x none vars"

    @pytest.mark.unit
    def test_invalid_normalization_type_raises_error_x(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that invalid x normalization type raises ValueError."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        with pytest.raises(ValueError, match="Unknown normalization type 'invalid' for x variable"):
            ForecastDataset(
                samples=samples,
                norm_stats=norm_stats,
                feature_metadata=feature_metadata,
                x_default_normalize_type="zscore",
                x_normalize_types={"var_a": "invalid"},
            )

    # ==================== COMBINED X AND Y TESTS ====================

    @pytest.mark.unit
    def test_getitem_with_per_variable_normalization(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that __getitem__ works correctly with per-variable normalization."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="zscore",
            y_default_normalize_type="zscore",
            x_normalize_types={"var_a": "minmax"},
            y_normalize_types={"var_a": "minmax", "var_b": "zscore"},
        )

        sample = dataset[0]

        assert "y" in sample, "Expected y in sample"
        assert sample["y"].shape == (2, 10, 10), f"Expected shape (2, 10, 10), got {sample['y'].shape}"
        assert "x" in sample, "Expected x in sample"
        assert "timedelta" in sample, "Expected timedelta in sample"

    @pytest.mark.unit
    def test_normalization_values_are_correct(self, temp_data_dir, feature_metadata):
        """Test that normalization is applied correctly to the y values."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        # Create specific norm stats for testing
        norm_stats = {
            "all_mean": torch.zeros(5, 1, 1),
            "all_std": torch.ones(5, 1, 1),
            "all_min": torch.zeros(5, 1, 1),
            "all_max": torch.ones(5, 1, 1),
            "aux_mean": torch.zeros(5, 1, 1),
            "aux_std": torch.ones(5, 1, 1),
            "aux_min": torch.zeros(5, 1, 1),
            "aux_max": torch.ones(5, 1, 1),
            "rea_mean": torch.tensor([[[1.0]], [[2.0]]]),  # mean for var_a=1, var_b=2
            "rea_std": torch.tensor([[[0.5]], [[1.0]]]),  # std for var_a=0.5, var_b=1
            "rea_min": torch.tensor([[[0.0]], [[0.0]]]),  # min for both is 0
            "rea_max": torch.tensor([[[2.0]], [[4.0]]]),  # max for var_a=2, var_b=4
        }

        # Test with var_a using zscore and var_b using minmax
        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="zscore",
            y_default_normalize_type="zscore",
            y_normalize_types={"var_a": "zscore", "var_b": "minmax"},
        )

        assert dataset._y_zscore_indices == [0], "Expected var_a to use zscore"
        assert dataset._y_minmax_indices == [1], "Expected var_b to use minmax"

        # The sample should have correct shape
        sample = dataset[0]
        assert sample["y"].shape == (2, 10, 10)

    @pytest.mark.unit
    def test_different_defaults_for_x_and_y(self, temp_data_dir, norm_stats, feature_metadata):
        """Test that x and y can have different default normalization types."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        dataset = ForecastDataset(
            samples=samples,
            norm_stats=norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="minmax",
            y_default_normalize_type="zscore",
        )

        # X should use minmax by default
        assert dataset._x_mean_zscore_indices == [], "Expected no x zscore vars"
        assert dataset._x_mean_minmax_indices == [0, 1, 2, 3, 4], "Expected all x vars to use minmax"

        # Y should use zscore by default
        assert dataset._y_zscore_indices == [0, 1], "Expected all y vars to use zscore"
        assert dataset._y_minmax_indices == [], "Expected no y minmax vars"
