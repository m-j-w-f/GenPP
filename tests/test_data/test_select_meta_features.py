"""Tests for ForecastDataset select_meta_features functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from genpp.data.icon.dataset import ForecastDataset, ForecastDataModule


class TestForecastDatasetSelectMetaFeatures:
    """Test suite for ForecastDataset select_meta_features parameter."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with synthetic test data using unified tensor format."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create unified tensor (c_total, x, y) with:
            # all_vars_mean (5), all_vars_std (5), meta_vars (4) = 14 total channels
            unified_tensor = torch.randn(14, 10, 10)
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
            "rea_max": torch.ones(2, 1, 1) * 2,
        }

    @pytest.fixture
    def feature_metadata(self):
        """Create synthetic feature metadata with 4 meta features."""
        return {
            "max_timedelta": 120.0,
            "all_var_mean_indices": list(range(5)),
            "all_var_std_indices": list(range(5, 10)),
            "meta_var_indices": list(range(10, 14)),  # 4 meta features
            "meta_var_names": [
                "sin_prediction_time",
                "cos_prediction_time",
                "latitude",
                "longitude",
            ],
            "predicted_var_mean_indices": [0, 1],
            "predicted_var_std_indices": [0, 1],
            "predicted_var_mean_names": ["var_a", "var_b"],
            "predicted_var_std_names": ["var_a", "var_b"],
            "all_var_mean_names": ["var_a", "var_b", "var_c", "var_d", "var_e"],
            "all_var_std_names": ["var_a", "var_b", "var_c", "var_d", "var_e"],
        }

    # ==================== SELECT_META_FEATURES TESTS ====================

    @pytest.mark.unit
    def test_select_all_meta_features_when_none(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that all meta features are selected when select_meta_features is None."""
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
            select_meta_features=None,  # All meta features
        )

        assert dataset._selected_meta_indices == [10, 11, 12, 13], (
            "Expected all meta indices when select_meta_features is None"
        )

    @pytest.mark.unit
    def test_select_no_meta_features_with_empty_list(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that no meta features are selected when select_meta_features is empty."""
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
            select_meta_features=[],  # No meta features (EMOS case)
        )

        assert dataset._selected_meta_indices == [], (
            "Expected empty meta indices when select_meta_features is []"
        )

    @pytest.mark.unit
    def test_select_subset_of_meta_features(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test selecting a subset of meta features."""
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
            select_meta_features=["sin_prediction_time", "latitude"],
        )

        # sin_prediction_time is at index 10, latitude is at index 12
        assert dataset._selected_meta_indices == [10, 12], (
            "Expected selected meta indices [10, 12]"
        )

    @pytest.mark.unit
    def test_select_single_meta_feature(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test selecting a single meta feature."""
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
            select_meta_features=["longitude"],
        )

        # longitude is at index 13
        assert dataset._selected_meta_indices == [13], (
            "Expected selected meta indices [13]"
        )

    @pytest.mark.unit
    def test_invalid_meta_feature_raises_error(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that an invalid meta feature name raises ValueError."""
        samples = [
            (
                temp_data_dir["fc_path"],
                temp_data_dir["rea_path"],
                np.datetime64("2023-01-01"),
                np.timedelta64(6, "h"),
            )
        ]

        with pytest.raises(ValueError, match="Metadata feature 'invalid_feature' not found"):
            ForecastDataset(
                samples=samples,
                norm_stats=norm_stats,
                feature_metadata=feature_metadata,
                select_meta_features=["invalid_feature"],
            )

    @pytest.mark.unit
    def test_getitem_returns_correct_meta_shape_all(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that __getitem__ returns meta_vars with correct shape when all selected."""
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
            select_meta_features=None,  # All 4 meta features
        )

        sample = dataset[0]

        assert sample["x"]["meta_vars"].shape == (4, 10, 10), (
            f"Expected shape (4, 10, 10), got {sample['x']['meta_vars'].shape}"
        )

    @pytest.mark.unit
    def test_getitem_returns_correct_meta_shape_subset(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that __getitem__ returns meta_vars with correct shape when subset selected."""
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
            select_meta_features=["sin_prediction_time", "latitude"],
        )

        sample = dataset[0]

        assert sample["x"]["meta_vars"].shape == (2, 10, 10), (
            f"Expected shape (2, 10, 10), got {sample['x']['meta_vars'].shape}"
        )

    @pytest.mark.unit
    def test_getitem_returns_empty_meta_when_none_selected(
        self, temp_data_dir, norm_stats, feature_metadata
    ):
        """Test that __getitem__ returns empty meta_vars when none selected."""
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
            select_meta_features=[],  # EMOS case: no meta features
        )

        sample = dataset[0]

        assert sample["x"]["meta_vars"].shape == (0, 10, 10), (
            f"Expected shape (0, 10, 10), got {sample['x']['meta_vars'].shape}"
        )


class TestForecastDataModuleSelectMetaFeatures:
    """Test suite for ForecastDataModule select_meta_features parameter."""

    @pytest.mark.unit
    def test_datamodule_stores_select_meta_features(self):
        """Test that ForecastDataModule stores select_meta_features."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            select_meta_features=["sin_prediction_time", "cos_prediction_time"],
        )

        assert dm.select_meta_features == ["sin_prediction_time", "cos_prediction_time"]

    @pytest.mark.unit
    def test_datamodule_stores_empty_select_meta_features(self):
        """Test that ForecastDataModule stores empty select_meta_features for EMOS."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            select_meta_features=[],  # EMOS case
        )

        assert dm.select_meta_features == []

    @pytest.mark.unit
    def test_datamodule_stores_none_select_meta_features(self):
        """Test that ForecastDataModule stores None select_meta_features (all features)."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            select_meta_features=None,  # All features
        )

        assert dm.select_meta_features is None
