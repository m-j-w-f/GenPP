"""Tests for spatial (per-coordinate) normalization in ForecastDataModule and ForecastDataset."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from genpp.data.icon.dataset import ForecastDataModule, ForecastDataset
from genpp.models.layers import ReverseAffineTransform


class TestSpatialNormMode:
    """Test suite for norm_mode parameter and spatial normalization."""

    @pytest.mark.unit
    def test_norm_mode_default_is_per_variable(self):
        """Test that default norm_mode is 'per_variable'."""
        dm = ForecastDataModule(
            x_select_variables=["var1"],
            y_select_variables=["var1"],
        )
        assert dm.norm_mode == "per_variable"

    @pytest.mark.unit
    def test_norm_mode_spatial_accepted(self):
        """Test that norm_mode='spatial' is accepted."""
        dm = ForecastDataModule(
            x_select_variables=["var1"],
            y_select_variables=["var1"],
            norm_mode="spatial",
        )
        assert dm.norm_mode == "spatial"

    @pytest.mark.unit
    def test_norm_mode_invalid_raises_error(self):
        """Test that invalid norm_mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown norm_mode"):
            ForecastDataModule(
                x_select_variables=["var1"],
                y_select_variables=["var1"],
                norm_mode="invalid",
            )


class TestSpatialReverseModules:
    """Test suite for y_reverseModules with spatial normalization."""

    @pytest.fixture
    def spatial_norm_stats(self):
        """Create spatial normalization statistics with shape [c, x, y]."""
        x_dim, y_dim = 10, 8
        return {
            "rea_mean": torch.randn(2, x_dim, y_dim) + torch.tensor([280.0, 5.0]).view(2, 1, 1),
            "rea_std": torch.rand(2, x_dim, y_dim) + torch.tensor([8.0, 2.0]).view(2, 1, 1),
            "rea_min": torch.randn(2, x_dim, y_dim) + torch.tensor([260.0, 0.0]).view(2, 1, 1),
            "rea_max": torch.randn(2, x_dim, y_dim) + torch.tensor([300.0, 30.0]).view(2, 1, 1),
        }

    @pytest.mark.unit
    def test_spatial_reverse_modules_zscore(self, spatial_norm_stats):
        """Test y_reverseModules with spatial zscore normalization."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="zscore",
            norm_mode="spatial",
        )
        dm.norm_stats = spatial_norm_stats

        modules = dm.y_reverseModules

        assert len(modules) == 2
        # mean and scale should be [x, y] tensors
        assert modules[0].mean.shape == (10, 8)
        assert modules[0].scale.shape == (10, 8)
        assert torch.allclose(modules[0].mean, spatial_norm_stats["rea_mean"][0])
        assert torch.allclose(modules[0].scale, spatial_norm_stats["rea_std"][0])
        assert torch.allclose(modules[1].mean, spatial_norm_stats["rea_mean"][1])
        assert torch.allclose(modules[1].scale, spatial_norm_stats["rea_std"][1])

    @pytest.mark.unit
    def test_spatial_reverse_modules_minmax(self, spatial_norm_stats):
        """Test y_reverseModules with spatial minmax normalization."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="minmax",
            norm_mode="spatial",
        )
        dm.norm_stats = spatial_norm_stats

        modules = dm.y_reverseModules

        assert len(modules) == 2
        assert modules[0].mean.shape == (10, 8)
        assert modules[0].scale.shape == (10, 8)
        # For minmax, mean = min, scale = max - min
        assert torch.allclose(modules[0].mean, spatial_norm_stats["rea_min"][0])
        expected_scale = spatial_norm_stats["rea_max"][0] - spatial_norm_stats["rea_min"][0]
        assert torch.allclose(modules[0].scale, expected_scale)

    @pytest.mark.unit
    def test_spatial_reverse_modules_none(self, spatial_norm_stats):
        """Test y_reverseModules with None normalization in spatial mode."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_normalize_types={"var1": None, "var2": None},
            norm_mode="spatial",
        )
        dm.norm_stats = spatial_norm_stats

        modules = dm.y_reverseModules
        # None normalization should produce scalar identity regardless of mode
        assert modules[0].mean.item() == pytest.approx(0.0)
        assert modules[0].scale.item() == pytest.approx(1.0)


class TestSpatialDatasetNormalization:
    """Test suite for ForecastDataset __getitem__ with spatial normalization stats."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with synthetic test data."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # unified tensor: 12 channels (5 mean + 5 std + 2 meta), spatial 10x8
            unified_tensor = torch.randn(12, 10, 8)
            rea_data = torch.randn(2, 10, 8)

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
    def spatial_norm_stats(self):
        """Create spatial normalization statistics with shape [c, x, y]."""
        return {
            "all_mean": torch.randn(5, 10, 8),
            "all_std": torch.rand(5, 10, 8) + 0.1,
            "all_min": torch.zeros(5, 10, 8),
            "all_max": torch.ones(5, 10, 8),
            "aux_mean": torch.randn(5, 10, 8),
            "aux_std": torch.rand(5, 10, 8) + 0.1,
            "aux_min": torch.zeros(5, 10, 8),
            "aux_max": torch.ones(5, 10, 8),
            "rea_mean": torch.randn(2, 10, 8),
            "rea_std": torch.rand(2, 10, 8) + 0.1,
            "rea_min": torch.zeros(2, 10, 8),
            "rea_max": torch.ones(2, 10, 8) * 2,
        }

    @pytest.fixture
    def feature_metadata(self):
        """Create feature metadata for 5 x vars, 2 y vars, 2 meta."""
        return {
            "max_timedelta": 120.0,
            "all_var_mean_indices": list(range(5)),
            "all_var_std_indices": list(range(5, 10)),
            "meta_var_indices": list(range(10, 12)),
            "meta_var_names": ["sin_prediction_time", "cos_prediction_time"],
            "predicted_var_mean_indices": [0, 1],
            "predicted_var_std_indices": [0, 1],
            "predicted_var_mean_names": ["var_a", "var_b"],
            "predicted_var_std_names": ["var_a", "var_b"],
            "all_var_mean_names": ["var_a", "var_b", "var_c", "var_d", "var_e"],
            "all_var_std_names": ["var_a", "var_b", "var_c", "var_d", "var_e"],
        }

    @pytest.mark.unit
    def test_spatial_zscore_normalization(self, temp_data_dir, spatial_norm_stats, feature_metadata):
        """Test that spatial zscore normalization produces correct shapes and values."""
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
            norm_stats=spatial_norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="zscore",
            y_default_normalize_type="zscore",
        )

        batch = dataset[0]
        # Verify shapes
        assert batch["y"].shape == (2, 10, 8)
        assert batch["x"]["all_vars_mean"].shape == (5, 10, 8)
        assert batch["x"]["all_vars_std"].shape == (5, 10, 8)

    @pytest.mark.unit
    def test_spatial_normalization_roundtrip(self, temp_data_dir, spatial_norm_stats, feature_metadata):
        """Test normalization -> denormalization roundtrip with spatial stats."""
        # Create known REA data
        rea_original = torch.randn(2, 10, 8)
        torch.save(rea_original, temp_data_dir["rea_path"])

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
            norm_stats=spatial_norm_stats,
            feature_metadata=feature_metadata,
            x_default_normalize_type="zscore",
            y_default_normalize_type="zscore",
        )

        batch = dataset[0]
        y_normalized = batch["y"]

        # Manually denormalize using the same stats
        y_denorm = y_normalized.clone()
        y_denorm[0] = y_normalized[0] * spatial_norm_stats["rea_std"][0] + spatial_norm_stats["rea_mean"][0]
        y_denorm[1] = y_normalized[1] * spatial_norm_stats["rea_std"][1] + spatial_norm_stats["rea_mean"][1]

        assert torch.allclose(y_denorm, rea_original, atol=1e-5)
