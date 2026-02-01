"""Tests for ForecastDataModule y_reverseModules property."""

import pytest
import torch

from genpp.data.icon.dataset import ForecastDataModule
from genpp.models.layers import ReverseAffineTransform


class TestForecastDataModuleReverseModules:
    """Test suite for ForecastDataModule y_reverseModules property."""

    @pytest.fixture
    def norm_stats(self):
        """Create synthetic normalization statistics."""
        return {
            "rea_mean": torch.tensor([[[10.0]], [[20.0]]]),  # shape [c, 1, 1]
            "rea_std": torch.tensor([[[2.0]], [[4.0]]]),     # shape [c, 1, 1]
            "rea_min": torch.tensor([[[5.0]], [[10.0]]]),    # shape [c, 1, 1]
            "rea_max": torch.tensor([[[15.0]], [[30.0]]]),   # shape [c, 1, 1]
        }

    @pytest.mark.unit
    def test_y_reverse_modules_without_norm_stats_raises_error(self):
        """Test that accessing y_reverseModules without norm_stats raises RuntimeError."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
        )
        # norm_stats is None by default

        with pytest.raises(RuntimeError, match="Normalization statistics not available"):
            _ = dm.y_reverseModules

    @pytest.mark.unit
    def test_y_reverse_modules_returns_list(self, norm_stats):
        """Test that y_reverseModules returns a list with one ReverseAffineTransform."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules

        assert isinstance(modules, list), "Expected list"
        assert len(modules) == 1, "Expected exactly one module"
        assert isinstance(modules[0], ReverseAffineTransform), "Expected ReverseAffineTransform"

    @pytest.mark.unit
    def test_y_reverse_modules_zscore_default(self, norm_stats):
        """Test y_reverseModules with default zscore normalization."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="zscore",
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules
        module = modules[0]

        # For zscore, mean = rea_mean and scale = rea_std
        assert module.mean.shape == torch.Size([2]), f"Expected shape [2], got {module.mean.shape}"
        assert module.scale.shape == torch.Size([2]), f"Expected shape [2], got {module.scale.shape}"

        # Check values (squeezed from [c, 1, 1])
        expected_mean = torch.tensor([10.0, 20.0])
        expected_scale = torch.tensor([2.0, 4.0])

        assert torch.allclose(module.mean, expected_mean), f"Mean mismatch: {module.mean} vs {expected_mean}"
        assert torch.allclose(module.scale, expected_scale), f"Scale mismatch: {module.scale} vs {expected_scale}"

    @pytest.mark.unit
    def test_y_reverse_modules_minmax_default(self, norm_stats):
        """Test y_reverseModules with default minmax normalization."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="minmax",
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules
        module = modules[0]

        # For minmax, mean = rea_min and scale = rea_max - rea_min
        expected_mean = torch.tensor([5.0, 10.0])
        expected_scale = torch.tensor([10.0, 20.0])  # (15-5), (30-10)

        assert torch.allclose(module.mean, expected_mean), f"Mean mismatch: {module.mean} vs {expected_mean}"
        assert torch.allclose(module.scale, expected_scale), f"Scale mismatch: {module.scale} vs {expected_scale}"

    @pytest.mark.unit
    def test_y_reverse_modules_mixed_normalization(self, norm_stats):
        """Test y_reverseModules with mixed per-variable normalization."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="zscore",
            y_normalize_types={"var2": "minmax"},  # var1 uses default zscore
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules
        module = modules[0]

        # var1 uses zscore: mean=10, scale=2
        # var2 uses minmax: mean=10, scale=20
        expected_mean = torch.tensor([10.0, 10.0])
        expected_scale = torch.tensor([2.0, 20.0])

        assert torch.allclose(module.mean, expected_mean), f"Mean mismatch: {module.mean} vs {expected_mean}"
        assert torch.allclose(module.scale, expected_scale), f"Scale mismatch: {module.scale} vs {expected_scale}"

    @pytest.mark.unit
    def test_y_reverse_modules_none_normalization(self, norm_stats):
        """Test y_reverseModules with None normalization (identity transform)."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="zscore",
            y_normalize_types={"var1": None},  # var1 has no normalization
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules
        module = modules[0]

        # var1 uses None: mean=0, scale=1 (identity)
        # var2 uses zscore: mean=20, scale=4
        expected_mean = torch.tensor([0.0, 20.0])
        expected_scale = torch.tensor([1.0, 4.0])

        assert torch.allclose(module.mean, expected_mean), f"Mean mismatch: {module.mean} vs {expected_mean}"
        assert torch.allclose(module.scale, expected_scale), f"Scale mismatch: {module.scale} vs {expected_scale}"

    @pytest.mark.unit
    def test_y_reverse_modules_forward_pass(self, norm_stats):
        """Test that the reverse module forward pass works correctly."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_default_normalize_type="zscore",
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules
        module = modules[0]

        # Create sample input: normalized values
        # Shape: [batch, channels, height, width]
        mu = torch.zeros(2, 2, 3, 3)  # All zeros (normalized mean)
        sigma = torch.ones(2, 2, 3, 3)  # All ones (normalized std)

        # Forward pass
        result_mu, result_sigma = module(mu, sigma)

        # Expected: mu_denorm = mu * scale + mean = 0 * [2, 4] + [10, 20] = [10, 20]
        # sigma_denorm = sigma * scale = 1 * [2, 4] = [2, 4]
        assert result_mu.shape == torch.Size([2, 2, 3, 3])
        assert result_sigma.shape == torch.Size([2, 2, 3, 3])

        # Check that denormalization is applied correctly
        # Channel 0 should have mu=10, sigma=2
        assert torch.allclose(result_mu[:, 0], torch.full((2, 3, 3), 10.0))
        assert torch.allclose(result_sigma[:, 0], torch.full((2, 3, 3), 2.0))

        # Channel 1 should have mu=20, sigma=4
        assert torch.allclose(result_mu[:, 1], torch.full((2, 3, 3), 20.0))
        assert torch.allclose(result_sigma[:, 1], torch.full((2, 3, 3), 4.0))

    @pytest.mark.unit
    def test_y_reverse_modules_invalid_norm_type(self, norm_stats):
        """Test that invalid normalization type raises ValueError."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
            y_normalize_types={"var1": "invalid_type"},
        )
        dm.norm_stats = norm_stats

        with pytest.raises(ValueError, match="Unknown normalization type"):
            _ = dm.y_reverseModules
