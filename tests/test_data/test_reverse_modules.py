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
        """Test that y_reverseModules returns a list with one ReverseAffineTransform per variable."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1", "var2"],
        )
        dm.norm_stats = norm_stats

        modules = dm.y_reverseModules

        assert isinstance(modules, list), "Expected list"
        assert len(modules) == 2, "Expected one module per y variable"
        for module in modules:
            assert isinstance(module, ReverseAffineTransform), "Expected ReverseAffineTransform"

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

        assert len(modules) == 2, "Expected one module per y variable"

        # For zscore, mean = rea_mean and scale = rea_std
        # Module 0 (var1): mean=10, scale=2
        assert modules[0].mean.squeeze().item() == pytest.approx(10.0)
        assert modules[0].scale.squeeze().item() == pytest.approx(2.0)

        # Module 1 (var2): mean=20, scale=4
        assert modules[1].mean.squeeze().item() == pytest.approx(20.0)
        assert modules[1].scale.squeeze().item() == pytest.approx(4.0)

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

        assert len(modules) == 2, "Expected one module per y variable"

        # For minmax, mean = rea_min and scale = rea_max - rea_min
        # Module 0 (var1): mean=5, scale=10 (15-5)
        assert modules[0].mean.squeeze().item() == pytest.approx(5.0)
        assert modules[0].scale.squeeze().item() == pytest.approx(10.0)

        # Module 1 (var2): mean=10, scale=20 (30-10)
        assert modules[1].mean.squeeze().item() == pytest.approx(10.0)
        assert modules[1].scale.squeeze().item() == pytest.approx(20.0)

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

        assert len(modules) == 2, "Expected one module per y variable"

        # var1 uses zscore: mean=10, scale=2
        assert modules[0].mean.squeeze().item() == pytest.approx(10.0)
        assert modules[0].scale.squeeze().item() == pytest.approx(2.0)

        # var2 uses minmax: mean=10, scale=20 (30-10)
        assert modules[1].mean.squeeze().item() == pytest.approx(10.0)
        assert modules[1].scale.squeeze().item() == pytest.approx(20.0)

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

        assert len(modules) == 2, "Expected one module per y variable"

        # var1 uses None: mean=0, scale=1 (identity)
        assert modules[0].mean.squeeze().item() == pytest.approx(0.0)
        assert modules[0].scale.squeeze().item() == pytest.approx(1.0)

        # var2 uses zscore: mean=20, scale=4
        assert modules[1].mean.squeeze().item() == pytest.approx(20.0)
        assert modules[1].scale.squeeze().item() == pytest.approx(4.0)

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

        assert len(modules) == 2, "Expected one module per y variable"

        # Test module 0 (var1): mean=10, scale=2
        # Create sample input: normalized values for single variable
        # Shape: [batch, height, width]
        mu0 = torch.zeros(2, 3, 3)  # All zeros (normalized mean)
        sigma0 = torch.ones(2, 3, 3)  # All ones (normalized std)

        # Forward pass
        result_mu0, result_sigma0 = modules[0](mu0, sigma0)

        # Expected: mu_denorm = mu * scale + mean = 0 * 2 + 10 = 10
        # sigma_denorm = sigma * scale = 1 * 2 = 2
        assert torch.allclose(result_mu0, torch.full((2, 3, 3), 10.0))
        assert torch.allclose(result_sigma0, torch.full((2, 3, 3), 2.0))

        # Test module 1 (var2): mean=20, scale=4
        mu1 = torch.zeros(2, 3, 3)
        sigma1 = torch.ones(2, 3, 3)

        result_mu1, result_sigma1 = modules[1](mu1, sigma1)

        # Expected: mu_denorm = mu * scale + mean = 0 * 4 + 20 = 20
        # sigma_denorm = sigma * scale = 1 * 4 = 4
        assert torch.allclose(result_mu1, torch.full((2, 3, 3), 20.0))
        assert torch.allclose(result_sigma1, torch.full((2, 3, 3), 4.0))

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
