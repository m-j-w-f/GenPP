"""Tests for ICON predict eval helper functions."""

import pytest
import torch

from genpp.eval.icon_predict_eval import _rescale_y
from genpp.models.layers import ReverseAffineTransform


class TestRescaleY:
    """Test suite for _rescale_y function."""

    @pytest.fixture
    def reverse_modules_zscore(self):
        """Create reverse modules for 2 variables with zscore normalization."""
        return [
            ReverseAffineTransform(mean=torch.tensor(280.0), scale=torch.tensor(10.0)),
            ReverseAffineTransform(mean=torch.tensor(5.0), scale=torch.tensor(3.0)),
        ]

    @pytest.mark.unit
    def test_rescale_ground_truth_4d(self, reverse_modules_zscore):
        """Test rescaling ground truth with shape [N, c, x, y]."""
        N, c, x, y = 4, 2, 8, 6
        y_normalized = torch.randn(N, c, x, y)
        y_rescaled = _rescale_y(y_normalized, reverse_modules_zscore)

        assert y_rescaled.shape == (N, c, x, y)
        # Channel 0: y * 10 + 280
        expected_ch0 = y_normalized[:, 0, :, :] * 10.0 + 280.0
        assert torch.allclose(y_rescaled[:, 0, :, :], expected_ch0)
        # Channel 1: y * 3 + 5
        expected_ch1 = y_normalized[:, 1, :, :] * 3.0 + 5.0
        assert torch.allclose(y_rescaled[:, 1, :, :], expected_ch1)

    @pytest.mark.unit
    def test_rescale_predictions_5d(self, reverse_modules_zscore):
        """Test rescaling predictions with shape [N, n_samples, c, x, y]."""
        N, n_samples, c, x, y = 4, 40, 2, 8, 6
        pred_normalized = torch.randn(N, n_samples, c, x, y)
        pred_rescaled = _rescale_y(pred_normalized, reverse_modules_zscore)

        assert pred_rescaled.shape == (N, n_samples, c, x, y)
        # Channel 0 across all samples: y * 10 + 280
        expected_ch0 = pred_normalized[:, :, 0, :, :] * 10.0 + 280.0
        assert torch.allclose(pred_rescaled[:, :, 0, :, :], expected_ch0)
        # Channel 1 across all samples: y * 3 + 5
        expected_ch1 = pred_normalized[:, :, 1, :, :] * 3.0 + 5.0
        assert torch.allclose(pred_rescaled[:, :, 1, :, :], expected_ch1)

    @pytest.mark.unit
    def test_rescale_does_not_modify_input(self, reverse_modules_zscore):
        """Test that _rescale_y does not modify the input tensor."""
        y = torch.randn(2, 2, 4, 4)
        y_original = y.clone()
        _rescale_y(y, reverse_modules_zscore)
        assert torch.equal(y, y_original)

    @pytest.mark.unit
    def test_rescale_roundtrip_zscore(self, reverse_modules_zscore):
        """Test normalization -> denormalization roundtrip preserves values."""
        # Original data
        y_original = torch.randn(3, 2, 5, 5)
        y_original[:, 0] = y_original[:, 0] * 10.0 + 280.0  # T_2M scale
        y_original[:, 1] = y_original[:, 1] * 3.0 + 5.0  # VMAX_10M scale

        # Normalize (as done in dataset.__getitem__)
        y_normalized = y_original.clone()
        y_normalized[:, 0] = (y_original[:, 0] - 280.0) / 10.0
        y_normalized[:, 1] = (y_original[:, 1] - 5.0) / 3.0

        # Denormalize (as done in _rescale_y)
        y_roundtrip = _rescale_y(y_normalized, reverse_modules_zscore)

        assert torch.allclose(y_roundtrip, y_original, atol=1e-5)

    @pytest.mark.unit
    def test_rescale_roundtrip_minmax(self):
        """Test minmax normalization -> denormalization roundtrip."""
        # minmax: mean = min, scale = max - min
        reverse_modules = [
            ReverseAffineTransform(mean=torch.tensor(0.0), scale=torch.tensor(10.0)),
            ReverseAffineTransform(mean=torch.tensor(-5.0), scale=torch.tensor(20.0)),
        ]

        y_original = torch.randn(3, 2, 5, 5)
        y_original[:, 0] = y_original[:, 0] * 10.0  # range [0, 10]
        y_original[:, 1] = y_original[:, 1] * 20.0 - 5.0  # range [-5, 15]

        # Normalize (minmax)
        y_normalized = y_original.clone()
        y_normalized[:, 0] = (y_original[:, 0] - 0.0) / 10.0
        y_normalized[:, 1] = (y_original[:, 1] - (-5.0)) / 20.0

        # Denormalize
        y_roundtrip = _rescale_y(y_normalized, reverse_modules)

        assert torch.allclose(y_roundtrip, y_original, atol=1e-5)
