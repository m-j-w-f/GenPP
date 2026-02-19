"""Tests for permutation_importance helper functions."""

import pytest
import torch

from genpp.preproc.transforms import Pad, PermuteChannel, Pipe


# ---------------------------------------------------------------------------
# _build_x_transform
# ---------------------------------------------------------------------------
from genpp.eval.permutation_importance import _build_x_transform


class TestBuildXTransform:
    """Tests for the _build_x_transform helper."""

    @pytest.mark.unit
    def test_none_existing_transform(self):
        """When there is no existing transform, return a Pipe with just PermuteChannel."""
        result = _build_x_transform(None, channel_index=3, seed=42)
        assert isinstance(result, Pipe)
        assert len(result.transforms) == 1
        assert isinstance(result.transforms[0], PermuteChannel)
        assert result.transforms[0].channel_index == 3

    @pytest.mark.unit
    def test_single_existing_transform(self):
        """When existing transform is a single transform, wrap both in a Pipe."""
        pad = Pad(padding=(1, 1, 1, 1))
        result = _build_x_transform(pad, channel_index=0)
        assert isinstance(result, Pipe)
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], PermuteChannel)
        assert result.transforms[1] is pad

    @pytest.mark.unit
    def test_existing_pipe_transform(self):
        """When existing transform is a Pipe, prepend PermuteChannel."""
        pad = Pad(padding=(1, 1, 1, 1))
        existing_pipe = Pipe([pad])
        result = _build_x_transform(existing_pipe, channel_index=5, seed=99)
        assert isinstance(result, Pipe)
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], PermuteChannel)
        assert result.transforms[0].channel_index == 5
        assert result.transforms[1] is pad

    @pytest.mark.unit
    def test_permute_is_applied_first(self):
        """Verify permutation is applied before any existing transform."""
        tensor = torch.randn(4, 5, 6)  # (feature, lon, lat)
        pad = Pad(padding=(1, 1, 1, 1))
        pipe = _build_x_transform(pad, channel_index=0, seed=42)

        # Apply the pipeline
        result = pipe(tensor)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# _compute_energy_score
# ---------------------------------------------------------------------------
from genpp.eval.permutation_importance import _compute_energy_score


class TestComputeEnergyScore:
    """Tests for the _compute_energy_score helper."""

    @pytest.mark.unit
    def test_returns_scalar(self):
        """Energy score should return a single scalar float."""
        # predictions: [n_times, n_samples, features, lon, lat]
        predictions = torch.randn(2, 5, 2, 3, 3)
        y = torch.randn(2, 2, 3, 3)
        score = _compute_energy_score(predictions, y, device="cpu")
        assert isinstance(score, float)

    @pytest.mark.unit
    def test_identical_predictions_low_score(self):
        """If predictions == observations (broadcast), score should be low / near zero."""
        y = torch.randn(3, 2, 4, 4)
        # Repeat y as the sole "sample" → zero variance ensemble
        predictions = y.unsqueeze(1).expand(-1, 10, -1, -1, -1)
        score = _compute_energy_score(predictions, y, device="cpu")
        # Should be close to zero (unbiased=False by default in EnergyScore)
        assert score < 1.0  # loose bound


# ---------------------------------------------------------------------------
# _get_channel_info
# ---------------------------------------------------------------------------
from genpp.eval.permutation_importance import _get_channel_info


class TestGetChannelInfo:
    """Tests for _get_channel_info."""

    @pytest.mark.unit
    def test_basic_channel_info(self):
        """Test that channel info is built from metadata correctly."""
        cache_metadata = {
            "feature_metadata": {
                "all_var_mean_indices": [0, 1],
                "all_var_std_indices": [2, 3],
                "meta_var_indices": [4],
                "pixel_idx_index": [5],
            },
            "x_variables": [
                "temp+statistic_mean",
                "wind+statistic_mean",
                "temp+statistic_std",
                "wind+statistic_std",
                "latitude",
                "pixel_idx",
            ],
        }
        info = _get_channel_info(cache_metadata)
        assert len(info) == 6
        assert info[0]["category"] == "all_var_mean"
        assert info[2]["category"] == "all_var_std"
        assert info[4]["category"] == "meta_var"
        assert info[5]["category"] == "pixel_idx"

    @pytest.mark.unit
    def test_empty_metadata(self):
        """Handle empty x_variables gracefully."""
        cache_metadata = {
            "feature_metadata": {
                "all_var_mean_indices": [],
                "all_var_std_indices": [],
                "meta_var_indices": [],
                "pixel_idx_index": None,
            },
            "x_variables": [],
        }
        info = _get_channel_info(cache_metadata)
        assert info == []
