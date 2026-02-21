"""Tests for permutation_importance_drn helper functions."""

import numpy as np
import pytest
import torch
import xarray as xr

from genpp.eval.permutation_importance_drn import _compute_copula_energy_score


class TestComputeCopulaEnergyScore:
    """Tests for _compute_copula_energy_score."""

    @pytest.fixture
    def sample_xarray_data(self):
        """Create sample xarray DataArrays for pred_samples and y_obs."""
        n_pred, n_samples, n_feat, n_lon, n_lat = 4, 10, 2, 3, 3
        rng = np.random.default_rng(42)

        pred_data = rng.standard_normal((n_pred, n_samples, n_feat, n_lon, n_lat))
        obs_data = rng.standard_normal((n_pred, n_feat, n_lon, n_lat))

        pred_index = list(range(n_pred))
        pred_samples = xr.DataArray(
            pred_data,
            dims=("prediction", "sample", "feature", "longitude", "latitude"),
            coords={
                "prediction": pred_index,
                "sample": np.arange(n_samples),
                "feature": ["2m_temperature", "10m_wind_speed"],
                "longitude": np.arange(n_lon),
                "latitude": np.arange(n_lat),
            },
        )
        y_obs = xr.DataArray(
            obs_data,
            dims=("prediction", "feature", "longitude", "latitude"),
            coords={
                "prediction": pred_index,
                "feature": ["2m_temperature", "10m_wind_speed"],
                "longitude": np.arange(n_lon),
                "latitude": np.arange(n_lat),
            },
        )
        return pred_samples, y_obs

    @pytest.mark.unit
    def test_returns_scalar(self, sample_xarray_data):
        """Energy score should return a single scalar float."""
        pred_samples, y_obs = sample_xarray_data
        score = _compute_copula_energy_score(pred_samples, y_obs, device="cpu")
        assert isinstance(score, float)

    @pytest.mark.unit
    def test_positive_score(self, sample_xarray_data):
        """Energy score should be positive for random data."""
        pred_samples, y_obs = sample_xarray_data
        score = _compute_copula_energy_score(pred_samples, y_obs, device="cpu")
        assert score > 0

    @pytest.mark.unit
    def test_perfect_predictions_low_score(self):
        """If all samples match the observation, energy score should be near zero."""
        n_pred, n_feat, n_lon, n_lat = 3, 2, 2, 2
        rng = np.random.default_rng(123)
        obs_data = rng.standard_normal((n_pred, n_feat, n_lon, n_lat))

        # All samples are the observation (perfect prediction)
        pred_data = np.broadcast_to(
            obs_data[:, None, :, :, :],
            (n_pred, 10, n_feat, n_lon, n_lat),
        ).copy()

        pred_index = list(range(n_pred))
        pred_samples = xr.DataArray(
            pred_data,
            dims=("prediction", "sample", "feature", "longitude", "latitude"),
            coords={
                "prediction": pred_index,
                "sample": np.arange(10),
                "feature": ["2m_temperature", "10m_wind_speed"],
                "longitude": np.arange(n_lon),
                "latitude": np.arange(n_lat),
            },
        )
        y_obs = xr.DataArray(
            obs_data,
            dims=("prediction", "feature", "longitude", "latitude"),
            coords={
                "prediction": pred_index,
                "feature": ["2m_temperature", "10m_wind_speed"],
                "longitude": np.arange(n_lon),
                "latitude": np.arange(n_lat),
            },
        )
        score = _compute_copula_energy_score(pred_samples, y_obs, device="cpu")
        assert score < 1.0  # loose bound — should be close to zero
