import numpy as np
import pytest
import torch
import xarray as xr

from genpp.preproc.utils import StandardScaler


class TestStandardScaler:
    """Test suite for StandardScaler class."""

    @pytest.fixture
    def sample_data_1d(self):
        """Create a simple 1D xarray DataArray for testing."""
        data = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=["time"], coords={"time": range(5)})
        return data

    @pytest.fixture
    def sample_data_2d(self):
        """Create a 2D xarray DataArray for testing."""
        data = xr.DataArray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dims=["time", "space"],
            coords={"time": range(3), "space": range(3)},
        )
        return data

    def test_init_single_dim(self):
        """Test initialization with a single dimension."""
        scaler = StandardScaler(dim="time")
        assert scaler.dim == "time"

    def test_fit_single_dim(self, sample_data_1d):
        """Test fitting the scaler on a single dimension."""
        scaler = StandardScaler(dim="time")
        scaler.fit(sample_data_1d)

        # Manual calculation: mean = 3.0, std = sqrt(2.5) ≈ 1.5811
        expected_mean = torch.tensor(3.0, dtype=torch.float32)
        expected_scale = torch.tensor(
            np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1), dtype=torch.float32
        )

        assert torch.allclose(scaler.mean, expected_mean, atol=1e-6)
        assert torch.allclose(scaler.scale, expected_scale, atol=1e-6)

    def test_transform(self, sample_data_1d):
        """Test transform method."""
        scaler = StandardScaler(dim="time")
        scaler.fit(sample_data_1d)
        result = scaler.transform(sample_data_1d)

        # After standardization, mean should be ~0, std should be ~1
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(result.std(unbiased=True), torch.tensor(1.0), atol=1e-6)

    def test_fit_transform(self, sample_data_1d):
        """Test fit_transform method."""
        scaler = StandardScaler(dim="time")
        result = scaler.fit_transform(sample_data_1d)

        # Should be equivalent to calling fit then transform
        scaler2 = StandardScaler(dim="time")
        scaler2.fit(sample_data_1d)
        expected = scaler2.transform(sample_data_1d)

        assert torch.allclose(result, expected)

    def test_call_method(self, sample_data_1d):
        """Test that __call__ works the same as transform."""
        scaler = StandardScaler(dim="time")
        scaler.fit(sample_data_1d)

        result1 = scaler.transform(sample_data_1d)
        result2 = scaler(sample_data_1d)

        assert torch.allclose(result1, result2)

    def test_inverse_transform(self, sample_data_1d):
        """Test inverse transformation."""
        scaler = StandardScaler(dim="time")
        transformed = scaler.fit_transform(sample_data_1d)
        reconstructed = scaler.inverse_transform(transformed)

        # Should recover original data
        original_tensor = torch.tensor(sample_data_1d.values, dtype=torch.float32)
        assert torch.allclose(reconstructed, original_tensor, atol=1e-6)

    def test_transform_different_data(self, sample_data_1d):
        """Test transforming data different from the fitted data."""
        scaler = StandardScaler(dim="time")
        scaler.fit(sample_data_1d)

        # Create new data with same structure
        new_data = xr.DataArray([10.0, 20.0, 30.0], dims=["time"], coords={"time": range(3)})

        result = scaler.transform(new_data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_partial_dimensions(self, sample_data_2d):
        """Test fitting on only some dimensions."""
        scaler = StandardScaler(dim="time")
        scaler.fit(sample_data_2d)

        # Should compute mean and std along time dimension only
        expected_mean = torch.tensor(sample_data_2d.mean(dim="time").values, dtype=torch.float32)
        expected_scale = torch.tensor(
            sample_data_2d.std(dim="time", ddof=1).values, dtype=torch.float32
        )

        assert torch.allclose(scaler.mean, expected_mean, atol=1e-6)
        assert torch.allclose(scaler.scale, expected_scale, atol=1e-6)

    def test_zero_std_handling(self):
        """Test behavior when standard deviation is zero."""
        # Create data with zero variance
        data = xr.DataArray([5.0, 5.0, 5.0, 5.0], dims=["time"])
        scaler = StandardScaler(dim="time")

        # Fit should emit a warning about zero standard deviation
        with pytest.warns(RuntimeWarning, match="Standard deviation is zero"):
            scaler.fit(data)

        # Standard deviation should be 0
        assert scaler.scale.item() == 0.0

        # Transform should result in NaN or inf (division by zero)
        result = scaler.transform(data)
        assert torch.isinf(result).any() or torch.isnan(result).any()

    def test_mismatched_dimensions_error(self, sample_data_1d):
        """Test error when dimension doesn't exist in data."""
        scaler = StandardScaler(dim="nonexistent_dim")

        with pytest.raises((ValueError, KeyError)):
            scaler.fit(sample_data_1d)

    def test_transform_before_fit_error(self, sample_data_1d):
        """Test that transform fails when called before fit."""
        scaler = StandardScaler(dim="time")

        with pytest.raises(AttributeError):
            scaler.transform(sample_data_1d)

    def test_inverse_transform_before_fit_error(self):
        """Test that inverse_transform fails when called before fit."""
        scaler = StandardScaler(dim="time")
        data = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(AttributeError):
            scaler.inverse_transform(data)
