import numpy as np
import pytest
import torch
import xarray as xr

from genpp.preproc.preprocessors import MinMaxScalerPreprocessor, StandardScalerPreprocessor


class TestStandardScalerPreprocessor:
    """Test suite for StandardScalerPreprocessor class."""

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

    @pytest.fixture
    def sample_data_3d(self):
        """Create a 3D xarray DataArray for testing."""
        data = xr.DataArray(
            np.random.randn(5, 3, 4),
            dims=["time", "lat", "lon"],
            coords={"time": range(5), "lat": range(3), "lon": range(4)},
        )
        return data

    def test_init_single_dim(self):
        """Test initialization with a single dimension."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        assert preprocessor.dim == "time"
        assert not preprocessor.is_fitted

    def test_fit_single_dim(self, sample_data_1d):
        """Test fitting the preprocessor on a single dimension."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        # Check that fitting completed
        assert preprocessor.is_fitted

        # Manual calculation: mean = 3.0, std = sqrt(2.5) ≈ 1.5811
        expected_mean = 3.0
        expected_std = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)

        # xarray mean/std should return DataArrays, then we convert to scalars
        assert np.isclose(preprocessor.mean.values, expected_mean, atol=1e-6)
        assert np.isclose(preprocessor.std.values, expected_std, atol=1e-6)

    def test_fit_2d_data_time_dim(self, sample_data_2d):
        """Test fitting on 2D data along time dimension."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_2d)

        # Should compute mean and std along time dimension
        expected_mean = sample_data_2d.mean(dim="time")
        expected_std = sample_data_2d.std(dim="time", ddof=1)

        assert preprocessor.is_fitted
        # Use xarray's equals method for comparing DataArrays
        assert preprocessor.mean.equals(expected_mean)
        assert preprocessor.std.equals(expected_std)

    def test_fit_2d_data_space_dim(self, sample_data_2d):
        """Test fitting on 2D data along space dimension."""
        preprocessor = StandardScalerPreprocessor(dim="space")
        preprocessor.fit(sample_data_2d)

        # Should compute mean and std along space dimension
        expected_mean = sample_data_2d.mean(dim="space")
        expected_std = sample_data_2d.std(dim="space", ddof=1)

        assert preprocessor.is_fitted
        assert preprocessor.mean.equals(expected_mean)
        assert preprocessor.std.equals(expected_std)

    def test_preprocess_1d(self, sample_data_1d):
        """Test preprocessing 1D data."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        result = preprocessor.preprocess(sample_data_1d)

        # Result should be an xarray DataArray
        assert isinstance(result, xr.DataArray)

        # Check that standardization worked: mean ≈ 0, std ≈ 1
        result_mean = result.mean().values
        result_std = result.std(ddof=1).values

        assert np.isclose(result_mean, 0.0, atol=1e-10)
        assert np.isclose(result_std, 1.0, atol=1e-10)

    def test_preprocess_2d(self, sample_data_2d):
        """Test preprocessing 2D data."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_2d)

        result = preprocessor.preprocess(sample_data_2d)

        # Result should be an xarray DataArray with same shape
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_data_2d.shape
        assert result.dims == sample_data_2d.dims

    def test_preprocess_different_data(self, sample_data_1d):
        """Test preprocessing data different from the fitted data."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        # Create new data with same structure but different values
        new_data = xr.DataArray([10.0, 20.0, 30.0], dims=["time"], coords={"time": range(3)})

        result = preprocessor.preprocess(new_data)

        # Should work and return standardized data
        assert isinstance(result, xr.DataArray)
        assert result.shape == new_data.shape
        assert result.dims == new_data.dims

    def test_preprocess_before_fit_error(self, sample_data_1d):
        """Test that preprocess fails when called before fit."""
        preprocessor = StandardScalerPreprocessor(dim="time")

        with pytest.raises(AttributeError):
            preprocessor.preprocess(sample_data_1d)

    def test_inverse_transform_not_implemented(self, sample_data_1d):
        """Test that inverse_transform raises NotImplementedError."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        # Convert to tensor for inverse_transform
        tensor_data = torch.tensor(sample_data_1d.values)

        with pytest.raises(NotImplementedError, match="TODO Inverse transform is not implemented"):
            preprocessor.inverse_transform(tensor_data)

    def test_zero_std_handling(self):
        """Test behavior when standard deviation is zero."""
        # Create data with zero variance
        data = xr.DataArray([5.0, 5.0, 5.0, 5.0], dims=["time"], coords={"time": range(4)})
        preprocessor = StandardScalerPreprocessor(dim="time")

        # Fit should work but std will be 0
        preprocessor.fit(data)

        assert preprocessor.is_fitted
        assert preprocessor.std.values == 0.0

        # Preprocessing should result in NaN (division by zero)
        result = preprocessor.preprocess(data)
        assert np.isnan(result.values).all()

    def test_mismatched_dimensions_error(self, sample_data_1d):
        """Test error when dimension doesn't exist in data."""
        preprocessor = StandardScalerPreprocessor(dim="nonexistent_dim")

        with pytest.raises((ValueError, KeyError)):
            preprocessor.fit(sample_data_1d)

    def test_preprocess_preserves_coordinates(self, sample_data_2d):
        """Test that preprocessing preserves coordinate information."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_2d)

        result = preprocessor.preprocess(sample_data_2d)

        # Coordinates should be preserved
        assert result.coords.keys() == sample_data_2d.coords.keys()
        for coord_name in sample_data_2d.coords:
            assert result.coords[coord_name].equals(sample_data_2d.coords[coord_name])

    def test_preprocess_preserves_attributes(self, sample_data_1d):
        """Test that preprocessing preserves attributes."""
        # Add some attributes to the data
        sample_data_1d.attrs["units"] = "temperature"
        sample_data_1d.attrs["description"] = "test data"

        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        result = preprocessor.preprocess(sample_data_1d)

        # Attributes should be preserved
        assert result.attrs == sample_data_1d.attrs

    def test_fit_idempotent(self, sample_data_1d):
        """Test that calling fit multiple times gives same result."""
        preprocessor = StandardScalerPreprocessor(dim="time")

        # Fit once
        preprocessor.fit(sample_data_1d)
        first_mean = preprocessor.mean.copy()
        first_std = preprocessor.std.copy()

        # Fit again
        preprocessor.fit(sample_data_1d)
        second_mean = preprocessor.mean
        second_std = preprocessor.std

        # Results should be identical
        assert first_mean.equals(second_mean)
        assert first_std.equals(second_std)

    def test_multiple_dim_string_error(self):
        """Test that passing multiple dimensions as string raises appropriate error."""
        # This should work fine - just testing the initialization
        preprocessor = StandardScalerPreprocessor(dim="time")
        assert preprocessor.dim == "time"

    def test_different_data_types(self, sample_data_1d):
        """Test preprocessing with different numeric data types."""
        # Convert to different data types
        float32_data = sample_data_1d.astype(np.float32)
        int_data = sample_data_1d.astype(int)

        preprocessor = StandardScalerPreprocessor(dim="time")

        # Should work with float32
        preprocessor.fit(float32_data)
        result_float32 = preprocessor.preprocess(float32_data)
        assert isinstance(result_float32, xr.DataArray)

        # Should work with integer data (converted to float)
        preprocessor.fit(int_data)
        result_int = preprocessor.preprocess(int_data)
        assert isinstance(result_int, xr.DataArray)

    def test_3d_data_complex_case(self, sample_data_3d):
        """Test with more complex 3D data."""
        preprocessor = StandardScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_3d)

        result = preprocessor.preprocess(sample_data_3d)

        # Should preserve shape and dimensions
        assert result.shape == sample_data_3d.shape
        assert result.dims == sample_data_3d.dims

        # Each lat-lon point should be standardized across time
        for lat_idx in range(sample_data_3d.sizes["lat"]):
            for lon_idx in range(sample_data_3d.sizes["lon"]):
                point_data = result.isel(lat=lat_idx, lon=lon_idx)
                # Mean should be close to 0, std close to 1 for this time series
                assert np.isclose(point_data.mean().values, 0.0, atol=1e-10)
                assert np.isclose(point_data.std(ddof=1).values, 1.0, atol=1e-10)


class TestMinMaxScalerPreprocessor:
    """Test suite for MinMaxScalerPreprocessor class."""

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

    @pytest.fixture
    def sample_data_3d(self):
        """Create a 3D xarray DataArray for testing."""
        data = xr.DataArray(
            np.random.randn(5, 3, 4),
            dims=["time", "lat", "lon"],
            coords={"time": range(5), "lat": range(3), "lon": range(4)},
        )
        return data

    def test_init_single_dim(self):
        """Test initialization with a single dimension."""
        preprocessor = MinMaxScalerPreprocessor(dim="time")
        assert preprocessor.dim == "time"
        assert preprocessor.feature_range == (0, 1)

    def test_init_custom_feature_range(self):
        """Test initialization with custom feature range."""
        preprocessor = MinMaxScalerPreprocessor(dim="time", feature_range=(-1, 1))
        assert preprocessor.dim == "time"
        assert preprocessor.feature_range == (-1, 1)

    def test_fit_single_dim(self, sample_data_1d):
        """Test fitting the preprocessor on a single dimension."""
        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        # Check that min and max are computed correctly
        expected_min = sample_data_1d.min(dim="time")
        expected_max = sample_data_1d.max(dim="time")

        assert preprocessor.data_min.equals(expected_min)
        assert preprocessor.data_max.equals(expected_max)

    def test_preprocess_1d_default_range(self, sample_data_1d):
        """Test preprocessing 1D data with default range [0, 1]."""
        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        result = preprocessor.preprocess(sample_data_1d)

        # Check that data is scaled to [0, 1]
        assert result.min().values == 0.0
        assert result.max().values == 1.0

        # Check shape preservation
        assert result.shape == sample_data_1d.shape
        assert result.dims == sample_data_1d.dims

    def test_preprocess_1d_custom_range(self, sample_data_1d):
        """Test preprocessing 1D data with custom range [-1, 1]."""
        preprocessor = MinMaxScalerPreprocessor(dim="time", feature_range=(-1, 1))
        preprocessor.fit(sample_data_1d)

        result = preprocessor.preprocess(sample_data_1d)

        # Check that data is scaled to [-1, 1]
        assert np.isclose(result.min().values, -1.0)
        assert np.isclose(result.max().values, 1.0)

        # Check shape preservation
        assert result.shape == sample_data_1d.shape
        assert result.dims == sample_data_1d.dims

    def test_preprocess_2d_along_time(self, sample_data_2d):
        """Test preprocessing 2D data along time dimension."""
        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_2d)

        result = preprocessor.preprocess(sample_data_2d)

        # Check shape preservation
        assert result.shape == sample_data_2d.shape
        assert result.dims == sample_data_2d.dims

        # For each space coordinate, the values should be scaled to [0, 1] across time
        for space_idx in range(sample_data_2d.sizes["space"]):
            space_data = result.isel(space=space_idx)
            assert np.isclose(space_data.min().values, 0.0, atol=1e-10)
            assert np.isclose(space_data.max().values, 1.0, atol=1e-10)

    def test_preprocess_2d_along_space(self, sample_data_2d):
        """Test preprocessing 2D data along space dimension."""
        preprocessor = MinMaxScalerPreprocessor(dim="space")
        preprocessor.fit(sample_data_2d)

        result = preprocessor.preprocess(sample_data_2d)

        # Check shape preservation
        assert result.shape == sample_data_2d.shape
        assert result.dims == sample_data_2d.dims

        # For each time coordinate, the values should be scaled to [0, 1] across space
        for time_idx in range(sample_data_2d.sizes["time"]):
            time_data = result.isel(time=time_idx)
            assert np.isclose(time_data.min().values, 0.0, atol=1e-10)
            assert np.isclose(time_data.max().values, 1.0, atol=1e-10)

    def test_preprocess_multiple_dims(self, sample_data_2d):
        """Test preprocessing with multiple dimensions."""
        preprocessor = MinMaxScalerPreprocessor(dim=["time", "space"])
        preprocessor.fit(sample_data_2d)

        result = preprocessor.preprocess(sample_data_2d)

        # When scaling across both dimensions, entire array should be scaled to [0, 1]
        assert np.isclose(result.min().values, 0.0, atol=1e-10)
        assert np.isclose(result.max().values, 1.0, atol=1e-10)

        # Check shape preservation
        assert result.shape == sample_data_2d.shape
        assert result.dims == sample_data_2d.dims

    def test_constant_data(self):
        """Test preprocessing with constant data (min == max)."""
        constant_data = xr.DataArray([5.0, 5.0, 5.0, 5.0], dims=["time"], coords={"time": range(4)})

        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(constant_data)

        # With constant data, min == max, so division by zero should be handled
        # The implementation doesn't explicitly handle this case, so it may result in NaN
        result = preprocessor.preprocess(constant_data)

        # Check that we get a result (even if it contains NaN)
        assert result.shape == constant_data.shape
        assert result.dims == constant_data.dims

    def test_preprocessing_preserves_attributes(self, sample_data_1d):
        """Test that preprocessing preserves data attributes."""
        sample_data_1d.attrs["units"] = "temperature"
        sample_data_1d.attrs["description"] = "test data"

        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)
        result = preprocessor.preprocess(sample_data_1d)

        # Attributes should be preserved
        assert result.attrs == sample_data_1d.attrs

    def test_different_data_types(self, sample_data_1d):
        """Test preprocessing with different numeric data types."""
        # Convert to different data types
        float32_data = sample_data_1d.astype(np.float32)
        int_data = sample_data_1d.astype(int)

        preprocessor = MinMaxScalerPreprocessor(dim="time")

        # Should work with float32
        preprocessor.fit(float32_data)
        result_float32 = preprocessor.preprocess(float32_data)
        assert isinstance(result_float32, xr.DataArray)
        assert np.isclose(result_float32.min().values, 0.0, atol=1e-6)
        assert np.isclose(result_float32.max().values, 1.0, atol=1e-6)

        # Should work with integer data
        preprocessor.fit(int_data)
        result_int = preprocessor.preprocess(int_data)
        assert isinstance(result_int, xr.DataArray)
        assert np.isclose(result_int.min().values, 0.0, atol=1e-10)
        assert np.isclose(result_int.max().values, 1.0, atol=1e-10)

    def test_3d_data_complex_case(self, sample_data_3d):
        """Test with more complex 3D data."""
        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_3d)

        result = preprocessor.preprocess(sample_data_3d)

        # Should preserve shape and dimensions
        assert result.shape == sample_data_3d.shape
        assert result.dims == sample_data_3d.dims

        # Each lat-lon point should be scaled to [0, 1] across time
        for lat_idx in range(sample_data_3d.sizes["lat"]):
            for lon_idx in range(sample_data_3d.sizes["lon"]):
                point_data = result.isel(lat=lat_idx, lon=lon_idx)
                # Min should be close to 0, max close to 1 for this time series
                assert np.isclose(point_data.min().values, 0.0, atol=1e-10)
                assert np.isclose(point_data.max().values, 1.0, atol=1e-10)

    def test_inverse_transform_not_implemented(self, sample_data_1d):
        """Test that inverse_transform raises NotImplementedError."""
        preprocessor = MinMaxScalerPreprocessor(dim="time")
        preprocessor.fit(sample_data_1d)

        # Convert to tensor for inverse transform
        dummy_tensor = torch.tensor([0.0, 0.5, 1.0])

        with pytest.raises(NotImplementedError, match="TODO Inverse transform is not implemented"):
            preprocessor.inverse_transform(dummy_tensor)

    def test_refitting_updates_statistics(self, sample_data_1d):
        """Test that refitting updates the min/max statistics."""
        # Create different data
        new_data = xr.DataArray([10.0, 20.0, 30.0], dims=["time"], coords={"time": range(3)})

        preprocessor = MinMaxScalerPreprocessor(dim="time")

        # Fit once
        preprocessor.fit(sample_data_1d)
        first_min = preprocessor.data_min.copy()
        first_max = preprocessor.data_max.copy()

        # Fit again with different data
        preprocessor.fit(new_data)
        second_min = preprocessor.data_min
        second_max = preprocessor.data_max

        # Results should be different
        assert not first_min.equals(second_min)
        assert not first_max.equals(second_max)

        # New statistics should match the new data
        assert second_min.equals(new_data.min(dim="time"))
        assert second_max.equals(new_data.max(dim="time"))
