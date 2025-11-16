import numpy as np
import pandas as pd
import pytest
import xarray as xr

from genpp.data.utils import flatten_levels, get_time_intersection


class TestFlattenLevels:
    """Test suite for flatten_levels function."""

    @pytest.fixture
    def sample_dataset_with_levels(self):
        """Create a sample dataset with level dimension."""
        # Create data with multiple features and levels
        data_3d = np.random.randn(3, 4, 5)  # time, lat, lon
        data_4d = np.random.randn(3, 2, 4, 5)  # time, level, lat, lon

        ds = xr.Dataset(
            {
                "temperature": (["time", "level", "lat", "lon"], data_4d),
                "pressure": (["time", "level", "lat", "lon"], data_4d * 2),
                "surface_var": (["time", "lat", "lon"], data_3d),  # No level dimension
            },
            coords={
                "time": pd.date_range("2023-01-01", periods=3),
                "level": [850, 500],
                "lat": np.linspace(-45, 45, 4),
                "lon": np.linspace(-90, 90, 5),
            },
        )
        return ds

    @pytest.fixture
    def sample_dataset_no_levels(self):
        """Create a sample dataset without level dimension."""
        data_3d = np.random.randn(3, 4, 5)  # time, lat, lon

        ds = xr.Dataset(
            {
                "temperature": (["time", "lat", "lon"], data_3d),
                "pressure": (["time", "lat", "lon"], data_3d * 2),
            },
            coords={
                "time": pd.date_range("2023-01-01", periods=3),
                "lat": np.linspace(-45, 45, 4),
                "lon": np.linspace(-90, 90, 5),
            },
        )
        return ds

    @pytest.fixture
    def sample_dataset_custom_level_dim(self):
        """Create a sample dataset with custom level dimension name."""
        data_4d = np.random.randn(3, 3, 4, 5)  # time, height, lat, lon

        ds = xr.Dataset(
            {
                "temperature": (["time", "height", "lat", "lon"], data_4d),
            },
            coords={
                "time": pd.date_range("2023-01-01", periods=3),
                "height": [100, 200, 300],
                "lat": np.linspace(-45, 45, 4),
                "lon": np.linspace(-90, 90, 5),
            },
        )
        return ds

    @pytest.mark.unit
    def test_flatten_levels_basic(self, sample_dataset_with_levels):
        """Test basic functionality of flatten_levels."""
        result = flatten_levels(sample_dataset_with_levels)

        # Should return a DataArray
        assert isinstance(result, xr.Dataset)

        # Should have flattened features for each level
        expected_vars = [
            "temperature+level_850",
            "temperature+level_500",
            "pressure+level_850",
            "pressure+level_500",
            "surface_var",
        ]

        # Check that feature names are correct
        var_names = list(result.data_vars)
        for expected_var in expected_vars:
            assert expected_var in var_names

    @pytest.mark.unit
    def test_flatten_levels_preserves_coordinates(self, sample_dataset_with_levels):
        """Test that non-level coordinates are preserved."""
        result = flatten_levels(sample_dataset_with_levels)

        # Should preserve all coordinates except level
        expected_coords = ["time", "lat", "lon"]
        assert set(result.coords.keys()) == set(expected_coords)

        # Check coordinate values are preserved
        np.testing.assert_array_equal(result.time.values, sample_dataset_with_levels.time.values)
        np.testing.assert_array_equal(result.lat.values, sample_dataset_with_levels.lat.values)
        np.testing.assert_array_equal(result.lon.values, sample_dataset_with_levels.lon.values)

    @pytest.mark.unit
    def test_flatten_levels_custom_level_dim(self, sample_dataset_custom_level_dim):
        """Test flatten_levels with custom level dimension name."""
        result = flatten_levels(sample_dataset_custom_level_dim, level_dim="height")

        # Should have flattened features for each height level
        expected_vars = [
            "temperature+height_100",
            "temperature+height_200",
            "temperature+height_300",
        ]
        var_names = list(result.data_vars)

        assert len(var_names) == 3
        for expected_var in expected_vars:
            assert expected_var in var_names

    @pytest.mark.unit
    def test_flatten_levels_no_level_dimension(self, sample_dataset_no_levels):
        """Test flatten_levels with dataset that has no level dimension."""
        with pytest.raises(KeyError):
            flatten_levels(sample_dataset_no_levels)

    @pytest.mark.unit
    def test_flatten_levels_data_integrity(self, sample_dataset_with_levels):
        """Test that data values are preserved during flattening."""
        original_ds = sample_dataset_with_levels
        result = flatten_levels(original_ds)

        # Check that flattened temperature data matches original
        temp_850 = result["temperature+level_850"]
        original_temp_850 = original_ds.temperature.sel(level=850)

        np.testing.assert_array_equal(temp_850.values, original_temp_850.values)

        # Check surface feature is unchanged
        surface_var = result["surface_var"]
        original_surface = original_ds.surface_var

        np.testing.assert_array_equal(surface_var.values, original_surface.values)

    @pytest.mark.unit
    def test_flatten_levels_mixed_variables(self, sample_dataset_with_levels):
        """Test that features with and without levels are handled correctly."""
        result = flatten_levels(sample_dataset_with_levels)

        # features with levels should be split
        var_names = list(result.data_vars)
        temp_vars = [var for var in var_names if "temperature+level_" in var]
        assert len(temp_vars) == 2  # Two levels

        # features without levels should remain single
        surface_vars = [var for var in var_names if var == "surface_var"]
        assert len(surface_vars) == 1

    @pytest.mark.unit
    def test_flatten_levels_empty_dataset(self):
        """Test behavior with empty dataset."""
        empty_ds = xr.Dataset()
        with pytest.raises(KeyError):
            flatten_levels(empty_ds)

    @pytest.mark.unit
    def test_flatten_levels_single_level(self):
        """Test with dataset having only one level."""
        data_4d = np.random.randn(2, 1, 3, 3)  # time, level, lat, lon

        ds = xr.Dataset(
            {
                "temperature": (["time", "level", "lat", "lon"], data_4d),
            },
            coords={
                "time": pd.date_range("2023-01-01", periods=2),
                "level": [850],
                "lat": [0, 1, 2],
                "lon": [0, 1, 2],
            },
        )

        result = flatten_levels(ds)
        var_names = list(result.data_vars)

        assert "temperature+level_850" in var_names
        assert len(var_names) == 1


@pytest.mark.skip(
    reason="TestGetTimeIntersection tests are deprecated along with get_time_intersection function"
)
class TestGetTimeIntersection:
    """Test suite for get_time_intersection function."""

    @pytest.fixture
    def dataset1_prediction_time(self):
        """Create dataset with prediction_time coordinate."""
        times = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        data = np.random.randn(len(times), 3, 3)

        ds = xr.Dataset(
            {"temperature": (["prediction_time", "lat", "lon"], data)},
            coords={"prediction_time": times, "lat": [0, 1, 2], "lon": [0, 1, 2]},
        )
        return ds

    @pytest.fixture
    def dataset2_time(self):
        """Create dataset with time coordinate."""
        times = pd.date_range("2023-01-05", "2023-01-15", freq="D")
        data = np.random.randn(len(times), 3, 3)

        ds = xr.Dataset(
            {"observations": (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": [0, 1, 2], "lon": [0, 1, 2]},
        )
        return ds

    @pytest.fixture
    def dataset3_custom_time_dim(self):
        """Create dataset with custom time dimension name."""
        times = pd.date_range("2023-01-03", "2023-01-12", freq="D")
        data = np.random.randn(len(times), 3, 3)

        ds = xr.Dataset(
            {"data": (["forecast_time", "lat", "lon"], data)},
            coords={"forecast_time": times, "lat": [0, 1, 2], "lon": [0, 1, 2]},
        )
        return ds

    @pytest.mark.unit
    def test_get_time_intersection_basic(self, dataset1_prediction_time, dataset2_time):
        """Test basic time intersection functionality."""
        result = get_time_intersection(dataset1_prediction_time, dataset2_time)

        # Should return pandas Index
        assert isinstance(result, pd.Index)

        # Should contain intersection of times
        # dataset1: 2023-01-01 to 2023-01-10
        # dataset2: 2023-01-05 to 2023-01-15
        # intersection: 2023-01-05 to 2023-01-10
        expected_times = pd.date_range("2023-01-05", "2023-01-10", freq="D")

        assert len(result) == len(expected_times)
        pd.testing.assert_index_equal(result, expected_times)

    @pytest.mark.unit
    def test_get_time_intersection_custom_dims(
        self, dataset1_prediction_time, dataset3_custom_time_dim
    ):
        """Test time intersection with custom dimension names."""
        result = get_time_intersection(
            dataset1_prediction_time,
            dataset3_custom_time_dim,
            time_dim1="prediction_time",
            time_dim2="forecast_time",
        )

        # dataset1: 2023-01-01 to 2023-01-10
        # dataset3: 2023-01-03 to 2023-01-12
        # intersection: 2023-01-03 to 2023-01-10
        expected_times = pd.date_range("2023-01-03", "2023-01-10", freq="D")

        assert len(result) == len(expected_times)
        pd.testing.assert_index_equal(result, expected_times)

    @pytest.mark.unit
    def test_get_time_intersection_no_overlap(self):
        """Test time intersection with no overlap."""
        # Create datasets with non-overlapping times
        ds1 = xr.Dataset(
            {"data": (["prediction_time", "lat"], np.random.randn(3, 2))},
            coords={"prediction_time": pd.date_range("2023-01-01", periods=3), "lat": [0, 1]},
        )

        ds2 = xr.Dataset(
            {"data": (["time", "lat"], np.random.randn(3, 2))},
            coords={"time": pd.date_range("2023-02-01", periods=3), "lat": [0, 1]},
        )

        result = get_time_intersection(ds1, ds2)

        # Should return empty index
        assert len(result) == 0
        assert isinstance(result, pd.Index)

    @pytest.mark.unit
    def test_get_time_intersection_identical_times(self):
        """Test time intersection with identical time coordinates."""
        times = pd.date_range("2023-01-01", periods=5)

        ds1 = xr.Dataset(
            {"data1": (["prediction_time", "lat"], np.random.randn(5, 2))},
            coords={"prediction_time": times, "lat": [0, 1]},
        )

        ds2 = xr.Dataset(
            {"data2": (["time", "lat"], np.random.randn(5, 2))},
            coords={"time": times, "lat": [0, 1]},
        )

        result = get_time_intersection(ds1, ds2)

        # Should return all times
        assert len(result) == 5
        pd.testing.assert_index_equal(result, times)

    @pytest.mark.unit
    def test_get_time_intersection_partial_overlap(self):
        """Test time intersection with partial overlap and different frequencies."""
        # Dataset 1: daily from Jan 1-10
        ds1 = xr.Dataset(
            {"data1": (["prediction_time", "lat"], np.random.randn(10, 2))},
            coords={
                "prediction_time": pd.date_range("2023-01-01", periods=10, freq="D"),
                "lat": [0, 1],
            },
        )

        # Dataset 2: every other day from Jan 5-15
        ds2_times = pd.date_range("2023-01-05", "2023-01-15", freq="2D")
        ds2 = xr.Dataset(
            {"data2": (["time", "lat"], np.random.randn(len(ds2_times), 2))},
            coords={"time": ds2_times, "lat": [0, 1]},
        )

        result = get_time_intersection(ds1, ds2)

        # Should only include dates that exist in both
        # ds1 has daily dates 1-10, ds2 has dates 5, 7, 9, 11, 13, 15
        # intersection should be 5, 7, 9
        expected_dates = pd.to_datetime(["2023-01-05", "2023-01-07", "2023-01-09"])

        assert len(result) == 3
        pd.testing.assert_index_equal(result.sort_values(), expected_dates)

    @pytest.mark.unit
    def test_get_time_intersection_error_missing_dimension(self, dataset1_prediction_time):
        """Test error handling when dimension doesn't exist."""
        with pytest.raises(KeyError):
            get_time_intersection(
                dataset1_prediction_time,
                dataset1_prediction_time,
                time_dim1="nonexistent_dim",
                time_dim2="prediction_time",
            )

    @pytest.mark.unit
    def test_get_time_intersection_different_time_types(self):
        """Test time intersection with different time types/formats."""
        # Dataset with datetime64
        ds1 = xr.Dataset(
            {"data1": (["prediction_time", "lat"], np.random.randn(3, 2))},
            coords={"prediction_time": pd.date_range("2023-01-01", periods=3), "lat": [0, 1]},
        )

        # Dataset with cftime (if available) or different datetime format
        # For simplicity, using string dates that get converted
        times_str = ["2023-01-02", "2023-01-03", "2023-01-04"]
        ds2 = xr.Dataset(
            {"data2": (["time", "lat"], np.random.randn(3, 2))},
            coords={"time": pd.to_datetime(times_str), "lat": [0, 1]},
        )

        result = get_time_intersection(ds1, ds2)

        # Should find intersection despite different creation methods
        expected = pd.to_datetime(["2023-01-02", "2023-01-03"])
        assert len(result) == 2
        pd.testing.assert_index_equal(result.sort_values(), expected)
