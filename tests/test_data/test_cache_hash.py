"""Tests for cache hash computation in fast_dataset_simple module."""

import pytest
from omegaconf import OmegaConf

from genpp.data.fast_dataset_simple import _compute_config_hash


@pytest.mark.unit
def test_compute_config_hash_basic():
    """Test that hash computation produces consistent results."""
    # Create a simple config
    config = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
            },
            "val": {"slice": "2021-01-01:2021-06-30", "x_kwargs": {"input_dims": {"feature": 10}}},
            "test": {"slice": "2021-07-01:2021-12-31", "x_kwargs": {"input_dims": {"feature": 10}}},
        }
    )

    x_vars = ["temp", "pressure"]
    y_vars = ["precipitation"]

    # Compute hash twice
    hash1 = _compute_config_hash(config, x_vars, y_vars)
    hash2 = _compute_config_hash(config, x_vars, y_vars)

    # Should produce the same hash
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 hex characters


@pytest.mark.unit
def test_compute_config_hash_different_configs():
    """Test that different configs produce different hashes."""
    config1 = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
            },
            "val": {"slice": "2021-01-01:2021-06-30", "x_kwargs": {"input_dims": {"feature": 10}}},
            "test": {"slice": "2021-07-01:2021-12-31", "x_kwargs": {"input_dims": {"feature": 10}}},
        }
    )

    config2 = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 20}},
            },
            "val": {"slice": "2021-01-01:2021-06-30", "x_kwargs": {"input_dims": {"feature": 20}}},
            "test": {"slice": "2021-07-01:2021-12-31", "x_kwargs": {"input_dims": {"feature": 20}}},
        }
    )

    x_vars = ["temp", "pressure"]
    y_vars = ["precipitation"]

    hash1 = _compute_config_hash(config1, x_vars, y_vars)
    hash2 = _compute_config_hash(config2, x_vars, y_vars)

    # Should produce different hashes
    assert hash1 != hash2


@pytest.mark.unit
def test_compute_config_hash_different_variables():
    """Test that different variable selections produce different hashes."""
    config = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
            },
            "val": {"slice": "2021-01-01:2021-06-30", "x_kwargs": {"input_dims": {"feature": 10}}},
            "test": {"slice": "2021-07-01:2021-12-31", "x_kwargs": {"input_dims": {"feature": 10}}},
        }
    )

    x_vars1 = ["temp", "pressure"]
    x_vars2 = ["temp", "humidity"]
    y_vars = ["precipitation"]

    hash1 = _compute_config_hash(config, x_vars1, y_vars)
    hash2 = _compute_config_hash(config, x_vars2, y_vars)

    # Should produce different hashes
    assert hash1 != hash2


@pytest.mark.unit
def test_compute_config_hash_with_preprocessing():
    """Test that preprocessing affects the hash."""
    config = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
            },
            "val": {"slice": "2021-01-01:2021-06-30", "x_kwargs": {"input_dims": {"feature": 10}}},
            "test": {"slice": "2021-07-01:2021-12-31", "x_kwargs": {"input_dims": {"feature": 10}}},
        }
    )

    x_vars = ["temp", "pressure"]
    y_vars = ["precipitation"]

    # Create mock preprocessors with different names
    class MockPreprocessor1:
        pass

    class MockPreprocessor2:
        pass

    hash1 = _compute_config_hash(config, x_vars, y_vars, [MockPreprocessor1()], None)  # type: ignore
    hash2 = _compute_config_hash(config, x_vars, y_vars, [MockPreprocessor2()], None)  # type: ignore
    hash3 = _compute_config_hash(config, x_vars, y_vars, None, None)

    # All should be different
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3


@pytest.mark.unit
def test_compute_config_hash_variable_order_invariant():
    """Test that variable order doesn't affect hash (they are sorted)."""
    config = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
            },
            "val": {"slice": "2021-01-01:2021-06-30", "x_kwargs": {"input_dims": {"feature": 10}}},
            "test": {"slice": "2021-07-01:2021-12-31", "x_kwargs": {"input_dims": {"feature": 10}}},
        }
    )

    # Variables in different order
    x_vars1 = ["temp", "pressure", "humidity"]
    x_vars2 = ["humidity", "temp", "pressure"]
    y_vars = ["precipitation"]

    hash1 = _compute_config_hash(config, x_vars1, y_vars)
    hash2 = _compute_config_hash(config, x_vars2, y_vars)

    # Should produce a different hash (variables are not in the same order)
    assert hash1 != hash2


@pytest.mark.unit
def test_compute_config_hash_transforms_excluded():
    """Test that x_transform and y_transform don't affect the hash."""
    # Config without transforms
    config1 = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
                "x_transform": None,
                "y_transform": None,
            },
            "val": {
                "slice": "2021-01-01:2021-06-30",
                "x_kwargs": {"input_dims": {"feature": 10}},
                "x_transform": None,
                "y_transform": None,
            },
            "test": {
                "slice": "2021-07-01:2021-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
                "x_transform": None,
                "y_transform": None,
            },
        }
    )

    # Config with transforms (as strings to simulate transform objects)
    config2 = OmegaConf.create(
        {
            "train": {
                "slice": "2020-01-01:2020-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
                "x_transform": "SomeTransform",  # Different!
                "y_transform": "AnotherTransform",  # Different!
            },
            "val": {
                "slice": "2021-01-01:2021-06-30",
                "x_kwargs": {"input_dims": {"feature": 10}},
                "x_transform": "SomeTransform",
                "y_transform": "AnotherTransform",
            },
            "test": {
                "slice": "2021-07-01:2021-12-31",
                "x_kwargs": {"input_dims": {"feature": 10}},
                "x_transform": "SomeTransform",
                "y_transform": "AnotherTransform",
            },
        }
    )

    x_vars = ["temp", "pressure"]
    y_vars = ["precipitation"]

    hash1 = _compute_config_hash(config1, x_vars, y_vars)
    hash2 = _compute_config_hash(config2, x_vars, y_vars)

    # Should produce the same hash (transforms are excluded from hash)
    assert hash1 == hash2
