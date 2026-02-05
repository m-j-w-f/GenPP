"""Tests for ForecastDataModule separate batch size functionality."""

import pytest

from genpp.data.icon.dataset import ForecastDataModule


class TestForecastDataModuleBatchSizes:
    """Test suite for ForecastDataModule batch size parameters."""

    @pytest.mark.unit
    def test_default_batch_size_for_all(self):
        """Test that batch_size is used for all dataloaders when specific sizes not provided."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            batch_size=64,
        )

        assert dm.batch_size == 64
        assert dm.train_batch_size == 64
        assert dm.val_batch_size == 64
        assert dm.test_batch_size == 64

    @pytest.mark.unit
    def test_separate_batch_sizes(self):
        """Test that separate batch sizes can be specified for train, val, and test."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            batch_size=32,
            train_batch_size=64,
            val_batch_size=128,
            test_batch_size=256,
        )

        assert dm.batch_size == 32
        assert dm.train_batch_size == 64
        assert dm.val_batch_size == 128
        assert dm.test_batch_size == 256

    @pytest.mark.unit
    def test_partial_batch_sizes(self):
        """Test that unspecified batch sizes fall back to default batch_size."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            batch_size=32,
            train_batch_size=64,  # Only train specified
        )

        assert dm.batch_size == 32
        assert dm.train_batch_size == 64
        assert dm.val_batch_size == 32  # Falls back to batch_size
        assert dm.test_batch_size == 32  # Falls back to batch_size

    @pytest.mark.unit
    def test_none_batch_sizes_use_default(self):
        """Test that None batch sizes use the default batch_size."""
        dm = ForecastDataModule(
            x_select_variables=["var1", "var2"],
            y_select_variables=["var1"],
            batch_size=48,
            train_batch_size=None,
            val_batch_size=None,
            test_batch_size=None,
        )

        assert dm.batch_size == 48
        assert dm.train_batch_size == 48
        assert dm.val_batch_size == 48
        assert dm.test_batch_size == 48
