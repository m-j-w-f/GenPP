"""Tests for launch_permutation_importance_drn helper functions."""

import csv
import tempfile
from pathlib import Path

import pytest

from genpp.eval.launch_permutation_importance_drn import FIELDNAMES, _merge_results


class TestMergeResults:
    """Tests for _merge_results."""

    @pytest.fixture
    def tmp_dir(self):
        """Create a temporary directory for test CSV files."""
        d = tempfile.mkdtemp()
        yield Path(d)
        import shutil

        shutil.rmtree(d, ignore_errors=True)

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    @pytest.mark.unit
    def test_merge_two_channels(self, tmp_dir):
        """Results from two channels are merged and sorted by importance descending."""
        ch0 = tmp_dir / "channel_0.csv"
        ch1 = tmp_dir / "channel_1.csv"

        self._write_csv(
            ch0,
            [
                {
                    "channel_index": 0,
                    "channel_name": "temp_mean",
                    "category": "all_var_mean",
                    "baseline_es": 1.0,
                    "permuted_es": 1.5,
                    "importance": 0.5,
                    "importance_std": 0.01,
                }
            ],
        )
        self._write_csv(
            ch1,
            [
                {
                    "channel_index": 1,
                    "channel_name": "wind_mean",
                    "category": "all_var_mean",
                    "baseline_es": 1.0,
                    "permuted_es": 2.0,
                    "importance": 1.0,
                    "importance_std": 0.02,
                }
            ],
        )

        output = tmp_dir / "merged.csv"
        rows = _merge_results({0: ch0, 1: ch1}, output)

        assert len(rows) == 2
        # Sorted by importance descending — channel 1 first
        assert rows[0]["channel_name"] == "wind_mean"
        assert rows[1]["channel_name"] == "temp_mean"

        # Verify output file exists and has correct content
        assert output.exists()
        with open(output, newline="") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2

    @pytest.mark.unit
    def test_merge_missing_channel(self, tmp_dir, capsys):
        """Missing channel file produces a warning but doesn't crash."""
        ch0 = tmp_dir / "channel_0.csv"
        ch1 = tmp_dir / "channel_1.csv"  # this one won't exist

        self._write_csv(
            ch0,
            [
                {
                    "channel_index": 0,
                    "channel_name": "temp_mean",
                    "category": "all_var_mean",
                    "baseline_es": 1.0,
                    "permuted_es": 1.5,
                    "importance": 0.5,
                    "importance_std": 0.01,
                }
            ],
        )

        output = tmp_dir / "merged.csv"
        rows = _merge_results({0: ch0, 1: ch1}, output)

        assert len(rows) == 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    @pytest.mark.unit
    def test_merge_empty(self, tmp_dir):
        """Empty input produces empty output."""
        output = tmp_dir / "merged.csv"
        rows = _merge_results({}, output)
        assert rows == []
        assert output.exists()
