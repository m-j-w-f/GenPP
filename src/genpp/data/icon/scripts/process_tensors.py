#!/usr/bin/env python
"""
Standalone script for processing ICON forecast and reanalysis data into tensors.

This script uses static methods from dataset.py to process tensors, enabling
parallel processing via NQSV job submissions.

The new version creates unified tensor files where all features are concatenated
into a single tensor, with a metadata pickle file that maps feature names to indices.

Usage:
    For forecast tensors:
        JOB_TYPE=fc YEAR=2021 MONTH=01 DAY=15 pixi run python process_tensors.py

    For reanalysis tensors:
        JOB_TYPE=rea YEAR=2021 MONTH=01 DAY=15 pixi run python process_tensors.py

Environment Variables:
    JOB_TYPE: Either 'fc' or 'rea' to specify which tensor type to process
    YEAR: Year in format YYYY (e.g., '2021')
    MONTH: Month in format MM (e.g., '01')
    DAY: Day in format DD (e.g., '15')
"""

import os
import sys
from pathlib import Path

# Import constants and methods from the package
from genpp.data.icon import DATA_DIR, VARS_GRID_28, VARS_REA
from genpp.data.icon.dataset import ForecastDataModule


def filter_paths_by_day(paths: list[Path], year: str, month: str, day: str) -> list[Path]:
    """Filter file paths to only include those from a specific year-month-day.

    Args:
        paths: List of file paths where filename contains date like YYYYMMDDHH
        year: String in format 'YYYY'
        month: String in format 'MM'
        day: String in format 'DD'

    Returns:
        Filtered list of paths
    """
    prefix = f"{year}{month}{day}"
    filtered = []
    for path in paths:
        # Extract date from filename (format varies by file type)
        stem = path.stem
        # For ensmean/ensstd files: ens_YYYYMMDDHH_XXX.nc
        # For rea files: rea_YYYYMMDDHH.nc
        parts = stem.split("_")
        for part in parts:
            if len(part) >= 8 and part[:8] == prefix:
                filtered.append(path)
                break
    return filtered


def main():
    """Main entry point for the script."""
    # Get job type, year, month and day from environment variables
    job_type = os.environ.get("JOB_TYPE", "").lower()
    year = os.environ.get("YEAR", "")
    month = os.environ.get("MONTH", "")
    day = os.environ.get("DAY", "")

    usage = "Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 DAY=15 pixi run -e nb python process_tensors.py"

    if job_type not in ["fc", "rea"]:
        print(f"Error: JOB_TYPE must be 'fc' or 'rea', got '{job_type}'")
        print(usage)
        sys.exit(1)

    if not year or len(year) != 4 or not year.isdigit():
        print(f"Error: YEAR must be in format YYYY, got '{year}'")
        print(usage)
        sys.exit(1)

    if not month or len(month) != 2 or not month.isdigit():
        print(f"Error: MONTH must be in format MM, got '{month}'")
        print(usage)
        sys.exit(1)

    if not day or len(day) != 2 or not day.isdigit():
        print(f"Error: DAY must be in format DD, got '{day}'")
        print(usage)
        sys.exit(1)

    print(f"Processing {job_type.upper()} tensors for {year}-{month}-{day}")

    # Setup tensor directories
    fc_tensor_dir = DATA_DIR / "tensors" / "fc"
    rea_tensor_dir = DATA_DIR / "tensors" / "rea"

    if job_type == "fc":
        # Process forecast tensors
        ens_nc_paths = sorted(list((DATA_DIR / "ensmean").glob("*.nc")))
        if not ens_nc_paths:
            print(f"Error: No ensmean files found in {DATA_DIR / 'ensmean'}")
            sys.exit(1)

        # Filter to only this day
        ens_nc_paths = filter_paths_by_day(ens_nc_paths, year, month, day)
        print(f"Found {len(ens_nc_paths)} ensmean files for {year}-{month}-{day}")

        if ens_nc_paths:
            # Call the static method from ForecastDataModule
            feature_metadata = ForecastDataModule._get_fc_tensors_static(
                ens_nc_paths,
                VARS_GRID_28,
                VARS_REA,
                fc_tensor_dir,
            )
            print(f"Completed processing {len(ens_nc_paths)} FC tensors for {year}-{month}-{day}")
            if feature_metadata:
                print(f"Feature metadata: {list(feature_metadata.keys())}")
        else:
            print(f"No files found for {year}-{month}-{day}")

    elif job_type == "rea":
        # Process reanalysis tensors
        rea_nc_paths = sorted(list((DATA_DIR / "rea").glob("*.nc")))
        if not rea_nc_paths:
            print(f"Error: No rea files found in {DATA_DIR / 'rea'}")
            sys.exit(1)

        # Filter to only this day
        rea_nc_paths = filter_paths_by_day(rea_nc_paths, year, month, day)
        print(f"Found {len(rea_nc_paths)} rea files for {year}-{month}-{day}")

        if rea_nc_paths:
            # Call the static method from ForecastDataModule
            feature_metadata = ForecastDataModule._get_rea_tensors_static(
                rea_nc_paths,
                VARS_REA,
                rea_tensor_dir,
            )
            print(f"Completed processing {len(rea_nc_paths)} REA tensors for {year}-{month}-{day}")
            if feature_metadata:
                print(f"Feature metadata: {list(feature_metadata.keys())}")
        else:
            print(f"No files found for {year}-{month}-{day}")


if __name__ == "__main__":
    main()
