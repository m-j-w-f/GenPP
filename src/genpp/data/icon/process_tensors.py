#!/usr/bin/env python
"""
Standalone script for processing ICON forecast and reanalysis data into tensors.

This script uses static methods from dataset.py to process tensors, enabling
parallel processing via NQSV job submissions.

Usage:
    For forecast tensors:
        JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py

    For reanalysis tensors:
        JOB_TYPE=rea YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py

Environment Variables:
    JOB_TYPE: Either 'fc' or 'rea' to specify which tensor type to process
    YEAR: Year in format YYYY (e.g., '2021')
    MONTH: Month in format MM (e.g., '01')
"""

import os
import sys
from pathlib import Path

# Import constants and methods from the package
from genpp.data.icon import DATA_DIR, VARS_GRID_28, VARS_REA
from genpp.data.icon.dataset import ForecastDataModule


def filter_paths_by_month(paths: list[Path], year: str, month: str) -> list[Path]:
    """Filter file paths to only include those from a specific year-month.

    Args:
        paths: List of file paths where filename contains date like YYYYMMDDHH
        year: String in format 'YYYY'
        month: String in format 'MM'

    Returns:
        Filtered list of paths
    """
    prefix = f"{year}{month}"
    filtered = []
    for path in paths:
        # Extract date from filename (format varies by file type)
        stem = path.stem
        # For ensmean/ensstd files: ens_YYYYMMDDHH_XXX.nc
        # For rea files: rea_YYYYMMDDHH.nc
        parts = stem.split("_")
        for part in parts:
            if len(part) >= 6 and part[:6] == prefix:
                filtered.append(path)
                break
    return filtered


def main():
    """Main entry point for the script."""
    # Get job type, year and month from environment variables
    job_type = os.environ.get("JOB_TYPE", "").lower()
    year = os.environ.get("YEAR", "")
    month = os.environ.get("MONTH", "")

    if job_type not in ["fc", "rea"]:
        print(f"Error: JOB_TYPE must be 'fc' or 'rea', got '{job_type}'")
        print("Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py")
        sys.exit(1)

    if not year or len(year) != 4 or not year.isdigit():
        print(f"Error: YEAR must be in format YYYY, got '{year}'")
        print("Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py")
        sys.exit(1)

    if not month or len(month) != 2 or not month.isdigit():
        print(f"Error: MONTH must be in format MM, got '{month}'")
        print("Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py")
        sys.exit(1)

    print(f"Processing {job_type.upper()} tensors for {year}-{month}")

    # Setup tensor directories
    fc_tensor_dir = DATA_DIR / "tensors" / "fc"
    meta_tensor_dir = DATA_DIR / "tensors" / "meta"
    rea_tensor_dir = DATA_DIR / "tensors" / "rea"

    if job_type == "fc":
        # Process forecast tensors
        ens_nc_paths = sorted(list((DATA_DIR / "ensmean").glob("*.nc")))
        if not ens_nc_paths:
            print(f"Error: No ensmean files found in {DATA_DIR / 'ensmean'}")
            sys.exit(1)

        # Filter to only this month
        ens_nc_paths = filter_paths_by_month(ens_nc_paths, year, month)
        print(f"Found {len(ens_nc_paths)} ensmean files for {year}-{month}")

        if ens_nc_paths:
            # Call the static method from ForecastDataModule
            ForecastDataModule._get_fc_tensors_static(
                ens_nc_paths,
                VARS_GRID_28,
                VARS_REA,
                fc_tensor_dir,
                meta_tensor_dir,
            )
            print(f"Completed processing {len(ens_nc_paths)} FC tensors for {year}-{month}")
        else:
            print(f"No files found for {year}-{month}")

    elif job_type == "rea":
        # Process reanalysis tensors
        rea_nc_paths = sorted(list((DATA_DIR / "rea").glob("*.nc")))
        if not rea_nc_paths:
            print(f"Error: No rea files found in {DATA_DIR / 'rea'}")
            sys.exit(1)

        # Filter to only this month
        rea_nc_paths = filter_paths_by_month(rea_nc_paths, year, month)
        print(f"Found {len(rea_nc_paths)} rea files for {year}-{month}")

        if rea_nc_paths:
            # Call the static method from ForecastDataModule
            ForecastDataModule._get_rea_tensors_static(
                rea_nc_paths,
                VARS_REA,
                rea_tensor_dir,
            )
            print(f"Completed processing {len(rea_nc_paths)} REA tensors for {year}-{month}")
        else:
            print(f"No files found for {year}-{month}")


if __name__ == "__main__":
    main()
