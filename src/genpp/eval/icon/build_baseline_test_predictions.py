"""Build consolidated baseline test predictions from raw ICON ensemble files.

Writes output incrementally to avoid keeping the full 72GB prediction tensor in memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    class _NoOpTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self._iterable = iterable
            self.total = total

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _NoOpTqdm(iterable=iterable, **kwargs)


from genpp.data.icon import DATA_DIR


@dataclass(frozen=True)
class ReferenceMeta:
    init_date_decoded: np.ndarray
    leadtime_decoded: np.ndarray
    variable: np.ndarray
    spatial_x_name: str
    spatial_y_name: str
    spatial_x: np.ndarray
    spatial_y: np.ndarray
    time_encoded: np.ndarray
    init_encoded: np.ndarray
    lead_encoded: np.ndarray
    time_attrs: dict[str, object]
    init_attrs: dict[str, object]
    lead_attrs: dict[str, object]


def load_ensemble_tensor(path: Path) -> np.ndarray:
    """Load one raw ensemble NetCDF to ndarray with shape [sample, variable, x, y]."""
    with xr.open_dataset(path) as ds:
        t2m = ds["T_2M"].squeeze(drop=True)
        vmax = ds["VMAX_10M"].squeeze(drop=True)
        stacked = xr.concat(
            [
                t2m.assign_coords(variable="T_2M").expand_dims("variable"),
                vmax.assign_coords(variable="VMAX_10M").expand_dims("variable"),
            ],
            dim="variable",
        )
        stacked = stacked.transpose("time", "variable", "x", "y")
        return stacked.values.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build consolidated baseline predictions matching a reference schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-test-predictions",
        type=Path,
        default=Path("outputs/ENGRESSION/2026-03-23_18-58-40/test_predictions.nc"),
        help="Reference NetCDF providing canonical time/init_date/leadtime/variable order.",
    )
    parser.add_argument(
        "--ens-dir",
        type=Path,
        default=DATA_DIR / "ens",
        help="Directory containing ens_YYYYMMDDHH_lead.nc files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/BASELINE/test_predictions.nc"),
        help="Target consolidated baseline NetCDF path.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing ensemble files instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    return parser.parse_args()


def _find_time_like(ds: xr.Dataset, primary: str, fallback: str) -> xr.DataArray:
    if primary in ds.coords:
        return ds.coords[primary]
    if primary in ds.variables:
        return ds[primary]
    if fallback in ds.coords:
        return ds.coords[fallback]
    if fallback in ds.variables:
        return ds[fallback]
    raise ValueError(f"Missing {primary}/{fallback} in reference dataset")


def _extract_reference_meta(reference_path: Path) -> ReferenceMeta:
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    with xr.open_dataset(reference_path, decode_times=True) as ds_dec:
        init_da = _find_time_like(ds_dec, "init_date", "init_time")
        lead_da = _find_time_like(ds_dec, "leadtime", "lead_time")

        init_decoded = np.asarray(init_da.values)
        lead_decoded = np.asarray(lead_da.values)

        if "variable" not in ds_dec.coords:
            raise ValueError("Reference dataset is missing required variable coordinate")
        variable_vals = np.asarray(ds_dec.coords["variable"].values)

        if "rlon" in ds_dec.coords and "rlat" in ds_dec.coords:
            spatial_x_name = "rlon"
            spatial_y_name = "rlat"
            spatial_x = np.asarray(ds_dec.coords["rlon"].values)
            spatial_y = np.asarray(ds_dec.coords["rlat"].values)
        elif "x" in ds_dec.coords and "y" in ds_dec.coords:
            spatial_x_name = "x"
            spatial_y_name = "y"
            spatial_x = np.asarray(ds_dec.coords["x"].values)
            spatial_y = np.asarray(ds_dec.coords["y"].values)
        else:
            raise ValueError("Reference dataset must expose either rlon/rlat or x/y coordinates")

    with xr.open_dataset(reference_path, decode_times=False) as ds_raw:
        time_da = ds_raw["time"]
        init_raw_da = _find_time_like(ds_raw, "init_date", "init_time")
        lead_raw_da = _find_time_like(ds_raw, "leadtime", "lead_time")

        time_encoded = np.asarray(time_da.values)
        init_encoded = np.asarray(init_raw_da.values)
        lead_encoded = np.asarray(lead_raw_da.values)

        time_attrs = dict(time_da.attrs)
        init_attrs = dict(init_raw_da.attrs)
        lead_attrs = dict(lead_raw_da.attrs)

    n = len(time_encoded)
    if not (n == len(init_encoded) == len(lead_encoded) == len(init_decoded) == len(lead_decoded)):
        raise ValueError("Reference time/init/lead arrays have inconsistent lengths")

    return ReferenceMeta(
        init_date_decoded=init_decoded,
        leadtime_decoded=lead_decoded,
        variable=variable_vals,
        spatial_x_name=spatial_x_name,
        spatial_y_name=spatial_y_name,
        spatial_x=spatial_x,
        spatial_y=spatial_y,
        time_encoded=time_encoded,
        init_encoded=init_encoded,
        lead_encoded=lead_encoded,
        time_attrs=time_attrs,
        init_attrs=init_attrs,
        lead_attrs=lead_attrs,
    )


def _leadtime_hours(value: object) -> int:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return int(arr.astype("timedelta64[h]").astype(np.int64))
    td = pd.to_timedelta(value)  # type: ignore
    return int(td.total_seconds() // 3600)


def _init_str(value: object) -> str:
    ts = pd.Timestamp(value).to_pydatetime()  # type: ignore
    return ts.strftime("%Y%m%d%H")


def _resolve_entries(
    meta: ReferenceMeta, ens_dir: Path, allow_missing: bool
) -> list[tuple[int, Path]]:
    entries: list[tuple[int, Path]] = []
    missing: list[Path] = []

    total = len(meta.time_encoded)
    for i in tqdm(
        range(total),
        desc="Resolving ensemble files",
        unit="file",
    ):
        init_str = _init_str(meta.init_date_decoded[i])
        lead_h = _leadtime_hours(meta.leadtime_decoded[i])
        ens_path = ens_dir / f"ens_{init_str}_{lead_h}.nc"

        if not ens_path.exists():
            missing.append(ens_path)
            if not allow_missing:
                raise FileNotFoundError(f"Missing required ensemble file: {ens_path}")
            continue

        entries.append((i, ens_path))

    if not entries:
        raise RuntimeError("No matching ensemble files found for reference keys")

    if missing:
        print(f"Skipped {len(missing)} missing entries (allow-missing enabled).", flush=True)

    return entries


def _copy_attrs(nc_var, attrs: dict[str, object]) -> None:
    for key, value in attrs.items():
        if key in {"_FillValue", "dtype"}:
            continue
        setattr(nc_var, key, value)


def _initialize_output(
    meta: ReferenceMeta, kept_indices: np.ndarray, first_pred: np.ndarray, output_path: Path
) -> tuple[Dataset, object]:
    _, n_samples, n_vars, n_x, n_y = (1, *first_pred.shape)

    if len(meta.variable) != n_vars:
        raise ValueError(f"Variable count mismatch: pred={n_vars} ref={len(meta.variable)}")
    if len(meta.spatial_x) != n_x or len(meta.spatial_y) != n_y:
        raise ValueError(
            "Spatial coordinate length mismatch: "
            f"pred=({n_x},{n_y}) ref=({len(meta.spatial_x)},{len(meta.spatial_y)})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nc = Dataset(output_path, "w", format="NETCDF4")

    n_time = len(kept_indices)
    nc.createDimension("time", n_time)
    nc.createDimension("sample", n_samples)
    nc.createDimension("variable", n_vars)
    nc.createDimension(meta.spatial_x_name, n_x)
    nc.createDimension(meta.spatial_y_name, n_y)

    time_var = nc.createVariable("time", "i8", ("time",))
    time_var[:] = meta.time_encoded[kept_indices].astype(np.int64)
    _copy_attrs(time_var, meta.time_attrs)

    sample_var = nc.createVariable("sample", "i8", ("sample",))
    sample_var[:] = np.arange(n_samples, dtype=np.int64)

    variable_var = nc.createVariable("variable", str, ("variable",))
    variable_var[:] = meta.variable.astype(str)

    x_var = nc.createVariable(meta.spatial_x_name, "f8", (meta.spatial_x_name,))
    x_var[:] = meta.spatial_x.astype(np.float64)

    y_var = nc.createVariable(meta.spatial_y_name, "f8", (meta.spatial_y_name,))
    y_var[:] = meta.spatial_y.astype(np.float64)

    init_var = nc.createVariable("init_date", "i8", ("time",))
    init_var[:] = meta.init_encoded[kept_indices].astype(np.int64)
    _copy_attrs(init_var, meta.init_attrs)

    lead_var = nc.createVariable("leadtime", "i8", ("time",))
    lead_var[:] = meta.lead_encoded[kept_indices].astype(np.int64)
    _copy_attrs(lead_var, meta.lead_attrs)

    pred_var = nc.createVariable(
        "prediction",
        "f4",
        ("time", "sample", "variable", meta.spatial_x_name, meta.spatial_y_name),
        zlib=True,
        complevel=3,
        chunksizes=(1, n_samples, n_vars, n_x, n_y),
    )
    pred_var.setncattr("coordinates", "init_date leadtime")

    return nc, pred_var


def build_to_netcdf(
    meta: ReferenceMeta, ens_dir: Path, output_path: Path, allow_missing: bool
) -> None:
    entries = _resolve_entries(meta=meta, ens_dir=ens_dir, allow_missing=allow_missing)
    kept_indices = np.asarray([idx for idx, _ in entries], dtype=np.int64)

    first_idx, first_path = entries[0]
    first_pred = load_ensemble_tensor(first_path)

    nc, pred_var = _initialize_output(
        meta=meta,
        kept_indices=kept_indices,
        first_pred=first_pred,
        output_path=output_path,
    )

    total = len(entries)
    try:
        with tqdm(total=total, desc="Writing predictions", unit="file") as pbar:
            pred_var[0, :, :, :, :] = first_pred  # type: ignore
            pbar.update(1)

            for out_i, (_, ens_path) in enumerate(entries[1:], start=1):
                pred = load_ensemble_tensor(ens_path)
                pred_var[out_i, :, :, :, :] = pred  # type: ignore
                pbar.update(1)
    finally:
        nc.close()


def main() -> None:
    args = parse_args()

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_path}. Use --overwrite to replace it."
        )

    meta = _extract_reference_meta(args.reference_test_predictions)
    build_to_netcdf(
        meta=meta,
        ens_dir=args.ens_dir,
        output_path=args.output_path,
        allow_missing=args.allow_missing,
    )

    print(f"Wrote consolidated baseline predictions: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
