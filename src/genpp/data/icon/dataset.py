# %%
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from genpp import BASE_DIR
from genpp.data import MetadataVars
from genpp.data.icon import (
    AXIS_ORDER,
    LEVELS_TO_FLATTEN,
    VARS_GRID_28,
    VARS_TO_DROP,
)
from genpp.data.utils import flatten_levels

# %%
DATA_DIR = BASE_DIR / "data" / "icon" / "data"


# %%
# Note to self: opening a .nc file from the work dir takes a bit of time
# In the prepare data step we should probably create multiple tensor files (one per day for example)
# probably should copy the data to local
# - Local NVMe raid /raid with 8TB on smc and 24TB on dgx.
# - Local SSD scratch /scratch with 54GB on smc and 1,3TB on dgx.

# %%
# Check out some files
# test_idx = 208
# rea_nc_paths = sorted(list((DATA_DIR / "rea").glob("*.nc")))
# rea_nc_paths = rea_nc_paths[test_idx : test_idx + 1]

# ens_nc_paths = sorted(list((DATA_DIR / "ensmean").glob("*.nc")))
# ens_nc_paths = ens_nc_paths[:1] + ens_nc_paths[-1:]

# %% [markdown]
# ## Changes after 2022112300 06-UTC
#
# TODO investigate how the vars change after the switch date 2022112300.
# See [ICON-EPS Model Changes PDF (November 2022)](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon_eps/pdf_2022/pdf_icon_eps_23_11_2022.pdf?__blob=publicationFile&v=2)
# The changes will become effective for the 06-UTC forecast run
#
# ### How to deal with this?
#
# This change also adds vertical levels (for the newer forecasts some levels are in the data which are not in the old files)
# But all vars that are in the old files are also in the new ones which is good :)
# However we might still utilize only the new forecasts with more prediction_timedeltas to compensate as the dataset has no switch of grids then.
#
# ### Note on REA files
#
# Only after 01.01.2019 are the files complete and carry the wind speed variable!
#
# ## Other Info
#
# ### Some vars are 0
#
# These are snow vars in the summer
#
# ### Time Coordinate in forecasts
#
# The forecasts only carry the axis time which is the valid_time of the forecast.
# The file name is the init_time.
# The dimedelta is only in the name.
#
# ### Saving as Pytorch Tensor
#
# Each Tensor is about 20MB in size times 11896 files -> ~ 240 GB
# Each dgx node has 2.5 Tib in ram -> easy
#
#


# %%
# Add meta features
def _add_sincos_doy(da: xr.DataArray) -> xr.DataArray:
    doy = da.time.dt.dayofyear
    sin_time = np.sin(doy * 2 * np.pi / 365).astype(np.float32)
    cos_time = np.sin(doy * 2 * np.pi / 365).astype(np.float32)
    transformed_time = xr.concat([sin_time, cos_time], dim="feature")
    transformed_time["feature"] = [
        MetadataVars.SIN_PREDICTION_TIME.value,
        MetadataVars.COS_PREDICTION_TIME.value,
    ]
    transformed_time.expand_dims(
        {
            "x": da.x,
            "y": da.y,
        }
    )
    return transformed_time


def _add_xy(da: xr.DataArray) -> xr.DataArray:
    # normalize x per-axis (min-max) and expand to 2D feature map
    x = da.x
    x_mean = float(x.mean())
    x_std = float(x.std())
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    x_grid = x_norm.expand_dims({"y": da.y, "feature": [MetadataVars.LONGITUDE.value]})
    x_grid = x_grid.transpose("feature", "x", "y")

    # normalize y per-axis (min-max) and expand to 2D feature map
    y = da.y
    y_mean = float(y.mean())
    y_std = float(y.std())
    y_norm = ((y - y_mean) / y_std).astype(np.float32)
    y_grid = y_norm.expand_dims({"x": da.x, "feature": [MetadataVars.LATITUDE.value]})
    y_grid = y_grid.transpose("feature", "x", "y")

    return xr.concat([x_grid, y_grid], dim="feature")


def get_meatdata_features(da: xr.DataArray) -> xr.DataArray:
    sincos_doy = _add_sincos_doy(da)
    xy_grid = _add_xy(da)
    return xr.concat([sincos_doy, xy_grid], dim="feature").transpose(*AXIS_ORDER)


# %%
class ForecastDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path, Path, np.datetime64, np.timedelta64]],
        norm_stats: dict[str, torch.Tensor],
        feature_metadata: dict[str, float],
        normalize_type: str = "zscore",
    ) -> None:
        """Initialize the ForecastDataset.

        Args:
            samples (list[tuple[Path, Path, Path, np.datetime64, np.timedelta64]]): List of tuples containing
                (fc_path, meta_path, rea_path, init_date, leadtime).
            norm_stats (dict[str, torch.Tensor]): Dictionary with normalization statistics
                ('fc_mean', 'fc_std', 'fc_min', 'fc_max', 'rea_mean', 'rea_std', 'rea_min', 'rea_max').
            feature_metadata (dict[str, torch.Tensor]): Dictionary containing feature categorization info
                (predicted_var_indices, auxiliary_var_indices, meta_var_indices).
            normalize_type (str): Type of normalization, either 'zscore' or 'minmax'.
        """
        self.samples = samples
        self.norm_stats = norm_stats
        self.feature_metadata = feature_metadata
        self.normalize_type = normalize_type

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index, loading and normalizing data.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict[str, Any]: Dictionary containing:
                - x: dict with predicted_vars, auxiliary_vars, meta_vars, pixel_idx
                - y: target tensor
                - timedelta: normalized prediction timedelta
        """
        fc_path, meta_path, rea_path, _, leadtime = self.samples[idx]

        # Load tensors
        fc_dict = torch.load(fc_path)  # dict with 'predicted_vars' and 'auxiliary_vars'
        meta = torch.load(meta_path)  # shape [c, x, y]
        rea = torch.load(rea_path)  # shape [c, x, y]

        # Extract predicted and auxiliary variables
        predicted_vars = fc_dict["predicted_vars"]  # shape [c0, x, y]
        auxiliary_vars = fc_dict["auxiliary_vars"]  # shape [c1, x, y]

        # Normalize predicted variables
        if self.normalize_type == "zscore":
            predicted_vars = (predicted_vars - self.norm_stats["pred_mean"]) / self.norm_stats[
                "pred_std"
            ]
        elif self.normalize_type == "minmax":
            predicted_vars = (predicted_vars - self.norm_stats["pred_min"]) / (
                self.norm_stats["pred_max"] - self.norm_stats["pred_min"]
            )

        # Normalize auxiliary variables
        if self.normalize_type == "zscore":
            auxiliary_vars = (auxiliary_vars - self.norm_stats["aux_mean"]) / self.norm_stats[
                "aux_std"
            ]
        elif self.normalize_type == "minmax":
            auxiliary_vars = (auxiliary_vars - self.norm_stats["aux_min"]) / (
                self.norm_stats["aux_max"] - self.norm_stats["aux_min"]
            )

        # Normalize REA (reanalysis target)
        if self.normalize_type == "zscore":
            rea = (rea - self.norm_stats["rea_mean"]) / self.norm_stats["rea_std"]
        elif self.normalize_type == "minmax":
            rea = (rea - self.norm_stats["rea_min"]) / (
                self.norm_stats["rea_max"] - self.norm_stats["rea_min"]
            )

        # Convert timedelta to hours and normalize
        timedelta_hours = leadtime / np.timedelta64(1, "h")
        max_timedelta = self.feature_metadata.get("max_timedelta", 120.0)
        timedelta_normalized = torch.tensor(timedelta_hours / max_timedelta, dtype=torch.float32)

        return {
            "x": {
                "predicted_vars": predicted_vars,
                "auxiliary_vars": auxiliary_vars,
                "meta_vars": meta,
                "pixel_idx": None,
            },
            "y": rea,
            "timedelta": timedelta_normalized,
        }


# %%
class ForecastDataModule(L.LightningDataModule):
    def __init__(
        self,
        x_select_variables: list[str],
        y_select_variables: list[str],
        data_dir: Path | str = DATA_DIR,
        cache_dir: Path | str | None = None,
        batch_size: int = 32,
        normalize_type: str = "zscore",
        num_workers: int = 4,
    ) -> None:
        """Initialize the ForecastDataModule.

        Args:
            x_select_variables (list[str]): List of variable names to select from FC data.
            y_select_variables (list[str]): List of variable names to select from REA data.
            data_dir (str): Path to the data directory containing ensmean, ensstd, rea folders.
            cache_dir (str): Dir where the data will be copied to and read from. This is used
                             due to the DATA_DIR being very slow to read from.
            batch_size (int): Batch size for DataLoaders.
            normalize_type (str): Type of normalization ('zscore' or 'minmax').
            num_workers (int): Number of workers for DataLoaders.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size
        self.normalize_type = normalize_type  # 'zscore' or 'minmax'
        self.num_workers = num_workers
        self.x_select_variables = x_select_variables
        self.y_select_variables = y_select_variables
        self.x_select_variables_wo_y = [
            var for var in self.x_select_variables if var not in self.y_select_variables
        ]
        self.norm_stats: dict[str, torch.Tensor] | None = None
        self.feature_metadata = None

        self.fc_tensor_dir = DATA_DIR / "tensors" / "fc"
        self.meta_tensor_dir = DATA_DIR / "tensors" / "meta"
        self.rea_tensor_dir = DATA_DIR / "tensors" / "rea"
        self.norm_stats_file = DATA_DIR / "tensors" / "norm_stats.pt"

    def prepare_data(self) -> None:
        """Prepare data by computing normalization statistics from the test set.

        This method collects samples, splits them, and computes mean, std, min, max
        from the test set for normalization.
        """
        ens_nc_paths = sorted(list((DATA_DIR / "ensmean").glob("*.nc")))
        self._get_fc_tensors(ens_nc_paths)

        rea_nc_paths = sorted(list((DATA_DIR / "rea").glob("*.nc")))
        self._get_rea_tensors(rea_nc_paths)

        if not self.norm_stats_file.exists():
            self._compute_norm_stats()

        if self.feature_metadata is None:
            print("Computing feature metadata")
            self._compute_feature_metadata()

    def _compute_feature_metadata(self) -> None:
        """Compute feature metadata for storing max timedelta."""
        fc_paths = list(self.fc_tensor_dir.glob("fc_*.pt"))

        if not fc_paths:
            raise RuntimeError("No FC tensor files found. Run prepare_data() first.")

        # Find max timedelta from filenames
        max_timedelta = 0.0
        for fc_path in fc_paths:
            parts = fc_path.stem.split("_")
            if len(parts) >= 3:
                leadtime = int(parts[2])
                max_timedelta = max(max_timedelta, float(leadtime))

        self.feature_metadata = {
            "max_timedelta": max_timedelta,
        }

    def _compute_tensor_stats(
        self, tensor_paths: list[Path], tensor_key: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean, std, min, max statistics for tensors in a single pass.

        Args:
            tensor_paths: List of paths to tensor files
            tensor_key: Optional key to extract from dict tensors (e.g., 'predicted_vars', 'auxiliary_vars')

        Returns:
            Tuple of (mean, std, min, max) tensors with shape [c, 1, 1]
        """
        tensor_sum = None
        tensor_sum_sq = None
        tensor_min = None  # type: ignore
        tensor_max = None  # type: ignore
        tensor_count = 0

        for tensor_path in tqdm(tensor_paths):
            # Load tensor (either dict or tensor directly)
            loaded = torch.load(tensor_path)

            # Extract specific key if this is a dict
            if tensor_key is not None:
                if not isinstance(loaded, dict):
                    raise ValueError(f"Expected dict at {tensor_path}, got {type(loaded)}")
                tensor = loaded[tensor_key]
            else:
                tensor = loaded

            # Compute spatial statistics: [c, x, y] -> [c, 1, 1]
            spatial_sum = tensor.sum(dim=[-2, -1], keepdim=True)
            spatial_sum_sq = (tensor**2).sum(dim=[-2, -1], keepdim=True)

            if tensor_sum is None:
                tensor_sum = spatial_sum
                tensor_sum_sq = spatial_sum_sq
                tensor_min = tensor.amin(dim=[-2, -1], keepdim=True)
                tensor_max = tensor.amax(dim=[-2, -1], keepdim=True)
            else:
                tensor_sum += spatial_sum
                tensor_sum_sq += spatial_sum_sq
                tensor_min: torch.Tensor = torch.minimum(
                    tensor_min, tensor.amin(dim=[-2, -1], keepdim=True)
                )
                tensor_max: torch.Tensor = torch.maximum(
                    tensor_max, tensor.amax(dim=[-2, -1], keepdim=True)
                )

            tensor_count += tensor.shape[-2] * tensor.shape[-1]

        if tensor_sum is None or tensor_sum_sq is None:
            raise RuntimeError("No tensors were processed")

        mean = tensor_sum / tensor_count
        var = (tensor_sum_sq / tensor_count) - (mean**2)
        std = torch.sqrt(var)

        return mean, std, tensor_min, tensor_max

    def _compute_norm_stats(self) -> None:
        """Compute normalization statistics (mean, std, min, max) for FC and REA tensors.

        Computes statistics across all spatial dimensions in a single pass through the data
        to minimize I/O overhead. Results are stored in self.norm_stats and saved to disk.

        The computed statistics have shapes:
        - Predicted var statistics: [c0, 1, 1]
        - Auxiliary var statistics: [c1, 1, 1]
        - REA statistics: [c, 1, 1]
        """
        self.norm_stats = {}

        # Compute statistics for predicted variables in FC tensors
        fc_tensor_paths = list(self.fc_tensor_dir.glob("fc_*.pt"))
        if fc_tensor_paths:
            print("Computing predicted_vars stats")
            pred_mean, pred_std, pred_min, pred_max = self._compute_tensor_stats(
                fc_tensor_paths, tensor_key="predicted_vars"
            )
            self.norm_stats.update(
                {
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "pred_min": pred_min,
                    "pred_max": pred_max,
                }
            )

            # Compute statistics for auxiliary variables in FC tensors
            print("Computing aux_vars stats")
            aux_mean, aux_std, aux_min, aux_max = self._compute_tensor_stats(
                fc_tensor_paths, tensor_key="auxiliary_vars"
            )
            self.norm_stats.update(
                {
                    "aux_mean": aux_mean,
                    "aux_std": aux_std,
                    "aux_min": aux_min,
                    "aux_max": aux_max,
                }
            )

        # Compute statistics for REA tensors
        rea_tensor_paths = list(self.rea_tensor_dir.glob("rea_*.pt"))
        if rea_tensor_paths:
            print("Computing rea stats")
            rea_mean, rea_std, rea_min, rea_max = self._compute_tensor_stats(
                rea_tensor_paths, tensor_key=None
            )
            self.norm_stats.update(
                {
                    "rea_mean": rea_mean,
                    "rea_std": rea_std,
                    "rea_min": rea_min,
                    "rea_max": rea_max,
                }
            )

        torch.save(self.norm_stats, self.norm_stats_file)

    def setup(self, stage: str) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage (str): Stage of setup (e.g., 'fit', 'test').
        """
        # Load normalization statistics if not already loaded
        if self.norm_stats is None:
            if self.norm_stats_file.exists():
                self.norm_stats = torch.load(self.norm_stats_file)
            else:
                raise ValueError(
                    "norm_stats is None and norm_stats file does not exist. "
                    "Run prepare_data() first."
                )

        # Load feature metadata if not already loaded
        if self.feature_metadata is None:
            raise ValueError(
                "feature_metadata is None and feature_metadata file does not exist. "
                "Run prepare_data() first."
            )

        # Collect and sort samples by init_date
        all_samples = self._collect_samples()
        all_samples.sort(key=lambda x: x[3])  # Sort by init_date (index 3)

        # Split by init_date year
        train_samples, val_samples, test_samples = [], [], []
        for sample in all_samples:
            init_date = sample[3]  # np.datetime64
            year = init_date.astype("datetime64[Y]").astype(int) + 1970

            if year <= 2021:
                train_samples.append(sample)
            elif year == 2022:
                val_samples.append(sample)
            else:
                test_samples.append(sample)

        self.train_dataset = ForecastDataset(
            train_samples,
            self.norm_stats,  # type: ignore
            self.feature_metadata,
            self.normalize_type,
        )
        self.val_dataset = ForecastDataset(
            val_samples,
            self.norm_stats,  # type: ignore
            self.feature_metadata,
            self.normalize_type,
        )
        self.test_dataset = ForecastDataset(
            test_samples,
            self.norm_stats,  # type: ignore
            self.feature_metadata,
            self.normalize_type,
        )

    def _get_fc_tensors(self, ens_nc_paths: list[Path]) -> None:
        """Build and store forecast tensors from ensemble NetCDF paths.

        Args:
            ens_nc_paths (list[Path]): Paths to ensmean NetCDF files to process.

        Returns:
            None: Writes forecast and metadata tensors to disk.
        """
        # Skip entries with already materialized tensors
        filtered_paths: list[Path] = []
        for ens_path in ens_nc_paths:
            time_leadtime = "_".join(ens_path.stem.split("_")[1:])
            fc_path = self.fc_tensor_dir / f"fc_{time_leadtime}.pt"
            meta_path = self.meta_tensor_dir / f"meta_{time_leadtime}.pt"
            if fc_path.exists() and meta_path.exists():
                continue
            filtered_paths.append(ens_path)
        ens_nc_paths = filtered_paths

        # Build matching ensstd paths for remaining inputs
        std_nc_paths = [Path(str(p).replace("ensmean", "ensstd")) for p in ens_nc_paths]
        # Process mean/std pairs together
        for paths in tqdm(zip(ens_nc_paths, std_nc_paths), desc="Generating FC Tensors"):
            datasets = []
            time_leadtime = "_".join(paths[0].stem.split("_")[1:])
            for path in paths:
                print(f"Processing path {path}")
                ds = xr.open_dataset(path).drop_vars(VARS_TO_DROP)
                for level in LEVELS_TO_FLATTEN:
                    try:
                        ds = flatten_levels(ds, level)
                    except KeyError:
                        pass
                da = ds[VARS_GRID_28].to_dataarray("feature").squeeze().transpose(*AXIS_ORDER)
                datasets.append(da)
            da_stacked = xr.concat(datasets, dim="aggregation")
            da_stacked.coords["aggregation"] = ["mean", "std"]

            # Select the predicted vars (y_select_vars) and use only aggr dim mean
            predicted_vars = da_stacked.sel(aggregation="mean", feature=self.y_select_variables)
            predicted_vars = torch.from_numpy(predicted_vars.values)

            # Select the auxiliary vars (all vars in x_select_vars) and kick out the
            aux_vars_mean = da_stacked.sel(aggregation="mean", feature=self.x_select_variables_wo_y)
            aux_vars_std = da_stacked.sel(aggregation="std", feature=self.x_select_variables)
            aux_vars = xr.concat([aux_vars_mean, aux_vars_std], dim="feature")
            aux_vars = torch.from_numpy(aux_vars.values)

            fc_tensors = {
                "predicted_vars": predicted_vars,  # [c0,x,y]
                "auxiliary_vars": aux_vars,  # [c1,x,y]
            }
            fc_path = self.fc_tensor_dir / f"fc_{time_leadtime}.pt"
            torch.save(fc_tensors, fc_path)

            meta = get_meatdata_features(da_stacked)

            fc_path = self.fc_tensor_dir / f"fc_{time_leadtime}.pt"
            fc_tensor = torch.from_numpy(da_stacked.values)
            # Fc tensors have shape [agg, c, x, y]
            torch.save(fc_tensor, fc_path)

            meta_path = self.meta_tensor_dir / f"meta_{time_leadtime}.pt"
            meta_tensor = torch.from_numpy(meta.values)
            # Meta tensors have shape [c, x, y]
            torch.save(meta_tensor, meta_path)

    def _get_rea_tensors(self, rea_nc_paths: list[Path]) -> None:
        """Build and store reanalysis tensors from NetCDF paths, skipping existing outputs.

        Args:
            rea_nc_paths (list[Path]): Paths to reanalysis NetCDF files to process.

        Returns:
            None: Writes reanalysis tensors to disk.
        """
        # Skip entries with already materialized tensors
        filtered_paths: list[Path] = []
        for rea_path in rea_nc_paths:
            date = rea_path.stem.split("_")[-1]
            tens_path = self.rea_tensor_dir / f"rea_{date}.pt"
            if tens_path.exists():
                continue
            filtered_paths.append(rea_path)
        rea_nc_paths = filtered_paths

        for rea_path in tqdm(rea_nc_paths, desc="Generating REA Tensors"):
            date = rea_path.stem.split("_")[-1]
            rea = xr.open_dataset(rea_path)
            rea = rea.drop_vars("rotated_pole")
            for dim in ["height", "height_2"]:
                rea = flatten_levels(rea, level_dim=dim)
            rea = (
                rea.to_dataarray("feature")
                .sel(feature=self.y_select_variables)
                .transpose(..., *AXIS_ORDER)
            )
            tens_path = self.rea_tensor_dir / f"rea_{date}.pt"
            tens = torch.from_numpy(rea.values).squeeze()
            # Rea has shape [c, x, y]
            torch.save(tens, tens_path)

    def _collect_samples(self) -> list[tuple[Path, Path, Path, np.datetime64, np.timedelta64]]:
        """Collect valid samples from the data directories.

        Parses filenames in the tensors file to generate  tuples of (fc_path, meta_path, rea_path, init_date, leadtime).

        Returns:
            list[tuple[Path, Path, Path, np.datetime64, np.timedelta64]]: List of tuples (fc_path, meta_path, rea_path, init_date, leadtime).
        """
        samples: list[tuple[Path, Path, Path, np.datetime64, np.timedelta64]] = []

        # Get all FC tensor files
        for fc_path in self.fc_tensor_dir.glob("fc_*.pt"):
            # Parse filename: fc_YYYYMMDDHH_LT.pt
            parts = fc_path.stem.split("_")
            if len(parts) < 3:
                continue

            date_str = parts[1]  # YYYYMMDDHH
            leadtime_str = parts[2]  # leadtime in hours

            try:
                # Parse init_date
                init_date = np.datetime64(
                    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{date_str[8:10]}:00:00"
                )

                # Parse leadtime
                leadtime = np.timedelta64(int(leadtime_str), "h")

                # Calculate target date
                target_date = init_date + leadtime
                target_date_str = str(target_date)[:10].replace("-", "")  # YYYYMMDD

                # Build corresponding meta and rea paths
                meta_path = self.meta_tensor_dir / f"meta_{date_str}_{leadtime_str}.pt"
                rea_path = self.rea_tensor_dir / f"rea_{target_date_str}.pt"

                # Check if all files exist
                if fc_path.exists() and meta_path.exists() and rea_path.exists():
                    samples.append((fc_path, meta_path, rea_path, init_date, leadtime))

            except (ValueError, IndexError):
                # Skip malformed filenames
                continue

        return samples

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader.

        Returns:
            DataLoader: DataLoader for test data.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
