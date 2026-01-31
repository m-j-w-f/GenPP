# %%
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any
from collections.abc import Callable

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from genpp import BASE_DIR
from genpp.data.icon import (
    AXIS_ORDER,
    LEVELS_TO_FLATTEN,
    VARS_GRID_28,
    VARS_TO_DROP,
)
from genpp.data.utils import MetadataVars, flatten_levels

# %%
DATA_DIR = BASE_DIR / "data" / "icon" / "data"

# Setup logger
logger = logging.getLogger(__name__)


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
    cos_time = np.cos(doy * 2 * np.pi / 365).astype(np.float32)
    transformed_time = xr.concat([sin_time, cos_time], dim="feature")
    transformed_time["feature"] = [
        MetadataVars.SIN_PREDICTION_TIME.value,
        MetadataVars.COS_PREDICTION_TIME.value,
    ]
    transformed_time = transformed_time.expand_dims(
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


def get_metadata_features(da: xr.DataArray) -> xr.DataArray:
    """Get metadata features including day-of-year and coordinates."""
    sincos_doy = _add_sincos_doy(da)
    xy_grid = _add_xy(da)
    return xr.concat([sincos_doy, xy_grid], dim="feature", coords="minimal").transpose(*AXIS_ORDER)


# %%
class ForecastDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path, np.datetime64, np.timedelta64]],
        norm_stats: dict[str, torch.Tensor],
        feature_metadata: dict[str, Any],
        normalize_type: str = "zscore",
        x_transform: Callable | None = None,
        y_transform: Callable | None = None,
    ) -> None:
        """Initialize the ForecastDataset.

        Args:
            samples (list[tuple[Path, Path, np.datetime64, np.timedelta64]]): List of tuples containing
                (fc_path, rea_path, init_date, leadtime). Meta features are embedded in fc_path.
            norm_stats (dict[str, torch.Tensor]): Dictionary with normalization statistics
                ('all_mean', 'all_std', 'all_min', 'all_max', 'aux_mean', 'aux_std', 'aux_min', 'aux_max',
                'rea_mean', 'rea_std', 'rea_min', 'rea_max').
            feature_metadata (dict[str, Any]): Dictionary containing feature categorization info
                (predicted_var_indices, all_var_indices, meta_var_indices).
            normalize_type (str): Type of normalization, either 'zscore' or 'minmax'.
            x_transform (Callable | None): Optional transform to apply to input features (predicted_vars,
                all_vars, meta_vars) after normalization. Can be a function or nn.Module.
            y_transform (Callable | None): Optional transform to apply to target (rea) after normalization.
                Can be a function or nn.Module.
        """
        self.samples = samples
        self.norm_stats = norm_stats
        self.feature_metadata = feature_metadata
        self.normalize_type = normalize_type
        self.x_transform = x_transform
        self.y_transform = y_transform

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
                - x: dict with predicted_vars_mean, predicted_vars_std, all_vars_mean, all_vars_std, meta_vars, pixel_idx
                - y: target tensor
                - timedelta: normalized prediction timedelta
        """
        fc_path, rea_path, _, leadtime = self.samples[idx]

        # Load tensors (new unified format)
        fc_tensor = torch.load(fc_path)  # unified tensor with all features [c_total, x, y]
        rea = torch.load(rea_path)  # shape [c, x, y]

        # Extract features using indices from metadata
        # Note: predicted_var indices point INTO the all_var arrays (subset relationship)
        predicted_var_mean_indices = self.feature_metadata["predicted_var_mean_indices"]
        predicted_var_std_indices = self.feature_metadata["predicted_var_std_indices"]
        all_var_mean_indices = self.feature_metadata["all_var_mean_indices"]
        all_var_std_indices = self.feature_metadata["all_var_std_indices"]
        meta_var_indices = self.feature_metadata["meta_var_indices"]

        # Slice the unified tensor to get all_vars and meta
        all_vars_mean = fc_tensor[all_var_mean_indices]  # shape [c_all, x, y]
        all_vars_std = fc_tensor[all_var_std_indices]  # shape [c_all, x, y]
        meta = fc_tensor[meta_var_indices]  # shape [c_meta, x, y]
        
        # Normalize all variables (means)
        if self.normalize_type == "zscore":
            all_vars_mean = (
                all_vars_mean - self.norm_stats["all_mean"]
            ) / self.norm_stats["all_std"]
        elif self.normalize_type == "minmax":
            all_vars_mean = (
                all_vars_mean - self.norm_stats["all_min"]
            ) / (
                self.norm_stats["all_max"] - self.norm_stats["all_min"]
            )

        # Normalize all variables (stds)
        if self.normalize_type == "zscore":
            all_vars_std = (
                all_vars_std - self.norm_stats["aux_mean"]
            ) / self.norm_stats["aux_std"]
        elif self.normalize_type == "minmax":
            all_vars_std = (
                all_vars_std - self.norm_stats["aux_min"]
            ) / (
                self.norm_stats["aux_max"] - self.norm_stats["aux_min"]
            )

        # Extract predicted vars AFTER normalization (they're subsets of normalized all_vars)
        predicted_vars_mean = all_vars_mean[predicted_var_mean_indices]
        predicted_vars_std = all_vars_std[predicted_var_std_indices]

        # Normalize REA (reanalysis target)
        if self.normalize_type == "zscore":
            rea = (rea - self.norm_stats["rea_mean"]) / self.norm_stats["rea_std"]
        elif self.normalize_type == "minmax":
            rea = (rea - self.norm_stats["rea_min"]) / (
                self.norm_stats["rea_max"] - self.norm_stats["rea_min"]
            )

        # Apply transforms if provided (after normalization)
        if self.x_transform is not None:
            predicted_vars_mean = self.x_transform(predicted_vars_mean)
            predicted_vars_std = self.x_transform(predicted_vars_std)
            all_vars_mean = self.x_transform(all_vars_mean)
            all_vars_std = self.x_transform(all_vars_std)
            meta = self.x_transform(meta)

        if self.y_transform is not None:
            rea = self.y_transform(rea)

        # Convert timedelta to hours and normalize
        timedelta_hours = leadtime / np.timedelta64(1, "h")
        max_timedelta = self.feature_metadata.get("max_timedelta", 120.0)
        timedelta_normalized = torch.tensor(timedelta_hours / max_timedelta, dtype=torch.float32)

        return {
            "x": {
                "predicted_vars_mean": predicted_vars_mean,
                "predicted_vars_std": predicted_vars_std,
                "all_vars_mean": all_vars_mean,
                "all_vars_std": all_vars_std,
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
        x_transform: Callable | None = None,
        y_transform: Callable | None = None,
        prefetch_factor: int | None = None,
        multiprocessing_context: str | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_split: dict[str, str] | None = None,
        val_split: dict[str, str] | None = None,
        test_split: dict[str, str] | None = None,
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
            x_transform (Callable | None): Optional transform to apply to input features.
            y_transform (Callable | None): Optional transform to apply to targets.
            prefetch_factor (int | None): Number of batches to prefetch per worker.
            multiprocessing_context (str | None): Multiprocessing context ('fork', 'spawn', 'forkserver').
            pin_memory (bool): Whether to pin memory in DataLoader.
            persistent_workers (bool): Whether to keep workers alive between epochs.
            train_split (dict[str, str] | None): Train split config with 'start' and 'end' dates.
            val_split (dict[str, str] | None): Validation split config with 'start' and 'end' dates.
            test_split (dict[str, str] | None): Test split config with 'start' and 'end' dates.
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
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.norm_stats: dict[str, torch.Tensor] | None = None
        self.feature_metadata = None

        self.fc_tensor_dir = DATA_DIR / "tensors" / "fc"
        self.rea_tensor_dir = DATA_DIR / "tensors" / "rea"
        # norm_stats_file will be set with train set identifier in prepare_data
        self.norm_stats_file: Path | None = None

        # Store split configurations (dates for valid_time filtering)
        # Default to old year-based splits if not provided
        self.train_split = train_split or {"start": "2019-01-01", "end": "2021-12-31"}
        self.val_split = val_split or {"start": "2022-01-01", "end": "2022-12-31"}
        self.test_split = test_split or {"start": "2023-01-01", "end": "2023-12-31"}

    def _get_train_set_identifier(self) -> str:
        """Generate a unique identifier for the train set configuration.

        The identifier is based on the tensor paths that belong to the train set
        based on valid_time (init_date + leadtime) within the train split range.
        This allows us to detect if the train set has changed.

        Returns:
            str: A hash string representing the train set configuration.
        """
        # Collect all FC tensor paths
        fc_paths = sorted(list(self.fc_tensor_dir.glob("fc_*.pt")))

        # Parse train split dates
        train_start = np.datetime64(self.train_split["start"])
        train_end = np.datetime64(self.train_split["end"])

        # Filter to only train set samples (valid_time within train range)
        train_paths = []
        for fc_path in fc_paths:
            parts = fc_path.stem.split("_")
            if len(parts) >= 3:
                date_str = parts[1]  # YYYYMMDDHH
                leadtime_str = parts[2]  # leadtime in hours
                try:
                    # Parse init_date
                    init_date = np.datetime64(
                        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{date_str[8:10]}:00:00"
                    )
                    # Parse leadtime
                    leadtime = np.timedelta64(int(leadtime_str), "h")
                    # Calculate valid_time
                    valid_time = init_date + leadtime

                    if train_start <= valid_time <= train_end:
                        train_paths.append(fc_path.name)
                except (ValueError, IndexError):
                    continue

        # Create a hash from the list of train paths (already sorted from fc_paths)
        train_paths_str = ",".join(train_paths)
        hash_obj = hashlib.sha256(train_paths_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars of hash

    def _filter_train_tensor_paths(self, tensor_paths: list[Path]) -> list[Path]:
        """Filter tensor paths to only include train set samples based on valid_time.

        Args:
            tensor_paths: List of tensor file paths to filter.

        Returns:
            List of tensor paths that belong to the train set.
        """
        # Parse train split dates
        train_start = np.datetime64(self.train_split["start"])
        train_end = np.datetime64(self.train_split["end"])

        train_paths = []
        for tensor_path in tensor_paths:
            # Extract date and leadtime from filename
            # Format: fc_YYYYMMDDHH_LT.pt or rea_YYYYMMDD.pt
            parts = tensor_path.stem.split("_")
            if len(parts) >= 3:
                date_str = parts[1]  # YYYYMMDDHH
                leadtime_str = parts[2]  # leadtime in hours
                try:
                    # Parse init_date
                    init_date = np.datetime64(
                        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{date_str[8:10]}:00:00"
                    )
                    # Parse leadtime
                    leadtime = np.timedelta64(int(leadtime_str), "h")
                    # Calculate valid_time
                    valid_time = init_date + leadtime

                    if train_start <= valid_time <= train_end:
                        train_paths.append(tensor_path)
                except (ValueError, IndexError):
                    continue

        return train_paths

    def prepare_data(self) -> None:
        """Prepare data by computing normalization statistics from the train set.

        This method collects samples, splits them, and computes mean, std, min, max
        from the train set for normalization. The stats file includes a train set
        identifier to ensure we recompute if the train set changes.
        """
        # TODO fix some error here
        # For now this does not concern us as much as the data is complete
        # However some file errors here and should be skipped
        # ens_nc_paths = sorted(list((DATA_DIR / "ensmean").glob("*.nc")))
        # self._get_fc_tensors(ens_nc_paths)

        # rea_nc_paths = sorted(list((DATA_DIR / "rea").glob("*.nc")))
        # self._get_rea_tensors(rea_nc_paths)

        # Generate train set identifier and set norm_stats_file path
        train_set_id = self._get_train_set_identifier()
        self.norm_stats_file = DATA_DIR / "tensors" / f"norm_stats_train_{train_set_id}.pt"

        if not self.norm_stats_file.exists():
            print(f"Computing norm stats for train set (id: {train_set_id})...")
            self._compute_norm_stats()
        else:
            print(f"Norm stats file already exists for train set (id: {train_set_id})")

        if self.feature_metadata is None:
            print("Computing feature metadata...")
            self._compute_feature_metadata()

        # TODO if on gpu cluster, move files to specific locations

    def _compute_feature_metadata(self) -> None:
        """Load or compute feature metadata including max timedelta and feature indices."""
        # Try to load feature metadata from pickle file
        fc_metadata_path = self.fc_tensor_dir / "feature_metadata.pkl"
        
        if fc_metadata_path.exists():
            with open(fc_metadata_path, "rb") as f:
                self.feature_metadata = pickle.load(f)
            print(f"Loaded feature metadata from {fc_metadata_path}")
        else:
            # If metadata file doesn't exist, we need to create it
            # This should only happen if tensors were created with the old format
            raise RuntimeError(
                f"Feature metadata file not found at {fc_metadata_path}. "
                "Please regenerate tensors using the updated _get_fc_tensors method."
            )
        
        # Add max timedelta to metadata
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
        
        self.feature_metadata["max_timedelta"] = max_timedelta

    def _compute_tensor_stats(
        self, tensor_paths: list[Path], feature_indices: list[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean, std, min, max statistics for tensors in a single pass.

        Args:
            tensor_paths: List of paths to tensor files
            feature_indices: Optional list of indices to extract from unified tensors

        Returns:
            Tuple of (mean, std, min, max) tensors with shape [c, 1, 1]
        """
        tensor_sum = None
        tensor_sum_sq = None
        tensor_min = None  # type: ignore
        tensor_max = None  # type: ignore
        tensor_count = 0

        for tensor_path in tqdm(tensor_paths):
            # Load tensor (unified tensor format)
            loaded = torch.load(tensor_path)

            # Extract specific indices if provided
            if feature_indices is not None:
                if isinstance(loaded, dict):
                    raise ValueError(
                        f"Found old dict format at {tensor_path}. "
                        "Please regenerate tensors with the new unified format."
                    )
                tensor = loaded[feature_indices]
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

        Statistics are computed ONLY on the train set.

        The computed statistics have shapes:
        - All var (mean) statistics: [c_all, 1, 1]
        - All var (std) statistics: [c_all, 1, 1]
        - REA statistics: [c, 1, 1]
        """
        # Load feature metadata first
        if self.feature_metadata is None:
            self._compute_feature_metadata()
        
        self.norm_stats = {}

        # Compute statistics for all variables (mean) in FC tensors
        fc_tensor_paths = list(self.fc_tensor_dir.glob("fc_*.pt"))
        # Filter to only train set samples
        fc_tensor_paths = self._filter_train_tensor_paths(fc_tensor_paths)

        if fc_tensor_paths:
            print(
                f"Computing all_vars_mean stats from {len(fc_tensor_paths)} train set FC tensors"
            )
            # all_vars_mean are at the beginning of the tensor
            all_mean, all_std, all_min, all_max = self._compute_tensor_stats(
                fc_tensor_paths, feature_indices=self.feature_metadata["all_var_mean_indices"]
            )
            self.norm_stats.update(
                {
                    "all_mean": all_mean,
                    "all_std": all_std,
                    "all_min": all_min,
                    "all_max": all_max,
                }
            )

            # Compute statistics for all variables (std) in FC tensors
            print(f"Computing all_vars_std stats from {len(fc_tensor_paths)} train set FC tensors")
            # all_vars_std come after all_vars_mean
            aux_mean, aux_std, aux_min, aux_max = self._compute_tensor_stats(
                fc_tensor_paths, feature_indices=self.feature_metadata["all_var_std_indices"]
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
        # Filter to only train set samples
        rea_tensor_paths = self._filter_train_tensor_paths(rea_tensor_paths)

        if rea_tensor_paths:
            print(f"Computing rea stats from {len(rea_tensor_paths)} train set REA tensors")
            rea_mean, rea_std, rea_min, rea_max = self._compute_tensor_stats(
                rea_tensor_paths, feature_indices=None  # REA tensors are already just the y variables
            )
            self.norm_stats.update(
                {
                    "rea_mean": rea_mean,
                    "rea_std": rea_std,
                    "rea_min": rea_min,
                    "rea_max": rea_max,
                }
            )

        torch.save(self.norm_stats, self.norm_stats_file)  # type:ignore
        print(f"Saved norm stats to {self.norm_stats_file}")

    def setup(self, stage: str) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage (str): Stage of setup (e.g., 'fit', 'test').
        """
        # Load normalization statistics if not already loaded
        if self.norm_stats is None:
            # Set norm_stats_file path with train set identifier if not already set
            if self.norm_stats_file is None:
                train_set_id = self._get_train_set_identifier()
                self.norm_stats_file = DATA_DIR / "tensors" / f"norm_stats_train_{train_set_id}.pt"

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

        # Collect and sort samples by valid_time (init_date + leadtime)
        all_samples = self._collect_samples()
        all_samples.sort(key=lambda x: x[3] + x[4])  # Sort by valid_time (init_date + leadtime)

        # Parse split date ranges
        train_start = np.datetime64(self.train_split["start"])
        train_end = np.datetime64(self.train_split["end"])
        val_start = np.datetime64(self.val_split["start"])
        val_end = np.datetime64(self.val_split["end"])
        test_start = np.datetime64(self.test_split["start"])
        test_end = np.datetime64(self.test_split["end"])

        # Split by valid_time (forecast valid time = init_date + leadtime)
        train_samples, val_samples, test_samples = [], [], []
        dropped_samples = []
        for sample in all_samples:
            init_date = sample[3]  # np.datetime64
            leadtime = sample[4]  # np.timedelta64
            valid_time = init_date + leadtime  # Forecast valid time

            if train_start <= valid_time <= train_end:
                train_samples.append(sample)
            elif val_start <= valid_time <= val_end:
                val_samples.append(sample)
            elif test_start <= valid_time <= test_end:
                test_samples.append(sample)
            else:
                # Sample falls outside all split ranges
                dropped_samples.append((sample[0].name, valid_time))

        # Log dropped samples if any
        if dropped_samples:
            logger.warning(
                f"Dropped {len(dropped_samples)} samples that fall outside all split ranges. "
                f"First few: {dropped_samples[:5]}"
            )

        self.train_dataset = ForecastDataset(
            train_samples,
            self.norm_stats,  # type: ignore
            self.feature_metadata,
            self.normalize_type,
            self.x_transform,
            self.y_transform,
        )
        self.val_dataset = ForecastDataset(
            val_samples,
            self.norm_stats,  # type: ignore
            self.feature_metadata,
            self.normalize_type,
            self.x_transform,
            self.y_transform,
        )
        self.test_dataset = ForecastDataset(
            test_samples,
            self.norm_stats,  # type: ignore
            self.feature_metadata,
            self.normalize_type,
            self.x_transform,
            self.y_transform,
        )

    @staticmethod
    def _get_fc_tensors_static(
        ens_nc_paths: list[Path],
        x_select_variables: list[str],
        y_select_variables: list[str],
        fc_tensor_dir: Path,
    ) -> dict[str, Any]:
        """Build and store forecast tensors from ensemble NetCDF paths.

        Creates unified tensor files where all features (all_vars_mean, all_vars_std, meta_vars) 
        are concatenated into a single tensor. predicted_vars are identified as a subset of 
        all_vars via indices in the metadata.
        
        Also creates a metadata pickle file that maps feature names to their indices in the tensor.

        Args:
            ens_nc_paths (list[Path]): Paths to ensmean NetCDF files to process.
            x_select_variables (list[str]): List of input variable names (all variables).
            y_select_variables (list[str]): List of target variable names (predicted variables, subset of x).
            fc_tensor_dir (Path): Directory to store forecast tensors.

        Returns:
            dict: Feature metadata mapping feature names to indices.
        """
        # Ensure output directories exist
        fc_tensor_dir.mkdir(parents=True, exist_ok=True)
        
        # Assert that y_select_variables is a subset of x_select_variables
        assert all(
            y_var in x_select_variables for y_var in y_select_variables
        ), f"y_select_variables must be a subset of x_select_variables. " \
           f"Missing variables: {set(y_select_variables) - set(x_select_variables)}"

        # Skip entries with already materialized tensors
        filtered_paths: list[Path] = []
        for ens_path in ens_nc_paths:
            time_leadtime = "_".join(ens_path.stem.split("_")[1:])
            fc_path = fc_tensor_dir / f"fc_{time_leadtime}.pt"
            if fc_path.exists():
                continue
            filtered_paths.append(ens_path)
        ens_nc_paths = filtered_paths

        # Build feature metadata once (same for all files)
        feature_metadata = None
        
        # Build matching ensstd paths for remaining inputs
        std_nc_paths = [Path(str(p).replace("ensmean", "ensstd")) for p in ens_nc_paths]
        # Process mean/std pairs together
        for paths in tqdm(
            zip(ens_nc_paths, std_nc_paths), desc="Generating FC Tensors", total=len(ens_nc_paths)
        ):
            datasets = []
            time_leadtime = "_".join(paths[0].stem.split("_")[1:])
            missing_var = False
            for path in paths:
                ds = xr.open_dataset(path).drop_vars(VARS_TO_DROP)
                for level in LEVELS_TO_FLATTEN:
                    try:
                        ds = flatten_levels(ds, level)
                    except KeyError:
                        # Here KeyErrors are fine since a level of a var might be missing but we do not need that var
                        continue
                try:
                    da = ds[VARS_GRID_28].to_dataarray("feature").squeeze().transpose(*AXIS_ORDER)
                    datasets.append(da)
                except KeyError:
                    missing_var = True
            if missing_var:
                print(f"Skipping {paths} due to missing vars")
                continue
            da_stacked = xr.concat(datasets, dim="aggregation")
            da_stacked.coords["aggregation"] = ["mean", "std"]

            # Get metadata features
            meta = get_metadata_features(da_stacked)
            
            # Build unified tensor with all features
            # Order: all_vars_mean, all_vars_std, meta_vars
            # Note: predicted_vars are a SUBSET of all_vars (identified by indices)
            
            # 1. all_vars_mean (all x variables from mean aggregation)
            all_vars_mean = da_stacked.sel(aggregation="mean", feature=x_select_variables)
            all_vars_mean_tensor = torch.from_numpy(all_vars_mean.values)
            
            # 2. all_vars_std (all x variables from std aggregation)
            all_vars_std = da_stacked.sel(aggregation="std", feature=x_select_variables)
            all_vars_std_tensor = torch.from_numpy(all_vars_std.values)
            
            # 3. meta_vars
            meta_tensor = torch.from_numpy(meta.values)
            
            # Concatenate all features into a single tensor along feature dimension
            unified_tensor = torch.cat([
                all_vars_mean_tensor,
                all_vars_std_tensor,
                meta_tensor,
            ], dim=0)  # Concatenate along feature dimension (dim 0)
            
            # Build feature metadata on first iteration
            if feature_metadata is None:
                # Track indices for each feature category
                idx = 0
                
                # All vars mean
                all_var_mean_names = x_select_variables
                all_var_mean_indices = list(range(idx, idx + len(all_var_mean_names)))
                idx += len(all_var_mean_names)
                
                # All vars std
                all_var_std_names = x_select_variables
                all_var_std_indices = list(range(idx, idx + len(all_var_std_names)))
                idx += len(all_var_std_names)
                
                # Meta vars
                meta_var_names = meta.feature.values.tolist()
                meta_var_indices = list(range(idx, idx + len(meta_var_names)))
                idx += len(meta_var_names)
                
                # Predicted vars are a SUBSET of all_vars
                # Find which indices in all_var_mean/std correspond to y_select_variables
                predicted_var_mean_names = y_select_variables
                predicted_var_mean_indices = [
                    i for i, name in enumerate(all_var_mean_names) if name in y_select_variables
                ]
                
                predicted_var_std_names = y_select_variables
                # Use indices relative to all_vars_std (consistent with predicted_var_mean_indices)
                predicted_var_std_indices = [
                    i for i, name in enumerate(all_var_std_names) if name in y_select_variables
                ]
                
                feature_metadata = {
                    # Predicted variables (indices into all_vars arrays, not separate storage)
                    "predicted_var_mean_names": predicted_var_mean_names,
                    "predicted_var_mean_indices": predicted_var_mean_indices,
                    "predicted_var_std_names": predicted_var_std_names,
                    "predicted_var_std_indices": predicted_var_std_indices,
                    # All input variables (means and stds separately)
                    "all_var_mean_names": all_var_mean_names,
                    "all_var_mean_indices": all_var_mean_indices,
                    "all_var_std_names": all_var_std_names,
                    "all_var_std_indices": all_var_std_indices,
                    # Meta variables
                    "meta_var_names": meta_var_names,
                    "meta_var_indices": meta_var_indices,
                    "pixel_idx_index": None,  # ICON doesn't use pixel_idx
                }
            
            # Save unified tensor
            fc_path = fc_tensor_dir / f"fc_{time_leadtime}.pt"
            torch.save(unified_tensor, fc_path)
        
        # Save feature metadata to pickle file (only once)
        if feature_metadata is not None:
            metadata_path = fc_tensor_dir / "feature_metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump(feature_metadata, f)
            print(f"Saved feature metadata to {metadata_path}")
        
        return feature_metadata or {}

    def _get_fc_tensors(self, ens_nc_paths: list[Path]) -> dict[str, Any]:
        """Build and store forecast tensors from ensemble NetCDF paths.

        Args:
            ens_nc_paths (list[Path]): Paths to ensmean NetCDF files to process.

        Returns:
            dict: Feature metadata mapping feature names to indices.
        """
        return ForecastDataModule._get_fc_tensors_static(
            ens_nc_paths,
            self.x_select_variables,
            self.y_select_variables,
            self.fc_tensor_dir,
        )

    @staticmethod
    def _get_rea_tensors_static(
        rea_nc_paths: list[Path],
        y_select_variables: list[str],
        rea_tensor_dir: Path,
    ) -> dict[str, Any]:
        """Build and store reanalysis tensors from NetCDF paths, skipping existing outputs.

        Creates unified tensor files where all reanalysis features are stored in a single tensor.
        Also creates a metadata pickle file that maps feature names to their indices.

        Args:
            rea_nc_paths (list[Path]): Paths to reanalysis NetCDF files to process.
            y_select_variables (list[str]): List of target variable names.
            rea_tensor_dir (Path): Directory to store reanalysis tensors.

        Returns:
            dict: Feature metadata mapping feature names to indices.
        """
        # Ensure output directory exists
        rea_tensor_dir.mkdir(parents=True, exist_ok=True)

        # Skip entries with already materialized tensors
        filtered_paths: list[Path] = []
        for rea_path in rea_nc_paths:
            date = rea_path.stem.split("_")[-1]
            tens_path = rea_tensor_dir / f"rea_{date}.pt"
            if tens_path.exists():
                continue
            filtered_paths.append(rea_path)
        rea_nc_paths = filtered_paths

        # Build feature metadata once
        feature_metadata = None
        
        for rea_path in tqdm(rea_nc_paths, desc="Generating REA Tensors"):
            date = rea_path.stem.split("_")[-1]
            rea = xr.open_dataset(rea_path)
            rea = rea.drop_vars("rotated_pole")
            for dim in ["height", "height_2"]:
                try:
                    rea = flatten_levels(rea, level_dim=dim)
                except KeyError:
                    # Some files may not have all dimensions (e.g., early rea files missing height_2)
                    continue
            try:
                rea = (
                    rea.to_dataarray("feature")
                    .sel(feature=y_select_variables)
                    .transpose(..., *AXIS_ORDER)
                )
            except KeyError:
                print(f"Skipping {rea_path} due to missing vars")
                continue
            tens_path = rea_tensor_dir / f"rea_{date}.pt"
            tens = torch.from_numpy(rea.values).squeeze()
            # Rea has shape [c, x, y]
            torch.save(tens, tens_path)
            
            # Build feature metadata on first iteration
            if feature_metadata is None:
                feature_metadata = {
                    "y_var_names": y_select_variables,
                    "y_var_indices": list(range(len(y_select_variables))),
                }
        
        # Save feature metadata to pickle file (only once)
        if feature_metadata is not None:
            metadata_path = rea_tensor_dir / "feature_metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump(feature_metadata, f)
            print(f"Saved REA feature metadata to {metadata_path}")
        
        return feature_metadata or {}

    def _get_rea_tensors(self, rea_nc_paths: list[Path]) -> dict[str, Any]:
        """Build and store reanalysis tensors from NetCDF paths, skipping existing outputs.

        Args:
            rea_nc_paths (list[Path]): Paths to reanalysis NetCDF files to process.

        Returns:
            dict: Feature metadata mapping feature names to indices.
        """
        return ForecastDataModule._get_rea_tensors_static(
            rea_nc_paths,
            self.y_select_variables,
            self.rea_tensor_dir,
        )

    def _collect_samples(self) -> list[tuple[Path, Path, np.datetime64, np.timedelta64]]:
        """Collect valid samples from the data directories.

        Parses filenames in the tensors file to generate tuples of (fc_path, rea_path, init_date, leadtime).
        Meta features are embedded in fc_path.

        Returns:
            list[tuple[Path, Path, np.datetime64, np.timedelta64]]: List of tuples (fc_path, rea_path, init_date, leadtime).
        """
        samples: list[tuple[Path, Path, np.datetime64, np.timedelta64]] = []

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

                # Build corresponding rea path
                rea_path = self.rea_tensor_dir / f"rea_{target_date_str}.pt"

                # Check if both files exist
                if fc_path.exists() and rea_path.exists():
                    samples.append((fc_path, rea_path, init_date, leadtime))

            except (ValueError, IndexError):
                # Skip malformed filenames
                continue

        return samples

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        if self.multiprocessing_context is not None and self.num_workers > 0:
            dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context
        return DataLoader(self.train_dataset, **dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        if self.multiprocessing_context is not None and self.num_workers > 0:
            dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context
        return DataLoader(self.val_dataset, **dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader.

        Returns:
            DataLoader: DataLoader for test data.
        """
        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        if self.multiprocessing_context is not None and self.num_workers > 0:
            dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context
        return DataLoader(self.test_dataset, **dataloader_kwargs)
