"""
Optimized dataset implementation using PyTorch's native tensor saving for fast data loading.
As long as the data fits into memory this is the fastest way to load it.
TODO if we need it we could extend this to use numpy's memory mapping for larger-than-memory datasets.
For huge cloud datasets, xbatcher seems like the best option.
"""

import hashlib
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any

import lightning as L
import torch
import xarray as xr
import xbatcher
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xbatcher.loaders.torch import to_tensor

from genpp.data import (
    FORECAST_ENS_FLAT_AGG_NAME,
    FORECAST_ENS_NAME,
    OBSERVATIONS_FLAT_NAME,
    OBSERVATIONS_NAME,
    OUTPUT_DIR,
)
from genpp.data.utils import flatten_levels
from genpp.preproc.preprocessors import Preprocessor


def _compute_config_hash(
    dataset_config: DictConfig,
    x_select_variables: list[str],
    y_select_variables: list[str],
    x_preprocessing: list[Preprocessor] | None = None,
    y_preprocessing: list[Preprocessor] | None = None,
) -> str:
    """
    Compute a hash of the dataset configuration to detect if cached data can be reused.

    Args:
        dataset_config: Dataset configuration including time slices and batch kwargs
        x_select_variables: List of x variables to select
        y_select_variables: List of y variables to select
        x_preprocessing: List of preprocessing steps for x data
        y_preprocessing: List of preprocessing steps for y data

    Returns:
        Hash string representing the configuration
    """
    # Convert dataset config to dict and exclude transforms (they are applied after caching)
    config_container = OmegaConf.to_container(dataset_config, resolve=True)
    
    # Remove x_transform and y_transform from each split as they don't affect cached data
    for split in ["train", "val", "test"]:
        if split in config_container and isinstance(config_container[split], dict):
            config_container[split].pop("x_transform", None)
            config_container[split].pop("y_transform", None)
    
    # Create a dictionary with all relevant configuration
    config_dict = {
        "dataset_config": config_container,
        "x_select_variables": sorted(x_select_variables),
        "y_select_variables": sorted(y_select_variables),
        "x_preprocessing": [type(p).__name__ for p in x_preprocessing] if x_preprocessing else [],
        "y_preprocessing": [type(p).__name__ for p in y_preprocessing] if y_preprocessing else [],
    }

    # Convert to string and compute hash
    config_str = str(sorted(config_dict.items()))
    return hashlib.sha256(config_str.encode()).hexdigest()


class TransformTensorDataset(Dataset):
    """Custom dataset that applies transforms to tensors in __getitem__.

    This dataset wraps tensors and applies transforms on-the-fly during data loading,
    rather than pre-applying them during preprocessing.
    """

    def __init__(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        dt_tensor: torch.Tensor,
        x_transform: Any = None,
        y_transform: Any = None,
    ):
        """Initialize the dataset with tensors and optional transforms.

        Args:
            x_tensor: Input feature tensor
            y_tensor: Target tensor
            x_transform: Optional transform to apply to x data
            y_transform: Optional transform to apply to y data
        """
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.dt_tensor = dt_tensor
        self.x_transform = x_transform
        self.y_transform = y_transform

        if len(x_tensor) != len(y_tensor):
            raise ValueError(f"Tensor lengths don't match: {len(x_tensor)} vs {len(y_tensor)}")

    def __len__(self) -> int:
        return len(self.x_tensor)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        dt = self.dt_tensor[index]

        # Apply transforms if provided
        if self.x_transform is not None:
            x = self.x_transform(x)
        if self.y_transform is not None:
            y = self.y_transform(y)

        return x, y, dt


def _cache_data(
    x_da: xr.DataArray,
    y_da: xr.DataArray,
    time_slices: dict[str, slice],
    cache_dir: Path,
    batch_kwargs: dict,
) -> tuple[Any, dict[str, list[int]]]:
    """
    Pre-process xarray data and save as PyTorch tensor files.

    Args:
        x_da: Input xarray DataArray
              (feature: 63, time: 3651, prediction_timedelta: 5, latitude: 31, longitude: 37)
        y_da: Target xarray DataArray
              (feature: 2, time: 7304, latitude: 31, longitude: 37)
        time_slices: Dictionary mapping split names to time slices
        cache_dir: Directory to save cached files
        batch_kwargs: Batching configuration from xbatcher

    Returns:
        Tuple of (path_to_saved_dict, split_indices)
        The saved dict contains keys: "x", "y", "prediction_timedelta"
    """
    assert x_da.feature.shape[0] == batch_kwargs["input_dims"]["feature"]

    all_x, all_y, all_dt = [], [], []
    split_indices: dict[str, list[int]] = {}
    current_idx = 0

    for split_name, time_slice in time_slices.items():
        # We split both datasets right here to prevent leaking data
        x_split = x_da.sel(time=time_slice)
        y_split = y_da.sel(time=time_slice)

        # These batch kwargs are fixed so that the batches are random
        batch_kwargs["batch_dims"]["time"] = 1
        batch_kwargs["batch_dims"]["prediction_timedelta"] = 1

        gen = xbatcher.BatchGenerator(x_split, **batch_kwargs)

        split_batch_indices = []
        for x_batch in tqdm(gen):
            x_tensor = to_tensor(x_batch)
            t0 = x_batch.time.values[0]
            dt = x_batch.prediction_timedelta.values[0]
            y_t = t0 + dt
            if y_t not in y_split.time.values:
                # This data is not includeded as it would be in another split
                # thus leaking data
                continue
            y_batch = y_split.sel(time=y_t, longitude=x_batch.longitude, latitude=x_batch.latitude)
            y_tensor = to_tensor(y_batch.compute())

            if y_tensor.dim() == 3:
                y_tensor = y_tensor.unsqueeze(0).unsqueeze(0)

            tensor_dt = torch.tensor([dt], dtype=torch.float32)

            all_x.append(x_tensor)
            all_y.append(y_tensor)
            all_dt.append(tensor_dt)

            split_batch_indices.append(current_idx)
            current_idx += 1

        split_indices[split_name] = split_batch_indices

    if not all_x:
        raise ValueError("No batches generated from the data")

    x_tensor = torch.cat(all_x, dim=0)
    y_tensor = torch.cat(all_y, dim=0)
    dt_tensor = torch.cat(all_dt, dim=0)

    temp_file = tempfile.NamedTemporaryFile(dir=cache_dir, suffix="_tensor.pt", delete=False)
    torch.save({"x": x_tensor, "y": y_tensor, "prediction_timedelta": dt_tensor}, temp_file.name)
    return temp_file.name, split_indices


class FastWeatherBench2DataModule(L.LightningDataModule):
    """Optimized DataModule for WeatherBench2 dataset using cached PyTorch tensors."""

    def __init__(
        self,
        dataset_config: DictConfig,
        dataloader_config: DictConfig,
        x_select_variables: ListConfig | list[str],
        y_select_variables: ListConfig | list[str],
        save_dir: Path = OUTPUT_DIR,
        x_preprocessing: list[Preprocessor] | None = None,
        y_preprocessing: list[Preprocessor] | None = None,
    ) -> None:
        super().__init__()
        self.path = save_dir
        self.x_preprocessing = x_preprocessing
        self.y_preprocessing = y_preprocessing
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.x_select_variables = (
            x_select_variables if isinstance(x_select_variables, list) else list(x_select_variables)
        )
        self.y_select_variables = (
            y_select_variables if isinstance(y_select_variables, list) else list(y_select_variables)
        )
        self.already_prepared = False

        # Create cache directory for fast tensor data
        self.cache_dir = save_dir / "tmp"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize temporary file attributes
        self.x_tmp = None
        self.y_tmp = None
        self.metadata_tmp = None

    def _select_variables(
        self,
        da: xr.DataArray,
        select_variables: ListConfig | list[str],
        append_suffix: bool = True,
    ) -> xr.DataArray:
        """Select only specified variables from the dataset."""
        if select_variables is None:
            raise ValueError("No variables specified for selection.")

        # Convert ListConfig to list if necessary
        variables = (
            list(select_variables) if isinstance(select_variables, ListConfig) else select_variables
        )

        # Create list of feature names to keep (with _mean and _std suffixes)
        features_to_keep = []
        if append_suffix:
            # First all means, then all stds
            for var in variables:
                features_to_keep.append(f"{var}_mean")
            for var in variables:
                features_to_keep.append(f"{var}_std")
        else:
            features_to_keep = variables

        # Filter to only include the specified features
        available_features = da.feature.values
        selected_features = [f for f in features_to_keep if f in available_features]
        if not selected_features:
            raise ValueError(
                f"None of the requested variables {variables} were found in the dataset. "
                f"Available variables: {[f.replace('_mean', '').replace('_std', '') for f in available_features if f.endswith(('_mean', '_std'))]}"
            )

        return da.sel(feature=selected_features)

    def prepare_data(self) -> None:
        """Prepare and cache tensor data for fast loading."""
        if self.already_prepared:
            return

        # Compute hash of configuration to check if data is already cached
        config_hash = _compute_config_hash(
            self.dataset_config,
            self.x_select_variables,
            self.y_select_variables,
            self.x_preprocessing,
            self.y_preprocessing,
        )

        # Check if cached data with this hash exists
        cached_tensor_path = self.cache_dir / f"tensor_{config_hash}.pt"
        cached_metadata_path = self.cache_dir / f"metadata_{config_hash}.pkl"

        if cached_tensor_path.exists() and cached_metadata_path.exists():
            # Load existing cached data
            with open(cached_metadata_path, "rb") as f:
                cache_metadata = pickle.load(f)

            # Verify the hash matches
            if cache_metadata.get("config_hash") == config_hash:
                # Use existing cached files
                self.tmp = cached_tensor_path
                self.metadata_tmp = type("obj", (object,), {"name": str(cached_metadata_path)})

                # Store reverse modules for later use
                if self.x_preprocessing:
                    self.x_reverseModules = [
                        rm
                        for preprocessor in self.x_preprocessing
                        if (rm := preprocessor.get_reverse_module()) is not None
                    ]
                if self.y_preprocessing:
                    self.y_reverseModules = [
                        rm
                        for preprocessor in self.y_preprocessing
                        if (rm := preprocessor.get_reverse_module()) is not None
                    ]

                self.already_prepared = True
                return

        # Check if data exists
        if not os.path.exists(self.path / FORECAST_ENS_NAME):
            raise FileNotFoundError(
                f"Forecast ensemble data not found at {self.path / FORECAST_ENS_NAME}. "
                "Please download the dataset first."
            )

        if not os.path.exists(self.path / OBSERVATIONS_NAME):
            raise FileNotFoundError(
                f"Observations data not found at {self.path / OBSERVATIONS_NAME}. "
                "Please download the dataset first."
            )

        # Ensure flat and aggregated data exists
        if (
            not (self.path / FORECAST_ENS_FLAT_AGG_NAME).exists()
            or not (self.path / OBSERVATIONS_FLAT_NAME).exists()
        ):
            warnings.warn(
                "Flattening and aggregation of ensemble members and observations is not done yet. "
                "This will take some time..."
            )
            from genpp.data.flat_and_aggr import main as preprocess_main

            preprocess_main(base_dir=self.path)

        # Process X data
        x_da = xr.open_zarr(self.path / FORECAST_ENS_FLAT_AGG_NAME, consolidated=True)
        # Select only specified variables
        if self.x_select_variables:
            x_da = x_da[self.x_select_variables]

        # Turn into DataArray
        x_da = flatten_levels(x_da, level_dim="statistic", interleave=False)
        x_da = x_da.to_dataarray(dim="feature")

        # Apply preprocessing to x data
        if self.x_preprocessing:
            for preprocessor in self.x_preprocessing:
                # Preprocessors work on dataarrays
                preprocessor.fit(x_da.sel(time=self.dataset_config.train.slice))
                if not preprocessor.fit_only:
                    x_da = preprocessor.preprocess(x_da)

        # Process Y data
        y_da = xr.open_zarr(self.path / OBSERVATIONS_FLAT_NAME, consolidated=True)
        if self.y_select_variables:
            y_da = y_da[self.y_select_variables]

        # Turn into DataArray (there are no levels to flatten here)
        y_da = y_da.to_dataarray(dim="feature")

        # Apply preprocessing to y data
        if self.y_preprocessing:
            for preprocessor in self.y_preprocessing:
                preprocessor.fit(y_da.sel(time=self.dataset_config.train.slice))
                if not preprocessor.fit_only:
                    y_da = preprocessor.preprocess(y_da)

        # Define time slices for splits
        time_slice = {
            "train": self.dataset_config.train.slice,
            "val": self.dataset_config.val.slice,
            "test": self.dataset_config.test.slice,
        }

        # Preprocess and save x data
        tmp_tensor_path, split_indices = _cache_data(
            x_da, y_da, time_slice, self.cache_dir, batch_kwargs=self.dataset_config.train.x_kwargs
        )

        # Move temporary file to hash-based permanent location
        cached_tensor_path = self.cache_dir / f"tensor_{config_hash}.pt"
        os.rename(tmp_tensor_path, cached_tensor_path)
        self.tmp = cached_tensor_path

        # Store metadata with hash
        cache_metadata = {
            "config_hash": config_hash,
            "tmp_path": str(cached_tensor_path),
            "split_indices": split_indices,
        }

        # Store reverse modules for later use
        if self.x_preprocessing:
            self.x_reverseModules = [
                rm
                for preprocessor in self.x_preprocessing
                if (rm := preprocessor.get_reverse_module()) is not None
            ]
        if self.y_preprocessing:
            self.y_reverseModules = [
                rm
                for preprocessor in self.y_preprocessing
                if (rm := preprocessor.get_reverse_module()) is not None
            ]

        # Save metadata to hash-based permanent location
        cached_metadata_path = self.cache_dir / f"metadata_{config_hash}.pkl"
        with open(cached_metadata_path, "wb") as f:
            pickle.dump(cache_metadata, f)

        self.metadata_tmp = type("obj", (object,), {"name": str(cached_metadata_path)})
        self.already_prepared = True

    def setup(self, stage: str) -> None:
        """Setup datasets for the given stage."""
        if not hasattr(self, "metadata_tmp") or self.metadata_tmp is None:
            raise RuntimeError("prepare_data() must be called before setup()")

        # Load cache metadata from file
        with open(self.metadata_tmp.name, "rb") as f:
            cache_metadata = pickle.load(f)

        # Load cached tensors
        tmp_tensor = torch.load(cache_metadata["tmp_path"])
        x_tensor = tmp_tensor["x"]
        y_tensor = tmp_tensor["y"]
        dt_tensor = tmp_tensor["prediction_timedelta"]

        # Get transforms from metadata
        x_transform = self.dataset_config.train.x_transform
        y_transform = self.dataset_config.train.y_transform

        if stage == "fit":
            # Create training dataset
            train_indices = cache_metadata["split_indices"]["train"]
            self.train_dataset = TransformTensorDataset(
                x_tensor[train_indices],
                y_tensor[train_indices],
                dt_tensor[train_indices],
                x_transform=x_transform,
                y_transform=y_transform,
            )

        if stage in ("fit", "validate"):
            # Create validation dataset
            val_indices = cache_metadata["split_indices"]["val"]
            self.val_dataset = TransformTensorDataset(
                x_tensor[val_indices],
                y_tensor[val_indices],
                dt_tensor[val_indices],
                x_transform=x_transform,
                y_transform=y_transform,
            )

        if stage == "test":
            # Create test dataset
            test_indices = cache_metadata["split_indices"]["test"]
            self.test_dataset = TransformTensorDataset(
                x_tensor[test_indices],
                y_tensor[test_indices],
                dt_tensor[test_indices],
                x_transform=x_transform,
                y_transform=y_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self.dataloader_config.train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self.dataloader_config.val,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            **self.dataloader_config.test,
        )

    def cleanup(self) -> None:
        """Clean up any temporary files.

        Note: Cached tensor and metadata files are kept for future use.
        They are reused when the same configuration is used again.
        """
        # The tensor and metadata files are now persistent cache files
        # and should not be deleted. They will be reused when the same
        # configuration is used again based on the config hash.
        pass
