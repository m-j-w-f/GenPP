"""
Optimized dataset implementation using PyTorch's native tensor saving for fast data loading.
As long as the data fits into memory this is the fastest way to load it.
TODO if we need it we could extend this to use numpy's memory mapping for larger-than-memory datasets.
For huge cloud datasets, xbatcher seems like the best option.
"""

import hashlib
import pickle
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
    MetadataVars,
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
        if split in config_container and isinstance(config_container[split], dict):  # type: ignore
            config_container[split].pop("x_transform", None)  # type: ignore
            config_container[split].pop("y_transform", None)  # type: ignore

    # Create a dictionary with all relevant configuration
    config_dict = {
        "dataset_config": config_container,
        "x_select_variables": x_select_variables,
        "y_select_variables": y_select_variables,
        "x_preprocessing": [type(p).__name__ for p in x_preprocessing] if x_preprocessing else [],
        "y_preprocessing": [type(p).__name__ for p in y_preprocessing] if y_preprocessing else [],
    }

    # Convert to string and compute hash
    config_str = str(sorted(config_dict.items()))
    return hashlib.sha256(config_str.encode()).hexdigest()


def _cache_data(
    x_da: xr.DataArray,
    y_da: xr.DataArray,
    time_slices: dict[str, slice],
    tensor_save_path: Path,
    batch_kwargs: dict,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """
    Pre-process xarray data and save as PyTorch tensor files.

    Args:
        x_da: Input xarray DataArray
              (feature: 63, time: 3651, prediction_timedelta: 5, latitude: 31, longitude: 37)
        y_da: Target xarray DataArray
              (feature: 2, time: 7304, latitude: 31, longitude: 37)
        time_slices: Dictionary mapping split names to time slices
        tensor_save_path: Directory to save cached tensor files
        batch_kwargs: Batching configuration from xbatcher

    Returns:
        A tuple containing:
        - The split indices indicating the train, val and test sets.
        - Feature categorization metadata (predicted, auxiliary, meta variable info)
    """
    assert x_da.feature.shape[0] == batch_kwargs["input_dims"]["feature"]

    # Categorize features
    all_x_features = x_da.feature.values.tolist()
    all_y_features = y_da.feature.values.tolist()
    meta_var_values = [v.value for v in MetadataVars]

    # Identify feature categories
    meta_var_names = [f for f in all_x_features if f in meta_var_values and f != "pixel_idx"]
    predicted_var_names = [
        f for f in all_x_features if f.removesuffix("+statistic_mean") in all_y_features
    ]
    auxiliary_var_names = [
        f for f in all_x_features if f not in predicted_var_names and f not in meta_var_names
    ]

    # Get feature indices for each category
    meta_var_indices = [i for i, f in enumerate(all_x_features) if f in meta_var_names]
    predicted_var_indices = [i for i, f in enumerate(all_x_features) if f in predicted_var_names]
    auxiliary_var_indices = [i for i, f in enumerate(all_x_features) if f in auxiliary_var_names]
    pixel_idx_index = [all_x_features.index("pixel_idx")] if "pixel_idx" in all_x_features else None

    feature_metadata = {
        "predicted_var_names": predicted_var_names,
        "predicted_var_indices": predicted_var_indices,
        "auxiliary_var_names": auxiliary_var_names,
        "auxiliary_var_indices": auxiliary_var_indices,
        "meta_var_names": meta_var_names,
        "meta_var_indices": meta_var_indices,
        "pixel_idx_index": pixel_idx_index,
    }

    all_x, all_y, all_td = [], [], []
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
            td = x_batch.prediction_timedelta.values[0]
            y_t = t0 + td
            if y_t not in y_split.time.values:
                # This data is not includeded as it would be in another split
                # thus leaking data
                continue
            y_batch = y_split.sel(time=y_t, longitude=x_batch.longitude, latitude=x_batch.latitude)
            y_tensor = to_tensor(y_batch.compute())

            # Ensure y_tensor has the same number of dimensions as x_tensor by
            # prepending singleton dimensions as needed.
            y_tensor = y_tensor.unsqueeze(0)

            tensor_td = torch.tensor([td], dtype=torch.float32)

            all_x.append(x_tensor)
            all_y.append(y_tensor)
            all_td.append(tensor_td)

            split_batch_indices.append(current_idx)
            current_idx += 1

        split_indices[split_name] = split_batch_indices

    if not all_x:
        raise ValueError("No batches generated from the data")

    x_tensor = torch.cat(all_x, dim=0)
    y_tensor = torch.cat(all_y, dim=0)
    td_tensor = torch.cat(all_td, dim=0)

    # Normalize td_tensor to have max value of 1
    td_tensor = td_tensor / td_tensor.max()

    torch.save({"x": x_tensor, "y": y_tensor, "prediction_timedelta": td_tensor}, tensor_save_path)
    return split_indices, feature_metadata


class TransformTensorDataset(Dataset):
    """Custom dataset that applies transforms to tensors in __getitem__.

    This dataset wraps tensors and applies transforms on-the-fly during data loading,
    rather than pre-applying them during preprocessing.
    """

    def __init__(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        td_tensor: torch.Tensor,
        feature_metadata: dict[str, Any],
        x_transform: Any = None,
        y_transform: Any = None,
    ):
        """Initialize the dataset with tensors and optional transforms.

        Args:
            x_tensor: Input feature tensor
            y_tensor: Target tensor
            td_tensor: Prediction timedelta tensor
            feature_metadata: Dictionary containing feature categorization info
            x_transform: Optional transform to apply to x data
            y_transform: Optional transform to apply to y data
        """
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.td_tensor = td_tensor
        self.feature_metadata = feature_metadata
        self.x_transform = x_transform
        self.y_transform = y_transform

        if len(x_tensor) != len(y_tensor):
            raise ValueError(f"Tensor lengths don't match: {len(x_tensor)} vs {len(y_tensor)}")

    def __len__(self) -> int:
        return len(self.x_tensor)

    def __getitem__(self, index) -> dict[str, Any]:
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        td = self.td_tensor[index]

        # Apply transforms if provided
        if self.x_transform is not None:
            x = self.x_transform(x)
        if self.y_transform is not None:
            y = self.y_transform(y)

        # Split x into predicted, auxiliary, and meta variables
        predicted_var_indices = self.feature_metadata["predicted_var_indices"]
        auxiliary_var_indices = self.feature_metadata["auxiliary_var_indices"]
        meta_var_indices = self.feature_metadata["meta_var_indices"]
        pixel_idx_index = self.feature_metadata.get("pixel_idx_index", None)

        # Extract tensors for each category
        # x shape: [feature, ...] (the collate is not called yet -> no batch dimension)
        predicted_vars = x[predicted_var_indices] if predicted_var_indices else torch.tensor([])
        auxiliary_vars = x[auxiliary_var_indices] if auxiliary_var_indices else torch.tensor([])
        meta_vars = x[meta_var_indices] if meta_var_indices else torch.tensor([])
        pixel_idx = x[pixel_idx_index] if pixel_idx_index else torch.tensor([])

        return {
            "x": {
                "predicted_vars": predicted_vars,
                "auxiliary_vars": auxiliary_vars,
                "meta_vars": meta_vars,
                "pixel_idx": pixel_idx.int(),
            },
            "y": y,
            "timedelta": td,
        }


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
        self.cache_dir = save_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize temporary file attributes
        self.metadata_path = None

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

        # Check if data exists
        if not (self.path / FORECAST_ENS_NAME).exists():
            raise FileNotFoundError(
                f"Forecast ensemble data not found at {self.path / FORECAST_ENS_NAME}. "
                "Please download the dataset first."
            )

        if not (self.path / OBSERVATIONS_NAME).exists():
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
        x_da = x_da.transpose("time", "prediction_timedelta", "feature", "longitude", "latitude")

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
        y_da = y_da.transpose("time", "feature", "longitude", "latitude")

        # Apply preprocessing to y data
        if self.y_preprocessing:
            for preprocessor in self.y_preprocessing:
                preprocessor.fit(y_da.sel(time=self.dataset_config.train.slice))
                if not preprocessor.fit_only:
                    y_da = preprocessor.preprocess(y_da)

        # Store reverse modules for later use (requires fitted preprocessors)
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

        # Compute hash of configuration to check if data is already cached
        config_hash = _compute_config_hash(
            self.dataset_config,
            self.x_select_variables,
            self.y_select_variables,
            self.x_preprocessing,
            self.y_preprocessing,
        )
        print(f"Configuration hash: {config_hash}")

        # Check if cached data with this hash exists
        self.tensor_path = self.cache_dir / f"tensor_{config_hash}.pt"
        self.metadata_path = self.cache_dir / f"metadata_{config_hash}.pkl"

        if self.tensor_path.exists() and self.metadata_path.exists():
            # Load existing cached data
            print("Cached tensor data found. Verifying configuration...")
            with open(self.metadata_path, "rb") as f:
                cache_metadata = pickle.load(f)

            # Verify the hash matches
            if cache_metadata.get("config_hash") == config_hash:
                # Use existing cached files (skip expensive _cache_data call)
                self.already_prepared = True
                print("Using cached tensor data.")
                return

        # Define time slices for splits
        time_slice = {
            "train": self.dataset_config.train.slice,
            "val": self.dataset_config.val.slice,
            "test": self.dataset_config.test.slice,
        }

        # Cache is not found - run the expensive _cache_data function
        split_indices, feature_metadata = _cache_data(
            x_da,
            y_da,
            time_slice,
            self.tensor_path,
            batch_kwargs=self.dataset_config.train.x_kwargs,
        )

        # Store metadata with hash
        cache_metadata = {
            "config_hash": config_hash,
            "tmp_path": str(self.tensor_path),
            "split_indices": split_indices,
            "feature_metadata": feature_metadata,
            "x_variables": x_da.feature.values.tolist(),
            "y_variables": y_da.feature.values.tolist(),
            "x_preprocessing": [type(p).__name__ for p in self.x_preprocessing]
            if self.x_preprocessing
            else [],
            "y_preprocessing": [type(p).__name__ for p in self.y_preprocessing]
            if self.y_preprocessing
            else [],
        }

        # Save metadata to hash-based permanent location
        with open(self.metadata_path, "wb") as f:
            pickle.dump(cache_metadata, f)

        self.already_prepared = True

    def setup(self, stage: str) -> None:
        """Setup datasets for the given stage."""
        if self.metadata_path is None:
            raise RuntimeError("prepare_data() must be called before setup()")

        # Load cache metadata from file
        with open(self.metadata_path, "rb") as f:
            cache_metadata = pickle.load(f)

        # Load cached tensors
        tmp_tensor = torch.load(cache_metadata["tmp_path"])
        x_tensor = tmp_tensor["x"]
        y_tensor = tmp_tensor["y"]
        td_tensor = tmp_tensor["prediction_timedelta"]

        # Get feature metadata
        feature_metadata = cache_metadata["feature_metadata"]

        # Get transforms from metadata
        x_transform = self.dataset_config.train.x_transform
        y_transform = self.dataset_config.train.y_transform

        if stage == "fit":
            # Create training dataset
            train_indices = cache_metadata["split_indices"]["train"]
            self.train_dataset = TransformTensorDataset(
                x_tensor[train_indices],
                y_tensor[train_indices],
                td_tensor[train_indices],
                feature_metadata=feature_metadata,
                x_transform=x_transform,
                y_transform=y_transform,
            )

        if stage in ("fit", "validate"):
            # Create validation dataset
            val_indices = cache_metadata["split_indices"]["val"]
            self.val_dataset = TransformTensorDataset(
                x_tensor[val_indices],
                y_tensor[val_indices],
                td_tensor[val_indices],
                feature_metadata=feature_metadata,
                x_transform=x_transform,
                y_transform=y_transform,
            )

        if stage == "test":
            # Create test datasets grouped by unique lead times
            test_indices = cache_metadata["split_indices"]["test"]
            x_test = x_tensor[test_indices]
            y_test = y_tensor[test_indices]
            td_test = td_tensor[test_indices]
            unique_tds = torch.sort(torch.unique(td_test))[0]
            self.test_datasets = []
            for td in unique_tds:
                mask = td_test == td
                x_subset = x_test[mask]
                y_subset = y_test[mask]
                td_subset = td_test[mask]
                ds = TransformTensorDataset(
                    x_subset,
                    y_subset,
                    td_subset,
                    feature_metadata=feature_metadata,
                    x_transform=x_transform,
                    y_transform=y_transform,
                )
                self.test_datasets.append(ds)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            **self.dataloader_config.train,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            **self.dataloader_config.val,
        )

    def test_dataloader(self) -> list[DataLoader[Any]]:
        # Lightning implicitly uses a combined_loader here with mode="sequential"
        return [DataLoader(ds, **self.dataloader_config.test) for ds in self.test_datasets]

    def cleanup(self) -> None:
        """Clean up any temporary files.

        Note: Cached tensor and metadata files are kept for future use.
        They are reused when the same configuration is used again.
        """
        # The tensor and metadata files are now persistent cache files
        # and should not be deleted. They will be reused when the same
        # configuration is used again based on the config hash.
        pass
