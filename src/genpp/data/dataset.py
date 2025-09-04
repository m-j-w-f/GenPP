import atexit
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from warnings import warn

import lightning as L
import xarray as xr
import xbatcher
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from xbatcher.loaders.torch import MapDataset, to_tensor

from genpp.data import (
    FORECAST_ENS_FLAT_AGG_NAME,
    FORECAST_ENS_NAME,
    OBSERVATIONS_FLAT_NAME,
    OBSERVATIONS_NAME,
    OUTPUT_DIR,
)
from genpp.preproc.preprocessors import Preprocessor


def _get_MapDataset(
    x_ds: xr.DataArray,
    y_ds: xr.DataArray,
    x_kwargs: dict,
    y_kwargs: dict,
    x_transform: Callable = to_tensor,
    y_transform: Callable = to_tensor,
) -> MapDataset:
    """Creates a MapDataset for the given xarray Datasets.

    Args:
        x_ds (xr.Dataset): dataset containing the input data.
        y_ds (xr.Dataset): dataset containing the target data.
        x_kwargs (dict): keyword arguments for the xbatcher.BatchGenerator for input data.
        y_kwargs (dict): keyword arguments for the xbatcher.BatchGenerator for target data.
        x_transform (Callable, optional): transform function for input data. Defaults to to_tensor.
        y_transform (Callable, optional): transform function for target data. Defaults to to_tensor.

    Returns:
        MapDataset: A PyTorch-compatible MapDataset that generates batches from the input datasets
               using xbatcher BatchGenerators with the specified transforms applied.
    """

    x_gen = xbatcher.BatchGenerator(
        x_ds,
        **x_kwargs,
    )
    y_gen = xbatcher.BatchGenerator(
        y_ds,
        **y_kwargs,
    )

    map_ds = MapDataset(
        X_generator=x_gen,
        y_generator=y_gen,
        transform=x_transform if x_transform is not None else to_tensor,
        target_transform=y_transform if y_transform is not None else to_tensor,
    )
    return map_ds


class WeatherBench2DataModule(L.LightningDataModule):
    """DataModule for WeatherBench2 dataset."""

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
        self.x_select_variables = x_select_variables
        self.y_select_variables = y_select_variables
        self.already_prepared = False
        atexit.register(self.cleanup)

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

    def prepare_data(self):
        # This method is called only on 1 GPU/TPU in distributed training
        # Use it to download data, if necessary
        # Do not assign states here, they will nit be kept
        if self.already_prepared:
            return

        # Is data already downloaded?
        if not os.path.exists(self.path / FORECAST_ENS_NAME):
            raise FileNotFoundError(
                f"Forecast ensemble data not found at {self.path / FORECAST_ENS_NAME}. "
                "Please download the dataset first. A script is provided in src/genpp/data/download.py."
            )

        if not os.path.exists(self.path / OBSERVATIONS_NAME):
            raise FileNotFoundError(
                f"Observations data not found at {self.path / OBSERVATIONS_NAME}. "
                "Please download the dataset first. A script is provided in src/genpp/data/download.py."
            )

        # Aggregate the ensemble members and cut out missing values if not done yet
        if (
            not (self.path / FORECAST_ENS_FLAT_AGG_NAME).exists()
            or not (self.path / OBSERVATIONS_FLAT_NAME).exists()
        ):
            warn(
                "Flattening and aggregation of ensemble members and observations is not done yet. "
                "This will take some time..."
            )
            from genpp.preproc.flat_and_aggr import main as preprocess_main

            preprocess_main(base_dir=self.path)

        # Preprocess the forecasts and save it to disk if necessary
        if self.x_preprocessing is not None or self.x_select_variables is not None:
            # Load the data, fit the preprocessors, and save the preprocessed data
            da = xr.open_dataarray(self.path / FORECAST_ENS_FLAT_AGG_NAME)
            da = self._select_variables(
                da, self.x_select_variables, append_suffix=True
            )  # Apply variable selection before preprocessing
            for preprocessor in self.x_preprocessing if self.x_preprocessing else []:
                preprocessor.fit(da.sel(prediction_time=self.dataset_config.train.slice))
                if not preprocessor.fit_only:
                    da = preprocessor.preprocess(da)

            self.x_tmp = tempfile.NamedTemporaryFile(
                dir=self.path / "tmp", suffix=".nc", delete=False
            )
            da.to_netcdf(
                self.x_tmp.name,
                mode="w",
                format="NETCDF4",
            )
            # put the inverse transforms in a list to that they can be accessed by the model later
            if self.x_preprocessing is not None:
                self.x_reverseModules = [
                    rm
                    for preprocessor in self.x_preprocessing
                    if (rm := preprocessor.get_reverse_module()) is not None
                ]

        if self.y_preprocessing is not None or self.y_select_variables is not None:
            # Load the data, fit the preprocessors, and save the preprocessed data
            da = xr.open_dataarray(self.path / OBSERVATIONS_FLAT_NAME)
            da = self._select_variables(
                da, self.y_select_variables, append_suffix=False
            )  # Apply variable selection before preprocessing
            for preprocessor in self.y_preprocessing if self.y_preprocessing else []:
                preprocessor.fit(da.sel(time=self.dataset_config.train.slice))
                if not preprocessor.fit_only:
                    da = preprocessor.preprocess(da)

            self.y_tmp = tempfile.NamedTemporaryFile(
                dir=self.path / "tmp", suffix=".nc", delete=False
            )
            da.to_netcdf(
                self.y_tmp.name,
                mode="w",
                format="NETCDF4",
            )
            if self.y_preprocessing is not None:
                self.y_reverseModules = [
                    rm
                    for preprocessor in self.y_preprocessing
                    if (rm := preprocessor.get_reverse_module()) is not None
                ]

        self.already_prepared = True

    def setup(self, stage: str) -> None:
        # Load preprocessed data if preprocessing was configured, otherwise load original flat data
        if self.x_preprocessing or self.x_select_variables:
            self.x = xr.open_dataarray(self.x_tmp.name)
        else:
            self.x = xr.open_dataarray(self.path / FORECAST_ENS_FLAT_AGG_NAME)
        if self.y_preprocessing or self.y_select_variables:
            self.y = xr.open_dataarray(self.y_tmp.name)
        else:
            self.y = xr.open_dataarray(self.path / OBSERVATIONS_FLAT_NAME)
        self.x_feature_names = self.x.feature.values
        self.y_feature_names = self.y.feature.values

        if stage == "fit":
            self.train_dataset = _get_MapDataset(
                self.x.sel(prediction_time=self.dataset_config.train.slice),
                self.y.sel(time=self.dataset_config.train.slice),
                x_kwargs=self.dataset_config.train.x_kwargs,
                y_kwargs=self.dataset_config.train.y_kwargs,
                x_transform=self.dataset_config.train.x_transform,
                y_transform=self.dataset_config.train.y_transform,
            )

        if stage in ("fit", "validate"):
            self.val_dataset = _get_MapDataset(
                self.x.sel(prediction_time=self.dataset_config.val.slice),
                self.y.sel(time=self.dataset_config.val.slice),
                x_kwargs=self.dataset_config.val.x_kwargs,
                y_kwargs=self.dataset_config.val.y_kwargs,
                x_transform=self.dataset_config.val.x_transform,
                y_transform=self.dataset_config.val.y_transform,
            )
        if stage == "test":
            self.test_dataset = _get_MapDataset(
                self.x.sel(prediction_time=self.dataset_config.test.slice),
                self.y.sel(time=self.dataset_config.test.slice),
                x_kwargs=self.dataset_config.test.x_kwargs,
                y_kwargs=self.dataset_config.test.y_kwargs,
                x_transform=self.dataset_config.test.x_transform,
                y_transform=self.dataset_config.test.y_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            **self.dataloader_config.train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            **self.dataloader_config.val,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            **self.dataloader_config.test,
        )

    def cleanup(self) -> None:
        if self.x_tmp and os.path.exists(self.x_tmp.name):
            self.x_tmp.close()
            os.remove(self.x_tmp.name)
        if self.y_tmp and os.path.exists(self.y_tmp.name):
            self.y_tmp.close()
            os.remove(self.y_tmp.name)

    def __del__(self):
        self.cleanup()
