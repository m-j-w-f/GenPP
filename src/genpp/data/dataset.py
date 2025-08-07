import os
from pathlib import Path
from typing import Callable, List
from warnings import warn

import lightning as L
import xarray as xr
import xbatcher
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from xbatcher.loaders.torch import MapDataset, to_tensor

from genpp.data import (
    FORECAST_ENS_FLAT_AGG_NAME,
    FORECAST_ENS_FLAT_AGG_PREPROC_NAME,
    FORECAST_ENS_NAME,
    OBSERVATIONS_FLAT_NAME,
    OBSERVATIONS_NAME,
    OUTPUT_DIR,
)
from genpp.preproc.preprocessors import Preprocessor


def get_MapDataset(
    x_ds: xr.DataArray,
    y_ds: xr.DataArray,
    x_kwargs: dict,
    y_kwargs: dict,
    batch_size: int = 8,
    x_transform: Callable = to_tensor,
    y_transform: Callable = to_tensor,
) -> MapDataset:
    """Creates a MapDataset for the given xarray Datasets.

    Args:
        x_ds (xr.Dataset): dataset containing the input data.
        y_ds (xr.Dataset): dataset containing the target data.
        x_kwargs (dict): keyword arguments for the xbatcher.BatchGenerator for input data.
        y_kwargs (dict): keyword arguments for the xbatcher.BatchGenerator for target data.
        batch_size (int, optional): number of samples per batch. Defaults to 8.
        x_transform (Callable, optional): transform function for input data. Defaults to to_tensor.
        y_transform (Callable, optional): transform function for target data. Defaults to to_tensor.

    Returns:
        MapDataset: A PyTorch-compatible MapDataset that generates batches from the input datasets
               using xbatcher BatchGenerators with the specified transforms applied.
    """
    x_kwargs["input_dims"]["prediction_time"] = batch_size
    y_kwargs["input_dims"]["time"] = batch_size

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
        transform=x_transform,
        target_transform=y_transform,
    )
    return map_ds


class WeatherBench2DataModule(L.LightningDataModule):
    """DataModule for WeatherBench2 dataset."""

    def __init__(
        self,
        dataset_config: DictConfig,
        dataloader_config: DictConfig,
        save_dir: Path = OUTPUT_DIR,
        preprocssing: List[Preprocessor] | None = None,
    ) -> None:
        super().__init__()
        self.path = save_dir
        self.preprocssing = preprocssing
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config

    def prepare_data(self):
        # This method is called only on 1 GPU/TPU in distributed training
        # Use it to download data, if necessary
        # Do not assign states here, they will nit be kept

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
        # TODO: if we often change the preprocessing, we might want to use a tempfile here and delete it in teardown after training
        if self.preprocssing:
            if not (self.path / FORECAST_ENS_FLAT_AGG_PREPROC_NAME).exists():
                warn(
                    f"Preprocessing not done yet for {self.path / FORECAST_ENS_FLAT_AGG_PREPROC_NAME}. "
                    "This will take some time..."
                )
                # Load the data, fit the preprocessors, and save the preprocessed data
                da = xr.open_dataarray(self.path / FORECAST_ENS_FLAT_AGG_NAME)
                for preprocessor in self.preprocssing:
                    preprocessor.fit(da)
                    da = preprocessor.preprocess(da)
                da.to_netcdf(
                    self.path / FORECAST_ENS_FLAT_AGG_PREPROC_NAME,
                    mode="w",
                    format="NETCDF4",
                )

    def setup(self, stage: str):
        x = xr.open_dataarray(self.path / FORECAST_ENS_FLAT_AGG_PREPROC_NAME)
        y = xr.open_dataarray(self.path / OBSERVATIONS_FLAT_NAME)

        if stage == "fit":
            self.train_dataset = get_MapDataset(
                x.sel(time=self.dataset_config.train.slice),
                y.sel(time=self.dataset_config.train.slice),
                x_kwargs=self.dataset_config.train.x_kwargs,
                y_kwargs=self.dataset_config.train.y_kwargs,
                batch_size=self.dataset_config.train.batch_size,
            )
            self.val_dataset = get_MapDataset(
                x.sel(time=self.dataset_config.val.slice),
                y.sel(time=self.dataset_config.val.slice),
                x_kwargs=self.dataset_config.val.x_kwargs,
                y_kwargs=self.dataset_config.val.y_kwargs,
                batch_size=self.dataset_config.val.batch_size,
            )
        if stage == "test":
            self.test_dataset = get_MapDataset(
                x.sel(time=self.dataset_config.test.slice),
                y.sel(time=self.dataset_config.test.slice),
                x_kwargs=self.dataset_config.test.x_kwargs,
                y_kwargs=self.dataset_config.test.y_kwargs,
                batch_size=self.dataset_config.test.batch_size,
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
