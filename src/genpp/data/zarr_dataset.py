"""
This dataloader is primarily used for the autoencoder training.
There is no functionality to pre-process the data and there is no plan in adding this.
"""

from pathlib import Path

import lightning as L
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from genpp import BASE_DIR


class ZarrDataModule(L.LightningDataModule):
    def __init__(
        self,
        split: dict[str, slice],
        x_select_variables: list[str],
        dataloader_kwargs,
        zarr_path: Path = BASE_DIR / "data" / "weatherbench2" / "ifs_ens.zarr",
    ):
        super().__init__()
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.features = (
            list(x_select_variables)
            if not isinstance(x_select_variables, list)
            else x_select_variables
        )
        self.ds = self.ds[self.features]
        self.ds = (
            self.ds.stack(sample=("time", "prediction_timedelta", "number"))
            .to_dataarray("feature")
            .transpose("sample", "feature", "longitude", "latitude")
        )
        # Filter out all samples with NaNs
        valid_samples = ~self.ds.isnull().any(dim=("feature", "longitude", "latitude"))
        self.ds = self.ds.sel(sample=valid_samples.compute())

        self.split = split
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.ds.sel(time=self.split["train"])
        if stage == "validate" or stage == "fit" or stage is None:
            self.val_dataset = self.ds.sel(time=self.split["val"])
        if stage == "test" or stage is None:
            self.test_dataset = self.ds.sel(time=self.split["test"])

    def train_dataloader(self):
        t = torch.tensor(self.train_dataset.values)
        dataset = TensorDataset(t)
        return DataLoader(dataset, **self.dataloader_kwargs.train)

    def val_dataloader(self):
        t = torch.tensor(self.val_dataset.values)
        dataset = TensorDataset(t)
        return DataLoader(dataset, **self.dataloader_kwargs.val)

    def test_dataloader(self):
        t = torch.tensor(self.test_dataset.values)
        dataset = TensorDataset(t)
        return DataLoader(dataset, **self.dataloader_kwargs.test)

    def cleanup(self) -> None:
        pass
