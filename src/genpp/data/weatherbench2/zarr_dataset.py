"""Data module for loading Zarr datasets used in autoencoder training.

This dataloader is primarily used for the autoencoder training.
There is no functionality to pre-process the data and there is no plan in adding this.
"""

from pathlib import Path
from collections.abc import Callable

import lightning as L
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from genpp import BASE_DIR


class ZarrDataModule(L.LightningDataModule):
    """Lightning data module for loading Zarr datasets.

    Args:
        split (dict[str, slice]): Dictionary with train/val/test splits as slices.
        x_select_variables (list[str]): List of variable names to select from the dataset.
        dataloader_kwargs: Keyword arguments for the dataloaders.
        zarr_path (Path): Path to the Zarr dataset.
    """

    def __init__(
        self,
        split: dict[str, slice],
        x_select_variables: list[str],
        dataloader_kwargs,
        zarr_path: Path = BASE_DIR / "data" / "weatherbench2" / "ifs_ens.zarr",
    ):
        super().__init__()
        ds: xr.Dataset = xr.open_zarr(zarr_path, consolidated=True)
        self.features: list[str] = (
            list(x_select_variables)
            if not isinstance(x_select_variables, list)
            else x_select_variables
        )
        ds = ds[self.features]
        self.ds: xr.DataArray = (
            ds.stack(sample=("time", "prediction_timedelta", "number"))
            .to_dataarray("feature")
            .transpose("sample", "feature", "longitude", "latitude")
        )
        # Filter out all samples with NaNs
        valid_samples = ~self.ds.isnull().any(dim=("feature", "longitude", "latitude"))
        self.ds = self.ds.sel(sample=valid_samples.compute())
        self.ds = self.ds.unstack("sample")

        self.split = split
        self.dataloader_kwargs = dataloader_kwargs

    def _dataset_to_tensor_dataset(self, dataset: xr.DataArray) -> TensorDataset:
        """Convert an xarray DataArray to a PyTorch TensorDataset.

        Args:
            dataset (xr.DataArray): The input xarray DataArray.

        Returns:
            TensorDataset: The converted TensorDataset.
        """
        dataset = dataset.stack(sample=("time", "prediction_timedelta", "number")).transpose(
            "sample", "feature", "longitude", "latitude"
        )
        t = torch.tensor(dataset.values)
        return TensorDataset(t)

    def setup(self, stage=None):
        """Set up the datasets for training, validation, and testing.

        Args:
            stage (str): The stage to set up ('fit', 'validate', 'test', or None for all).
        """
        if stage == "fit" or stage is None:
            # TODO check if this time split works correctly
            # It does but we loose coords information
            # BUG this does not work correctly
            # FIX: could unstack select and restack?
            train_dataset = self.ds.sel(time=self.split["train"])
            self.train_tensor_dataset = self._dataset_to_tensor_dataset(train_dataset)
        if stage == "validate" or stage == "fit" or stage is None:
            val_dataset = self.ds.sel(time=self.split["val"])
            self.val_tensor_dataset = self._dataset_to_tensor_dataset(val_dataset)
        if stage == "test" or stage is None:
            test_dataset = self.ds.sel(time=self.split["test"])
            self.test_tensor_dataset = self._dataset_to_tensor_dataset(test_dataset)

    def train_dataloader(self):
        """Create the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(self.train_tensor_dataset, **self.dataloader_kwargs.train)

    def val_dataloader(self):
        """Create the validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(self.val_tensor_dataset, **self.dataloader_kwargs.val)

    def test_dataloader(self):
        """Create the test data loader.

        Returns:
            DataLoader: The test data loader.
        """
        return DataLoader(self.test_tensor_dataset, **self.dataloader_kwargs.test)

    def cleanup(self) -> None:
        """Clean up resources after training."""
        pass


class ZarrClassificationDataModule(ZarrDataModule):
    def _dataset_to_tensor_dataset(self, dataset: xr.DataArray) -> TensorDataset:
        """Convert an xarray DataArray to a PyTorch TensorDataset for classification.

        Creates balanced dataset with two classes:
        - Class 0: Single member forecast (randomly selected from ensemble)
        - Class 1: Mean forecast across ensemble members

        Args:
            dataset (xr.DataArray): The input xarray DataArray with dims (time, prediction_timedelta, number, feature, longitude, latitude).

        Returns:
            TensorDataset: The converted TensorDataset with (x, y) pairs.
        """
        members = len(dataset.number)

        # Class 1: Mean across number dimension (vectorized)
        mean_forecasts = (
            dataset.mean(dim="number")
            .stack(sample=("time", "prediction_timedelta"))
            .transpose("sample", ...)
        )  # (sample, feature, longitude, latitude)
        mean_tensor = torch.tensor(mean_forecasts.values)

        # Class 0: Randomly select single members (vectorized)
        dataset = dataset.stack(sample=("time", "prediction_timedelta"))
        n_samples = len(dataset.sample)
        random_indices = torch.randint(0, members, (n_samples,))
        # Select the random members for each (time, prediction_timedelta) pair
        single_member_data = dataset.isel(
            number=xr.DataArray(
                random_indices.numpy(),
                dims=["sample"],
            )
        ).transpose("sample", ...)
        single_tensor = torch.tensor(single_member_data.values)

        x_tensor = torch.cat([mean_tensor, single_tensor], dim=0)

        # Create labels: [1,...,1, 0,...,0] (1 for mean, 0 for single member)
        y_tensor = torch.tensor([1] * n_samples + [0] * n_samples, dtype=torch.long)

        return TensorDataset(x_tensor, y_tensor)


class TransformTensorDataset(TensorDataset):
    """
    Extension of torch.utils.data.TensorDataset that adds transform support.
    Designed for unlabeled image data.

    Args:
        *tensors (torch.Tensor): Tensors of shape (N, C, H, W) where N is number of samples,
                  C is channels, H is height, W is width.
        transform (Callable | None): Optional transform to be applied on a sample.
    """

    def __init__(self, *tensors: torch.Tensor, transform: Callable | None = None):
        """
        Initialize the dataset.

        Args:
            *tensors (torch.Tensor): Image data as torch.Tensor
            transform (Callable | None): Optional transform/augmentation function
        """
        # Initialize parent TensorDataset with the data
        super().__init__(*tensors)

        if all(tensor.dim() != 4 for tensor in self.tensors):
            raise ValueError("Expected only 4D tensors (N, C, H, W)")

        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple[torch.Tensor, ...]: Transformed image tensor
        """
        # Get the image from parent class
        # TensorDataset returns a tuple
        images: tuple[torch.Tensor, ...] = super().__getitem__(idx)

        # Apply transform if available
        if self.transform is not None:
            return tuple(self.transform(i) for i in images)

        return images
