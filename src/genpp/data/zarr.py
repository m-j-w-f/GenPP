import pytorch_lightning as pl
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


class ZarrDataset(Dataset):
    def __init__(self, zarr_file, transform=None):
        self.zarr_file = zarr.open(zarr_file, mode="r")
        self.transform = transform
        self.num_samples = self.zarr_file.shape[0]  # Assuming the first dimension is samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.zarr_file[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32)


class ZarrDataModule(pl.LightningDataModule):
    def __init__(self, zarr_file, batch_size: int, transform, dataloader_kwargs: dict):
        super().__init__()
        self.zarr_file = zarr_file
        self.batch_size = batch_size
        self.transform = transform
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage=None):
        self.dataset = ZarrDataset(self.zarr_file, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)
