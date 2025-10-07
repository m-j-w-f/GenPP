import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.data import FORECAST_ENS_FLAT_AGG_PATH, OBSERVATIONS_FLAT_PATH
from genpp.data.fast_dataset_simple import FastWeatherBench2DataModule

CONFIG_DIR = BASE_DIR / "configs"

register_resolvers()


def _data_ready() -> bool:
    return FORECAST_ENS_FLAT_AGG_PATH.exists() and OBSERVATIONS_FLAT_PATH.exists()


@pytest.mark.integration
@pytest.mark.skipif(not _data_ready(), reason="WeatherBench2 preprocessed data not available")
def test_fast_weatherbench2_train_cycle():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base_drn")

    datamodule: FastWeatherBench2DataModule = instantiate(cfg.data.module)

    try:
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        batch_idx = None
        for batch_idx, batch in enumerate(loader):
            x, y, dt = batch
            if batch_idx == 0:
                print(x.shape, y.shape, dt.shape)
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert isinstance(dt, torch.Tensor)
            assert x.ndim == 4
            assert y.ndim == 4
            assert dt.ndim == 1
            # all dt values should be in [0, 1]
            expected = torch.ones_like(x, dtype=torch.bool)
            torch.testing.assert_close(dt <= 1.0, expected)
            torch.testing.assert_close(dt >= 0.0, expected)
        print(f"Total batches in train dataloader: {batch_idx + 1 if batch_idx is not None else 0}")
        if batch_idx is None:
            pytest.fail("Train dataloader did not yield any batches")
    finally:
        datamodule.cleanup()
