import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.data import FORECAST_ENS_FLAT_AGG_PATH, OBSERVATIONS_FLAT_PATH

CONFIG_DIR = BASE_DIR / "configs"

register_resolvers()


def _data_ready() -> bool:
    return FORECAST_ENS_FLAT_AGG_PATH.exists() and OBSERVATIONS_FLAT_PATH.exists()


@pytest.mark.integration
@pytest.mark.skipif(not _data_ready(), reason="WeatherBench2 preprocessed data not available")
def test_fast_weatherbench2_train_cycle():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base_drn")

    datamodule = instantiate(cfg.data.module)

    try:
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        for batch_idx, (x, y) in enumerate(loader):
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert x.ndim >= 3
            assert y.ndim >= 3
            if batch_idx >= 1:
                break
        else:
            pytest.fail("Train dataloader did not yield any batches")
    finally:
        datamodule.cleanup()
