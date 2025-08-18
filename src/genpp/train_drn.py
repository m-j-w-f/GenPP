import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from genpp.configs import register_resolvers
from genpp.models.layers import ReverseMinMaxScaling, ReverseStandardization


@hydra.main(version_base=None, config_path="configs", config_name="base_drn")
def train(cfg: DictConfig) -> None:
    register_resolvers()
    torch.set_float32_matmul_precision("medium")

    # Set seed for reproducibility
    if hasattr(cfg, "seed"):
        L.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data.module)
    try:
        datamodule.prepare_data()

        rs = ReverseStandardization(
            datamodule.y_preprocessing[0].mean_tensor, datamodule.y_preprocessing[0].std_tensor
        )
        rmm = ReverseMinMaxScaling(
            datamodule.y_preprocessing[1].min_tensor, datamodule.y_preprocessing[1].max_tensor
        )
        model = hydra.utils.instantiate(cfg.model, rescalers=[rs, rmm])
        model.compile()
        logger = None
        if hasattr(cfg, "logger") and cfg.logger:
            logger = hydra.utils.instantiate(cfg.logger)
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, detect_anomaly=True)

        trainer.fit(model, datamodule)
    finally:
        datamodule.cleanup()


if __name__ == "__main__":
    train()
