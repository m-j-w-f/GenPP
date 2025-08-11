import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from genpp.configs import register_resolvers


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    register_resolvers()
    torch.set_float32_matmul_precision("medium")

    # Set seed for reproducibility
    if hasattr(cfg, "seed"):
        L.seed_everything(cfg.seed)

    model = hydra.utils.instantiate(cfg.model)
    model.compile()
    datamodule = hydra.utils.instantiate(cfg.data.module)

    logger = None
    if hasattr(cfg, "logger") and cfg.logger:
        logger = hydra.utils.instantiate(cfg.logger)

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
