import hydra
import pytorch_lightning as L
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    # Set seed for reproducibility
    if hasattr(cfg, "seed"):
        L.seed_everything(cfg.seed)

    # Instantiate model using Hydra
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate datamodule using Hydra
    datamodule = hydra.utils.instantiate(cfg.data.module)

    # TODO Instantiate logger using Hydra (if specified)
    logger = None
    if hasattr(cfg, "logger") and cfg.logger:
        logger = hydra.utils.instantiate(cfg.logger)

    # Instantiate trainer using Hydra
    trainer = L.Trainer(logger=logger, **cfg.trainer)

    # Train the model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
