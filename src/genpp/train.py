import warnings

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from genpp.configs import register_resolvers


@hydra.main(version_base=None, config_path="configs", config_name="DUMMY")
def train(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    register_resolvers()
    torch.set_float32_matmul_precision("medium")

    # Set seed for reproducibility
    if hasattr(cfg, "seed"):
        L.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data.module)
    try:
        datamodule.prepare_data()

        model = hydra.utils.instantiate(
            cfg.model, rescaler=datamodule.y_reverseModules if cfg.model.use_rescaler else None
        )
        model.compile()
        logger = None
        if hasattr(cfg, "logger") and cfg.logger:
            logger = hydra.utils.instantiate(cfg.logger)
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

        # Skip sanity checks since the callback needs to be fitted first and this would fail otherwise
        trainer = hydra.utils.instantiate(
            cfg.trainer, logger=logger, detect_anomaly=False, num_sanity_val_steps=0
        )

        trainer.fit(model, datamodule)
    finally:
        datamodule.cleanup()


if __name__ == "__main__":
    train()
