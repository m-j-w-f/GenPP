import hydra
import pytorch_lightning as L
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    # Initialize W&B with Hydra config
    wandb.init(
        project="genpp-models",
        config=cfg,  # type: ignore
        tags=[cfg.model._target_.split(".")[-1]],  # e.g., "FcChenModel"
    )

    # Instantiate model using Hydra
    model = hydra.utils.instantiate(cfg.model)
    model = model.compile()

    # Instantiate datamodule using Hydra
    datamodule = hydra.utils.instantiate(cfg.data)

    # Use W&B logger
    logger = WandbLogger()

    trainer = L.Trainer(logger=logger, **cfg.trainer)

    # Train the model
    trainer.fit(model, datamodule)

    print(f"Training {cfg.model._target_} with config:")
    print(f"  - Model type: {cfg.model._target_.split('.')[-1]}")
    if hasattr(cfg.model, "hidden_dim_std"):
        print(
            f"  - Hidden dims: std={cfg.model.hidden_dim_std}, decoder={cfg.model.hidden_dim_decoder}"
        )
    if hasattr(cfg.model, "padding"):
        print(f"  - Padding: {cfg.model.padding}")
    print(f"  - Noise dim: {cfg.model.noise_dim}")
    print(f"  - Learning rate: {cfg.model.lr}")
    print(f"  - Loss function: {cfg.model.loss_fn._target_}")
    print(f"  - Data splits: train={cfg.data.dataset_config.train.slice}")
    print(f"  - Batch size: {cfg.data.dataset_config.train.batch_size}")
    print(
        f"  - Spatial dims: {cfg.data.dataset_config.train.x_kwargs.input_dims.latitude}x{cfg.data.dataset_config.train.x_kwargs.input_dims.longitude}"
    )

    # Model info
    print(f"Model instantiated successfully: {type(model).__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    train()
