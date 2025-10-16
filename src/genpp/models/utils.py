from abc import ABC
from collections.abc import Callable

import lightning as L
import torch
from einops import rearrange
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from tqdm import tqdm


def _instantiate_partial_scheduler(
    partial_scheduler: DictConfig, optimizer: torch.optim.Optimizer
) -> DictConfig:
    # This is ugly but works because the lr_scheduler_partial is a DictConfig
    if (
        partial_scheduler.scheduler.func is not torch.optim.lr_scheduler.ChainedScheduler
    ):  # Just a single scheduler
        partial_scheduler.scheduler = partial_scheduler.scheduler(optimizer)
    else:  # We need to instantiate the chained scheduler with the optimizer
        # It gets even uglier
        inner_schedulers = [
            p(optimizer) for p in partial_scheduler.scheduler.keywords["schedulers"]
        ]
        # Overwrite the inner schedulers with the instantiated ones by calling the func directly
        partial_scheduler.scheduler = partial_scheduler.scheduler.func(
            *partial_scheduler.scheduler.args,
            schedulers=inner_schedulers,
            **{k: v for k, v in partial_scheduler.scheduler.keywords.items() if k != "schedulers"},
        )

    return partial_scheduler


class BaseModule(L.LightningModule, ABC):
    def __init__(
        self, optimizer: Callable[..., torch.optim.Optimizer], lr_scheduler: DictConfig
    ) -> None:
        super().__init__()
        self.optimizer_partial = optimizer
        self.lr_scheduler_partial = lr_scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Instantiate the optimizer and scheduler from the config
        self.optimizer = self.optimizer_partial(self.parameters())
        self.lr_scheduler_partial = _instantiate_partial_scheduler(
            self.lr_scheduler_partial, self.optimizer
        )

        return {  # type: ignore
            "optimizer": self.optimizer,
            "lr_scheduler": {**self.lr_scheduler_partial},
        }


class FitScaleVarianceTDMixin:
    """Mixin class to add scale_variance_td fitting functionality to LightningModules."""

    def _fit_scale_variance_td(self) -> None:
        """Fit a regression of the form absolute_prediction_error ~ LeadTime for each variable.
        The regression coefficients are stored as a buffer called 'scale_variance_td'.
        """
        # If already fitted, skip
        if (
            hasattr(self, "scale_variance_td") and self.scale_variance_td is not None  # type: ignore
        ):
            print("scale_variance_td already fitted, skipping.")
            return

        # Access the training dataloader
        train_loader = self.trainer.datamodule.train_dataloader()  # type: ignore
        if train_loader is None:
            raise ValueError("Training dataloader is not available.")

        crop_layer = self.crop if hasattr(self, "crop") else None  # type: ignore
        A = None
        b = None

        with torch.no_grad():
            pbar = tqdm(train_loader, desc="Fitting scale_variance~TD")
            for batch in pbar:
                nwp, obs, td = batch["x"], batch["y"], batch["timedelta"]
                # The NWP vars we need are the first n_vars
                nwp = nwp["predicted_vars"]

                # Initialize A and b
                if A is None:
                    A = torch.zeros((nwp.shape[1], 2, 2)).to(nwp)
                if b is None:
                    b = torch.zeros((nwp.shape[1], 2)).to(nwp)

                if crop_layer is not None:
                    nwp = crop_layer(nwp)

                # If shapes don't match, crop obs as well
                if nwp.shape != obs.shape:
                    obs = crop_layer(obs)  # type: ignore

                # Now nwp and obs should have the same shape and no padding
                # We can run the regression now
                td = rearrange(td, "b -> b 1 1 1")
                td = td.expand_as(obs)
                td = rearrange(td, "b c h w -> c (b h w)")
                nwp = rearrange(nwp, "b c h w -> c (b h w)")
                obs = rearrange(obs, "b c h w -> c (b h w)")
                diff = (obs - nwp).abs()
                ones = torch.ones_like(diff)
                X = torch.stack([ones, td], dim=1)
                y = diff

                # The rearrange is the same as calling the transpose on the last two dims if X
                A += torch.bmm(X, rearrange(X, "c a b -> c b a"))
                b += torch.bmm(X, y.unsqueeze(-1)).squeeze(-1)

        betas = torch.linalg.solve(A, b)
        # The first dimension is the variable dimension
        # The second dimension is the intercept and slope
        self.register_buffer("scale_variance_td", betas)  # Shape [n_vars, 2] # type: ignore
