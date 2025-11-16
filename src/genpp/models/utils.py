from abc import ABC, abstractmethod
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


class BaseInternalTDScaling(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.is_fitted = False

    @abstractmethod
    def fit(self, model: L.LightningModule) -> None:
        """Fit the TD Scaling.

        Args:
            model (L.LightningModule): The outer model to fit the scaling for.
            This is needed to access the training data loader.
        """
        pass

    @abstractmethod
    def get_scale(self, td: torch.Tensor) -> torch.Tensor:
        pass


class LinearTDScaling(BaseInternalTDScaling):
    def __init__(self, mode: str) -> None:
        """Module used for scaling the predicted noise of the model so that the model has to only learn one scale.
        The linear model utilizes a linear regression of
        abs(err) ~ lead time in case of mode == "abs" and
        std(err) ~ lead time in case of mode == "std".

        Args:
            mode (str): The mode of scaling, either "abs" or "std".
        """
        super().__init__()
        if mode not in ["abs", "std"]:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def fit(self, model: torch.nn.Module) -> None:
        train_loader = model.trainer.datamodule.train_dataloader()  # type: ignore
        if train_loader is None:
            raise ValueError("Training dataloader is not available.")

        crop_layer = model.crop if hasattr(model, "crop") else None  # type: ignore
        A = None
        b = None

        with torch.no_grad():
            regressand = "abs(err)" if self.mode == "abs" else "std(err)"
            pbar = tqdm(train_loader, desc=f"Fitting {regressand}~TD")
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
                    nwp = crop_layer(nwp)  # type: ignore

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

                if self.mode == "abs":
                    diff = (obs - nwp).abs()
                elif self.mode == "std":
                    diff = (obs - nwp).std(dim=1)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

                ones = torch.ones_like(diff)
                X = torch.stack([ones, td], dim=1)  # [n_vars, 2, n_samples]
                y = diff

                # The rearrange is the same as calling the transpose on the last two dims if X
                A += torch.bmm(X, rearrange(X, "c a b -> c b a"))
                b += torch.bmm(X, y.unsqueeze(-1)).squeeze(-1)

        betas = torch.linalg.solve(A, b)  # Shape [n_vars, 2]
        # The first dimension is the variable dimension
        # The second dimension is the intercept and slope
        self.intercepts = betas[:, 0]
        self.slopes = betas[:, 1]
        self.is_fitted = True

    def get_scale(self, td: torch.Tensor) -> torch.Tensor:
        """Return the per-sample, per-variable scale for a batch of lead times.

        The scale is computed with the linear model fitted in `fit`:
            scale = intercept + slope * td
        where intercept and slope are learned independently for each variable.

        Args:
            td (torch.Tensor): 1D tensor of lead times with shape [batch]. Values must be float.

        Raises:
            ValueError: If the scaling model has not been fitted via `fit`.

        Returns:
            torch.Tensor: A tensor with shape [batch, n_vars, 1, 1]. The scale is broadcastable to
                match model prediction shapes (batch, n_vars, height, width). The returned tensor
                is placed on the same device as `td`.

        Notes:
            - If the instance was fitted in mode "abs", the scale is an estimate of E[|err|] per variable.
            - If the instance was fitted in mode "std", the scale is an estimate of std(err) per variable.
        """
        if not self.is_fitted:
            raise ValueError("TD Scaling is not fitted yet.")
        # Ensure betas is on the same device as td
        self.intercepts = self.intercepts.to(td)  # Shape [n_vars]
        self.slopes = self.slopes.to(td)  # Shape [n_vars]
        scale = rearrange(self.intercepts, "c -> 1 c") + rearrange(
            self.slopes, "c -> 1 c"
        ) * rearrange(td, "b -> b 1")  # Shape [b, n_vars]
        scale = rearrange(scale, "b c -> b c 1 1")  # Shape [b, n_vars, 1, 1]
        return scale
