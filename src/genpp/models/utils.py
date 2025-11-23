from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

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
    """Base Module for all Lightning Modules in GenPP.
    This class handles the optimizer and learning rate scheduler configuration.
    """

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
        """Return the scaling for a batch of lead times of shape [b, n_vars, 1, 1]"""
        pass


class LinearAbsTDScaling(BaseInternalTDScaling):
    def __init__(self) -> None:
        """Module used for scaling the predicted noise of the model so that the model has to only learn one scale.
        The linear model utilizes a linear regression of abs(err) ~ lead time.
        """
        super().__init__()

    def fit(self, model: torch.nn.Module) -> None:
        train_loader = model.trainer.datamodule.train_dataloader()  # type: ignore
        if train_loader is None:
            raise ValueError("Training dataloader is not available.")

        crop_layer = model.crop if hasattr(model, "crop") else None  # type: ignore
        A = None
        b = None

        with torch.no_grad():
            pbar = tqdm(train_loader, desc="Fitting abs(err)~TD")
            for batch in pbar:
                nwp, obs, td = batch["x"], batch["y"], batch["timedelta"]
                nwp = nwp["predicted_vars"]  # We only need these vars

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

                diff = (obs - nwp).abs()

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
            - The scale is an estimate of E[|err|] per variable.
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


@dataclass
class AbsStats:
    sum_abs: torch.Tensor
    count: int = 0


@dataclass
class StdStats:
    mean: torch.Tensor
    M2: torch.Tensor
    count: int = 0


class FixedTDScaling(LinearAbsTDScaling):
    def __init__(self, mode: str, n_vars: int = 2, n_leadtimes: int = 5) -> None:
        """Per-timedelta lookup table for absolute error or std scaling."""
        super().__init__()
        valid_modes = {"abs", "std"}
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got '{mode}'.")
        self.mode = mode
        self.n_vars: int | None = None
        self.register_buffer("lead_times", torch.zeros(n_leadtimes))
        self.register_buffer("lookup_table", torch.zeros((n_leadtimes, n_vars)))

    def fit(self, model: torch.nn.Module) -> None:
        train_loader = model.trainer.datamodule.train_dataloader()  # type: ignore
        if train_loader is None:
            raise ValueError("Training dataloader is not available.")

        crop_layer = model.crop if hasattr(model, "crop") else None  # type: ignore
        stats: dict[float, AbsStats | StdStats] = {}

        with torch.no_grad():
            pbar = tqdm(train_loader, desc="Fitting fixed TD scaling")
            for batch in pbar:
                nwp, obs, td = batch["x"], batch["y"], batch["timedelta"]
                nwp = nwp["predicted_vars"]

                if crop_layer is not None:
                    nwp = crop_layer(nwp)  # type: ignore
                if nwp.shape != obs.shape:
                    obs = crop_layer(obs)  # type: ignore

                diff = obs - nwp
                if self.n_vars is None:
                    self.n_vars = diff.shape[1]

                unique_td = torch.unique(td)
                for lead in unique_td:
                    mask = td == lead
                    if not mask.any():
                        continue

                    lead_diff = diff[mask]
                    if lead_diff.numel() == 0:
                        continue

                    lead_key = self._td_to_key(lead)
                    if lead_key not in stats:
                        stats[lead_key] = self._init_stats(lead_diff)

                    self._update_stats(stats[lead_key], lead_diff)

        if not stats:
            raise ValueError("No timedelta values were observed during fitting.")

        sorted_leads = sorted(stats.keys())
        lookup_rows = [self._finalize_stats(stats[lead]) for lead in sorted_leads]
        lookup_tensor = torch.stack(lookup_rows, dim=0)
        self.lead_times = torch.tensor(
            sorted_leads,
            dtype=lookup_tensor.dtype,
            device=lookup_tensor.device,
        )
        self.lookup_table = lookup_tensor
        self.is_fitted = True

    def get_scale(self, td: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise ValueError("TD Scaling is not fitted yet.")
        if self.n_vars is None:
            raise ValueError("Number of variables is unknown. Did you call fit()?")

        if self.lead_times.numel() == 0 or self.lookup_table.numel() == 0:
            raise ValueError("Lookup table is empty. Did you call fit()?")

        lead_times = self.lead_times.to(td)
        lookup = self.lookup_table.to(td)
        distances = torch.abs(lead_times.unsqueeze(0) - td.unsqueeze(1))
        indices = torch.argmin(distances, dim=1)
        scales = lookup[indices]
        scales = rearrange(scales, "b c -> b c 1 1")
        return scales

    def _td_to_key(self, value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    def _init_stats(self, reference: torch.Tensor) -> AbsStats | StdStats:
        if self.n_vars is None:
            raise ValueError("Number of variables must be set before initializing stats.")
        device = reference.device
        dtype = reference.dtype
        if self.mode == "abs":
            zeros = torch.zeros(self.n_vars, device=device, dtype=dtype)
            return AbsStats(sum_abs=zeros, count=0)

        mean = torch.zeros(self.n_vars, device=device, dtype=dtype)
        M2 = torch.zeros(self.n_vars, device=device, dtype=dtype)
        return StdStats(mean=mean, M2=M2, count=0)

    def _update_stats(self, stat: AbsStats | StdStats, lead_diff: torch.Tensor) -> None:
        if isinstance(stat, AbsStats):
            abs_vals = lead_diff.abs()
            stat.sum_abs = stat.sum_abs + abs_vals.sum(dim=(0, 2, 3))
            batch_count = lead_diff.shape[0] * lead_diff.shape[2] * lead_diff.shape[3]
            stat.count += batch_count
            return

        flat = rearrange(lead_diff, "b c h w -> c (b h w)")
        batch_count = flat.shape[1]
        if batch_count == 0:
            return

        batch_mean = flat.mean(dim=1)
        batch_M2 = ((flat - batch_mean[:, None]) ** 2).sum(dim=1)

        if stat.count == 0:
            stat.mean = batch_mean
            stat.M2 = batch_M2
            stat.count = batch_count
            return

        total_count = stat.count + batch_count
        delta = batch_mean - stat.mean
        stat.mean = stat.mean + delta * (batch_count / total_count)
        stat.M2 = stat.M2 + batch_M2 + (delta**2) * stat.count * batch_count / total_count
        stat.count = total_count

    def _finalize_stats(self, stat: AbsStats | StdStats) -> torch.Tensor:
        if stat.count == 0:
            raise ValueError("Encountered timedelta with zero samples during fitting.")
        if isinstance(stat, AbsStats):
            return stat.sum_abs / stat.count

        variance = stat.M2 / max(stat.count, 1)
        variance = torch.clamp(variance, min=1e-12)
        return torch.sqrt(variance)


class LearnedTDScaling(BaseInternalTDScaling):
    def __init__(self, n_vars: int = 2) -> None:
        """Module used for scaling the predicted noise of the model so that the model has to only learn one scale.
        The learned model utilizes a small neural network that takes the lead time as input and outputs
        a scale per variable.

        Args:
            n_vars (int): Number of variables to scale.
        """
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_vars),  # Output one scale per time delta per variable
            torch.nn.Softplus(),  # Ensure positivity
        )
        self.is_fitted = True  # No fitting needed for learned scaling

    def fit(self, model: L.LightningModule) -> None:
        # No fitting needed for learned scaling
        pass

    def get_scale(self, td: torch.Tensor) -> torch.Tensor:
        """Return the per-sample, per-variable scale for a batch of lead times.

        Args:
            td (torch.Tensor): 1D tensor of lead times with shape [batch]. Values must be float.
        """
        td = rearrange(td, "b -> b 1")  # Shape [b, 1]
        scale = self.model(td)  # Shape [b, n_vars]
        scale = rearrange(scale, "b c -> b c 1 1")  # Shape [b, n_vars, 1, 1]
        return scale
