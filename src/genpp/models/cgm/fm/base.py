from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from omegaconf import DictConfig

from genpp.models.cgm.utils import BaseGenerativeModule
from genpp.models.cgm.utils.td_scaling import InternalTDScalingMixin
from genpp.models.layers import CropND


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c, h, w]
            t (torch.Tensor): [bs, 1, 1, 1]
            conditioning (dict[str, torch.Tensor]): [bs,...] previously refered to as 'y' in the paper
            however this was confusing as the ground truth forecast is also refered to as 'y'
        Returns:
            torch.Tensor: u_t^theta(x|conditioning) [bs, c, h, w]
        """
        pass


class BaseFlowMatchingModel(BaseGenerativeModule):
    """
    Base Flow Matching Model that works with different backbone architectures.

    NOTE that in this class the naming convention is different than in the other classes:
    - x_1 is the target (i.e. the ground truth forecasts, for which we want to generate samples that are similar to)
    - the nwp forecast is the conditioning
    - td is the lead time (between 0 and 1) for which the prediction is made

    This is an abstract base class; use FlowMatchingNoiseModel or FlowMatchingDirectModel instead.
    """

    def __init__(
        self,
        backbone: ConditionalVectorField,
        n_samples: int,
        solver_iter: int,
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ):
        """Initialize the Flow Matching Model.

        Args:
            backbone (ConditionalVectorField): The neural network backbone that parameterizes the
                conditional vector field u_t^theta(x|conditioning).
            n_samples (int): Number of ensemble samples to generate during prediction.
            solver_iter (int): Number of ODE solver iterations. The step size is computed as 1/solver_iter.
            padding (Sequence[int]): Padding values to crop from the output. If empty, no cropping is applied.
            optimizer (Callable[..., torch.optim.Optimizer]): Factory function to create the optimizer.
            lr_scheduler (DictConfig): Configuration for the learning rate scheduler.
            use_rescaler (bool): Whether to use rescaling modules for the outputs.
            rescaler (Sequence[nn.Module  |  None] | nn.Module | None, optional): Rescaling module(s) to
                apply if use_rescaler is True. Can be a single module, a sequence of modules, or None.
                Defaults to None.
        """
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_samples=n_samples,
        )
        self.save_hyperparameters()
        if use_rescaler:
            raise NotImplementedError("Rescaling is not implemented yet.")
            # TODO implement rescaling
            if isinstance(rescaler, Sequence):
                filtered = [m for m in rescaler if m is not None]
                self.rescaler = nn.ModuleList(filtered) if filtered else None
        self.backbone = backbone
        self.padding = padding
        self.crop = CropND(padding=padding) if padding else nn.Identity()
        self.path = CondOTProbPath()
        self.solver = ODESolver(self.backbone)
        self.step_size = 1 / solver_iter

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: dict[str, torch.Tensor]):
        """
        Args:
            x (torch.Tensor): the (noisy) input [bs, 2, 48, 32]
            t (torch.Tensor): the timestep [bs, 1, 1, 1]
            conditioning (dict[str, torch.Tensor]): the conditioning dict with tensors of shape [bs, ...]
        Returns:
            torch.Tensor: [bs, 2, 48, 32], h and w dim might be cropped
        """
        res = self.backbone(x, t, conditioning)
        return self.crop(res)

    @abstractmethod
    def _calc_loss(self, batch) -> torch.Tensor:
        """Calculate the loss for the flow matching model.

        Subclasses must implement this method with their specific loss calculation logic.
        """
        pass

    def training_step(self, batch) -> torch.Tensor:
        loss = self._calc_loss(batch)
        loss = loss.mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    @abstractmethod
    def predict_step(self, batch) -> torch.Tensor:
        pass

    def validation_step(self, batch) -> torch.Tensor:
        loss = self._calc_loss(batch)
        loss = reduce(loss, "b c h w -> c", "mean")  # How good are we per channel
        for i, lo in enumerate(loss):
            self.log(
                f"val_loss_var_{i}",
                lo,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        loss = torch.mean(loss)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        loss = self._calc_loss(batch)
        loss = reduce(loss, "b c h w -> c", "mean")  # How good are we per channel
        for i, lo in enumerate(loss):
            self.log(
                f"test_loss_var_{i}",
                lo,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        loss = torch.mean(loss)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


class FlowMatchingNoiseModel(InternalTDScalingMixin, BaseFlowMatchingModel):
    """
    Flow Matching Model that predicts noise/deviations from the NWP forecast.

    How the prediction works:
    - Instead of generating samples similar to the ground truth directly, we want to sample the deviation (x_1 - nwp-fc)
    - Then these sampled deviations are added to the nwp forecasts to get the final samples
    - Also the deviations are scaled according to the lead time. The scaling factor is learned via linear regression.
    """

    def __init__(
        self,
        backbone: ConditionalVectorField,
        n_samples: int,
        solver_iter: int,
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ):
        """Initialize the FlowMatchingNoiseModel.

        Args:
            backbone (ConditionalVectorField): The neural network backbone.
            n_samples (int): Number of ensemble samples to generate.
            solver_iter (int): Number of ODE solver iterations.
            padding (Sequence[int]): Padding values to crop from the output.
            optimizer (Callable[..., torch.optim.Optimizer]): Factory function to create the optimizer.
            lr_scheduler (DictConfig): Configuration for the learning rate scheduler.
            internal_td_scaling (str): Scaling strategy ("abs", "std", "learned", or "linear_abs").
            use_rescaler (bool): Whether to use rescaling modules.
            rescaler (Sequence[nn.Module | None] | nn.Module | None, optional): Rescaling modules.
        """
        # Initialize base class first (calls nn.Module.__init__)
        BaseFlowMatchingModel.__init__(
            self,
            backbone=backbone,
            n_samples=n_samples,
            solver_iter=solver_iter,
            padding=padding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_rescaler=use_rescaler,
            rescaler=rescaler,
        )
        # Then initialize mixin (which assigns nn.Module attributes)
        InternalTDScalingMixin.__init__(self, internal_td_scaling=internal_td_scaling)

    def _calc_loss(self, batch) -> torch.Tensor:
        # Sample Data (X_0,X_1) ~ π(X_0,X_1) = N(X_0|0,I)q(X_1)
        nwp_fc, ground_truth, td = batch["x"], batch["y"], batch["timedelta"]
        # We want to predict the errors of the NWP forecasts
        # Now x_1 contains the errors
        x_1 = ground_truth - nwp_fc["predicted_vars_mean"]
        # x_1 should always have roughly the same magnitude, independent of the lead time
        # NOTE a larger scale leads to smaller loss values, as the model has to predict smaller deviations
        # this effect should be counteracted by the internal_td_scaling that is learned beforehand
        scale = self.internal_td_scaling.get_scale(td=td)  # Shape [b, n_vars, 1, 1]
        # Now x_1 contains the scaled errors, the model has to learn only one scale
        # NOTE the predicted noise needs to be scaled back during inference
        # to get the actual noise that is added to the NWP forecasts
        x_1 = x_1 / scale

        # Sample x_0 ~ N(0,I)
        x_0 = torch.randn_like(x_1).to(x_1)

        # Sample a random timestep
        t = torch.rand(x_1.size(0)).to(x_1)

        # Get the probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        u_t_theta = self.backbone(x=path_sample.x_t, t=path_sample.t, conditioning=nwp_fc)
        u_t_ref = path_sample.dx_t

        # Calc the l2 loss
        loss = torch.pow(u_t_theta - u_t_ref, 2)
        return loss

    def predict_step(self, batch) -> torch.Tensor:
        nwp_fc, x_1, td = batch["x"], batch["y"], batch["timedelta"]

        # repeat shapes to be able to generate 50 different samples
        nwp_fc_expanded = {}
        for k, v in nwp_fc.items():
            nwp_fc_expanded[k] = repeat(v, "b ... -> (n_samples b) ...", n_samples=self.n_samples)

        # Sample batch_size * n_samples random images. Keep the other dimensions as x_1
        x_init = torch.randn(x_1.size(0) * self.n_samples, *x_1.shape[1:]).to(x_1)
        sol = self.solver.sample(
            x_init=x_init,
            conditioning=nwp_fc_expanded,
            method="dopri5",
            step_size=self.step_size,
        )
        # Sol now contains the deviations that need to be added to the nwp forecasts
        sol = rearrange(sol, "(n_samples b) ... -> b n_samples ...", n_samples=self.n_samples)

        # Calculate the scale factor based on the lead time
        scale = self.internal_td_scaling.get_scale(td=td)  # Shape [b, n_vars, 1, 1]
        # Rescale the deviations according to the lead time (inverse of what was done during training)
        sol = sol * rearrange(scale, "b n_vars 1 1 -> b 1 n_vars 1 1")  # type: ignore
        ens_mean = rearrange(nwp_fc["predicted_vars_mean"], "b c h w -> b 1 c h w")
        res = ens_mean + sol  # Add the nwp forecasts to the deviations to get the final samples
        res_cropped = self.crop(res)
        return res_cropped


class FlowMatchingDirectModel(BaseFlowMatchingModel):
    """
    Flow Matching Model that directly predicts the output without noise addition.

    The model learns to predict the ground truth directly, with timedelta information
    passed to the backbone to enable lead-time-aware predictions.
    """

    def _calc_loss(self, batch) -> torch.Tensor:
        # Sample Data (X_0,X_1) ~ π(X_0,X_1) = N(X_0|0,I)q(X_1)
        nwp_fc, ground_truth, td = batch["x"], batch["y"], batch["timedelta"]
        # x_1 is the ground truth directly (no deviation calculation)
        x_1 = ground_truth

        # Sample x_0 ~ N(0,I)
        x_0 = torch.randn_like(x_1).to(x_1)

        # Sample a random timestep
        t = torch.rand(x_1.size(0)).to(x_1)

        # Add timedelta to conditioning
        conditioning = nwp_fc
        conditioning["timedelta"] = td

        # Get the probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        u_t_theta = self.backbone(x=path_sample.x_t, t=path_sample.t, conditioning=conditioning)
        u_t_ref = path_sample.dx_t

        # Calc the l2 loss
        loss = torch.pow(u_t_theta - u_t_ref, 2)
        return loss

    def predict_step(self, batch) -> torch.Tensor:
        nwp_fc, x_1, td = batch["x"], batch["y"], batch["timedelta"]

        # repeat shapes to be able to generate 50 different samples
        nwp_fc_expanded = {}
        for k, v in nwp_fc.items():
            nwp_fc_expanded[k] = repeat(v, "b ... -> (n_samples b) ...", n_samples=self.n_samples)

        # Expand timedelta for all samples
        td_expanded = repeat(td, "b ... -> (n_samples b) ...", n_samples=self.n_samples)
        nwp_fc_expanded["timedelta"] = td_expanded

        # Sample batch_size * n_samples random images. Keep the other dimensions as x_1
        x_init = torch.randn(x_1.size(0) * self.n_samples, *x_1.shape[1:]).to(x_1)
        sol = self.solver.sample(
            x_init=x_init,
            conditioning=nwp_fc_expanded,
            method="dopri5",
            step_size=self.step_size,
        )
        # sol now contains the direct predictions
        sol = rearrange(sol, "(n_samples b) ... -> b n_samples ...", n_samples=self.n_samples)
        res_cropped = self.crop(sol)
        return res_cropped


# Backwards compatibility alias
FlowMatchingModel = FlowMatchingNoiseModel
