"""Classifier-free guidance for Flow Matching models.

This module provides classifier-free guidance (CFG) variants of the flow matching models.
CFG allows trading off diversity for quality by interpolating between unconditional
and conditional vector fields during sampling.

Given:
- u^target_t(x|∅): unguided marginal vector field (unconditional)
- u^target_t(x|y): guided marginal vector field (conditional)
- w > 1: guidance scale

The classifier-free guided vector field is:
    u_t(x|y) = (1 - w) * u^target_t(x|∅) + w * u^target_t(x|y)

When w=1, this reduces to standard conditional generation (no guidance).
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, repeat
from flow_matching.solver import ODESolver
from omegaconf import DictConfig

from genpp.models.cgm.fm.base import (
    BaseFlowMatchingModel,
    ConditionalVectorField,
    FlowMatchingDirectModel,
    FlowMatchingNoiseModel,
)
from genpp.models.cgm.utils.td_scaling import InternalTDScalingMixin


def _is_guidance_scale_one(guidance_scale: float) -> bool:
    """Check if guidance scale is effectively 1.0 (no guidance).

    Uses tolerance-based comparison for floating point safety.
    """
    return abs(guidance_scale - 1.0) < 1e-9


class _CFGVectorFieldWrapper(nn.Module):
    """Wrapper that applies classifier-free guidance to a conditional vector field.

    This wrapper computes both conditional and unconditional vector fields
    and combines them according to the CFG formula during sampling.

    Args:
        backbone: The underlying conditional vector field network.
        guidance_scale: The guidance scale w. When w=1, no guidance is applied.
        null_conditioning_fn: Function that creates null/unconditional conditioning.
    """

    def __init__(
        self,
        backbone: ConditionalVectorField,
        guidance_scale: float,
        null_conditioning_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    ):
        super().__init__()
        self.backbone = backbone
        self.guidance_scale = guidance_scale
        self.null_conditioning_fn = null_conditioning_fn

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply CFG during forward pass.

        Args:
            x: Input tensor [bs, c, h, w]
            t: Timestep tensor [bs] or scalar
            conditioning: Conditioning dictionary

        Returns:
            Guided vector field output [bs, c, h, w]
        """
        # When guidance_scale is 1, just return the conditional output
        if _is_guidance_scale_one(self.guidance_scale):
            return self.backbone(x, t, conditioning)

        # Compute conditional vector field: u^target_t(x|y)
        u_cond = self.backbone(x, t, conditioning)

        # Create null conditioning for unconditional pass
        null_conditioning = self.null_conditioning_fn(conditioning)

        # Compute unconditional vector field: u^target_t(x|∅)
        u_uncond = self.backbone(x, t, null_conditioning)

        # Apply CFG formula: u_t(x|y) = (1 - w) * u_uncond + w * u_cond
        w = self.guidance_scale
        return (1 - w) * u_uncond + w * u_cond


class FlowMatchingNoiseModelCFG(InternalTDScalingMixin, BaseFlowMatchingModel):
    """Flow Matching Noise Model with Classifier-Free Guidance support.

    This model extends FlowMatchingNoiseModel with CFG capabilities for sampling.
    During training, the model is trained with conditioning dropout to learn both
    conditional and unconditional distributions.

    During inference, CFG interpolates between the two:
        u_t(x|y) = (1 - w) * u^target_t(x|∅) + w * u^target_t(x|y)

    Args:
        backbone: The neural network backbone for the conditional vector field.
        n_samples: Number of ensemble samples to generate.
        solver_iter: Number of ODE solver iterations.
        padding: Padding values for output cropping.
        optimizer: Factory function to create the optimizer.
        lr_scheduler: Configuration for learning rate scheduler.
        internal_td_scaling: Scaling strategy for timedelta.
        use_rescaler: Whether to use rescaling modules.
        guidance_scale: The guidance scale w (default 1.0, no guidance).
        conditioning_dropout_prob: Probability of dropping conditioning during training.
        rescaler: Optional rescaling modules.
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
        guidance_scale: float = 1.0,
        conditioning_dropout_prob: float = 0.0,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ):
        """Initialize the FlowMatchingNoiseModelCFG."""
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
        InternalTDScalingMixin.__init__(self, internal_td_scaling=internal_td_scaling)

        self.guidance_scale = guidance_scale
        self.conditioning_dropout_prob = conditioning_dropout_prob

    def _create_null_conditioning(
        self, conditioning: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Create null conditioning by zeroing out the conditioning tensors.

        Args:
            conditioning: Original conditioning dictionary

        Returns:
            Null conditioning dictionary with zeroed tensors
        """
        null_conditioning = {}
        for key, value in conditioning.items():
            if isinstance(value, torch.Tensor):
                null_conditioning[key] = torch.zeros_like(value)
            else:
                null_conditioning[key] = value
        return null_conditioning

    def _calc_loss(self, batch) -> torch.Tensor:
        """Calculate the loss with optional conditioning dropout for CFG training."""
        nwp_fc, ground_truth, td = batch["x"], batch["y"], batch["timedelta"]

        # We want to predict the errors of the NWP forecasts
        x_1 = ground_truth - nwp_fc["predicted_vars_mean"]

        # Scale errors according to lead time
        scale = self.internal_td_scaling.get_scale(td=td)
        x_1 = x_1 / scale

        # Sample x_0 ~ N(0,I)
        x_0 = torch.randn_like(x_1).to(x_1)

        # Sample a random timestep
        t = torch.rand(x_1.size(0)).to(x_1)

        # Get the probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # Apply conditioning dropout during training for CFG
        conditioning = nwp_fc
        if self.training and self.conditioning_dropout_prob > 0:
            # Create mask for which samples should have null conditioning
            dropout_mask = torch.rand(x_1.size(0), device=x_1.device) < self.conditioning_dropout_prob
            if dropout_mask.any():
                # Create null conditioning
                null_cond = self._create_null_conditioning(conditioning)
                # Apply dropout to each tensor in conditioning
                conditioning = {}
                for key, value in nwp_fc.items():
                    if isinstance(value, torch.Tensor):
                        mask_shape = [value.size(0)] + [1] * (value.dim() - 1)
                        mask = dropout_mask.view(*mask_shape).expand_as(value)
                        conditioning[key] = torch.where(mask, null_cond[key], value)
                    else:
                        conditioning[key] = value

        u_t_theta = self.backbone(x=path_sample.x_t, t=path_sample.t, conditioning=conditioning)
        u_t_ref = path_sample.dx_t

        # Calc the l2 loss
        loss = torch.pow(u_t_theta - u_t_ref, 2)
        return loss

    def predict_step(self, batch, guidance_scale: float | None = None) -> torch.Tensor:
        """Generate predictions using classifier-free guidance.

        Args:
            batch: Input batch containing 'x', 'y', and 'timedelta'.
            guidance_scale: Optional override for the guidance scale. If None,
                uses the model's default guidance_scale set during initialization.

        Returns:
            Generated predictions tensor.
        """
        nwp_fc, x_1, td = batch["x"], batch["y"], batch["timedelta"]

        # Use provided guidance_scale or fall back to model default
        effective_guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Repeat shapes to generate n_samples different samples
        nwp_fc_expanded = {}
        for k, v in nwp_fc.items():
            nwp_fc_expanded[k] = repeat(v, "b ... -> (n_samples b) ...", n_samples=self.n_samples)

        # Create CFG wrapper if guidance_scale != 1
        if not _is_guidance_scale_one(effective_guidance_scale):
            cfg_backbone = _CFGVectorFieldWrapper(
                backbone=self.backbone,
                guidance_scale=effective_guidance_scale,
                null_conditioning_fn=self._create_null_conditioning,
            )
            solver = ODESolver(cfg_backbone)
        else:
            solver = self.solver

        # Sample batch_size * n_samples random images
        x_init = torch.randn(x_1.size(0) * self.n_samples, *x_1.shape[1:]).to(x_1)
        sol = solver.sample(
            x_init=x_init,
            conditioning=nwp_fc_expanded,
            method="midpoint",
            step_size=self.step_size,
        )

        # Reshape output
        sol = rearrange(sol, "(n_samples b) ... -> b n_samples ...", n_samples=self.n_samples)

        # Calculate the scale factor based on the lead time
        scale = self.internal_td_scaling.get_scale(td=td)
        # Rescale the deviations according to the lead time
        sol = sol * rearrange(scale, "b n_vars 1 1 -> b 1 n_vars 1 1")
        ens_mean = rearrange(nwp_fc["predicted_vars_mean"], "b c h w -> b 1 c h w")
        res = ens_mean + sol
        res_cropped = self.crop(res)
        return res_cropped


class FlowMatchingDirectModelCFG(BaseFlowMatchingModel):
    """Flow Matching Direct Model with Classifier-Free Guidance support.

    This model extends FlowMatchingDirectModel with CFG capabilities for sampling.
    During training, the model is trained with conditioning dropout to learn both
    conditional and unconditional distributions.

    During inference, CFG interpolates between the two:
        u_t(x|y) = (1 - w) * u^target_t(x|∅) + w * u^target_t(x|y)

    Args:
        backbone: The neural network backbone for the conditional vector field.
        n_samples: Number of ensemble samples to generate.
        solver_iter: Number of ODE solver iterations.
        padding: Padding values for output cropping.
        optimizer: Factory function to create the optimizer.
        lr_scheduler: Configuration for learning rate scheduler.
        use_rescaler: Whether to use rescaling modules.
        guidance_scale: The guidance scale w (default 1.0, no guidance).
        conditioning_dropout_prob: Probability of dropping conditioning during training.
        rescaler: Optional rescaling modules.
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
        guidance_scale: float = 1.0,
        conditioning_dropout_prob: float = 0.0,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ):
        """Initialize the FlowMatchingDirectModelCFG."""
        super().__init__(
            backbone=backbone,
            n_samples=n_samples,
            solver_iter=solver_iter,
            padding=padding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_rescaler=use_rescaler,
            rescaler=rescaler,
        )

        self.guidance_scale = guidance_scale
        self.conditioning_dropout_prob = conditioning_dropout_prob

    def _create_null_conditioning(
        self, conditioning: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Create null conditioning by zeroing out the conditioning tensors.

        Args:
            conditioning: Original conditioning dictionary

        Returns:
            Null conditioning dictionary with zeroed tensors
        """
        null_conditioning = {}
        for key, value in conditioning.items():
            if isinstance(value, torch.Tensor):
                null_conditioning[key] = torch.zeros_like(value)
            else:
                null_conditioning[key] = value
        return null_conditioning

    def _calc_loss(self, batch) -> torch.Tensor:
        """Calculate the loss with optional conditioning dropout for CFG training."""
        nwp_fc, ground_truth, td = batch["x"], batch["y"], batch["timedelta"]
        x_1 = ground_truth

        # Sample x_0 ~ N(0,I)
        x_0 = torch.randn_like(x_1).to(x_1)

        # Sample a random timestep
        t = torch.rand(x_1.size(0)).to(x_1)

        # Add timedelta to conditioning
        conditioning = dict(nwp_fc)
        conditioning["timedelta"] = td

        # Apply conditioning dropout during training for CFG
        if self.training and self.conditioning_dropout_prob > 0:
            # Create mask for which samples should have null conditioning
            dropout_mask = torch.rand(x_1.size(0), device=x_1.device) < self.conditioning_dropout_prob
            if dropout_mask.any():
                # Create null conditioning
                null_cond = self._create_null_conditioning(conditioning)
                # Apply dropout to each tensor in conditioning
                new_conditioning = {}
                for key, value in conditioning.items():
                    if isinstance(value, torch.Tensor):
                        mask_shape = [value.size(0)] + [1] * (value.dim() - 1)
                        mask = dropout_mask.view(*mask_shape).expand_as(value)
                        new_conditioning[key] = torch.where(mask, null_cond[key], value)
                    else:
                        new_conditioning[key] = value
                conditioning = new_conditioning

        # Get the probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        u_t_theta = self.backbone(x=path_sample.x_t, t=path_sample.t, conditioning=conditioning)
        u_t_ref = path_sample.dx_t

        # Calc the l2 loss
        loss = torch.pow(u_t_theta - u_t_ref, 2)
        return loss

    def predict_step(self, batch, guidance_scale: float | None = None) -> torch.Tensor:
        """Generate predictions using classifier-free guidance.

        Args:
            batch: Input batch containing 'x', 'y', and 'timedelta'.
            guidance_scale: Optional override for the guidance scale. If None,
                uses the model's default guidance_scale set during initialization.

        Returns:
            Generated predictions tensor.
        """
        nwp_fc, x_1, td = batch["x"], batch["y"], batch["timedelta"]

        # Use provided guidance_scale or fall back to model default
        effective_guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Repeat shapes to generate n_samples different samples
        nwp_fc_expanded = {}
        for k, v in nwp_fc.items():
            nwp_fc_expanded[k] = repeat(v, "b ... -> (n_samples b) ...", n_samples=self.n_samples)

        # Expand timedelta for all samples
        td_expanded = repeat(td, "b ... -> (n_samples b) ...", n_samples=self.n_samples)
        nwp_fc_expanded["timedelta"] = td_expanded

        # Create CFG wrapper if guidance_scale != 1
        if not _is_guidance_scale_one(effective_guidance_scale):
            cfg_backbone = _CFGVectorFieldWrapper(
                backbone=self.backbone,
                guidance_scale=effective_guidance_scale,
                null_conditioning_fn=self._create_null_conditioning,
            )
            solver = ODESolver(cfg_backbone)
        else:
            solver = self.solver

        # Sample batch_size * n_samples random images
        x_init = torch.randn(x_1.size(0) * self.n_samples, *x_1.shape[1:]).to(x_1)
        sol = solver.sample(
            x_init=x_init,
            conditioning=nwp_fc_expanded,
            method="midpoint",
            step_size=self.step_size,
        )

        # Reshape output
        sol = rearrange(sol, "(n_samples b) ... -> b n_samples ...", n_samples=self.n_samples)
        res_cropped = self.crop(sol)
        return res_cropped
