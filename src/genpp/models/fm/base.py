from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from omegaconf import DictConfig

from genpp.models.fm.helpers import ConditionalVectorField
from genpp.models.layers import CropND
from genpp.models.utils import BaseInternalTDScaling, BaseModule


class FlowMatchingModel(BaseModule):
    """
    Generic Flow Matching Model that works with different backbone architectures.

    NOTE that in this class the naming convention is different than in the other classes:
    - x_1 is the target (i.e. the ground truth forecasts, for which we want to generate samples that are similar to)
    - the nwp forecast is the conditioning
    - td is the lead time (between 0 and 1) for which the prediction is made

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
        internal_td_scaling: BaseInternalTDScaling,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ):
        """_summary_

        Args:
            backbone (ConditionalVectorField): _description_
            n_samples (int): _description_
            solver_iter (int): _description_
            padding (Sequence[int]): _description_
            optimizer (Callable[..., torch.optim.Optimizer]): _description_
            lr_scheduler (DictConfig): _description_
            internal_td_scaling (str): _description_
            use_rescaler (bool): _description_
            rescaler (Sequence[nn.Module  |  None] | nn.Module | None, optional): _description_. Defaults to None.
        """
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.save_hyperparameters()
        if use_rescaler:
            # TODO implement rescaling
            if isinstance(rescaler, Sequence):
                filtered = [m for m in rescaler if m is not None]
                self.rescaler = nn.ModuleList(filtered) if filtered else None
        self.backbone = backbone
        self.n_samples = n_samples
        self.padding = padding
        self.crop = CropND(padding=padding) if padding else nn.Identity()
        self.path = CondOTProbPath()
        self.solver = ODESolver(self.backbone)
        self.step_size = 1 / solver_iter
        self.internal_td_scaling = internal_td_scaling

    # TODO add a parameter to choose which kind of normalization we will use
    def setup(self, stage: str | None = None):
        """This fits the submodel to predict the size of the standard deviation so that the final modle only has to learn one scale.

        Args:
            stage (str | None): The stage of setup, e.g., "fit"
        """
        if stage == "fit":
            self.internal_td_scaling.fit(self)

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

    def _calc_loss(self, batch) -> torch.Tensor:
        # Sample Data (X_0,X_1) ~ π(X_0,X_1) = N(X_0|0,I)q(X_1)
        nwp_fc, ground_truth, td = batch["x"], batch["y"], batch["timedelta"]
        # We want to predict the errors of the NWP forecasts
        # Now x_1 contains the errors
        x_1 = ground_truth - nwp_fc["predicted_vars"]
        # x_1 should always have roughly the same magnitude, independent of the lead time
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

    def predict_step(self, batch) -> torch.Tensor:
        nwp_fc, x_1, td = batch["x"], batch["y"], batch["timedelta"]

        ens_mean = rearrange(nwp_fc["predicted_vars"], "b c h w -> b 1 c h w")

        # repeat shapes to be able to generate 50 different samples
        for k, v in nwp_fc.items():
            nwp_fc[k] = repeat(v, "b ... -> (n_samples b) ...", n_samples=self.n_samples)

        # Sample n_samples * batch size random images. Keep the other dimensions as x_1
        x_init = torch.randn(nwp_fc["predicted_vars"].size(0), *x_1.shape[1:]).to(x_1)

        sol = self.solver.sample(
            x_init=x_init,
            conditioning=nwp_fc,
            method="midpoint",
            step_size=self.step_size,
        )
        # Sol now contains the deviations that need to be added to the nwp forecasts
        sol = rearrange(sol, "(n_samples b) ... -> b n_samples ...", n_samples=self.n_samples)

        # Calculate the scale factor based on the lead time
        scale = self.internal_td_scaling.get_scale(td=td)  # Shape [b, n_vars, 1, 1]
        # Rescale the deviations according to the lead time (inverse of what was done during training)
        sol = sol * rearrange(scale, "b n_vars 1 1 -> b 1 n_vars 1 1")  # type: ignore
        res = ens_mean + sol  # Add the nwp forecasts to the deviations to get the final samples
        res_cropped = self.crop(res)
        return res_cropped

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

    def on_load_checkpoint(self, checkpoint):
        # TODO modify this to be able to handle muliple kinds of normalization
        # If buffer exists in checkpoint, load it
        if "scale_variance_td" in checkpoint["state_dict"]:
            print("Loading scale_variance_td from checkpoint")
            self.register_buffer("scale_variance_td", checkpoint["state_dict"]["scale_variance_td"])
