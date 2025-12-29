"""Base Engression model for grid-based weather forecast post-processing.

This module implements the engression approach adapted for spatial data,
using stochastic neural networks that inject noise at each layer.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, reduce
from omegaconf import DictConfig

from genpp.models.cgm.utils import BaseGenerativeModule
from genpp.models.cgm.utils.td_scaling import InternalTDScalingMixin
from genpp.models.layers import CropND
from genpp.models.loss import EnergyScore


class StochasticLayer2D(nn.Module):
    """A 2D stochastic layer that injects noise for grid-based data.

    This layer concatenates random noise with the input features before
    applying a convolutional transformation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        noise_channels (int): Number of noise channels to inject. Defaults to 32.
        kernel_size (int): Kernel size for convolution. Defaults to 3.
        add_bn (bool): Whether to add batch normalization. Defaults to True.
        activation (nn.Module | None): Activation function. Defaults to nn.ReLU().
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_channels: int = 32,
        kernel_size: int = 3,
        add_bn: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_channels = noise_channels
        padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels + noise_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
        ]
        if add_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise injection.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channels, height, width].
        """
        batch_size, _, height, width = x.shape
        noise = torch.randn(
            batch_size, self.noise_channels, height, width, device=x.device, dtype=x.dtype
        )
        x_with_noise = torch.cat([x, noise], dim=1)
        return self.layer(x_with_noise)


class StochasticResBlock2D(nn.Module):
    """A 2D stochastic residual block that injects noise for grid-based data.

    Args:
        channels (int): Number of input and output channels.
        hidden_channels (int | None): Hidden channels. Defaults to same as channels.
        noise_channels (int): Number of noise channels to inject. Defaults to 32.
        kernel_size (int): Kernel size for convolution. Defaults to 3.
        add_bn (bool): Whether to add batch normalization. Defaults to True.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        noise_channels: int = 32,
        kernel_size: int = 3,
        add_bn: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels

        # First stochastic layer with ReLU activation
        self.block1 = StochasticLayer2D(
            in_channels=channels,
            out_channels=hidden_channels,
            noise_channels=noise_channels,
            kernel_size=kernel_size,
            add_bn=add_bn,
            activation=nn.ReLU(),
        )

        # Second stochastic layer without activation (applied after residual)
        self.block2 = StochasticLayer2D(
            in_channels=hidden_channels,
            out_channels=channels,
            noise_channels=noise_channels,
            kernel_size=kernel_size,
            add_bn=add_bn,
            activation=None,
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise injection and residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch, channels, height, width].
        """
        residual = x

        # First stochastic layer
        out = self.block1(x)

        # Second stochastic layer
        out = self.block2(out)

        # Residual connection
        out = out + residual
        return self.relu(out)


class StochasticBackbone(nn.Module, ABC):
    """Abstract base class for stochastic backbones.

    Subclasses must implement the forward method.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stochastic backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    @abstractmethod
    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Generate multiple samples for the same input.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples of shape [batch, n_samples, out_channels, height, width].
        """
        pass


class BaseEngressionModel(BaseGenerativeModule, ABC):
    """Base engression model for grid-based weather forecast post-processing.

    This model uses a stochastic neural network that injects noise at each layer,
    enabling the generation of ensemble members by running the network multiple times
    with different noise realizations.

    Args:
        backbone (StochasticBackbone): Stochastic neural network backbone.
        n_samples (int): Number of samples to generate during training/inference.
        padding (Sequence[int]): Padding values to crop from the output.
        optimizer (Callable[..., torch.optim.Optimizer]): Optimizer factory.
        lr_scheduler (DictConfig): Learning rate scheduler config.
        use_rescaler (bool): Whether to use rescaling modules.
        rescaler (Sequence[nn.Module | None] | nn.Module | None): Rescaling modules.
        loss_fn (nn.Module | None): Loss function. Defaults to EnergyScore.
    """

    def __init__(
        self,
        backbone: StochasticBackbone,
        n_samples: int,
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        loss_fn: nn.Module = EnergyScore(),
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_samples=n_samples,
        )
        self.save_hyperparameters(ignore=["backbone", "loss_fn"])
        if use_rescaler:
            raise NotImplementedError("Rescaling is not implemented yet.")

        self.backbone = backbone
        self.padding = padding
        self.crop = CropND(padding=padding) if padding else nn.Identity()

        # Loss function
        self.loss_fn = loss_fn
        self.loss_is_energy_score = type(self.loss_fn) is EnergyScore
        if not self.loss_is_energy_score:
            self.es = EnergyScore()

    @abstractmethod
    def prepare_input(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare input for the backbone.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary with predicted_vars,
                auxiliary_vars, meta_vars, and pixel_idx.

        Returns:
            torch.Tensor: Prepared input tensor.
        """
        pass

    @abstractmethod
    def forward(self, x: dict[str, torch.Tensor], td: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary.
            td (torch.Tensor): Time delta tensor.

        Returns:
            torch.Tensor: Output tensor of shape [batch, n_samples, out_features, height, width].
        """
        pass

    def predict_step(self, batch: dict) -> torch.Tensor:
        """Prediction step.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predictions.
        """
        x, td = batch["x"], batch["timedelta"]
        return self.forward(x, td)

    def training_step(self, batch: dict) -> torch.Tensor:
        """Training step.

        Args:
            batch (dict): Input batch with x, y, and timedelta.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y, td = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, td)  # [batch, n_samples, out_features, height, width]

        # Reshape for loss computation
        res_reshape = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape = rearrange(y, "b c h w -> b (c h w)")

        loss = self.loss_fn(res_reshape, y_reshape, mode="complete")
        loss = torch.mean(loss)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _score_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Unified scoring step for validation and testing.

        Args:
            batch (dict): Input batch.
            stage (str): Stage name ("val" or "test").

        Returns:
            torch.Tensor: Loss value.
        """
        x, y, td = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, td)

        # Reshape for per-variable and overall loss computation
        res_reshape = rearrange(res, "b n c h w -> b c n (h w)")
        y_reshape = rearrange(y, "b c h w -> b c (h w)")
        res_reshape2 = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape2 = rearrange(y, "b c h w -> b (c h w)")

        # Compute energy score
        if self.loss_is_energy_score:
            es_per_var = self.loss_fn(res_reshape, y_reshape, mode="per_var")
            es_overall = torch.mean(self.loss_fn(res_reshape2, y_reshape2, mode="complete"))
        else:
            es_per_var = self.es(res_reshape, y_reshape, mode="per_var")
            es_overall = torch.mean(self.es(res_reshape2, y_reshape2, mode="complete"))

        # Log per-variable energy score
        out_features = res.shape[2]
        es_per_var_mean = reduce(es_per_var, "b c -> c", "mean")
        for i in range(out_features):
            self.log(
                f"{stage}_loss_var_{i}",
                es_per_var_mean[i],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # Log overall energy score
        self.log(
            f"{stage}_loss",
            es_overall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # If using a different loss function, also log it
        if not self.loss_is_energy_score:
            loss_fn_per_var = self.loss_fn(res_reshape, y_reshape, mode="per_var")
            loss_fn_per_var_mean = reduce(loss_fn_per_var, "b c -> c", "mean")
            for i in range(out_features):
                self.log(
                    f"{stage}_loss_fn_var_{i}",
                    loss_fn_per_var_mean[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            loss_fn_overall = torch.mean(self.loss_fn(res_reshape2, y_reshape2, mode="complete"))
            self.log(
                f"{stage}_loss_fn",
                loss_fn_overall,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return es_overall

    def validation_step(self, batch: dict) -> torch.Tensor:
        """Validation step.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        return self._score_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Test step.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            dataloader_idx (int): Dataloader index.

        Returns:
            torch.Tensor: Test loss.
        """
        return self._score_step(batch, stage="test")


class BaseEngressionNoiseModel(InternalTDScalingMixin, BaseEngressionModel, ABC):
    """Base engression model that predicts noise with internal TD scaling.

    This model predicts deviations from the NWP forecast, which are then scaled
    by the internal TD scaling and added to the NWP mean.
    """

    def __init__(
        self,
        backbone: StochasticBackbone,
        n_samples: int,
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        loss_fn: nn.Module = EnergyScore(),
    ) -> None:
        """Initialize BaseEngressionNoiseModel.

        Args:
            backbone (StochasticBackbone): Stochastic neural network backbone.
            n_samples (int): Number of samples to generate.
            padding (Sequence[int]): Padding values to crop from the output.
            optimizer (Callable[..., torch.optim.Optimizer]): Optimizer factory.
            lr_scheduler (DictConfig): Learning rate scheduler config.
            internal_td_scaling (str): Scaling strategy ("abs", "std", "learned", or "linear_abs").
            use_rescaler (bool): Whether to use rescaling modules.
            rescaler (Sequence[nn.Module | None] | nn.Module | None): Rescaling modules.
            loss_fn (nn.Module): Loss function.
        """
        BaseEngressionModel.__init__(
            self,
            backbone=backbone,
            n_samples=n_samples,
            padding=padding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_rescaler=use_rescaler,
            rescaler=rescaler,
            loss_fn=loss_fn,
        )
        InternalTDScalingMixin.__init__(self, internal_td_scaling=internal_td_scaling)

    def forward(self, x: dict[str, torch.Tensor], td: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model with noise prediction and scaling.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary.
            td (torch.Tensor): Time delta tensor.

        Returns:
            torch.Tensor: Output tensor of shape [batch, n_samples, out_features, height, width].
        """
        # Prepare input for backbone
        backbone_input = self.prepare_input(x)

        # Generate samples using the stochastic backbone
        samples = self.backbone.sample(backbone_input, self.n_samples)

        # Get NWP forecast mean for residual connection
        nwp_mean = x["predicted_vars"]  # [batch, channels, height, width]

        # Scale by TD and add to NWP mean
        scale = self.internal_td_scaling.get_scale(td=td)  # [batch, n_vars, 1, 1]
        scale = rearrange(scale, "b c 1 1 -> b 1 c 1 1")

        # samples contains the deviation from NWP mean
        nwp_mean_expanded = rearrange(nwp_mean, "b c h w -> b 1 c h w")
        result = nwp_mean_expanded + scale * samples

        # Crop padding
        result = self.crop(result)
        return result


class BaseEngressionDirectModel(BaseEngressionModel, ABC):
    """Base engression model that predicts targets directly without internal TD scaling.

    This model directly predicts the target values without using internal TD scaling.
    The samples represent the full prediction, not deviations from NWP.
    """

    def forward(self, x: dict[str, torch.Tensor], td: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model with direct prediction.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary.
            td (torch.Tensor): Time delta tensor.

        Returns:
            torch.Tensor: Output tensor of shape [batch, n_samples, out_features, height, width].
        """
        # Prepare input for backbone
        backbone_input = self.prepare_input(x)

        # Generate samples using the stochastic backbone
        # samples contains the deviation from NWP mean
        samples = self.backbone.sample(backbone_input, self.n_samples)

        # Get NWP forecast mean for residual connection
        nwp_mean = x["predicted_vars"]  # [batch, channels, height, width]
        nwp_mean_expanded = rearrange(nwp_mean, "b c h w -> b 1 c h w")
        result = nwp_mean_expanded + samples

        # Crop padding
        result = self.crop(result)
        return result
