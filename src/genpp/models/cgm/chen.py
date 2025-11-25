from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from genpp.models.layers import CropND, LocallyConnected2D, UNet
from genpp.models.loss import EnergyScore
from genpp.models.utils import BaseGenerativeModule


class BaseChenModel(BaseGenerativeModule, ABC):
    """Base class for generative models with mean, std, and noise decoder components.

    Args:
        in_features (int): Number of input features.
        meta_features (int): Number of metadata features. !THIS DOES NOT INCLUDE THE PIXEL INDEX.
        out_features (int): Number of output features.
        width (int): Width of the input feature map.
        height (int): Height of the input feature map.
        noise_dim (int): Dimensionality of the latent space.
        embedding_dim (int): Dimensionality of the embeddings. Defaults to 5. If set to 0, no embeddings are used.
        n_samples (int): Number of samples to generate during training. Defaults to 50.
        final_activation (nn.Module): Activation function to apply at the end of the model.
        loss_fn (nn.Module): Loss function to use for training. Defaults to EnergyScore with beta=1.0.
        lr (float): Learning rate for the optimizer. Defaults to 3e-4.
        optimizer (Type[torch.optim.Optimizer]): Optimizer class to use. Defaults to torch.optim.AdamW.
        **kwargs: Any additional keyword arguments. These are here for compatibility and are ignored.

    Attributes:
        mean_model (nn.Module): Model to compute the mean of the input features. Must be set by subclasses.
        std_model (nn.Module): Model to compute the standard deviation of the input features. Must be set by subclasses.
        noise_decoder (nn.Module): Model to decode the noise and generate samples. Must be set by subclasses.
    """

    # Required attributes - must be set by subclasses in __init__
    mean_model: nn.Module
    std_model: nn.Module
    noise_decoder: nn.Module

    def __init__(
        self,
        in_features: int,
        meta_features: int,
        out_features: int,
        width: int,  # latitude
        height: int,  # longitude
        noise_dim: int,
        embedding_dim: int,
        n_samples: int,
        final_activation: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str = "abs",
        use_rescaler: bool = False,
        rescaler: Sequence[nn.Module | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            internal_td_scaling=internal_td_scaling,
            n_samples=n_samples,
        )
        if use_rescaler:
            raise NotImplementedError("Rescaling is not implemented yet.")
        if kwargs:
            warn(f"Ignoring additional arguments: {kwargs}")
        self.in_features = in_features
        self.meta_dim = meta_features
        self.out_features = out_features
        self.width = width
        self.height = height
        self.gridpoints = self.width * self.height
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_activation = final_activation
        self.use_embedding = embedding_dim > 0
        self.loss_fn = loss_fn
        # We need this for the validation step where we always use Energy Score
        # and if a different loss is used for training we want to record that too
        self.loss_is_energy_score = type(self.loss_fn) is EnergyScore
        if not self.loss_is_energy_score:
            self.es = EnergyScore()

        if self.use_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=self.gridpoints, embedding_dim=embedding_dim
            )

    @abstractmethod
    def concat_noise_decoder_input(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        meta: torch.Tensor,
        embedding: torch.Tensor | None,
        noise: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Concatenate the mean and standard deviation tensors for the noise decoder input.

        Args:
            mean (torch.Tensor): Mean tensor. Output of the mean_model.
            std (torch.Tensor): Standard deviation tensor. Output of the std_model.
            meta (torch.Tensor): Metadata tensor. Probably contains sin/cos doy, lat and long.
            !MAKE SURE TO REMOVE THE PIXEL INDEX FROM THE META TENSOR BEFORE PASSING IT TO THIS METHOD.
            embedding (torch.Tensor): Embedding tensor. Output of the get_embedding method.
            noise (torch.Tensor): Noise tensor. Output of the noise_model.
            *args: Additional positional arguments for subclass implementations
            **kwargs: Additional keyword arguments for subclass implementations
        """
        pass

    @abstractmethod
    def get_noise(self, batch_size: int) -> torch.Tensor:
        """Get the noise tensor for the model.

        Returns:
            torch.Tensor: Noise tensor.
        """
        pass

    def forward(self, x: dict[str, torch.Tensor], td: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (dict[str, torch.Tensor]): the input dictionary.
            td (torch.Tensor): the time delta tensor (used to scale the predicted noise). Shape [batch_size]

        Returns:
            torch.Tensor: the output tensor. Shape [batch_size, n_samples, out_features, height, width]
        """
        batch_size = x["predicted_vars"].shape[0]
        x_cat = torch.cat([x["predicted_vars"], x["auxiliary_vars"]], dim=1)
        mean, std = torch.chunk(x_cat, 2, dim=1)
        meta = x["meta_vars"]

        if self.use_embedding:
            pixel_idx = x["pixel_idx"]  # Shape [batch_size, lon, lat]
            emb = self.embedding(pixel_idx)  # Shape [batch_size, embedding_dim, lon, lat]
            emb = rearrange(emb, "b 1 h w c -> b c h w")
        else:
            emb = None

        # TODO it would make sense to use a residual connection here, but the original paper does not use it.
        # Also we have to figure out how to find the mean of the correct variable (2m_temperature or 10m_wind_speed).
        # This is easy with the improved data loading
        pred_mean = self.mean_model(mean)  # Shape [batch_size, out_features, lon, lat]
        delta = self.std_model(std)
        z = self.get_noise(batch_size).to(delta)  # Must be on the same device as delta

        noise = z * delta

        full_input_repeated_noise = self.concat_noise_decoder_input(
            mean=mean, std=std, meta=meta, embedding=emb, noise=noise
        )  # Shape [batch_size * n_samples, ...]
        std_samples = self.noise_decoder(
            full_input_repeated_noise
        )  # Shape [batch_size, n_samples, out_features, lon, lat]
        scales = rearrange(
            self.internal_td_scaling.get_scale(td=td),
            "b c h w -> b 1 c h w",
        )
        res = (
            pred_mean + scales * std_samples
        )  # Shape [batch_size, n_samples, out_features, lon, lat]
        return self.final_activation(res)

    def predict_step(self, batch) -> Any:
        x, td = batch["x"], batch["timedelta"]
        return self.forward(x, td)

    def training_step(self, batch) -> torch.Tensor:
        x, y, td = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, td)  # shape [b, n_samples, out_features, lon, lat]
        res_reshape = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape = rearrange(y, "b c h w -> b (c h w)")
        loss = self.loss_fn(res_reshape, y_reshape, mode="complete")  # shape [b]
        loss = torch.mean(loss)  # Take mean across batches
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # New unified scoring step for validation & testing.
    def score_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Unified scoring step used by validation_step and test_step.

        Computes ensemble predictions, calculates per-variable loss and overall loss,
        and logs metrics prefixed with the provided `stage` (e.g., "val" or "test").

        Args:
            batch (dict): The input batch with keys 'x', 'y', 'timedelta'.
            stage (str): Log prefix/prefix for metrics.
        """
        x, y, td = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, td)

        # Reshape for per-variable and overall loss computation
        res_reshape = rearrange(res, "b n c h w -> b c n (h w)")
        y_reshape = rearrange(y, "b c h w -> b c (h w)")
        res_reshape2 = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape2 = rearrange(y, "b c h w -> b (c h w)")

        # Compute energy score (always logged as {stage}_loss)
        if self.loss_is_energy_score:
            es_per_var = self.loss_fn(res_reshape, y_reshape, mode="per_var")  # shape [b, c]
            es_overall = torch.mean(self.loss_fn(res_reshape2, y_reshape2, mode="complete"))
        else:
            es_per_var = self.es(res_reshape, y_reshape, mode="per_var")  # shape [b, c]
            es_overall = torch.mean(self.es(res_reshape2, y_reshape2, mode="complete"))

        # Log per-variable energy score
        es_per_var_mean = reduce(es_per_var, "b c -> c", "mean")
        for i in range(self.out_features):
            self.log(
                f"{stage}_loss_var_{i}",
                es_per_var_mean[i],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # Log overall energy score as {stage}_loss
        self.log(
            f"{stage}_loss", es_overall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # If using a different loss function, also log it and return it
        if not self.loss_is_energy_score:
            loss_fn_per_var = self.loss_fn(res_reshape, y_reshape, mode="per_var")  # shape [b, c]
            loss_fn_per_var_mean = reduce(loss_fn_per_var, "b c -> c", "mean")
            for i in range(self.out_features):
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

    def validation_step(self, batch) -> torch.Tensor:
        return self.score_step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        return self.score_step(batch, stage="test")

    def on_load_checkpoint(self, checkpoint):
        # If buffer exists in checkpoint, load it
        if "scale_variance_td" in checkpoint["state_dict"]:
            print("Loading scale_variance_td from checkpoint")
            self.register_buffer("scale_variance_td", checkpoint["state_dict"]["scale_variance_td"])


class FcChenModel(BaseChenModel):
    """Model from GENERATIVE MACHINE LEARNING METHODS FOR MULTIVARIATE ENSEMBLE POSTPROCESSING by J. Chen et al. (2024).

    Args:
        hidden_dim_std (int): Dimensionality of the hidden layers for the standard deviation model.
        hidden_dim_decoder (int): Dimensionality of the hidden layers for the decoder.
    """

    # NOTE: This model now has an insane number (9M) of parameters,
    # the linear layers in the std_model and in the noise_decoder should be replaced
    # with a LocallyConnected2D layer followed by fully connected layers or a cnn.
    def __init__(
        self,
        *args,
        hidden_dim_std: int,
        hidden_dim_decoder: int,
        **kwargs,
    ) -> None:
        warn(
            "FcChenModel is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.save_hyperparameters()
        super().__init__(*args, **kwargs)

        self.hidden_dim_std = hidden_dim_std
        self.hidden_dim_decoder = hidden_dim_decoder

        self.mean_model = nn.Sequential(
            LocallyConnected2D(
                height=self.height,
                width=self.width,
                in_features=self.in_features,
                out_features=self.out_features,
            ),
            Rearrange("b c h w -> b 1 c h w"),
        )

        self.std_model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=self.in_features * self.gridpoints,
                out_features=self.hidden_dim_std,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim_std, self.noise_dim),
            nn.Softplus(),  # Keep the scaling factor delta positive
            Rearrange("b noise_dim -> b 1 noise_dim"),
        )

        self.noise_decoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            # Here the input is the concatenation of the mean and std model outputs with the embeddings, a doy feature and the latent noise
            nn.Linear(
                in_features=2
                * self.in_features
                * self.gridpoints  # mean and std for each gridpoint
                + self.embedding_dim * self.gridpoints  # embedding dimension for each gridpoint
                + self.meta_dim * self.gridpoints  # metadata for each gridpoint
                + self.noise_dim,  # latent noise for each gridpoint
                out_features=self.hidden_dim_decoder,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim_decoder, self.gridpoints * self.out_features),
            Rearrange(
                "(b n) (c h w) -> b n c h w",
                n=self.n_samples,
                h=self.height,
                w=self.width,
                c=self.out_features,
            ),
        )

    def get_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(size=(batch_size, self.n_samples, self.noise_dim))

    def concat_noise_decoder_input(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        meta: torch.Tensor,
        embedding: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the mean and standard deviation tensors for the noise decoder input."""
        mean_flat = mean.flatten(start_dim=1)  # Shape [batch_size, lat * lon * in_features]
        std_flat = std.flatten(start_dim=1)  # Shape [batch_size, lat * lon * in_features]
        meta_flat = meta.flatten(start_dim=1)  # Shape [batch_size, lat * lon * meta_dim]

        if self.use_embedding:
            embedding_flat = embedding.flatten(
                start_dim=1
            )  # Shape [batch_size, lat * lon * embedding_dim]

            full_input = torch.cat(
                [mean_flat, std_flat, meta_flat, embedding_flat], dim=-1
            )  # Shape [batch_size, lat * lon * (2 * in_features + meta_dim + embedding_dim)]
        else:
            full_input = torch.cat(
                [mean_flat, std_flat, meta_flat], dim=-1
            )  # Shape [batch_size, lat * lon * (2 * in_features + meta_dim)]

        # Shape of full_input: [batch_size, lat * lon * some_features]

        full_input_repeated = repeat(
            full_input, "b d -> b n d", n=self.n_samples
        )  # Shape [batch_size, n_samples, lat * lon * some_features]

        # Concatenate along the last dimension
        full_input_repeated_noise = torch.cat(
            [full_input_repeated, noise], dim=-1
        )  # Shape [batch_size, n_samples, lat * lon * some_features + noise_dim]

        full_input_repeated_noise = rearrange(
            full_input_repeated_noise, "b n d -> (b n) d"
        )  # Reshape so that all processing of all samples can be done in parallel.
        # Shape [batch_size * n_samples, lat * lon * (2 * in_features + embedding_dim + meta_dim) + noise_dim]
        return full_input_repeated_noise


class CNNChenModel(BaseChenModel):
    """CNN-based Chen model.
    In this model, both the std_model and the noise_decoder are separate UNets.

    Args:
        in_features (int): Number of input features. Passed to BaseChenModel.
        meta_features (int): Number of metadata features. Passed to BaseChenModel.
        out_features (int): Number of output features. Passed to BaseChenModel.
        width (int): Width of the input feature map. Passed to BaseChenModel.
        height (int): Height of the input feature map. Passed to BaseChenModel.
        noise_dim (int): Dimensionality of the latent space. Passed to BaseChenModel.
        embedding_dim (int): Dimensionality of the embeddings. Passed to BaseChenModel.
        n_samples (int): Number of samples to generate during training. Passed to BaseChenModel.
        final_activation (nn.Module): Activation function to apply at the end. Passed to BaseChenModel.
        loss_fn (nn.Module): Loss function to use for training. Passed to BaseChenModel.
        optimizer (Callable[..., torch.optim.Optimizer]): Optimizer class. Passed to BaseChenModel.
        lr_scheduler (DictConfig): Learning rate scheduler config. Passed to BaseChenModel.
        internal_td_scaling (str): TD scaling mode. Passed to BaseChenModel. Default is "abs".
        padding (Tuple[int, int, int, int]): Padding already applied to the input tensor.
            This is used as a final step to crop the output tensor to the original size
            so it can be compared with y to calculate the loss.
        std_unet_channels (Sequence[int]): Number of channels at each encoder level for the std_model UNet.
            Default is (32, 64, 64).
        std_unet_kernel_size (int): Kernel size for convolutions in std_model UNet. Default is 3.
        std_unet_use_batchnorm (bool): Whether to use batch normalization in std_model UNet. Default is False.
        std_unet_pool_type (str): Type of pooling for std_model UNet ("max" or "avg"). Default is "max".
        decoder_unet_channels (Sequence[int]): Number of channels at each encoder level for the noise_decoder UNet.
            Default is (32, 64, 64).
        decoder_unet_kernel_size (int): Kernel size for convolutions in noise_decoder UNet. Default is 3.
        decoder_unet_use_batchnorm (bool): Whether to use batch normalization in noise_decoder UNet. Default is False.
        decoder_unet_pool_type (str): Type of pooling for noise_decoder UNet ("max" or "avg"). Default is "max".
    """

    def __init__(
        self,
        # BaseChenModel parameters
        in_features: int,
        meta_features: int,
        out_features: int,
        width: int,
        height: int,
        noise_dim: int,
        embedding_dim: int,
        n_samples: int,
        final_activation: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str,
        # For compatibility with other models
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | None,
        # CNNChenModel-specific parameters
        padding: tuple[int, int, int, int],
        # UNet parameters for std_model
        std_unet_channels: Sequence[int] = (32, 64, 64),
        std_unet_kernel_size: int = 3,
        std_unet_use_batchnorm: bool = False,
        std_unet_pool_type: str = "max",
        # UNet parameters for noise_decoder
        decoder_unet_channels: Sequence[int] = (32, 64, 64),
        decoder_unet_kernel_size: int = 3,
        decoder_unet_use_batchnorm: bool = False,
        decoder_unet_pool_type: str = "max",
    ) -> None:
        self.save_hyperparameters()
        super().__init__(
            in_features=in_features,
            meta_features=meta_features,
            out_features=out_features,
            width=width,
            height=height,
            noise_dim=noise_dim,
            embedding_dim=embedding_dim,
            n_samples=n_samples,
            final_activation=final_activation,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            internal_td_scaling=internal_td_scaling,
        )
        self.padding = padding
        self.height_no_pad = self.height - self.padding[2] - self.padding[3]  # longitude
        self.width_no_pad = self.width - self.padding[0] - self.padding[1]  # latitude

        self.crop = CropND(padding=self.padding)

        self.mean_model = nn.Sequential(  # This model operates on the cropped input
            self.crop,
            LocallyConnected2D(
                height=self.height_no_pad,
                width=self.width_no_pad,
                in_features=self.in_features,
                out_features=self.out_features,
            ),
            Rearrange("b c h w -> b 1 c h w"),
        )

        # [batch_size, lat, lon, var]
        self.std_model = nn.Sequential(
            UNet(
                in_features=self.in_features,
                out_features=self.noise_dim,
                channels=std_unet_channels,
                kernel_size=std_unet_kernel_size,
                use_batchnorm=std_unet_use_batchnorm,
                pool_type=std_unet_pool_type,
            ),
            Rearrange("b c h w -> b 1 c h w"),
        )

        self.noise_decoder = nn.Sequential(
            UNet(
                in_features=2 * self.in_features
                + self.meta_dim
                + self.embedding_dim
                + self.noise_dim,
                out_features=self.out_features,
                channels=decoder_unet_channels,
                kernel_size=decoder_unet_kernel_size,
                use_batchnorm=decoder_unet_use_batchnorm,
                pool_type=decoder_unet_pool_type,
            ),
            Rearrange("(b n) c h w -> b n c h w", n=self.n_samples),
            self.crop,  # Crop back to the original size
        )

        if self.use_embedding:
            self.embedding = nn.Embedding(self.height * self.width, self.embedding_dim)

    def get_noise(self, batch_size: int) -> torch.Tensor:
        """Get the noise tensor for the model."""
        return torch.randn(
            size=(batch_size, self.n_samples, self.noise_dim, self.height, self.width)
        )

    def concat_noise_decoder_input(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        meta: torch.Tensor,
        embedding: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the mean and standard deviation tensors for the noise decoder input.
        mean, std have shape [batch_size, var, height, width]
        meta has shape [batch_size, meta_dim, height, width]
        embeddings have shape [batch_size, embedding_dim, height, width]
        noise has shape [batch_size, n_samples, noise_dim, height, width]
        """
        if self.use_embedding:
            full_det = torch.cat([mean, std, meta, embedding], dim=1)
        else:
            full_det = torch.cat([mean, std, meta], dim=1)
        full_det = repeat(full_det, "b c h w -> b n c h w", n=self.n_samples)
        full_stoch = torch.cat([full_det, noise], dim=2)  # Concat along channel dim
        full_stoch = rearrange(
            full_stoch, "b n c h w -> (b n) c h w"
        )  # Can be processed in parallel now.
        return full_stoch  # Shape [batch_size * n_samples, (2 * var + meta_var + embedding_dim + noise_dim), height, width]
