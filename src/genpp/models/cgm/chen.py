from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from genpp.models.layers import CropND, LocallyConnected2D, UNet, _get_scale_td
from genpp.models.utils import BaseModule


class BaseChenModel(BaseModule, ABC):
    """Base class for generative models with mean, std, and noise decoder components.

    Args:
        in_features (int): Number of input features.
        meta_features (int): Number of metadata features. !THIS DOES NOT INCLUDE THE PIXEL INDEX.
        out_features (int): Number of output features.
        width (int): Width of the input feature map.
        height (int): Height of the input feature map.
        noise_dim (int): Dimensionality of the latent space.
        embedding_dim (int): Dimensionality of the embeddings. Defaults to 5. If set to 0, no embeddings are used.
        n_samples_train (int): Number of samples to generate during training. Defaults to 50.
        final_activation (nn.Module): Activation function to apply at the end of the model.
        loss_fn (nn.Module): Loss function to use for training. Defaults to EnergyScore with beta=1.0.
        lr (float): Learning rate for the optimizer. Defaults to 3e-4.
        optimizer (Type[torch.optim.Optimizer]): Optimizer class to use. Defaults to torch.optim.AdamW.
        **kwargs: Any additional keyword arguments. These are here for compatibility and are ignored.
    """

    def __init__(
        self,
        in_features: int,
        meta_features: int,
        out_features: int,
        width: int,  # latitude
        height: int,  # longitude
        noise_dim: int,
        embedding_dim: int,
        n_samples_train: int,
        final_activation: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        use_rescaler: bool = False,
        rescaler: Sequence[nn.Module | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
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
        self.n_samples_train = n_samples_train
        self.final_activation = final_activation
        self.use_embedding = embedding_dim > 0
        self.loss_fn = loss_fn

        if self.use_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=self.gridpoints, embedding_dim=embedding_dim
            )
        self.scale_variance_td = None  # To be fitted via callback

    # Abstract components - to be implemented by subclasses
    @property
    @abstractmethod
    def mean_model(self) -> nn.Module:
        """Model to compute the mean of the input features."""
        pass

    @property
    @abstractmethod
    def std_model(self) -> nn.Module:
        """Model to compute the standard deviation of the input features."""
        pass

    @property
    @abstractmethod
    def noise_decoder(self) -> nn.Module:
        """Model to decode the noise and generate samples.
        Expected output shape [batch_size, n_samples_train, height, width, out_features]"""
        pass

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

    def forward(self, x: torch.Tensor, td: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): the input tensor. Shape [batch_size, in_features, height, width]
            td (torch.Tensor): the time delta tensor (used to scale the predicted noise). Shape [batch_size]

        Returns:
            torch.Tensor: the output tensor. Shape [batch_size, n_samples_train, out_features, height, width]
        """
        batch_size = x.shape[0]
        mean, std, meta = x.split(
            (self.in_features, self.in_features, self.meta_dim + self.use_embedding), dim=1
        )  # Mean, Std, Meta have now shape [batch_size, var, lon, lat]
        if self.use_embedding:
            pixel_idx = meta[:, -1, ...].int()  # Shape [batch_size, lon, lat]
            meta = meta[
                :, :-1, ...
            ]  # Remove the pixel index from the meta tensor. Shape [batch_size, meta_dim, lon, lat]
            emb = self.embedding(pixel_idx)  # Shape [batch_size, embedding_dim, lon, lat]
            emb = rearrange(emb, "b h w c -> b c h w")
        else:
            emb = None

        # NOTE it would make sense to use a residual connection here, but the original paper does not use it.
        # Also we have to figure out how to find the mean of the correct variable (2m_temperature or 10m_wind_speed).
        pred_mean = self.mean_model(mean)  # Shape [batch_size, out_features, lon, lat]
        delta = self.std_model(std)
        z = self.get_noise(batch_size).to(delta)  # Must be on the same device as delta

        noise = z * delta

        full_input_repeated_noise = self.concat_noise_decoder_input(
            mean=mean, std=std, meta=meta, embedding=emb, noise=noise
        )  # Shape [batch_size * n_samples_train, ...]
        std_samples = self.noise_decoder(
            full_input_repeated_noise
        )  # Shape [batch_size, n_samples_train, out_features, lon, lat]
        scales = rearrange(
            _get_scale_td(td=td, betas=self.scale_variance_td), "b c h w -> b 1 c h w"
        )
        res = (
            pred_mean + scales * std_samples
        )  # Shape [batch_size, n_samples_train, out_features, lon, lat]
        return self.final_activation(res)

    def predict_step(self, batch) -> Any:
        x, _, td = batch
        return self.forward(x, td)

    def training_step(self, batch) -> torch.Tensor:
        x, y, td = batch
        res = self.forward(x, td)  # shape [b, n_samples, out_features, lon, lat]
        res_reshape = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape = rearrange(y, "b c h w -> b (c h w)")
        loss = self.loss_fn(res_reshape, y_reshape)
        loss = torch.mean(loss)  # Take mean across batches
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        x, y, td = batch
        res = self.forward(x, td)
        res_reshape = rearrange(res, "b n c h w -> b c n (h w)")
        y_reshape = rearrange(y, "b c h w -> b c (h w)")
        loss_per_var = self.loss_fn(res_reshape, y_reshape)  # shape [b, c]
        loss_per_var = reduce(loss_per_var, "b c -> c", "mean")
        # Log the loss for each variable separately
        for i in range(self.out_features):
            self.log(
                f"val_loss_var_{i}",
                loss_per_var[i],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        # Log the overall loss
        res_reshape2 = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape2 = rearrange(y, "b c h w -> b (c h w)")
        loss = torch.mean(self.loss_fn(res_reshape2, y_reshape2))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch) -> torch.Tensor:
        x, y, td = batch
        res = self.forward(x, td)
        res_reshape = rearrange(res, "b n c h w -> b c n (h w)")
        y_reshape = rearrange(y, "b c h w -> b c (h w)")
        loss_per_var = self.loss_fn(res_reshape, y_reshape)  # shape [b, c]
        loss_per_var = reduce(loss_per_var, "b c -> c", "mean")
        # Log the loss for each variable separately
        for i in range(self.out_features):
            self.log(
                f"test_loss_var_{i}",
                loss_per_var[i],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        # Log the overall loss
        res_reshape2 = rearrange(res, "b n c h w -> b n (c h w)")
        y_reshape2 = rearrange(y, "b c h w -> b (c h w)")
        loss = torch.mean(self.loss_fn(res_reshape2, y_reshape2))
        return loss


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
        self.save_hyperparameters()
        super().__init__(*args, **kwargs)

        self.hidden_dim_std = hidden_dim_std
        self.hidden_dim_decoder = hidden_dim_decoder

        self._mean_model = nn.Sequential(
            LocallyConnected2D(
                height=self.height,
                width=self.width,
                in_features=self.in_features,
                out_features=self.out_features,
            ),
            Rearrange("b c h w -> b 1 c h w"),
        )

        self._std_model = nn.Sequential(
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

        self._noise_decoder = nn.Sequential(
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
                n=self.n_samples_train,
                h=self.height,
                w=self.width,
                c=self.out_features,
            ),
        )

    @property
    def mean_model(self) -> nn.Module:
        """Model to compute the mean of the input features."""
        return self._mean_model

    @property
    def std_model(self) -> nn.Module:
        """Model to compute the standard deviation of the input features."""
        return self._std_model

    @property
    def noise_decoder(self) -> nn.Module:
        """Model to decode the noise and generate samples."""
        return self._noise_decoder

    def get_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(size=(batch_size, self.n_samples_train, self.noise_dim))

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
            full_input, "b d -> b n d", n=self.n_samples_train
        )  # Shape [batch_size, n_samples_train, lat * lon * some_features]

        # Concatenate along the last dimension
        full_input_repeated_noise = torch.cat(
            [full_input_repeated, noise], dim=-1
        )  # Shape [batch_size, n_samples_train, lat * lon * some_features + noise_dim]

        full_input_repeated_noise = rearrange(
            full_input_repeated_noise, "b n d -> (b n) d"
        )  # Reshape so that all processing of all samples can be done in parallel.
        # Shape [batch_size * n_samples_train, lat * lon * (2 * in_features + embedding_dim + meta_dim) + noise_dim]
        return full_input_repeated_noise


class CNNChenModel(BaseChenModel):
    """CNN-based Chen model.
    In this model, both the std_model and the noise_decoder are separate UNets.
    Args:
        padding (Tuple[int, int, int, int]): Padding already applied to the input tensor.
        This is used as a final step to crop the output tensor to the original size so it can be compared with y to calculate the loss.

    """

    def __init__(self, *args, padding: tuple[int, int, int, int], **kwargs) -> None:
        self.save_hyperparameters(ignore=["final_activation", "loss_fn"])
        super().__init__(*args, **kwargs)
        self.padding = padding
        self.height_no_pad = self.height - self.padding[2] - self.padding[3]  # longitude
        self.width_no_pad = self.width - self.padding[0] - self.padding[1]  # latitude

        self.crop = CropND(padding=self.padding)

        self._mean_model = nn.Sequential(  # This model operates on the cropped input
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
        self._std_model = nn.Sequential(
            UNet(
                in_features=self.in_features,
                out_features=self.noise_dim,
            ),
            Rearrange("b c h w -> b 1 c h w"),
        )

        self._noise_decoder = nn.Sequential(
            UNet(
                in_features=2 * self.in_features
                + self.meta_dim
                + self.embedding_dim
                + self.noise_dim,
                out_features=self.out_features,
            ),
            Rearrange("(b n) c h w -> b n c h w", n=self.n_samples_train),
            self.crop,  # Crop back to the original size
        )

        if self.use_embedding:
            self.embedding = nn.Embedding(self.height * self.width, self.embedding_dim)

    @property
    def mean_model(self) -> nn.Module:
        """Model to compute the mean of the input features."""
        return self._mean_model

    @property
    def std_model(self) -> nn.Module:
        """Model to compute the standard deviation of the input features."""
        return self._std_model

    @property
    def noise_decoder(self) -> nn.Module:
        """Model to decode the noise and generate samples."""
        return self._noise_decoder

    def get_noise(self, batch_size: int) -> torch.Tensor:
        """Get the noise tensor for the model."""
        return torch.randn(
            size=(batch_size, self.n_samples_train, self.noise_dim, self.height, self.width)
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
        noise has shape [batch_size, n_samples_train, noise_dim, height, width]
        """
        if self.use_embedding:
            full_det = torch.cat([mean, std, meta, embedding], dim=1)
        else:
            full_det = torch.cat([mean, std, meta], dim=1)
        full_det = repeat(full_det, "b c h w -> b n c h w", n=self.n_samples_train)
        full_stoch = torch.cat([full_det, noise], dim=2)  # Concat along channel dim
        full_stoch = rearrange(
            full_stoch, "b n c h w -> (b n) c h w"
        )  # Can be processed in parallel now.
        return full_stoch  # Shape [batch_size * n_samples_train, (2 * var + meta_var + embedding_dim + noise_dim), height, width]


class PatchwiseChenModel(BaseChenModel):
    """Patchwise Chen model. This is a placeholder for a patchwise implementation."""

    def __init__(self, *args, **kwargs) -> None:
        self.save_hyperparameters()
        super().__init__(*args, **kwargs)
