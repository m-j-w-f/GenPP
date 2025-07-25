from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor


class LocallyConnected2D(nn.Module):
    def __init__(self, height: int, width: int, in_features: int, out_features: int) -> None:
        """
        A custom layer that applies a separate linear transformation for each (height, width) location.

        Args:
            height (int): Height of the input feature map.
            width (int): Width of the input feature map.
            in_features (int): Number of input features per location.
            out_features (int): Number of output features per location.
        """
        super(LocallyConnected2D, self).__init__()
        self.height = height
        self.width = width
        self.in_features = in_features
        self.out_features = out_features

        # Create a weight tensor for all spatial locations
        self.weight = nn.Parameter(torch.randn(height, width, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(height, width, out_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LocallyConnected2D layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_features, height, width].

        Returns:
            Tensor: Output tensor of shape [batch_size, out_features, height, width].
        """
        # Perform the linear transformation for all spatial locations in parallel
        out = torch.einsum("bhwc,hwco->bhwo", x, self.weight) + self.bias
        return out


# TODO: can this model already handle multivariate predictions? -> Yes

# NOTE: This model now has an insane number (9M) of parameters,
# the linear layers in the std_model and in the noise_decoder should be replaced
# with a LocallyConnected2D layer followed by fully connected layers or a cnn.


# TODO generate a superclas of the chen model which can then be used as a consturct to parametrize the model.
class BaseChenModel(ABC, nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_dim: int,
        embedding_dim: int = 5,
        n_samples_train: int = 50,
        final_activation: nn.Module = nn.Identity(),
    ) -> None:
        """Base class for generative models with mean, std, and noise decoder components.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            latent_dim (int): Dimensionality of the latent space.
            embedding_dim (int, optional): Dimensionality of the embeddings. Defaults to 5. If set to 0, no embeddings are used.
            n_samples_train (int, optional): Number of samples to generate during training. Defaults to 50.
            final_activation (nn.Module, optional): Activation function to apply at the end of the model.
                Defaults to nn.Identity(), which means no activation is applied.
        """
        super(BaseChenModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.n_samples_train = n_samples_train
        self.final_activation = final_activation
        self.use_embedding = embedding_dim > 0

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

    @abstractmethod
    def get_embedding(self, batch_size: int) -> Tensor:
        """Get the embedding layer."""
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
        noise: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Concatenate the mean and standard deviation tensors for the noise decoder input.

        Args:
            mean (torch.Tensor): Mean tensor. Output of the mean_model.
            std (torch.Tensor): Standard deviation tensor. Output of the std_model.
            noise (torch.Tensor): Noise tensor. Output of the noise_model.
            *args: Additional positional arguments for subclass implementations
            **kwargs: Additional keyword arguments for subclass implementations
        """
        pass

    def forward(self, x: torch.Tensor, doy: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        mean, std = rearrange(
            x, "batch lat lon (two aggr) var -> two batch lat lon aggr var", two=2
        )  # Mean, Std have now shape [batch_size, lat, lon, 1, var]
        mean = mean.squeeze()
        std = std.squeeze()

        # TODO it would make sense to use a residual connection here, but the original paper does not use it.
        # Also we have to figure out how to find the mean of the correct variable (2m_temperature or 10m_wind_speed).
        pred_mean = self.mean_model(mean)  # Shape [batch_size, lat, lon, out_features]
        delta = self.std_model(std)  # Shape [batch_size, latent_dim]
        # NOTE: which device should this be on? (PyTorch lighning should handle this)
        z = torch.randn(
            size=(batch_size, self.n_samples_train, self.latent_dim)
        )  # Shape [batch_size, n_samples_train, latent_dim]

        noise = z * delta  # Shape [batch_size, n_samples_train, latent_dim]

        full_input_repeated_noise = self.concat_noise_decoder_input(
            mean=mean, std=std, noise=noise
        )  # Shape [batch_size * n_samples_train, lat * lon * (2 * embedding_dim + latent_dim)]
        std_samples = self.noise_decoder(
            full_input_repeated_noise
        )  # Shape [batch_size, n_samples_train, lat, lon, out_features]

        res = pred_mean + std_samples  # Shape [batch_size, n_samples_train, lat, lon, out_features]

        return self.final_activation(res)


class FcChenModel(BaseChenModel):
    def __init__(
        self,
        *args,
        width: int,
        height: int,
        hidden_dim_std: int,
        hidden_dim_decoder: int,
        **kwargs,
    ) -> None:
        """Model from GENERATIVE MACHINE LEARNING METHODS FOR MULTIVARIATE ENSEMBLE POSTPROCESSING by J. Chen et al. (2024).

        Args:
            width (int): Width of the input feature map.
            height (int): Height of the input feature map.
            hidden_dim_std (int): Dimensionality of the hidden layers for the standard deviation model.
            hidden_dim_decoder (int): Dimensionality of the hidden layers for the decoder.
        """
        super(FcChenModel, self).__init__(*args, **kwargs)

        self.height = height
        self.width = width
        self.gridpoints = width * height
        self.hidden_dim_std = hidden_dim_std
        self.hidden_dim_decoder = hidden_dim_decoder

        self._mean_model = nn.Sequential(
            LocallyConnected2D(
                height=self.height,
                width=self.width,
                in_features=self.in_features,
                out_features=self.out_features,
            ),
            Rearrange("b h w o -> b 1 h w o"),
        )

        self._std_model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=self.in_features * self.gridpoints,
                out_features=self.hidden_dim_std,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim_std, self.latent_dim),
            nn.Softplus(),  # Keep the scaling factor delta positive
            Rearrange("b latent_dim -> b 1 latent_dim"),
        )

        self._noise_decoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            # Here the input is the concatenation of the mean and std model outputs with the embeddings, a doy feature and the latent noise
            nn.Linear(
                in_features=2
                * self.in_features
                * self.gridpoints  # mean and std for each gridpoint
                + self.embedding_dim * self.gridpoints  # embedding dimension for each gridpoint
                + self.latent_dim,  # latent noise for each gridpoint
                out_features=self.hidden_dim_decoder,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim_decoder, self.gridpoints * self.out_features),
            Rearrange(
                "(b n) (h w o) -> b n h w o",
                n=self.n_samples_train,
                h=self.height,
                w=self.width,
                o=self.out_features,
            ),
        )

        if self.use_embedding:
            self.embedding = nn.Embedding(self.gridpoints, self.embedding_dim)

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

    def get_embedding(self, batch_size: int) -> torch.Tensor:
        """Get the embedding layer."""
        embedding = self.embedding.weight  # Shape [gridpoints, embedding_dim]
        grid_embeddings_flat = rearrange(
            embedding, "gridpoints emb -> 1 (gridpoints emb)"
        )  # Shape [1, gridpoints *embedding_dim]
        grid_embeddings_flat = grid_embeddings_flat.expand(batch_size, -1)
        return grid_embeddings_flat

    def concat_noise_decoder_input(
        self, mean: torch.Tensor, std: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate the mean and standard deviation tensors for the noise decoder input."""
        mean_flat = mean.flatten(start_dim=1)  # Shape [batch_size, lat * lon * in_features]
        std_flat = std.flatten(start_dim=1)  # Shape [batch_size, lat * lon * in_features]

        if self.use_embedding:
            embedding = self.get_embedding(
                batch_size=mean.shape[0]
            )  # Shape [batch_size, lat * lon * embedding_dim]

            full_input = torch.cat(
                [mean_flat, std_flat, embedding], dim=-1
            )  # Shape [batch_size, lat * lon * (2 * in_features + embedding_dim)]
        else:
            full_input = torch.cat(
                [mean_flat, std_flat], dim=-1
            )  # Shape [batch_size, lat * lon * (2 * in_features)]

        full_input_repeated = repeat(
            full_input, "b d -> b n d", n=self.n_samples_train
        )  # Shape [batch_size, n_samples_train, lat * lon * (2 * in_features + embedding_dim)]

        # Concatenate along the last dimension
        full_input_repeated_noise = torch.cat(
            [full_input_repeated, noise], dim=-1
        )  # Shape [batch_size, lat * lon * (2 * in_features + embedding_dim) + latent_dim, n_samples_train]

        full_input_repeated_noise = rearrange(
            full_input_repeated_noise, "b n d -> (b n) d"
        )  # Reshape so that all processing of all samples can be done in parallel.
        # Shape [batch_size * n_samples_train, lat * lon * (2 * in_features + embedding_dim) + latent_dim]
        return full_input_repeated_noise
