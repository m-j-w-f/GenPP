import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor


class LocallyConnected2D(nn.Module):
    def __init__(
        self, height: int, width: int, in_features: int, out_features: int
    ) -> None:
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
        self.weight = nn.Parameter(
            torch.randn(height, width, in_features, out_features)
        )
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
class ChenModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        width: int,
        height: int,
        latent_dim: int,
        hidden_dim_std: int,
        hidden_dim_decoder: int,
        embedding_dim: int = 5,
        n_samples_train: int = 50,
        final_activation: nn.Module = nn.Identity(),
    ) -> None:
        """Model from GENERATIVE MACHINE LEARNING METHODS FOR MULTIVARIATE ENSEMBLE POSTPROCESSING by J. Chen et al. (2024).

        Args:
            in_features (int): Number of input features. The predicted features should be listed first in the feature dimension of the input tensor.
            out_features (int): Number of output features. These features should be listed first in the feature dimension of the input tensor.
            width (int): Width of the input feature map.
            height (int): Height of the input feature map.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim_std (int): Dimensionality of the hidden layers for the standard deviation model.
            hidden_dim_decoder (int): Dimensionality of the hidden layers for the decoder.
            embedding_dim (int, optional): Dimensionality of the embeddings. Defaults to 5.
            n_samples_train (int, optional): Number of samples to generate during training. Defaults to 50.
            final_activation (nn.Module, optional): Activation function to apply at the end of the model.
                Defaults to nn.Identity(), which means no activation is applied.
        """
        super(ChenModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.height = height
        self.width = width
        self.gridpoints = width * height
        self.latent_dim = latent_dim
        self.hidden_dim_std = hidden_dim_std
        self.hidden_dim_decoder = hidden_dim_decoder
        self.embedding_dim = embedding_dim
        self.n_samples_train = n_samples_train
        self.final_activation = final_activation

        self.mean_model = LocallyConnected2D(
            height=self.height,
            width=self.width,
            in_features=self.in_features,
            out_features=self.out_features,
        )

        self.std_model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=self.in_features * self.gridpoints,
                out_features=self.hidden_dim_std,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim_std, self.latent_dim),
            nn.Softplus(),  # Keep the scaling factor delta positive
        )

        self.noise_decoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            # Here the input is the concatenation of the mean and std model outputs with the embeddings, a doy feature and the latent noise
            nn.Linear(
                in_features=2
                * in_features
                * self.gridpoints  # mean and std for each gridpoint
                + self.embedding_dim
                * self.gridpoints  # embedding dimension for each gridpoint
                + self.latent_dim,  # latent noise for each gridpoint
                out_features=self.hidden_dim_decoder,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim_decoder, self.gridpoints * self.out_features),
        )

        self.embedding = nn.Embedding(self.gridpoints, self.embedding_dim)

    def forward(self, x: torch.Tensor, doy: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the Chen model.

        Args:
            x (torch.Tensor): Contains the mean and standard deviation of the input features.
            The predicted features should be listed first in the feature dimension of the input tensor.
            Shape [batch_size, in_features, width, height].
            doy (torch.Tensor): Contains the day of the year feature.

        Returns:
            torch.Tensor: Samples from the Chen model. Shape [batch_size, n_samples_train, height, width, out_features].
        """
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
        delta = rearrange(
            delta, "batch latent_dim -> batch 1 latent_dim"
        )  # Shape [batch_size, 1, latent_dim]
        noise = z * delta  # Shape [batch_size, n_samples_train, latent_dim]

        # Compute the latent representation
        grid_embeddings = self.embedding.weight  # Shape [gridpoints, embedding_dim]
        grid_embeddings_flat = repeat(
            grid_embeddings, "gridpoints emb -> b (gridpoints emb)", b=batch_size
        )  # Shape [batch_size, gridpoints *embedding_dim]

        # Mean has shape [batch_size, lat, lon, in_features]
        # Std has shape [batch_size, lat, lon, in_features]
        # Grid embeddings has shape [batch_size, gridpoints * embedding_dim]
        # Noise has shape [batch_size, n_samples_train, latent_dim]
        # doy has shape [batch_size, 1]

        mean_flat = mean.flatten(
            start_dim=1
        )  # Shape [batch_size, lat * lon * in_features]
        std_flat = std.flatten(
            start_dim=1
        )  # Shape [batch_size, lat * lon * in_features]

        full_input = torch.cat(
            [mean_flat, std_flat, grid_embeddings_flat], dim=-1
        )  # Shape [batch_size, lat * lon * (2 * in_features + embedding_dim)]

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

        std_samples = self.noise_decoder(
            full_input_repeated_noise
        )  # Shape [batch_size * n_samples_train, gridpoints * out_features]

        std_samples = rearrange(
            std_samples,
            "(b n) (h w o) -> b n h w o",
            b=batch_size,
            n=self.n_samples_train,
            h=self.height,
            w=self.width,
            o=self.out_features,
        )

        pred_mean = rearrange(
            pred_mean, "b h w o -> b 1 h w o"
        )  # Shape [batch_size, 1, lat, lon, out_features]

        res = (
            pred_mean + std_samples
        )  # Shape [batch_size, n_samples_train, lat, lon, out_features]

        return self.final_activation(res)
