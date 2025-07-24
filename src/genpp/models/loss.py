import torch
import torch.nn as nn
from einops import rearrange


class EnergyScore(nn.Module):
    def __init__(self) -> None:
        super(EnergyScore, self).__init__()

    # TODO: vibe coded this, need to check if it is correct
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the energy score between the predicted and true values.

        Args:
            x (torch.Tensor): The predicted values with shape [batch_size, n_samples, lat, lon, out_features].
            y (torch.Tensor): The true values with shape [batch_size, lat, lon, out_features].

        Returns:
            torch.Tensor: The computed energy score with shape [out_features].
        """

        batch_size, n_samples, lat, lon, out_features = x.shape

        # Reshape tensors for easier computation
        # x: [batch_size, out_features, n_samples, lat * lon]
        # y: [batch_size, out_features, 1, lat * lon]
        x_reshaped = rearrange(x, "b n lat lon d -> b d n (lat lon)")
        y_reshaped = rearrange(y, "b lat lon d -> b d 1 (lat lon)")

        n_samples_model = float(n_samples)
        eps = 1e-8

        # Calculate first term: E[||y_true - y_pred||]
        # Compute ||y_true||^2 + ||y_pred||^2 - 2 * y_true^T * y_pred
        y_true_norm_sq = torch.sum(
            y_reshaped**2, dim=3, keepdim=True
        )  # [batch_size, out_features, 1, 1]
        y_pred_norm_sq = torch.sum(
            x_reshaped**2, dim=3, keepdim=True
        )  # [batch_size, out_features, n_samples, 1]
        cross_term = torch.matmul(
            y_reshaped, rearrange(x_reshaped, "b d n spatial -> b d spatial n")
        )  # [batch_size, out_features, 1, n_samples]

        distances_12 = (
            y_true_norm_sq
            + rearrange(y_pred_norm_sq, "b d n 1 -> b d 1 n")
            - 2 * cross_term
        )
        distances_12 = torch.clamp(distances_12, min=eps, max=1e10)
        es_12 = torch.sum(
            torch.sqrt(distances_12), dim=(2, 3)
        )  # [batch_size, out_features]

        # Calculate second term: E[||y_pred_i - y_pred_j||] for i != j
        # Compute Gram matrix G = y_pred^T * y_pred
        G = torch.matmul(
            rearrange(x_reshaped, "b d n spatial -> b d spatial n"), x_reshaped
        )  # [batch_size, out_features, n_samples, n_samples]

        # Extract diagonal elements (||y_pred_i||^2)
        d = torch.diagonal(G, dim1=-2, dim2=-1).unsqueeze(
            -1
        )  # [batch_size, out_features, n_samples, 1]

        # Compute pairwise distances: ||y_pred_i||^2 + ||y_pred_j||^2 - 2 * y_pred_i^T * y_pred_j
        distances_22 = d + rearrange(d, "b d n 1 -> b d 1 n") - 2 * G
        distances_22 = torch.clamp(distances_22, min=eps, max=1e10)

        # Sum over all pairs (including diagonal, but we'll account for that)
        es_22 = torch.sum(
            torch.sqrt(distances_22), dim=(2, 3)
        )  # [batch_size, out_features]

        # Subtract diagonal terms (distance from sample to itself = 0)
        # Since sqrt(0) = 0, we don't need to explicitly subtract anything
        # But we need to account for the fact that we're averaging over n*(n-1) pairs, not n^2

        # Final energy score calculation
        energy_score = es_12 / n_samples_model - es_22 / (
            2 * n_samples_model * (n_samples_model - 1)
        )

        # Average across the batch dimension to get shape [out_features]
        energy_score = torch.mean(energy_score, dim=0)

        return energy_score  # [out_features]
