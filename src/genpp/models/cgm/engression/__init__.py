"""Engression models for generative ensemble post-processing.

This module implements engression models based on the approach from:
X. Shen et al., "Engression: Extrapolation through the Lens of Distributional Regression"

The models are adapted for grid-based weather forecast post-processing.
"""

from .base import BaseEngressionDirectModel, BaseEngressionModel, BaseEngressionNoiseModel
from .cnn import CNNEngressionDirectModel, CNNEngressionModel, CNNEngressionNoiseModel

__all__ = [
    "BaseEngressionModel",
    "BaseEngressionNoiseModel",
    "BaseEngressionDirectModel",
    "CNNEngressionModel",  # Backwards compatibility alias (same as CNNEngressionNoiseModel)
    "CNNEngressionNoiseModel",
    "CNNEngressionDirectModel",
]
