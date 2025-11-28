"""Engression models for generative ensemble post-processing.

This module implements engression models based on the approach from:
X. Shen et al., "Engression: Extrapolation through the Lens of Distributional Regression"

The models are adapted for grid-based weather forecast post-processing.
"""

from .base import BaseEngressionModel
from .cnn import CNNEngressionModel

__all__ = ["BaseEngressionModel", "CNNEngressionModel"]
