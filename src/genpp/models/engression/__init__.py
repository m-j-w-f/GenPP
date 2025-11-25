"""Engression models for generative ensemble post-processing.

This module implements engression models based on the approach from:
X. Shen et al., "Engression: Extrapolation through the Lens of Distributional Regression"

The models are adapted for grid-based weather forecast post-processing.
"""

from genpp.models.engression.base import EngressionModel
from genpp.models.engression.cnn import CNNEngressionModel

__all__ = ["EngressionModel", "CNNEngressionModel"]
