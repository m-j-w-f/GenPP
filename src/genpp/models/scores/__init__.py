"""Scoring rules for probabilistic predictions.

This module provides various scoring functions for evaluating probabilistic
forecasts, including:

- Energy Score and variants (EnergyScore, PatchwiseEnergyScore)
- RBF-based scores (RBFScore, PatchwiseRBFScore, MultiPatchwiseRBFScore, MultiScaleRBFScore)
- Variogram Score (VariogramScore)
- CRPS for parametric distributions (CRPS_Normal, CRPS_TruncatedNormal)
- Sample-based CRPS (EnsembleCRPS)
"""

from .crps import CRPS_Normal, CRPS_TruncatedNormal, EnsembleCRPS, crps_ensemble
from .energy import (
    EnergyScore,
    MultiScaleEnergyScore,
    MultiScalePatchwiseEnergyScore,
    PatchwiseEnergyScore,
    energy_score,
)
from .rbf import MultiScalePatchwiseRBFScore, MultiScaleRBFScore, PatchwiseRBFScore, RBFScore
from .variogram import VariogramScore

__all__ = [
    "EnergyScore",
    "PatchwiseEnergyScore",
    "MultiScaleEnergyScore",
    "MultiScalePatchwiseEnergyScore",
    "energy_score",
    "RBFScore",
    "PatchwiseRBFScore",
    "MultiScaleRBFScore",
    "MultiScalePatchwiseRBFScore",
    "VariogramScore",
    "CRPS_Normal",
    "CRPS_TruncatedNormal",
    "EnsembleCRPS",
    "crps_ensemble",
]
