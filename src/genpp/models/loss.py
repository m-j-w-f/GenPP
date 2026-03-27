"""Scoring rules for probabilistic predictions.

.. deprecated::
    This module is deprecated. Import from ``genpp.models.scores`` instead.
    This file is kept for backward compatibility and will be removed in a future version.

Example migration::

    # Old (deprecated)
    from genpp.models.loss import EnergyScore

    # New (recommended)
    from genpp.models.scores import EnergyScore
"""

import warnings

# Re-export everything from the new scores module for backward compatibility
from .scores import (
    CRPS_Normal,
    CRPS_TruncatedNormal,
    EnergyScore,
    EnsembleCRPS,
    MultiScaleEnergyScore,
    MultiScalePatchwiseEnergyScore,
    MultiScalePatchwiseRBFScore,
    MultiScaleRBFScore,
    PatchwiseEnergyScore,
    PatchwiseRBFScore,
    RBFScore,
    VariogramScore,
)

# Emit an import-time deprecation warning so that typical imports like
# "from genpp.models.loss import EnergyScore" are clearly flagged.
warnings.warn(
    "Importing from 'genpp.models.loss' is deprecated and will be removed in a future version. "
    "Please import scoring classes from 'genpp.models.scores' instead.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    # Energy scores
    "EnergyScore",
    "PatchwiseEnergyScore",
    "MultiScaleEnergyScore",
    "MultiScalePatchwiseEnergyScore",
    # RBF scores
    "RBFScore",
    "PatchwiseRBFScore",
    "MultiScalePatchwiseRBFScore",
    "MultiScaleRBFScore",
    # Variogram score
    "VariogramScore",
    # CRPS (distribution-based)
    "CRPS_Normal",
    "CRPS_TruncatedNormal",
    # CRPS (sample-based)
    "EnsembleCRPS",
]
