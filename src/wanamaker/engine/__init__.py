"""Bayesian engine abstraction layer.

PyMC is the selected Bayesian engine, but feature code must import from
``wanamaker.engine`` rather than importing PyMC directly. PyMC-specific code
belongs in ``wanamaker.engine.pymc`` and backend-specific tests.

xgboost is explicitly not a candidate for the modeling role; see Hard Rule 3.
"""

from wanamaker.engine.base import Engine, FitResult, Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
    PredictiveSummary,
)

__all__ = [
    "Engine",
    "FitResult",
    "Posterior",
    "PosteriorSummary",
    "ParameterSummary",
    "ChannelContributionSummary",
    "PredictiveSummary",
    "ConvergenceSummary",
]
