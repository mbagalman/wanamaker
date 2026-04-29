"""Bayesian engine abstraction layer.

Per AGENTS.md: the engine choice (PyMC vs. NumPyro vs. Stan) is the output
of Phase -1 and is **not yet decided**. Feature code must import from
``wanamaker.engine`` and never from a specific library, so we can swap
without rewriting features.

The current leading candidate is PyMC (BRD/PRD Section 9). xgboost is
explicitly *not* a candidate for the modeling role — see Hard Rule 3.
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
