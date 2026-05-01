"""Engine protocol — the contract that any backend must implement.

Kept deliberately small. Anything an engine exposes here becomes a constraint
on every candidate (PyMC, NumPyro, Stan), so we add only what features
actually need.

Reproducibility (NFR-2): the engine receives an explicit ``seed`` argument
and must use it for all sampling. No reading of global numpy/random state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from wanamaker.engine.summary import PosteriorSummary


@dataclass(frozen=True)
class Posterior:
    """Engine-agnostic handle to posterior draws.

    The internal representation is engine-native (e.g., an ``arviz.InferenceData``
    for PyMC). Code that needs marginal summaries should go through helper
    functions on this class rather than reaching into ``raw``.
    """

    raw: Any
    """The engine-native posterior object. Treat as opaque."""


@dataclass(frozen=True)
class FitResult:
    """Output of an engine fit."""

    posterior: Posterior
    summary: PosteriorSummary
    """Engine-neutral posterior summary consumed by downstream modules."""
    diagnostics: dict[str, Any]
    """Convergence statistics: r-hat, effective sample size, divergences, etc."""


class Engine(Protocol):
    """The interface every Bayesian engine must satisfy.

    Implementations live alongside this file. The production backend is
    currently ``engine/pymc.py``; the protocol stays intentionally narrow so
    feature code does not depend on backend-native objects.
    """

    name: str
    """Stable identifier (``"pymc"``, ``"numpyro"``, ``"stan"``)."""

    def fit(
        self,
        model_spec: Any,
        data: Any,
        seed: int,
        runtime_mode: str,
    ) -> FitResult:
        """Fit ``model_spec`` to ``data`` with the given seed and runtime tier.

        Args:
            model_spec: The engine-agnostic model specification produced by
                :mod:`wanamaker.model`.
            data: Pre-validated input frame (date, target, transformed media,
                controls).
            seed: Single integer seed; the engine must use only this seed
                for sampling, never global state.
            runtime_mode: One of ``"quick"``, ``"standard"``, ``"full"`` —
                see FR-3.5 for the runtime tier contract.
        """
        ...

    def posterior_predictive(
        self,
        posterior: Posterior,
        new_data: Any,
        seed: int,
    ) -> Any:
        """Draw from the posterior predictive distribution for new inputs."""
        ...
