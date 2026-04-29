"""Refresh diff report (FR-4.2).

Compares a current run's marginal posterior summaries to a previous run's,
emitting a structured per-parameter movement record. The classification of
each movement lives in ``classify.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterMovement:
    """A single parameter's movement between two runs."""

    name: str
    previous_mean: float
    current_mean: float
    previous_ci: tuple[float, float]
    current_ci: tuple[float, float]


@dataclass(frozen=True)
class RefreshDiff:
    """The full diff between two runs."""

    previous_run_id: str
    current_run_id: str
    movements: list[ParameterMovement]


def compute_diff(previous_summary: dict, current_summary: dict) -> RefreshDiff:
    raise NotImplementedError("Phase 1: refresh diff computation")
