"""Refresh diff report (FR-4.2).

Compares a current run's marginal posterior summaries to a previous run's,
emitting a structured per-parameter movement record. The classification of
each movement lives in ``classify.py``.

Two kinds of estimates are diffed:

- **Scalar parameters** (``PosteriorSummary.parameters``): adstock half-life,
  saturation EC50, saturation slope, ROI coefficients, intercept, controls.
  Identified by their dot-separated ``ParameterSummary.name``.

- **Channel contributions** (``PosteriorSummary.channel_contributions``):
  absolute contribution in target units. Named
  ``"channel.<channel>.contribution"`` in the diff output so they share the
  same namespace as scalar parameters.

Parameters that appear in one summary but not the other (e.g. new channels
added on refresh) are silently skipped — the diff only covers the
intersection of both run's parameter sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wanamaker.engine.summary import PosteriorSummary
    from wanamaker.refresh.classify import MovementClass


@dataclass(frozen=True)
class ParameterMovement:
    """A single parameter's movement between two runs."""

    name: str
    previous_mean: float
    current_mean: float
    previous_ci: tuple[float, float]
    current_ci: tuple[float, float]
    movement_class: "MovementClass | None" = field(default=None)
    """Classification of the movement. ``None`` for diffs produced before
    classification was implemented (schema v1 artifacts)."""


@dataclass(frozen=True)
class RefreshDiff:
    """The full diff between two runs."""

    previous_run_id: str
    current_run_id: str
    movements: list[ParameterMovement]


def compute_diff(
    previous_summary: "PosteriorSummary",
    current_summary: "PosteriorSummary",
    previous_run_id: str,
    current_run_id: str,
) -> RefreshDiff:
    """Compute the movement of every shared parameter between two model fits.

    Iterates over the intersection of parameters present in both summaries,
    builds a ``ParameterMovement`` for each, and classifies the movement
    using ``classify.classify_movement``.

    Channel contributions are included under the name
    ``"channel.<channel>.contribution"`` so they appear alongside scalar
    parameters in the diff output.

    Args:
        previous_summary: ``PosteriorSummary`` from the earlier run.
        current_summary: ``PosteriorSummary`` from the later run.
        previous_run_id: Storage key for the earlier run.
        current_run_id: Storage key for the later run.

    Returns:
        A frozen ``RefreshDiff`` with one ``ParameterMovement`` per shared
        parameter, each carrying its ``MovementClass``.
    """
    from wanamaker.refresh.classify import classify_movement

    # --- scalar parameters ---
    prev_params = {p.name: p for p in previous_summary.parameters}
    curr_params = {p.name: p for p in current_summary.parameters}

    movements: list[ParameterMovement] = []

    for name, curr in curr_params.items():
        prev = prev_params.get(name)
        if prev is None:
            continue
        cls = classify_movement(
            (prev.hdi_low, prev.hdi_high),
            curr.mean,
            (curr.hdi_low, curr.hdi_high),
        )
        movements.append(
            ParameterMovement(
                name=name,
                previous_mean=prev.mean,
                current_mean=curr.mean,
                previous_ci=(prev.hdi_low, prev.hdi_high),
                current_ci=(curr.hdi_low, curr.hdi_high),
                movement_class=cls,
            )
        )

    # --- channel contributions ---
    prev_contribs = {c.channel: c for c in previous_summary.channel_contributions}
    curr_contribs = {c.channel: c for c in current_summary.channel_contributions}

    for channel, curr in curr_contribs.items():
        prev = prev_contribs.get(channel)
        if prev is None:
            continue
        cls = classify_movement(
            (prev.hdi_low, prev.hdi_high),
            curr.mean_contribution,
            (curr.hdi_low, curr.hdi_high),
        )
        movements.append(
            ParameterMovement(
                name=f"channel.{channel}.contribution",
                previous_mean=prev.mean_contribution,
                current_mean=curr.mean_contribution,
                previous_ci=(prev.hdi_low, prev.hdi_high),
                current_ci=(curr.hdi_low, curr.hdi_high),
                movement_class=cls,
            )
        )

    return RefreshDiff(
        previous_run_id=previous_run_id,
        current_run_id=current_run_id,
        movements=movements,
    )
