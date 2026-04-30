"""Scenario comparison (FR-5.2).

User supplies 2–3 budget plans; we rank them with uncertainty and flag any
plan that extrapolates beyond the historical observed spend range.
Per FR-3.2, plans involving spend-invariant channels do not get
reallocation recommendations.

This module is engine-neutral: it accepts a ``PosteriorPredictiveEngine``
(the same Protocol consumed by ``forecast()``) and never imports any
specific Bayesian backend. The forecast layer's no-engine-imports
invariant from ``posterior_predictive`` carries through here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from wanamaker.engine.summary import PosteriorSummary
from wanamaker.forecast.posterior_predictive import (
    ExtrapolationFlag,
    PosteriorPredictiveEngine,
    forecast,
    load_plan,
)


@dataclass(frozen=True)
class ScenarioComparisonResult:
    """Outcome of forecasting a single scenario.

    Attributes:
        plan_name: Either the CSV file's stem or ``"Plan {i}"`` for
            DataFrame inputs.
        expected_outcome_mean: Sum of period-wise posterior predictive means
            across the plan horizon.
        expected_outcome_hdi_low: Sum of period-wise lower HDI bounds.
            This is a conservative aggregate, not the true HDI of the sum
            (which would require joint posterior draws).
        expected_outcome_hdi_high: Sum of period-wise upper HDI bounds;
            conservative, see ``expected_outcome_hdi_low``.
        total_spend_by_channel: Total planned spend per channel, computed
            from the same normalised plan that the forecast consumed.
        extrapolation_flags: Per-cell warnings for plans that exceed the
            channel's historical observed spend range.
        spend_invariant_channels: Channels whose saturation could not be
            estimated from training data (FR-3.2). These are excluded from
            reallocation recommendations.
    """

    plan_name: str
    expected_outcome_mean: float
    expected_outcome_hdi_low: float
    expected_outcome_hdi_high: float
    total_spend_by_channel: dict[str, float]
    extrapolation_flags: list[ExtrapolationFlag] = field(default_factory=list)
    spend_invariant_channels: list[str] = field(default_factory=list)


def compare_scenarios(
    posterior_summary: PosteriorSummary,
    plans: list[str | Path | pd.DataFrame],
    seed: int,
    engine: PosteriorPredictiveEngine,
) -> list[ScenarioComparisonResult]:
    """Rank multiple budget plans with uncertainty (FR-5.2).

    Each plan is normalised, forecast via the supplied engine, and summarised
    into a ``ScenarioComparisonResult``. Results are returned ranked by
    descending ``expected_outcome_mean``; ties break on ``plan_name`` so
    ordering is deterministic.

    Args:
        posterior_summary: Engine-neutral summary from a completed fit.
            Must contain ``channel_contributions`` so the spend ranges are
            available for extrapolation flagging.
        plans: 1+ future spend plans, each as a CSV path or a DataFrame in
            wide / long / transposed-wide format. See ``load_plan`` for the
            accepted shapes.
        seed: Posterior-predictive sampling seed.
        engine: Implementation of ``PosteriorPredictiveEngine``. The same
            engine is reused across all plans to keep results comparable.

    Returns:
        List of ``ScenarioComparisonResult``, ranked best-first.

    Raises:
        ValueError: If ``plans`` is empty.
        TypeError: If a plan element is not a CSV path or DataFrame.
        Plus any exception raised by ``forecast()`` for a malformed plan.
    """
    if not plans:
        raise ValueError("compare_scenarios requires at least one plan")

    required_channels = [c.channel for c in posterior_summary.channel_contributions]

    results: list[ScenarioComparisonResult] = []
    for index, plan in enumerate(plans):
        plan_name = _plan_name(plan, index)
        normalized = load_plan(plan, required_channels)
        forecast_result = forecast(posterior_summary, plan, seed, engine)

        total_spend = {
            channel: float(normalized.data[channel].sum())
            for channel in required_channels
        }

        results.append(
            ScenarioComparisonResult(
                plan_name=plan_name,
                expected_outcome_mean=float(sum(forecast_result.mean)),
                expected_outcome_hdi_low=float(sum(forecast_result.hdi_low)),
                expected_outcome_hdi_high=float(sum(forecast_result.hdi_high)),
                total_spend_by_channel=total_spend,
                extrapolation_flags=list(forecast_result.extrapolation_flags),
                spend_invariant_channels=list(forecast_result.spend_invariant_channels),
            )
        )

    return sorted(
        results,
        key=lambda r: (-r.expected_outcome_mean, r.plan_name),
    )


def _plan_name(plan: str | Path | pd.DataFrame, index: int) -> str:
    """Stable display name for a plan input."""
    if isinstance(plan, (str, Path)):
        return Path(plan).stem
    if isinstance(plan, pd.DataFrame):
        return f"Plan {index + 1}"
    raise TypeError(
        f"plan must be a CSV path or pandas DataFrame; got {type(plan).__name__}"
    )
