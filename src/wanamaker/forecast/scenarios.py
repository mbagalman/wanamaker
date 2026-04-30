"""Scenario comparison (FR-5.2).

User supplies 2–3 budget plans; we rank them with uncertainty and flag any
plan that extrapolates beyond the historical observed spend range.
Per FR-3.2, plans involving spend-invariant channels do not get
reallocation recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from wanamaker.engine.base import FitResult
from wanamaker.engine.pymc import PyMCEngine
from wanamaker.engine.summary import PosteriorSummary
from wanamaker.forecast.posterior_predictive import (
    ExtrapolationFlag,
    ForecastResult,
    PosteriorPredictiveEngine,
    forecast,
)


@dataclass(frozen=True)
class ScenarioComparisonResult:
    """Outcome of forecasting a single scenario."""

    plan_name: str
    expected_outcome_mean: float
    expected_outcome_hdi_low: float
    expected_outcome_hdi_high: float
    total_spend_by_channel: dict[str, float]
    extrapolation_flags: list[ExtrapolationFlag] = field(default_factory=list)
    spend_invariant_channels: list[str] = field(default_factory=list)


def compare_scenarios(
    posterior: Any,
    plans: list[str | Path | pd.DataFrame],
    seed: int,
) -> list[ScenarioComparisonResult]:
    """Rank multiple budget plans with uncertainty (FR-5.2).

    Args:
        posterior: Either a ``FitResult`` containing the posterior summary and raw
            posterior, or a duck-typed test object that exposes ``.summary`` and
            ``.posterior_predictive``.
        plans: List of future spend plans (CSVs or DataFrames).
        seed: Random seed for posterior predictive sampling.

    Returns:
        List of ``ScenarioComparisonResult`` ranked by descending expected outcome.
    """
    if isinstance(posterior, FitResult):
        posterior_summary = posterior.summary
        actual_posterior = posterior.posterior
        pymc_engine = PyMCEngine()

        class _Adapter(PosteriorPredictiveEngine):
            def posterior_predictive(
                self,
                summary: PosteriorSummary,
                new_data: pd.DataFrame,
                seed: int,
            ) -> Any:
                return pymc_engine.posterior_predictive(actual_posterior, new_data, seed)

        engine = _Adapter()
    elif hasattr(posterior, "summary") and hasattr(posterior, "posterior_predictive"):
        posterior_summary = posterior.summary

        class _MockAdapter(PosteriorPredictiveEngine):
            def posterior_predictive(
                self,
                summary: PosteriorSummary,
                new_data: pd.DataFrame,
                seed: int,
            ) -> Any:
                return posterior.posterior_predictive(summary, new_data, seed)

        engine = _MockAdapter()
    else:
        raise TypeError(
            "compare_scenarios requires a FitResult or a test mock exposing "
            "summary and posterior_predictive."
        )

    results: list[ScenarioComparisonResult] = []

    for i, plan in enumerate(plans):
        if isinstance(plan, (str, Path)):
            plan_name = Path(plan).stem
            data = pd.read_csv(plan)
        elif isinstance(plan, pd.DataFrame):
            plan_name = f"Plan {i + 1}"
            data = plan.copy()
        else:
            raise TypeError("plan must be a CSV path or pandas DataFrame")

        channel_names = [
            c.channel for c in posterior_summary.channel_contributions
        ]

        normalized_spend = {}
        for c in channel_names:
            if c in data.columns:
                normalized_spend[c] = float(pd.to_numeric(data[c], errors="coerce").sum())
            else:
                lower_cols = {str(col).lower(): col for col in data.columns}
                if "channel" in lower_cols and "spend" in lower_cols:
                    channel_col = lower_cols["channel"]
                    spend_col = lower_cols["spend"]
                    channel_spend = data[data[channel_col] == c][spend_col]
                    normalized_spend[c] = float(pd.to_numeric(channel_spend, errors="coerce").sum())
                elif "channel" in lower_cols:
                    channel_col = lower_cols["channel"]
                    row = data[data[channel_col] == c]
                    if not row.empty:
                        numeric_row = row.drop(columns=[channel_col]).apply(pd.to_numeric, errors="coerce")
                        row_spend = numeric_row.sum(axis=1).values[0]
                        normalized_spend[c] = float(row_spend)
                    else:
                        normalized_spend[c] = 0.0

        forecast_result = forecast(posterior_summary, plan, seed, engine)

        expected_outcome_mean = sum(forecast_result.mean)
        expected_outcome_hdi_low = sum(forecast_result.hdi_low)
        expected_outcome_hdi_high = sum(forecast_result.hdi_high)

        results.append(
            ScenarioComparisonResult(
                plan_name=plan_name,
                expected_outcome_mean=expected_outcome_mean,
                expected_outcome_hdi_low=expected_outcome_hdi_low,
                expected_outcome_hdi_high=expected_outcome_hdi_high,
                total_spend_by_channel=normalized_spend,
                extrapolation_flags=forecast_result.extrapolation_flags,
                spend_invariant_channels=forecast_result.spend_invariant_channels,
            )
        )

    return sorted(results, key=lambda x: x.expected_outcome_mean, reverse=True)
