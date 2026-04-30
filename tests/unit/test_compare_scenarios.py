from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.forecast.posterior_predictive import ExtrapolationFlag
from wanamaker.forecast.scenarios import ScenarioComparisonResult, compare_scenarios


@dataclass
class MockFitResultContext:
    """Mocks the interface expected by `compare_scenarios` for duck-typing."""

    summary: PosteriorSummary
    seed_used: int | None = None

    def posterior_predictive(
        self,
        summary: PosteriorSummary,
        new_data: pd.DataFrame,
        seed: int,
    ) -> PredictiveSummary:
        self.seed_used = seed

        # We will use the sum of "search" and "tv" to generate a totally deterministic result
        # Plan 1 vs Plan 2 should yield different means based on data
        # For simplicity, mean = sum of spend
        search_spend = new_data.get("search", pd.Series(dtype=float)).fillna(0)
        tv_spend = new_data.get("tv", pd.Series(dtype=float)).fillna(0)

        mean_series = search_spend * 2.0 + tv_spend * 0.5

        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=mean_series.tolist(),
            hdi_low=(mean_series * 0.8).tolist(),
            hdi_high=(mean_series * 1.2).tolist(),
        )


def _summary() -> PosteriorSummary:
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=100.0,
                hdi_low=80.0,
                hdi_high=120.0,
                observed_spend_min=10.0,
                observed_spend_max=50.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=200.0,
                hdi_low=150.0,
                hdi_high=250.0,
                observed_spend_min=100.0,
                observed_spend_max=100.0,
                spend_invariant=True,
            ),
        ]
    )


def test_compare_scenarios_ranking() -> None:
    plan1 = pd.DataFrame({
        "period": ["2026-01-01"],
        "search": [20.0],
        "tv": [100.0],
    })

    plan2 = pd.DataFrame({
        "period": ["2026-01-01"],
        "search": [40.0], # higher search spend = higher outcome (40*2 + 100*0.5 = 130 vs 20*2 + 100*0.5 = 90)
        "tv": [100.0],
    })

    posterior = MockFitResultContext(summary=_summary())

    results = compare_scenarios(posterior, [plan1, plan2], seed=42)

    assert len(results) == 2
    assert results[0].plan_name == "Plan 2" # Sorted descending by mean
    assert results[1].plan_name == "Plan 1"

    assert results[0].expected_outcome_mean == 130.0
    assert results[1].expected_outcome_mean == 90.0

    assert results[0].total_spend_by_channel["search"] == 40.0
    assert results[1].total_spend_by_channel["search"] == 20.0


def test_extrapolation_warnings() -> None:
    plan_extrapolates = pd.DataFrame({
        "period": ["2026-01-01"],
        "search": [60.0], # Max is 50.0
        "tv": [100.0],
    })

    posterior = MockFitResultContext(summary=_summary())

    results = compare_scenarios(posterior, [plan_extrapolates], seed=42)
    res = results[0]

    assert len(res.extrapolation_flags) == 1
    flag = res.extrapolation_flags[0]
    assert flag.channel == "search"
    assert flag.direction == "above_historical_max"
    assert flag.planned_spend == 60.0


def test_spend_invariant_warnings() -> None:
    plan = pd.DataFrame({
        "period": ["2026-01-01"],
        "search": [20.0],
        "tv": [100.0],
    })

    posterior = MockFitResultContext(summary=_summary())

    results = compare_scenarios(posterior, [plan], seed=42)
    res = results[0]

    # TV is spend invariant in _summary()
    assert res.spend_invariant_channels == ["tv"]
