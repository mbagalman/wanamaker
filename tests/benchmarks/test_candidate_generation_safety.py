"""Safety benchmarks for candidate scenario generation (issue #89).

Tests that generated candidate plans obey safety constraints and remain
compatible with scenario comparison and ramp recommendations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wanamaker.config import (
    ChannelConfig,
    DataConfig,
    ScenarioGenerationConfig,
    WanamakerConfig,
)
from wanamaker.forecast.constraints import resolve_scenario_generation_constraints
from wanamaker.forecast.generator import suggest_scenarios
from wanamaker.engine.summary import ChannelContributionSummary, PosteriorSummary, PredictiveSummary
from wanamaker.forecast.posterior_predictive import PosteriorPredictiveEngine

class _MockEngine(PosteriorPredictiveEngine):
    def posterior_predictive(
        self, posterior_summary: PosteriorSummary, new_data: pd.DataFrame, seed: int
    ) -> PredictiveSummary:
        return PredictiveSummary(periods=new_data["period"].tolist(),
            mean=new_data.sum(axis=1, numeric_only=True).tolist(),
            hdi_low=(new_data.sum(axis=1, numeric_only=True) * 0.8).tolist(),
            hdi_high=(new_data.sum(axis=1, numeric_only=True) * 1.2).tolist(),
            draws=[new_data.sum(axis=1, numeric_only=True).tolist()]
        )

@pytest.fixture
def base_config():
    return WanamakerConfig(
        data=DataConfig(
            csv_path="dummy.csv",
            date_column="week",
            target_column="sales",
            spend_columns=["tv", "search", "social"],
        ),
        channels=[
            ChannelConfig(name="tv", category="linear_tv"),
            ChannelConfig(name="search", category="paid_search"),
            ChannelConfig(name="social", category="paid_social"),
        ],
        scenario_generation=ScenarioGenerationConfig(
            budget_mode="hold_total",
            max_channel_change=0.20,
            max_total_moved_budget=0.10,
            require_historical_support=False,
        )
    )

@pytest.fixture
def baseline_data():
    return pd.DataFrame({
        "week": pd.date_range("2024-01-01", periods=10, freq="W").astype(str),
        "tv": [100.0] * 10,       # Total 1000
        "search": [200.0] * 10,   # Total 2000
        "social": [50.0] * 10,    # Total 500
    })

@pytest.fixture
def posterior_summary():
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=1500.0,
                roi_mean=1.5,
                spend_invariant=False,
                observed_spend_min=50.0,
                observed_spend_max=150.0,
                hdi_low=1000.0,
                hdi_high=2000.0,
            ),
            ChannelContributionSummary(
                channel="search",
                mean_contribution=4000.0,
                roi_mean=2.0,
                spend_invariant=False,
                observed_spend_min=100.0,
                observed_spend_max=300.0,
                hdi_low=3000.0,
                hdi_high=5000.0,
            ),
            ChannelContributionSummary(
                channel="social",
                mean_contribution=250.0,
                roi_mean=0.5,
                spend_invariant=False,
                observed_spend_min=20.0,
                observed_spend_max=80.0,
                hdi_low=100.0,
                hdi_high=400.0,
            ),
        ]
    )

@pytest.mark.benchmark
def test_benchmark_naive_optimizer_unsafe_move_rejected(benchmark, base_config, baseline_data, posterior_summary):
    # A naive optimizer would move all budget from social (ROI 0.5) to search (ROI 2.0).
    # But Wanamaker should bound it by max_channel_change and max_total_moved_budget.
    constraints = resolve_scenario_generation_constraints(base_config)
    engine = _MockEngine()

    result = benchmark(suggest_scenarios,
        posterior_summary,
        baseline_data,
        constraints,
        seed=42,
        engine=engine,
    )

    # Assert generated candidates don't exceed constraints
    for candidate in result.candidates:
        assert candidate.moved_share <= constraints.max_total_moved_budget + 1e-9

        # Check channel change ratios
        baseline_sums = baseline_data.sum(numeric_only=True)
        candidate_sums = candidate.plan.sum(numeric_only=True)
        for channel in ["tv", "search", "social"]:
            change = abs(candidate_sums[channel] - baseline_sums[channel]) / baseline_sums[channel]
            assert change <= constraints.max_channel_change + 1e-9

        # Check total budget held
        assert abs(candidate_sums.sum() - baseline_sums.sum()) < 1e-9

@pytest.mark.benchmark
def test_benchmark_constrained_to_locked_channels(benchmark, base_config, baseline_data, posterior_summary):
    # Lock 'search' and see if it is preserved
    base_config.scenario_generation = ScenarioGenerationConfig(
        budget_mode="hold_total",
        max_channel_change=0.20,
        max_total_moved_budget=0.10,
        locked_channels=["search"],
        require_historical_support=False,
    )
    constraints = resolve_scenario_generation_constraints(base_config)
    engine = _MockEngine()

    result = benchmark(suggest_scenarios,
        posterior_summary,
        baseline_data,
        constraints,
        seed=42,
        engine=engine,
    )

    for candidate in result.candidates:
        baseline_sums = baseline_data.sum(numeric_only=True)
        candidate_sums = candidate.plan.sum(numeric_only=True)
        # Search should remain unchanged
        assert abs(candidate_sums["search"] - baseline_sums["search"]) < 1e-9

@pytest.mark.benchmark
def test_benchmark_no_plans_if_all_unsafe(benchmark, base_config, baseline_data, posterior_summary):
    # If all options exceed bounds, it should generate no plans and provide proper rejection reason.
    # Exclude all channels or lock them
    base_config.scenario_generation = ScenarioGenerationConfig(
        budget_mode="hold_total",
        max_channel_change=0.20,
        max_total_moved_budget=0.10,
        locked_channels=["search", "tv", "social"],
        require_historical_support=False,
    )
    constraints = resolve_scenario_generation_constraints(base_config)
    engine = _MockEngine()

    result = benchmark(suggest_scenarios,
        posterior_summary,
        baseline_data,
        constraints,
        seed=42,
        engine=engine,
    )

    assert len(result.candidates) == 0
    assert result.blocked_channels["search"] == "locked"
    assert result.blocked_channels["tv"] == "locked"
    assert result.blocked_channels["social"] == "locked"

@pytest.mark.benchmark
def test_benchmark_spend_invariant_channels(benchmark, base_config, baseline_data, posterior_summary):
    # Make social spend_invariant, it shouldn't be used as source or destination
    from dataclasses import replace
    new_contributions = list(posterior_summary.channel_contributions)
    new_contributions[2] = replace(new_contributions[2], spend_invariant=True)
    posterior_summary = replace(posterior_summary, channel_contributions=new_contributions)
    constraints = resolve_scenario_generation_constraints(base_config)
    engine = _MockEngine()

    result = benchmark(suggest_scenarios,
        posterior_summary,
        baseline_data,
        constraints,
        seed=42,
        engine=engine,
    )

    for candidate in result.candidates:
        assert candidate.donor_channel != "social"
        assert candidate.recipient_channel != "social"
        baseline_sums = baseline_data.sum(numeric_only=True)
        candidate_sums = candidate.plan.sum(numeric_only=True)
        assert abs(candidate_sums["social"] - baseline_sums["social"]) < 1e-9


@pytest.mark.benchmark
def test_benchmark_compatibility_with_ramp_recommendations(benchmark, base_config, baseline_data, posterior_summary):
    # Tests that generated plans can be passed to ramp recommendation
    from wanamaker.forecast.ramp import recommend_ramp
    from wanamaker.forecast.scenarios import compare_scenarios

    constraints = resolve_scenario_generation_constraints(base_config)
    engine = _MockEngine()

    result = benchmark(suggest_scenarios,
        posterior_summary,
        baseline_data,
        constraints,
        seed=42,
        engine=engine,
        baseline_label="Baseline",
    )

    # Assert plans are compatible with compare_scenarios which rank uses
    plans = [baseline_data] + [c.plan for c in result.candidates]
    comparisons = compare_scenarios(posterior_summary, plans, seed=42, engine=engine)

    assert len(comparisons) == 1 + len(result.candidates)

    # We can also attempt a ramp on one of the candidates vs the baseline
    assert len(result.candidates) > 0
    ramp_rec = recommend_ramp(
        posterior_summary,
        baseline_data,
        result.candidates[0].plan,
        seed=42,
        engine=engine,
        trust_card=None,
    )

    # Make sure ramp generation works on the same config
    assert ramp_rec is not None

@pytest.mark.benchmark
def test_benchmark_weak_trust_card_dimensions_reduce_candidate_moves(benchmark, base_config, baseline_data, posterior_summary):
    # This tests the ramp integration with TrustCard
    from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus
    from wanamaker.forecast.ramp import recommend_ramp

    trust_card = TrustCard(
        dimensions=[
            TrustDimension(name="data_recency", status=TrustStatus.WEAK, explanation="Old data")
        ]
    )

    constraints = resolve_scenario_generation_constraints(base_config)
    engine = _MockEngine()

    result = benchmark(suggest_scenarios,
        posterior_summary,
        baseline_data,
        constraints,
        seed=42,
        engine=engine,
        baseline_label="Baseline",
    )

    if len(result.candidates) > 0:
        # A weak trust card dimension typically caps recommendations lower.
        ramp_rec_weak = recommend_ramp(
            posterior_summary,
            baseline_data,
            result.candidates[0].plan,
            seed=42,
            engine=engine,
            trust_card=trust_card,
        )

        ramp_rec_strong = recommend_ramp(
            posterior_summary,
            baseline_data,
            result.candidates[0].plan,
            seed=42,
            engine=engine,
            trust_card=TrustCard(dimensions=[]), # No weak dimensions
        )

        # Depending on ramp logic, weak dimensions block moves or reduce fractional_kelly.
        # Ensure that trust card has an effect
        assert ramp_rec_weak.candidates[0].fractional_kelly < ramp_rec_strong.candidates[0].fractional_kelly or                not ramp_rec_weak.candidates[0].passes or                len(ramp_rec_weak.candidates[0].failed_gates) > len(ramp_rec_strong.candidates[0].failed_gates)
