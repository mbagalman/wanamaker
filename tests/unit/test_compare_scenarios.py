"""Tests for ``compare_scenarios`` (issue #21, FR-5.2).

The function is engine-neutral: it consumes a ``PosteriorPredictiveEngine``
Protocol implementation. These tests use a deterministic stub engine so
the behaviour can be exercised without invoking PyMC.

Coverage:

- Ranking direction and deterministic tie-breaking
- All input shapes accepted by ``load_plan``: CSV file, wide DataFrame,
  long format, transposed wide (channel-rows)
- Multi-period plans: HDI bounds and means aggregate correctly
- Both extrapolation directions (above_historical_max, below_historical_min)
- ``spend_invariant_channels`` surfaced from the underlying summary
- ``total_spend_by_channel`` derived from the normalised plan, so it
  agrees with what the forecast actually consumed
- Empty plans raise; bad plan types raise
- Malformed plans surface ``ValueError`` via the ``forecast`` validation
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.forecast.posterior_predictive import ExtrapolationFlag
from wanamaker.forecast.scenarios import (
    ScenarioComparisonResult,
    compare_scenarios,
)

# ---------------------------------------------------------------------------
# Fixtures: a deterministic stub engine and a reusable summary
# ---------------------------------------------------------------------------


class StubEngine:
    """``PosteriorPredictiveEngine`` whose predictive mean is a fixed linear
    combination of the planned spend. Lets the test assert on exact totals."""

    def __init__(self, search_coef: float = 2.0, tv_coef: float = 0.5) -> None:
        self.search_coef = search_coef
        self.tv_coef = tv_coef
        self.last_seed: int | None = None
        self.call_count = 0

    def posterior_predictive(
        self,
        posterior_summary: PosteriorSummary,  # noqa: ARG002
        new_data: pd.DataFrame,
        seed: int,
    ) -> PredictiveSummary:
        self.last_seed = seed
        self.call_count += 1
        mean = (
            new_data["search"].astype(float) * self.search_coef
            + new_data["tv"].astype(float) * self.tv_coef
        )
        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=mean.tolist(),
            hdi_low=(mean * 0.8).tolist(),
            hdi_high=(mean * 1.2).tolist(),
        )


def _summary(spend_invariant_tv: bool = True) -> PosteriorSummary:
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=100.0, hdi_low=80.0, hdi_high=120.0,
                observed_spend_min=10.0, observed_spend_max=50.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=200.0, hdi_low=150.0, hdi_high=250.0,
                observed_spend_min=80.0, observed_spend_max=120.0,
                spend_invariant=spend_invariant_tv,
            ),
        ]
    )


def _wide_plan(
    search: list[float],
    tv: list[float],
    periods: list[str] | None = None,
) -> pd.DataFrame:
    n = len(search)
    return pd.DataFrame(
        {
            "period": periods or [f"2026-01-{i + 1:02d}" for i in range(n)],
            "search": search,
            "tv": tv,
        }
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_ranks_descending_by_expected_outcome(self) -> None:
        plan_a = _wide_plan([20.0], [100.0])  # 2*20 + 0.5*100 = 90
        plan_b = _wide_plan([40.0], [100.0])  # 2*40 + 0.5*100 = 130
        results = compare_scenarios(_summary(), [plan_a, plan_b], seed=0, engine=StubEngine())
        assert [r.expected_outcome_mean for r in results] == pytest.approx([130.0, 90.0])
        assert results[0].plan_name == "Plan 2"
        assert results[1].plan_name == "Plan 1"

    def test_three_plans_returned_in_descending_order(self) -> None:
        plans = [
            _wide_plan([10.0], [100.0]),  # 70
            _wide_plan([30.0], [100.0]),  # 110
            _wide_plan([20.0], [100.0]),  # 90
        ]
        results = compare_scenarios(_summary(), plans, seed=0, engine=StubEngine())
        means = [r.expected_outcome_mean for r in results]
        assert means == sorted(means, reverse=True)

    def test_tie_break_is_deterministic(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        # Two identical plans → same mean → tie. plan_name dictates order.
        results = compare_scenarios(_summary(), [plan, plan], seed=0, engine=StubEngine())
        assert [r.plan_name for r in results] == ["Plan 1", "Plan 2"]


# ---------------------------------------------------------------------------
# Input shapes
# ---------------------------------------------------------------------------


class TestInputShapes:
    def test_csv_file_input_uses_file_stem_as_name(self, tmp_path: Path) -> None:
        path = tmp_path / "q3_aggressive.csv"
        path.write_text(
            textwrap.dedent(
                """\
                period,search,tv
                2026-07-01,30,100
                """
            )
        )
        results = compare_scenarios(_summary(), [path], seed=0, engine=StubEngine())
        assert results[0].plan_name == "q3_aggressive"
        assert results[0].expected_outcome_mean == pytest.approx(110.0)

    def test_csv_path_as_string_works(self, tmp_path: Path) -> None:
        path = tmp_path / "named_plan.csv"
        path.write_text("period,search,tv\n2026-01-01,20,100\n")
        results = compare_scenarios(_summary(), [str(path)], seed=0, engine=StubEngine())
        assert results[0].plan_name == "named_plan"

    def test_long_format_dataframe(self) -> None:
        long = pd.DataFrame(
            {
                "period": ["2026-01-01", "2026-01-01"],
                "channel": ["search", "tv"],
                "spend": [25.0, 100.0],
            }
        )
        results = compare_scenarios(_summary(), [long], seed=0, engine=StubEngine())
        # mean = 2*25 + 0.5*100 = 100
        assert results[0].expected_outcome_mean == pytest.approx(100.0)
        # Spend totals derived from normalised plan, not raw long frame.
        assert results[0].total_spend_by_channel == pytest.approx(
            {"search": 25.0, "tv": 100.0}
        )

    def test_channel_rows_dataframe(self) -> None:
        # Transposed-wide: channel column with one row per channel.
        transposed = pd.DataFrame(
            {
                "channel": ["search", "tv"],
                "2026-01-01": [25.0, 100.0],
                "2026-01-08": [30.0, 100.0],
            }
        )
        results = compare_scenarios(_summary(), [transposed], seed=0, engine=StubEngine())
        # Period 1: 50 + 50 = 100; Period 2: 60 + 50 = 110; sum = 210
        assert results[0].expected_outcome_mean == pytest.approx(210.0)
        assert results[0].total_spend_by_channel == pytest.approx(
            {"search": 55.0, "tv": 200.0}
        )


# ---------------------------------------------------------------------------
# Multi-period and aggregate fields
# ---------------------------------------------------------------------------


class TestAggregateFields:
    def test_multi_period_plan_aggregates_means_and_hdis(self) -> None:
        plan = _wide_plan([20.0, 30.0, 25.0], [100.0, 100.0, 100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        # Period means: 90, 110, 100 → sum 300.
        assert results[0].expected_outcome_mean == pytest.approx(300.0)
        # HDI bounds are sum of period bounds: low = 0.8 * mean, high = 1.2 * mean.
        assert results[0].expected_outcome_hdi_low == pytest.approx(300.0 * 0.8)
        assert results[0].expected_outcome_hdi_high == pytest.approx(300.0 * 1.2)

    def test_total_spend_matches_normalised_plan(self) -> None:
        plan = _wide_plan([10.0, 20.0, 30.0], [100.0, 100.0, 100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        assert results[0].total_spend_by_channel == pytest.approx(
            {"search": 60.0, "tv": 300.0}
        )


# ---------------------------------------------------------------------------
# Extrapolation flags (FR-5.2 acceptance)
# ---------------------------------------------------------------------------


class TestExtrapolationFlags:
    def test_above_historical_max_flag_surfaced(self) -> None:
        # search.observed_spend_max = 50.0
        plan = _wide_plan([60.0], [100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        flags = results[0].extrapolation_flags
        assert len(flags) == 1
        assert flags[0].channel == "search"
        assert flags[0].direction == "above_historical_max"
        assert flags[0].planned_spend == pytest.approx(60.0)

    def test_below_historical_min_flag_surfaced(self) -> None:
        # tv.observed_spend_min = 80.0
        plan = _wide_plan([20.0], [50.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        directions = {f.direction for f in results[0].extrapolation_flags}
        channels = {f.channel for f in results[0].extrapolation_flags}
        assert "below_historical_min" in directions
        assert "tv" in channels

    def test_in_range_plan_has_no_flags(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        assert results[0].extrapolation_flags == []


# ---------------------------------------------------------------------------
# Spend-invariant channels
# ---------------------------------------------------------------------------


class TestSpendInvariantChannels:
    def test_invariant_channel_surfaced_from_summary(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        results = compare_scenarios(
            _summary(spend_invariant_tv=True), [plan], seed=0, engine=StubEngine()
        )
        assert results[0].spend_invariant_channels == ["tv"]

    def test_no_invariant_channels_when_none_marked(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        results = compare_scenarios(
            _summary(spend_invariant_tv=False), [plan], seed=0, engine=StubEngine()
        )
        assert results[0].spend_invariant_channels == []


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


class TestEngineWiring:
    def test_seed_passed_through_to_engine(self) -> None:
        engine = StubEngine()
        compare_scenarios(_summary(), [_wide_plan([20.0], [100.0])], seed=4242, engine=engine)
        assert engine.last_seed == 4242

    def test_engine_called_once_per_plan(self) -> None:
        engine = StubEngine()
        compare_scenarios(
            _summary(),
            [_wide_plan([20.0], [100.0]), _wide_plan([30.0], [100.0])],
            seed=0, engine=engine,
        )
        assert engine.call_count == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_plans_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one plan"):
            compare_scenarios(_summary(), [], seed=0, engine=StubEngine())

    def test_invalid_plan_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="CSV path or pandas DataFrame"):
            compare_scenarios(
                _summary(),
                [12345],  # type: ignore[list-item]
                seed=0, engine=StubEngine(),
            )

    def test_negative_spend_propagates_value_error(self) -> None:
        plan = _wide_plan([-5.0], [100.0])
        with pytest.raises(ValueError, match="negative spend"):
            compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())

    def test_missing_channel_propagates_value_error(self) -> None:
        plan = pd.DataFrame({"period": ["2026-01-01"], "search": [20.0]})  # tv missing
        with pytest.raises(ValueError, match="missing channel columns"):
            compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_result_is_frozen(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        with pytest.raises((AttributeError, TypeError)):
            results[0].plan_name = "other"  # type: ignore[misc]

    def test_extrapolation_flag_type(self) -> None:
        plan = _wide_plan([60.0], [100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        assert all(isinstance(f, ExtrapolationFlag) for f in results[0].extrapolation_flags)

    def test_returned_objects_are_scenario_comparison_results(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        results = compare_scenarios(_summary(), [plan], seed=0, engine=StubEngine())
        assert all(isinstance(r, ScenarioComparisonResult) for r in results)
