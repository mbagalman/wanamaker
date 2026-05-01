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


# ---------------------------------------------------------------------------
# Decision-grade metrics (issue #86)
#
# Probability and delta fields are computed from a per-draw outcome matrix
# (PredictiveSummary.draws). DrawsStubEngine returns deterministic draws so
# the tests can assert exact numbers without invoking PyMC.
# ---------------------------------------------------------------------------


class DrawsStubEngine:
    """Deterministic engine that returns fixed per-draw outcomes.

    The same seed produces the same draws across every plan, so paired
    deltas have no within-engine noise — the engine simply maps the
    plan's spend through fixed coefficients with a small fixed offset
    per draw, then sums per period to get a total per draw.
    """

    def __init__(
        self,
        search_coef: float = 2.0,
        tv_coef: float = 0.5,
        draw_offsets: list[float] | None = None,
    ) -> None:
        self.search_coef = search_coef
        self.tv_coef = tv_coef
        # Default offsets create a small spread so the HDI is non-degenerate.
        self.draw_offsets = draw_offsets or [-5.0, -2.0, 0.0, 2.0, 5.0]

    def posterior_predictive(
        self,
        posterior_summary: PosteriorSummary,  # noqa: ARG002
        new_data: pd.DataFrame,
        seed: int,  # noqa: ARG002
    ) -> PredictiveSummary:
        period_means = (
            new_data["search"].astype(float) * self.search_coef
            + new_data["tv"].astype(float) * self.tv_coef
        ).tolist()
        # draws shape = (n_draws, n_periods).
        draws = [
            [m + offset for m in period_means]
            for offset in self.draw_offsets
        ]
        # Use the per-draw matrix's own min/max as conservative HDI bounds.
        column_lows = [min(d[i] for d in draws) for i in range(len(period_means))]
        column_highs = [max(d[i] for d in draws) for i in range(len(period_means))]
        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=period_means,
            hdi_low=column_lows,
            hdi_high=column_highs,
            draws=draws,
        )


class TestBaselineMarker:
    def test_first_input_plan_is_baseline(self) -> None:
        plans = [
            _wide_plan([20.0], [100.0]),  # baseline
            _wide_plan([40.0], [100.0]),
        ]
        results = compare_scenarios(
            _summary(), plans, seed=0, engine=DrawsStubEngine(),
        )
        baseline = next(r for r in results if r.is_baseline)
        assert baseline.plan_name == "Plan 1"
        # And only one row is the baseline.
        assert sum(r.is_baseline for r in results) == 1

    def test_baseline_row_has_zero_delta_and_full_p_beats(self) -> None:
        plans = [
            _wide_plan([20.0], [100.0]),
            _wide_plan([40.0], [100.0]),
        ]
        results = compare_scenarios(
            _summary(), plans, seed=0, engine=DrawsStubEngine(),
        )
        baseline = next(r for r in results if r.is_baseline)
        assert baseline.delta_vs_baseline_mean == 0.0
        assert baseline.delta_vs_baseline_hdi_low == 0.0
        assert baseline.delta_vs_baseline_hdi_high == 0.0
        assert baseline.probability_beats_baseline == 1.0
        assert baseline.probability_material_loss == 0.0


class TestDeltaAndProbability:
    def test_higher_mean_plan_has_positive_delta_and_high_p_beats(self) -> None:
        plans = [
            _wide_plan([20.0], [100.0]),  # mean = 90
            _wide_plan([40.0], [100.0]),  # mean = 130
        ]
        results = compare_scenarios(
            _summary(), plans, seed=0, engine=DrawsStubEngine(),
        )
        non_baseline = next(r for r in results if not r.is_baseline)
        assert non_baseline.delta_vs_baseline_mean == pytest.approx(40.0)
        # Every paired draw improves by exactly +40, so P(beats) = 100%.
        assert non_baseline.probability_beats_baseline == 1.0
        assert non_baseline.probability_material_loss == 0.0

    def test_lower_mean_plan_has_negative_delta_and_low_p_beats(self) -> None:
        plans = [
            _wide_plan([40.0], [100.0]),  # baseline = 130
            _wide_plan([20.0], [100.0]),  # mean = 90
        ]
        results = compare_scenarios(
            _summary(), plans, seed=0, engine=DrawsStubEngine(),
        )
        non_baseline = next(r for r in results if not r.is_baseline)
        assert non_baseline.delta_vs_baseline_mean == pytest.approx(-40.0)
        assert non_baseline.probability_beats_baseline == 0.0
        # 5% material-loss threshold of 130 = 6.5; -40 is far past that.
        assert non_baseline.probability_material_loss == 1.0

    def test_paired_delta_collapses_zero_when_plans_are_identical(self) -> None:
        plan = _wide_plan([20.0], [100.0])
        results = compare_scenarios(
            _summary(), [plan, plan], seed=0, engine=DrawsStubEngine(),
        )
        # The non-baseline duplicate has zero paired delta on every draw.
        non_baseline = next(r for r in results if not r.is_baseline)
        assert non_baseline.delta_vs_baseline_mean == 0.0
        # All draws are exactly equal -> P(strict beat) is 0%.
        assert non_baseline.probability_beats_baseline == 0.0
        assert non_baseline.probability_material_loss == 0.0


class TestInterpretationSentences:
    """Each interpretation bucket should produce a sentence the deterministic
    template can render. The terminology guardrail enforces that none of
    these sentences leaks optimizer language."""

    def test_baseline_sentence_identifies_as_reference(self) -> None:
        plans = [_wide_plan([20.0], [100.0]), _wide_plan([40.0], [100.0])]
        results = compare_scenarios(
            _summary(), plans, seed=0, engine=DrawsStubEngine(),
        )
        baseline = next(r for r in results if r.is_baseline)
        assert "baseline" in baseline.interpretation.lower()

    def test_directional_positive_sentence_for_high_p_beats(self) -> None:
        plans = [
            _wide_plan([20.0], [100.0]),  # baseline
            _wide_plan([40.0], [100.0]),  # P(beats) = 100%
        ]
        # Use a TV-channel summary that's NOT spend-invariant so we hit
        # the "no caveats" branch of the decision tree.
        results = compare_scenarios(
            _summary(spend_invariant_tv=False),
            plans, seed=0, engine=DrawsStubEngine(),
        )
        non_baseline = next(r for r in results if not r.is_baseline)
        assert "P(beats baseline) = 100%" in non_baseline.interpretation
        assert "directional" in non_baseline.interpretation.lower()

    def test_directional_negative_sentence_for_low_p_beats(self) -> None:
        plans = [
            _wide_plan([40.0], [100.0]),  # baseline
            _wide_plan([20.0], [100.0]),  # P(beats) = 0%
        ]
        results = compare_scenarios(
            _summary(spend_invariant_tv=False),
            plans, seed=0, engine=DrawsStubEngine(),
        )
        non_baseline = next(r for r in results if not r.is_baseline)
        assert "Lower expected outcome" in non_baseline.interpretation
        assert "does not support" in non_baseline.interpretation

    def test_extrapolation_sentence_when_plan_exceeds_observed_range(self) -> None:
        plans = [
            _wide_plan([20.0], [100.0]),  # baseline
            _wide_plan([60.0], [100.0]),  # search=60 exceeds observed_max=50
        ]
        results = compare_scenarios(
            _summary(spend_invariant_tv=False),
            plans, seed=0, engine=DrawsStubEngine(),
        )
        non_baseline = next(r for r in results if not r.is_baseline)
        assert "exceeds historical spend ranges" in non_baseline.interpretation
        assert "controlled-test candidate" in non_baseline.interpretation

    def test_spend_invariant_sentence_takes_precedence(self) -> None:
        """Spend-invariant caveats should win the interpretation
        regardless of probability metrics — FR-3.2 is the higher rule."""
        plans = [_wide_plan([20.0], [100.0]), _wide_plan([20.0], [100.0])]
        # tv is spend-invariant in this summary fixture.
        results = compare_scenarios(
            _summary(spend_invariant_tv=True),
            plans, seed=0, engine=DrawsStubEngine(),
        )
        non_baseline = next(r for r in results if not r.is_baseline)
        assert "spend-invariant" in non_baseline.interpretation
        assert "FR-3.2" in non_baseline.interpretation

    def test_indistinguishable_sentence_for_zero_straddling_delta(self) -> None:
        """Direct test of the interpretation helper for the
        ``delta credible interval straddles zero`` bucket.

        Going through ``compare_scenarios`` here would be misleading: a
        deterministic engine with paired draws produces a fully resolved
        delta (paired noise cancels), so it can't naturally hit the
        ``straddles zero`` decision branch. The helper itself is the
        decision tree, so testing it directly is the correct level.
        """
        from wanamaker.forecast.scenarios import _interpretation_sentence

        sentence = _interpretation_sentence(
            is_baseline=False,
            plan_name="alt",
            extrapolation_flags=[],
            spend_invariant_channels=[],
            probability_beats_baseline=0.55,
            probability_material_loss=0.05,
            delta_hdi_low=-100.0,
            delta_hdi_high=200.0,
        )
        assert "not meaningfully distinguishable" in sentence

    def test_mixed_signal_sentence_for_middle_p_beats_with_one_sided_delta(
        self,
    ) -> None:
        """Direct test: P(beats) in the middle band but delta CI fully
        positive routes to the 'mixed signal' bucket."""
        from wanamaker.forecast.scenarios import _interpretation_sentence

        sentence = _interpretation_sentence(
            is_baseline=False,
            plan_name="alt",
            extrapolation_flags=[],
            spend_invariant_channels=[],
            probability_beats_baseline=0.55,
            probability_material_loss=0.05,
            delta_hdi_low=10.0,  # entirely above zero — but P(beats) is mid
            delta_hdi_high=200.0,
        )
        assert "mixed signal" in sentence.lower()


class TestGracefulFallbackWithoutDraws:
    """When the stub engine returns no per-draw matrix, the new fields
    fall back to neutral defaults instead of crashing. The original
    StubEngine in this file does not produce draws, so this is the
    backward-compat path."""

    def test_probabilities_fall_back_to_neutral_without_draws(self) -> None:
        plans = [_wide_plan([20.0], [100.0]), _wide_plan([40.0], [100.0])]
        results = compare_scenarios(_summary(), plans, seed=0, engine=StubEngine())
        non_baseline = next(r for r in results if not r.is_baseline)
        assert non_baseline.probability_beats_baseline == 0.5
        assert non_baseline.probability_material_loss == 0.0
        # Delta fields are zero (the function can't compute paired
        # comparison without draws).
        assert non_baseline.delta_vs_baseline_mean == 0.0


class TestNoBannedTerminology:
    """The interpretation sentences must not leak optimizer language."""

    def test_no_banned_phrases_in_any_interpretation(self) -> None:
        # Mirrors AGENTS.md → "Product terminology" and the canonical
        # list in tests/unit/test_terminology_guardrails.py. Defined
        # locally because tests/ is not a package.
        banned_phrases = (
            "optimized budget",
            "optimal allocation",
            "best budget",
            "guaranteed lift",
            "maximize roi",
        )

        plans = [
            _wide_plan([20.0], [100.0]),
            _wide_plan([40.0], [100.0]),
            _wide_plan([60.0], [100.0]),  # extrapolation
        ]
        # Cover the spend-invariant branch too.
        for spend_invariant in (True, False):
            results = compare_scenarios(
                _summary(spend_invariant_tv=spend_invariant),
                plans, seed=0, engine=DrawsStubEngine(),
            )
            for r in results:
                lowered = r.interpretation.lower()
                for phrase in banned_phrases:
                    assert phrase not in lowered, (
                        f"interpretation for {r.plan_name!r} contains "
                        f"banned phrase {phrase!r}: {r.interpretation!r}"
                    )
