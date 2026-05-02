"""Tests for bounded candidate scenario generation (issue #85).

The generator is engine-neutral: it takes a ``PosteriorPredictiveEngine``
and a ``PosteriorSummary``, and validates every candidate plan via
``validate_candidate_spend`` before forecasting. These tests exercise:

- Constraint enforcement: every produced candidate satisfies the
  baseline-vs-candidate validator (no max-channel-change violation,
  total budget preserved when ``hold_total``, no movement above
  ``max_total_moved_budget``, no min/max spend bound violation).
- Channel blocking: locked, excluded, spend-invariant, and
  zero-baseline channels never appear as donors or recipients.
- Deduplication: equivalent plans collapse so ``top_n`` is a count of
  distinct candidates, not raw enumeration steps.
- Historical-support gating: when ``require_historical_support`` is
  on, candidates that step outside observed spend ranges are rejected
  and surfaced in the rejections audit trail.
- Ranking integration: ``compare_scenarios`` is called once with the
  baseline plus every candidate; result labels match the candidate
  ``label`` (not ``Plan N``).
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
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.forecast import (
    resolve_scenario_generation_constraints,
    suggest_scenarios,
)
from wanamaker.forecast.constraints import validate_candidate_spend

# ---------------------------------------------------------------------------
# Stub engine + reusable summary
# ---------------------------------------------------------------------------


class _LinearEngine:
    """Predictive mean is a fixed linear combination of planned spend.

    Returning a non-trivial ``draws`` matrix lets ``compare_scenarios``
    compute paired deltas (otherwise it falls back to 0.5 / 0.0
    placeholders).
    """

    def __init__(self, coefficients: dict[str, float]) -> None:
        self.coefficients = dict(coefficients)
        self.calls = 0

    def posterior_predictive(
        self,
        posterior_summary: PosteriorSummary,  # noqa: ARG002
        new_data: pd.DataFrame,
        seed: int,  # noqa: ARG002
    ) -> PredictiveSummary:
        self.calls += 1
        per_period_mean = pd.Series(0.0, index=new_data.index, dtype=float)
        for channel, coef in self.coefficients.items():
            per_period_mean = per_period_mean + new_data[channel].astype(float) * coef
        # Two draws: a tight band around the mean. Enough to give
        # compare_scenarios a real per-draw matrix without making the
        # test sensitive to sampling noise.
        draws = [
            (per_period_mean * 0.95).tolist(),
            (per_period_mean * 1.05).tolist(),
        ]
        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=per_period_mean.tolist(),
            hdi_low=(per_period_mean * 0.80).tolist(),
            hdi_high=(per_period_mean * 1.20).tolist(),
            draws=draws,
        )


def _summary(
    *,
    spend_invariant: dict[str, bool] | None = None,
    rois: dict[str, float] | None = None,
    spend_ranges: dict[str, tuple[float, float]] | None = None,
) -> PosteriorSummary:
    """Build a posterior summary with three channels: search, tv, affiliate."""
    spend_invariant = spend_invariant or {}
    rois = rois or {"search": 3.0, "tv": 1.0, "affiliate": 0.5}
    spend_ranges = spend_ranges or {
        "search": (10.0, 200.0),
        "tv": (10.0, 200.0),
        "affiliate": (10.0, 200.0),
    }
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel=name,
                mean_contribution=100.0,
                hdi_low=80.0,
                hdi_high=120.0,
                roi_mean=rois[name],
                observed_spend_min=spend_ranges[name][0],
                observed_spend_max=spend_ranges[name][1],
                spend_invariant=spend_invariant.get(name, False),
            )
            for name in ("search", "tv", "affiliate")
        ]
    )


def _baseline(
    *,
    search: float = 100.0,
    tv: float = 100.0,
    affiliate: float = 100.0,
    n_periods: int = 4,
) -> pd.DataFrame:
    """A flat-across-periods baseline plan."""
    per_period = {
        "search": search / n_periods,
        "tv": tv / n_periods,
        "affiliate": affiliate / n_periods,
    }
    return pd.DataFrame(
        {
            "period": [f"2026-W{i + 1:02d}" for i in range(n_periods)],
            "search": [per_period["search"]] * n_periods,
            "tv": [per_period["tv"]] * n_periods,
            "affiliate": [per_period["affiliate"]] * n_periods,
        }
    )


def _config(
    tmp_path,
    *,
    scenario_generation: ScenarioGenerationConfig | None = None,
) -> WanamakerConfig:
    csv = tmp_path / "data.csv"
    csv.write_text(
        "week,revenue,search,tv,affiliate\n2024-01-01,100,10,20,5\n",
        encoding="utf-8",
    )
    return WanamakerConfig(
        data=DataConfig(
            csv_path=csv,
            date_column="week",
            target_column="revenue",
            spend_columns=["search", "tv", "affiliate"],
        ),
        channels=[
            ChannelConfig(name="search", category="paid_search"),
            ChannelConfig(name="tv", category="linear_tv"),
            ChannelConfig(name="affiliate", category="affiliate"),
        ],
        scenario_generation=scenario_generation,
    )


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------


def test_every_candidate_passes_validate_candidate_spend(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=5,
                max_channel_change=0.15,
                max_total_moved_budget=0.20,
            ),
        )
    )
    baseline = _baseline()
    summary = _summary()

    result = suggest_scenarios(
        summary, baseline, constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.candidates, "expected at least one candidate"
    baseline_totals = {
        channel: float(baseline[channel].sum())
        for channel in ("search", "tv", "affiliate")
    }
    for candidate in result.candidates:
        candidate_totals = {
            channel: float(candidate.plan[channel].sum())
            for channel in ("search", "tv", "affiliate")
        }
        validate_candidate_spend(baseline_totals, candidate_totals, constraints)


def test_hold_total_preserved_across_every_candidate(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=5,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )
    baseline = _baseline()
    baseline_total = baseline[["search", "tv", "affiliate"]].sum().sum()

    result = suggest_scenarios(
        _summary(), baseline, constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.candidates
    for candidate in result.candidates:
        candidate_total = candidate.plan[["search", "tv", "affiliate"]].sum().sum()
        assert candidate_total == pytest.approx(baseline_total, rel=1e-9)


def test_max_channel_change_respected_per_channel(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=5,
                max_channel_change=0.10,
                max_total_moved_budget=0.50,
            ),
        )
    )
    baseline = _baseline()

    result = suggest_scenarios(
        _summary(), baseline, constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.candidates
    for candidate in result.candidates:
        for channel in ("search", "tv", "affiliate"):
            base_total = float(baseline[channel].sum())
            new_total = float(candidate.plan[channel].sum())
            relative = abs(new_total - base_total) / base_total
            assert relative <= constraints.max_channel_change + 1e-9


# ---------------------------------------------------------------------------
# Channel blocking
# ---------------------------------------------------------------------------


def test_locked_channel_never_changes(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
                locked_channels=["tv"],
            ),
        )
    )
    baseline = _baseline()

    result = suggest_scenarios(
        _summary(), baseline, constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.blocked_channels.get("tv") == "locked"
    for candidate in result.candidates:
        assert candidate.donor_channel != "tv"
        assert candidate.recipient_channel != "tv"
        assert float(candidate.plan["tv"].sum()) == pytest.approx(
            float(baseline["tv"].sum()), rel=1e-9
        )


def test_excluded_channel_never_used(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
                excluded_channels=["affiliate"],
            ),
        )
    )
    result = suggest_scenarios(
        _summary(), _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.blocked_channels.get("affiliate") == "excluded"
    for candidate in result.candidates:
        assert candidate.donor_channel != "affiliate"
        assert candidate.recipient_channel != "affiliate"


def test_spend_invariant_channel_blocked_with_explanation(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )
    summary = _summary(spend_invariant={"affiliate": True})

    result = suggest_scenarios(
        summary, _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.blocked_channels.get("affiliate") == "spend_invariant"
    for candidate in result.candidates:
        assert candidate.donor_channel != "affiliate"
        assert candidate.recipient_channel != "affiliate"


def test_zero_baseline_channel_blocked(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )
    baseline = _baseline(affiliate=0.0)

    result = suggest_scenarios(
        _summary(), baseline, constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.blocked_channels.get("affiliate") == "zero_baseline"
    for candidate in result.candidates:
        assert candidate.donor_channel != "affiliate"
        assert candidate.recipient_channel != "affiliate"


def test_no_candidates_when_only_one_eligible_channel(tmp_path) -> None:
    """A single eligible channel can't donate to itself; no candidate is possible."""
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=5,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
                locked_channels=["tv"],
                excluded_channels=["affiliate"],
            ),
        )
    )

    result = suggest_scenarios(
        _summary(), _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.candidates == []
    assert "tv" in result.blocked_channels
    assert "affiliate" in result.blocked_channels


# ---------------------------------------------------------------------------
# Deduplication and ordering
# ---------------------------------------------------------------------------


def test_distinct_candidates_no_duplicates(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=5,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )

    result = suggest_scenarios(
        _summary(), _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    signatures = set()
    for candidate in result.candidates:
        signature = tuple(
            (channel, round(float(candidate.plan[channel].sum()), 6))
            for channel in ("affiliate", "search", "tv")
        )
        assert signature not in signatures, "duplicate candidate plan was emitted"
        signatures.add(signature)


def test_top_n_is_a_hard_cap(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=2,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )

    result = suggest_scenarios(
        _summary(), _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert len(result.candidates) <= 2


# ---------------------------------------------------------------------------
# Historical support
# ---------------------------------------------------------------------------


def test_historical_support_gate_records_rejection(tmp_path) -> None:
    """Tighten the recipient's observed range so a 15% increase steps outside it."""
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
                require_historical_support=True,
            ),
        )
    )
    summary = _summary(
        spend_ranges={
            "search": (10.0, 26.0),  # baseline 25/period; 15% bump = 28.75 > 26
            "tv": (10.0, 200.0),
            "affiliate": (10.0, 200.0),
        }
    )

    result = suggest_scenarios(
        summary, _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    rejection_reasons = " ".join(r.reason for r in result.rejections)
    assert "historical" in rejection_reasons.lower(), (
        "expected at least one rejection citing historical support"
    )


def test_historical_support_disabled_allows_extrapolation(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
                require_historical_support=False,
            ),
        )
    )
    summary = _summary(
        spend_ranges={
            "search": (10.0, 26.0),
            "tv": (10.0, 200.0),
            "affiliate": (10.0, 200.0),
        }
    )

    result = suggest_scenarios(
        summary, _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
    )

    assert result.candidates, (
        "with historical support disabled, candidates should still be produced"
    )
    historical_rejections = [
        r for r in result.rejections if "historical" in r.reason.lower()
    ]
    assert historical_rejections == []


# ---------------------------------------------------------------------------
# Ranking integration
# ---------------------------------------------------------------------------


def test_rankings_use_candidate_labels_not_plan_n(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )

    result = suggest_scenarios(
        _summary(), _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
        baseline_label="baseline_plan",
    )

    ranking_names = {ranked.plan_name for ranked in result.rankings}
    expected = {"baseline_plan", *(c.label for c in result.candidates)}
    assert ranking_names == expected
    assert all(
        not name.startswith("Plan ") for name in ranking_names
    ), "candidate ranking still has the placeholder 'Plan N' labels"


def test_baseline_present_in_ranking_with_is_baseline_flag(tmp_path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )

    result = suggest_scenarios(
        _summary(), _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 3.0, "tv": 1.0, "affiliate": 0.5}),
        baseline_label="baseline_plan",
    )

    baseline_rows = [r for r in result.rankings if r.is_baseline]
    assert len(baseline_rows) == 1
    assert baseline_rows[0].plan_name == "baseline_plan"


def test_higher_roi_recipient_is_preferred(tmp_path) -> None:
    """First donor->recipient pair should move from lowest-ROI to highest-ROI."""
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=1,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )

    result = suggest_scenarios(
        _summary(rois={"search": 5.0, "tv": 1.0, "affiliate": 0.2}),
        _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 5.0, "tv": 1.0, "affiliate": 0.2}),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.donor_channel == "affiliate"
    assert candidate.recipient_channel == "search"


# ---------------------------------------------------------------------------
# Empty paths
# ---------------------------------------------------------------------------


def test_empty_when_no_eligible_pair_records_no_rejections(tmp_path) -> None:
    """All channels blocked → no rejections (nothing was attempted), no candidates."""
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                top_n=3,
                max_channel_change=0.15,
                max_total_moved_budget=0.50,
            ),
        )
    )
    summary = _summary(
        spend_invariant={"search": True, "tv": True, "affiliate": True}
    )

    result = suggest_scenarios(
        summary, _baseline(), constraints, seed=0,
        engine=_LinearEngine({"search": 1.0, "tv": 1.0, "affiliate": 1.0}),
    )

    assert result.candidates == []
    assert result.rejections == []
    assert all(
        reason == "spend_invariant"
        for reason in result.blocked_channels.values()
    )
