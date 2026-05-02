"""Scenario comparison (FR-5.2).

User supplies 2–3 budget plans; we rank them with uncertainty, flag any
plan that extrapolates beyond the historical observed spend range, and
report decision-grade summary metrics (probability of beating baseline,
downside risk, plain-English interpretation). Per FR-3.2, plans involving
spend-invariant channels do not get reallocation recommendations.

Per AGENTS.md "Product terminology" the output uses cautious decision
language ("ranks first", "directional", "not meaningfully distinguishable")
and avoids optimizer-grade promises ("best budget", "guaranteed lift").

This module is engine-neutral: it accepts a ``PosteriorPredictiveEngine``
(the same Protocol consumed by ``forecast()``) and never imports any
specific Bayesian backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wanamaker.engine.summary import PosteriorSummary
from wanamaker.forecast.posterior_predictive import (
    ExtrapolationFlag,
    PosteriorPredictiveEngine,
    forecast,
    load_plan,
)

# Material loss threshold for the downside-probability metric:
# probability that the plan underperforms baseline by more than this
# fraction of the baseline outcome. Mirrors the ramp module's default
# ``RiskTolerance.min_relative_loss`` so the two surfaces use the same
# downside definition.
_MATERIAL_LOSS_THRESHOLD = 0.05


@dataclass(frozen=True)
class ScenarioComparisonResult:
    """Outcome of forecasting a single scenario.

    The first plan in the input list is treated as the baseline; the
    delta and probability fields below are computed vs. that baseline.
    For the baseline row itself, all delta and probability fields are
    zero / 1.0 / informational (see ``is_baseline``).

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
        is_baseline: True for the first plan in the input list. Delta /
            probability fields are not meaningful on the baseline row;
            consumers should display them as ``—`` or skip them.
        delta_vs_baseline_mean: Posterior mean of (this plan − baseline).
            Computed paired-by-draw using the same seed across forecasts,
            so this captures correlations the per-plan HDIs cannot.
        delta_vs_baseline_hdi_low: 95% HDI lower bound for the delta.
        delta_vs_baseline_hdi_high: 95% HDI upper bound for the delta.
        probability_beats_baseline: Posterior probability that this
            plan's outcome exceeds the baseline's, computed paired by
            draw. ``1.0`` for the baseline row by definition.
        probability_material_loss: Posterior probability that this plan
            underperforms the baseline by more than 5% of the baseline
            mean. ``0.0`` for the baseline row.
        interpretation: One-sentence plain-English summary of how to
            read this scenario relative to baseline. Generated from a
            deterministic decision tree over the metrics above; never
            uses optimizer-grade language (no "best budget", no
            "guaranteed lift").
    """

    plan_name: str
    expected_outcome_mean: float
    expected_outcome_hdi_low: float
    expected_outcome_hdi_high: float
    total_spend_by_channel: dict[str, float]
    extrapolation_flags: list[ExtrapolationFlag] = field(default_factory=list)
    spend_invariant_channels: list[str] = field(default_factory=list)
    is_baseline: bool = False
    delta_vs_baseline_mean: float = 0.0
    delta_vs_baseline_hdi_low: float = 0.0
    delta_vs_baseline_hdi_high: float = 0.0
    probability_beats_baseline: float = 1.0
    probability_material_loss: float = 0.0
    interpretation: str = ""


def compare_scenarios(
    posterior_summary: PosteriorSummary,
    plans: list[str | Path | pd.DataFrame],
    seed: int,
    engine: PosteriorPredictiveEngine,
) -> list[ScenarioComparisonResult]:
    """Rank multiple budget plans with uncertainty (FR-5.2).

    Each plan is normalised, forecast via the supplied engine, and summarised
    into a ``ScenarioComparisonResult``. The first plan in ``plans`` is
    treated as the baseline; delta and probability fields on every other
    result are computed paired-by-draw against it (so correlations between
    plans are preserved).

    Results are returned ranked by descending ``expected_outcome_mean``;
    ties break on ``plan_name`` so ordering is deterministic. The
    baseline plan is always present in the output but its position
    depends on its expected outcome.

    Args:
        posterior_summary: Engine-neutral summary from a completed fit.
            Must contain ``channel_contributions`` so the spend ranges are
            available for extrapolation flagging.
        plans: 1+ future spend plans, each as a CSV path or a DataFrame in
            wide / long / transposed-wide format. See ``load_plan`` for the
            accepted shapes. The first entry is the baseline.
        seed: Posterior-predictive sampling seed. The same seed is passed
            to every per-plan forecast so draws are aligned across plans
            (paired comparison).
        engine: Implementation of ``PosteriorPredictiveEngine``. The same
            engine is reused across all plans to keep results comparable.

    Returns:
        List of ``ScenarioComparisonResult``, ranked best-first by
        expected outcome.

    Raises:
        ValueError: If ``plans`` is empty.
        TypeError: If a plan element is not a CSV path or DataFrame.
        ValueError: If a plan is malformed or cannot be forecast.

    Examples:
        Compare a baseline plan to an aggressive alternative using a stub
        engine that returns three deterministic per-draw outcomes. The first
        plan is the baseline; results come back ranked by descending expected
        outcome (so the winning plan is at index 0). The delta and
        probability fields are computed paired-by-draw against the baseline.

        Engine outcome rule: 2 × ``paid_search`` per period. Three draws
        offset by ``-1, 0, +1`` give a tiny non-degenerate spread without
        sampling noise.

        >>> import pandas as pd
        >>> from wanamaker.engine.summary import (
        ...     ChannelContributionSummary, PosteriorSummary, PredictiveSummary,
        ... )
        >>> from wanamaker.forecast import compare_scenarios
        >>>
        >>> summary = PosteriorSummary(
        ...     channel_contributions=[
        ...         ChannelContributionSummary(
        ...             channel="paid_search", mean_contribution=0.0,
        ...             hdi_low=0.0, hdi_high=0.0,
        ...             observed_spend_min=10.0, observed_spend_max=50.0,
        ...         ),
        ...     ]
        ... )
        >>>
        >>> class StubEngine:
        ...     def posterior_predictive(self, summary, new_data, seed):
        ...         period_mean = (new_data["paid_search"].astype(float) * 2.0).tolist()
        ...         draws = [
        ...             [m - 1.0 for m in period_mean],
        ...             period_mean,
        ...             [m + 1.0 for m in period_mean],
        ...         ]
        ...         return PredictiveSummary(
        ...             periods=new_data["period"].astype(str).tolist(),
        ...             mean=period_mean,
        ...             hdi_low=draws[0], hdi_high=draws[2], draws=draws,
        ...         )
        >>>
        >>> baseline = pd.DataFrame({"period": ["2026-01-05"], "paid_search": [20.0]})
        >>> aggressive = pd.DataFrame({"period": ["2026-01-05"], "paid_search": [40.0]})
        >>> results = compare_scenarios(
        ...     summary, [baseline, aggressive], seed=0, engine=StubEngine(),
        ... )
        >>> [(r.plan_name, r.is_baseline, r.expected_outcome_mean) for r in results]
        [('Plan 2', False, 80.0), ('Plan 1', True, 40.0)]
        >>> winner = results[0]
        >>> winner.delta_vs_baseline_mean       # 80 - 40, paired by draw
        40.0
        >>> winner.probability_beats_baseline   # all three draws improve
        1.0
        >>> winner.probability_material_loss
        0.0
    """
    if not plans:
        raise ValueError("compare_scenarios requires at least one plan")

    required_channels = [c.channel for c in posterior_summary.channel_contributions]

    # Forecast every plan first (in input order) so we can compute
    # paired deltas vs the baseline (plans[0]).
    raw_results: list[tuple[str, dict[str, float], object]] = []
    per_draw_totals: list[np.ndarray | None] = []
    for index, plan in enumerate(plans):
        plan_name = _plan_name(plan, index)
        normalized = load_plan(plan, required_channels)
        forecast_result = forecast(posterior_summary, plan, seed, engine)

        total_spend = {
            channel: float(normalized.data[channel].sum())
            for channel in required_channels
        }
        raw_results.append((plan_name, total_spend, forecast_result))
        per_draw_totals.append(_per_draw_totals(forecast_result))

    baseline_totals = per_draw_totals[0]
    baseline_mean = float(sum(raw_results[0][2].mean))

    results: list[ScenarioComparisonResult] = []
    for index, (plan_name, total_spend, forecast_result) in enumerate(raw_results):
        is_baseline = index == 0
        scenario_totals = per_draw_totals[index]

        if is_baseline or scenario_totals is None or baseline_totals is None:
            delta_mean = 0.0
            delta_hdi_low = 0.0
            delta_hdi_high = 0.0
            p_beats = 1.0 if is_baseline else 0.5
            p_material_loss = 0.0
        else:
            delta = scenario_totals - baseline_totals
            delta_mean = float(delta.mean())
            delta_hdi_low, delta_hdi_high = _hdi(delta, mass=0.95)
            p_beats = float((delta > 0).mean())
            loss_threshold = _MATERIAL_LOSS_THRESHOLD * abs(baseline_mean)
            p_material_loss = float((delta < -loss_threshold).mean())

        interpretation = _interpretation_sentence(
            is_baseline=is_baseline,
            plan_name=plan_name,
            extrapolation_flags=list(forecast_result.extrapolation_flags),
            spend_invariant_channels=list(forecast_result.spend_invariant_channels),
            probability_beats_baseline=p_beats,
            probability_material_loss=p_material_loss,
            delta_hdi_low=delta_hdi_low,
            delta_hdi_high=delta_hdi_high,
        )

        results.append(
            ScenarioComparisonResult(
                plan_name=plan_name,
                expected_outcome_mean=float(sum(forecast_result.mean)),
                expected_outcome_hdi_low=float(sum(forecast_result.hdi_low)),
                expected_outcome_hdi_high=float(sum(forecast_result.hdi_high)),
                total_spend_by_channel=total_spend,
                extrapolation_flags=list(forecast_result.extrapolation_flags),
                spend_invariant_channels=list(forecast_result.spend_invariant_channels),
                is_baseline=is_baseline,
                delta_vs_baseline_mean=delta_mean,
                delta_vs_baseline_hdi_low=delta_hdi_low,
                delta_vs_baseline_hdi_high=delta_hdi_high,
                probability_beats_baseline=p_beats,
                probability_material_loss=p_material_loss,
                interpretation=interpretation,
            )
        )

    return sorted(
        results,
        key=lambda r: (-r.expected_outcome_mean, r.plan_name),
    )


def _per_draw_totals(forecast_result: object) -> np.ndarray | None:
    """Total outcome per posterior-predictive draw, summed across periods.

    Returns ``None`` when the forecast has no per-draw matrix (e.g., a
    summary-only adapter). In that case downstream code falls back to
    point estimates and the probability fields are reported as ``0.5``
    / ``0.0``.
    """
    draws = getattr(forecast_result, "draws", None)
    if draws is None:
        return None
    arr = np.asarray(draws, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr.sum(axis=1)


def _hdi(values: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Return shortest interval containing ``mass`` of the sample draws."""
    if values.size == 0:
        return 0.0, 0.0
    if not 0 < mass <= 1:
        raise ValueError(f"mass must be in (0, 1]; got {mass!r}")
    sorted_values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    interval_size = max(1, int(np.ceil(mass * sorted_values.size)))
    if interval_size >= sorted_values.size:
        return float(sorted_values[0]), float(sorted_values[-1])
    lows = sorted_values[: sorted_values.size - interval_size + 1]
    highs = sorted_values[interval_size - 1 :]
    widths = highs - lows
    best = int(np.argmin(widths))
    return float(lows[best]), float(highs[best])


def _interpretation_sentence(
    *,
    is_baseline: bool,
    plan_name: str,
    extrapolation_flags: list[ExtrapolationFlag],
    spend_invariant_channels: list[str],
    probability_beats_baseline: float,
    probability_material_loss: float,
    delta_hdi_low: float,
    delta_hdi_high: float,
) -> str:
    """Deterministic plain-English read of a scenario vs. the baseline.

    Decision tree, in order:

    1. Baseline row → short identifying sentence.
    2. Spend-invariant channels involved → recommend caveats; FR-3.2.
    3. Plan extrapolates → flag and ask for a controlled test.
    4. Probability of beating baseline ≥ 0.7 with low downside →
       directional positive.
    5. Probability of beating baseline ≤ 0.3 → directional negative.
    6. Delta credible interval straddles zero → not meaningfully
       distinguishable.
    7. Otherwise → mixed signal sentence.

    All sentences avoid optimizer language per AGENTS.md → "Product
    terminology". The terminology guardrail test enforces that.
    """
    if is_baseline:
        return (
            f"`{plan_name}` is the baseline plan; other scenarios are "
            "compared against it."
        )
    if spend_invariant_channels:
        joined = ", ".join(f"`{c}`" for c in spend_invariant_channels)
        return (
            "Reallocation involves spend-invariant channel(s) "
            f"({joined}); FR-3.2 excludes these from reallocation guidance."
        )
    if extrapolation_flags:
        channels = sorted({flag.channel for flag in extrapolation_flags})
        joined = ", ".join(f"`{c}`" for c in channels)
        return (
            f"Plan exceeds historical spend ranges for {joined}; treat as "
            "a controlled-test candidate, not as supported headroom."
        )
    straddles_zero = delta_hdi_low <= 0 <= delta_hdi_high
    if probability_beats_baseline >= 0.7 and probability_material_loss <= 0.10:
        return (
            "Higher expected outcome than baseline "
            f"(P(beats baseline) = {probability_beats_baseline:.0%}) with "
            "limited downside; treat as directional, not as a guaranteed move."
        )
    if probability_beats_baseline <= 0.3:
        return (
            "Lower expected outcome than baseline "
            f"(P(beats baseline) = {probability_beats_baseline:.0%}); the "
            "model does not support this reallocation."
        )
    if straddles_zero:
        return (
            "Delta credible interval straddles zero; not meaningfully "
            "distinguishable from baseline."
        )
    return (
        "Mixed signal: expected outcome differs from baseline but "
        f"P(beats baseline) is {probability_beats_baseline:.0%}; treat as "
        "directional and revisit after more data arrives."
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
