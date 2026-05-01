"""Risk-adjusted allocation ramps (FR-5.6, design note: docs/risk_adjusted_allocation.md).

Given a baseline plan ``x0`` and a model-favored target plan ``x_star``,
``recommend_ramp`` evaluates a ladder of partial moves
``x(f) = x0 + f * (x_star - x0)`` and recommends the largest ``f`` that
passes a fixed gate set: posterior probability of improvement, downside
probability of material loss, lower-tail (CVaR_5) tolerance, Trust Card
gating, extrapolation severity, spend-invariant channel exclusion, and a
fractional-Kelly cap clamped to ``[0, 1]``.

The output is a discrete verdict — ``proceed``, ``stage``, ``test_first``,
``do_not_recommend`` — not a continuous "optimized budget." This is
deliberate: the design note explains that v1 stays user-driven and
conservative, with the user supplying both plans.

This module is engine-neutral. Like ``forecast()`` and
``compare_scenarios()``, it consumes a ``PosteriorPredictiveEngine`` and
never imports any specific Bayesian backend. The risk math runs over
the per-draw outcome matrix exposed in ``PredictiveSummary.draws``
(added in #64).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wanamaker.engine.summary import PosteriorSummary, PredictiveSummary
from wanamaker.forecast.posterior_predictive import (
    ExtrapolationFlag,
    PosteriorPredictiveEngine,
    load_plan,
)
from wanamaker.trust_card.card import TrustCard, TrustStatus

# v1 ramp ladder. 0.0 is *not* a candidate — it is the implicit
# do-nothing fallback we drop to when no positive fraction passes.
_RAMP_LADDER: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)

# Trust Card → ramp cap. Any weak dimension materially related to the
# move caps the recommendation hard; any moderate dim caps it softly;
# clean cards leave the cap alone.
_TRUST_RAMP_CAP_WEAK = 0.10
_TRUST_RAMP_CAP_MODERATE = 0.50
_TRUST_RAMP_CAP_PASS = 1.00

# Trust Card → Kelly multiplier (see design note's "Kelly-Inspired
# Sizing" section). Mirrors the same pass / moderate / weak buckets.
_KELLY_MULTIPLIER_WEAK = 0.10
_KELLY_MULTIPLIER_MODERATE = 0.25
_KELLY_MULTIPLIER_PASS = 0.50

RampStatus = Literal["proceed", "stage", "test_first", "do_not_recommend"]

# Gate names that block on uncertainty / evidence quality (a future
# experiment or refresh would change the answer). When *only* these
# gates fail, we route to ``test_first`` rather than
# ``do_not_recommend``.
_EVIDENCE_GATES: frozenset[str] = frozenset(
    {"trust_card", "extrapolation", "fractional_kelly"}
)


@dataclass(frozen=True)
class RiskTolerance:
    """User-tunable risk preferences for a ramp recommendation.

    Defaults are deliberately conservative for v1; thresholds are not
    exposed via CLI yet (issue #66 may add them). They are kept here so
    the math layer doesn't bake in magic numbers.
    """

    min_relative_loss: float = 0.05
    """Material-loss threshold expressed as a fraction of baseline outcome."""

    min_absolute_loss: float = 0.0
    """Absolute floor for the material-loss threshold (target-unit dollars)."""

    p_positive_threshold: float = 0.75
    """Minimum P(delta > 0) for a candidate to pass."""

    p_material_loss_threshold: float = 0.10
    """Maximum P(delta < -loss_threshold) for a candidate to pass."""

    cvar_relative_tolerance: float = 2.0
    """``CVaR_5`` must be no worse than ``-cvar_relative_tolerance * loss_threshold``."""

    extrapolation_severity_cap: float = 0.25
    """Max planned-spend overshoot above ``observed_spend_max`` for a candidate
    to pass extrapolation gating, expressed as a fraction (e.g. 0.25 = 25 %
    above the historical max). Below-historical-min is gated identically."""

    kelly_multiplier_override: float | None = None
    """If set, replaces the Trust-Card-derived Kelly multiplier. The default
    multipliers (0.5 / 0.25 / 0.10 for pass / moderate / weak) are
    intentionally conservative for v1; an override lets the caller tune the
    Kelly cap when supplemental evidence (e.g. recent lift tests) justifies
    a less restrictive ceiling. Tests also use this hook to exercise the
    full ramp ladder without changing the Trust Card surface."""


@dataclass(frozen=True)
class RampCandidate:
    """One ramp fraction's metrics, gate verdict, and reported context.

    All fields are JSON-friendly so a ``RampRecommendation`` round-trips
    through the artifact envelope.
    """

    fraction: float
    total_spend_by_channel: dict[str, float]
    expected_increment: float
    probability_positive: float
    probability_material_loss: float
    q05_increment: float
    cvar_5: float
    largest_move_share: float
    fractional_kelly: float
    extrapolation_flags: list[ExtrapolationFlag] = field(default_factory=list)
    passes: bool = False
    failed_gates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RampRecommendation:
    """Final verdict for a baseline → target ramp evaluation."""

    baseline_plan_name: str
    target_plan_name: str
    recommended_fraction: float
    status: RampStatus
    candidates: list[RampCandidate]
    explanation: str
    blocking_reason: str | None = None
    """When non-empty, names the up-front block (e.g.
    ``"spend_invariant_reallocation"``) that short-circuited the
    recommendation before any candidate metrics were computed."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def recommend_ramp(
    posterior_summary: PosteriorSummary,
    baseline_plan: str | Path | pd.DataFrame,
    target_plan: str | Path | pd.DataFrame,
    seed: int,
    engine: PosteriorPredictiveEngine,
    *,
    trust_card: TrustCard | None = None,
    risk_tolerance: RiskTolerance | None = None,
) -> RampRecommendation:
    """Recommend the largest defensible move from ``baseline_plan`` toward ``target_plan``.

    Args:
        posterior_summary: Engine-neutral summary from a completed fit.
            Must contain ``channel_contributions`` so per-channel
            extrapolation flagging and spend-invariant detection work.
        baseline_plan: The user's current or status-quo plan.
        target_plan: The user's preferred candidate plan, typically the
            top-ranked entry from ``compare_scenarios``. Must align with
            ``baseline_plan`` on periods and channels.
        seed: Posterior-predictive sampling seed. Reused for every
            engine call in this evaluation so the ladder of ramped plans
            is comparable.
        engine: Posterior-predictive engine. Must populate
            ``PredictiveSummary.draws`` (added in #64) — risk metrics
            need the per-draw outcome matrix.
        trust_card: Optional Trust Card from the same fit. ``weak`` and
            ``moderate`` dimensions cap the ramp; ``None`` is treated
            equivalently to a ``pass`` card for gating but still
            documented in the explanation.
        risk_tolerance: Optional thresholds. ``None`` uses conservative
            v1 defaults.

    Returns:
        A ``RampRecommendation`` with one of four discrete statuses:
        ``proceed`` (full move passes), ``stage`` (a partial move
        passes), ``test_first`` (failures are evidence-quality
        problems an experiment could resolve), or
        ``do_not_recommend`` (failures are about expected value or
        downside risk).

    Raises:
        ValueError: If the summary has no channel contributions, the
            two plans don't align, the engine doesn't return per-draw
            outcomes, or the plans are otherwise malformed.
    """
    risk_tolerance = risk_tolerance or RiskTolerance()
    contributions = list(posterior_summary.channel_contributions)
    if not contributions:
        raise ValueError(
            "posterior_summary.channel_contributions is empty; "
            "ramp evaluation needs at least one channel."
        )

    required_channels = [c.channel for c in contributions]
    baseline = load_plan(baseline_plan, required_channels)
    target = load_plan(target_plan, required_channels)
    _validate_plans_align(baseline, target)

    baseline_name = _plan_name(baseline_plan, "baseline")
    target_name = _plan_name(target_plan, "target")

    # Up-front block: any spend-invariant channel that would be moved by
    # the target plan triggers ``do_not_recommend`` regardless of
    # expected value. Saturation can't be learned from the data for
    # those channels (FR-3.2), so the model is in no position to
    # endorse reallocating them.
    invariant_reallocated = _spend_invariant_reallocations(
        contributions, baseline.data, target.data,
    )
    if invariant_reallocated:
        names = ", ".join(sorted(invariant_reallocated))
        return RampRecommendation(
            baseline_plan_name=baseline_name,
            target_plan_name=target_name,
            recommended_fraction=0.0,
            status="do_not_recommend",
            candidates=[],
            explanation=(
                f"Target plan reallocates spend-invariant channel(s) "
                f"({names}). Saturation cannot be estimated from "
                "training data for those channels (FR-3.2), so the "
                "model is not in a position to endorse moving spend "
                "into or out of them. Run a controlled experiment or "
                "vary spend over a longer history before reconsidering "
                "this reallocation."
            ),
            blocking_reason="spend_invariant_reallocation",
        )

    # Engine call on baseline once — needed for normalisation across
    # all candidates and for Kelly's r_s = delta / baseline_outcome.
    baseline_predictive = _engine_call(engine, posterior_summary, baseline.data, seed)
    baseline_draws = _draws_array(baseline_predictive, "baseline")
    baseline_outcome_s = baseline_draws.sum(axis=1)
    baseline_outcome_mean = float(baseline_outcome_s.mean())

    # Engine call on the full target — used both to seed the Kelly
    # fraction and as the f=1.0 candidate's draws.
    target_predictive = _engine_call(engine, posterior_summary, target.data, seed)
    target_draws = _draws_array(target_predictive, "target")
    delta_full_s = target_draws.sum(axis=1) - baseline_outcome_s

    fractional_kelly = _fractional_kelly(
        delta_full_s, baseline_outcome_s, trust_card,
        override=risk_tolerance.kelly_multiplier_override,
    )
    ramp_cap = _trust_card_ramp_cap(trust_card)

    loss_threshold = max(
        risk_tolerance.min_absolute_loss,
        baseline_outcome_mean * risk_tolerance.min_relative_loss,
    )
    cvar_tolerance = -risk_tolerance.cvar_relative_tolerance * loss_threshold

    # Build candidates. The f=1.0 candidate reuses ``target_draws`` to
    # avoid a duplicate engine call.
    candidates: list[RampCandidate] = []
    for fraction in _RAMP_LADDER:
        if fraction == 1.0:
            candidate_data = target.data
            draws = target_draws
        else:
            candidate_data = _interpolate_plan(baseline.data, target.data, fraction)
            candidate_predictive = _engine_call(
                engine, posterior_summary, candidate_data, seed,
            )
            draws = _draws_array(candidate_predictive, f"f={fraction}")

        candidates.append(
            _build_candidate(
                fraction=fraction,
                candidate_data=candidate_data,
                baseline_data=baseline.data,
                draws=draws,
                baseline_outcome_s=baseline_outcome_s,
                contributions=contributions,
                loss_threshold=loss_threshold,
                cvar_tolerance=cvar_tolerance,
                fractional_kelly=fractional_kelly,
                ramp_cap=ramp_cap,
                risk_tolerance=risk_tolerance,
                trust_card=trust_card,
            )
        )

    return _build_recommendation(
        candidates=candidates,
        baseline_name=baseline_name,
        target_name=target_name,
        trust_card=trust_card,
        fractional_kelly=fractional_kelly,
    )


# ---------------------------------------------------------------------------
# Plan handling
# ---------------------------------------------------------------------------


def _validate_plans_align(baseline: Any, target: Any) -> None:
    if list(baseline.periods) != list(target.periods):
        raise ValueError(
            "baseline and target plans must cover identical periods; got "
            f"baseline={list(baseline.periods)} target={list(target.periods)}"
        )
    if list(baseline.data.columns) != list(target.data.columns):
        raise ValueError(
            "baseline and target plans must have identical channel columns "
            "after normalisation."
        )


def _interpolate_plan(
    baseline: pd.DataFrame, target: pd.DataFrame, fraction: float,
) -> pd.DataFrame:
    """Linear interpolation ``x(f) = x0 + f * (x_star - x0)`` per channel."""
    out = baseline.copy()
    for column in baseline.columns:
        if column == "period":
            continue
        out[column] = (
            baseline[column].astype(float)
            + fraction * (target[column].astype(float) - baseline[column].astype(float))
        )
    return out


def _plan_name(plan: str | Path | pd.DataFrame, fallback: str) -> str:
    if isinstance(plan, (str, Path)):
        return Path(plan).stem
    return fallback


def _spend_invariant_reallocations(
    contributions: Iterable[Any],
    baseline_data: pd.DataFrame,
    target_data: pd.DataFrame,
) -> list[str]:
    """Channels marked spend-invariant whose target spend differs from baseline."""
    invariant = [c.channel for c in contributions if c.spend_invariant]
    moved = []
    for channel in invariant:
        if channel not in baseline_data.columns or channel not in target_data.columns:
            continue
        if not np.allclose(
            baseline_data[channel].to_numpy(dtype=float),
            target_data[channel].to_numpy(dtype=float),
        ):
            moved.append(channel)
    return moved


# ---------------------------------------------------------------------------
# Engine plumbing
# ---------------------------------------------------------------------------


def _engine_call(
    engine: PosteriorPredictiveEngine,
    posterior_summary: PosteriorSummary,
    data: pd.DataFrame,
    seed: int,
) -> PredictiveSummary:
    return engine.posterior_predictive(posterior_summary, data, seed)


def _draws_array(predictive: PredictiveSummary, label: str) -> NDArray[np.float64]:
    if predictive.draws is None:
        raise ValueError(
            f"Engine returned no per-draw outcomes for the {label} plan. "
            "Risk-adjusted ramp evaluation requires PredictiveSummary.draws "
            "(added in #64); upgrade the engine adapter or pass an engine "
            "that populates draws."
        )
    return np.asarray(predictive.draws, dtype=np.float64)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


def _fractional_kelly(
    delta_full_s: NDArray[np.float64],
    baseline_outcome_s: NDArray[np.float64],
    trust_card: TrustCard | None,
    *,
    override: float | None = None,
) -> float:
    """Fractional Kelly clamped to ``[0, 1]`` then scaled by Trust Card multiplier.

    Without the clamp the raw Kelly value is unbounded above and the
    ``f <= fractional_kelly`` gate becomes decorative. Clamping keeps it
    in the same scale as the candidate ramp fractions.
    """
    safe_baseline = np.maximum(baseline_outcome_s, 1.0)
    r_s = delta_full_s / safe_baseline
    var_r = float(np.var(r_s, ddof=1))
    if var_r <= 0.0:
        return 0.0
    raw = float(np.mean(r_s) / var_r)
    clamped = max(0.0, min(1.0, raw))
    multiplier = override if override is not None else _kelly_multiplier(trust_card)
    return clamped * multiplier


def _kelly_multiplier(trust_card: TrustCard | None) -> float:
    if trust_card is None:
        return _KELLY_MULTIPLIER_PASS
    if any(d.status == TrustStatus.WEAK for d in trust_card.dimensions):
        return _KELLY_MULTIPLIER_WEAK
    if any(d.status == TrustStatus.MODERATE for d in trust_card.dimensions):
        return _KELLY_MULTIPLIER_MODERATE
    return _KELLY_MULTIPLIER_PASS


def _trust_card_ramp_cap(trust_card: TrustCard | None) -> float:
    if trust_card is None:
        return _TRUST_RAMP_CAP_PASS
    if any(d.status == TrustStatus.WEAK for d in trust_card.dimensions):
        return _TRUST_RAMP_CAP_WEAK
    if any(d.status == TrustStatus.MODERATE for d in trust_card.dimensions):
        return _TRUST_RAMP_CAP_MODERATE
    return _TRUST_RAMP_CAP_PASS


def _build_candidate(
    *,
    fraction: float,
    candidate_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    draws: NDArray[np.float64],
    baseline_outcome_s: NDArray[np.float64],
    contributions: list[Any],
    loss_threshold: float,
    cvar_tolerance: float,
    fractional_kelly: float,
    ramp_cap: float,
    risk_tolerance: RiskTolerance,
    trust_card: TrustCard | None,
) -> RampCandidate:
    delta_s = draws.sum(axis=1) - baseline_outcome_s

    expected_increment = float(np.mean(delta_s))
    probability_positive = float(np.mean(delta_s > 0))
    probability_material_loss = float(np.mean(delta_s < -loss_threshold))
    q05_increment = float(np.quantile(delta_s, 0.05))
    tail_mask = delta_s <= q05_increment
    cvar_5 = float(np.mean(delta_s[tail_mask])) if tail_mask.any() else q05_increment

    total_spend = {
        column: float(candidate_data[column].sum())
        for column in candidate_data.columns
        if column != "period"
    }
    largest_move_share = _largest_move_share(baseline_data, candidate_data)
    extrapolation_flags = _extrapolation_flags(candidate_data, contributions)

    failed: list[str] = []
    if probability_positive < risk_tolerance.p_positive_threshold:
        failed.append("p_positive")
    if probability_material_loss > risk_tolerance.p_material_loss_threshold:
        failed.append("p_material_loss")
    if cvar_5 < cvar_tolerance:
        failed.append("cvar_5")
    if fraction > ramp_cap:
        failed.append("trust_card")
    if fraction > fractional_kelly + 1e-9:
        failed.append("fractional_kelly")
    if _extrapolation_severe(extrapolation_flags, risk_tolerance):
        failed.append("extrapolation")
    # Trust Card weakness alone (without an over-cap fraction) doesn't
    # fail the gate — the cap above already enforces that. Spend-
    # invariant channels are blocked up-front, never reach here.

    return RampCandidate(
        fraction=fraction,
        total_spend_by_channel=total_spend,
        expected_increment=expected_increment,
        probability_positive=probability_positive,
        probability_material_loss=probability_material_loss,
        q05_increment=q05_increment,
        cvar_5=cvar_5,
        largest_move_share=largest_move_share,
        fractional_kelly=fractional_kelly,
        extrapolation_flags=extrapolation_flags,
        passes=not failed,
        failed_gates=failed,
    )


def _largest_move_share(
    baseline_data: pd.DataFrame, candidate_data: pd.DataFrame,
) -> float:
    """Largest channel's share of the absolute moved budget.

    Reported, not gated, in v1. ``0.0`` for the do-nothing case.
    """
    moved = []
    for column in baseline_data.columns:
        if column == "period":
            continue
        diff = abs(
            candidate_data[column].astype(float).sum()
            - baseline_data[column].astype(float).sum()
        )
        moved.append(float(diff))
    total = sum(moved)
    if total <= 0.0:
        return 0.0
    return max(moved) / total


def _extrapolation_flags(
    candidate_data: pd.DataFrame,
    contributions: list[Any],
) -> list[ExtrapolationFlag]:
    """Per-cell flag where candidate spend leaves the observed historical range."""
    flags: list[ExtrapolationFlag] = []
    by_channel = {c.channel: c for c in contributions}
    for _, row in candidate_data.iterrows():
        period = str(row["period"])
        for channel, summary in by_channel.items():
            if channel not in candidate_data.columns:
                continue
            planned = float(row[channel])
            if planned > summary.observed_spend_max:
                flags.append(ExtrapolationFlag(
                    period=period, channel=channel, planned_spend=planned,
                    observed_spend_min=summary.observed_spend_min,
                    observed_spend_max=summary.observed_spend_max,
                    direction="above_historical_max",
                ))
            elif planned < summary.observed_spend_min:
                flags.append(ExtrapolationFlag(
                    period=period, channel=channel, planned_spend=planned,
                    observed_spend_min=summary.observed_spend_min,
                    observed_spend_max=summary.observed_spend_max,
                    direction="below_historical_min",
                ))
    return flags


def _extrapolation_severe(
    flags: list[ExtrapolationFlag], risk_tolerance: RiskTolerance,
) -> bool:
    cap = risk_tolerance.extrapolation_severity_cap
    for flag in flags:
        if (
            flag.direction == "above_historical_max"
            and flag.observed_spend_max > 0
            and (flag.planned_spend - flag.observed_spend_max)
            / flag.observed_spend_max
            > cap
        ):
            return True
        if (
            flag.direction == "below_historical_min"
            and flag.observed_spend_min > 0
            and (flag.observed_spend_min - flag.planned_spend)
            / flag.observed_spend_min
            > cap
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def _build_recommendation(
    *,
    candidates: list[RampCandidate],
    baseline_name: str,
    target_name: str,
    trust_card: TrustCard | None,
    fractional_kelly: float,
) -> RampRecommendation:
    passing = [c for c in candidates if c.passes]

    if passing:
        chosen = max(passing, key=lambda c: c.fraction)
        higher_failed = [c for c in candidates if c.fraction > chosen.fraction]
        if not higher_failed:
            # ``chosen`` is at the top of the ladder (f=1.0); nothing
            # higher to consider.
            status: RampStatus = "proceed"
        else:
            # Distinguish "proceed" from "stage" on *why* higher
            # fractions failed:
            #
            # - If at least one higher fraction failed on a value /
            #   downside gate (p_positive, p_material_loss, cvar_5),
            #   the model genuinely doesn't believe in larger moves —
            #   no amount of refresh waiting will change that.
            #   ``proceed`` at the chosen fraction is the model's
            #   verdict.
            #
            # - If higher fractions failed *only* on evidence-quality
            #   gates (extrapolation, trust_card, fractional_kelly),
            #   a refresh after new data could lift the cap. Recommend
            #   the chosen fraction and ``stage`` the move.
            all_higher_failures: set[str] = set()
            for cand in higher_failed:
                all_higher_failures.update(cand.failed_gates)
            if all_higher_failures and all_higher_failures.issubset(_EVIDENCE_GATES):
                status = "stage"
            else:
                status = "proceed"
        explanation = _explain_recommendation(
            chosen, candidates, status, trust_card,
        )
        return RampRecommendation(
            baseline_plan_name=baseline_name,
            target_plan_name=target_name,
            recommended_fraction=chosen.fraction,
            status=status,
            candidates=candidates,
            explanation=explanation,
        )

    # No positive-fraction candidate passes. Inspect why.
    smallest = min(candidates, key=lambda c: c.fraction)
    failure_set = set(smallest.failed_gates)
    if failure_set and failure_set.issubset(_EVIDENCE_GATES):
        status = "test_first"
    else:
        status = "do_not_recommend"
    explanation = _explain_no_recommendation(smallest, status, trust_card)
    return RampRecommendation(
        baseline_plan_name=baseline_name,
        target_plan_name=target_name,
        recommended_fraction=0.0,
        status=status,
        candidates=candidates,
        explanation=explanation,
    )


def _explain_recommendation(
    chosen: RampCandidate,
    candidates: list[RampCandidate],
    status: RampStatus,
    trust_card: TrustCard | None,
) -> str:
    full_move = next((c for c in candidates if c.fraction >= 1.0 - 1e-9), None)
    pieces = [
        f"The largest move that passes every gate is {chosen.fraction:.0%} "
        f"of the way from baseline to target."
    ]
    pieces.append(
        f"Expected gain {chosen.expected_increment:,.0f}; "
        f"{chosen.probability_positive:.0%} probability of beating the "
        f"baseline; {chosen.probability_material_loss:.0%} probability "
        f"of a material loss."
    )
    if status == "stage" and full_move is not None:
        gate_text = ", ".join(full_move.failed_gates)
        pieces.append(
            f"The full 100% move did not pass: {gate_text}. "
            "Stage the change and refresh after new data arrives."
        )
    if trust_card is not None and trust_card.has_weak_dimension:
        weak = ", ".join(trust_card.weak_dimension_names)
        pieces.append(f"Trust Card flags weakness on: {weak}.")
    return " ".join(pieces)


def _explain_no_recommendation(
    smallest: RampCandidate,
    status: RampStatus,
    trust_card: TrustCard | None,
) -> str:
    gate_text = ", ".join(smallest.failed_gates) or "no positive evidence"
    if status == "test_first":
        body = (
            "No positive ramp fraction passed the gates, but the failures "
            "are about evidence quality rather than expected value: "
            f"{gate_text}. A controlled experiment or refresh after more "
            "data would change the answer."
        )
    else:
        body = (
            "No positive ramp fraction passed the gates. The failures are "
            f"about expected value or downside: {gate_text}. The model "
            "does not currently support this reallocation."
        )
    if trust_card is not None and trust_card.has_weak_dimension:
        weak = ", ".join(trust_card.weak_dimension_names)
        body += f" Trust Card flags weakness on: {weak}."
    return body
