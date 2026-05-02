"""Bounded candidate scenario generation (issue #85).

v1 takes a baseline plan plus the run's posterior summary and produces a
small, ranked set of candidate budget plans that each respect the
explicit constraint contract from issue #88. It is *not* an optimizer:
candidates are deterministic donor-to-recipient reallocations between
eligible channels, every candidate is gated through
``validate_candidate_spend`` before forecasting, and the surviving plans
are ranked using the existing ``compare_scenarios`` machinery so deltas
are paired by draw against the baseline.

Eligibility rules (the safety contract #89 will benchmark):

- Locked channels never change.
- Excluded channels are never sources or destinations.
- ``spend_invariant`` channels (FR-3.2) are never sources or destinations.
- Channels with zero baseline spend are not used because percentage
  changes are undefined from zero (matches ``validate_candidate_spend``).
- ``require_historical_support=True`` rejects candidates whose per-period
  spend would step outside the channel's observed historical range.

Per AGENTS.md "Product terminology": output uses cautious decision
language. These are *candidate plans*, not "optimal" or "best" budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from wanamaker.engine.summary import PosteriorSummary
from wanamaker.forecast.constraints import (
    ScenarioGenerationConstraints,
    validate_candidate_spend,
)
from wanamaker.forecast.posterior_predictive import (
    PosteriorPredictiveEngine,
    load_plan,
)
from wanamaker.forecast.scenarios import (
    ScenarioComparisonResult,
    compare_scenarios,
)


@dataclass(frozen=True)
class CandidateRejection:
    """Audit record for a candidate that failed the safety gate."""

    candidate_label: str
    reason: str


@dataclass(frozen=True)
class CandidatePlan:
    """A bounded candidate plan that survived every constraint check.

    Attributes:
        label: Stable identifier suitable for filenames and table rows.
        plan: Wide period-channel DataFrame ready for ``forecast`` /
            ``compare_scenarios`` consumption.
        donor_channel: The channel whose spend was reduced.
        recipient_channel: The channel whose spend was increased.
        moved_amount: Absolute spend transferred from donor to recipient.
        moved_share: ``moved_amount`` as a fraction of baseline total spend.
    """

    label: str
    plan: pd.DataFrame
    donor_channel: str
    recipient_channel: str
    moved_amount: float
    moved_share: float


@dataclass(frozen=True)
class CandidateScenarioSet:
    """The complete result of ``suggest_scenarios``.

    Attributes:
        baseline_label: Display name used for the baseline plan.
        constraints: Resolved constraints that shaped this run.
        candidates: Surviving candidate plans, in generation order.
        rankings: ``ScenarioComparisonResult`` list including the
            baseline plus every candidate, sorted as
            ``compare_scenarios`` returned them.
        rejections: Candidates that failed the safety gate, with the
            triggering reason. Useful to surface in the report so users
            see why some intuitive moves were not produced.
        blocked_channels: Channels excluded from generation, mapped to
            the disqualifying reason: ``"locked"``, ``"excluded"``,
            ``"spend_invariant"``, ``"zero_baseline"``.
    """

    baseline_label: str
    constraints: ScenarioGenerationConstraints
    candidates: list[CandidatePlan] = field(default_factory=list)
    rankings: list[ScenarioComparisonResult] = field(default_factory=list)
    rejections: list[CandidateRejection] = field(default_factory=list)
    blocked_channels: dict[str, str] = field(default_factory=dict)


def suggest_scenarios(
    posterior_summary: PosteriorSummary,
    baseline: str | Path | pd.DataFrame,
    constraints: ScenarioGenerationConstraints,
    seed: int,
    engine: PosteriorPredictiveEngine,
    *,
    baseline_label: str = "baseline",
) -> CandidateScenarioSet:
    """Generate up to ``top_n`` bounded candidate budget plans, ranked.

    Args:
        posterior_summary: Engine-neutral summary from a completed fit.
            Provides per-channel ROI means, observed spend ranges, and
            ``spend_invariant`` flags.
        baseline: Baseline plan as a CSV path or a DataFrame in any of
            the shapes ``load_plan`` accepts.
        constraints: Resolved scenario-generation constraints. Generation
            never violates these; ``validate_candidate_spend`` is the
            fail-closed gate.
        seed: Posterior-predictive seed. Reused across all forecasts so
            ``compare_scenarios`` can pair deltas by draw.
        engine: Implementation of ``PosteriorPredictiveEngine``.
        baseline_label: Display name to use for the baseline plan.

    Returns:
        A ``CandidateScenarioSet`` with every surviving candidate, the
        ranking, the audit trail of rejections, and the blocked-channel
        map. ``candidates`` is empty when no eligible donor/recipient
        pair exists or when every attempt failed the safety gate.

    Examples:
        Generate one bounded candidate from a 50/50 baseline by donating
        from the lowest-ROI channel (``display``, ROI 0.5) to the
        highest-ROI channel (``paid_search``, ROI 4.0). The constraint
        contract caps any single channel's relative change at 10%, so the
        candidate moves exactly ``100 × 0.10 = 10`` from donor to
        recipient.

        >>> import pandas as pd
        >>> from wanamaker.engine.summary import (
        ...     ChannelContributionSummary, PosteriorSummary, PredictiveSummary,
        ... )
        >>> from wanamaker.forecast import (
        ...     ScenarioGenerationConstraints, suggest_scenarios,
        ... )
        >>>
        >>> summary = PosteriorSummary(
        ...     channel_contributions=[
        ...         ChannelContributionSummary(
        ...             channel="paid_search", mean_contribution=0.0,
        ...             hdi_low=0.0, hdi_high=0.0, roi_mean=4.0,
        ...             observed_spend_min=0.0, observed_spend_max=200.0,
        ...         ),
        ...         ChannelContributionSummary(
        ...             channel="display", mean_contribution=0.0,
        ...             hdi_low=0.0, hdi_high=0.0, roi_mean=0.5,
        ...             observed_spend_min=0.0, observed_spend_max=200.0,
        ...         ),
        ...     ]
        ... )
        >>>
        >>> baseline = pd.DataFrame({
        ...     "period": ["2026-01"],
        ...     "paid_search": [100.0],
        ...     "display":     [100.0],
        ... })
        >>> constraints = ScenarioGenerationConstraints(
        ...     budget_mode="hold_total", top_n=1,
        ...     max_channel_change=0.10, max_total_moved_budget=0.20,
        ...     locked_channels=(), excluded_channels=(),
        ...     min_spend=(), max_spend=(),
        ...     require_historical_support=False,
        ... )
        >>>
        >>> class StubEngine:
        ...     def posterior_predictive(self, summary, new_data, seed):
        ...         m = (
        ...             new_data["paid_search"].astype(float) * 4.0
        ...             + new_data["display"].astype(float) * 0.5
        ...         ).tolist()
        ...         return PredictiveSummary(
        ...             periods=new_data["period"].astype(str).tolist(),
        ...             mean=m, hdi_low=m, hdi_high=m, draws=[m, m],
        ...         )
        >>>
        >>> result = suggest_scenarios(
        ...     summary, baseline, constraints, seed=0, engine=StubEngine(),
        ... )
        >>> [c.label for c in result.candidates]
        ['candidate_1_display_to_paid_search']
        >>> only = result.candidates[0]
        >>> only.donor_channel, only.recipient_channel, only.moved_amount
        ('display', 'paid_search', 10.0)
        >>> result.blocked_channels       # nothing locked / invariant / zero
        {}
        >>> # Ranking always includes the baseline plus every surviving candidate.
        >>> [(r.plan_name, r.is_baseline) for r in result.rankings]
        [('candidate_1_display_to_paid_search', False), ('baseline', True)]
    """
    required_channels = [c.channel for c in posterior_summary.channel_contributions]
    baseline_plan = load_plan(baseline, required_channels)
    baseline_totals = {
        channel: float(baseline_plan.data[channel].sum())
        for channel in required_channels
    }

    blocked = _compute_blocked_channels(
        posterior_summary, constraints, baseline_totals
    )
    eligible = [
        contribution.channel
        for contribution in posterior_summary.channel_contributions
        if contribution.channel not in blocked
    ]
    roi_by_channel = {
        contribution.channel: contribution.roi_mean
        for contribution in posterior_summary.channel_contributions
    }
    donors_ranked = sorted(eligible, key=lambda ch: (roi_by_channel[ch], ch))
    recipients_ranked = sorted(eligible, key=lambda ch: (-roi_by_channel[ch], ch))

    candidates: list[CandidatePlan] = []
    rejections: list[CandidateRejection] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = {
        _plan_signature(baseline_totals)
    }

    for donor, recipient, fraction in _candidate_moves(
        donors_ranked, recipients_ranked, constraints
    ):
        if len(candidates) >= constraints.top_n:
            break
        if donor == recipient:
            continue
        amount = baseline_totals[donor] * fraction
        if amount <= 0:
            continue

        candidate_totals = dict(baseline_totals)
        candidate_totals[donor] = baseline_totals[donor] - amount
        candidate_totals[recipient] = baseline_totals[recipient] + amount

        label = _candidate_label(len(candidates) + 1, donor, recipient)

        try:
            validate_candidate_spend(baseline_totals, candidate_totals, constraints)
        except ValueError as exc:
            rejections.append(CandidateRejection(label, str(exc)))
            continue

        signature = _plan_signature(candidate_totals)
        if signature in seen_signatures:
            continue

        candidate_df = _build_plan_dataframe(
            baseline_plan.data,
            baseline_totals,
            candidate_totals,
            required_channels,
        )

        if constraints.require_historical_support and _violates_historical_support(
            candidate_df, posterior_summary, required_channels
        ):
            rejections.append(
                CandidateRejection(
                    label,
                    "candidate exceeds observed historical spend range "
                    "(require_historical_support is true)",
                )
            )
            continue

        seen_signatures.add(signature)
        baseline_total = sum(baseline_totals.values())
        moved_share = amount / baseline_total if baseline_total > 0 else 0.0
        candidates.append(
            CandidatePlan(
                label=label,
                plan=candidate_df,
                donor_channel=donor,
                recipient_channel=recipient,
                moved_amount=amount,
                moved_share=moved_share,
            )
        )

    rankings = _rank_candidates(
        posterior_summary,
        baseline_plan.data,
        baseline_label,
        candidates,
        seed,
        engine,
    )

    return CandidateScenarioSet(
        baseline_label=baseline_label,
        constraints=constraints,
        candidates=candidates,
        rankings=rankings,
        rejections=rejections,
        blocked_channels=blocked,
    )


def _compute_blocked_channels(
    posterior_summary: PosteriorSummary,
    constraints: ScenarioGenerationConstraints,
    baseline_totals: dict[str, float],
) -> dict[str, str]:
    """Reasons each channel is excluded from reallocation, in priority order.

    Locked first, then excluded, then spend_invariant, then zero_baseline.
    The first matching reason wins so the caller can show the most
    actionable explanation (e.g. "you locked this channel" beats "this
    channel is spend-invariant").
    """
    blocked: dict[str, str] = {}
    locked = set(constraints.locked_channels)
    excluded = set(constraints.excluded_channels)
    for contribution in posterior_summary.channel_contributions:
        channel = contribution.channel
        if channel in locked:
            blocked[channel] = "locked"
        elif channel in excluded:
            blocked[channel] = "excluded"
        elif contribution.spend_invariant:
            blocked[channel] = "spend_invariant"
        elif baseline_totals.get(channel, 0.0) <= 0:
            blocked[channel] = "zero_baseline"
    return blocked


def _candidate_moves(
    donors_ranked: list[str],
    recipients_ranked: list[str],
    constraints: ScenarioGenerationConstraints,
) -> list[tuple[str, str, float]]:
    """Enumerate (donor, recipient, move_fraction) tuples deterministically.

    The order fans across pairs first, then move sizes, so the first
    survivors are diverse pairs rather than many sizes of the same pair.
    """
    if not donors_ranked or not recipients_ranked:
        return []

    max_change = constraints.max_channel_change
    if max_change <= 0:
        return []

    fractions = (max_change, max_change * 0.5, max_change * 0.25)
    moves: list[tuple[str, str, float]] = []
    n_donors = len(donors_ranked)
    n_recipients = len(recipients_ranked)

    for fraction in fractions:
        for donor_idx in range(n_donors):
            for recipient_idx in range(n_recipients):
                donor = donors_ranked[donor_idx]
                recipient = recipients_ranked[recipient_idx]
                if donor == recipient:
                    continue
                moves.append((donor, recipient, fraction))
    return moves


def _plan_signature(totals: dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Stable hashable signature of channel totals for deduplication.

    Rounded so floating-point noise across two equivalent reallocations
    doesn't slip past the dedupe check.
    """
    return tuple(sorted((channel, round(value, 6)) for channel, value in totals.items()))


def _build_plan_dataframe(
    baseline_data: pd.DataFrame,
    baseline_totals: dict[str, float],
    candidate_totals: dict[str, float],
    required_channels: list[str],
) -> pd.DataFrame:
    """Scale baseline columns proportionally to hit candidate channel totals.

    Keeps each channel's temporal shape identical to the baseline; only
    the scale changes. Channels with zero baseline get a uniform spread,
    but those are blocked upstream so this branch is defensive.
    """
    candidate = baseline_data.copy()
    n_periods = max(len(candidate), 1)
    for channel in required_channels:
        baseline_total = baseline_totals[channel]
        target_total = candidate_totals[channel]
        if baseline_total > 0:
            scale = target_total / baseline_total
            candidate[channel] = candidate[channel].astype(float) * scale
        else:
            candidate[channel] = float(target_total) / n_periods
    return candidate


def _violates_historical_support(
    plan_df: pd.DataFrame,
    posterior_summary: PosteriorSummary,
    required_channels: list[str],
) -> bool:
    ranges = {
        contribution.channel: (
            float(contribution.observed_spend_min),
            float(contribution.observed_spend_max),
        )
        for contribution in posterior_summary.channel_contributions
    }
    for channel in required_channels:
        low, high = ranges[channel]
        column = plan_df[channel].astype(float)
        if (column > high).any() or (column < low).any():
            return True
    return False


def _candidate_label(index: int, donor: str, recipient: str) -> str:
    return f"candidate_{index}_{donor}_to_{recipient}"


def _rank_candidates(
    posterior_summary: PosteriorSummary,
    baseline_data: pd.DataFrame,
    baseline_label: str,
    candidates: list[CandidatePlan],
    seed: int,
    engine: PosteriorPredictiveEngine,
) -> list[ScenarioComparisonResult]:
    """Rank baseline + candidates via ``compare_scenarios`` with stable labels.

    ``compare_scenarios`` labels DataFrame inputs ``Plan {i+1}``; we
    relabel the results to the baseline label and the candidate labels
    we minted above so the ranking ties back to the candidate CSVs.
    """
    if not candidates:
        return []
    plans: list[str | Path | pd.DataFrame] = [baseline_data.copy()]
    plans.extend(candidate.plan.copy() for candidate in candidates)

    raw_rankings = compare_scenarios(posterior_summary, plans, seed, engine)
    label_for_position = [baseline_label, *(candidate.label for candidate in candidates)]
    name_map = {
        f"Plan {position + 1}": label_for_position[position]
        for position in range(len(label_for_position))
    }
    return [
        replace(result, plan_name=name_map.get(result.plan_name, result.plan_name))
        for result in raw_rankings
    ]
