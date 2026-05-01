"""Jinja2 rendering for the executive summary and trust card.

Two layers:

- ``build_executive_summary_context`` / ``build_trust_card_context`` turn
  domain objects (``PosteriorSummary``, ``TrustCard``, ``RefreshDiff``)
  into plain dicts suitable for the templates. Per-channel confidence
  tagging, weak-dimension extraction, period range derivation, and
  refresh-diff narrative bullets all live here so the templates stay
  declarative.
- ``render_executive_summary`` / ``render_trust_card`` apply the Jinja2
  templates to a ready-made context dict.

Per AGENTS.md Hard Rule 2: **no LLM calls for output generation**, ever.
If a template feels limiting, expand the template logic. Do not reach
for an LLM.

Templates are loaded from ``wanamaker.reports.templates`` via the
package loader so they ship with installed wheels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
)
from wanamaker.refresh.classify import MovementClass, unexplained_fraction
from wanamaker.refresh.diff import RefreshDiff
from wanamaker.trust_card.card import TrustCard, TrustStatus

# A channel is shown as "high confidence" only when its ROI HDI width is
# at most this fraction of |roi_mean|. Above the moderate threshold it
# becomes "weak" and the template hedges. Tuned to match the trust card's
# WEAKLY_IDENTIFIED rule (CI width / mean > 1.0) so the two surfaces
# agree on "we can't tell from the data".
_HIGH_CONFIDENCE_RATIO = 0.5
_MODERATE_CONFIDENCE_RATIO = 1.0


_env = Environment(
    loader=PackageLoader("wanamaker.reports", "templates"),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render(template_name: str, context: dict[str, Any]) -> str:
    return _env.get_template(template_name).render(**context)


def render_executive_summary(context: dict[str, Any]) -> str:
    """Render the plain-English executive summary (FR-5.3).

    The template adjusts language based on per-channel confidence and
    trust-card weakness (both pre-computed in the context). No LLM
    involvement; the templates are code.
    """
    return _render("executive_summary.md.j2", context)


def render_trust_card(context: dict[str, Any]) -> str:
    """Render the model Trust Card (FR-5.4)."""
    return _render("trust_card.md.j2", context)


def render_ramp_recommendation(context: dict[str, Any]) -> str:
    """Render a risk-adjusted allocation ramp recommendation (FR-5.6)."""
    return _render("ramp_recommendation.md.j2", context)


# ---------------------------------------------------------------------------
# Context shaping
# ---------------------------------------------------------------------------


def build_executive_summary_context(
    summary: PosteriorSummary,
    trust_card: TrustCard,
    *,
    period_labels: Sequence[str] | None = None,
    refresh_diff: RefreshDiff | None = None,
    advisor_recommendations: Iterable[str] | None = None,
    trust_card_link: str | None = None,
) -> dict[str, Any]:
    """Shape the inputs the executive-summary template expects.

    Args:
        summary: Engine-neutral posterior summary from the completed fit.
        trust_card: ``TrustCard`` for the same run.
        period_labels: Ordered period labels (typically ISO date strings)
            covering the training window. Used to set the headline date
            range. ``None`` falls back to the ``in_sample_predictive``
            periods, then to a generic placeholder.
        refresh_diff: Optional diff against a prior run. When present, the
            summary includes a brief refresh narrative.
        advisor_recommendations: Optional plain-English bullets from the
            Experiment Advisor. Defaults to an empty list when absent.
        trust_card_link: Optional anchor / filename for the linked Trust
            Card section in the report. Defaults to ``"trust_card.md"``.

    Returns:
        Plain dict ready to feed into ``render_executive_summary``.
    """
    contributions = list(summary.channel_contributions)
    total_media = sum(c.mean_contribution for c in contributions) or 0.0

    channels = [_channel_view(c, total_media) for c in contributions]
    channels_ranked = sorted(
        channels, key=lambda c: c["contribution_mean"], reverse=True
    )

    period_start, period_end = _period_range(summary, period_labels)

    weak_dimensions = [
        {"name": d.name, "explanation": d.explanation}
        for d in trust_card.dimensions
        if d.status == TrustStatus.WEAK
    ]
    moderate_dimensions = [
        {"name": d.name, "explanation": d.explanation}
        for d in trust_card.dimensions
        if d.status == TrustStatus.MODERATE
    ]

    refresh_narrative = (
        _refresh_narrative(refresh_diff) if refresh_diff is not None else None
    )

    return {
        "period_start": period_start,
        "period_end": period_end,
        "n_periods": len(period_labels)
        if period_labels is not None
        else (
            len(summary.in_sample_predictive.periods)
            if summary.in_sample_predictive is not None
            else 0
        ),
        "total_media_contribution": float(total_media),
        "channels": channels_ranked,
        "trust_card": trust_card,
        "trust_card_link": trust_card_link or "trust_card.md",
        "weak_dimensions": weak_dimensions,
        "moderate_dimensions": moderate_dimensions,
        "has_weak_trust": bool(weak_dimensions),
        "advisor_recommendations": list(advisor_recommendations or []),
        "refresh_narrative": refresh_narrative,
    }


def build_trust_card_context(
    summary: PosteriorSummary,
    trust_card: TrustCard,
) -> dict[str, Any]:
    """Shape the inputs the trust-card template expects.

    The template needs both the dimension list and the per-channel
    saturation breakdown (FR-5.4: "Spend-invariant channels shown with
    'saturation cannot be estimated from observed data'"). The latter is
    derived from ``summary.channel_contributions`` here so the template
    stays declarative.
    """
    saturation_channels = [
        {
            "name": c.channel,
            "spend_invariant": bool(c.spend_invariant),
            "observed_spend_min": float(c.observed_spend_min),
            "observed_spend_max": float(c.observed_spend_max),
        }
        for c in summary.channel_contributions
    ]
    has_invariant = any(c["spend_invariant"] for c in saturation_channels)

    return {
        "dimensions": [
            {
                "name": d.name,
                "status": d.status.value,
                "explanation": d.explanation,
            }
            for d in trust_card.dimensions
        ],
        "saturation_channels": saturation_channels,
        "has_invariant_channels": has_invariant,
    }


def build_ramp_recommendation_context(
    recommendation: Any,
    *,
    run_id: str,
    baseline_path: Any,
    target_path: Any,
    advisor_handoff: str | None = None,
) -> dict[str, Any]:
    """Shape the inputs the ramp-recommendation template expects.

    Args:
        recommendation: ``RampRecommendation`` returned by
            ``wanamaker.forecast.ramp.recommend_ramp``.
        run_id: Run ID used to reconstruct the posterior.
        baseline_path: User-supplied baseline plan path.
        target_path: User-supplied target plan path.
        advisor_handoff: Optional Experiment Advisor sentence for
            evidence-bound recommendations.

    Returns:
        Plain dict ready to feed into ``render_ramp_recommendation``.
    """
    status_labels = {
        "proceed": "Proceed",
        "stage": "Stage",
        "test_first": "Test first",
        "do_not_recommend": "Do not recommend",
    }
    candidates = []
    for candidate in recommendation.candidates:
        candidates.append({
            "fraction": float(candidate.fraction),
            "fraction_label": f"{candidate.fraction:.0%}",
            "expected_increment": float(candidate.expected_increment),
            "probability_positive": float(candidate.probability_positive),
            "probability_material_loss": float(candidate.probability_material_loss),
            "q05_increment": float(candidate.q05_increment),
            "cvar_5": float(candidate.cvar_5),
            "largest_move_share": float(candidate.largest_move_share),
            "fractional_kelly": float(candidate.fractional_kelly),
            "passes": bool(candidate.passes),
            "failed_gates": list(candidate.failed_gates),
            "failed_gates_label": (
                ", ".join(candidate.failed_gates) if candidate.failed_gates else "none"
            ),
            "extrapolation_count": len(candidate.extrapolation_flags),
        })

    return {
        "run_id": run_id,
        "baseline_path": str(baseline_path),
        "target_path": str(target_path),
        "baseline_plan_name": recommendation.baseline_plan_name,
        "target_plan_name": recommendation.target_plan_name,
        "recommended_fraction": float(recommendation.recommended_fraction),
        "recommended_fraction_label": f"{recommendation.recommended_fraction:.0%}",
        "status": recommendation.status,
        "status_label": status_labels.get(recommendation.status, recommendation.status),
        "explanation": recommendation.explanation,
        "blocking_reason": recommendation.blocking_reason,
        "candidates": candidates,
        "advisor_handoff": advisor_handoff,
    }


# ---------------------------------------------------------------------------
# Internal helpers — keep templates declarative
# ---------------------------------------------------------------------------


def _channel_view(
    contribution: ChannelContributionSummary, total_media: float,
) -> dict[str, Any]:
    """Per-channel context entry, with confidence pre-computed."""
    confidence = _channel_confidence(contribution)
    share = (
        contribution.mean_contribution / total_media
        if total_media > 0
        else 0.0
    )
    return {
        "name": contribution.channel,
        "contribution_mean": float(contribution.mean_contribution),
        "contribution_hdi_low": float(contribution.hdi_low),
        "contribution_hdi_high": float(contribution.hdi_high),
        "share_of_effect": float(share),
        "roi_mean": float(contribution.roi_mean),
        "roi_hdi_low": float(contribution.roi_hdi_low),
        "roi_hdi_high": float(contribution.roi_hdi_high),
        "spend_invariant": bool(contribution.spend_invariant),
        "observed_spend_min": float(contribution.observed_spend_min),
        "observed_spend_max": float(contribution.observed_spend_max),
        "confidence": confidence,
    }


def _channel_confidence(contribution: ChannelContributionSummary) -> str:
    """Tag a channel as ``high`` / ``moderate`` / ``weak`` for the template.

    Spend-invariant channels are weak by definition (saturation can't be
    learned). Otherwise we use the ROI HDI width relative to ``|roi_mean|``
    so the rule lines up with the trust card's WEAKLY_IDENTIFIED check.
    """
    if contribution.spend_invariant:
        return "weak"
    abs_mean = abs(contribution.roi_mean)
    if abs_mean == 0.0:
        return "weak"
    width = contribution.roi_hdi_high - contribution.roi_hdi_low
    ratio = width / abs_mean
    if ratio <= _HIGH_CONFIDENCE_RATIO:
        return "high"
    if ratio <= _MODERATE_CONFIDENCE_RATIO:
        return "moderate"
    return "weak"


def _period_range(
    summary: PosteriorSummary, period_labels: Sequence[str] | None,
) -> tuple[str, str]:
    """Pick the period range string for the headline."""
    if period_labels:
        return str(period_labels[0]), str(period_labels[-1])
    predictive = summary.in_sample_predictive
    if predictive and predictive.periods:
        return str(predictive.periods[0]), str(predictive.periods[-1])
    return "unknown", "unknown"


def _refresh_narrative(refresh_diff: RefreshDiff) -> Mapping[str, Any]:
    """Build a brief diff narrative for inclusion in the executive summary."""
    movements = list(refresh_diff.movements)
    fraction = unexplained_fraction(movements)
    classes: dict[str, int] = {}
    for movement in movements:
        if movement.movement_class is None:
            continue
        classes[movement.movement_class.value] = classes.get(
            movement.movement_class.value, 0
        ) + 1
    unexplained_count = classes.get(MovementClass.UNEXPLAINED.value, 0)
    return {
        "previous_run_id": refresh_diff.previous_run_id,
        "n_movements": len(movements),
        "n_unexplained": unexplained_count,
        "unexplained_fraction": fraction,
        "movements_by_class": classes,
    }
