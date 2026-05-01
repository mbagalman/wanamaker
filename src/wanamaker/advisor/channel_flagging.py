"""Channel flagging for experimental validation (FR-5.5).

Given a completed fit, ``flag_channels`` returns the channels where a
controlled experiment would most improve decision confidence — either
because the model literally cannot learn saturation from the observed
data (spend invariant) or because the channel is influential enough
that its wide posterior would change reallocation guidance.

The function is intentionally pure: it consumes the engine-neutral
``PosteriorSummary`` plus an optional spend-totals mapping and returns a
sorted list of ``ChannelFlag`` records ready to drop into the executive
summary's recommended-actions section.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from wanamaker.engine.summary import ChannelContributionSummary, PosteriorSummary

# A channel's ROI HDI is "wide" when its width exceeds this fraction of
# |roi_mean|. At 0.5 the HDI spans more than half the magnitude of the
# point estimate — i.e. the model can't tell apart "ROI is X" from "ROI
# is 1.5X". Tuned to match the channel-confidence threshold the report
# template uses, so the advisor's "experiment me" set lines up with the
# template's "weak" tag.
_DEFAULT_UNCERTAINTY_THRESHOLD = 0.5

# A channel needs at least this share of total media contribution before
# we recommend an experiment on it. Below this, the cost of running a
# controlled test almost certainly exceeds the decision value of better
# information about the channel.
_DEFAULT_HIGH_SHARE_THRESHOLD = 0.10


@dataclass(frozen=True)
class ChannelFlag:
    """A single flagged channel and the reason.

    Attributes:
        channel: Channel name as it appears in the fit's ``ModelSpec``.
        posterior_ci: Lower / upper bounds of the channel's contribution
            HDI in target-unit dollars. Reported alongside the rationale
            so the recipient can see *how* uncertain the estimate is.
        spend: Total spend over the training window if the caller
            supplied it; otherwise the channel's mean modelled
            contribution as a proxy. The unit is target-unit dollars in
            either case.
        rationale: Plain-English sentence ready to render into the
            executive-summary recommended-actions list. Includes the
            channel name so the bullet is self-contained.
    """

    channel: str
    posterior_ci: tuple[float, float]
    spend: float
    rationale: str


def flag_channels(
    summary: PosteriorSummary,
    *,
    spend_by_channel: Mapping[str, float] | None = None,
    high_share_threshold: float = _DEFAULT_HIGH_SHARE_THRESHOLD,
    uncertainty_threshold: float = _DEFAULT_UNCERTAINTY_THRESHOLD,
) -> list[ChannelFlag]:
    """Identify channels that would benefit most from experimental validation.

    Two flagging conditions:

    1. **Spend invariant** — saturation cannot be learned from the
       observed data, so prior knowledge alone shapes the curve. The
       only way to improve the estimate is a controlled experiment that
       varies spend, so these are flagged at the highest priority
       regardless of magnitude.
    2. **Wide posterior on an influential channel** — the channel's
       share of media contribution is at or above
       ``high_share_threshold`` *and* the ROI HDI width is at least
       ``uncertainty_threshold * |roi_mean|``. The combination matters:
       a tiny channel with a wide CI is not worth running a test for,
       and a large channel with a tight CI doesn't need one.

    Channels that satisfy neither condition are not flagged.

    Args:
        summary: Engine-neutral posterior summary from a completed fit.
        spend_by_channel: Optional total spend per channel in
            target-unit dollars (typically computed from the training
            data). When supplied, the rationale uses real spend; the
            priority ranking also uses these dollars. When absent, the
            rationale falls back to the channel's mean contribution as
            a spend proxy.
        high_share_threshold: Minimum share of total media contribution
            for a non-invariant channel to be eligible for flagging.
        uncertainty_threshold: ROI HDI width / ``|roi_mean|``. Above
            this, the posterior on a non-invariant channel is "too
            wide" given its influence.

    Returns:
        Flagged channels sorted with spend-invariant entries first and
        the rest by descending spend (or contribution proxy when spend
        is not supplied).
    """
    contributions = list(summary.channel_contributions)
    total_media = sum(c.mean_contribution for c in contributions)
    flags: list[tuple[bool, float, ChannelFlag]] = []

    for contribution in contributions:
        spend_value = float(
            (spend_by_channel or {}).get(
                contribution.channel, contribution.mean_contribution,
            )
        )
        ci = (float(contribution.hdi_low), float(contribution.hdi_high))

        if contribution.spend_invariant:
            flags.append((True, spend_value, _invariant_flag(contribution, spend_value, ci)))
            continue

        share = (
            contribution.mean_contribution / total_media if total_media > 0 else 0.0
        )
        roi_uncertainty = _roi_uncertainty_ratio(contribution)
        if (
            share >= high_share_threshold
            and roi_uncertainty >= uncertainty_threshold
        ):
            flags.append(
                (
                    False,
                    spend_value,
                    _wide_posterior_flag(
                        contribution, spend_value, ci,
                        spend_supplied=spend_by_channel is not None,
                    ),
                )
            )

    # Sort so spend-invariant entries appear first, then by descending
    # spend (or contribution proxy) within their tier. Tie-break on
    # channel name for deterministic output.
    flags.sort(key=lambda entry: (-int(entry[0]), -entry[1], entry[2].channel))
    return [flag for _, _, flag in flags]


# ---------------------------------------------------------------------------
# Rationale construction
# ---------------------------------------------------------------------------


def _invariant_flag(
    contribution: ChannelContributionSummary,
    spend_value: float,
    ci: tuple[float, float],
) -> ChannelFlag:
    rationale = (
        f"Channel `{contribution.channel}` has invariant spend over the "
        "training window, so saturation cannot be estimated from the "
        "data. A controlled experiment that varies spend is the only way "
        "to learn the curve."
    )
    return ChannelFlag(
        channel=contribution.channel,
        posterior_ci=ci,
        spend=spend_value,
        rationale=rationale,
    )


def _wide_posterior_flag(
    contribution: ChannelContributionSummary,
    spend_value: float,
    ci: tuple[float, float],
    *,
    spend_supplied: bool,
) -> ChannelFlag:
    spend_phrase = (
        f"significant spend ({_format_currency(spend_value)})"
        if spend_supplied
        else (
            "a meaningful share of media impact "
            f"({_format_currency(spend_value)} in modelled contribution)"
        )
    )
    rationale = (
        f"Channel `{contribution.channel}` has high posterior uncertainty "
        f"(95% HDI: {_format_currency(ci[0])} to {_format_currency(ci[1])}) "
        f"and {spend_phrase}; a controlled experiment would substantially "
        "improve confidence."
    )
    return ChannelFlag(
        channel=contribution.channel,
        posterior_ci=ci,
        spend=spend_value,
        rationale=rationale,
    )


def _roi_uncertainty_ratio(contribution: ChannelContributionSummary) -> float:
    abs_mean = abs(contribution.roi_mean)
    if abs_mean == 0.0:
        return float("inf")
    width = contribution.roi_hdi_high - contribution.roi_hdi_low
    return width / abs_mean


def _format_currency(value: float) -> str:
    """Compact currency-style formatting for the rationale.

    The advisor doesn't know the user's currency symbol; we emit a bare
    number with thousands separators so the template author can prepend
    the appropriate symbol if needed.
    """
    return f"{value:,.0f}"
