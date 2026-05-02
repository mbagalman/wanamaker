"""Calibrated-vs-uncalibrated comparison report (issue #80).

Given two completed runs that differ only by whether lift-test
calibration was supplied, this module produces a deterministic
side-by-side comparison: per-channel ROI / contribution / share-of-effect
before and after calibration, plus a plain-English classification of
how each channel moved (agrees, experiment-dominant, directional shift,
history-dominant) and a one-sentence stakeholder summary.

Per AGENTS.md Hard Rule 2: **no LLM calls for output generation**, ever.
Every interpretation comes from a deterministic decision tree.
Per AGENTS.md "Product terminology", language is decision-support, not
optimizer-grade promises (the guardrail test enforces this on
``src/wanamaker/reports/`` files).

Public surface:

- ``compare_calibration`` — turn two ``PosteriorSummary`` objects (plus
  the calibrated run's lift-test channel set) into a
  ``CalibrationComparison`` value object.
- ``build_calibration_comparison_context`` — shape the comparison for
  the Markdown template.
- ``render_calibration_comparison`` — apply the template.
- Validation errors: ``CalibrationComparisonError`` and named subclasses
  surface mismatch reasons (different data hash, channel-set drift,
  both-calibrated, neither-calibrated, etc.) with actionable messages.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from wanamaker.engine.summary import ChannelContributionSummary, PosteriorSummary

# A calibrated HDI narrower than this fraction of the uncalibrated HDI
# means the experiment dominated the data-driven posterior — its
# precision tightened the estimate beyond what history alone supports.
_EXPERIMENT_DOMINANT_HDI_RATIO = 0.7


# ---------------------------------------------------------------------------
# Error types — surface every mismatch with a recognisable subclass so
# the CLI can map errors back to user actions.
# ---------------------------------------------------------------------------


class CalibrationComparisonError(ValueError):
    """Base class for all calibration-comparison validation failures."""


class DataHashMismatchError(CalibrationComparisonError):
    """The two runs were fit on different training CSVs."""


class ChannelSetMismatchError(CalibrationComparisonError):
    """The two runs have different channel sets in their summaries."""


class CalibrationModeError(CalibrationComparisonError):
    """Both runs are calibrated, or neither is — exactly one is required."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelComparison:
    """One channel's before/after numbers and classification."""

    channel: str
    is_calibrated: bool
    """True when this channel had a lift-test prior in the calibrated run."""
    roi_mean_uncal: float
    roi_hdi_low_uncal: float
    roi_hdi_high_uncal: float
    roi_mean_cal: float
    roi_hdi_low_cal: float
    roi_hdi_high_cal: float
    contribution_uncal: float
    contribution_cal: float
    share_uncal: float
    share_cal: float
    classification: str
    """One of: ``agrees``, ``experiment-dominant``, ``directional-shift``,
    ``history-dominant``, ``secondary-shift``. See ``compare_calibration``
    for the decision tree."""

    @property
    def is_material_change(self) -> bool:
        """True when the calibrated mean falls outside the uncalibrated HDI
        or vice versa — i.e. the experiment moved the estimate beyond
        what the uncalibrated posterior could explain."""
        return (
            self.roi_mean_cal < self.roi_hdi_low_uncal
            or self.roi_mean_cal > self.roi_hdi_high_uncal
            or self.roi_mean_uncal < self.roi_hdi_low_cal
            or self.roi_mean_uncal > self.roi_hdi_high_cal
        )

    @property
    def roi_delta(self) -> float:
        return self.roi_mean_cal - self.roi_mean_uncal

    @property
    def roi_relative_change(self) -> float:
        if self.roi_mean_uncal == 0:
            return 0.0
        return (self.roi_mean_cal - self.roi_mean_uncal) / abs(self.roi_mean_uncal)


@dataclass(frozen=True)
class CalibrationComparison:
    """Full comparison between an uncalibrated run and a calibrated run."""

    uncalibrated_run_id: str
    calibrated_run_id: str
    calibrated_channels: list[str]
    """Channels that had a lift-test prior in the calibrated run."""
    channels: list[ChannelComparison] = field(default_factory=list)
    total_media_uncal: float = 0.0
    total_media_cal: float = 0.0
    summary_sentence: str = ""


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_calibration(
    uncalibrated_summary: PosteriorSummary,
    calibrated_summary: PosteriorSummary,
    *,
    uncalibrated_run_id: str,
    calibrated_run_id: str,
    calibrated_channels: Iterable[str],
) -> CalibrationComparison:
    """Build a per-channel comparison between two posterior summaries.

    Args:
        uncalibrated_summary: ``PosteriorSummary`` from the run fitted
            without a lift-test prior.
        calibrated_summary: ``PosteriorSummary`` from the run fitted with
            a lift-test prior on the same data.
        uncalibrated_run_id: Source run id; surfaces in the Markdown.
        calibrated_run_id: Source run id; surfaces in the Markdown.
        calibrated_channels: Channels in the calibrated run's
            lift-test prior dictionary. Used to flag which channels were
            directly calibrated vs. only secondarily shifted.

    Raises:
        ChannelSetMismatchError: When the two summaries describe different
            channel sets. Same data, same channels — that's the contract.
    """
    uncal_by_channel = {c.channel: c for c in uncalibrated_summary.channel_contributions}
    cal_by_channel = {c.channel: c for c in calibrated_summary.channel_contributions}
    uncal_set = set(uncal_by_channel)
    cal_set = set(cal_by_channel)
    if uncal_set != cal_set:
        only_in_uncal = sorted(uncal_set - cal_set)
        only_in_cal = sorted(cal_set - uncal_set)
        raise ChannelSetMismatchError(
            "uncalibrated and calibrated runs describe different channel "
            f"sets. Only in uncalibrated: {only_in_uncal}; only in "
            f"calibrated: {only_in_cal}. Both runs must be fit on the same "
            "channel set."
        )

    calibrated_set = set(calibrated_channels)
    total_media_uncal = sum(c.mean_contribution for c in uncalibrated_summary.channel_contributions)
    total_media_cal = sum(c.mean_contribution for c in calibrated_summary.channel_contributions)

    # Order channels by uncalibrated contribution so the largest media
    # contributors lead the table. Stable for downstream rendering.
    ordered_channels = sorted(
        uncal_set,
        key=lambda c: uncal_by_channel[c].mean_contribution,
        reverse=True,
    )

    channel_comparisons: list[ChannelComparison] = []
    for channel_name in ordered_channels:
        uncal = uncal_by_channel[channel_name]
        cal = cal_by_channel[channel_name]
        is_cal = channel_name in calibrated_set
        classification = _classify_channel(uncal, cal, is_calibrated=is_cal)
        channel_comparisons.append(
            ChannelComparison(
                channel=channel_name,
                is_calibrated=is_cal,
                roi_mean_uncal=float(uncal.roi_mean),
                roi_hdi_low_uncal=float(uncal.roi_hdi_low),
                roi_hdi_high_uncal=float(uncal.roi_hdi_high),
                roi_mean_cal=float(cal.roi_mean),
                roi_hdi_low_cal=float(cal.roi_hdi_low),
                roi_hdi_high_cal=float(cal.roi_hdi_high),
                contribution_uncal=float(uncal.mean_contribution),
                contribution_cal=float(cal.mean_contribution),
                share_uncal=(
                    float(uncal.mean_contribution / total_media_uncal)
                    if total_media_uncal > 0 else 0.0
                ),
                share_cal=(
                    float(cal.mean_contribution / total_media_cal)
                    if total_media_cal > 0 else 0.0
                ),
                classification=classification,
            )
        )

    return CalibrationComparison(
        uncalibrated_run_id=uncalibrated_run_id,
        calibrated_run_id=calibrated_run_id,
        calibrated_channels=sorted(calibrated_set),
        channels=channel_comparisons,
        total_media_uncal=total_media_uncal,
        total_media_cal=total_media_cal,
        summary_sentence=_summary_sentence(channel_comparisons),
    )


def _classify_channel(
    uncal: ChannelContributionSummary,
    cal: ChannelContributionSummary,
    *,
    is_calibrated: bool,
) -> str:
    """Five-bucket per-channel classification.

    - ``no-material-change`` — calibrated mean inside the uncalibrated
      HDI and vice versa; the experiment didn't reposition this channel
      meaningfully.
    - ``experiment-dominant`` — only on directly-calibrated channels:
      the calibrated HDI is much tighter than the uncalibrated HDI,
      meaning the lift-test prior dominated the data-driven posterior.
    - ``directional-shift`` — directly-calibrated channel where the
      mean moved outside the uncalibrated HDI but the calibrated HDI
      didn't tighten dramatically (the data still has a say).
    - ``history-dominant`` — directly-calibrated channel whose
      posterior barely moved despite having a lift-test prior. Means
      the experiment was noisy enough that history outweighed it.
    - ``secondary-shift`` — non-calibrated channel that moved
      materially because calibration of *other* channels reshaped the
      total-contribution share.
    """
    uncal_mean_in_cal_hdi = cal.roi_hdi_low <= uncal.roi_mean <= cal.roi_hdi_high
    cal_mean_in_uncal_hdi = uncal.roi_hdi_low <= cal.roi_mean <= uncal.roi_hdi_high
    is_material = not (uncal_mean_in_cal_hdi and cal_mean_in_uncal_hdi)

    uncal_width = max(uncal.roi_hdi_high - uncal.roi_hdi_low, 1e-12)
    cal_width = max(cal.roi_hdi_high - cal.roi_hdi_low, 1e-12)
    width_ratio = cal_width / uncal_width

    if not is_calibrated:
        return "secondary-shift" if is_material else "no-material-change"

    # Directly calibrated channels.
    if width_ratio <= _EXPERIMENT_DOMINANT_HDI_RATIO:
        return "experiment-dominant"
    if is_material:
        return "directional-shift"
    return "history-dominant"


def _summary_sentence(channels: list[ChannelComparison]) -> str:
    """One-sentence stakeholder takeaway from the per-channel classifications.

    Picks the most consequential signal: experiment-dominant beats
    directional-shift beats secondary-shift beats history-dominant
    beats no-material-change. Names the affected channels by name when
    there are 1–3, otherwise gives a count.
    """
    by_class: dict[str, list[ChannelComparison]] = {}
    for c in channels:
        by_class.setdefault(c.classification, []).append(c)

    def joined(items: list[ChannelComparison]) -> str:
        names = [f"`{c.channel}`" for c in items]
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    if "experiment-dominant" in by_class:
        items = by_class["experiment-dominant"]
        if len(items) > 3:
            return (
                f"Experiment evidence dominated for {len(items)} channels; "
                "treat the calibrated estimates as experiment-led."
            )
        return (
            f"Experiment evidence dominated the posterior for {joined(items)}; "
            "treat the calibrated estimates as experiment-led."
        )
    if "directional-shift" in by_class:
        items = by_class["directional-shift"]
        if len(items) > 3:
            return (
                f"Adding the lift test moved {len(items)} channel ROIs "
                "beyond the uncalibrated credible interval; the experiment "
                "and the data partly disagree."
            )
        return (
            f"Adding the lift test pulled {joined(items)} beyond the "
            "uncalibrated credible interval; treat the direction as the "
            "real signal but the magnitude as still uncertain."
        )
    if "secondary-shift" in by_class:
        items = by_class["secondary-shift"]
        return (
            f"Calibrating other channels reshaped {joined(items)} "
            "indirectly; review the per-channel table before acting."
        )
    if "history-dominant" in by_class:
        return (
            "The lift test was supplied but did not materially move the "
            "posterior — historical data outweighed the experiment's "
            "precision."
        )
    return (
        "Adding the lift test produced no material change; the experiment "
        "is consistent with what the data already indicated."
    )


# ---------------------------------------------------------------------------
# Template plumbing — context shaper + render function
# ---------------------------------------------------------------------------


def build_calibration_comparison_context(
    comparison: CalibrationComparison,
) -> dict[str, Any]:
    """Shape the dataclass into a flat dict for the Jinja template.

    Numeric values are pre-formatted into display strings here so the
    template stays declarative.
    """
    rows = []
    for c in comparison.channels:
        rows.append(
            {
                "channel": c.channel,
                "is_calibrated": c.is_calibrated,
                "classification": c.classification,
                "classification_label": _CLASSIFICATION_LABELS.get(
                    c.classification, c.classification,
                ),
                "roi_mean_uncal": c.roi_mean_uncal,
                "roi_hdi_uncal": (c.roi_hdi_low_uncal, c.roi_hdi_high_uncal),
                "roi_mean_cal": c.roi_mean_cal,
                "roi_hdi_cal": (c.roi_hdi_low_cal, c.roi_hdi_high_cal),
                "roi_delta": c.roi_delta,
                "roi_relative_change_pct": c.roi_relative_change * 100.0,
                "contribution_uncal": c.contribution_uncal,
                "contribution_cal": c.contribution_cal,
                "share_uncal": c.share_uncal,
                "share_cal": c.share_cal,
                "is_material_change": c.is_material_change,
            }
        )

    return {
        "uncalibrated_run_id": comparison.uncalibrated_run_id,
        "calibrated_run_id": comparison.calibrated_run_id,
        "calibrated_channels": comparison.calibrated_channels,
        "n_calibrated_channels": len(comparison.calibrated_channels),
        "channels": rows,
        "n_channels": len(rows),
        "n_material_changes": sum(1 for r in rows if r["is_material_change"]),
        "total_media_uncal": comparison.total_media_uncal,
        "total_media_cal": comparison.total_media_cal,
        "total_media_delta": comparison.total_media_cal - comparison.total_media_uncal,
        "summary_sentence": comparison.summary_sentence,
    }


_CLASSIFICATION_LABELS: dict[str, str] = {
    "no-material-change": "no material change",
    "experiment-dominant": "experiment-dominant",
    "directional-shift": "directional shift",
    "history-dominant": "history-dominant",
    "secondary-shift": "secondary shift",
}


def render_calibration_comparison(context: dict[str, Any]) -> str:
    """Render the comparison Markdown report from a ready-made context."""
    from wanamaker.reports.render import _env  # local import to avoid a cycle

    template = _env.get_template("calibration_comparison.md.j2")
    return template.render(**context)
