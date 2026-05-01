"""Posterior-predictive forecasting (FR-5.1 mode 2).

This module owns the downstream forecasting contract: future plans are
validated against the engine-neutral ``PosteriorSummary`` so forecast warnings
can be produced without reaching into an engine-native posterior object.
Sampling remains the engine's job through ``posterior_predictive``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from wanamaker.engine.base import Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
    PredictiveSummary,
)

PERIOD_COLUMN_CANDIDATES = ("period", "date", "week", "month")
LONG_PLAN_COLUMNS = {"period", "channel", "spend"}
CHANNEL_COLUMN = "channel"


class PosteriorPredictiveEngine(Protocol):
    """Minimum engine surface needed for posterior predictive forecasting."""

    def posterior_predictive(
        self,
        posterior_summary: PosteriorSummary,
        new_data: pd.DataFrame,
        seed: int,
    ) -> PredictiveSummary:
        """Draw target samples for ``new_data`` using ``posterior_summary`` context.

        The returned ``PredictiveSummary`` must include the per-draw outcome matrix
        in its ``draws`` field with shape ``(n_draws, n_periods)``.
        """
        ...


@dataclass(frozen=True)
class ExtrapolationFlag:
    """One plan cell outside the historical observed spend range."""

    period: str
    channel: str
    planned_spend: float
    observed_spend_min: float
    observed_spend_max: float
    direction: str


@dataclass(frozen=True)
class ForecastResult(PredictiveSummary):
    """Predictive summary plus forecast-specific warning metadata.

    ``ForecastResult`` subclasses ``PredictiveSummary`` so existing consumers
    that only need period means and HDIs can use it directly. The extra fields
    are for CLI/report layers that need to show extrapolation warnings and note
    channels whose saturation was not identifiable from training data.
    """

    extrapolation_flags: list[ExtrapolationFlag] = field(default_factory=list)
    spend_invariant_channels: list[str] = field(default_factory=list)

    @property
    def extrapolated_periods(self) -> list[str]:
        """Return unique period labels that contain any extrapolated spend."""
        return list(dict.fromkeys(flag.period for flag in self.extrapolation_flags))


@dataclass(frozen=True)
class ForecastPlan:
    """A normalised future spend plan: ordered period labels plus a wide
    period-by-channel frame (``period`` column followed by one column per
    required channel)."""

    periods: list[str]
    data: pd.DataFrame


def forecast(
    posterior_summary: PosteriorSummary,
    future_spend: str | Path | pd.DataFrame,
    seed: int,
    engine: PosteriorPredictiveEngine,
) -> ForecastResult:
    """Forecast the target metric for a future spend plan.

    Args:
        posterior_summary: Engine-neutral summary from a completed fit. A bare
            ``Posterior`` is deliberately rejected because it does not carry the
            observed spend ranges needed for extrapolation warnings.
        future_spend: Future plan as either a CSV path or a ``DataFrame``. Wide
            format is ``period,<channel_1>,<channel_2>,...``. Transposed wide
            format is ``channel,<period_1>,<period_2>,...``. Long format is
            ``period,channel,spend``.
        seed: Explicit posterior-predictive seed.
        engine: Engine implementation that can draw posterior predictive
            samples from the summary and normalized future plan.

    Returns:
        A ``ForecastResult`` containing point estimates, HDIs, and warning
        metadata for extrapolated or spend-invariant channels.

    Raises:
        TypeError: If ``posterior_summary`` is a bare ``Posterior`` or if
            ``future_spend`` is not a supported type.
        ValueError: If the summary lacks channel spend ranges, the plan is
            malformed, or the engine returns a summary with incompatible length.
    """
    if isinstance(posterior_summary, Posterior):
        raise TypeError(
            "forecast requires a PosteriorSummary, not a bare Posterior. "
            "Downstream forecast logic needs observed spend ranges from summary.json."
        )
    if not isinstance(posterior_summary, PosteriorSummary):
        raise TypeError("posterior_summary must be a PosteriorSummary")

    channel_ranges = _channel_ranges(posterior_summary)
    channels = list(channel_ranges)
    plan = load_plan(future_spend, channels)
    flags = _extrapolation_flags(plan, channel_ranges)
    spend_invariant_channels = [
        c.channel for c in posterior_summary.channel_contributions if c.spend_invariant
    ]

    predictive = engine.posterior_predictive(posterior_summary, plan.data, seed)
    _validate_predictive(predictive, plan.periods)

    return ForecastResult(
        periods=list(predictive.periods),
        mean=list(predictive.mean),
        hdi_low=list(predictive.hdi_low),
        hdi_high=list(predictive.hdi_high),
        interval_mass=predictive.interval_mass,
        extrapolation_flags=flags,
        spend_invariant_channels=spend_invariant_channels,
    )


def _channel_ranges(
    posterior_summary: PosteriorSummary,
) -> dict[str, ChannelContributionSummary]:
    ranges = {
        contribution.channel: contribution
        for contribution in posterior_summary.channel_contributions
    }
    if not ranges:
        raise ValueError(
            "PosteriorSummary.channel_contributions is empty; cannot validate "
            "future spend ranges for forecast."
        )
    return ranges


def load_plan(
    future_spend: str | Path | pd.DataFrame,
    required_channels: list[str],
) -> ForecastPlan:
    """Load and normalise a future spend plan into a wide period-channel frame.

    Accepts wide (``period,<ch1>,<ch2>,...``), long (``period,channel,spend``),
    or transposed (``channel,<p1>,<p2>,...``) shapes. Returns a ``ForecastPlan``
    whose ``data`` is always wide-format with a single ``period`` column.

    Used by ``forecast()`` and ``compare_scenarios()`` so both consume the
    same normalised plan and produce consistent spend totals.

    Raises:
        TypeError: If ``future_spend`` is not a CSV path or DataFrame.
        ValueError: If the plan is empty, missing required channels, has
            duplicates, or contains non-numeric / negative spend.
    """
    if isinstance(future_spend, (str, Path)):
        raw = pd.read_csv(future_spend)
    elif isinstance(future_spend, pd.DataFrame):
        raw = future_spend.copy()
    else:
        raise TypeError("future_spend must be a CSV path or pandas DataFrame")

    if raw.empty:
        raise ValueError("future spend plan has no rows")

    if _is_long_plan(raw):
        normalized = _normalize_long_plan(raw)
    elif _is_channel_rows_plan(raw):
        normalized = _normalize_channel_rows_plan(raw, required_channels)
    else:
        normalized = raw
    period_column = _period_column(normalized)

    if period_column is None:
        periods = [str(i) for i in range(len(normalized))]
        data = normalized.copy()
        data.insert(0, "period", periods)
        period_column = "period"
    else:
        periods = normalized[period_column].astype(str).tolist()
        data = normalized.copy()
        if period_column != "period":
            data = data.rename(columns={period_column: "period"})
            period_column = "period"

    _validate_channels(data, required_channels, period_column)
    for channel in required_channels:
        data[channel] = pd.to_numeric(data[channel], errors="raise")
        if data[channel].isna().any():
            raise ValueError(f"future spend plan contains missing values for {channel!r}")
        if (data[channel] < 0).any():
            raise ValueError(f"future spend plan contains negative spend for {channel!r}")

    return ForecastPlan(periods=periods, data=data[["period", *required_channels]])


def _is_long_plan(data: pd.DataFrame) -> bool:
    return LONG_PLAN_COLUMNS.issubset({str(column).lower() for column in data.columns})


def _is_channel_rows_plan(data: pd.DataFrame) -> bool:
    columns = {str(column).lower() for column in data.columns}
    return CHANNEL_COLUMN in columns and "spend" not in columns


def _normalize_long_plan(data: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = {str(column).lower(): column for column in data.columns}
    period_col = lower_to_original["period"]
    channel_col = lower_to_original["channel"]
    spend_col = lower_to_original["spend"]
    duplicates = data.duplicated(subset=[period_col, channel_col])
    if duplicates.any():
        duplicated_rows = data.loc[duplicates, [period_col, channel_col]].to_dict("records")
        raise ValueError(f"future spend plan has duplicate period/channel rows: {duplicated_rows}")
    pivoted = data.pivot(index=period_col, columns=channel_col, values=spend_col)
    pivoted = pivoted.reset_index()
    pivoted.columns = [str(column) for column in pivoted.columns]
    return pivoted


def _normalize_channel_rows_plan(
    data: pd.DataFrame,
    required_channels: list[str],
) -> pd.DataFrame:
    lower_to_original = {str(column).lower(): column for column in data.columns}
    channel_col = lower_to_original[CHANNEL_COLUMN]
    duplicate_channels = data[channel_col].duplicated()
    if duplicate_channels.any():
        duplicates = data.loc[duplicate_channels, channel_col].astype(str).tolist()
        raise ValueError(f"future spend plan has duplicate channel rows: {duplicates}")

    indexed = data.set_index(channel_col)
    missing = sorted(set(required_channels) - set(map(str, indexed.index)))
    if missing:
        raise ValueError(f"future spend plan is missing channel rows: {missing}")
    extras = sorted(set(map(str, indexed.index)) - set(required_channels))
    if extras:
        raise ValueError(f"future spend plan has unrecognized channel rows: {extras}")

    transposed = indexed.loc[required_channels].transpose().reset_index()
    transposed = transposed.rename(columns={"index": "period"})
    transposed.columns = [str(column) for column in transposed.columns]
    return transposed


def _period_column(data: pd.DataFrame) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in data.columns}
    for candidate in PERIOD_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _validate_channels(
    data: pd.DataFrame,
    required_channels: list[str],
    period_column: str,
) -> None:
    columns = set(map(str, data.columns))
    required = set(required_channels)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"future spend plan is missing channel columns: {missing}")
    extras = sorted(columns - required - {period_column})
    if extras:
        raise ValueError(f"future spend plan has unrecognized columns: {extras}")


def _extrapolation_flags(
    plan: ForecastPlan,
    channel_ranges: dict[str, ChannelContributionSummary],
) -> list[ExtrapolationFlag]:
    flags: list[ExtrapolationFlag] = []
    for _, row in plan.data.iterrows():
        period = str(row["period"])
        for channel, summary in channel_ranges.items():
            planned = float(row[channel])
            if planned > summary.observed_spend_max:
                flags.append(
                    ExtrapolationFlag(
                        period=period,
                        channel=channel,
                        planned_spend=planned,
                        observed_spend_min=summary.observed_spend_min,
                        observed_spend_max=summary.observed_spend_max,
                        direction="above_historical_max",
                    )
                )
            elif planned < summary.observed_spend_min:
                flags.append(
                    ExtrapolationFlag(
                        period=period,
                        channel=channel,
                        planned_spend=planned,
                        observed_spend_min=summary.observed_spend_min,
                        observed_spend_max=summary.observed_spend_max,
                        direction="below_historical_min",
                    )
                )
    return flags


def _validate_predictive(predictive: Any, periods: list[str]) -> None:
    if not isinstance(predictive, PredictiveSummary):
        raise TypeError("engine.posterior_predictive must return a PredictiveSummary")
    expected = len(periods)
    lengths = {
        "periods": len(predictive.periods),
        "mean": len(predictive.mean),
        "hdi_low": len(predictive.hdi_low),
        "hdi_high": len(predictive.hdi_high),
    }
    bad = {name: length for name, length in lengths.items() if length != expected}
    if bad:
        raise ValueError(
            "engine.posterior_predictive returned lengths that do not match "
            f"the future plan ({expected} rows): {bad}"
        )
