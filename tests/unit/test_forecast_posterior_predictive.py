"""Unit tests for posterior-predictive forecast planning (issue #20)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from wanamaker.engine.base import Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.forecast.posterior_predictive import (
    ForecastResult,
    forecast,
)


@dataclass
class FakeEngine:
    calls: list[tuple[PosteriorSummary, pd.DataFrame, int]] = field(default_factory=list)

    def posterior_predictive(
        self,
        posterior_summary: PosteriorSummary,
        new_data: pd.DataFrame,
        seed: int,
    ) -> PredictiveSummary:
        self.calls.append((posterior_summary, new_data.copy(), seed))
        scale = 1.0 + (seed % 7) * 0.01
        mean = [
            float(row.search * 2.0 + row.tv * 0.5) * scale
            for row in new_data.itertuples(index=False)
        ]
        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=mean,
            hdi_low=[value - 1.0 for value in mean],
            hdi_high=[value + 1.0 for value in mean],
        )


def _summary() -> PosteriorSummary:
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=100.0,
                hdi_low=80.0,
                hdi_high=120.0,
                observed_spend_min=10.0,
                observed_spend_max=50.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=200.0,
                hdi_low=150.0,
                hdi_high=250.0,
                observed_spend_min=100.0,
                observed_spend_max=100.0,
                spend_invariant=True,
            ),
        ]
    )


def _wide_plan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["2026-01-05", "2026-01-12", "2026-01-19"],
            "search": [20.0, 75.0, 5.0],
            "tv": [100.0, 100.0, 120.0],
        }
    )


def test_forecast_calls_engine_with_summary_plan_and_seed() -> None:
    summary = _summary()
    engine = FakeEngine()

    result = forecast(summary, _wide_plan(), seed=123, engine=engine)

    assert isinstance(result, PredictiveSummary)
    assert isinstance(result, ForecastResult)
    assert result.periods == ["2026-01-05", "2026-01-12", "2026-01-19"]
    assert len(engine.calls) == 1
    called_summary, called_plan, called_seed = engine.calls[0]
    assert called_summary is summary
    assert called_seed == 123
    assert list(called_plan.columns) == ["period", "search", "tv"]


def test_forecast_flags_periods_above_historical_max() -> None:
    result = forecast(_summary(), _wide_plan(), seed=0, engine=FakeEngine())

    above_max = [
        flag for flag in result.extrapolation_flags
        if flag.direction == "above_historical_max"
    ]
    assert [(flag.period, flag.channel) for flag in above_max] == [
        ("2026-01-12", "search"),
        ("2026-01-19", "tv"),
    ]
    assert result.extrapolated_periods == ["2026-01-12", "2026-01-19"]


def test_forecast_flags_periods_below_historical_min() -> None:
    result = forecast(_summary(), _wide_plan(), seed=0, engine=FakeEngine())

    below_min = [
        flag for flag in result.extrapolation_flags
        if flag.direction == "below_historical_min"
    ]
    assert [(flag.period, flag.channel) for flag in below_min] == [
        ("2026-01-19", "search")
    ]


def test_forecast_records_spend_invariant_channels() -> None:
    result = forecast(_summary(), _wide_plan(), seed=0, engine=FakeEngine())

    assert result.spend_invariant_channels == ["tv"]


def test_forecast_accepts_csv_path(tmp_path: Path) -> None:
    path = tmp_path / "plan.csv"
    _wide_plan().to_csv(path, index=False)

    result = forecast(_summary(), path, seed=0, engine=FakeEngine())

    assert result.periods == ["2026-01-05", "2026-01-12", "2026-01-19"]


def test_forecast_accepts_long_channel_period_plan() -> None:
    long_plan = pd.DataFrame(
        {
            "period": ["2026-01-05", "2026-01-05", "2026-01-12", "2026-01-12"],
            "channel": ["search", "tv", "search", "tv"],
            "spend": [20.0, 100.0, 30.0, 100.0],
        }
    )
    engine = FakeEngine()

    result = forecast(_summary(), long_plan, seed=0, engine=engine)

    assert result.periods == ["2026-01-05", "2026-01-12"]
    assert engine.calls[0][1]["search"].tolist() == pytest.approx([20.0, 30.0])
    assert engine.calls[0][1]["tv"].tolist() == pytest.approx([100.0, 100.0])


def test_forecast_accepts_channel_rows_period_columns_plan() -> None:
    channel_rows = pd.DataFrame(
        {
            "channel": ["search", "tv"],
            "2026-01-05": [20.0, 100.0],
            "2026-01-12": [30.0, 100.0],
        }
    )
    engine = FakeEngine()

    result = forecast(_summary(), channel_rows, seed=0, engine=engine)

    assert result.periods == ["2026-01-05", "2026-01-12"]
    assert engine.calls[0][1]["search"].tolist() == pytest.approx([20.0, 30.0])
    assert engine.calls[0][1]["tv"].tolist() == pytest.approx([100.0, 100.0])


def test_forecast_rejects_bare_posterior() -> None:
    with pytest.raises(TypeError, match="PosteriorSummary"):
        forecast(  # type: ignore[arg-type]
            Posterior(raw=object()),
            _wide_plan(),
            seed=0,
            engine=FakeEngine(),
        )


def test_forecast_rejects_missing_channel() -> None:
    plan = _wide_plan().drop(columns=["tv"])

    with pytest.raises(ValueError, match="missing channel"):
        forecast(_summary(), plan, seed=0, engine=FakeEngine())


def test_forecast_rejects_extra_channel() -> None:
    plan = _wide_plan().assign(display=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="unrecognized columns"):
        forecast(_summary(), plan, seed=0, engine=FakeEngine())


def test_forecast_rejects_negative_spend() -> None:
    plan = _wide_plan()
    plan.loc[0, "search"] = -1.0

    with pytest.raises(ValueError, match="negative spend"):
        forecast(_summary(), plan, seed=0, engine=FakeEngine())


def test_forecast_is_reproducible_for_same_engine_and_seed() -> None:
    plan = _wide_plan()
    first = forecast(_summary(), plan, seed=44, engine=FakeEngine())
    second = forecast(_summary(), plan, seed=44, engine=FakeEngine())

    assert first == second


def test_forecast_validates_engine_output_length() -> None:
    class BadEngine:
        def posterior_predictive(
            self,
            posterior_summary: PosteriorSummary,
            new_data: pd.DataFrame,
            seed: int,
        ) -> PredictiveSummary:
            del posterior_summary, new_data, seed
            return PredictiveSummary(
                periods=["2026-01-05"],
                mean=[1.0],
                hdi_low=[0.0],
                hdi_high=[2.0],
            )

    with pytest.raises(ValueError, match="do not match"):
        forecast(_summary(), _wide_plan(), seed=0, engine=BadEngine())
