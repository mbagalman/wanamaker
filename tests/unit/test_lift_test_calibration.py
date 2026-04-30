"""Unit tests for lift-test calibration wiring (issue #16)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wanamaker.config import ChannelConfig, DataConfig, WanamakerConfig
from wanamaker.data.io import load_lift_test_csv
from wanamaker.engine.pymc import _coefficient_prior
from wanamaker.model.builder import build_model_spec


def _write_lift_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "lift_tests.csv"
    path.write_text(content, encoding="utf-8")
    return path


def _config(tmp_path: Path, lift_test_csv: Path | None = None) -> WanamakerConfig:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text(
        "week,revenue,search,tv\n2024-01-01,100,10,20\n2024-01-08,110,12,22\n",
        encoding="utf-8",
    )
    return WanamakerConfig(
        data=DataConfig(
            csv_path=data_csv,
            date_column="week",
            target_column="revenue",
            spend_columns=["search", "tv"],
            lift_test_csv=lift_test_csv,
        ),
        channels=[
            ChannelConfig(name="search", category="paid_search"),
            ChannelConfig(name="tv", category="linear_tv"),
        ],
    )


def test_load_lift_test_csv_parses_dates_and_numeric_fields(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
    )

    df = load_lift_test_csv(path)

    assert list(df.columns) == [
        "channel",
        "test_start",
        "test_end",
        "lift_estimate",
        "ci_lower",
        "ci_upper",
    ]
    assert df.loc[0, "channel"] == "search"
    assert pd.api.types.is_datetime64_any_dtype(df["test_start"])
    assert df.loc[0, "lift_estimate"] == pytest.approx(2.0)


def test_load_lift_test_csv_requires_all_columns(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower\n"
        "search,2024-01-01,2024-01-14,2.0,1.2\n",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_lift_test_csv(path)


def test_load_lift_test_csv_rejects_duplicate_channel_rows(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n"
        "search,2024-02-01,2024-02-14,2.2,1.4,3.0\n",
    )

    with pytest.raises(ValueError, match="duplicate channel"):
        load_lift_test_csv(path)


def test_load_lift_test_csv_rejects_invalid_interval(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,2.8,1.2\n",
    )

    with pytest.raises(ValueError, match="ci_upper <= ci_lower"):
        load_lift_test_csv(path)


def test_build_model_spec_converts_lift_tests_to_priors(tmp_path: Path) -> None:
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.216014,2.783986\n",
    )

    spec = build_model_spec(_config(tmp_path, lift_csv))

    assert set(spec.lift_test_priors) == {"search"}
    prior = spec.lift_test_priors["search"]
    assert prior.mean_roi == pytest.approx(2.0)
    assert prior.sd_roi == pytest.approx(0.4, rel=1e-6)
    assert prior.confidence == pytest.approx(0.95)


def test_build_model_spec_rejects_lift_test_for_unknown_channel(tmp_path: Path) -> None:
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "social,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
    )

    with pytest.raises(ValueError, match="not configured"):
        build_model_spec(_config(tmp_path, lift_csv))


def test_lift_tests_do_not_modify_adstock_or_saturation_priors(tmp_path: Path) -> None:
    base = build_model_spec(_config(tmp_path))
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
    )
    calibrated = build_model_spec(_config(tmp_path, lift_csv))

    assert calibrated.channel_priors == base.channel_priors
    assert calibrated.lift_test_priors["search"].mean_roi == pytest.approx(2.0)


class _FakePM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, float]]] = []

    def HalfNormal(self, name: str, **kwargs: float) -> str:  # noqa: N802
        self.calls.append(("HalfNormal", name, kwargs))
        return "half-normal"

    def TruncatedNormal(self, name: str, **kwargs: float) -> str:  # noqa: N802
        self.calls.append(("TruncatedNormal", name, kwargs))
        return "truncated-normal"


def test_pymc_coefficient_prior_uses_lift_prior_when_present(tmp_path: Path) -> None:
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.216014,2.783986\n",
    )
    spec = build_model_spec(_config(tmp_path, lift_csv))
    fake_pm = _FakePM()

    result = _coefficient_prior(
        pm=fake_pm,
        name="channel__search__coefficient",
        lift_prior=spec.lift_test_priors["search"],
        default_sigma=99.0,
    )

    assert result == "truncated-normal"
    assert fake_pm.calls == [
        (
            "TruncatedNormal",
            "channel__search__coefficient",
            {"mu": pytest.approx(2.0), "sigma": pytest.approx(0.4), "lower": 0.0},
        )
    ]


def test_pymc_coefficient_prior_uses_default_without_lift_prior() -> None:
    fake_pm = _FakePM()

    result = _coefficient_prior(
        pm=fake_pm,
        name="channel__tv__coefficient",
        lift_prior=None,
        default_sigma=10.0,
    )

    assert result == "half-normal"
    assert fake_pm.calls == [
        ("HalfNormal", "channel__tv__coefficient", {"sigma": 10.0})
    ]
