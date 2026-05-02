"""Unit tests for lift-test calibration wiring (issue #16)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from wanamaker.cli import _load_lift_priors_if_any
from wanamaker.config import (
    CalibrationConfig,
    ChannelConfig,
    DataConfig,
    LiftTestCalibrationConfig,
    WanamakerConfig,
)
from wanamaker.data.io import load_lift_test_csv
from wanamaker.engine.pymc import _coefficient_prior
from wanamaker.model.builder import build_model_spec, pool_lift_priors
from wanamaker.model.spec import LiftPrior


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
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
    )

    df = load_lift_test_csv(path)

    assert list(df.columns) == [
        "channel",
        "test_start",
        "test_end",
        "roi_estimate",
        "roi_ci_lower",
        "roi_ci_upper",
    ]
    assert df.loc[0, "channel"] == "search"
    assert pd.api.types.is_datetime64_any_dtype(df["test_start"])
    assert df.loc[0, "roi_estimate"] == pytest.approx(2.0)


def test_load_lift_test_csv_requires_all_columns(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower\n"
        "search,2024-01-01,2024-01-14,2.0,1.2\n",
    )

    with pytest.raises(ValueError, match="does not match any supported schema"):
        load_lift_test_csv(path)


def test_load_lift_test_csv_allows_multiple_rows_per_channel(tmp_path: Path) -> None:
    """Multiple rows for the same channel are now allowed (#78). They're
    pooled in ``model.builder._load_lift_test_priors``; the loader just
    preserves them."""
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n"
        "search,2024-02-01,2024-02-14,2.2,1.4,3.0\n",
    )

    df = load_lift_test_csv(path)
    assert len(df) == 2
    assert (df["channel"] == "search").all()


def test_load_lift_test_csv_warns_on_overlapping_test_windows(tmp_path: Path) -> None:
    """Two tests for the same channel with overlapping date windows
    violate the independence assumption the pooling formula relies on
    (#78). The loader still returns the rows; it just warns."""
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-31,2.0,1.2,2.8\n"
        # Overlaps with the first row (2024-01-01..31).
        "search,2024-01-15,2024-02-14,2.2,1.4,3.0\n",
    )

    with pytest.warns(UserWarning, match="overlapping test windows"):
        df = load_lift_test_csv(path)

    assert len(df) == 2


def test_load_lift_test_csv_does_not_warn_on_disjoint_windows(tmp_path: Path) -> None:
    """Two tests for the same channel with disjoint date windows are the
    intended use case for pooling and should not warn."""
    import warnings as _warnings

    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n"
        "search,2024-02-01,2024-02-14,2.2,1.4,3.0\n",
    )

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        df = load_lift_test_csv(path)

    assert len(df) == 2


def test_load_lift_test_csv_rejects_invalid_interval(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,2.8,1.2\n",
    )

    with pytest.raises(ValueError, match="roi_ci_upper <= roi_ci_lower"):
        load_lift_test_csv(path)


def test_load_lift_test_csv_warns_on_legacy_schema(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
    )

    with pytest.warns(FutureWarning, match="uses legacy columns"):
        df = load_lift_test_csv(path)

    assert "roi_estimate" in df.columns
    assert df.loc[0, "roi_estimate"] == pytest.approx(2.0)


def test_load_lift_test_csv_handles_outcome_schema(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,incremental_outcome,incremental_spend,ci_lower_outcome,ci_upper_outcome\n"
        "search,2024-01-01,2024-01-14,100,50,60,140\n",
    )

    df = load_lift_test_csv(path)

    assert "roi_estimate" in df.columns
    assert df.loc[0, "roi_estimate"] == pytest.approx(2.0)
    assert df.loc[0, "roi_ci_lower"] == pytest.approx(1.2)
    assert df.loc[0, "roi_ci_upper"] == pytest.approx(2.8)


def test_load_lift_test_csv_rejects_mixed_schemas(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper,lift_estimate,ci_lower,ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8,2.0,1.2,2.8\n",
    )

    with pytest.raises(ValueError, match="mixes multiple schemas"):
        load_lift_test_csv(path)


def test_load_lift_test_csv_outcome_schema_rejects_non_numeric(tmp_path: Path) -> None:
    path = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,incremental_outcome,incremental_spend,ci_lower_outcome,ci_upper_outcome\n"
        "search,2024-01-01,2024-01-14,abc,50,60,140\n",
    )

    # pd.to_numeric(errors="raise") drops down to a base ValueError
    with pytest.raises(ValueError, match="Unable to parse string"):
        load_lift_test_csv(path)


def test_build_model_spec_converts_lift_tests_to_priors(tmp_path: Path) -> None:
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.216014,2.783986\n",
    )

    spec = build_model_spec(_config(tmp_path, lift_csv))

    assert set(spec.lift_test_priors) == {"search"}
    prior = spec.lift_test_priors["search"]
    assert prior.mean_roi == pytest.approx(2.0)
    assert prior.sd_roi == pytest.approx(0.4, rel=1e-6)
    assert prior.confidence == pytest.approx(0.95)


def test_build_model_spec_uses_calibration_config_priors(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text("week,revenue,search\n2024-01-01,100,10\n", encoding="utf-8")
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.216014,2.783986\n",
    )

    cfg = WanamakerConfig(
        data=DataConfig(
            csv_path=data_csv,
            date_column="week",
            target_column="revenue",
            spend_columns=["search"],
        ),
        channels=[ChannelConfig(name="search", category="paid_search")],
        calibration=CalibrationConfig(
            lift_tests=LiftTestCalibrationConfig(path=lift_csv)
        ),
    )

    spec = build_model_spec(cfg)
    assert set(spec.lift_test_priors) == {"search"}
    prior = spec.lift_test_priors["search"]
    assert prior.mean_roi == pytest.approx(2.0)


def test_cli_lift_prior_helper_uses_calibration_config(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text("week,revenue,search\n2024-01-01,100,10\n", encoding="utf-8")
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.216014,2.783986\n",
    )

    cfg = WanamakerConfig(
        data=DataConfig(
            csv_path=data_csv,
            date_column="week",
            target_column="revenue",
            spend_columns=["search"],
        ),
        channels=[ChannelConfig(name="search", category="paid_search")],
        calibration=CalibrationConfig(
            lift_tests=LiftTestCalibrationConfig(path=lift_csv)
        ),
    )

    priors = _load_lift_priors_if_any(cfg)

    assert priors is not None
    assert set(priors) == {"search"}
    assert priors["search"].mean_roi == pytest.approx(2.0)


def test_config_rejects_conflicting_lift_test_paths(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text("week,revenue,search\n2024-01-01,100,10\n", encoding="utf-8")
    lift_csv = tmp_path / "lift_tests.csv"
    lift_csv.write_text(
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cannot specify both"):
        WanamakerConfig(
            data=DataConfig(
                csv_path=data_csv,
                date_column="week",
                target_column="revenue",
                spend_columns=["search"],
                lift_test_csv=lift_csv,
            ),
            channels=[ChannelConfig(name="search", category="paid_search")],
            calibration=CalibrationConfig(
                lift_tests=LiftTestCalibrationConfig(path=lift_csv)
            ),
        )


def test_build_model_spec_rejects_lift_test_for_unknown_channel(tmp_path: Path) -> None:
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "social,2024-01-01,2024-01-14,2.0,1.2,2.8\n",
    )

    with pytest.raises(ValueError, match="not configured"):
        build_model_spec(_config(tmp_path, lift_csv))


def test_lift_tests_do_not_modify_adstock_or_saturation_priors(tmp_path: Path) -> None:
    base = build_model_spec(_config(tmp_path))
    lift_csv = _write_lift_csv(
        tmp_path,
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
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
        "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
        "search,2024-01-01,2024-01-14,2.0,1.216014,2.783986\n",
    )
    spec = build_model_spec(_config(tmp_path, lift_csv))
    fake_pm = _FakePM()

    result = _coefficient_prior(
        pm=fake_pm,
        name="channel__search__coefficient",
        lift_prior=spec.lift_test_priors["search"],
        anchor_prior=None,
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
        anchor_prior=None,
        default_sigma=10.0,
    )

    assert result == "half-normal"
    assert fake_pm.calls == [
        ("HalfNormal", "channel__tv__coefficient", {"sigma": 10.0})
    ]


# ---------------------------------------------------------------------------
# Multi-test-per-channel pooling (#78)
# ---------------------------------------------------------------------------


class TestPoolLiftPriors:
    """Worked-example tests for the precision-weighted pooling formula."""

    def test_single_prior_passes_through_with_n_tests_one(self) -> None:
        """A single-element pool should be the input itself, with
        ``n_tests=1`` made explicit even if the input had a default."""
        original = LiftPrior(mean_roi=2.0, sd_roi=0.5)
        pooled = pool_lift_priors([original])
        assert pooled.mean_roi == pytest.approx(2.0)
        assert pooled.sd_roi == pytest.approx(0.5)
        assert pooled.n_tests == 1

    def test_two_equal_precision_priors_average_means_and_shrink_sd(self) -> None:
        """Equal precisions → average mean; pooled sd = σ / √2."""
        priors = [
            LiftPrior(mean_roi=1.0, sd_roi=0.5),
            LiftPrior(mean_roi=3.0, sd_roi=0.5),
        ]
        pooled = pool_lift_priors(priors)
        assert pooled.mean_roi == pytest.approx(2.0)
        # σ_pooled = √(1 / (1/0.25 + 1/0.25)) = √(1/8) = 0.3536...
        assert pooled.sd_roi == pytest.approx(0.5 / math.sqrt(2))
        assert pooled.n_tests == 2

    def test_unequal_precision_pulls_toward_tighter_estimate(self) -> None:
        """A tight test should dominate a wide test in the pooled mean."""
        # σ1 = 0.1, σ2 = 1.0 → precisions 100, 1.
        # μ_pooled = (1.0*100 + 3.0*1) / 101 ≈ 1.0198
        priors = [
            LiftPrior(mean_roi=1.0, sd_roi=0.1),
            LiftPrior(mean_roi=3.0, sd_roi=1.0),
        ]
        pooled = pool_lift_priors(priors)
        assert pooled.mean_roi == pytest.approx(1.0198, abs=1e-3)
        # σ_pooled = √(1/101) ≈ 0.0995
        assert pooled.sd_roi == pytest.approx(math.sqrt(1.0 / 101.0), rel=1e-6)
        # And the pooled sd is below even the tighter input — that is the
        # whole point of pooling additional information.
        assert pooled.sd_roi < 0.1

    def test_pooled_n_tests_sums_inputs(self) -> None:
        """Pooling already-pooled priors keeps ``n_tests`` cumulative."""
        a = pool_lift_priors([
            LiftPrior(mean_roi=2.0, sd_roi=0.5),
            LiftPrior(mean_roi=2.5, sd_roi=0.5),
        ])
        b = LiftPrior(mean_roi=1.5, sd_roi=0.3)
        combined = pool_lift_priors([a, b])
        assert a.n_tests == 2
        assert combined.n_tests == 3

    def test_pool_with_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            pool_lift_priors([])


class TestMultipleLiftTestsPerChannel:
    """End-to-end tests through ``build_model_spec``."""

    def test_pooled_prior_is_tighter_than_single_test_prior(self, tmp_path: Path) -> None:
        # Write two distinct CSVs side by side; helper overwrites a single
        # path so we use sub-directories to keep them apart.
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        double_dir = tmp_path / "double"
        double_dir.mkdir()
        single = _write_lift_csv(
            single_dir,
            "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
            "search,2024-01-01,2024-01-14,2.0,1.6,2.4\n",
        )
        double = _write_lift_csv(
            double_dir,
            "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
            "search,2024-01-01,2024-01-14,2.0,1.6,2.4\n"
            "search,2024-02-01,2024-02-14,2.0,1.6,2.4\n",
        )

        single_spec = build_model_spec(_config(tmp_path, single))
        double_spec = build_model_spec(_config(tmp_path, double))

        single_prior = single_spec.lift_test_priors["search"]
        double_prior = double_spec.lift_test_priors["search"]

        # Same point estimate.
        assert single_prior.mean_roi == pytest.approx(double_prior.mean_roi)
        # Pooled sd is √2 times tighter (two equal-precision tests).
        assert double_prior.sd_roi == pytest.approx(
            single_prior.sd_roi / math.sqrt(2), rel=1e-6,
        )
        # n_tests reflects both rows.
        assert single_prior.n_tests == 1
        assert double_prior.n_tests == 2

    def test_unknown_channel_still_fails(self, tmp_path: Path) -> None:
        """Multi-row support doesn't relax the unknown-channel check."""
        path = _write_lift_csv(
            tmp_path,
            "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
            "ghost,2024-01-01,2024-01-14,2.0,1.6,2.4\n"
            "ghost,2024-02-01,2024-02-14,2.0,1.6,2.4\n",
        )
        with pytest.raises(ValueError, match="not configured"):
            build_model_spec(_config(tmp_path, path))


class TestTrustCardReportsPooledTestCount:
    """The Trust Card calibration message should report the total number
    of underlying lift-test rows, not the number of channels."""

    def test_message_counts_tests_not_channels(self) -> None:
        from wanamaker.engine.summary import ChannelContributionSummary
        from wanamaker.trust_card.compute import lift_test_consistency_dimension

        contributions = [
            ChannelContributionSummary(
                channel="search",
                mean_contribution=100.0,
                hdi_low=80.0,
                hdi_high=120.0,
                roi_mean=2.0,
                roi_hdi_low=1.6,
                roi_hdi_high=2.4,
                observed_spend_min=10.0,
                observed_spend_max=50.0,
            ),
        ]
        # One channel, but the underlying prior pooled three rows.
        priors = {
            "search": LiftPrior(mean_roi=2.0, sd_roi=0.2, n_tests=3),
        }

        result = lift_test_consistency_dimension(contributions, priors)
        # Must say "3 lift tests", not "1 lift test".
        assert "3 lift tests" in result.explanation
