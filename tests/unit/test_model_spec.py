"""Unit tests for the full ModelSpec schema (issue #7).

Verifies the contract that all engines depend on:
- Required fields are enforced
- Optional fields have correct defaults
- All types are frozen dataclasses
- Supporting types (LiftPrior, HoldoutConfig, SeasonalitySpec, AnchoredPrior)
  are constructible and frozen
"""

from __future__ import annotations

import math
import pytest

from wanamaker.model.priors import ChannelPriors, default_priors_for_category
from wanamaker.model.spec import (
    AnchoredPrior,
    ChannelSpec,
    HoldoutConfig,
    LiftPrior,
    ModelSpec,
    SeasonalitySpec,
)


# ---------------------------------------------------------------------------
# ChannelSpec
# ---------------------------------------------------------------------------


class TestChannelSpec:
    def test_required_fields(self) -> None:
        ch = ChannelSpec(name="paid_search", category="paid_search")
        assert ch.name == "paid_search"
        assert ch.category == "paid_search"

    def test_default_adstock_family_is_geometric(self) -> None:
        ch = ChannelSpec(name="tv", category="linear_tv")
        assert ch.adstock_family == "geometric"

    def test_weibull_family_accepted(self) -> None:
        ch = ChannelSpec(name="tv", category="linear_tv", adstock_family="weibull")
        assert ch.adstock_family == "weibull"

    def test_is_frozen(self) -> None:
        ch = ChannelSpec(name="tv", category="linear_tv")
        with pytest.raises((AttributeError, TypeError)):
            ch.name = "new_name"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LiftPrior
# ---------------------------------------------------------------------------


class TestLiftPrior:
    def test_construction(self) -> None:
        lp = LiftPrior(mean_roi=2.5, sd_roi=0.4)
        assert lp.mean_roi == 2.5
        assert lp.sd_roi == 0.4
        assert lp.confidence == 0.95  # default

    def test_custom_confidence(self) -> None:
        lp = LiftPrior(mean_roi=1.0, sd_roi=0.2, confidence=0.90)
        assert lp.confidence == 0.90

    def test_is_frozen(self) -> None:
        lp = LiftPrior(mean_roi=1.0, sd_roi=0.2)
        with pytest.raises((AttributeError, TypeError)):
            lp.mean_roi = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HoldoutConfig
# ---------------------------------------------------------------------------


class TestHoldoutConfig:
    def test_construction(self) -> None:
        hc = HoldoutConfig(start_date="2023-01-01", end_date="2023-03-31")
        assert hc.start_date == "2023-01-01"
        assert hc.end_date == "2023-03-31"

    def test_is_frozen(self) -> None:
        hc = HoldoutConfig(start_date="2023-01-01", end_date="2023-03-31")
        with pytest.raises((AttributeError, TypeError)):
            hc.start_date = "2024-01-01"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SeasonalitySpec
# ---------------------------------------------------------------------------


class TestSeasonalitySpec:
    def test_defaults(self) -> None:
        ss = SeasonalitySpec()
        assert ss.fourier_order == 2
        assert ss.period_weeks == pytest.approx(365.25 / 7.0)
        assert ss.include_trend is False

    def test_custom_values(self) -> None:
        ss = SeasonalitySpec(fourier_order=4, period_weeks=52.0, include_trend=True)
        assert ss.fourier_order == 4
        assert ss.period_weeks == 52.0
        assert ss.include_trend is True

    def test_no_seasonality_via_fourier_zero(self) -> None:
        ss = SeasonalitySpec(fourier_order=0)
        assert ss.fourier_order == 0

    def test_is_frozen(self) -> None:
        ss = SeasonalitySpec()
        with pytest.raises((AttributeError, TypeError)):
            ss.fourier_order = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AnchoredPrior
# ---------------------------------------------------------------------------


class TestAnchoredPrior:
    def test_construction(self) -> None:
        ap = AnchoredPrior(mean=1.5, sd=0.3, weight=0.3)
        assert ap.mean == 1.5
        assert ap.sd == 0.3
        assert ap.weight == 0.3

    def test_zero_weight_means_no_anchoring(self) -> None:
        ap = AnchoredPrior(mean=0.0, sd=1.0, weight=0.0)
        assert ap.weight == 0.0

    def test_full_weight_means_full_anchoring(self) -> None:
        ap = AnchoredPrior(mean=2.0, sd=0.1, weight=1.0)
        assert ap.weight == 1.0

    def test_is_frozen(self) -> None:
        ap = AnchoredPrior(mean=1.0, sd=0.2, weight=0.3)
        with pytest.raises((AttributeError, TypeError)):
            ap.weight = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelSpec — required fields
# ---------------------------------------------------------------------------


class TestModelSpecRequiredFields:
    def test_minimal_construction(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
        )
        assert spec.target_column == "revenue"
        assert spec.date_column == "week"
        assert len(spec.channels) == 1

    def test_missing_target_column_raises(self) -> None:
        with pytest.raises(TypeError):
            ModelSpec(  # type: ignore[call-arg]
                channels=[ChannelSpec(name="tv", category="linear_tv")],
                date_column="week",
            )

    def test_missing_date_column_raises(self) -> None:
        with pytest.raises(TypeError):
            ModelSpec(  # type: ignore[call-arg]
                channels=[ChannelSpec(name="tv", category="linear_tv")],
                target_column="revenue",
            )

    def test_missing_channels_raises(self) -> None:
        with pytest.raises(TypeError):
            ModelSpec(  # type: ignore[call-arg]
                target_column="revenue",
                date_column="week",
            )


# ---------------------------------------------------------------------------
# ModelSpec — defaults
# ---------------------------------------------------------------------------


class TestModelSpecDefaults:
    def _minimal(self) -> ModelSpec:
        return ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
        )

    def test_control_columns_defaults_empty(self) -> None:
        assert self._minimal().control_columns == []

    def test_frequency_defaults_weekly(self) -> None:
        assert self._minimal().frequency == "weekly"

    def test_channel_priors_defaults_empty(self) -> None:
        assert self._minimal().channel_priors == {}

    def test_lift_test_priors_defaults_empty(self) -> None:
        assert self._minimal().lift_test_priors == {}

    def test_holdout_config_defaults_none(self) -> None:
        assert self._minimal().holdout_config is None

    def test_seasonality_defaults_none(self) -> None:
        assert self._minimal().seasonality is None

    def test_anchor_priors_defaults_none(self) -> None:
        assert self._minimal().anchor_priors is None

    def test_spend_invariant_channels_defaults_empty(self) -> None:
        assert self._minimal().spend_invariant_channels == set()

    def test_runtime_mode_defaults_standard(self) -> None:
        assert self._minimal().runtime_mode == "standard"


# ---------------------------------------------------------------------------
# ModelSpec — optional fields round-trip
# ---------------------------------------------------------------------------


class TestModelSpecOptionalFields:
    def test_channel_priors_override(self) -> None:
        tv_prior = default_priors_for_category("linear_tv")
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
            channel_priors={"tv": tv_prior},
        )
        assert spec.channel_priors["tv"] is tv_prior

    def test_lift_test_prior(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="search", category="paid_search")],
            target_column="revenue",
            date_column="week",
            lift_test_priors={"search": LiftPrior(mean_roi=3.0, sd_roi=0.5)},
        )
        assert spec.lift_test_priors["search"].mean_roi == 3.0

    def test_holdout_config(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
            holdout_config=HoldoutConfig("2023-10-01", "2023-12-31"),
        )
        assert spec.holdout_config is not None
        assert spec.holdout_config.end_date == "2023-12-31"

    def test_seasonality(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
            seasonality=SeasonalitySpec(fourier_order=4, include_trend=True),
        )
        assert spec.seasonality is not None
        assert spec.seasonality.fourier_order == 4
        assert spec.seasonality.include_trend is True

    def test_anchor_priors(self) -> None:
        anchors = {"channel.tv.half_life": AnchoredPrior(mean=6.0, sd=1.0, weight=0.3)}
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
            anchor_priors=anchors,
        )
        assert spec.anchor_priors is not None
        assert spec.anchor_priors["channel.tv.half_life"].weight == 0.3

    def test_spend_invariant_channels(self) -> None:
        spec = ModelSpec(
            channels=[
                ChannelSpec(name="tv", category="linear_tv"),
                ChannelSpec(name="search", category="paid_search"),
            ],
            target_column="revenue",
            date_column="week",
            spend_invariant_channels={"tv"},
        )
        assert "tv" in spec.spend_invariant_channels
        assert "search" not in spec.spend_invariant_channels

    def test_control_columns(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
            control_columns=["holiday", "promo"],
        )
        assert spec.control_columns == ["holiday", "promo"]

    def test_runtime_mode_quick(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
            runtime_mode="quick",
        )
        assert spec.runtime_mode == "quick"


# ---------------------------------------------------------------------------
# ModelSpec — frozen
# ---------------------------------------------------------------------------


class TestModelSpecFrozen:
    def test_is_frozen(self) -> None:
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.target_column = "sales"  # type: ignore[misc]

    def test_channels_field_is_frozen_reference(self) -> None:
        """The channels list is a frozen reference — the spec can't be reassigned."""
        spec = ModelSpec(
            channels=[ChannelSpec(name="tv", category="linear_tv")],
            target_column="revenue",
            date_column="week",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.channels = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Multi-channel ModelSpec
# ---------------------------------------------------------------------------


class TestMultiChannelModelSpec:
    def test_multiple_channels(self) -> None:
        spec = ModelSpec(
            channels=[
                ChannelSpec(name="search", category="paid_search"),
                ChannelSpec(name="social", category="paid_social"),
                ChannelSpec(name="tv", category="linear_tv"),
            ],
            target_column="revenue",
            date_column="week",
        )
        assert len(spec.channels) == 3
        assert spec.channels[2].category == "linear_tv"

    def test_mixed_adstock_families(self) -> None:
        spec = ModelSpec(
            channels=[
                ChannelSpec(name="search", category="paid_search"),
                ChannelSpec(name="tv", category="linear_tv", adstock_family="weibull"),
            ],
            target_column="revenue",
            date_column="week",
        )
        assert spec.channels[0].adstock_family == "geometric"
        assert spec.channels[1].adstock_family == "weibull"

    def test_per_channel_prior_overrides_subset(self) -> None:
        """Only tv is overridden; search falls back to the taxonomy default."""
        tv_prior = default_priors_for_category("linear_tv")
        spec = ModelSpec(
            channels=[
                ChannelSpec(name="search", category="paid_search"),
                ChannelSpec(name="tv", category="linear_tv"),
            ],
            target_column="revenue",
            date_column="week",
            channel_priors={"tv": tv_prior},
        )
        assert "search" not in spec.channel_priors
        assert "tv" in spec.channel_priors
        assert math.exp(spec.channel_priors["tv"].half_life_mu) == pytest.approx(6.0, rel=1e-6)
