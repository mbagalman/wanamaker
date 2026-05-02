"""Unit tests for Trust Card computation (issue #19)."""

from __future__ import annotations

import pytest

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.model.spec import LiftPrior
from wanamaker.refresh.classify import MovementClass
from wanamaker.refresh.diff import ParameterMovement, RefreshDiff
from wanamaker.trust_card import (
    PriorSensitivityResult,
    TrustStatus,
    build_trust_card,
    convergence_dimension,
    holdout_accuracy_dimension,
    lift_test_consistency_dimension,
    prior_sensitivity_dimension,
    refresh_stability_dimension,
    saturation_identifiability_dimension,
)
from wanamaker.trust_card.card import TrustCard, TrustDimension


def _summary(
    convergence: ConvergenceSummary | None = None,
    predictive: PredictiveSummary | None = None,
    contributions: list[ChannelContributionSummary] | None = None,
) -> PosteriorSummary:
    return PosteriorSummary(
        convergence=convergence,
        in_sample_predictive=predictive,
        channel_contributions=contributions or [],
    )


def _convergence(
    r_hat: float | None = 1.005,
    ess: float | None = 500.0,
    divergences: int = 0,
) -> ConvergenceSummary:
    return ConvergenceSummary(
        max_r_hat=r_hat,
        min_ess_bulk=ess,
        n_divergences=divergences,
        n_chains=4,
        n_draws=1000,
    )


def _predictive(values: list[float]) -> PredictiveSummary:
    return PredictiveSummary(
        periods=[str(i) for i in range(len(values))],
        mean=values,
        hdi_low=[value - 1.0 for value in values],
        hdi_high=[value + 1.0 for value in values],
    )


def _contribution(
    channel: str,
    roi_mean: float = 2.0,
    roi_low: float = 1.5,
    roi_high: float = 2.5,
    spend_invariant: bool = False,
) -> ChannelContributionSummary:
    return ChannelContributionSummary(
        channel=channel,
        mean_contribution=100.0,
        hdi_low=80.0,
        hdi_high=120.0,
        roi_mean=roi_mean,
        roi_hdi_low=roi_low,
        roi_hdi_high=roi_high,
        observed_spend_min=10.0,
        observed_spend_max=100.0,
        spend_invariant=spend_invariant,
    )


def _movement(cls: MovementClass) -> ParameterMovement:
    return ParameterMovement(
        name="channel.search.roi",
        previous_mean=1.0,
        current_mean=1.1,
        previous_ci=(0.5, 1.5),
        current_ci=(0.6, 1.6),
        movement_class=cls,
    )


class TestTrustCardHelpers:
    def test_has_weak_dimension(self) -> None:
        card = TrustCard(
            [
                TrustDimension("convergence", TrustStatus.PASS, "ok"),
                TrustDimension("holdout_accuracy", TrustStatus.WEAK, "bad"),
            ]
        )

        assert card.has_weak_dimension is True
        assert card.weak_dimension_names == ["holdout_accuracy"]
        assert card.dimension("convergence") is not None
        assert card.dimension("missing") is None


class TestConvergenceDimension:
    def test_pass(self) -> None:
        result = convergence_dimension(_summary(_convergence()))

        assert result.status == TrustStatus.PASS
        assert result.name == "convergence"

    def test_moderate_for_degraded_but_usable_diagnostics(self) -> None:
        result = convergence_dimension(_summary(_convergence(r_hat=1.03, ess=250.0)))

        assert result.status == TrustStatus.MODERATE

    def test_weak_for_bad_rhat(self) -> None:
        result = convergence_dimension(_summary(_convergence(r_hat=1.08, ess=500.0)))

        assert result.status == TrustStatus.WEAK

    def test_moderate_when_rhat_unavailable(self) -> None:
        result = convergence_dimension(_summary(_convergence(r_hat=None)))

        assert result.status == TrustStatus.MODERATE

    def test_weak_when_missing(self) -> None:
        result = convergence_dimension(_summary(convergence=None))

        assert result.status == TrustStatus.WEAK


class TestHoldoutAccuracyDimension:
    def test_pass_for_low_wape(self) -> None:
        result = holdout_accuracy_dimension(
            _summary(predictive=_predictive([100.0, 110.0])),
            actuals=[100.0, 100.0],
        )

        assert result.status == TrustStatus.PASS
        assert "WAPE" in result.explanation

    def test_moderate_when_not_evaluated(self) -> None:
        result = holdout_accuracy_dimension(_summary(), actuals=None)

        assert result.status == TrustStatus.MODERATE

    def test_weak_for_high_wape(self) -> None:
        result = holdout_accuracy_dimension(
            _summary(predictive=_predictive([50.0, 50.0])),
            actuals=[100.0, 100.0],
        )

        assert result.status == TrustStatus.WEAK

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            holdout_accuracy_dimension(
                _summary(predictive=_predictive([100.0])),
                actuals=[100.0, 110.0],
            )


class TestRefreshStabilityDimension:
    def test_pass_for_low_unexplained_fraction(self) -> None:
        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[_movement(MovementClass.WITHIN_PRIOR_CI)] * 10,
        )

        result = refresh_stability_dimension(diff)

        assert result.status == TrustStatus.PASS

    def test_moderate_for_some_unexplained_movements(self) -> None:
        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[
                _movement(MovementClass.UNEXPLAINED),
                *_movement_list(MovementClass.WITHIN_PRIOR_CI, 4),
            ],
        )

        result = refresh_stability_dimension(diff)

        assert result.status == TrustStatus.MODERATE

    def test_weak_for_many_unexplained_movements(self) -> None:
        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[
                _movement(MovementClass.UNEXPLAINED),
                _movement(MovementClass.UNEXPLAINED),
                _movement(MovementClass.WITHIN_PRIOR_CI),
            ],
        )

        result = refresh_stability_dimension(diff)

        assert result.status == TrustStatus.WEAK


class TestPriorSensitivityDimension:
    def test_moderate_when_missing(self) -> None:
        assert prior_sensitivity_dimension(None).status == TrustStatus.MODERATE

    def test_pass_for_small_shift(self) -> None:
        result = prior_sensitivity_dimension(PriorSensitivityResult(max_relative_shift=0.05))

        assert result.status == TrustStatus.PASS

    def test_moderate_for_mid_shift(self) -> None:
        assert prior_sensitivity_dimension(0.15).status == TrustStatus.MODERATE

    def test_weak_for_large_shift(self) -> None:
        assert prior_sensitivity_dimension(0.40).status == TrustStatus.WEAK


class TestSaturationIdentifiabilityDimension:
    def test_pass_when_all_channels_have_variation(self) -> None:
        result = saturation_identifiability_dimension(
            [_contribution("search"), _contribution("tv")],
            {"search": 0.3, "tv": 0.2},
        )

        assert result.status == TrustStatus.PASS

    def test_weak_for_spend_invariant_channel(self) -> None:
        result = saturation_identifiability_dimension(
            [_contribution("tv", spend_invariant=True)],
            {"tv": 0.0},
        )

        assert result.status == TrustStatus.WEAK
        assert "tv" in result.explanation

    def test_weak_for_low_cv_channel(self) -> None:
        result = saturation_identifiability_dimension(
            [_contribution("display")],
            {"display": 0.02},
        )

        assert result.status == TrustStatus.WEAK

    def test_moderate_when_cv_unavailable(self) -> None:
        result = saturation_identifiability_dimension([_contribution("search")], None)

        assert result.status == TrustStatus.MODERATE


class TestLiftTestConsistencyDimension:
    def test_pass_when_intervals_overlap_and_mean_inside_lift_interval(self) -> None:
        result = lift_test_consistency_dimension(
            [_contribution("search", roi_mean=2.0, roi_low=1.8, roi_high=2.2)],
            {"search": LiftPrior(mean_roi=2.0, sd_roi=0.2)},
            {"search": 100.0, "tv": 100.0},
        )

        assert result.status == TrustStatus.PASS
        assert "covering 50% of media spend" in result.explanation
        assert "1 channel agrees with experiment evidence" in result.explanation

    def test_moderate_when_intervals_overlap_but_mean_outside_lift_interval(self) -> None:
        result = lift_test_consistency_dimension(
            [_contribution("search", roi_mean=2.6, roi_low=2.2, roi_high=2.7)],
            {"search": LiftPrior(mean_roi=2.0, sd_roi=0.2)},
            {"search": 100.0, "tv": 100.0},
        )

        assert result.status == TrustStatus.MODERATE
        assert (
            "search was pulled higher and should be treated as experiment-led"
            in result.explanation
        )

    def test_moderate_when_pulled_lower(self) -> None:
        result = lift_test_consistency_dimension(
            [_contribution("search", roi_mean=1.4, roi_low=1.2, roi_high=1.8)],
            {"search": LiftPrior(mean_roi=2.0, sd_roi=0.2)},
        )

        assert result.status == TrustStatus.MODERATE
        assert (
            "search was pulled lower and should be treated as experiment-led"
            in result.explanation
        )

    def test_weak_when_intervals_do_not_overlap(self) -> None:
        result = lift_test_consistency_dimension(
            [_contribution("search", roi_mean=4.0, roi_low=3.8, roi_high=4.2)],
            {"search": LiftPrior(mean_roi=2.0, sd_roi=0.2)},
        )

        assert result.status == TrustStatus.WEAK
        assert "search conflicts with test" in result.explanation

    def test_moderate_when_posterior_summary_missing_channel(self) -> None:
        result = lift_test_consistency_dimension(
            [_contribution("tv")],
            {"search": LiftPrior(mean_roi=2.0, sd_roi=0.2)},
        )

        assert result.status == TrustStatus.MODERATE
        assert "search missing posterior summary" in result.explanation


class TestBuildTrustCard:
    def test_builds_expected_core_dimensions(self) -> None:
        card = build_trust_card(
            _summary(
                convergence=_convergence(),
                predictive=_predictive([100.0, 110.0]),
                contributions=[_contribution("search")],
            ),
            holdout_actuals=[100.0, 100.0],
            prior_sensitivity=0.05,
            spend_cv_by_channel={"search": 0.2},
        )

        names = [dimension.name for dimension in card.dimensions]
        assert names == [
            "convergence",
            "holdout_accuracy",
            "prior_sensitivity",
            "saturation_identifiability",
        ]
        assert card.has_weak_dimension is False

    def test_includes_optional_refresh_and_lift_dimensions(self) -> None:
        card = build_trust_card(
            _summary(
                convergence=_convergence(),
                contributions=[_contribution("search", roi_mean=2.0)],
            ),
            prior_sensitivity=0.05,
            spend_cv_by_channel={"search": 0.2},
            refresh_diff=RefreshDiff(
                previous_run_id="a",
                current_run_id="b",
                movements=[_movement(MovementClass.WITHIN_PRIOR_CI)],
            ),
            lift_test_priors={"search": LiftPrior(mean_roi=2.0, sd_roi=0.2)},
        )

        names = [dimension.name for dimension in card.dimensions]
        assert "refresh_stability" in names
        assert "lift_test_consistency" in names


def _movement_list(cls: MovementClass, count: int) -> list[ParameterMovement]:
    return [_movement(cls) for _ in range(count)]
