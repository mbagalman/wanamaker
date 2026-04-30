"""Trust Card computation (FR-5.4).

This module converts completed fit artifacts into the six v1 Trust Card
dimensions. It is deliberately deterministic and template-friendly: every
dimension returns a status plus a plain-English explanation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from wanamaker.engine.summary import ChannelContributionSummary, PosteriorSummary
from wanamaker.model.spec import LiftPrior
from wanamaker.refresh.classify import unexplained_fraction
from wanamaker.refresh.diff import RefreshDiff
from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus

CONVERGENCE_RHAT_PASS = 1.01
CONVERGENCE_RHAT_MODERATE = 1.05
CONVERGENCE_ESS_PASS = 400.0
CONVERGENCE_ESS_MODERATE = 100.0
HOLDOUT_WAPE_PASS = 0.10
HOLDOUT_WAPE_MODERATE = 0.20
REFRESH_UNEXPLAINED_PASS = 0.10
REFRESH_UNEXPLAINED_MODERATE = 0.25
PRIOR_SHIFT_PASS = 0.10
PRIOR_SHIFT_MODERATE = 0.25
SPEND_CV_THRESHOLD = 0.10
NORMAL_95_Z = 1.959963984540054


@dataclass(frozen=True)
class PriorSensitivityResult:
    """Summary of a prior-sensitivity refit.

    ``max_relative_shift`` is the largest posterior mean movement observed
    under a prior perturbation, expressed relative to the original posterior
    magnitude. For example, ``0.12`` means the largest movement was 12%.
    """

    max_relative_shift: float


def build_trust_card(
    summary: PosteriorSummary,
    *,
    holdout_actuals: Sequence[float] | None = None,
    refresh_diff: RefreshDiff | None = None,
    prior_sensitivity: PriorSensitivityResult | float | None = None,
    spend_cv_by_channel: Mapping[str, float] | None = None,
    lift_test_priors: Mapping[str, LiftPrior] | None = None,
) -> TrustCard:
    """Build a v1 Trust Card from completed fit artifacts.

    Args:
        summary: Engine-neutral posterior summary from the completed fit.
        holdout_actuals: Actual target values for the same periods as
            ``summary.in_sample_predictive`` when that predictive summary is
            being used as a held-out-period prediction.
        refresh_diff: Optional diff against a previous run. When absent, the
            refresh-stability dimension is omitted.
        prior_sensitivity: Optional prior sensitivity result. When absent, the
            dimension is included as ``moderate`` because the check has not yet
            been run.
        spend_cv_by_channel: Optional coefficient of variation by channel,
            usually from the data diagnostic.
        lift_test_priors: Optional lift-test priors keyed by channel. When
            absent or empty, the lift-test consistency dimension is omitted.

    Returns:
        Trust card with deterministic v1 dimensions.
    """
    dimensions = [
        convergence_dimension(summary),
        holdout_accuracy_dimension(summary, holdout_actuals),
        prior_sensitivity_dimension(prior_sensitivity),
        saturation_identifiability_dimension(summary.channel_contributions, spend_cv_by_channel),
    ]

    if refresh_diff is not None:
        dimensions.append(refresh_stability_dimension(refresh_diff))
    if lift_test_priors:
        dimensions.append(
            lift_test_consistency_dimension(summary.channel_contributions, lift_test_priors)
        )

    return TrustCard(dimensions=dimensions)


def convergence_dimension(summary: PosteriorSummary) -> TrustDimension:
    """Assess sampler convergence from ``ConvergenceSummary``."""
    convergence = summary.convergence
    if convergence is None:
        return TrustDimension(
            name="convergence",
            status=TrustStatus.WEAK,
            explanation="Convergence diagnostics are missing.",
        )

    r_hat = convergence.max_r_hat
    ess = convergence.min_ess_bulk
    divergences = convergence.n_divergences

    if r_hat is None or ess is None:
        return TrustDimension(
            name="convergence",
            status=TrustStatus.MODERATE,
            explanation="R-hat or ESS was not available; review sampler output manually.",
        )

    if r_hat < CONVERGENCE_RHAT_PASS and ess > CONVERGENCE_ESS_PASS and divergences == 0:
        return TrustDimension(
            name="convergence",
            status=TrustStatus.PASS,
            explanation=(
                f"Sampler diagnostics passed: max R-hat {r_hat:.3f}, "
                f"min ESS {ess:.0f}, divergences {divergences}."
            ),
        )

    if r_hat <= CONVERGENCE_RHAT_MODERATE and ess >= CONVERGENCE_ESS_MODERATE:
        return TrustDimension(
            name="convergence",
            status=TrustStatus.MODERATE,
            explanation=(
                f"Sampler diagnostics are mixed: max R-hat {r_hat:.3f}, "
                f"min ESS {ess:.0f}, divergences {divergences}."
            ),
        )

    return TrustDimension(
        name="convergence",
        status=TrustStatus.WEAK,
        explanation=(
            f"Sampler diagnostics are weak: max R-hat {r_hat:.3f}, "
            f"min ESS {ess:.0f}, divergences {divergences}."
        ),
    )


def holdout_accuracy_dimension(
    summary: PosteriorSummary,
    actuals: Sequence[float] | None,
) -> TrustDimension:
    """Assess held-out predictive accuracy using WAPE."""
    predictive = summary.in_sample_predictive
    if predictive is None or actuals is None:
        return TrustDimension(
            name="holdout_accuracy",
            status=TrustStatus.MODERATE,
            explanation="Holdout accuracy was not evaluated for this run.",
        )

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(predictive.mean, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "holdout_actuals must have the same length as PredictiveSummary.mean "
            f"({len(y_true)} != {len(y_pred)})"
        )
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0.0:
        return TrustDimension(
            name="holdout_accuracy",
            status=TrustStatus.MODERATE,
            explanation="Holdout actuals sum to zero, so WAPE is undefined.",
        )

    wape = float(np.sum(np.abs(y_true - y_pred)) / denominator)
    if wape <= HOLDOUT_WAPE_PASS:
        status = TrustStatus.PASS
    elif wape <= HOLDOUT_WAPE_MODERATE:
        status = TrustStatus.MODERATE
    else:
        status = TrustStatus.WEAK

    return TrustDimension(
        name="holdout_accuracy",
        status=status,
        explanation=f"Holdout WAPE is {wape:.1%}.",
    )


def refresh_stability_dimension(refresh_diff: RefreshDiff) -> TrustDimension:
    """Assess refresh stability from the fraction of unexplained movements."""
    fraction = unexplained_fraction(refresh_diff.movements)
    n_movements = len(refresh_diff.movements)

    if fraction <= REFRESH_UNEXPLAINED_PASS:
        status = TrustStatus.PASS
    elif fraction <= REFRESH_UNEXPLAINED_MODERATE:
        status = TrustStatus.MODERATE
    else:
        status = TrustStatus.WEAK

    return TrustDimension(
        name="refresh_stability",
        status=status,
        explanation=(
            f"{fraction:.1%} of {n_movements} parameter movement(s) were unexplained."
        ),
    )


def prior_sensitivity_dimension(
    prior_sensitivity: PriorSensitivityResult | float | None,
) -> TrustDimension:
    """Assess posterior stability under perturbed priors."""
    if prior_sensitivity is None:
        return TrustDimension(
            name="prior_sensitivity",
            status=TrustStatus.MODERATE,
            explanation="Prior sensitivity was not evaluated for this run.",
        )

    shift = (
        float(prior_sensitivity.max_relative_shift)
        if isinstance(prior_sensitivity, PriorSensitivityResult)
        else float(prior_sensitivity)
    )

    if shift <= PRIOR_SHIFT_PASS:
        status = TrustStatus.PASS
    elif shift <= PRIOR_SHIFT_MODERATE:
        status = TrustStatus.MODERATE
    else:
        status = TrustStatus.WEAK

    return TrustDimension(
        name="prior_sensitivity",
        status=status,
        explanation=f"Maximum posterior shift under prior perturbation was {shift:.1%}.",
    )


def saturation_identifiability_dimension(
    channel_contributions: Sequence[ChannelContributionSummary],
    spend_cv_by_channel: Mapping[str, float] | None,
) -> TrustDimension:
    """Assess whether channel saturation can be learned from observed spend."""
    weak_channels = []
    moderate_channels = []
    cv_by_channel = spend_cv_by_channel or {}

    for contribution in channel_contributions:
        cv = cv_by_channel.get(contribution.channel)
        if contribution.spend_invariant or (cv is not None and cv < SPEND_CV_THRESHOLD):
            weak_channels.append(contribution.channel)
        elif cv is None:
            moderate_channels.append(contribution.channel)

    if weak_channels:
        names = ", ".join(sorted(weak_channels))
        return TrustDimension(
            name="saturation_identifiability",
            status=TrustStatus.WEAK,
            explanation=(
                f"Saturation is weakly identified for: {names}. "
                "Spend variation is too limited to estimate curve shape from data."
            ),
        )

    if moderate_channels:
        return TrustDimension(
            name="saturation_identifiability",
            status=TrustStatus.MODERATE,
            explanation=(
                "Spend variation diagnostics were unavailable for "
                f"{len(moderate_channels)} channel(s); identifiability is only partially assessed."
            ),
        )

    return TrustDimension(
        name="saturation_identifiability",
        status=TrustStatus.PASS,
        explanation="All channel spend series have enough variation for saturation checks.",
    )


def lift_test_consistency_dimension(
    channel_contributions: Sequence[ChannelContributionSummary],
    lift_test_priors: Mapping[str, LiftPrior],
) -> TrustDimension:
    """Compare posterior ROI intervals to supplied lift-test estimates."""
    contributions = {contribution.channel: contribution for contribution in channel_contributions}
    weak = []
    moderate = []
    missing = []

    for channel, prior in lift_test_priors.items():
        contribution = contributions.get(channel)
        if contribution is None:
            missing.append(channel)
            continue

        lift_low = prior.mean_roi - NORMAL_95_Z * prior.sd_roi
        lift_high = prior.mean_roi + NORMAL_95_Z * prior.sd_roi
        posterior_low = contribution.roi_hdi_low
        posterior_high = contribution.roi_hdi_high
        overlaps = posterior_low <= lift_high and lift_low <= posterior_high
        posterior_mean_in_lift_interval = lift_low <= contribution.roi_mean <= lift_high

        if not overlaps:
            weak.append(channel)
        elif not posterior_mean_in_lift_interval:
            moderate.append(channel)

    if weak:
        names = ", ".join(sorted(weak))
        return TrustDimension(
            name="lift_test_consistency",
            status=TrustStatus.WEAK,
            explanation=f"Posterior ROI does not overlap lift-test intervals for: {names}.",
        )

    if moderate or missing:
        details = []
        if moderate:
            details.append(
                "posterior means outside lift-test intervals: "
                f"{', '.join(sorted(moderate))}"
            )
        if missing:
            details.append(f"no posterior ROI summary: {', '.join(sorted(missing))}")
        return TrustDimension(
            name="lift_test_consistency",
            status=TrustStatus.MODERATE,
            explanation="Lift-test consistency is mixed (" + "; ".join(details) + ").",
        )

    return TrustDimension(
        name="lift_test_consistency",
        status=TrustStatus.PASS,
        explanation="Posterior ROI intervals are consistent with supplied lift tests.",
    )
