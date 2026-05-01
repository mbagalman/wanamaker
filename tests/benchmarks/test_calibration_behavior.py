"""Calibration-behavior benchmarks (issue #81).

Integration-level tests proving lift-test calibration moves the
posterior in the expected direction with the expected precision
sensitivity, without leaking into adstock or saturation parameters,
and that the Trust Card's ``lift_test_consistency`` dimension fires
correctly when the lift test agrees / disagrees with the data.

The unit tests in ``tests/unit/test_lift_test_calibration.py`` already
cover loading, schema validation, and the ``LiftPrior`` conversion.
This file complements those with **engine-level** behavior — real
PyMC fits — to make calibration a tested product contract, not just a
loader feature.

All tests are gated behind ``benchmark`` and ``engine`` markers, so the
fast unit-test CI tier does not run them; the engine-aware tier (or a
manual ``pytest -m benchmark`` invocation) does.

Runtime budget: each fit uses ``runtime_mode="quick"`` on the small
``lift_test_calibration`` benchmark dataset (80 weeks, 3 channels). On
a modern laptop a fit completes in ~30–90 seconds. The tests share
four module-scoped fits across all assertions to keep total runtime
bounded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from wanamaker.benchmarks.loaders import load_lift_test_calibration

pytestmark = [pytest.mark.benchmark, pytest.mark.engine]

# True ROI from the calibration benchmark dataset metadata. The dataset
# was generated with this value, so a tight lift-test prior centred here
# should pull the posterior toward 1.8.
_TRUE_ROI = 1.8

# Lift-test value used for the disagreement case. Far enough from the
# true ROI that even a wide model-implied posterior would not overlap a
# tight prior anchored here, which is what triggers ``weak`` on the
# Trust Card.
_DISAGREEMENT_ROI = 0.5

# Quick-mode runtime keeps total test time tractable. Standard or full
# would tighten the posteriors but would also push the suite well past
# the runtime budget. Tolerances below are sized for quick mode.
_RUNTIME_MODE = "quick"
_SEED = 7


def _build_spec(lift_priors: Mapping[str, Any] | None = None):
    """Construct a ``ModelSpec`` for the calibration benchmark dataset.

    Lazily imports engine modules so the file imports cleanly under the
    no-engine unit tier.
    """
    from wanamaker.model.spec import ChannelSpec, ModelSpec

    _, _, metadata = load_lift_test_calibration()
    channels = [ChannelSpec(name=name, category=name) for name in metadata["spend_columns"]]
    return ModelSpec(
        channels=channels,
        target_column=metadata["target_column"],
        date_column=metadata["date_column"],
        control_columns=list(metadata["control_columns"]),
        lift_test_priors=dict(lift_priors or {}),
    )


def _fit(lift_priors: Mapping[str, Any] | None = None):
    """Run a quick-mode PyMC fit with the given lift-test priors."""
    pytest.importorskip("pymc")
    from wanamaker.engine.pymc import PyMCEngine

    df, _, _ = load_lift_test_calibration()
    spec = _build_spec(lift_priors)
    engine = PyMCEngine()
    return engine.fit(spec, df, seed=_SEED, runtime_mode=_RUNTIME_MODE)


def _lift_prior(mean_roi: float, sd_roi: float):
    """Build a single-channel ``LiftPrior`` for ``paid_search``."""
    from wanamaker.model.spec import LiftPrior

    return {"paid_search": LiftPrior(mean_roi=mean_roi, sd_roi=sd_roi, confidence=0.95)}


def _paid_search_roi(fit_result: Any) -> tuple[float, float, float]:
    """Pull the ``paid_search`` posterior ROI mean and 95% HDI."""
    for c in fit_result.summary.channel_contributions:
        if c.channel == "paid_search":
            return c.roi_mean, c.roi_hdi_low, c.roi_hdi_high
    raise AssertionError("paid_search not in fit result")


def _parameter_mean(fit_result: Any, name: str) -> float:
    """Pull a posterior parameter mean by stable name (engine.summary)."""
    for p in fit_result.summary.parameters:
        if p.name == name:
            return float(p.mean)
    raise AssertionError(f"parameter {name!r} not in fit result")


# ---------------------------------------------------------------------------
# Fixtures: four shared fits
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_fit():
    """Uncalibrated fit. Reference for all comparison tests."""
    return _fit(lift_priors=None)


@pytest.fixture(scope="module")
def calibrated_at_true_tight():
    """Lift-test centred at true ROI (1.8) with a tight ±0.05 sd."""
    return _fit(lift_priors=_lift_prior(mean_roi=_TRUE_ROI, sd_roi=0.05))


@pytest.fixture(scope="module")
def calibrated_at_true_wide():
    """Lift-test centred at true ROI (1.8) with a wide ±0.5 sd.

    Same point estimate as ``calibrated_at_true_tight`` so the only
    difference between the two posteriors is precision.
    """
    return _fit(lift_priors=_lift_prior(mean_roi=_TRUE_ROI, sd_roi=0.5))


@pytest.fixture(scope="module")
def calibrated_at_disagreement():
    """Lift-test wildly disagreeing with the data (0.5 vs true 1.8).

    The model wants the posterior near 1.8 (the true ROI), but a tight
    calibration prior pulls toward 0.5. With a sufficiently tight
    prior, the posterior cannot reconcile with both — the Trust Card's
    ``lift_test_consistency`` dimension should fire ``weak``.
    """
    return _fit(lift_priors=_lift_prior(mean_roi=_DISAGREEMENT_ROI, sd_roi=0.05))


# ---------------------------------------------------------------------------
# Behavior contracts
# ---------------------------------------------------------------------------


def test_calibration_pulls_posterior_toward_disagreeing_lift_test(
    baseline_fit, calibrated_at_disagreement,
) -> None:
    """Directionality: a lift-test placed below the data-driven posterior
    pulls the calibrated posterior down toward it.

    The benchmark dataset's true ROI is 1.8, so the uncalibrated fit
    lands somewhere near 1.8 (within sampling). Setting the lift-test
    at 0.5 pulls the calibrated posterior down; the calibrated mean
    must be strictly below the baseline mean.
    """
    baseline_mean, _, _ = _paid_search_roi(baseline_fit)
    calibrated_mean, _, _ = _paid_search_roi(calibrated_at_disagreement)

    assert calibrated_mean < baseline_mean, (
        f"calibrated posterior mean {calibrated_mean:.3f} should be below "
        f"baseline mean {baseline_mean:.3f} when the lift-test is anchored "
        f"at {_DISAGREEMENT_ROI}"
    )
    # And materially below — sampling noise alone should not produce a
    # gap this large in quick mode.
    assert (baseline_mean - calibrated_mean) > 0.20, (
        "expected the disagreement lift-test to move the posterior at "
        f"least 0.20 ROI units below baseline; saw {baseline_mean - calibrated_mean:.3f}"
    )


def test_narrow_ci_pulls_posterior_more_than_wide_ci(
    calibrated_at_true_tight, calibrated_at_true_wide,
) -> None:
    """Precision sensitivity: same lift-test point (1.8), different CI
    widths. The narrow-CI fit's posterior should be closer to 1.8 than
    the wide-CI fit's posterior. Equivalently, the narrow fit's HDI
    should be narrower than the wide fit's HDI.
    """
    tight_mean, tight_low, tight_high = _paid_search_roi(calibrated_at_true_tight)
    wide_mean, wide_low, wide_high = _paid_search_roi(calibrated_at_true_wide)

    tight_distance = abs(tight_mean - _TRUE_ROI)
    wide_distance = abs(wide_mean - _TRUE_ROI)
    tight_width = tight_high - tight_low
    wide_width = wide_high - wide_low

    # Either the mean is pulled closer or the HDI is tighter — both are
    # legitimate signs of stronger pull. Requiring both would be too
    # brittle in quick mode where chains are short.
    assert tight_distance <= wide_distance + 0.01 or tight_width < wide_width, (
        "narrow-CI calibration should produce a posterior at least as "
        "close to the lift-test point or tighter than wide-CI calibration. "
        f"narrow: mean={tight_mean:.3f} HDI=[{tight_low:.3f}, {tight_high:.3f}] "
        f"wide: mean={wide_mean:.3f} HDI=[{wide_low:.3f}, {wide_high:.3f}]"
    )
    # Stronger separately-checkable invariant: tight HDI is narrower than
    # wide HDI by a non-trivial margin.
    assert tight_width < wide_width, (
        f"narrow-CI HDI width {tight_width:.3f} should be strictly less "
        f"than wide-CI HDI width {wide_width:.3f}"
    )


def test_calibration_does_not_modify_adstock_or_saturation(
    baseline_fit, calibrated_at_true_tight,
) -> None:
    """No transform leakage: calibration only constrains the channel
    coefficient prior. Adstock half-life, EC50, and slope should be
    unchanged within sampling noise between the calibrated and
    uncalibrated fits.

    Quick-mode chains are short, so we use generous relative-difference
    tolerances. A bug that re-routed the calibration prior into adstock
    or saturation would shift these by orders of magnitude — well
    outside the tolerance.
    """
    for parameter in (
        "channel.paid_search.half_life",
        "channel.paid_search.ec50",
        "channel.paid_search.slope",
    ):
        baseline_value = _parameter_mean(baseline_fit, parameter)
        calibrated_value = _parameter_mean(calibrated_at_true_tight, parameter)
        # ``half_life`` and ``ec50`` are positive; ``slope`` typically is
        # too. Use relative difference normalised by the larger magnitude
        # so the test is symmetric under tiny baseline values.
        scale = max(abs(baseline_value), abs(calibrated_value), 1e-6)
        relative_diff = abs(calibrated_value - baseline_value) / scale
        assert relative_diff <= 0.30, (
            f"{parameter}: baseline={baseline_value:.4f}, "
            f"calibrated={calibrated_value:.4f}, relative diff "
            f"{relative_diff:.2%} exceeds the 30% leakage tolerance"
        )


def test_trust_card_passes_when_lift_test_agrees_with_data(
    calibrated_at_true_tight,
) -> None:
    """Lift-test centred at the true ROI with a tight CI: the posterior
    should land near 1.8, overlapping the lift-test interval, and the
    Trust Card's ``lift_test_consistency`` dimension should be ``pass``.
    """
    from wanamaker.trust_card.card import TrustStatus
    from wanamaker.trust_card.compute import build_trust_card

    summary = calibrated_at_true_tight.summary
    lift_priors = _lift_prior(mean_roi=_TRUE_ROI, sd_roi=0.05)
    card = build_trust_card(summary, lift_test_priors=lift_priors)

    dim = next(
        d for d in card.dimensions if d.name == "lift_test_consistency"
    )
    assert dim.status == TrustStatus.PASS, (
        f"lift_test_consistency expected PASS when prior agrees with data; "
        f"got {dim.status.value} ({dim.explanation!r})"
    )


def test_trust_card_weak_when_lift_test_disagrees_with_data(
    calibrated_at_disagreement,
) -> None:
    """Lift-test centred at 0.5 (true ROI is 1.8) with a tight CI: even
    after the prior pulls the posterior down, the posterior interval
    will not fully overlap the lift-test interval. ``lift_test_consistency``
    should be ``weak``.

    This is the integration-level coverage the issue calls for —
    the unit tests in ``test_trust_card_compute.py`` cover the logic
    with synthetic inputs, but only an engine-level fit with a real
    disagreeing prior demonstrates that the dimension lands ``weak``
    end-to-end.
    """
    from wanamaker.trust_card.card import TrustStatus
    from wanamaker.trust_card.compute import build_trust_card

    summary = calibrated_at_disagreement.summary
    lift_priors = _lift_prior(mean_roi=_DISAGREEMENT_ROI, sd_roi=0.05)
    card = build_trust_card(summary, lift_test_priors=lift_priors)

    dim = next(
        d for d in card.dimensions if d.name == "lift_test_consistency"
    )
    assert dim.status == TrustStatus.WEAK, (
        f"lift_test_consistency expected WEAK when prior disagrees sharply "
        f"with data; got {dim.status.value} ({dim.explanation!r})"
    )
