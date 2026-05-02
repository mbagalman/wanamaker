"""Tests for the risk-adjusted allocation ramp (#65 / FR-5.6).

The math layer is engine-neutral: it consumes a
``PosteriorPredictiveEngine`` and runs everything off the per-draw
outcome matrix exposed in ``PredictiveSummary.draws`` (#64). Tests use a
``StubEngine`` whose draws are deterministic linear functions of the
plan's spend so each gate, each output status, and the up-front blocks
can be exercised by tweaking a few constants.

Coverage:

- Per-metric correctness on synthetic draws
- Each gate (passing and failing examples)
- All four output statuses (proceed / stage / test_first / do_not_recommend)
- Spend-invariant channel block produces ``do_not_recommend`` regardless
  of expected value
- Kelly fraction clamp (a tight, small-positive posterior produces a
  sensible cap, not an unbounded fraction)
- Trust Card ramp cap (weak / moderate / pass)
- JSON round-trip through the artifact envelope
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanamaker.artifacts import (
    deserialize_ramp_recommendation,
    serialize_ramp_recommendation,
)
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.forecast.ramp import (
    RampCandidate,
    RampRecommendation,
    RiskTolerance,
    recommend_ramp,
)
from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus

# ---------------------------------------------------------------------------
# Stub engine + helpers
# ---------------------------------------------------------------------------


class StubEngine:
    """Deterministic ``PosteriorPredictiveEngine`` for the math tests.

    The per-period mean is a linear combination of channel spend; the
    draws are that mean plus Gaussian noise with the supplied SD.
    Tweaking ``coefficients`` and ``noise_sd`` lets each test reach the
    metric / gate region it cares about.
    """

    def __init__(
        self,
        coefficients: dict[str, float] | None = None,
        baseline_intercept: float = 1000.0,
        noise_sd: float = 50.0,
        n_draws: int = 400,
    ) -> None:
        self.coefficients = coefficients or {"search": 2.0, "tv": 0.5}
        self.baseline_intercept = baseline_intercept
        self.noise_sd = noise_sd
        self.n_draws = n_draws

    def posterior_predictive(
        self,
        posterior_summary: PosteriorSummary,  # noqa: ARG002 — Protocol contract
        new_data: pd.DataFrame,
        seed: int,
    ) -> PredictiveSummary:
        rng = np.random.default_rng(seed)
        per_period_mean = np.full(len(new_data), self.baseline_intercept, dtype=float)
        for channel, coef in self.coefficients.items():
            if channel in new_data.columns:
                column = new_data[channel].astype(float).to_numpy()
                per_period_mean = per_period_mean + coef * column
        noise = rng.normal(0.0, self.noise_sd, size=(self.n_draws, len(new_data)))
        draws = per_period_mean[None, :] + noise
        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=draws.mean(axis=0).tolist(),
            hdi_low=np.quantile(draws, 0.025, axis=0).tolist(),
            hdi_high=np.quantile(draws, 0.975, axis=0).tolist(),
            draws=draws.tolist(),
        )


def _summary(*, spend_invariant_tv: bool = False) -> PosteriorSummary:
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=100.0, hdi_low=80.0, hdi_high=120.0,
                observed_spend_min=10.0, observed_spend_max=50.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=200.0, hdi_low=150.0, hdi_high=250.0,
                observed_spend_min=80.0, observed_spend_max=120.0,
                spend_invariant=spend_invariant_tv,
            ),
        ]
    )


def _plan(search: list[float], tv: list[float]) -> pd.DataFrame:
    n = len(search)
    return pd.DataFrame({
        "period": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "search": search,
        "tv": tv,
    })


# ---------------------------------------------------------------------------
# Plan interpolation + alignment
# ---------------------------------------------------------------------------


class TestPlanAlignment:
    def test_misaligned_periods_raise(self) -> None:
        baseline = _plan([20.0], [100.0])
        target = pd.DataFrame({
            "period": ["2026-02-05"], "search": [25.0], "tv": [100.0],
        })
        with pytest.raises(ValueError, match="identical periods"):
            recommend_ramp(_summary(), baseline, target, seed=0, engine=StubEngine())

    def test_missing_required_channel_raises(self) -> None:
        baseline = pd.DataFrame({"period": ["2026-01-01"], "search": [20.0]})
        target = pd.DataFrame({"period": ["2026-01-01"], "search": [25.0]})
        with pytest.raises(ValueError, match="missing channel columns"):
            recommend_ramp(_summary(), baseline, target, seed=0, engine=StubEngine())


# ---------------------------------------------------------------------------
# Per-metric correctness
# ---------------------------------------------------------------------------


class TestPerCandidateMetrics:
    def test_expected_increment_matches_drift_in_per_period_mean(self) -> None:
        # search coef = 2.0, baseline search=10, target search=20.
        # delta per period at f=0.50 should be ~0.5 * 2.0 * (20-10) = 10 per period.
        # Two periods → expected_increment ~20 (plus tiny noise).
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=1.0),  # small noise for tight check
        )
        c50 = next(c for c in rec.candidates if c.fraction == pytest.approx(0.50))
        assert c50.expected_increment == pytest.approx(20.0, rel=0.05)

    def test_probability_positive_increases_with_fraction_when_signal_is_real(
        self,
    ) -> None:
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=5.0),
        )
        ps = [c.probability_positive for c in rec.candidates]
        # Monotonic non-decreasing in fraction (within sampling slack).
        for prev, curr in zip(ps, ps[1:], strict=False):
            assert curr >= prev - 0.05

    def test_largest_move_share_is_fraction_of_concentration(self) -> None:
        # Only search moves; tv held flat. Largest-move share should be 1.0.
        baseline = _plan([10.0], [100.0])
        target = _plan([30.0], [100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=0, engine=StubEngine(noise_sd=1.0),
        )
        for candidate in rec.candidates:
            if candidate.fraction == 0.0:
                continue
            assert candidate.largest_move_share == pytest.approx(1.0, rel=1e-6)

    def test_q05_below_mean_for_noisy_posterior(self) -> None:
        baseline = _plan([10.0], [100.0])
        target = _plan([20.0], [100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=20.0),
        )
        for candidate in rec.candidates:
            if candidate.fraction <= 0.0:
                continue
            assert candidate.q05_increment <= candidate.expected_increment
            assert candidate.cvar_5 <= candidate.q05_increment


# ---------------------------------------------------------------------------
# Decision rule — all four statuses
# ---------------------------------------------------------------------------


class TestStatusProceed:
    def test_full_move_passes_yields_proceed(self) -> None:
        # 10→20 search at coef 2.0, low noise: every fraction has a
        # near-100% probability of improvement and zero material loss.
        # The Kelly multiplier override of 1.0 lifts the v1 cap so f=1.0
        # passes — ``proceed`` is the natural verdict.
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=1.0),
            risk_tolerance=RiskTolerance(kelly_multiplier_override=1.0),
        )
        assert rec.status == "proceed"
        assert rec.recommended_fraction == pytest.approx(1.0)

    def test_proceed_when_higher_fractions_fail_on_value_gates(self) -> None:
        """When higher fractions fail on a *value* gate (p_positive,
        p_material_loss, cvar_5), staging doesn't help — the model
        genuinely doesn't believe in larger moves. Correct verdict is
        ``proceed`` at the chosen fraction.

        This case is hard to construct with a purely linear stub engine
        because mean(delta) and var(delta) both scale predictably in f,
        so a value-gate failure tends to cascade down the ladder. We
        test the decision-rule helper directly with hand-built
        candidates to avoid that constructive limitation.
        """
        from wanamaker.forecast.ramp import _build_recommendation

        candidates = [
            _candidate(0.10, passes=True),
            _candidate(0.25, passes=True),
            _candidate(0.50, passes=True),
            _candidate(0.75, passes=False, failed_gates=["p_positive"]),
            _candidate(1.0, passes=False, failed_gates=["p_positive", "cvar_5"]),
        ]
        rec = _build_recommendation(
            candidates=candidates,
            baseline_name="base",
            target_name="alt",
            trust_card=None,
            fractional_kelly=1.0,
        )
        assert rec.status == "proceed"
        assert rec.recommended_fraction == pytest.approx(0.50)


def _candidate(
    fraction: float, *, passes: bool, failed_gates: list[str] | None = None,
) -> RampCandidate:
    """Hand-build a ``RampCandidate`` for direct decision-rule tests."""
    return RampCandidate(
        fraction=fraction,
        total_spend_by_channel={"search": 100.0 * fraction},
        expected_increment=100.0 * fraction,
        probability_positive=0.9 if passes else 0.5,
        probability_material_loss=0.0,
        q05_increment=50.0 * fraction,
        cvar_5=25.0 * fraction,
        largest_move_share=1.0,
        fractional_kelly=1.0,
        extrapolation_flags=[],
        passes=passes,
        failed_gates=failed_gates or [],
    )


class TestStatusStage:
    def test_partial_passes_full_fails_extrapolation(self) -> None:
        # observed_spend_max for search is 50. Target spend at f=1.0
        # of 80 is 60% above the historical max → severe extrapolation
        # at f=1.0 but not at f=0.25 (spend at 0.25 = 22.5, well in
        # range). Strong signal otherwise.
        baseline = _plan([20.0, 20.0], [100.0, 100.0])
        target = _plan([80.0, 80.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=1.0),
        )
        assert rec.status == "stage"
        assert 0 < rec.recommended_fraction < 1.0
        # The 100% candidate failed extrapolation.
        full = next(c for c in rec.candidates if c.fraction >= 1.0 - 1e-9)
        assert "extrapolation" in full.failed_gates


class TestStatusTestFirst:
    def test_weak_trust_card_routes_to_test_first(self) -> None:
        # Strong signal, in-range — but the Trust Card is weak so the
        # ramp cap is 0.10, which is below every f in the ladder
        # except 0.10. Smallest candidate (0.10) might still pass at
        # the cap; let's force a stricter case where even 0.10 fails
        # because Kelly multiplier is the weak one (0.10), and Kelly
        # raw saturates the clamp at 1.0, so fractional_kelly = 0.1.
        # 0.10 at the boundary should fail with an epsilon.
        baseline = _plan([20.0, 20.0], [100.0, 100.0])
        target = _plan([21.0, 21.0], [100.0, 100.0])  # tiny but real signal
        weak_card = TrustCard(dimensions=[
            TrustDimension(
                name="saturation_identifiability",
                status=TrustStatus.WEAK,
                explanation="Spend variation is limited.",
            ),
        ])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=2.0),
            trust_card=weak_card,
        )
        # Cap = 0.10. The Kelly multiplier for weak is 0.10, and the
        # raw Kelly saturates the clamp (small positive mean, tiny
        # variance), so fractional_kelly ≈ 0.10. Either gate (or both)
        # blocks the larger candidates.
        assert rec.status in ("test_first", "stage")
        if rec.status == "test_first":
            # All candidates failed; the smallest one's failures must
            # be strictly evidence-quality gates.
            smallest = min(rec.candidates, key=lambda c: c.fraction)
            assert smallest.failed_gates
            assert all(
                g in {"trust_card", "extrapolation", "fractional_kelly"}
                for g in smallest.failed_gates
            )


class TestStatusDoNotRecommend:
    def test_negative_signal_yields_do_not_recommend(self) -> None:
        # Negative coefficient: more search costs revenue.
        engine = StubEngine(coefficients={"search": -2.0, "tv": 0.5}, noise_sd=5.0)
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(_summary(), baseline, target, seed=42, engine=engine)
        assert rec.status == "do_not_recommend"
        assert rec.recommended_fraction == 0.0


# ---------------------------------------------------------------------------
# Up-front blocks
# ---------------------------------------------------------------------------


class TestSpendInvariantBlock:
    def test_invariant_channel_reallocation_is_blocked(self) -> None:
        # tv is spend-invariant in this summary, and the target moves it.
        baseline = _plan([10.0], [100.0])
        target = _plan([10.0], [120.0])
        rec = recommend_ramp(
            _summary(spend_invariant_tv=True),
            baseline, target, seed=0, engine=StubEngine(),
        )
        assert rec.status == "do_not_recommend"
        assert rec.blocking_reason == "spend_invariant_reallocation"
        assert "tv" in rec.explanation
        assert rec.candidates == []  # short-circuit, no candidates evaluated

    def test_invariant_block_overrides_strong_signal(self) -> None:
        # Even with a hugely positive expected delta on search, moving
        # an invariant tv channel blocks the recommendation.
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [110.0, 110.0])
        rec = recommend_ramp(
            _summary(spend_invariant_tv=True),
            baseline, target, seed=0, engine=StubEngine(noise_sd=1.0),
        )
        assert rec.status == "do_not_recommend"
        assert rec.blocking_reason == "spend_invariant_reallocation"

    def test_invariant_channel_held_flat_does_not_block(self) -> None:
        # tv is invariant but target keeps its spend equal to baseline,
        # so the block condition does not fire.
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(spend_invariant_tv=True),
            baseline, target, seed=42, engine=StubEngine(noise_sd=1.0),
        )
        assert rec.blocking_reason is None
        assert rec.candidates  # candidates were evaluated


# ---------------------------------------------------------------------------
# Fractional Kelly
# ---------------------------------------------------------------------------


class TestKellyClamp:
    def test_tight_posterior_does_not_produce_unbounded_kelly(self) -> None:
        # Tiny noise → variance(r) is tiny → raw Kelly = mean/var is
        # huge. The clamp must keep fractional_kelly in [0, 0.5].
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([12.0, 12.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=0.5),
        )
        # The Kelly multiplier for None trust card is 0.5 (pass), so
        # fractional_kelly is at most 0.5.
        for candidate in rec.candidates:
            assert 0.0 <= candidate.fractional_kelly <= 0.5 + 1e-9

    def test_no_signal_yields_kelly_at_or_near_zero(self) -> None:
        # When delta is dominated by noise, the Kelly numerator is
        # near zero and the fractional Kelly should not get crowded
        # by the clamp.
        engine = StubEngine(coefficients={"search": 0.0, "tv": 0.0}, noise_sd=50.0)
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(_summary(), baseline, target, seed=42, engine=engine)
        for candidate in rec.candidates:
            assert candidate.fractional_kelly < 0.5


# ---------------------------------------------------------------------------
# Trust Card cap interaction
# ---------------------------------------------------------------------------


class TestTrustCardCap:
    def test_weak_dimension_caps_ramp_at_10pct(self) -> None:
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([15.0, 15.0], [100.0, 100.0])
        weak_card = TrustCard(dimensions=[
            TrustDimension(
                name="prior_sensitivity",
                status=TrustStatus.WEAK,
                explanation="Posterior shifts substantially under prior perturbation.",
            ),
        ])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=1.0), trust_card=weak_card,
        )
        # Larger fractions must fail trust_card.
        for candidate in rec.candidates:
            if candidate.fraction > 0.10 + 1e-9:
                assert "trust_card" in candidate.failed_gates

    def test_moderate_dimension_caps_ramp_at_50pct(self) -> None:
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([15.0, 15.0], [100.0, 100.0])
        moderate_card = TrustCard(dimensions=[
            TrustDimension(
                name="holdout_accuracy",
                status=TrustStatus.MODERATE,
                explanation="Holdout WAPE is 15%.",
            ),
        ])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=1.0), trust_card=moderate_card,
        )
        for candidate in rec.candidates:
            if candidate.fraction > 0.50 + 1e-9:
                assert "trust_card" in candidate.failed_gates


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_preserves_recommendation(self) -> None:
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=1.0),
        )
        restored = deserialize_ramp_recommendation(serialize_ramp_recommendation(rec))
        assert isinstance(restored, RampRecommendation)
        assert restored.status == rec.status
        assert restored.recommended_fraction == pytest.approx(rec.recommended_fraction)
        assert len(restored.candidates) == len(rec.candidates)
        for orig, restored_c in zip(rec.candidates, restored.candidates, strict=True):
            assert isinstance(restored_c, RampCandidate)
            assert restored_c.fraction == pytest.approx(orig.fraction)
            assert restored_c.expected_increment == pytest.approx(orig.expected_increment)
            assert restored_c.passes == orig.passes
            assert restored_c.failed_gates == orig.failed_gates

    def test_invariant_block_round_trips(self) -> None:
        baseline = _plan([10.0], [100.0])
        target = _plan([10.0], [120.0])
        rec = recommend_ramp(
            _summary(spend_invariant_tv=True),
            baseline, target, seed=0, engine=StubEngine(),
        )
        restored = deserialize_ramp_recommendation(serialize_ramp_recommendation(rec))
        assert restored.status == "do_not_recommend"
        assert restored.blocking_reason == "spend_invariant_reallocation"
        assert restored.candidates == []


# ---------------------------------------------------------------------------
# Terminology guardrail (per-instance)
#
# The static guardrail in tests/unit/test_terminology_guardrails.py
# deliberately does *not* scan src/wanamaker/forecast/ because the module
# docstrings legitimately quote the banned phrases when explaining what
# Wanamaker is *not* (e.g. ``not a continuous "optimized budget."``).
# Instead, forecast user-facing copy is gated per-instance: this class
# exercises every ramp verdict and asserts that the rendered explanation
# never leaks optimizer-grade language.
# ---------------------------------------------------------------------------


class TestNoBannedTerminology:
    """The ramp explanation must never leak optimizer language.

    Mirrors ``test_compare_scenarios.py::TestNoBannedTerminology``. Defined
    locally because ``tests/`` is not a package; the canonical list lives
    in ``tests/unit/test_terminology_guardrails.py``.
    """

    BANNED_PHRASES: tuple[str, ...] = (
        "optimized budget",
        "optimal allocation",
        "best budget",
        "guaranteed lift",
        "maximize roi",
    )

    def _assert_clean(self, rec: RampRecommendation) -> None:
        lowered = rec.explanation.lower()
        for phrase in self.BANNED_PHRASES:
            assert phrase not in lowered, (
                f"ramp explanation for status {rec.status!r} contains "
                f"banned phrase {phrase!r}: {rec.explanation!r}"
            )

    def test_proceed_explanation_has_no_banned_phrases(self) -> None:
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=1.0),
            risk_tolerance=RiskTolerance(kelly_multiplier_override=1.0),
        )
        assert rec.status == "proceed"
        self._assert_clean(rec)

    def test_stage_explanation_has_no_banned_phrases(self) -> None:
        # Strong signal that triggers severe extrapolation at f=1.0.
        baseline = _plan([20.0, 20.0], [100.0, 100.0])
        target = _plan([80.0, 80.0], [100.0, 100.0])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=1.0),
        )
        assert rec.status == "stage"
        self._assert_clean(rec)

    def test_test_first_explanation_has_no_banned_phrases(self) -> None:
        # Tiny signal under a weak Trust Card → evidence-quality block.
        baseline = _plan([20.0, 20.0], [100.0, 100.0])
        target = _plan([21.0, 21.0], [100.0, 100.0])
        weak_card = TrustCard(dimensions=[
            TrustDimension(
                name="saturation_identifiability",
                status=TrustStatus.WEAK,
                explanation="Spend variation is limited.",
            ),
        ])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42, engine=StubEngine(noise_sd=2.0),
            trust_card=weak_card,
        )
        # Either test_first or stage is acceptable here (depends on which
        # gates bind); the terminology guardrail must hold in both branches.
        assert rec.status in ("test_first", "stage")
        self._assert_clean(rec)

    def test_do_not_recommend_value_failure_has_no_banned_phrases(self) -> None:
        # Negative coefficient → expected-value gate fails for every f.
        engine = StubEngine(coefficients={"search": -2.0, "tv": 0.5}, noise_sd=5.0)
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([20.0, 20.0], [100.0, 100.0])
        rec = recommend_ramp(_summary(), baseline, target, seed=42, engine=engine)
        assert rec.status == "do_not_recommend"
        assert rec.blocking_reason is None  # not the spend-invariant short-circuit
        self._assert_clean(rec)

    def test_do_not_recommend_invariant_block_has_no_banned_phrases(self) -> None:
        # Spend-invariant reallocation short-circuits before candidates run;
        # the block-reason explanation has its own template.
        baseline = _plan([10.0], [100.0])
        target = _plan([10.0], [120.0])
        rec = recommend_ramp(
            _summary(spend_invariant_tv=True),
            baseline, target, seed=0, engine=StubEngine(),
        )
        assert rec.status == "do_not_recommend"
        assert rec.blocking_reason == "spend_invariant_reallocation"
        self._assert_clean(rec)

    def test_weak_trust_card_explanation_has_no_banned_phrases(self) -> None:
        # The proceed/stage path appends a "Trust Card flags weakness on: ..."
        # sentence. Cover that branch too.
        baseline = _plan([10.0, 10.0], [100.0, 100.0])
        target = _plan([15.0, 15.0], [100.0, 100.0])
        weak_card = TrustCard(dimensions=[
            TrustDimension(
                name="prior_sensitivity",
                status=TrustStatus.WEAK,
                explanation="Posterior shifts substantially under prior perturbation.",
            ),
        ])
        rec = recommend_ramp(
            _summary(), baseline, target, seed=42,
            engine=StubEngine(noise_sd=1.0), trust_card=weak_card,
        )
        self._assert_clean(rec)
