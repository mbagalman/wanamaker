"""Unit tests for geometric adstock (FR-3.1, FR-3.4).

Every test uses known inputs and known outputs, as required by AGENTS.md
Hard Rule 4. The worked examples are derived from the canonical formula
documented in docs/references/adstock_and_saturation.md sec. 2.1.

The closed-form equivalence tests serve as a second independent check:
if the recursive implementation and the closed-form summation agree on
the same inputs, the recurrence is correct.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from wanamaker.transforms.adstock import (
    geometric_adstock,
    half_life_to_decay,
    weibull_adstock,
)

# ---------------------------------------------------------------------------
# half_life_to_decay
# ---------------------------------------------------------------------------


class TestHalfLifeToDecay:
    def test_half_life_1_gives_decay_half(self) -> None:
        # A half-life of 1 period means the effect halves every period.
        assert half_life_to_decay(1.0) == pytest.approx(0.5)

    def test_half_life_2(self) -> None:
        # theta = 0.5 ** (1/2) = sqrt(0.5)
        assert half_life_to_decay(2.0) == pytest.approx(0.5 ** 0.5)

    def test_half_life_4(self) -> None:
        assert half_life_to_decay(4.0) == pytest.approx(0.5 ** 0.25)

    def test_half_life_8(self) -> None:
        assert half_life_to_decay(8.0) == pytest.approx(0.5 ** (1 / 8))

    def test_zero_half_life_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            half_life_to_decay(0.0)

    def test_negative_half_life_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            half_life_to_decay(-1.0)

    def test_result_is_in_zero_one(self) -> None:
        for h in [0.5, 1, 2, 4, 8, 52]:
            decay = half_life_to_decay(h)
            assert 0.0 < decay < 1.0, f"half_life={h} produced decay={decay}"


# ---------------------------------------------------------------------------
# geometric_adstock — parameter validation
# ---------------------------------------------------------------------------


class TestGeometricAdstockValidation:
    def test_2d_input_rejected(self) -> None:
        """2-D arrays must raise ValueError, matching weibull_adstock behaviour."""
        with pytest.raises(ValueError, match="1-D array"):
            geometric_adstock(np.ones((3, 4)), decay=0.5)

    def test_decay_zero_is_valid(self) -> None:
        # decay=0 is the boundary case (no carryover); must not raise.
        result = geometric_adstock(np.array([1.0, 2.0]), decay=0.0)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_decay_one_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            geometric_adstock(np.array([1.0]), decay=1.0)

    def test_decay_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            geometric_adstock(np.array([1.0]), decay=1.5)

    def test_decay_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            geometric_adstock(np.array([1.0]), decay=-0.1)


# ---------------------------------------------------------------------------
# geometric_adstock — known worked examples
# ---------------------------------------------------------------------------


class TestGeometricAdstockKnownOutputs:
    def test_single_impulse_decay_half(self) -> None:
        """Canonical example: unit impulse, decay=0.5 (half-life=1 period).

        A_0 = 1.0
        A_1 = 0 + 0.5 * 1.0 = 0.5
        A_2 = 0 + 0.5 * 0.5 = 0.25
        A_3 = 0 + 0.5 * 0.25 = 0.125
        """
        spend = np.array([1.0, 0.0, 0.0, 0.0])
        result = geometric_adstock(spend, decay=0.5)
        np.testing.assert_allclose(result, [1.0, 0.5, 0.25, 0.125])

    def test_two_consecutive_periods(self) -> None:
        """Two periods of spend followed by silence.

        A_0 = 1.0
        A_1 = 1.0 + 0.5 * 1.0 = 1.5
        A_2 = 0.0 + 0.5 * 1.5 = 0.75
        """
        spend = np.array([1.0, 1.0, 0.0])
        result = geometric_adstock(spend, decay=0.5)
        np.testing.assert_allclose(result, [1.0, 1.5, 0.75])

    def test_zero_decay_is_identity(self) -> None:
        """decay=0 means no carryover: output must equal input exactly."""
        spend = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = geometric_adstock(spend, decay=0.0)
        np.testing.assert_array_equal(result, spend)

    def test_all_zeros_returns_zeros(self) -> None:
        spend = np.zeros(10)
        result = geometric_adstock(spend, decay=0.7)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_constant_spend_convergence(self) -> None:
        """Constant spend X converges to X / (1 - decay) as t -> inf.

        With spend=1 and decay=0.5, the steady-state value is 1/(1-0.5)=2.
        After 30 periods the series should be within 0.001 of 2.0.
        """
        spend = np.ones(30)
        result = geometric_adstock(spend, decay=0.5)
        assert result[-1] == pytest.approx(2.0, abs=1e-3)

    def test_half_life_property(self) -> None:
        """After h periods, a unit impulse should decay to exactly 0.5.

        Verified for several half-lives using the half_life_to_decay helper.
        """
        for half_life in [1, 2, 4, 8]:
            decay = half_life_to_decay(float(half_life))
            spend = np.zeros(half_life + 1)
            spend[0] = 1.0
            result = geometric_adstock(spend, decay=decay)
            assert result[half_life] == pytest.approx(0.5, rel=1e-6), (
                f"half_life={half_life}: expected 0.5 at t={half_life}, "
                f"got {result[half_life]}"
            )

    def test_three_period_manual(self) -> None:
        """Hand-computed example with decay=0.8.

        A_0 = 2.0
        A_1 = 3.0 + 0.8 * 2.0 = 4.6
        A_2 = 1.0 + 0.8 * 4.6 = 4.68
        """
        spend = np.array([2.0, 3.0, 1.0])
        result = geometric_adstock(spend, decay=0.8)
        np.testing.assert_allclose(result, [2.0, 4.6, 4.68])


# ---------------------------------------------------------------------------
# geometric_adstock — closed-form equivalence
# ---------------------------------------------------------------------------


class TestGeometricAdstockClosedForm:
    """The recursive implementation must agree with the closed-form sum.

    Closed form: A_t = sum_{k=0}^{t} decay^k * X_{t-k}

    This is an independent correctness check — if both agree, the
    recurrence is implemented correctly.
    """

    @staticmethod
    def _closed_form(spend: NDArray[np.float64], decay: float) -> NDArray[np.float64]:
        n = len(spend)
        out = np.zeros(n)
        for t in range(n):
            out[t] = sum(decay**k * spend[t - k] for k in range(t + 1))
        return out

    @pytest.mark.parametrize("decay", [0.0, 0.3, 0.5, 0.8, 0.99])
    def test_matches_closed_form(self, decay: float) -> None:
        rng = np.random.default_rng(42)
        spend = rng.uniform(0, 100, size=20).astype(np.float64)
        recursive = geometric_adstock(spend, decay=decay)
        closed = self._closed_form(spend, decay=decay)
        np.testing.assert_allclose(recursive, closed, rtol=1e-10)


# ---------------------------------------------------------------------------
# geometric_adstock — output properties
# ---------------------------------------------------------------------------


class TestGeometricAdstockOutputProperties:
    def test_output_shape_matches_input(self) -> None:
        spend = np.ones(52)
        result = geometric_adstock(spend, decay=0.5)
        assert result.shape == spend.shape

    def test_output_dtype_is_float64(self) -> None:
        spend = np.ones(5, dtype=np.float64)
        result = geometric_adstock(spend, decay=0.5)
        assert result.dtype == np.float64

    def test_integer_input_accepted(self) -> None:
        """Integer arrays must be coerced to float64 without error."""
        spend = np.array([1, 2, 3, 4], dtype=np.int32)
        result = geometric_adstock(spend, decay=0.5)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result[0], 1.0)

    def test_non_negative_spend_produces_non_negative_output(self) -> None:
        rng = np.random.default_rng(0)
        spend = rng.uniform(0, 1000, size=100)
        result = geometric_adstock(spend, decay=0.6)
        assert (result >= 0).all()

    def test_long_series_numerical_stability(self) -> None:
        """300 periods of constant spend should not accumulate floating-point
        error beyond float64 tolerance relative to the analytic steady state."""
        decay = 0.9
        spend = np.ones(300)
        result = geometric_adstock(spend, decay=decay)
        steady_state = 1.0 / (1.0 - decay)  # = 10.0
        assert result[-1] == pytest.approx(steady_state, rel=1e-9)


# ---------------------------------------------------------------------------
# weibull_adstock
# ---------------------------------------------------------------------------


class TestWeibullAdstockValidation:
    def test_shape_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="shape must be strictly positive"):
            weibull_adstock(np.array([1.0]), shape=0.0, scale=2.0)

    def test_scale_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="scale must be strictly positive"):
            weibull_adstock(np.array([1.0]), shape=2.0, scale=0.0)

    def test_variant_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="variant must be 'cdf' or 'pdf'"):
            weibull_adstock(np.array([1.0]), shape=2.0, scale=2.0, variant="gamma")

    def test_spend_must_be_one_dimensional(self) -> None:
        with pytest.raises(ValueError, match="1-D array"):
            weibull_adstock(np.ones((2, 2)), shape=2.0, scale=2.0)


class TestWeibullAdstockKnownOutputs:
    def test_cdf_single_impulse_default_shape_scale(self) -> None:
        """CDF variant with shape=2, scale=2 uses Weibull survival weights.

        For a unit impulse, output equals the lag weights:
        w_lag = exp(-((lag / 2) ** 2)).
        """
        spend = np.array([1.0, 0.0, 0.0, 0.0])
        result = weibull_adstock(spend)
        expected = np.exp(-np.array([0.0, 0.25, 1.0, 2.25]))
        np.testing.assert_allclose(result, expected)

    def test_cdf_shape_one_matches_geometric_decay(self) -> None:
        """CDF Weibull with shape=1 is geometric adstock.

        exp(-(lag / scale)) = exp(-1 / scale) ** lag.
        """
        spend = np.array([2.0, 3.0, 1.0, 0.0, 4.0])
        scale = 2.0
        weibull = weibull_adstock(spend, shape=1.0, scale=scale, variant="cdf")
        geometric = geometric_adstock(spend, decay=float(np.exp(-1.0 / scale)))
        np.testing.assert_allclose(weibull, geometric, rtol=1e-12)

    def test_pdf_single_impulse_has_lagged_peak_shape(self) -> None:
        """PDF variant uses normalized Weibull density weights.

        With shape=2 and scale=2, the first few raw weights are:
        (lag / 2) * exp(-((lag / 2) ** 2)), normalized by the maximum.
        """
        spend = np.array([1.0, 0.0, 0.0, 0.0])
        result = weibull_adstock(spend, shape=2.0, scale=2.0, variant="pdf")
        lags = np.array([1.0, 2.0, 3.0, 4.0])
        raw = (lags / 2.0) * np.exp(-((lags / 2.0) ** 2))
        expected = raw / raw.max()
        np.testing.assert_allclose(result, expected)

    def test_two_consecutive_periods_cdf(self) -> None:
        spend = np.array([1.0, 1.0, 0.0])
        weights = np.exp(-np.array([0.0, 0.25, 1.0]))
        result = weibull_adstock(spend, shape=2.0, scale=2.0)
        expected = np.array([
            1.0,
            1.0 + weights[1],
            weights[1] + weights[2],
        ])
        np.testing.assert_allclose(result, expected)


class TestWeibullAdstockOutputProperties:
    def test_output_shape_matches_input(self) -> None:
        spend = np.ones(52)
        result = weibull_adstock(spend, shape=2.0, scale=2.0)
        assert result.shape == spend.shape

    def test_output_dtype_is_float64(self) -> None:
        spend = np.ones(5, dtype=np.float64)
        result = weibull_adstock(spend, shape=2.0, scale=2.0)
        assert result.dtype == np.float64

    def test_integer_input_accepted(self) -> None:
        spend = np.array([1, 2, 3, 4], dtype=np.int32)
        result = weibull_adstock(spend, shape=2.0, scale=2.0)
        assert result.dtype == np.float64

    def test_empty_input_returns_empty_array(self) -> None:
        result = weibull_adstock(np.array([], dtype=np.float64), shape=2.0, scale=2.0)
        assert result.dtype == np.float64
        assert result.size == 0

    def test_non_negative_spend_produces_non_negative_output(self) -> None:
        rng = np.random.default_rng(0)
        spend = rng.uniform(0, 1000, size=100)
        result = weibull_adstock(spend, shape=2.0, scale=2.0)
        assert (result >= 0).all()
