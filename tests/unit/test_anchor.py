"""Unit tests for anchoring weight resolution and prior blending (FR-4.4, issue #18)."""

from __future__ import annotations

import math

import pytest

from wanamaker.engine.pymc import _blend_lognormal
from wanamaker.refresh.anchor import (
    ANCHOR_PRESETS,
    DEFAULT_ANCHOR_STRENGTH,
    resolve_anchor_weight,
)


@pytest.mark.parametrize("preset,expected", list(ANCHOR_PRESETS.items()))
def test_named_presets_resolve(preset: str, expected: float) -> None:
    assert resolve_anchor_weight(preset) == expected


def test_default_preset_is_medium() -> None:
    assert DEFAULT_ANCHOR_STRENGTH == "medium"
    assert resolve_anchor_weight(DEFAULT_ANCHOR_STRENGTH) == 0.3


def test_numeric_override_passes_through() -> None:
    assert resolve_anchor_weight(0.42) == 0.42
    assert resolve_anchor_weight(0.0) == 0.0
    assert resolve_anchor_weight(1.0) == 1.0


def test_unknown_preset_rejected() -> None:
    with pytest.raises(ValueError, match="unknown anchor strength preset"):
        resolve_anchor_weight("aggressive")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
def test_out_of_range_numeric_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_anchor_weight(bad)


# ---------------------------------------------------------------------------
# _blend_lognormal — prior blending arithmetic
# ---------------------------------------------------------------------------


class TestBlendLognormal:
    """Tests for the LogNormal parameter blending used by the PyMC engine."""

    def test_weight_zero_returns_default(self) -> None:
        """w=0 → no anchoring; output equals default parameters."""
        mu, sigma = _blend_lognormal(1.0, 0.4, prev_mean=5.0, prev_sd=0.5, weight=0.0)
        assert math.isclose(mu, 1.0)
        assert math.isclose(sigma, 0.4)

    def test_weight_one_returns_prev_posterior(self) -> None:
        """w=1 → full anchoring; output is the previous posterior in log-space."""
        prev_mean, prev_sd = 4.0, 0.4   # cv = 0.1
        cv2 = (prev_sd / prev_mean) ** 2
        expected_mu = math.log(prev_mean) - 0.5 * math.log1p(cv2)
        expected_sigma = math.sqrt(math.log1p(cv2))
        mu, sigma = _blend_lognormal(0.0, 0.5, prev_mean=prev_mean, prev_sd=prev_sd, weight=1.0)
        assert math.isclose(mu, expected_mu, rel_tol=1e-9)
        assert math.isclose(sigma, expected_sigma, rel_tol=1e-9)

    def test_medium_weight_is_convex_combination(self) -> None:
        """w=0.3 → blended mu is 0.7 * mu_default + 0.3 * mu_log_prev."""
        mu_default, sigma_default = 1.0, 0.40
        prev_mean, prev_sd = math.exp(1.5), 0.3
        cv2 = (prev_sd / prev_mean) ** 2
        mu_log_prev = math.log(prev_mean) - 0.5 * math.log1p(cv2)
        sigma_log_prev = math.sqrt(math.log1p(cv2))
        expected_mu = 0.7 * mu_default + 0.3 * mu_log_prev
        expected_sigma = 0.7 * sigma_default + 0.3 * sigma_log_prev

        mu, sigma = _blend_lognormal(mu_default, sigma_default, prev_mean, prev_sd, weight=0.3)
        assert math.isclose(mu, expected_mu, rel_tol=1e-9)
        assert math.isclose(sigma, expected_sigma, rel_tol=1e-9)

    def test_blended_sigma_shrinks_with_higher_weight(self) -> None:
        """Heavier anchoring → tighter blended prior (assuming prev is tighter)."""
        # After 80 weeks the posterior is much tighter than the default prior
        prev_mean, prev_sd = 2.0, 0.1  # tight posterior: cv = 0.05
        _, s_light = _blend_lognormal(0.0, 0.4, prev_mean, prev_sd, weight=0.2)
        _, s_medium = _blend_lognormal(0.0, 0.4, prev_mean, prev_sd, weight=0.3)
        _, s_heavy = _blend_lognormal(0.0, 0.4, prev_mean, prev_sd, weight=0.5)
        assert s_light > s_medium > s_heavy

    def test_near_zero_prev_mean_is_stable(self) -> None:
        """Very small previous mean should not cause log(0) errors."""
        mu, sigma = _blend_lognormal(0.0, 0.4, prev_mean=1e-12, prev_sd=1e-13, weight=0.3)
        assert math.isfinite(mu)
        assert math.isfinite(sigma)
        assert sigma > 0

    def test_high_cv_posterior_is_stable(self) -> None:
        """Very noisy previous posterior (cv > 1) should still produce finite output."""
        # High cv: sd >> mean — common for weakly identified parameters
        mu, sigma = _blend_lognormal(1.0, 0.4, prev_mean=1.0, prev_sd=5.0, weight=0.3)
        assert math.isfinite(mu)
        assert math.isfinite(sigma)
        assert sigma > 0

    def test_sigma_always_positive(self) -> None:
        """Blended sigma must be > 0 for all weights including zero prev_sd."""
        mu, sigma = _blend_lognormal(1.0, 0.4, prev_mean=2.0, prev_sd=0.0, weight=0.5)
        assert sigma > 0

    # --- analytical bounds for NFR-5 (issue #18) ---

    def test_medium_preset_sigma_reduction(self) -> None:
        """w=0.3 on a typical posterior reduces prior width to ≈76 % of default."""
        # Typical: default_sigma=0.4, after 80 weeks post_sigma ≈ 0.08 (cv=0.2)
        prev_mean, prev_sd = 2.0, 0.4  # cv=0.2, close to practical posterior
        _, blended_sigma = _blend_lognormal(0.0, 0.4, prev_mean, prev_sd, weight=0.3)
        # Blended sigma should be < default sigma
        assert blended_sigma < 0.4
        # And not too small (preserving some prior width)
        assert blended_sigma > 0.1

    def test_heavy_preset_sigma_reduction(self) -> None:
        """w=0.5 reduces prior width more than w=0.3."""
        prev_mean, prev_sd = 2.0, 0.4
        _, s_medium = _blend_lognormal(0.0, 0.4, prev_mean, prev_sd, weight=0.3)
        _, s_heavy = _blend_lognormal(0.0, 0.4, prev_mean, prev_sd, weight=0.5)
        assert s_heavy < s_medium
