"""Unit tests for default channel-category priors (FR-1.2).

Every test uses known inputs and known outputs (AGENTS.md Hard Rule 4).
The expected values are derived from docs/references/adstock_and_saturation.md
sec. 4 and the inline comments in model/priors.py.
"""

from __future__ import annotations

import math

import pytest

from wanamaker.data.taxonomy import DEFAULT_CHANNEL_CATEGORIES
from wanamaker.model.priors import ChannelPriors, default_priors_for_category


# ---------------------------------------------------------------------------
# ChannelPriors dataclass contract
# ---------------------------------------------------------------------------


class TestChannelPriorsContract:
    def test_is_frozen(self) -> None:
        priors = default_priors_for_category("paid_search")
        with pytest.raises((AttributeError, TypeError)):
            priors.half_life_mu = 99.0  # type: ignore[misc]

    def test_has_expected_fields(self) -> None:
        priors = default_priors_for_category("paid_search")
        assert hasattr(priors, "half_life_mu")
        assert hasattr(priors, "half_life_sigma")
        assert hasattr(priors, "hill_alpha_mu")
        assert hasattr(priors, "hill_alpha_sigma")

    def test_all_fields_are_floats(self) -> None:
        priors = default_priors_for_category("linear_tv")
        assert isinstance(priors.half_life_mu, float)
        assert isinstance(priors.half_life_sigma, float)
        assert isinstance(priors.hill_alpha_mu, float)
        assert isinstance(priors.hill_alpha_sigma, float)


# ---------------------------------------------------------------------------
# Coverage: all 10 taxonomy categories must have priors
# ---------------------------------------------------------------------------


class TestAllCategoriesCovered:
    @pytest.mark.parametrize("category", DEFAULT_CHANNEL_CATEGORIES)
    def test_category_has_priors(self, category: str) -> None:
        priors = default_priors_for_category(category)
        assert isinstance(priors, ChannelPriors)

    def test_all_ten_categories_covered(self) -> None:
        assert len(DEFAULT_CHANNEL_CATEGORIES) == 10
        for category in DEFAULT_CHANNEL_CATEGORIES:
            default_priors_for_category(category)  # must not raise


# ---------------------------------------------------------------------------
# Known prior medians (half-life)
# ---------------------------------------------------------------------------


class TestHalfLifeMedians:
    """exp(half_life_mu) should equal the documented median in weeks."""

    def _median(self, category: str) -> float:
        return math.exp(default_priors_for_category(category).half_life_mu)

    def test_paid_search_median_1_week(self) -> None:
        assert self._median("paid_search") == pytest.approx(1.0, rel=1e-6)

    def test_paid_social_median_2_weeks(self) -> None:
        assert self._median("paid_social") == pytest.approx(2.0, rel=1e-6)

    def test_video_median_3_weeks(self) -> None:
        assert self._median("video") == pytest.approx(3.0, rel=1e-6)

    def test_linear_tv_median_6_weeks(self) -> None:
        assert self._median("linear_tv") == pytest.approx(6.0, rel=1e-6)

    def test_ctv_median_4_weeks(self) -> None:
        assert self._median("ctv") == pytest.approx(4.0, rel=1e-6)

    def test_audio_podcast_median_4_5_weeks(self) -> None:
        assert self._median("audio_podcast") == pytest.approx(4.5, rel=1e-6)

    def test_display_programmatic_median_1_5_weeks(self) -> None:
        assert self._median("display_programmatic") == pytest.approx(1.5, rel=1e-6)

    def test_affiliate_median_1_week(self) -> None:
        assert self._median("affiliate") == pytest.approx(1.0, rel=1e-6)

    def test_email_crm_median_0_7_weeks(self) -> None:
        assert self._median("email_crm") == pytest.approx(0.7, rel=1e-6)

    def test_promotions_discounting_median_0_5_weeks(self) -> None:
        assert self._median("promotions_discounting") == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Ordering: long-memory channels must have higher medians than short-memory
# ---------------------------------------------------------------------------


class TestHalfLifeOrdering:
    def _median(self, category: str) -> float:
        return math.exp(default_priors_for_category(category).half_life_mu)

    def test_linear_tv_longer_than_ctv(self) -> None:
        assert self._median("linear_tv") > self._median("ctv")

    def test_ctv_longer_than_video(self) -> None:
        assert self._median("ctv") > self._median("video")

    def test_video_longer_than_paid_social(self) -> None:
        assert self._median("video") > self._median("paid_social")

    def test_paid_social_longer_than_paid_search(self) -> None:
        assert self._median("paid_social") > self._median("paid_search")

    def test_paid_search_longer_than_promotions(self) -> None:
        assert self._median("paid_search") > self._median("promotions_discounting")


# ---------------------------------------------------------------------------
# Hill alpha: all categories share documented median and sigma
# ---------------------------------------------------------------------------


class TestHillAlphaPriors:
    @pytest.mark.parametrize("category", DEFAULT_CHANNEL_CATEGORIES)
    def test_hill_alpha_median_is_1_5(self, category: str) -> None:
        priors = default_priors_for_category(category)
        median = math.exp(priors.hill_alpha_mu)
        assert median == pytest.approx(1.5, rel=1e-6), (
            f"{category}: expected Hill alpha median 1.5, got {median}"
        )

    @pytest.mark.parametrize("category", DEFAULT_CHANNEL_CATEGORIES)
    def test_hill_alpha_sigma_is_0_50(self, category: str) -> None:
        priors = default_priors_for_category(category)
        assert priors.hill_alpha_sigma == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Sigma values: short-memory vs long-memory convention
# ---------------------------------------------------------------------------


class TestHalfLifeSigmaConvention:
    """Short-memory channels use sigma 0.35; long-memory use 0.45."""

    def test_paid_search_sigma_0_35(self) -> None:
        assert default_priors_for_category("paid_search").half_life_sigma == pytest.approx(0.35)

    def test_affiliate_sigma_0_35(self) -> None:
        assert default_priors_for_category("affiliate").half_life_sigma == pytest.approx(0.35)

    def test_email_crm_sigma_0_35(self) -> None:
        assert default_priors_for_category("email_crm").half_life_sigma == pytest.approx(0.35)

    def test_promotions_sigma_0_35(self) -> None:
        assert default_priors_for_category("promotions_discounting").half_life_sigma == pytest.approx(0.35)

    def test_linear_tv_sigma_0_45(self) -> None:
        assert default_priors_for_category("linear_tv").half_life_sigma == pytest.approx(0.45)

    def test_ctv_sigma_0_45(self) -> None:
        assert default_priors_for_category("ctv").half_life_sigma == pytest.approx(0.45)

    def test_audio_podcast_sigma_0_45(self) -> None:
        assert default_priors_for_category("audio_podcast").half_life_sigma == pytest.approx(0.45)

    def test_paid_social_sigma_0_40(self) -> None:
        assert default_priors_for_category("paid_social").half_life_sigma == pytest.approx(0.40)

    def test_video_sigma_0_40(self) -> None:
        assert default_priors_for_category("video").half_life_sigma == pytest.approx(0.40)

    def test_display_programmatic_sigma_0_40(self) -> None:
        assert default_priors_for_category("display_programmatic").half_life_sigma == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Validation: unknown category raises ValueError
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_category_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown channel category"):
            default_priors_for_category("out_of_home")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            default_priors_for_category("")

    def test_error_message_lists_valid_categories(self) -> None:
        with pytest.raises(ValueError, match="paid_search"):
            default_priors_for_category("typo_channel")

    def test_case_sensitive(self) -> None:
        with pytest.raises(ValueError):
            default_priors_for_category("Paid_Search")


# ---------------------------------------------------------------------------
# Prior range sanity: medians produce credible adstock decay rates
# ---------------------------------------------------------------------------


class TestDecayRateSanity:
    """Converting half_life_mu to decay theta should give plausible values."""

    def _median_decay(self, category: str) -> float:
        half_life = math.exp(default_priors_for_category(category).half_life_mu)
        return 0.5 ** (1.0 / half_life)

    @pytest.mark.parametrize("category", DEFAULT_CHANNEL_CATEGORIES)
    def test_decay_in_zero_one(self, category: str) -> None:
        theta = self._median_decay(category)
        assert 0.0 < theta < 1.0, (
            f"{category}: decay theta {theta} outside (0, 1)"
        )

    def test_paid_search_decay_below_0_5(self) -> None:
        # Half-life = 1 week → theta = 0.5; any shorter is < 0.5.
        # paid_search median = 1.0 week → theta = exactly 0.5.
        assert self._median_decay("paid_search") == pytest.approx(0.5, rel=1e-6)

    def test_linear_tv_decay_above_0_85(self) -> None:
        # Half-life = 6 weeks → theta = 0.5^(1/6) ≈ 0.891.
        assert self._median_decay("linear_tv") > 0.85
