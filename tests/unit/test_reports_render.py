"""Tests for the executive summary and Trust Card templates (#27, #28).

These tests render against synthetic ``PosteriorSummary`` / ``TrustCard``
inputs covering every condition the templates branch on:

- All-pass / mixed / all-weak Trust Cards
- Spend-invariant channels and the FR-3.2 callout
- Conditional refresh-stability / lift-test sections
- Confidence-driven channel language (high / moderate / weak)
- Refresh-diff narrative

Acceptance from #27 includes user testing against ≥10 testers; that is
out of scope for unit tests, but every conditional and language branch
the testers would experience is exercised here.
"""

from __future__ import annotations

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.refresh.classify import MovementClass
from wanamaker.refresh.diff import ParameterMovement, RefreshDiff
from wanamaker.reports import (
    build_executive_summary_context,
    build_trust_card_context,
    render_executive_summary,
    render_trust_card,
)
from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _channel(
    name: str,
    *,
    contribution: float = 5000.0,
    contribution_hdi: tuple[float, float] = (4500.0, 5500.0),
    roi_mean: float = 2.0,
    roi_hdi: tuple[float, float] = (1.8, 2.2),
    spend_invariant: bool = False,
    spend_min: float = 10.0,
    spend_max: float = 50.0,
) -> ChannelContributionSummary:
    return ChannelContributionSummary(
        channel=name,
        mean_contribution=contribution,
        hdi_low=contribution_hdi[0],
        hdi_high=contribution_hdi[1],
        roi_mean=roi_mean,
        roi_hdi_low=roi_hdi[0],
        roi_hdi_high=roi_hdi[1],
        observed_spend_min=spend_min,
        observed_spend_max=spend_max,
        spend_invariant=spend_invariant,
    )


def _summary(
    *channels: ChannelContributionSummary,
    periods: list[str] | None = None,
) -> PosteriorSummary:
    if periods is None:
        periods = ["2024-01-01", "2024-01-08", "2024-01-15"]
    return PosteriorSummary(
        channel_contributions=list(channels),
        convergence=ConvergenceSummary(
            max_r_hat=1.005, min_ess_bulk=500.0,
            n_divergences=0, n_chains=4, n_draws=1000,
        ),
        in_sample_predictive=PredictiveSummary(
            periods=periods,
            mean=[100.0] * len(periods),
            hdi_low=[90.0] * len(periods),
            hdi_high=[110.0] * len(periods),
        ),
    )


def _card(
    *dims: tuple[str, TrustStatus, str],
) -> TrustCard:
    return TrustCard(
        dimensions=[
            TrustDimension(name=name, status=status, explanation=expl)
            for (name, status, expl) in dims
        ]
    )


# ---------------------------------------------------------------------------
# Per-channel confidence tagging
# ---------------------------------------------------------------------------


class TestChannelConfidence:
    def test_tight_hdi_yields_high_confidence(self) -> None:
        ctx = build_executive_summary_context(
            _summary(_channel("paid_search", roi_mean=2.0, roi_hdi=(1.9, 2.1))),
            _card(("convergence", TrustStatus.PASS, "ok")),
        )
        assert ctx["channels"][0]["confidence"] == "high"

    def test_wide_hdi_yields_moderate_or_weak(self) -> None:
        ctx = build_executive_summary_context(
            _summary(_channel("paid_search", roi_mean=2.0, roi_hdi=(0.6, 3.4))),
            _card(("convergence", TrustStatus.PASS, "ok")),
        )
        assert ctx["channels"][0]["confidence"] in ("moderate", "weak")

    def test_spend_invariant_channel_is_weak(self) -> None:
        ctx = build_executive_summary_context(
            _summary(
                _channel(
                    "tv", roi_mean=1.0, roi_hdi=(0.95, 1.05), spend_invariant=True,
                ),
            ),
            _card(("convergence", TrustStatus.PASS, "ok")),
        )
        assert ctx["channels"][0]["confidence"] == "weak"


# ---------------------------------------------------------------------------
# Executive summary template — branches across trust card states
# ---------------------------------------------------------------------------


class TestExecutiveSummaryTemplate:
    def test_all_pass_card_renders_definitive_headline(self) -> None:
        summary = _summary(
            _channel("paid_search", contribution=5000.0, roi_hdi=(1.9, 2.1)),
            _channel("tv", contribution=2000.0, roi_hdi=(0.45, 0.55)),
        )
        card = _card(
            ("convergence", TrustStatus.PASS, "Sampler converged."),
            ("holdout_accuracy", TrustStatus.PASS, "WAPE 5%."),
        )
        out = render_executive_summary(
            build_executive_summary_context(summary, card)
        )
        assert "# Executive summary" in out
        assert "## Headline finding" in out
        assert "## Channel contributions" in out
        assert "## ROI summary" in out
        assert "## Recommended actions" in out
        # No weak callout when nothing is weak.
        assert "Trust Card flags weakness" not in out
        # The clean-trust action fallback fires.
        assert "No specific actions" in out

    def test_mixed_card_hedges_headline_and_lists_uncertainties(self) -> None:
        summary = _summary(
            _channel("paid_search", roi_mean=2.0, roi_hdi=(0.6, 3.4)),  # moderate
            _channel("tv", roi_hdi=(0.45, 0.55)),
        )
        card = _card(
            ("convergence", TrustStatus.PASS, "Sampler converged."),
            ("holdout_accuracy", TrustStatus.MODERATE, "WAPE is 15%."),
            ("prior_sensitivity", TrustStatus.WEAK, "Posterior moves 30%."),
        )
        out = render_executive_summary(
            build_executive_summary_context(summary, card)
        )
        assert "Trust Card flags weakness" in out
        assert "prior_sensitivity" in out
        # Both moderate and weak dimensions appear in Key uncertainties.
        assert "**holdout_accuracy (moderate):**" in out
        assert "**prior_sensitivity (weak):**" in out

    def test_spend_invariant_channel_listed_with_fr32_callout(self) -> None:
        summary = _summary(
            _channel("paid_search", contribution=5000.0, roi_hdi=(1.9, 2.1)),
            _channel("tv", contribution=2000.0, spend_invariant=True),
        )
        card = _card(
            ("saturation_identifiability", TrustStatus.WEAK, "tv weak."),
        )
        out = render_executive_summary(
            build_executive_summary_context(summary, card)
        )
        assert "Saturation cannot be estimated from observed data" in out
        assert "`tv`" in out
        # FR-3.2 reallocation exclusion is called out explicitly.
        assert "excluded from reallocation guidance" in out

    def test_weak_channels_get_experiment_recommendation_fallback(self) -> None:
        summary = _summary(
            _channel("paid_search", roi_mean=2.0, roi_hdi=(0.6, 3.4)),  # weak
        )
        card = _card(("convergence", TrustStatus.PASS, "ok"))
        out = render_executive_summary(
            build_executive_summary_context(summary, card)
        )
        assert "Consider an experiment for **`paid_search`**" in out

    def test_advisor_recommendations_replace_fallback(self) -> None:
        summary = _summary(
            _channel("paid_search", roi_mean=2.0, roi_hdi=(0.6, 3.4)),
        )
        card = _card(("convergence", TrustStatus.PASS, "ok"))
        out = render_executive_summary(
            build_executive_summary_context(
                summary, card,
                advisor_recommendations=[
                    "Run a geo holdout for `paid_search` next quarter.",
                ],
            )
        )
        assert "Run a geo holdout for `paid_search`" in out
        # When advisor supplied bullets, the auto-generated fallback is suppressed.
        assert "Consider an experiment for **`paid_search`**" not in out

    def test_refresh_narrative_appears_when_diff_supplied(self) -> None:
        summary = _summary(_channel("paid_search"))
        card = _card(("convergence", TrustStatus.PASS, "ok"))
        diff = RefreshDiff(
            previous_run_id="abcd1234_20250101T000000Z",
            current_run_id="efgh5678_20260101T000000Z",
            movements=[
                ParameterMovement(
                    name="channel.paid_search.coefficient",
                    previous_mean=2.0, current_mean=2.05,
                    previous_ci=(1.8, 2.2), current_ci=(1.85, 2.25),
                    movement_class=MovementClass.WITHIN_PRIOR_CI,
                ),
                ParameterMovement(
                    name="channel.tv.coefficient",
                    previous_mean=1.0, current_mean=5.0,
                    previous_ci=(0.9, 1.1), current_ci=(4.9, 5.1),
                    movement_class=MovementClass.UNEXPLAINED,
                ),
            ],
        )
        out = render_executive_summary(
            build_executive_summary_context(summary, card, refresh_diff=diff)
        )
        assert "## Refresh notes" in out
        assert "abcd1234_20250101T000000Z" in out
        assert "1 were classified as **unexplained**" in out

    def test_period_range_uses_period_labels_when_supplied(self) -> None:
        summary = _summary(_channel("paid_search"))
        card = _card(("convergence", TrustStatus.PASS, "ok"))
        out = render_executive_summary(
            build_executive_summary_context(
                summary, card,
                period_labels=["2026-01-05", "2026-01-12", "2026-01-19"],
            )
        )
        assert "2026-01-05 – 2026-01-19" in out


# ---------------------------------------------------------------------------
# Trust Card template — per-dimension sections + conditional behaviour
# ---------------------------------------------------------------------------


class TestTrustCardTemplate:
    def test_renders_one_section_per_dimension_with_status(self) -> None:
        summary = _summary(_channel("paid_search"))
        card = _card(
            ("convergence", TrustStatus.PASS, "Sampler converged."),
            ("holdout_accuracy", TrustStatus.MODERATE, "WAPE 15%."),
            ("prior_sensitivity", TrustStatus.WEAK, "Shift 30%."),
        )
        out = render_trust_card(build_trust_card_context(summary, card))
        assert "# Model Trust Card" in out
        assert "### convergence — pass" in out
        assert "### holdout_accuracy — moderate" in out
        assert "### prior_sensitivity — weak" in out

    def test_per_channel_saturation_status_listed(self) -> None:
        summary = _summary(
            _channel("paid_search", spend_min=10.0, spend_max=50.0),
            _channel("tv", spend_invariant=True, spend_min=100.0, spend_max=100.0),
        )
        card = _card(
            ("saturation_identifiability", TrustStatus.WEAK, "tv weak."),
        )
        out = render_trust_card(build_trust_card_context(summary, card))
        assert "Per-channel status" in out
        assert "`paid_search`" in out and "identifiable" in out
        assert "`tv`" in out and "spend invariant" in out
        # FR-5.4 acceptance: spend-invariant note appears.
        assert "Saturation cannot be estimated from observed data" in out

    def test_no_saturation_block_when_dimension_absent(self) -> None:
        summary = _summary(_channel("paid_search"))
        card = _card(
            ("convergence", TrustStatus.PASS, "ok"),
            ("holdout_accuracy", TrustStatus.PASS, "ok"),
        )
        out = render_trust_card(build_trust_card_context(summary, card))
        assert "Per-channel status" not in out

    def test_refresh_stability_only_present_when_dimension_included(self) -> None:
        summary = _summary(_channel("paid_search"))
        without = render_trust_card(
            build_trust_card_context(
                summary,
                _card(("convergence", TrustStatus.PASS, "ok")),
            )
        )
        assert "refresh_stability" not in without

        with_dim = render_trust_card(
            build_trust_card_context(
                summary,
                _card(
                    ("convergence", TrustStatus.PASS, "ok"),
                    ("refresh_stability", TrustStatus.PASS, "All within prior CI."),
                ),
            )
        )
        assert "refresh_stability" in with_dim

    def test_lift_test_consistency_only_present_when_dimension_included(self) -> None:
        summary = _summary(_channel("paid_search"))
        with_dim = render_trust_card(
            build_trust_card_context(
                summary,
                _card(
                    ("convergence", TrustStatus.PASS, "ok"),
                    ("lift_test_consistency", TrustStatus.PASS, "Consistent."),
                ),
            )
        )
        assert "lift_test_consistency" in with_dim

    def test_all_weak_card_renders_without_error(self) -> None:
        summary = _summary(
            _channel("paid_search", spend_invariant=True),
            _channel("tv", spend_invariant=True),
        )
        card = _card(
            ("convergence", TrustStatus.WEAK, "Divergences."),
            ("holdout_accuracy", TrustStatus.WEAK, "WAPE 50%."),
            ("prior_sensitivity", TrustStatus.WEAK, "Shift 80%."),
            ("saturation_identifiability", TrustStatus.WEAK, "All weak."),
        )
        out = render_trust_card(build_trust_card_context(summary, card))
        # All four dimensions present.
        for dim in (
            "convergence",
            "holdout_accuracy",
            "prior_sensitivity",
            "saturation_identifiability",
        ):
            assert f"### {dim} — weak" in out
        # Saturation block lists both invariant channels.
        assert "spend invariant" in out

    def test_empty_dimensions_renders_explanatory_message(self) -> None:
        summary = _summary(_channel("paid_search"))
        card = TrustCard(dimensions=[])
        out = render_trust_card(build_trust_card_context(summary, card))
        assert "No Trust Card dimensions were computed" in out
