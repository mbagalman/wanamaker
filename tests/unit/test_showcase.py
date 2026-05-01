"""Tests for the HTML showcase renderer (`build_showcase_context` /
`render_showcase`).

We assert the structural promises that make the showcase a self-contained
"show-to-CMO" artifact: a single HTML file with inlined CSS, inlined SVG
charts, no external resource references, no Jinja leakage, and the
expected sections present (or absent) given the optional inputs.

The chart helpers are exercised separately in ``test_charts``; here we
only assert that the SVG appears in the rendered document and that the
right section turns on/off for optional inputs (scenario forecast,
refresh diff).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.forecast.posterior_predictive import (
    ExtrapolationFlag,
    ForecastResult,
)
from wanamaker.refresh.classify import MovementClass
from wanamaker.refresh.diff import ParameterMovement, RefreshDiff
from wanamaker.reports import build_showcase_context, render_showcase
from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus


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


def _saturation_params(channel_name: str) -> list[ParameterSummary]:
    """Realistic Hill saturation parameters for one channel."""
    return [
        ParameterSummary(
            name=f"channel.{channel_name}.half_life",
            mean=1.0,
            sd=0.2,
            hdi_low=0.7,
            hdi_high=1.4,
        ),
        ParameterSummary(
            name=f"channel.{channel_name}.ec50",
            mean=3000.0,
            sd=400.0,
            hdi_low=2200.0,
            hdi_high=3800.0,
        ),
        ParameterSummary(
            name=f"channel.{channel_name}.slope",
            mean=1.5,
            sd=0.2,
            hdi_low=1.1,
            hdi_high=1.9,
        ),
        ParameterSummary(
            name=f"channel.{channel_name}.coefficient",
            mean=12000.0,
            sd=1500.0,
            hdi_low=9000.0,
            hdi_high=15000.0,
        ),
    ]


def _summary(
    *channels: ChannelContributionSummary,
    periods: list[str] | None = None,
    parameters: list[ParameterSummary] | None = None,
) -> PosteriorSummary:
    if periods is None:
        periods = ["2024-01-01", "2024-01-08", "2024-01-15"]
    return PosteriorSummary(
        parameters=list(parameters) if parameters else [],
        channel_contributions=list(channels),
        convergence=ConvergenceSummary(
            max_r_hat=1.005,
            min_ess_bulk=500.0,
            n_divergences=0,
            n_chains=4,
            n_draws=1000,
        ),
        in_sample_predictive=PredictiveSummary(
            periods=periods,
            mean=[100.0] * len(periods),
            hdi_low=[90.0] * len(periods),
            hdi_high=[110.0] * len(periods),
        ),
    )


def _card(*dims: tuple[str, TrustStatus, str]) -> TrustCard:
    return TrustCard(
        dimensions=[
            TrustDimension(name=name, status=status, explanation=expl)
            for (name, status, expl) in dims
        ]
    )


def _render(
    summary: PosteriorSummary,
    card: TrustCard,
    **overrides,
) -> str:
    defaults = {
        "title": "Test Showcase",
        "run_id": "abc123de_20260101T000000Z",
        "generated_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        "runtime_mode": "quick",
        "package_version": "0.1.0",
        "data_hash": "deadbeefcafef00d1234",
        "run_fingerprint": "abc123de4567890f",
        "engine_label": "pymc 5.10.0",
    }
    defaults.update(overrides)
    context = build_showcase_context(summary, card, **defaults)
    return render_showcase(context)


# ---------------------------------------------------------------------------
# Structural promises
# ---------------------------------------------------------------------------


def test_renders_to_self_contained_html() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "<style>" in html and "</style>" in html
    # No external CSS, fonts, or scripts.
    assert "<link" not in html
    assert "<script" not in html
    assert "https://" not in html
    assert "http://" not in html or "http://www.w3.org/2000/svg" in html
    # Jinja leakage check — neither delimiter pair should survive rendering.
    assert "{{" not in html
    assert "{%" not in html


def test_includes_all_required_sections() -> None:
    summary = _summary(_channel("paid_search"), _channel("paid_social"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    for section_id in (
        "verdict",
        "executive-summary",
        "contribution-waterfall",
        "channel-contributions",
        "channel-roi",
        "response-curves",
        "trust-card",
    ):
        assert f'id="{section_id}"' in html, f"missing #{section_id}"


def test_omits_scenario_section_when_no_forecast_supplied() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert 'id="scenario"' not in html


def test_omits_refresh_section_when_no_refresh_diff() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert 'id="refresh-diff"' not in html


def test_charts_inline_as_svg() -> None:
    summary = _summary(_channel("paid_search"), _channel("paid_social"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    # Four SVGs minimum (waterfall + contribution + ROI + response curves).
    # Scenario chart only when --scenario is supplied; tested separately.
    svg_count = html.count("<svg")
    assert svg_count >= 4, f"expected >= 4 inline SVGs, got {svg_count}"
    assert "wmk-chart--waterfall" in html
    assert "wmk-chart--contributions" in html
    assert "wmk-chart--roi" in html
    assert "wmk-chart--response-curves" in html


# ---------------------------------------------------------------------------
# Verdict pill
# ---------------------------------------------------------------------------


def test_verdict_pill_pass_when_all_dimensions_pass() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.PASS, "OK."),
    )
    html = _render(summary, card)

    assert "wmk-pill--pass" in html
    assert "Trust Card clean" in html


def test_verdict_pill_weak_when_any_dimension_weak() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        (
            "saturation_identifiability",
            TrustStatus.WEAK,
            "Insufficient spend variation.",
        ),
    )
    html = _render(summary, card)

    assert "wmk-pill--weak" in html
    assert "Use with caution" in html


def test_verdict_pill_moderate_when_only_moderates() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.MODERATE, "Mixed."),
    )
    html = _render(summary, card)

    assert "wmk-pill--moderate" in html
    assert "Mixed evidence" in html


# ---------------------------------------------------------------------------
# Contribution waterfall
# ---------------------------------------------------------------------------


def test_waterfall_baseline_derived_from_predictive_minus_media() -> None:
    """Baseline = sum(in_sample_predictive.mean) − sum(channel contributions).

    Here the predictive mean is 100 per period × 3 periods = 300; channel
    contribution is 5,000. Negative gap is clamped to 0 in the helper, so
    the waterfall renders media-only.
    """
    summary = _summary(_channel("paid_search", contribution=5000.0))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert 'id="contribution-waterfall"' in html
    assert "wmk-chart--waterfall" in html
    # Predicted total (300) < media (5000), so baseline clamps to 0 and the
    # waterfall shows only the media segment.
    assert ">Baseline<" not in html
    # The total label reflects the visible total (just media in this case).
    assert "Total: 5,000" in html


def test_waterfall_includes_baseline_when_predictive_exceeds_media() -> None:
    """When predictive > media, the baseline segment shows up."""
    summary = _summary(
        _channel("paid_search", contribution=200.0),  # tiny media
        periods=["2024-01-01", "2024-01-08", "2024-01-15"],
    )
    # Predictive mean is 100 per period × 3 = 300; media = 200; baseline = 100.
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert ">Baseline<" in html
    assert "Total: 300" in html


# ---------------------------------------------------------------------------
# Response curves
# ---------------------------------------------------------------------------


def test_response_curves_rendered_when_saturation_params_present() -> None:
    summary = _summary(
        _channel("paid_search"),
        parameters=_saturation_params("paid_search"),
    )
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert 'id="response-curves"' in html
    assert "wmk-chart--response-curves" in html
    # Polyline = the saturation curve. Substring check is enough; the
    # chart helper's own tests cover the shape.
    assert "<polyline" in html
    assert "Saturation curve not identifiable" not in html


def test_response_curves_section_falls_back_when_params_missing() -> None:
    """No parameters -> the section still renders, but each panel shows
    the placeholder text instead of a curve."""
    summary = _summary(_channel("paid_search"), parameters=None)
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert 'id="response-curves"' in html
    assert "Saturation curve not identifiable" in html


# ---------------------------------------------------------------------------
# Spend-invariant rendering
# ---------------------------------------------------------------------------


def test_spend_invariant_channel_marked_in_table_and_chart() -> None:
    summary = _summary(
        _channel("paid_search"),
        _channel("affiliate", spend_invariant=True),
    )
    card = _card(
        (
            "saturation_identifiability",
            TrustStatus.WEAK,
            "Affiliate spend is invariant.",
        )
    )
    html = _render(summary, card)

    # Hatch pattern is defined in the contribution chart for invariant fills.
    assert "wmk-hatch-invariant" in html
    # ROI chart shows the explanatory text in place of a dot/HDI line.
    assert "spend invariant" in html.lower()


# ---------------------------------------------------------------------------
# Optional sections
# ---------------------------------------------------------------------------


def test_scenario_section_appears_when_forecast_supplied() -> None:
    summary = _summary(
        _channel("paid_search", spend_min=100.0, spend_max=500.0)
    )
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    forecast = ForecastResult(
        periods=["2026-01-05", "2026-01-12", "2026-01-19"],
        mean=[1000.0, 1100.0, 1050.0],
        hdi_low=[900.0, 980.0, 940.0],
        hdi_high=[1100.0, 1220.0, 1160.0],
        extrapolation_flags=[
            ExtrapolationFlag(
                period="2026-01-05",
                channel="paid_search",
                planned_spend=900.0,
                observed_spend_min=100.0,
                observed_spend_max=500.0,
                direction="above",
            )
        ],
        spend_invariant_channels=[],
    )
    html = _render(
        summary,
        card,
        scenario_forecast=forecast,
        scenario_plan_name="q1_plan",
    )

    assert 'id="scenario"' in html
    assert "wmk-chart--scenario" in html
    assert "q1_plan" in html
    assert "Extrapolation warning" in html
    assert "paid_search" in html


def test_refresh_section_appears_when_diff_supplied() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    refresh_diff = RefreshDiff(
        previous_run_id="prev_run_id_abc",
        current_run_id="current_run_id_xyz",
        movements=[
            ParameterMovement(
                name="channel.paid_search.roi",
                previous_mean=2.0,
                current_mean=2.05,
                previous_ci=(1.8, 2.2),
                current_ci=(1.85, 2.25),
                movement_class=MovementClass.WITHIN_PRIOR_CI,
            ),
        ],
    )
    html = _render(summary, card, refresh_diff=refresh_diff)

    assert 'id="refresh-diff"' in html
    assert "prev_run_id_abc" in html


# ---------------------------------------------------------------------------
# Decision notes only on weak dimensions
# ---------------------------------------------------------------------------


def test_decision_note_present_for_weak_dimension_only() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        (
            "prior_sensitivity",
            TrustStatus.WEAK,
            "Posterior moves with priors.",
        ),
    )
    html = _render(summary, card)

    assert "Posterior moves with priors." in html
    # The decision-note styling is applied at least once (for prior_sensitivity).
    assert "wmk-trust-card__decision" in html

    # Slice the convergence card by its name landmarks: from
    # ">convergence<" to the next card opener. The PASS card must not
    # contain a decision note.
    convergence_idx = html.find(">convergence<")
    assert convergence_idx >= 0
    next_card_idx = html.find(
        'class="wmk-trust-card"', convergence_idx + 1
    )
    assert next_card_idx > convergence_idx
    convergence_card = html[convergence_idx:next_card_idx]
    assert "wmk-trust-card__decision" not in convergence_card


# ---------------------------------------------------------------------------
# Title and short ID handling
# ---------------------------------------------------------------------------


def test_title_appears_in_h1_and_html_title() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card, title="Q4 2026 plan")

    assert "<title>Q4 2026 plan</title>" in html
    assert "<h1>Q4 2026 plan</h1>" in html


def test_data_hash_truncated_to_16_chars_in_footer() -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    full_hash = "0123456789abcdef0123456789abcdef"
    html = _render(summary, card, data_hash=full_hash)

    # The full hash should not appear in the footer; only its first 16 chars.
    assert full_hash not in html
    assert "0123456789abcdef" in html


# ---------------------------------------------------------------------------
# Empty-state safety
# ---------------------------------------------------------------------------


def test_renders_without_channels() -> None:
    summary = PosteriorSummary(
        channel_contributions=[],
        convergence=None,
        in_sample_predictive=PredictiveSummary(
            periods=["2024-01-01"],
            mean=[0.0],
            hdi_low=[0.0],
            hdi_high=[0.0],
        ),
    )
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card)

    assert "No channel contributions were available" in html
    # The empty-chart fallback should appear in place of bars/dots.
    assert "wmk-chart--empty" in html


def test_renders_without_trust_dimensions() -> None:
    summary = _summary(_channel("paid_search"))
    card = TrustCard(dimensions=[])
    html = _render(summary, card)

    assert 'id="trust-card"' in html
    # Verdict pill falls back to "No verdict".
    assert "No verdict" in html or "No Trust Card dimensions" in html


# ---------------------------------------------------------------------------
# HTML escaping safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("malicious", ["<script>x()</script>", "</style>", "&"])
def test_title_is_html_escaped(malicious: str) -> None:
    summary = _summary(_channel("paid_search"))
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(summary, card, title=malicious)

    # Raw script tags should not survive into the output.
    assert "<script>x()</script>" not in html
