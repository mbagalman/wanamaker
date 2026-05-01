"""Tests for the executive-facing Trust Card one-pager renderer.

The one-pager is forwarded to non-technical executives, so the assertions
focus on three contracts:

1. **Self-contained.** Single HTML file, inline CSS, no external assets,
   no Jinja leakage.
2. **No jargon.** None of the analyst-facing terms in
   ``JARGON_BLACKLIST`` appear in the rendered output. This is the
   load-bearing test — if it fails, the one-pager is no longer
   executive-friendly even if everything else looks fine.
3. **Verdict matches the worst dimension.** The big pill at the top is
   chosen from the worst status across all dimensions; mixed cards do
   not lull readers with a green pill.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.reports import (
    build_trust_card_one_pager_context,
    render_trust_card_one_pager,
)
from wanamaker.reports._trust_card_translations import JARGON_BLACKLIST
from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus


def _channel(name: str) -> ChannelContributionSummary:
    return ChannelContributionSummary(
        channel=name,
        mean_contribution=5000.0,
        hdi_low=4500.0,
        hdi_high=5500.0,
        roi_mean=2.0,
        roi_hdi_low=1.8,
        roi_hdi_high=2.2,
        observed_spend_min=10.0,
        observed_spend_max=50.0,
        spend_invariant=False,
    )


def _summary(*channels: ChannelContributionSummary) -> PosteriorSummary:
    periods = ["2024-01-01", "2024-01-08", "2024-01-15"]
    return PosteriorSummary(
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


def _render(card: TrustCard, **overrides) -> str:
    summary = _summary(_channel("paid_search"))
    defaults = {
        "title": "Test One-Pager",
        "run_id": "abc123de_20260101T000000Z",
        "generated_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        "package_version": "0.1.0",
        "run_fingerprint": "abc123de4567890f",
    }
    defaults.update(overrides)
    context = build_trust_card_one_pager_context(summary, card, **defaults)
    return render_trust_card_one_pager(context)


# ---------------------------------------------------------------------------
# Self-contained
# ---------------------------------------------------------------------------


def test_renders_to_self_contained_html() -> None:
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(card)

    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "<style>" in html and "</style>" in html
    assert "<link" not in html
    assert "<script" not in html
    assert "https://" not in html
    assert "{{" not in html
    assert "{%" not in html


def test_includes_required_sections() -> None:
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(card)

    for section_id in ("dimensions", "decisions", "not-covered"):
        assert f'id="{section_id}"' in html, f"missing #{section_id}"


def test_size_under_threshold() -> None:
    """The one-pager should stay tiny for email forwarding."""
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.PASS, "OK."),
        ("refresh_stability", TrustStatus.PASS, "OK."),
        ("prior_sensitivity", TrustStatus.PASS, "OK."),
        ("saturation_identifiability", TrustStatus.PASS, "OK."),
        ("lift_test_consistency", TrustStatus.PASS, "OK."),
    )
    html = _render(card)
    assert len(html) < 25_000, f"one-pager grew to {len(html)} bytes"


# ---------------------------------------------------------------------------
# Plain-English guard — the load-bearing test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "card",
    [
        _card(("convergence", TrustStatus.PASS, "OK.")),
        _card(("convergence", TrustStatus.MODERATE, "OK.")),
        _card(("convergence", TrustStatus.WEAK, "OK.")),
        _card(
            ("convergence", TrustStatus.PASS, "Max R-hat is 1.005 across all parameters."),
            ("holdout_accuracy", TrustStatus.MODERATE, "Holdout MAPE is 12%."),
            (
                "saturation_identifiability",
                TrustStatus.WEAK,
                "Affiliate spend is invariant; saturation curve from priors only.",
            ),
            (
                "prior_sensitivity",
                TrustStatus.PASS,
                "Posterior is robust to ±50% prior perturbations.",
            ),
            (
                "lift_test_consistency",
                TrustStatus.PASS,
                "Posterior agrees with provided lift tests.",
            ),
            ("refresh_stability", TrustStatus.PASS, "Movements explained."),
        ),
    ],
)
def test_no_jargon_appears_in_rendered_output(card: TrustCard) -> None:
    """The whole point of the one-pager: no analyst jargon survives.

    Even when the underlying ``TrustDimension.explanation`` contains
    technical terms, the rendered one-pager uses the translated strings
    from ``_trust_card_translations`` instead.

    Acronyms (HDI, ESS, MCMC) must be matched case-sensitively with
    word boundaries so they don't false-positive against substrings like
    ``weakness`` or ``Express``.
    """
    import re

    html = _render(card)
    for term in JARGON_BLACKLIST:
        # Word-boundary, case-sensitive match. Catches "ESS" but not "ess"
        # inside "weakness"; catches "Bayesian" but not arbitrary substrings.
        pattern = r"\b" + re.escape(term) + r"\b"
        match = re.search(pattern, html)
        assert match is None, (
            f"jargon term {term!r} leaked into the one-pager output: "
            f"...{html[max(0, match.start() - 20):match.end() + 20]}..."
        )


# ---------------------------------------------------------------------------
# Verdict matches the worst dimension
# ---------------------------------------------------------------------------


def test_verdict_pass_when_all_pass() -> None:
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.PASS, "OK."),
    )
    html = _render(card)
    assert "wmk-verdict--pass" in html
    assert "wmk-pill--pass" in html
    assert "Trust Card clean" in html


def test_verdict_moderate_when_only_moderates() -> None:
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.MODERATE, "Mixed."),
    )
    html = _render(card)
    assert "wmk-verdict--moderate" in html
    assert "Mixed evidence" in html


def test_verdict_weak_when_any_weak() -> None:
    """The worst dimension wins — a single weak still tints the verdict red."""
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.PASS, "OK."),
        ("saturation_identifiability", TrustStatus.WEAK, "Insufficient spend variation."),
    )
    html = _render(card)
    assert "wmk-verdict--weak" in html
    assert "Use with caution" in html


def test_verdict_handles_empty_card() -> None:
    card = TrustCard(dimensions=[])
    html = _render(card)
    assert "No verdict" in html or "No Trust Card dimensions" in html


# ---------------------------------------------------------------------------
# Decisions block
# ---------------------------------------------------------------------------


def test_decisions_lists_one_bullet_per_weak_dimension() -> None:
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("saturation_identifiability", TrustStatus.WEAK, "Insufficient spend variation."),
        ("prior_sensitivity", TrustStatus.WEAK, "Sensitive to priors."),
    )
    html = _render(card)
    # Two weak dimensions -> two <li> entries inside the decisions section.
    decisions_start = html.index('id="decisions"')
    decisions_end = html.index('id="not-covered"')
    decisions_block = html[decisions_start:decisions_end]
    assert decisions_block.count("<li>") == 2


def test_decisions_default_when_all_pass() -> None:
    card = _card(
        ("convergence", TrustStatus.PASS, "OK."),
        ("holdout_accuracy", TrustStatus.PASS, "OK."),
    )
    html = _render(card)
    assert "acted on with normal caution" in html


# ---------------------------------------------------------------------------
# Per-dimension cell uses translated label and translated text
# ---------------------------------------------------------------------------


def test_dimensions_use_friendly_labels() -> None:
    card = _card(
        ("saturation_identifiability", TrustStatus.WEAK, "Insufficient spend variation."),
    )
    html = _render(card)
    # Friendly label appears; raw underscored name does not.
    assert "Diminishing-returns curves" in html
    assert "saturation_identifiability" not in html


def test_dimensions_use_translated_text_not_raw_explanation() -> None:
    """The technical explanation should not survive into the one-pager —
    only the translated consequence sentence."""
    card = _card(
        (
            "convergence",
            TrustStatus.WEAK,
            "Max R-hat is 1.27, well above the 1.01 threshold.",
        ),
    )
    html = _render(card)
    assert "Max R-hat" not in html
    assert "did not settle cleanly" in html


# ---------------------------------------------------------------------------
# Title and escape safety
# ---------------------------------------------------------------------------


def test_title_appears_in_h1_and_html_title() -> None:
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(card, title="Q4 2026 Trust Card")
    assert "<title>Q4 2026 Trust Card</title>" in html
    assert "<h1>Q4 2026 Trust Card</h1>" in html


@pytest.mark.parametrize("malicious", ["<script>x()</script>", "</style>", "&"])
def test_title_is_html_escaped(malicious: str) -> None:
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(card, title=malicious)
    assert "<script>x()</script>" not in html


# ---------------------------------------------------------------------------
# Print contract
# ---------------------------------------------------------------------------


def test_print_css_present() -> None:
    """The @page rule is what makes the one-pager print to one sheet."""
    card = _card(("convergence", TrustStatus.PASS, "OK."))
    html = _render(card)
    assert "@page" in html
    assert "@media print" in html
