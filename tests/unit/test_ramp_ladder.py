"""Tests for the executive-facing ramp ladder visualization.

Covers issue #87's acceptance criteria:

- Ramp Markdown includes a per-fraction ladder with pass/fail reasons.
- Selected ramp fraction is easy to identify.
- All four statuses render: ``proceed``, ``stage``, ``test_first``,
  ``do_not_recommend``.
- At least one failure due to extrapolation, one due to downside risk,
  one due to Trust Card gating.

Tests build ``RampRecommendation`` objects directly and feed them
through ``build_ramp_recommendation_context`` + the Markdown template.
No engine is invoked — these tests stay at the rendering layer.
"""

from __future__ import annotations

from wanamaker.forecast.posterior_predictive import ExtrapolationFlag
from wanamaker.forecast.ramp import RampCandidate, RampRecommendation
from wanamaker.reports import (
    build_ramp_recommendation_context,
    render_ramp_recommendation,
)


def _candidate(
    fraction: float,
    *,
    expected_increment: float = 1000.0,
    p_positive: float = 0.92,
    p_material_loss: float = 0.05,
    q05_increment: float = -200.0,
    cvar_5: float = -300.0,
    largest_move_share: float = 0.10,
    fractional_kelly: float = 0.50,
    extrapolation_flags: list[ExtrapolationFlag] | None = None,
    failed_gates: list[str] | None = None,
) -> RampCandidate:
    failed = failed_gates or []
    return RampCandidate(
        fraction=fraction,
        total_spend_by_channel={"paid_search": 10000.0},
        expected_increment=expected_increment,
        probability_positive=p_positive,
        probability_material_loss=p_material_loss,
        q05_increment=q05_increment,
        cvar_5=cvar_5,
        largest_move_share=largest_move_share,
        fractional_kelly=fractional_kelly,
        extrapolation_flags=extrapolation_flags or [],
        passes=not failed,
        failed_gates=failed,
    )


def _recommendation(
    candidates: list[RampCandidate],
    *,
    recommended_fraction: float,
    status: str = "stage",
    explanation: str = "Stage 25%.",
    blocking_reason: str | None = None,
) -> RampRecommendation:
    return RampRecommendation(
        baseline_plan_name="base",
        target_plan_name="alt",
        recommended_fraction=recommended_fraction,
        status=status,  # type: ignore[arg-type]
        candidates=candidates,
        explanation=explanation,
        blocking_reason=blocking_reason,
    )


def _render(rec: RampRecommendation) -> str:
    ctx = build_ramp_recommendation_context(
        rec,
        run_id="abc12345",
        baseline_path="base.csv",
        target_path="alt.csv",
    )
    return render_ramp_recommendation(ctx)


# ---------------------------------------------------------------------------
# Context shaping
# ---------------------------------------------------------------------------


def test_context_marks_recommended_fraction_as_selected() -> None:
    cands = [
        _candidate(0.10),
        _candidate(0.25),
        _candidate(0.50, failed_gates=["p_material_loss"]),
    ]
    rec = _recommendation(cands, recommended_fraction=0.25, status="stage")
    ctx = build_ramp_recommendation_context(
        rec, run_id="x", baseline_path="b.csv", target_path="t.csv",
    )

    selected = [c for c in ctx["candidates"] if c["is_selected"]]
    assert len(selected) == 1
    assert selected[0]["fraction"] == 0.25


def test_historical_support_buckets() -> None:
    """`extrapolation` in failed_gates → severe; flags but no fail → mild;
    no flags → in range."""
    cands = [
        _candidate(0.10),  # in range
        _candidate(
            0.25,
            extrapolation_flags=[
                ExtrapolationFlag(
                    period="2026-01-05",
                    channel="paid_search",
                    planned_spend=900.0,
                    observed_spend_min=100.0,
                    observed_spend_max=500.0,
                    direction="above",
                ),
            ],
        ),  # mild
        _candidate(
            0.50,
            extrapolation_flags=[
                ExtrapolationFlag(
                    period=f"2026-01-{i:02d}",
                    channel="paid_search",
                    planned_spend=900.0,
                    observed_spend_min=100.0,
                    observed_spend_max=500.0,
                    direction="above",
                )
                for i in range(5, 12)
            ],
            failed_gates=["extrapolation"],
        ),  # severe
    ]
    rec = _recommendation(cands, recommended_fraction=0.10, status="stage")
    ctx = build_ramp_recommendation_context(
        rec, run_id="x", baseline_path="b.csv", target_path="t.csv",
    )
    labels = [c["historical_support_label"] for c in ctx["candidates"]]
    assert labels == ["in range", "mild", "severe"]


def test_failure_reasons_translate_each_gate_into_plain_english() -> None:
    cand = _candidate(
        1.0,
        p_positive=0.40,
        p_material_loss=0.30,
        cvar_5=-2000.0,
        fractional_kelly=0.25,
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
        failed_gates=[
            "p_positive",
            "p_material_loss",
            "cvar_5",
            "trust_card",
            "fractional_kelly",
            "extrapolation",
        ],
    )
    rec = _recommendation([cand], recommended_fraction=0.0, status="do_not_recommend")
    ctx = build_ramp_recommendation_context(
        rec, run_id="x", baseline_path="b.csv", target_path="t.csv",
    )
    reasons = ctx["candidates"][0]["failure_reasons"]
    text = " | ".join(reasons)

    # Every failed gate produced a translation.
    assert len(reasons) == 6
    assert "low probability of beating baseline" in text
    assert "downside risk too high" in text
    assert "worst-case tail loss too large" in text
    assert "Trust Card caps the move" in text
    assert "exceeds the risk-adjusted sizing cap" in text
    assert "exceeds historical spend range" in text
    # Concrete numbers attached where they aid the reader.
    assert "30% chance of a material loss" in text
    assert "P(beats baseline) = 40%" in text
    assert "1 cell(s) outside training range" in text
    assert "sizing cap = 25%" in text


def test_no_failure_reasons_for_passing_candidates() -> None:
    cand = _candidate(0.25)
    rec = _recommendation([cand], recommended_fraction=0.25, status="stage")
    ctx = build_ramp_recommendation_context(
        rec, run_id="x", baseline_path="b.csv", target_path="t.csv",
    )
    assert ctx["candidates"][0]["failure_reasons"] == []


def test_signed_int_formatting_for_expected_lift() -> None:
    """Positive lift gets +; negative gets ASCII -; zero is just 0."""
    cands = [
        _candidate(0.10, expected_increment=1234.6),
        _candidate(0.25, expected_increment=-1500.4),
        _candidate(0.50, expected_increment=0.0),
    ]
    rec = _recommendation(cands, recommended_fraction=0.10, status="stage")
    ctx = build_ramp_recommendation_context(
        rec, run_id="x", baseline_path="b.csv", target_path="t.csv",
    )
    labels = [c["expected_lift_label"] for c in ctx["candidates"]]
    assert labels == ["+1,235", "-1,500", "0"]


# ---------------------------------------------------------------------------
# Template output — covers the issue's acceptance criteria
# ---------------------------------------------------------------------------


def test_decision_ladder_header_matches_spec() -> None:
    rec = _recommendation([_candidate(0.10)], recommended_fraction=0.10, status="stage")
    body = _render(rec)
    assert (
        "| Ramp | Expected lift | Downside risk | Historical support | "
        "Trust Card gate | Verdict |"
    ) in body


def test_selected_fraction_is_visually_marked() -> None:
    cands = [
        _candidate(0.10),
        _candidate(0.25),
        _candidate(0.50, failed_gates=["p_material_loss"]),
    ]
    rec = _recommendation(cands, recommended_fraction=0.25, status="stage")
    body = _render(rec)

    # Only the 25% row shows the (selected) marker.
    selected_lines = [line for line in body.splitlines() if "(selected)" in line]
    assert len(selected_lines) == 1
    assert "25%" in selected_lines[0]


def test_why_each_fractions_verdict_section_explains_failures() -> None:
    cands = [
        _candidate(0.10),
        _candidate(
            0.25,
            failed_gates=["extrapolation"],
            extrapolation_flags=[
                ExtrapolationFlag(
                    period=f"2026-01-{i:02d}",
                    channel="paid_search",
                    planned_spend=900.0,
                    observed_spend_min=100.0,
                    observed_spend_max=500.0,
                    direction="above",
                )
                for i in (5, 12, 19)
            ],
        ),
    ]
    rec = _recommendation(cands, recommended_fraction=0.10, status="stage")
    body = _render(rec)

    assert "Why each fraction's verdict" in body
    # Plain-English explanation appears for the failing fraction.
    assert "exceeds historical spend range" in body
    assert "(3 cell(s) outside training range)" in body


def test_no_kelly_language_in_executive_ladder() -> None:
    """`Kelly` only appears in the analyst-detail section, not in the lead
    ladder. Per the issue: don't use Kelly language in executive output."""
    cand = _candidate(1.0, fractional_kelly=0.25, failed_gates=["fractional_kelly"])
    rec = _recommendation([cand], recommended_fraction=0.0, status="do_not_recommend")
    body = _render(rec)

    ladder_start = body.index("## Decision ladder")
    detail_start = body.index("## Sizing detail (for analysts)")
    ladder_section = body[ladder_start:detail_start]
    assert "Kelly" not in ladder_section
    assert "kelly" not in ladder_section
    # The analyst-detail section may legitimately use the word.
    assert "Kelly" in body[detail_start:]


def test_analyst_detail_table_preserves_raw_metrics() -> None:
    cand = _candidate(0.25, p_positive=0.85, q05_increment=-100.0, cvar_5=-150.0)
    rec = _recommendation([cand], recommended_fraction=0.25, status="stage")
    body = _render(rec)

    detail_start = body.index("## Sizing detail (for analysts)")
    detail = body[detail_start:]
    assert "P(improve)" in detail
    assert "CVaR 5" in detail
    assert "q05 increment" in detail


# ---------------------------------------------------------------------------
# All four statuses render
# ---------------------------------------------------------------------------


def test_proceed_status_renders() -> None:
    cand = _candidate(1.0)
    rec = _recommendation(
        [cand],
        recommended_fraction=1.0,
        status="proceed",
        explanation="Full move is supported.",
    )
    body = _render(rec)
    assert "(`proceed`)" in body
    assert "Recommended ramp: **100%**" in body


def test_stage_status_renders_and_marks_partial_fraction() -> None:
    cands = [
        _candidate(0.10),
        _candidate(0.25),
        _candidate(0.50, failed_gates=["p_material_loss"]),
        _candidate(1.0, failed_gates=["p_material_loss", "cvar_5"]),
    ]
    rec = _recommendation(cands, recommended_fraction=0.25, status="stage")
    body = _render(rec)
    assert "(`stage`)" in body
    assert "Recommended ramp: **25%**" in body
    # All four candidates appear in the ladder.
    for label in ("10%", "25%", "50%", "100%"):
        assert label in body


def test_test_first_status_renders_with_evidence_gate_failures() -> None:
    """`test_first` is the route when only evidence gates (extrapolation,
    trust_card, fractional_kelly) failed at higher fractions."""
    cands = [
        _candidate(
            0.10, failed_gates=["extrapolation"],
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
        ),
        _candidate(0.25, failed_gates=["trust_card"]),
    ]
    rec = _recommendation(
        cands, recommended_fraction=0.0, status="test_first",
        explanation="Run an experiment first.",
    )
    body = _render(rec)
    assert "(`test_first`)" in body
    # The Trust Card gate column shows fail for the 25% row.
    twentyfive_line = next(
        line for line in body.splitlines()
        if line.startswith("| 25%") and "Verdict" not in line
    )
    assert "fail" in twentyfive_line


def test_do_not_recommend_status_renders_with_blocking_reason() -> None:
    rec = _recommendation(
        [],
        recommended_fraction=0.0,
        status="do_not_recommend",
        explanation="Spend-invariant channel reallocation is unsafe.",
        blocking_reason="spend_invariant_reallocation",
    )
    body = _render(rec)
    assert "(`do_not_recommend`)" in body
    assert "Blocking reason: `spend_invariant_reallocation`" in body
    # No candidate ladder when blocked up-front.
    assert "blocked\nbefore candidate scoring" in body


# ---------------------------------------------------------------------------
# Specific failure-driver coverage (issue acceptance)
# ---------------------------------------------------------------------------


def test_extrapolation_driven_failure_appears_in_why_section() -> None:
    cand = _candidate(
        0.50,
        failed_gates=["extrapolation"],
        extrapolation_flags=[
            ExtrapolationFlag(
                period="2026-01-05",
                channel="ctv",
                planned_spend=20000.0,
                observed_spend_min=8000.0,
                observed_spend_max=15000.0,
                direction="above",
            )
        ],
    )
    rec = _recommendation([cand], recommended_fraction=0.0, status="test_first")
    body = _render(rec)

    why_start = body.index("Why each fraction's verdict")
    why_section = body[why_start:body.index("## How to use this", why_start)]
    assert "exceeds historical spend range" in why_section


def test_downside_risk_driven_failure_appears_in_why_section() -> None:
    cand = _candidate(
        0.50,
        p_material_loss=0.30,
        failed_gates=["p_material_loss"],
    )
    rec = _recommendation([cand], recommended_fraction=0.0, status="do_not_recommend")
    body = _render(rec)

    why_start = body.index("Why each fraction's verdict")
    why_section = body[why_start:body.index("## How to use this", why_start)]
    assert "downside risk too high" in why_section
    assert "30% chance of a material loss" in why_section


def test_trust_card_driven_failure_appears_in_why_section() -> None:
    cand = _candidate(
        0.50,
        failed_gates=["trust_card"],
    )
    rec = _recommendation([cand], recommended_fraction=0.10, status="stage")
    body = _render(rec)

    why_start = body.index("Why each fraction's verdict")
    why_section = body[why_start:body.index("## How to use this", why_start)]
    assert "Trust Card caps the move" in why_section
