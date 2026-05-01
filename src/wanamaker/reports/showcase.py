"""HTML "show-to-CMO" showcase rendering.

The showcase exists so a reader can forward a single self-contained file
("show your CMO") and have it render identically in any modern browser,
Outlook preview, or print. No CDN, no JS, no external assets — the
stylesheet is inlined and charts are rendered as inline SVG.

Per AGENTS.md Hard Rule 2: **no LLM calls for output generation**, ever.
The showcase reuses the same context shapers as the Markdown report.

Public surface:

- ``build_showcase_context`` — turn ``PosteriorSummary`` + ``TrustCard``
  (plus optional refresh diff and optional scenario forecast) into the
  flat dict the template consumes.
- ``render_showcase`` — apply the Jinja2 template to a ready-made context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from importlib.resources import files
from typing import Any

from wanamaker.engine.summary import PosteriorSummary
from wanamaker.forecast.posterior_predictive import ForecastResult
from wanamaker.refresh.classify import MovementClass, unexplained_fraction
from wanamaker.refresh.diff import RefreshDiff
from wanamaker.reports._charts import (
    contribution_bars_svg,
    response_curves_svg,
    roi_dotplot_svg,
    scenario_delta_svg,
)
from wanamaker.reports.render import (
    build_executive_summary_context,
    build_trust_card_context,
)
from wanamaker.trust_card.card import TrustCard, TrustStatus

_DECISION_NOTES: dict[str, str] = {
    "convergence": (
        "If sampler diagnostics are weak, treat the posterior as approximate "
        "and avoid fine-grained reallocation calls until the fit is rerun."
    ),
    "holdout_accuracy": (
        "If holdout accuracy is weak, the model is not predicting recent "
        "history well — be cautious about forward forecasts."
    ),
    "refresh_stability": (
        "If refresh stability is weak, recent estimates moved enough that "
        "prior decisions made on the old run may no longer hold."
    ),
    "prior_sensitivity": (
        "If prior sensitivity is weak, the data is not pinning down the "
        "answer — priors are doing more work than the evidence."
    ),
    "saturation_identifiability": (
        "If saturation is weakly identifiable, do not make strong "
        "reallocation calls based on response-curve shape."
    ),
    "lift_test_consistency": (
        "If lift-test consistency is weak, the model disagrees with recent "
        "experiments — reconcile before acting on either."
    ),
}


def _trust_pill(trust_card: TrustCard) -> dict[str, str]:
    """Top-of-page verdict pill summarising the Trust Card.

    Worst dimension wins. Empty Trust Card defaults to "moderate".
    """
    if not trust_card.dimensions:
        return {
            "status": "moderate",
            "label": "No verdict",
            "explanation": "No Trust Card dimensions were computed for this run.",
        }
    statuses = {d.status for d in trust_card.dimensions}
    if TrustStatus.WEAK in statuses:
        weak_names = [
            d.name for d in trust_card.dimensions if d.status == TrustStatus.WEAK
        ]
        return {
            "status": "weak",
            "label": "Use with caution",
            "explanation": (
                "The Trust Card flags weakness on "
                + ", ".join(weak_names)
                + ". Outputs touching those dimensions need careful interpretation."
            ),
        }
    if TrustStatus.MODERATE in statuses:
        return {
            "status": "moderate",
            "label": "Mixed evidence",
            "explanation": (
                "The Trust Card is broadly OK, with moderate evidence on "
                "some dimensions. Read each dimension before strong action."
            ),
        }
    return {
        "status": "pass",
        "label": "Trust Card clean",
        "explanation": (
            "The Trust Card passes on every dimension. The fit is well "
            "identified and outputs can be acted on with normal caution."
        ),
    }


def _refresh_narrative(refresh_diff: RefreshDiff) -> Mapping[str, Any]:
    """Mirror of the Markdown report's refresh narrative."""
    movements = list(refresh_diff.movements)
    fraction = unexplained_fraction(movements)
    classes: dict[str, int] = {}
    for movement in movements:
        if movement.movement_class is None:
            continue
        classes[movement.movement_class.value] = (
            classes.get(movement.movement_class.value, 0) + 1
        )
    unexplained_count = classes.get(MovementClass.UNEXPLAINED.value, 0)
    return {
        "previous_run_id": refresh_diff.previous_run_id,
        "n_movements": len(movements),
        "n_unexplained": unexplained_count,
        "unexplained_fraction": fraction,
        "movements_by_class": classes,
    }


def _scenario_block(
    forecast_result: ForecastResult,
    plan_name: str,
    baseline_total: float | None,
) -> dict[str, Any]:
    """Per-period mean+HDI plus the totals the prose paragraph needs."""
    mean_total = float(sum(forecast_result.mean))
    hdi_low_total = float(sum(forecast_result.hdi_low))
    hdi_high_total = float(sum(forecast_result.hdi_high))
    extrap = sorted({flag.channel for flag in forecast_result.extrapolation_flags})
    return {
        "plan_name": plan_name,
        "mean_total": mean_total,
        "hdi_low_total": hdi_low_total,
        "hdi_high_total": hdi_high_total,
        "baseline_total": baseline_total,
        "delta": (mean_total - baseline_total) if baseline_total is not None else 0.0,
        "extrapolation_warnings": extrap,
        "periods": list(forecast_result.periods),
        "mean": list(forecast_result.mean),
        "hdi_low": list(forecast_result.hdi_low),
        "hdi_high": list(forecast_result.hdi_high),
    }


def build_showcase_context(
    summary: PosteriorSummary,
    trust_card: TrustCard,
    *,
    title: str,
    run_id: str,
    generated_at: datetime,
    runtime_mode: str,
    package_version: str,
    data_hash: str,
    run_fingerprint: str,
    engine_label: str,
    refresh_diff: RefreshDiff | None = None,
    scenario_forecast: ForecastResult | None = None,
    scenario_plan_name: str | None = None,
    baseline_total: float | None = None,
    advisor_recommendations: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Shape every input the showcase template needs.

    Reuses the Markdown report's context shapers for executive-summary
    fields and trust-card per-channel saturation rendering, then adds the
    showcase-only chrome (verdict pill, decision notes, charts, footer
    metadata).
    """
    exec_ctx = build_executive_summary_context(
        summary,
        trust_card,
        refresh_diff=refresh_diff,
        advisor_recommendations=advisor_recommendations,
        trust_card_link="#trust-card",
    )
    trust_ctx = build_trust_card_context(summary, trust_card)

    trust_dimensions = [
        {
            "name": d["name"],
            "status": d["status"],
            "explanation": d["explanation"],
            "decision_note": (
                _DECISION_NOTES.get(d["name"]) if d["status"] == "weak" else None
            ),
        }
        for d in trust_ctx["dimensions"]
    ]

    scenario_block: dict[str, Any] | None = None
    scenario_chart_svg = ""
    if scenario_forecast is not None:
        scenario_block = _scenario_block(
            scenario_forecast,
            plan_name=scenario_plan_name or "scenario",
            baseline_total=baseline_total,
        )
        scenario_chart_svg = scenario_delta_svg(
            scenario_block["periods"],
            scenario_block["mean"],
            scenario_block["hdi_low"],
            scenario_block["hdi_high"],
        )

    refresh_narrative = (
        _refresh_narrative(refresh_diff) if refresh_diff is not None else None
    )

    response_curve_channels = _response_curve_channels(summary, exec_ctx["channels"])
    response_curves_chart_svg = response_curves_svg(response_curve_channels)

    return {
        # Header / footer.
        "title": title,
        "run_id": run_id,
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        "runtime_mode": runtime_mode,
        "package_version": package_version,
        "data_hash": data_hash[:16] if data_hash else "—",
        "run_fingerprint": run_fingerprint or "—",
        "engine_label": engine_label,
        # Period.
        "period_start": exec_ctx["period_start"],
        "period_end": exec_ctx["period_end"],
        "n_periods": exec_ctx["n_periods"],
        # Channels (already ranked + tagged).
        "channels": exec_ctx["channels"],
        "total_media_contribution": exec_ctx["total_media_contribution"],
        # Trust card.
        "trust_card_pill": _trust_pill(trust_card),
        "trust_dimensions": trust_dimensions,
        "has_invariant_channels": trust_ctx["has_invariant_channels"],
        # Charts.
        "contribution_chart_svg": contribution_bars_svg(exec_ctx["channels"]),
        "roi_chart_svg": roi_dotplot_svg(exec_ctx["channels"]),
        "response_curves_chart_svg": response_curves_chart_svg,
        "scenario_chart_svg": scenario_chart_svg,
        # Optional sections.
        "scenario": scenario_block,
        "refresh_narrative": refresh_narrative,
    }


def _response_curve_channels(
    summary: PosteriorSummary,
    channels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join channel summaries with their saturation parameters from ``summary.parameters``.

    The PyMC engine emits parameters with stable names like
    ``channel.<name>.ec50``, ``channel.<name>.slope``, and
    ``channel.<name>.coefficient`` (see ``ParameterSummary.name`` in
    ``engine.summary``). Look those up by channel name; channels missing
    any of the three are passed through and the chart helper falls back
    to the spend-invariant rendering for them.
    """
    by_name: dict[str, float] = {}
    for p in summary.parameters:
        by_name[p.name] = float(p.mean)

    enriched: list[dict[str, Any]] = []
    for ch in channels:
        name = ch["name"]
        enriched.append(
            {
                **ch,
                "ec50": by_name.get(f"channel.{name}.ec50"),
                "slope": by_name.get(f"channel.{name}.slope"),
                "coefficient": by_name.get(f"channel.{name}.coefficient"),
            }
        )
    return enriched


def _read_stylesheet() -> str:
    """Read the bundled ``showcase.css`` so it can be inlined into ``<style>``.

    Reading at render time (not import time) keeps the package import
    lazy and avoids any I/O on bare ``import wanamaker``.
    """
    return (
        files("wanamaker.reports.templates")
        .joinpath("showcase.css")
        .read_text(encoding="utf-8")
    )


def render_showcase(context: dict[str, Any]) -> str:
    """Render the showcase HTML from a ready-made context.

    The stylesheet is read from package resources at call time and
    injected so the output is a single self-contained file. The Jinja2
    environment is the one configured in ``reports.render`` and uses
    ``select_autoescape(["html"])``; SVG payloads are explicitly
    ``| safe`` in the template.
    """
    from wanamaker.reports.render import _env  # local import to avoid a cycle

    template = _env.get_template("showcase.html.j2")
    enriched = {**context, "stylesheet": _read_stylesheet()}
    return template.render(**enriched)
