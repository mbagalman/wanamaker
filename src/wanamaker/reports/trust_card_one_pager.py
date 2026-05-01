"""Trust Card one-pager rendering — executive-facing standalone artifact.

The one-pager is the "if you read only one page of this analysis" output.
It is designed for forwarding to a non-technical executive audience
(CMO, CFO, CEO) and for printing to a single physical page.

Two design rules differentiate it from the full showcase:

1. **No jargon.** Every dimension is translated to consequence language
   via ``_trust_card_translations``. The unit test asserts that none of
   ``credible interval``, ``HDI``, ``R-hat``, ``MCMC``, ``Bayesian``,
   ``posterior``, ``ESS``, or ``Gelman-Rubin`` appear in the rendered
   HTML.
2. **One page.** Print CSS uses ``@page size: auto; margin: 0.5in;`` so
   it renders on US Letter or A4 without manual sizing.

Per AGENTS.md Hard Rule 2: **no LLM calls for output generation**, ever.
The verdict sentence is selected from a fixed three-line set; per-dimension
language comes from the translation dictionary.
"""

from __future__ import annotations

from datetime import datetime
from importlib.resources import files
from typing import Any

from wanamaker.engine.summary import PosteriorSummary
from wanamaker.reports._trust_card_translations import (
    consequence_for,
    decision_for,
    label_for,
)
from wanamaker.trust_card.card import TrustCard, TrustStatus

# Verdict line text — one per status. Fixed phrasing wins over varied
# phrasing because executives read these scanning for one signal: green
# / amber / red. Predictable language is the feature.
_VERDICT_TEXT: dict[str, dict[str, str]] = {
    "pass": {
        "label": "Trust Card clean",
        "sentence": (
            "Every check on the model passed. The analysis can be acted "
            "on with normal caution."
        ),
    },
    "moderate": {
        "label": "Mixed evidence",
        "sentence": (
            "The model is broadly trustworthy, with caveats on a few "
            "checks. Read the items below before strong action."
        ),
    },
    "weak": {
        "label": "Use with caution",
        "sentence": (
            "One or more checks on the model flagged a weakness. Read "
            "the items below before acting on the analysis."
        ),
    },
    "empty": {
        "label": "No verdict",
        "sentence": (
            "No Trust Card dimensions were computed for this run. The "
            "full report has the available diagnostics."
        ),
    },
}


def _verdict_for(trust_card: TrustCard) -> dict[str, str]:
    """Return the pill + sentence for the worst dimension's status."""
    if not trust_card.dimensions:
        v = _VERDICT_TEXT["empty"]
        return {"status": "moderate", "label": v["label"], "sentence": v["sentence"]}
    statuses = {d.status for d in trust_card.dimensions}
    if TrustStatus.WEAK in statuses:
        key = "weak"
    elif TrustStatus.MODERATE in statuses:
        key = "moderate"
    else:
        key = "pass"
    v = _VERDICT_TEXT[key]
    return {"status": key, "label": v["label"], "sentence": v["sentence"]}


def build_trust_card_one_pager_context(
    summary: PosteriorSummary,
    trust_card: TrustCard,
    *,
    title: str,
    run_id: str,
    generated_at: datetime,
    package_version: str,
    run_fingerprint: str,
) -> dict[str, Any]:
    """Shape every input the Trust Card one-pager template needs.

    The shaper applies the plain-English translation dictionary to each
    dimension, picks the verdict line based on the worst status, and
    builds the per-weak-dimension decision bullets. No technical strings
    survive into the output.
    """
    dimensions = []
    decisions: list[dict[str, str]] = []
    for d in trust_card.dimensions:
        dimensions.append(
            {
                "name_label": label_for(d.name),
                "status": d.status.value,
                "sentence": consequence_for(d.name, d.status),
            }
        )
        if d.status == TrustStatus.WEAK:
            decisions.append(
                {
                    "dimension_label": label_for(d.name),
                    "sentence": decision_for(d.name),
                }
            )

    period_start, period_end, n_periods = _period_range(summary)

    return {
        "title": title,
        "run_id": run_id,
        "generated_at": generated_at.strftime("%Y-%m-%d"),
        "package_version": package_version,
        "run_fingerprint": run_fingerprint or "—",
        "period_start": period_start,
        "period_end": period_end,
        "n_periods": n_periods,
        "verdict": _verdict_for(trust_card),
        "dimensions": dimensions,
        "decisions": decisions,
    }


def _period_range(summary: PosteriorSummary) -> tuple[str, str, int]:
    """Pull the date range from in-sample predictive periods, when present."""
    predictive = summary.in_sample_predictive
    if predictive and predictive.periods:
        return (
            str(predictive.periods[0]),
            str(predictive.periods[-1]),
            len(predictive.periods),
        )
    return ("unknown", "unknown", 0)


def _read_stylesheet() -> str:
    """Read ``trust_card_one_pager.css`` from the package for inlining."""
    return (
        files("wanamaker.reports.templates")
        .joinpath("trust_card_one_pager.css")
        .read_text(encoding="utf-8")
    )


def render_trust_card_one_pager(context: dict[str, Any]) -> str:
    """Render the one-pager HTML from a ready-made context.

    The stylesheet is read from package resources at call time and
    injected so the output is a single self-contained file. Uses the
    Jinja2 environment configured in ``reports.render``, which
    autoescapes ``.html.j2`` templates.
    """
    from wanamaker.reports.render import _env  # local import to avoid a cycle

    template = _env.get_template("trust_card_one_pager.html.j2")
    enriched = {**context, "stylesheet": _read_stylesheet()}
    return template.render(**enriched)
