"""Jinja2 rendering for the executive summary and trust card.

The renderer is intentionally thin. All language adjustment based on
confidence level (FR-5.3 — "high-confidence channels get definitive
statements; low-confidence channels get hedged language") lives inside
the template files via Jinja2 conditionals, not in Python code.

Templates are loaded from ``wanamaker.reports.templates`` via the package
loader so they ship with installed wheels.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

_env = Environment(
    loader=PackageLoader("wanamaker.reports", "templates"),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render(template_name: str, context: dict[str, Any]) -> str:
    return _env.get_template(template_name).render(**context)


def render_executive_summary(context: dict[str, Any]) -> str:
    """Render the plain-English executive summary (FR-5.3).

    The template adjusts language based on confidence level. No LLM
    involvement; the templates are code.
    """
    return _render("executive_summary.md.j2", context)


def render_trust_card(context: dict[str, Any]) -> str:
    """Render the model Trust Card (FR-5.4)."""
    return _render("trust_card.md.j2", context)
