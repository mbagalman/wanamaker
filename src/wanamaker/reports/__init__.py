"""Report rendering.

All user-facing report text — executive summary, trust card, refresh diff
narrative — is generated from deterministic Jinja2 templates driven by
posterior statistics.

Per AGENTS.md Hard Rule 2: **no LLM calls for output generation**, ever.
If a template feels limiting, expand the template logic. Do not reach for
an LLM.

Templates ship inside this subpackage so the package is self-contained.
"""

from wanamaker.reports.render import render_executive_summary, render_trust_card

__all__ = ["render_executive_summary", "render_trust_card"]
