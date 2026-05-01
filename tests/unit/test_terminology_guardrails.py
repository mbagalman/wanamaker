"""Issue #90: Product terminology guardrails.

Wanamaker is a decision-support tool, not a constrained-optimization
budget engine. User-facing output must use cautious decision language
("candidate scenarios", "risk-adjusted ramp", "largest defensible
move") and avoid optimizer-grade promises ("optimized budget", "optimal
allocation", "best budget", "guaranteed lift", "maximize ROI" as a
promise).

The full rule lives in ``AGENTS.md`` under "Product terminology". This
test enforces it on:

1. Every Jinja2 template under ``src/wanamaker/reports/templates/``.
2. Every Python helper in ``src/wanamaker/reports/`` that injects
   literal copy into a rendered artifact (decision notes, verdict
   text, gate explanations, etc.).

Contributor-facing or design docs (``AGENTS.md``, ``docs/architecture.md``,
``docs/risk_adjusted_allocation.md``) are out of scope — those
legitimately discuss the banned phrases when explaining what Wanamaker
does *not* do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Mirrors the "Avoid" list in AGENTS.md → "Product terminology". Keep in
# sync; this is the load-bearing definition for the test, AGENTS.md is
# the load-bearing definition for humans.
BANNED_PHRASES: tuple[str, ...] = (
    "optimized budget",
    "optimal allocation",
    "best budget",
    "guaranteed lift",
    "maximize roi",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "src" / "wanamaker" / "reports"
TEMPLATES_DIR = REPORTS_DIR / "templates"


def _matches_in(text: str) -> list[str]:
    """Return banned phrases that appear in ``text`` (case-insensitive)."""
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def _template_files() -> list[Path]:
    return sorted(TEMPLATES_DIR.glob("*.j2"))


def _reports_python_files() -> list[Path]:
    """Python files in ``src/wanamaker/reports/`` that emit user-facing copy.

    Excludes ``__init__.py`` (re-exports only, no literal copy) but keeps
    every other module — context shapers, verdict text, decision notes,
    chart helpers, Excel exporter — since any of them could leak banned
    phrasing into rendered output.
    """
    return sorted(p for p in REPORTS_DIR.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.name)
def test_no_banned_phrases_in_jinja_templates(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    matches = _matches_in(text)
    assert not matches, (
        f"{path.relative_to(REPO_ROOT)} contains banned phrase(s): {matches}. "
        "See AGENTS.md → 'Product terminology' for the rule and "
        "approved alternatives."
    )


@pytest.mark.parametrize("path", _reports_python_files(), ids=lambda p: p.name)
def test_no_banned_phrases_in_reports_python_modules(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    matches = _matches_in(text)
    assert not matches, (
        f"{path.relative_to(REPO_ROOT)} contains banned phrase(s): {matches}. "
        "See AGENTS.md → 'Product terminology' for the rule and "
        "approved alternatives."
    )


def test_guardrail_at_least_one_template_was_scanned() -> None:
    """Sanity: glob picked up the templates that exist today.

    Catches the silent-failure mode where a future repo restructure
    moves the templates and the guardrail test passes by scanning
    nothing.
    """
    templates = _template_files()
    assert templates, "no .j2 templates were discovered under TEMPLATES_DIR"
    names = {p.name for p in templates}
    # The five templates we ship today must all be in scope. If one is
    # renamed or removed, update this set deliberately.
    expected = {
        "executive_summary.md.j2",
        "trust_card.md.j2",
        "ramp_recommendation.md.j2",
        "showcase.html.j2",
        "trust_card_one_pager.html.j2",
    }
    missing = expected - names
    assert not missing, f"expected templates not found: {missing}"


def test_guardrail_at_least_one_python_helper_was_scanned() -> None:
    """Sanity: glob picked up the Python helpers that emit copy.

    Same silent-failure guard as above for the Python side.
    """
    helpers = _reports_python_files()
    assert helpers, "no Python helpers discovered under REPORTS_DIR"
    names = {p.name for p in helpers}
    # At minimum the showcase shaper, the one-pager shaper, and the
    # render module must be in scope. Update deliberately if the layout
    # moves.
    expected_minimum = {"showcase.py", "trust_card_one_pager.py", "render.py"}
    missing = expected_minimum - names
    assert not missing, f"expected helpers not found: {missing}"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("This plan is the optimized budget.", ["optimized budget"]),
        ("We compute the OPTIMAL ALLOCATION at scale.", ["optimal allocation"]),
        ("Pick the best budget.", ["best budget"]),
        ("Guaranteed lift on every channel.", ["guaranteed lift"]),
        ("We Maximize ROI subject to constraints.", ["maximize roi"]),
        # Multiple in the same blob.
        (
            "The optimized budget gives a guaranteed lift.",
            ["optimized budget", "guaranteed lift"],
        ),
        # Defensive mention with the negation does still match — the
        # guardrail can't tell context from text. That is *why* the
        # scan is scoped to template + helper files only; design docs
        # that legitimately discuss the banned terms are out of scope.
        ('Wanamaker does not produce an "optimized budget."', ["optimized budget"]),
        # No matches.
        ("Risk-adjusted ramp recommendation for paid_search.", []),
        ("", []),
    ],
)
def test_matcher_catches_known_violations(
    text: str, expected: list[str],
) -> None:
    assert _matches_in(text) == expected
