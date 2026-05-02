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
2. Every Python module under ``USER_FACING_COPY_DIRS`` whose strings
   flow into a rendered artifact — currently ``src/wanamaker/reports/``
   plus ``src/wanamaker/trust_card/`` (Trust-Card dimension explanations
   render directly into the executive summary, showcase, and one-pager).
   New modules that emit literal copy into user-facing output should be
   added to ``USER_FACING_COPY_DIRS``.

**Why ``src/wanamaker/forecast/`` is NOT in ``USER_FACING_COPY_DIRS``.**
Forecast modules (``forecast/scenarios.py``, ``forecast/ramp.py``) also
emit user-facing copy — scenario interpretation sentences and ramp
verdict explanations both flow into rendered Markdown — but their
*module docstrings* legitimately quote the banned phrases when
explaining what Wanamaker is *not* (e.g. ``not a continuous "optimized
budget."``). A whole-file scan would produce false-positive failures on
those defensive mentions, so forecast copy is gated **per-instance**
instead:

- ``tests/unit/test_compare_scenarios.py::TestNoBannedTerminology``
  asserts every interpretation sentence under multiple scenarios.
- ``tests/unit/test_ramp.py::TestNoBannedTerminology`` asserts the ramp
  explanation under every verdict (proceed / stage / test_first /
  do_not_recommend, with and without a weak Trust Card, and the
  spend-invariant short-circuit).

When you add a new forecast user-facing string emitter, add a
matching per-instance test in the relevant ``test_<module>.py`` file —
do *not* add the directory to ``USER_FACING_COPY_DIRS`` here.

Contributor-facing or design docs (``AGENTS.md``, ``docs/internal/architecture.md``,
``docs/internal/risk_adjusted_allocation.md``) are out of scope — those
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

# Directories whose Python modules produce literal strings that get
# rendered into user-facing artifacts. Add a directory here when its
# output is consumed by reports / showcase / trust-card surfaces.
USER_FACING_COPY_DIRS: tuple[Path, ...] = (
    REPORTS_DIR,
    REPO_ROOT / "src" / "wanamaker" / "trust_card",
)


def _matches_in(text: str) -> list[str]:
    """Return banned phrases that appear in ``text`` (case-insensitive)."""
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def _template_files() -> list[Path]:
    return sorted(TEMPLATES_DIR.glob("*.j2"))


def _user_facing_python_files() -> list[Path]:
    """Python files across every ``USER_FACING_COPY_DIRS`` directory.

    Excludes ``__init__.py`` (re-exports only, no literal copy) but keeps
    every other module — context shapers, verdict text, decision notes,
    chart helpers, Excel exporter, Trust-Card dimension explanations —
    since any of them could leak banned phrasing into rendered output.
    """
    files: list[Path] = []
    for directory in USER_FACING_COPY_DIRS:
        files.extend(p for p in directory.glob("*.py") if p.name != "__init__.py")
    return sorted(files)


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.name)
def test_no_banned_phrases_in_jinja_templates(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    matches = _matches_in(text)
    assert not matches, (
        f"{path.relative_to(REPO_ROOT)} contains banned phrase(s): {matches}. "
        "See AGENTS.md → 'Product terminology' for the rule and "
        "approved alternatives."
    )


@pytest.mark.parametrize(
    "path", _user_facing_python_files(), ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_no_banned_phrases_in_user_facing_python_modules(path: Path) -> None:
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


def test_guardrail_covers_each_user_facing_directory() -> None:
    """Sanity: every directory in ``USER_FACING_COPY_DIRS`` contributed at
    least one file to the scan. Catches the silent-failure mode where
    a future restructure leaves a directory empty without anyone
    noticing the guardrail effectively dropped it.
    """
    discovered = _user_facing_python_files()
    assert discovered, "no Python helpers discovered across USER_FACING_COPY_DIRS"
    parents = {p.parent for p in discovered}
    for directory in USER_FACING_COPY_DIRS:
        assert directory in parents, (
            f"directory {directory.relative_to(REPO_ROOT)} is in "
            "USER_FACING_COPY_DIRS but contributed zero files to the scan"
        )


def test_guardrail_at_least_one_python_helper_was_scanned() -> None:
    """Sanity: glob picked up the Python helpers that emit copy.

    Same silent-failure guard as above for the Python side, anchored on
    a few specific files from each scanned directory so a layout change
    that moves them surfaces explicitly.
    """
    helpers = _user_facing_python_files()
    assert helpers, "no Python helpers discovered under USER_FACING_COPY_DIRS"
    names = {p.name for p in helpers}
    # At minimum the showcase shaper, the one-pager shaper, the render
    # module, and the Trust-Card compute module must all be in scope.
    # Update deliberately if the layout moves.
    expected_minimum = {
        "showcase.py",
        "trust_card_one_pager.py",
        "render.py",
        "compute.py",  # src/wanamaker/trust_card/compute.py
    }
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
