"""Scenario comparison (FR-5.2).

User supplies 2–3 budget plans; we rank them with uncertainty and flag any
plan that extrapolates beyond the historical observed spend range.
Per FR-3.2, plans involving spend-invariant channels do not get
reallocation recommendations.
"""

from __future__ import annotations


def compare_scenarios(posterior, plans, seed: int):  # noqa: ANN001
    raise NotImplementedError("Phase 1: scenario comparison")
