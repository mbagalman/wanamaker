"""Refresh accountability — Wanamaker's headline differentiator (FR-4).

Three pieces:

- ``anchor``  — light posterior anchoring as a mixture prior (FR-4.4)
- ``diff``    — refresh diff report against a previous run's posterior (FR-4.2)
- ``classify``— movement classification (FR-4.3): within-CI, improved-holdout,
                unexplained, user-induced, weakly-identified

Acceptance target (NFR-5): on the refresh stability benchmark, >=90% of
historical contribution estimates classified as "within prior credible
interval" with default anchoring on.
"""
