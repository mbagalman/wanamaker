"""Adstock and saturation transforms.

These are the load-bearing statistical primitives in the model — per AGENTS.md
Hard Rule 4, bugs here produce silently wrong ROI numbers and destroy the
trust thesis. Every function in this subpackage must:

- Cite the canonical mathematical form in its docstring
- Have unit tests with worked examples (input → known output)
- Be reviewed by a human before merge
- Cross-check default parameter values against the PRD
"""
