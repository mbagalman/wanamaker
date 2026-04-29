"""Default priors per channel category (FR-1.2, FR-3.1).

Each channel category in the default taxonomy has prior shapes for adstock
half-life and saturation parameters that reflect typical behavior for that
media type.

Per FR-1.2 acceptance, the empirical or theoretical basis for each chosen
range must be documented inline alongside the values when they land.
"""

from __future__ import annotations

raise_msg = "Phase 0: default priors per channel category — needs PRD-cited values"


def default_priors_for_category(category: str) -> dict[str, object]:
    raise NotImplementedError(raise_msg)
