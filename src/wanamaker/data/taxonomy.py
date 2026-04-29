"""Channel taxonomy and category-driven default priors (FR-1.2).

The default taxonomy covers the standard categories that show up in
mid-market media plans. Each category carries default prior shapes for
adstock half-life and saturation parameters that reflect typical behavior
for that media type.

Per FR-1.2 acceptance: the empirical or theoretical basis for each chosen
range must be documented. Those citations land alongside the actual prior
values in Phase 0; for now this is a name list.
"""

from __future__ import annotations

from typing import Final

DEFAULT_CHANNEL_CATEGORIES: Final[tuple[str, ...]] = (
    "paid_search",
    "paid_social",
    "video",
    "linear_tv",
    "ctv",
    "audio_podcast",
    "display_programmatic",
    "affiliate",
    "email_crm",
    "promotions_discounting",
)
