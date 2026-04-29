"""Light posterior anchoring as a mixture prior (FR-4.4).

For each channel-level parameter θ:

    Prior_new(θ) = (1 - w) · Prior_default(θ) + w · Posterior_previous(θ)

with ``w ∈ [0, 1]``. Anchoring is applied to *marginals only* — joint
posterior structure (parameter correlations) is regenerated from new data.

Default ``w`` is 0.3 (placeholder pending Phase 1 empirical tuning against
the refresh-stability benchmark and NFR-5 unexplained-movement target).

The named presets (FR-4.4 table) exposed via ``--anchor-strength``:

    none → 0.0    light → 0.2    medium → 0.3 (default)    heavy → 0.5

Anchoring applies to channel-level parameters only:
- channel ROI / coefficient
- adstock half-life
- saturation slope
- saturation EC50

Control coefficients, baseline parameters, and global hyperparameters
re-estimate freely.
"""

from __future__ import annotations

from typing import Final

ANCHOR_PRESETS: Final[dict[str, float]] = {
    "none": 0.0,
    "light": 0.2,
    "medium": 0.3,
    "heavy": 0.5,
}
DEFAULT_ANCHOR_STRENGTH: Final[str] = "medium"


def resolve_anchor_weight(value: str | float) -> float:
    """Translate a CLI/YAML ``anchor_strength`` value to a numeric weight.

    Args:
        value: Either a preset name (``"none"``, ``"light"``, ``"medium"``,
            ``"heavy"``) or a float in ``[0, 1]``.

    Returns:
        The numeric anchoring weight ``w``.

    Raises:
        ValueError: If ``value`` is an unknown preset or a float outside
            ``[0, 1]``.
    """
    if isinstance(value, str):
        try:
            return ANCHOR_PRESETS[value]
        except KeyError as exc:
            valid = ", ".join(ANCHOR_PRESETS)
            raise ValueError(
                f"unknown anchor strength preset {value!r}; expected one of: {valid}"
            ) from exc
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"anchor weight must be in [0, 1]; got {value}")
    return float(value)
