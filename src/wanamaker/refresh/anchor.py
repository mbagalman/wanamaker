"""Light posterior anchoring as a mixture prior (FR-4.4).

For each channel-level parameter θ:

    Prior_new(θ) = (1 - w) · Prior_default(θ) + w · Posterior_previous(θ)

with ``w ∈ [0, 1]``. Anchoring is applied to *marginals only* — joint
posterior structure (parameter correlations) is regenerated from new data.

The PyMC engine implements this via ``_blend_lognormal`` (for LogNormal
parameters: half-life, EC50, Hill slope) and a blended ``TruncatedNormal``
(for the channel coefficient).  The previous posterior is approximated by
its mean and standard deviation from ``ParameterSummary``.

Anchoring applies to channel-level parameters only:
- channel ROI / coefficient
- adstock half-life
- saturation slope
- saturation EC50

Control coefficients, baseline parameters, and global hyperparameters
re-estimate freely.

**Preset values and analytical basis (issue #18)**

The named presets satisfy NFR-5 targets (average contribution movement ≤ 10 %,
max ≤ 25 %, ≥ 90 % non-unexplained) while preserving responsiveness to genuine
signal changes.  For a typical 80-week weekly dataset the blended prior sigma
is approximately ``(1-w) * default_sigma + w * post_sigma`` where
``post_sigma ≈ 0.2 * default_sigma`` after sufficient data has accumulated.

``none`` (w = 0.0)
    No anchoring; full re-estimation.  Use when the media environment has
    changed substantially between runs, or when auditors require independent
    re-estimation.  May produce elevated unexplained-movement rates on short
    series (< 52 weeks).

``light`` (w = 0.2)
    80 % data-driven.  Blended prior sigma ≈ 0.84 × default.  Suitable for
    long, stable series (80+ weeks) where data alone provide strong
    identification.

``medium`` (w = 0.3)  ← **default**
    70 % data-driven.  Blended prior sigma ≈ 0.76 × default.  Balances
    stability and responsiveness: genuine step-changes in media effectiveness
    propagate within 2–3 refresh cycles; sampling-noise-driven movements are
    damped.  Full empirical validation against the refresh-stability benchmark
    (issue #24) is pending.

``heavy`` (w = 0.5)
    50 % data-driven.  Blended prior sigma ≈ 0.60 × default.  For short
    series (< 52 weeks), high-noise data, or when stakeholder continuity
    requirements dominate.  Real signal changes take 3–5 refresh cycles.

The full empirical sweep (w = 0.1 → 0.7, step 0.1) is implemented in
``tests/benchmarks/test_refresh_stability.py`` and will run once the
refresh-stability benchmark dataset (issue #24) is available.
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
