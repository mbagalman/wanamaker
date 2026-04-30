"""Build an engine-ready ``ModelSpec`` from a ``WanamakerConfig``.

This module is the bridge between the user-facing YAML configuration and the
engine-agnostic ``ModelSpec``.  It is the *only* place that should translate
config vocabulary into model vocabulary — keeping the transformation in one
place makes it easy to test and to extend as new ModelSpec fields land.

Separation of concerns:
- ``WanamakerConfig`` owns the user-facing schema (pydantic, YAML-derived).
- ``ModelSpec`` owns the engine contract (frozen dataclasses, no I/O).
- ``build_model_spec`` converts one to the other, applying taxonomy defaults
  and any per-channel overrides from the config.
"""

from __future__ import annotations

import math

from wanamaker.config import WanamakerConfig
from wanamaker.data.io import load_lift_test_csv
from wanamaker.data.taxonomy import DEFAULT_CHANNEL_CATEGORIES
from wanamaker.model.priors import default_priors_for_category
from wanamaker.model.spec import ChannelSpec, LiftPrior, ModelSpec

_NORMAL_95_Z = 1.959963984540054


def build_model_spec(cfg: WanamakerConfig) -> ModelSpec:
    """Convert a validated ``WanamakerConfig`` into an engine-ready ``ModelSpec``.

    Resolution order for channel priors:
    1. If the config names a channel that is one of the 10 default taxonomy
       categories and no explicit override is given, the taxonomy default
       from ``default_priors_for_category`` is used.
    2. Per-channel prior overrides (future: FR-1.2 YAML overrides) will be
       layered on top once issue #7 per-channel prior config lands.

    Refresh anchoring (``anchor_priors``) is intentionally *not* applied
    here — that is the responsibility of the refresh path (issue #22).
    ``build_model_spec`` always produces an unanchored spec suitable for
    a fresh first fit.

    Args:
        cfg: Fully validated configuration produced by ``load_config``.

    Returns:
        A frozen ``ModelSpec`` ready to pass to ``Engine.fit``.

    Raises:
        ValueError: If a channel in ``cfg.channels`` has a category that is
            not in ``DEFAULT_CHANNEL_CATEGORIES``.
    """
    unknown = [
        ch.category for ch in cfg.channels
        if ch.category not in DEFAULT_CHANNEL_CATEGORIES
    ]
    if unknown:
        raise ValueError(
            f"Unknown channel categories: {unknown}. "
            f"Valid categories: {sorted(DEFAULT_CHANNEL_CATEGORIES)}"
        )

    channel_specs = [
        ChannelSpec(
            name=ch.name,
            category=ch.category,
            adstock_family=ch.adstock_family,
        )
        for ch in cfg.channels
    ]

    # Resolve priors for all channels from taxonomy defaults.
    # Future: per-channel YAML overrides will populate channel_priors selectively.
    channel_priors = {
        ch.name: default_priors_for_category(ch.category)
        for ch in cfg.channels
    }
    lift_test_priors = _load_lift_test_priors(cfg)

    return ModelSpec(
        channels=channel_specs,
        target_column=cfg.data.target_column,
        date_column=cfg.data.date_column,
        control_columns=list(cfg.data.control_columns),
        channel_priors=channel_priors,
        lift_test_priors=lift_test_priors,
        runtime_mode=cfg.run.runtime_mode,
    )


def _load_lift_test_priors(cfg: WanamakerConfig) -> dict[str, LiftPrior]:
    if cfg.data.lift_test_csv is None:
        return {}

    channel_names = {channel.name for channel in cfg.channels}
    lift_tests = load_lift_test_csv(cfg.data.lift_test_csv)
    unknown = sorted(set(lift_tests["channel"]) - channel_names)
    if unknown:
        raise ValueError(
            "lift-test CSV contains channels that are not configured: "
            f"{unknown}. Add them to channels or remove the lift-test rows."
        )

    priors: dict[str, LiftPrior] = {}
    for row in lift_tests.itertuples(index=False):
        interval_width = float(row.ci_upper) - float(row.ci_lower)
        sd_roi = interval_width / (2.0 * _NORMAL_95_Z)
        if not math.isfinite(sd_roi) or sd_roi <= 0.0:
            raise ValueError(
                f"lift-test CSV produced a non-positive prior sd for channel {row.channel!r}"
            )
        priors[str(row.channel)] = LiftPrior(
            mean_roi=float(row.lift_estimate),
            sd_roi=sd_roi,
            confidence=0.95,
        )

    return priors
