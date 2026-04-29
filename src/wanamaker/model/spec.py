"""Engine-agnostic model specification.

A ``ModelSpec`` is a declarative description of the model: which channels,
which transforms, which priors. It is pure data — no engine, no sampling.
Engines consume it via ``Engine.fit(model_spec, ...)``.

Keeping the spec data-only means the same description can be rendered into
PyMC, NumPyro, or Stan code without leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ChannelSpec:
    """Per-channel modeling spec."""

    name: str
    category: str
    adstock_family: Literal["geometric", "weibull"] = "geometric"


@dataclass(frozen=True)
class ModelSpec:
    """Top-level model specification consumed by engines."""

    channels: list[ChannelSpec]
    control_columns: list[str] = field(default_factory=list)
