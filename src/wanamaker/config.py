"""YAML configuration loading and validation.

Configuration is the user's primary contract with the tool (Layer 2 of the
three-layer progressive disclosure architecture in FR-7). Every field here
should be motivated by an actual user need — per AGENTS.md, do not add
config options speculatively.

The schema is intentionally minimal in the scaffold; fields will accrete
as features land.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

RuntimeMode = Literal["quick", "standard", "full"]
AnchorStrength = Literal["none", "light", "medium", "heavy"]


class DataConfig(BaseModel):
    """Where the input CSV lives and how to read it (FR-1.1)."""

    model_config = ConfigDict(extra="forbid")

    csv_path: Path
    date_column: str
    target_column: str
    spend_columns: list[str] = Field(default_factory=list)
    control_columns: list[str] = Field(default_factory=list)
    lift_test_csv: Path | None = None


class ChannelConfig(BaseModel):
    """Per-channel overrides on top of the channel-category defaults (FR-1.2)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    adstock_family: Literal["geometric", "weibull"] = "geometric"


class RefreshConfig(BaseModel):
    """Refresh accountability settings (FR-4)."""

    model_config = ConfigDict(extra="forbid")

    anchor_strength: AnchorStrength | float = "medium"
    """Named preset or numeric weight in [0, 1]. See FR-4.4."""


class RunConfig(BaseModel):
    """Top-level run-control settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    runtime_mode: RuntimeMode = "standard"
    artifact_dir: Path = Path(".wanamaker")


class WanamakerConfig(BaseModel):
    """The full validated configuration for a Wanamaker run."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    channels: list[ChannelConfig] = Field(default_factory=list)
    refresh: RefreshConfig = Field(default_factory=RefreshConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def load_config(path: Path) -> WanamakerConfig:
    """Load and validate a YAML config file.

    Args:
        path: Filesystem path to a YAML config.

    Returns:
        A validated ``WanamakerConfig``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the YAML does not match the schema.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return WanamakerConfig.model_validate(raw)
