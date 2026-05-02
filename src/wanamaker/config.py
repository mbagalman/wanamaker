"""YAML configuration loading and validation.

Configuration is the user's primary contract with the tool (Layer 2 of the
three-layer progressive disclosure architecture in FR-7). Every field here
should be motivated by an actual user need — per AGENTS.md, do not add
config options speculatively.

The schema is intentionally compact. New fields should be added only when a
real workflow needs them because every option becomes part of the user-facing
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RuntimeMode = Literal["quick", "standard", "full"]
AnchorStrength = Literal["none", "light", "medium", "heavy"]
BudgetMode = Literal["hold_total", "allow_increase", "allow_decrease", "allow_change"]


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


class LiftTestCalibrationConfig(BaseModel):
    """Configuration for lift-test / experiment calibration."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    mode: Literal["roi_prior"] = "roi_prior"


class CalibrationConfig(BaseModel):
    """Evidence and priors to calibrate the model."""

    model_config = ConfigDict(extra="forbid")

    lift_tests: LiftTestCalibrationConfig | None = None


class ScenarioGenerationConfig(BaseModel):
    """Explicit constraints for future bounded candidate scenario generation."""

    model_config = ConfigDict(extra="forbid")

    budget_mode: BudgetMode = "hold_total"
    top_n: int = Field(default=5, ge=1, le=20)
    max_channel_change: float = Field(default=0.15, ge=0.0, le=1.0)
    max_total_moved_budget: float = Field(default=0.20, ge=0.0, le=1.0)
    locked_channels: list[str] = Field(default_factory=list)
    excluded_channels: list[str] = Field(default_factory=list)
    min_spend: dict[str, float] = Field(default_factory=dict)
    max_spend: dict[str, float] = Field(default_factory=dict)
    require_historical_support: bool = True

    @model_validator(mode="after")
    def _check_channel_constraints(self) -> ScenarioGenerationConfig:
        locked = set(self.locked_channels)
        excluded = set(self.excluded_channels)
        overlap = sorted(locked & excluded)
        if overlap:
            raise ValueError(
                "scenario_generation channels cannot be both locked and excluded: "
                f"{overlap}"
            )

        for field_name, values in (
            ("locked_channels", self.locked_channels),
            ("excluded_channels", self.excluded_channels),
        ):
            blanks = [value for value in values if not value.strip()]
            if blanks:
                raise ValueError(f"scenario_generation.{field_name} cannot contain blanks")

        for field_name, bounds in (("min_spend", self.min_spend), ("max_spend", self.max_spend)):
            blank_keys = [channel for channel in bounds if not channel.strip()]
            if blank_keys:
                raise ValueError(f"scenario_generation.{field_name} cannot contain blank channels")
            negative = {channel: value for channel, value in bounds.items() if value < 0}
            if negative:
                raise ValueError(
                    f"scenario_generation.{field_name} values must be non-negative: {negative}"
                )

        inverted = sorted(
            channel
            for channel, min_value in self.min_spend.items()
            if channel in self.max_spend and min_value > self.max_spend[channel]
        )
        if inverted:
            raise ValueError(
                "scenario_generation min_spend cannot exceed max_spend for channels: "
                f"{inverted}"
            )
        return self


class WanamakerConfig(BaseModel):
    """The full validated configuration for a Wanamaker run."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    channels: list[ChannelConfig] = Field(default_factory=list)
    refresh: RefreshConfig = Field(default_factory=RefreshConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    calibration: CalibrationConfig | None = None
    scenario_generation: ScenarioGenerationConfig | None = None

    @model_validator(mode="after")
    def _check_lift_test_conflict(self) -> WanamakerConfig:
        has_legacy = self.data.lift_test_csv is not None
        has_new = (
            self.calibration is not None
            and self.calibration.lift_tests is not None
            and self.calibration.lift_tests.path is not None
        )
        if has_legacy and has_new:
            raise ValueError(
                "Cannot specify both data.lift_test_csv and calibration.lift_tests.path. "
                "Use calibration.lift_tests.path."
            )
        return self


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
