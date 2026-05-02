"""Tests for explicit scenario-generation constraints (issue #88)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from wanamaker.config import (
    ChannelConfig,
    DataConfig,
    ScenarioGenerationConfig,
    WanamakerConfig,
)
from wanamaker.forecast.constraints import (
    ScenarioGenerationConstraints,
    format_constraints_markdown,
    resolve_scenario_generation_constraints,
    validate_candidate_spend,
)


def _config(
    tmp_path: Path,
    *,
    scenario_generation: ScenarioGenerationConfig | None = None,
) -> WanamakerConfig:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text(
        "week,revenue,search,tv,affiliate\n2024-01-01,100,10,20,5\n",
        encoding="utf-8",
    )
    return WanamakerConfig(
        data=DataConfig(
            csv_path=data_csv,
            date_column="week",
            target_column="revenue",
            spend_columns=["search", "tv", "affiliate"],
        ),
        channels=[
            ChannelConfig(name="search", category="paid_search"),
            ChannelConfig(name="tv", category="linear_tv"),
            ChannelConfig(name="affiliate", category="affiliate"),
        ],
        scenario_generation=scenario_generation,
    )


def test_missing_scenario_generation_uses_conservative_defaults(tmp_path: Path) -> None:
    constraints = resolve_scenario_generation_constraints(_config(tmp_path))

    assert constraints == ScenarioGenerationConstraints(
        budget_mode="hold_total",
        top_n=5,
        max_channel_change=0.15,
        max_total_moved_budget=0.20,
        locked_channels=(),
        excluded_channels=(),
        min_spend=(),
        max_spend=(),
        require_historical_support=True,
    )


def test_resolved_constraints_are_immutable_and_sorted(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        scenario_generation=ScenarioGenerationConfig(
            locked_channels=["tv"],
            excluded_channels=["affiliate"],
            min_spend={"search": 5000.0},
            max_spend={"search": 20000.0},
        ),
    )

    constraints = resolve_scenario_generation_constraints(cfg)

    assert constraints.locked_channels == ("tv",)
    assert constraints.excluded_channels == ("affiliate",)
    assert constraints.min_spend == (("search", 5000.0),)
    assert constraints.max_spend == (("search", 20000.0),)
    with pytest.raises(FrozenInstanceError):
        constraints.top_n = 10  # type: ignore[misc]


def test_config_rejects_invalid_percentages_and_top_n() -> None:
    with pytest.raises(ValidationError, match="max_channel_change"):
        ScenarioGenerationConfig(max_channel_change=1.2)

    with pytest.raises(ValidationError, match="top_n"):
        ScenarioGenerationConfig(top_n=0)


def test_config_rejects_locked_and_excluded_overlap() -> None:
    with pytest.raises(ValidationError, match="both locked and excluded"):
        ScenarioGenerationConfig(
            locked_channels=["tv"],
            excluded_channels=["tv"],
        )


def test_config_rejects_invalid_spend_bounds() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        ScenarioGenerationConfig(min_spend={"search": -1.0})

    with pytest.raises(ValidationError, match="min_spend cannot exceed max_spend"):
        ScenarioGenerationConfig(
            min_spend={"search": 20000.0},
            max_spend={"search": 5000.0},
        )


def test_resolver_rejects_unknown_constraint_channels(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        scenario_generation=ScenarioGenerationConfig(
            locked_channels=["ghost"],
        ),
    )

    with pytest.raises(ValueError, match="not configured"):
        resolve_scenario_generation_constraints(cfg)


def test_validate_candidate_preserves_total_budget_when_required(tmp_path: Path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                max_channel_change=1.0,
                max_total_moved_budget=1.0,
            ),
        )
    )

    validate_candidate_spend(
        {"search": 100.0, "tv": 100.0, "affiliate": 0.0},
        {"search": 120.0, "tv": 80.0, "affiliate": 0.0},
        constraints,
    )

    with pytest.raises(ValueError, match="hold_total"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 0.0},
            {"search": 130.0, "tv": 100.0, "affiliate": 0.0},
            constraints,
        )


def test_validate_candidate_enforces_locked_and_excluded_channels(tmp_path: Path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                max_channel_change=1.0,
                max_total_moved_budget=1.0,
                locked_channels=["tv"],
                excluded_channels=["affiliate"],
            ),
        )
    )

    with pytest.raises(ValueError, match="locked channel 'tv'"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 50.0},
            {"search": 120.0, "tv": 80.0, "affiliate": 50.0},
            constraints,
        )

    with pytest.raises(ValueError, match="excluded channel 'affiliate'"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 50.0},
            {"search": 110.0, "tv": 100.0, "affiliate": 40.0},
            constraints,
        )


def test_validate_candidate_enforces_movement_and_spend_bounds(tmp_path: Path) -> None:
    change_constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                max_channel_change=0.10,
                max_total_moved_budget=0.05,
                min_spend={"search": 95.0},
                max_spend={"tv": 105.0},
            ),
        )
    )

    with pytest.raises(ValueError, match="max_channel_change"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 100.0},
            {"search": 120.0, "tv": 90.0, "affiliate": 90.0},
            change_constraints,
        )

    moved_constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                max_channel_change=1.0,
                max_total_moved_budget=0.05,
            ),
        )
    )
    with pytest.raises(ValueError, match="max_total_moved_budget"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 100.0},
            {"search": 110.0, "tv": 110.0, "affiliate": 80.0},
            moved_constraints,
        )

    bound_constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                max_channel_change=1.0,
                max_total_moved_budget=1.0,
                min_spend={"search": 95.0},
                max_spend={"tv": 105.0},
            ),
        )
    )
    with pytest.raises(ValueError, match="min_spend"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 100.0},
            {"search": 94.0, "tv": 103.0, "affiliate": 103.0},
            bound_constraints,
        )

    with pytest.raises(ValueError, match="max_spend"):
        validate_candidate_spend(
            {"search": 100.0, "tv": 100.0, "affiliate": 100.0},
            {"search": 98.0, "tv": 106.0, "affiliate": 96.0},
            bound_constraints,
        )


def test_constraints_markdown_is_report_ready(tmp_path: Path) -> None:
    constraints = resolve_scenario_generation_constraints(
        _config(
            tmp_path,
            scenario_generation=ScenarioGenerationConfig(
                budget_mode="hold_total",
                top_n=3,
                locked_channels=["tv"],
                excluded_channels=["affiliate"],
                min_spend={"search": 5000.0},
                max_spend={"search": 20000.0},
                require_historical_support=True,
            ),
        )
    )

    markdown = format_constraints_markdown(constraints)

    assert markdown.startswith("## Constraints used")
    assert "- Budget mode: `hold_total`" in markdown
    assert "- Locked channels: `tv`" in markdown
    assert "- Excluded channels: `affiliate`" in markdown
    assert "- Minimum spend bounds: `search`=5,000" in markdown
    assert "- Maximum spend bounds: `search`=20,000" in markdown
