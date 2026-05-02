"""Candidate scenario-generation constraints.

This module defines the explicit, auditable constraint contract for future
bounded candidate generation. It does not generate scenarios; it validates
and resolves user configuration into immutable objects that a generator can
consume before any mechanical allocation search starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from wanamaker.config import ScenarioGenerationConfig, WanamakerConfig

BudgetMode = Literal["hold_total", "allow_increase", "allow_decrease", "allow_change"]


@dataclass(frozen=True)
class ScenarioGenerationConstraints:
    """Resolved immutable constraints for bounded scenario generation.

    Attributes:
        budget_mode: How total budget may change relative to the baseline.
        top_n: Maximum number of candidate scenarios to return.
        max_channel_change: Maximum relative change for any channel.
        max_total_moved_budget: Maximum moved budget as a share of baseline total.
        locked_channels: Channels that must remain unchanged.
        excluded_channels: Channels excluded from generated reallocations.
        min_spend: Per-channel lower spend bounds, sorted by channel.
        max_spend: Per-channel upper spend bounds, sorted by channel.
        require_historical_support: Whether candidates must stay inside observed
            historical spend ranges.
    """

    budget_mode: BudgetMode
    top_n: int
    max_channel_change: float
    max_total_moved_budget: float
    locked_channels: tuple[str, ...]
    excluded_channels: tuple[str, ...]
    min_spend: tuple[tuple[str, float], ...]
    max_spend: tuple[tuple[str, float], ...]
    require_historical_support: bool


def resolve_scenario_generation_constraints(
    cfg: WanamakerConfig,
) -> ScenarioGenerationConstraints:
    """Resolve YAML scenario-generation config into immutable constraints.

    Args:
        cfg: Validated Wanamaker configuration.

    Returns:
        Immutable constraints with sorted channel lists and spend bounds.

    Raises:
        ValueError: If constraints refer to channels not declared in
            ``cfg.channels``.

    Examples:
        >>> from wanamaker.config import ChannelConfig, DataConfig, WanamakerConfig
        >>> cfg = WanamakerConfig(
        ...     data=DataConfig(csv_path="data.csv", date_column="week", target_column="sales"),
        ...     channels=[ChannelConfig(name="search", category="paid_search")],
        ... )
        >>> resolve_scenario_generation_constraints(cfg).budget_mode
        'hold_total'
    """
    raw = cfg.scenario_generation or ScenarioGenerationConfig()
    configured_channels = {channel.name for channel in cfg.channels}
    referenced = set(raw.locked_channels)
    referenced.update(raw.excluded_channels)
    referenced.update(raw.min_spend)
    referenced.update(raw.max_spend)
    unknown = sorted(referenced - configured_channels)
    if unknown:
        raise ValueError(
            "scenario_generation references channels that are not configured: "
            f"{unknown}"
        )

    return ScenarioGenerationConstraints(
        budget_mode=raw.budget_mode,
        top_n=raw.top_n,
        max_channel_change=raw.max_channel_change,
        max_total_moved_budget=raw.max_total_moved_budget,
        locked_channels=tuple(sorted(raw.locked_channels)),
        excluded_channels=tuple(sorted(raw.excluded_channels)),
        min_spend=tuple(sorted((k, float(v)) for k, v in raw.min_spend.items())),
        max_spend=tuple(sorted((k, float(v)) for k, v in raw.max_spend.items())),
        require_historical_support=raw.require_historical_support,
    )


def validate_candidate_spend(
    baseline_spend: Mapping[str, float],
    candidate_spend: Mapping[str, float],
    constraints: ScenarioGenerationConstraints,
) -> None:
    """Validate a generated candidate against resolved constraints.

    This is the fail-closed gate a future candidate generator can call before
    scoring or reporting a scenario.

    Args:
        baseline_spend: Baseline total spend by channel.
        candidate_spend: Candidate total spend by channel.
        constraints: Resolved scenario generation constraints.

    Raises:
        ValueError: If the candidate violates budget mode, movement, locked
            channels, excluded channels, or spend bounds.
    """
    baseline = {k: float(v) for k, v in baseline_spend.items()}
    candidate = {k: float(v) for k, v in candidate_spend.items()}
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "candidate spend channels must match baseline channels; "
            f"missing={missing}, extra={extra}"
        )

    baseline_total = sum(baseline.values())
    candidate_total = sum(candidate.values())
    tolerance = max(1e-9, abs(baseline_total) * 1e-9)
    total_changed = abs(candidate_total - baseline_total) > tolerance
    if constraints.budget_mode == "hold_total" and total_changed:
        raise ValueError(
            "candidate violates scenario_generation.budget_mode=hold_total: "
            f"baseline total {baseline_total:.6g}, candidate total {candidate_total:.6g}"
        )
    if constraints.budget_mode == "allow_increase" and candidate_total + tolerance < baseline_total:
        raise ValueError("candidate total decreased but budget_mode=allow_increase")
    if constraints.budget_mode == "allow_decrease" and candidate_total > baseline_total + tolerance:
        raise ValueError("candidate total increased but budget_mode=allow_decrease")

    locked = set(constraints.locked_channels)
    excluded = set(constraints.excluded_channels)
    for channel in sorted(locked | excluded):
        if abs(candidate[channel] - baseline[channel]) > tolerance:
            status = "locked" if channel in locked else "excluded"
            raise ValueError(
                f"candidate changes {status} channel {channel!r}: "
                f"baseline {baseline[channel]:.6g}, candidate {candidate[channel]:.6g}"
            )

    for channel, base_value in baseline.items():
        candidate_value = candidate[channel]
        delta = abs(candidate_value - base_value)
        if base_value == 0:
            if delta > tolerance:
                raise ValueError(
                    f"candidate changes zero-baseline channel {channel!r}; "
                    "max_channel_change is undefined from zero"
                )
            continue
        relative_change = delta / abs(base_value)
        if relative_change > constraints.max_channel_change + 1e-12:
            raise ValueError(
                f"candidate changes channel {channel!r} by {relative_change:.1%}, "
                "above scenario_generation.max_channel_change"
            )

    if baseline_total > 0:
        moved_budget = sum(max(candidate[ch] - baseline[ch], 0.0) for ch in baseline)
        moved_share = moved_budget / baseline_total
        if moved_share > constraints.max_total_moved_budget + 1e-12:
            raise ValueError(
                f"candidate moves {moved_share:.1%} of baseline budget, above "
                "scenario_generation.max_total_moved_budget"
            )

    min_spend = dict(constraints.min_spend)
    max_spend = dict(constraints.max_spend)
    for channel, minimum in min_spend.items():
        if candidate[channel] < minimum - tolerance:
            raise ValueError(
                f"candidate spend for {channel!r} is below scenario_generation.min_spend"
            )
    for channel, maximum in max_spend.items():
        if candidate[channel] > maximum + tolerance:
            raise ValueError(
                f"candidate spend for {channel!r} is above scenario_generation.max_spend"
            )


def format_constraints_markdown(
    constraints: ScenarioGenerationConstraints,
    *,
    heading: str = "## Constraints used",
) -> str:
    """Render a human-readable Markdown section for generated reports."""
    lines = [
        heading,
        "",
        f"- Budget mode: `{constraints.budget_mode}`",
        f"- Top candidates requested: {constraints.top_n}",
        f"- Maximum per-channel change: {constraints.max_channel_change:.0%}",
        f"- Maximum total moved budget: {constraints.max_total_moved_budget:.0%}",
        "- Historical support required: "
        + ("yes" if constraints.require_historical_support else "no"),
        "- Locked channels: " + _channel_list_label(constraints.locked_channels),
        "- Excluded channels: " + _channel_list_label(constraints.excluded_channels),
        "- Minimum spend bounds: " + _bounds_label(constraints.min_spend),
        "- Maximum spend bounds: " + _bounds_label(constraints.max_spend),
        "",
    ]
    return "\n".join(lines)


def _channel_list_label(channels: tuple[str, ...]) -> str:
    if not channels:
        return "none"
    return ", ".join(f"`{channel}`" for channel in channels)


def _bounds_label(bounds: tuple[tuple[str, float], ...]) -> str:
    if not bounds:
        return "none"
    return ", ".join(f"`{channel}`={value:,.0f}" for channel, value in bounds)
