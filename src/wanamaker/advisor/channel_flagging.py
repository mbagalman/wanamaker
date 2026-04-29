"""Channel flagging for experimental validation (FR-5.5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelFlag:
    """A single flagged channel and the reason."""

    channel: str
    posterior_ci: tuple[float, float]
    spend: float
    rationale: str


def flag_channels(posterior, spend_summary) -> list[ChannelFlag]:  # noqa: ANN001
    raise NotImplementedError("Phase 2: experiment-advisor channel flagging")
