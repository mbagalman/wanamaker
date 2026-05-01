"""Discrete readiness levels and the structured report (FR-2.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ReadinessLevel(StrEnum):
    """The four discrete readiness verdicts. Deliberately not a numeric score."""

    READY = "ready"
    USABLE_WITH_WARNINGS = "usable_with_warnings"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    NOT_RECOMMENDED = "not_recommended"


class CheckSeverity(StrEnum):
    """Severity levels emitted by individual readiness checks."""

    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single diagnostic check."""

    name: str
    severity: CheckSeverity
    message: str


@dataclass(frozen=True)
class ReadinessReport:
    """Structured output of ``wanamaker diagnose``."""

    level: ReadinessLevel
    checks: list[CheckResult] = field(default_factory=list)
