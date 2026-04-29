"""Pre-flight data readiness diagnostic.

Implements ``wanamaker diagnose`` (FR-2). The diagnostic runs a battery of
checks (FR-2.2) and returns a discrete readiness level (FR-2.3) — never a
composite numeric score, deliberately.

Per FR-2.4, validation is required before fitting in the default flow.
"""

from wanamaker.diagnose.readiness import ReadinessLevel, ReadinessReport

__all__ = ["ReadinessLevel", "ReadinessReport"]
