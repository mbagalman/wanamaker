"""Trust Card data structures (FR-5.4).

v1 dimensions (FR-5.4):

- convergence            (R-hat, effective sample size)
- holdout_accuracy
- refresh_stability      (only when there's a previous run)
- prior_sensitivity
- saturation_identifiability  (per channel)
- lift_test_consistency  (only when calibration data was provided)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TrustStatus(StrEnum):
    PASS = "pass"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass(frozen=True)
class TrustDimension:
    name: str
    status: TrustStatus
    explanation: str


@dataclass(frozen=True)
class TrustCard:
    dimensions: list[TrustDimension]

    def dimension(self, name: str) -> TrustDimension | None:
        """Return one dimension by name, or ``None`` when absent."""
        return next((dimension for dimension in self.dimensions if dimension.name == name), None)

    @property
    def has_weak_dimension(self) -> bool:
        """Whether any dimension should trigger hedged report language."""
        return any(dimension.status == TrustStatus.WEAK for dimension in self.dimensions)

    @property
    def weak_dimension_names(self) -> list[str]:
        """Names of dimensions with ``weak`` status, for report templates."""
        return [
            dimension.name
            for dimension in self.dimensions
            if dimension.status == TrustStatus.WEAK
        ]
