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
from enum import Enum


class TrustStatus(str, Enum):
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
