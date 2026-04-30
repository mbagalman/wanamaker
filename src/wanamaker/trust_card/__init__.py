"""Model Trust Card (FR-5.4).

A single-page summary with named credibility dimensions, each with a
status (pass / moderate / weak) and a one-line explanation.

The Trust Card is the connective tissue between model output and
recommendations: weak status on a dimension translates into specific
hedged language in the executive summary and specific warnings in
scenario comparison.
"""

from wanamaker.trust_card.card import (
    TrustCard,
    TrustDimension,
    TrustStatus,
)
from wanamaker.trust_card.compute import (
    PriorSensitivityResult,
    build_trust_card,
    convergence_dimension,
    holdout_accuracy_dimension,
    lift_test_consistency_dimension,
    prior_sensitivity_dimension,
    refresh_stability_dimension,
    saturation_identifiability_dimension,
)

__all__ = [
    "PriorSensitivityResult",
    "TrustCard",
    "TrustDimension",
    "TrustStatus",
    "build_trust_card",
    "convergence_dimension",
    "holdout_accuracy_dimension",
    "lift_test_consistency_dimension",
    "prior_sensitivity_dimension",
    "refresh_stability_dimension",
    "saturation_identifiability_dimension",
]
