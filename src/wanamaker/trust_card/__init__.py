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

__all__ = ["TrustCard", "TrustDimension", "TrustStatus"]
