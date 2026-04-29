"""Model specification and default priors.

The model is engine-agnostic by design — see ``wanamaker.engine``. This
subpackage produces a ``ModelSpec`` that any backend can consume.

Default model (FR-3.1): geometric adstock + Hill saturation per channel,
additive linear combination of transformed media plus controls,
weakly-informative priors on all coefficients.
"""

from wanamaker.model.spec import (
    AnchoredPrior,
    ChannelSpec,
    HoldoutConfig,
    LiftPrior,
    ModelSpec,
    SeasonalitySpec,
)

__all__ = [
    "AnchoredPrior",
    "ChannelSpec",
    "HoldoutConfig",
    "LiftPrior",
    "ModelSpec",
    "SeasonalitySpec",
]
