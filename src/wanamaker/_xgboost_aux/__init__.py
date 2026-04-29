"""xgboost in a strictly supporting role.

Per AGENTS.md Hard Rule 3: xgboost is in this project for two narrow
purposes only:

1. **Preview forecast** (internal label: ``xgb_preview``) -- a fast
   tree-based forecast with conformal intervals as a sanity check before
   the full Bayesian fit. Output is explicitly labeled as a forecast
   preview, not as ROI. This is NOT the same as ``runtime_mode="quick"``,
   which controls Bayesian sampler effort. These are separate concepts.

2. **Validation cross-check** (internal label: ``xgb_crosscheck``) -- an
   independent tree-based fit used to detect specification problems in the
   Bayesian model. If the Bayesian and tree forecasts disagree
   substantially, the Trust Card raises a flag.

Note on naming: ``runtime_mode`` ("quick" / "standard" / "full") refers
exclusively to Bayesian sampler effort. The xgboost paths above are
identified by the labels ``xgb_preview`` and ``xgb_crosscheck``
internally (e.g. in ``seeding.derive_seed``). Never call these paths
"quick mode" -- that phrase is reserved for the sampler tier.

xgboost does **not** produce ROI estimates, saturation curves, adstock
parameters, or any other channel-level inference output. If a feature
request seems to want xgboost for those, re-read the BRD/PRD.

The module name is leading-underscore to signal that it is internal
support, not a user-facing API surface.
"""
