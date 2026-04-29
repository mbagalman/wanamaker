"""xgboost in a strictly supporting role.

Per AGENTS.md Hard Rule 3: xgboost is in this project for two narrow
purposes only:

1. **Quick-mode forecast preview** — a fast tree-based forecast with
   conformal intervals as a sanity check before the full Bayesian fit.
   Output is explicitly labeled as a forecast preview, not as ROI.

2. **Validation cross-check** — an independent tree-based fit used to
   detect specification problems in the Bayesian model. If the Bayesian
   and tree forecasts disagree substantially, the Trust Card raises a flag.

xgboost does **not** produce ROI estimates, saturation curves, adstock
parameters, or any other channel-level inference output. If a feature
request seems to want xgboost for those, re-read the BRD/PRD.

The module name is leading-underscore to signal that it is internal
support, not a user-facing API surface.
"""
