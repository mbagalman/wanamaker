"""Forecasting and scenario comparison (FR-5.1 mode 2, FR-5.2).

Two responsibilities:

- ``posterior_predictive`` — forecast the target metric given a future
  spend plan, returning point estimate and credible interval.
- ``scenarios`` — rank user-supplied budget plans (2–3 typical) with
  uncertainty, plus explicit warnings when plans extrapolate beyond the
  historical observed spend range.

v1 deliberately ships scenario comparison rather than constrained inverse
optimization (deferred to v1.1, BRD/PRD §4.2). User-driven scenarios keep
the human in control of strategic decisions.
"""
