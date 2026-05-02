# Wanamaker Documentation

Wanamaker is an open-source Bayesian marketing mix model for teams that need credible
measurement without a deep data science bench.

The project is in active pre-1.0 development. The PyMC engine decision is made,
the local workflow is implemented, and the docs now describe source-checkout
usage while public package and image release work is still pending.

## Start Here

- [Installation](installation.md) describes source-checkout and local Docker usage.
- [Quickstart](quickstart.md) walks through the bundled public benchmark.
- [Analyst's Guide](guides/analyst_guide.md) explains MMM concepts at a working level.
- [What Wanamaker Tells Your CMO](guides/cmo_guide.md) explains outputs in business terms.
- [API Reference](api_reference.md) documents the CLI, YAML config, Python modules, and artifact files.

## Project References

- [BRD/PRD](internal/wanamaker_brd_prd.md) is the locked product and strategy source of truth.
- [Architecture](internal/architecture.md) defines module responsibilities and engineering boundaries.
- [Adstock and Saturation](references/adstock_and_saturation.md) is required reading before
  changing statistical transforms.
- [Risk-Adjusted Allocation: Math Reference](references/risk_adjusted_allocation_math.md)
  is required reading before changing the ramp recommender.
- [Verification Guide](verification.md) maps every BRD/PRD claim and
  AGENTS.md hard rule to the module that implements it and the test
  that proves it. Read this if you want to audit Wanamaker rather
  than take its word.

