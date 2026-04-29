# Feedback on `architecture.md`

Overall: the architecture plan is sound and aligned with the BRD/PRD. The module boundaries support the project's main thesis: local-first execution, deterministic reporting, refresh accountability, and an engine-agnostic Bayesian core. I would proceed with this plan, but I would tighten the points below before Phase 0 implementation begins.

## Substantive Criticism

### 1. The transform data flow is conceptually ambiguous

The data-flow diagram shows `transforms/` applying geometric/Weibull adstock and Hill saturation before `engine.fit`. That is only correct if the transform parameters are fixed before fitting. The PRD, however, treats adstock and saturation parameters as inferred channel-level quantities, with posterior intervals and refresh anchoring.

If decay, half-life, EC50, and slope are sampled parameters, the transforms must be part of the probabilistic program implemented by the engine backend, not a pre-engine preprocessing step. The pure NumPy transform functions are still useful for tests, documentation, benchmarks, fixed-parameter previews, and backend validation, but the architecture should say explicitly how backends consume the canonical transform definitions.

Suggested fix: revise Section 4 so `model/` produces a parameterized `ModelSpec`, `transforms/` owns canonical formulas and test fixtures, and `engine/` applies those formulas inside the backend-specific model graph.

### 2. The engine abstraction is too narrow for downstream features

`Engine.fit(...) -> FitResult` and `Posterior(raw: Any)` are a good starting point for a spike, but too opaque for the planned architecture. Refresh diffs, trust cards, reports, scenario comparison, lift-test consistency, prior sensitivity, and benchmark acceptance tests all need stable access to the same posterior summaries.

Without an engine-neutral posterior summary contract, each downstream module will either reach into `Posterior.raw` or invent its own summary shape. That would weaken the engine boundary at the exact point the project needs it most.

Suggested fix: add a small typed summary layer before implementing a real backend. For example:

- `PosteriorSummary`
- `ParameterSummary`
- `ChannelContributionSummary`
- `PredictiveSummary`
- helper functions owned by `engine/summary.py` or `model/summaries.py`

The raw engine object can still be exposed for expert users, but core modules should depend on typed summaries.

### 3. `ModelSpec` is under-specified relative to the architecture

The document describes `ModelSpec` as a pure data description of channels, transforms, and priors, but the current scaffold only contains channel names, categories, adstock family, and controls. That is fine for now, but the architecture should define the intended shape before the backend spike hardens around an accidental minimum.

The spec likely needs to represent:

- target column and date/frequency assumptions
- channel prior references or resolved prior objects
- lift-test calibration inputs
- holdout configuration
- seasonality/trend/control treatment
- runtime mode interpretation
- refresh anchoring priors
- whether a channel is spend-invariant or weakly identified

Suggested fix: include a planned `ModelSpec` schema in `architecture.md`, even if many fields remain unimplemented.

### 4. Artifact identity and reproducibility need sharper separation

The architecture says run IDs are content-addressed from data hash, config, and timestamp, and are stable for the same `(data, config)` pair within the same minute. This mixes two concerns:

- reproducible model outputs for the same data/config/seed
- unique run records for repeated executions

Including timestamps in run identity means identical runs across different minutes produce different run IDs. That may be acceptable for artifact bookkeeping, but it should not be described as content-addressed in the reproducibility sense.

Suggested fix: define both concepts:

- a deterministic `run_fingerprint` from data hash, normalized config, code/package version, engine name/version, and seed
- a unique `run_id` for storage, optionally derived from fingerprint plus timestamp or a collision suffix

That gives refresh/accountability reports a stable way to say "same analytical setup" without forbidding repeated run artifacts.

### 5. The validation skip path needs a product decision

The data-flow diagram mentions `--skip-validation`, but the CLI scaffold does not expose it, and the PRD emphasizes that readiness diagnostics cannot be skipped silently. A skip flag may be reasonable for expert users, tests, or benchmark workflows, but it is risky for the target persona.

Suggested fix: either remove `--skip-validation` from the architecture for now, or document it as an expert-only flag that requires an explicit reason and is recorded in the run artifact and Trust Card.

### 6. "Quick mode" has two meanings

Section 3.2 says runtime modes (`quick`, `standard`, `full`) control sampler effort only and keep the model specification identical. Section 3.16 describes xgboost as a "quick-mode forecast preview." These should not share the same label unless they are intentionally part of one user-facing mode.

Suggested fix: reserve `runtime_mode=quick` for Bayesian sampler effort, and call the xgboost path something like `preview_forecast` or `xgb_crosscheck`. This avoids accidental misuse of xgboost as a substitute model.

### 7. The no-network CI gate is helpful but overstated

`tests/test_no_network_in_core.py` is a good fast guardrail, but the architecture describes it as walking the import graph and rejecting banned modules. In practice, it catches modules imported at import time. It may miss optional or lazy imports inside functions until those functions execute.

Suggested fix: keep the current test, but describe it as a fast import-time guardrail. Add a planned source scan or command-level network-isolated integration test for the six core commands once they are implemented.

### 8. Reference paths have drifted

The architecture links to `adstock_and_saturation.md` as if it lives in `docs/`, while `AGENTS.md` says the canonical path is `docs/references/adstock_and_saturation.md`. In the current repo, the file appears at the repository root as `adstock_and_saturation.md`.

This matters because the transform rules say contributors must read the canonical reference before changing statistical code.

Suggested fix: move or copy the reference to `docs/references/adstock_and_saturation.md`, then update `architecture.md`, `AGENTS.md`, and transform docstrings to point to the same path.

### 9. Encoding/mojibake should be fixed early

Several docs and code comments display mojibake: punctuation and arrows are rendered as garbled `a`-with-circumflex sequences, and box-drawing diagrams do not render cleanly. This is not an architectural flaw, but documentation quality is a v1 deliverable and trust is part of the product. Broken punctuation in the core docs undercuts that tone.

Suggested fix: normalize affected Markdown and docstrings to UTF-8, or convert decorative punctuation and diagrams to plain ASCII.

## Smaller Suggestions

- Consider adding a formal `docs/decisions/` directory for Phase -1 engine selection and other architectural decisions. The PRD refers to a decision log, but the repository does not yet have one.
- Define ownership for plotting/data visualization. Reports own templates, but response curve extrapolation warnings will likely need reusable chart data contracts.
- Consider making benchmark dataset generation scripts explicit. Loaders are planned, but reproducible synthetic data generation is as important as loading committed CSVs.
- Clarify whether `diagnose` operates from CSV-only input or from full YAML config. The CLI takes only a CSV, while many checks need spend/control column knowledge.
- Add a planned "run manifest" artifact that records config, package version, engine version, seed, data hash, readiness result, validation-skip status if any, and summary schema version.

## Recommendation

Proceed with the documented architecture. The plan is coherent and appropriately scoped for Phase -1. The main thing I would fix before implementing the engine spike is the contract between `ModelSpec`, `transforms`, `engine`, and posterior summaries. That is the load-bearing boundary: if it stays vague, the first backend implementation will define the architecture by accident.
