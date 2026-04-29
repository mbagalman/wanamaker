# Feedback on `architecture.md` and GitHub Issues

Overall: the updated architecture is much stronger than the first draft. The main concerns from the prior review have been addressed: transforms are no longer treated as fixed preprocessing, `PosteriorSummary` is now the downstream contract, run identity is split into `run_fingerprint` and `run_id`, `--skip-validation` is explicitly handled, xgboost is no longer conflated with Bayesian quick mode, and the reference path now points to `docs/references/adstock_and_saturation.md`.

The GitHub Issues are also well-scoped and mostly line up with the architecture. I would proceed with the project. The criticism below is about tightening a few load-bearing details before implementation hardens.

## Important Concerns

### 1. The benchmark and engine issues have a sequencing loop

Issue #1 says the engine selection spike benchmarks each candidate on the synthetic ground-truth dataset. Issue #8 uses the synthetic benchmark for backend acceptance. But issue #9, which creates the synthetic ground-truth benchmark, is marked as blocked by #8.

That creates a practical loop: the engine cannot be selected or accepted without a benchmark, but the benchmark is said to wait for the engine.

Suggested fix: split #9 into two pieces:

- create the synthetic data generator, committed CSV, and ground-truth JSON before #1/#8
- verify model recovery on that dataset after #8

The generator should be engine-independent and should use the canonical NumPy transform functions once #3-#5 are done.

### 2. Credible interval mass is inconsistent with the PRD

The PRD and issue #8 use 95% credible interval coverage as an acceptance criterion. The current `engine/summary.py` docstrings describe `hdi_low` and `hdi_high` as 94% HDIs because that is the ArviZ default.

That mismatch will quietly leak into reports, Trust Card logic, benchmark tests, and user-facing wording.

Suggested fix: make interval mass explicit in the summary contract. Either standardize on 95% intervals everywhere or add a field such as `interval_mass: float = 0.95` to `ParameterSummary`, `ChannelContributionSummary`, and `PredictiveSummary`. For this project, 95% is the safer default because it matches the locked PRD.

### 3. Forecast and scenario comparison need historical spend context

The architecture notes that extrapolation warnings require observed spend ranges, but issue #20 only has `forecast` accept a `Posterior` and a future spend plan. That is not enough information to flag spend outside the historical range unless the forecast function also receives fit metadata or the posterior summary contains spend ranges.

Suggested fix: add a small `SpendRange` or `FitContext` contract before implementing #20/#21. It should include observed min/max per channel and spend-invariant channel flags. Then update #20 and #21 so extrapolation warnings are not left as ad hoc logic.

### 4. Artifact serialization is under-specified

Several issues assume typed objects are written to and loaded from artifacts:

- `summary.json` stores `PosteriorSummary`
- `manifest.json` stores identity and validation state
- `report` loads `TrustCard`
- `refresh` persists a diff report

The architecture names these files, but there is no issue dedicated to JSON schema, serialization/deserialization, schema versioning, or backward compatibility. This will become painful once reports and refresh compare old runs.

Suggested fix: add a dedicated issue for artifact schemas and serializers. It should cover `PosteriorSummary`, `TrustCard`, `RefreshDiff`, and `manifest.json`, including `summary_schema_version` and round-trip tests.

### 5. Reproducibility should be tested during engine selection, not only after

Issue #11 adds a bit-for-bit reproducibility CI test after the backend exists. That is necessary, but reproducibility is also a deciding constraint for #1. Different engines, BLAS settings, chain parallelism, and sampler implementations may behave differently across platforms.

Suggested fix: add reproducibility to the Phase -1 engine decision criteria: same data, config, and seed should produce byte-identical `PosteriorSummary` on repeated runs in the same clean environment. If cross-platform bit-for-bit identity is required, test that explicitly during #1 rather than discovering it after the engine has been chosen.

### 6. Trust Card persistence is unclear

Issue #19 implements Trust Card computation. Issue #30 says `report` loads a `TrustCard` from the run, but #10 does not list a Trust Card artifact and the architecture's run directory does not include one.

Suggested fix: decide whether Trust Cards are persisted as `trust_card.json` or recomputed from `summary.json`, `manifest.json`, and any `refresh_diff.json` each time. I prefer persisting the computed card with a schema version, because it makes reports auditable and keeps the exact status visible after thresholds change in future versions.

### 7. Structural break detection may need a dependency decision

Issue #14 suggests using a robust change-point library such as `ruptures`. That may be reasonable, but it would become a production dependency in a core command. It should be evaluated with the same install-friction and local-first discipline as the engine, even though it is smaller.

Suggested fix: make #14 include a mini dependency decision: dependency size, install behavior on Windows/macOS/Linux, deterministic behavior, and whether a simpler in-house method is sufficient for v1.

## Smaller Notes

- `engine/summary.py` says summaries are serialized to the run manifest, but the architecture says `PosteriorSummary` is written to `summary.json`. The latter is correct; update the docstring when touching that file.
- Issue #15 says structural breaks run only with `--config`, while the updated architecture marks structural breaks as config-independent. Pick one behavior and align the issue.
- Issue #33 should probably run on release branches or nightly if running every PR is too heavy once `fit` exists. The acceptance is right; the cadence may need practicality.
- Issue #40 mentions a one-command example, but the CLI has six commands and no `wanamaker run --example` command. Either add an issue for a one-command demo wrapper or revise the README/release checklist language.

## Recommendation

Proceed with the plan. The architecture and issues are coherent enough to start work. Before implementing the engine backend, I would first clean up the sequencing loop around the synthetic benchmark, standardize credible interval mass at 95%, and add an artifact schema/serialization issue. Those are the places most likely to create avoidable rework later.
