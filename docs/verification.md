# Verification Guide: How to Audit What Wanamaker Claims

This guide is for **auditors, reviewers, and skeptical readers** — people who do not want to take the project's word for it. Every Wanamaker claim worth auditing is mapped below to (a) the module that implements it and (b) the test that proves the implementation behaves as advertised. Where a claim has only partial test coverage, that gap is stated openly: silent overclaiming is the single thing this guide exists to prevent.

The guide is organised by claim category. The BRD/PRD lives at [`docs/internal/wanamaker_brd_prd.md`](internal/wanamaker_brd_prd.md); the architectural invariants at [`AGENTS.md`](../AGENTS.md). This file is the *map* between those documents and the code.

> **Caveat conventions used below.** A claim is reported as **gated in CI** if a test file under `tests/` exercises it on every fast unit run. **Engine-gated** means a test exists but is skipped unless `WANAMAKER_RUN_ENGINE_TESTS=1` (the test fits a real PyMC model and is too slow for the unit lane). **Partially gated** means an aspect of the claim is unit-tested but a published acceptance threshold is not hard-asserted. **Not currently gated** means the implementation exists and the documentation describes the behaviour, but no automated test asserts it.

---

## 1. Architectural Hard Rules (AGENTS.md)

Violating these breaks the credibility thesis the project is built on.

| Rule | Implementation | Test | Status |
|---|---|---|---|
| 1. No network calls in core operations | All core subpackages listed in [`tests/test_no_network_in_core.py:CORE_SUBPACKAGES`](../tests/test_no_network_in_core.py); banned modules in `BANNED_MODULES` | [`tests/test_no_network_in_core.py`](../tests/test_no_network_in_core.py) (static import-graph walk) and [`tests/test_network_isolation.py`](../tests/test_network_isolation.py) (runtime, engine-gated) | **Gated in CI** (static); runtime engine-gated. |
| 2. No LLM calls for output generation | All report text rendered from Jinja2 templates under [`src/wanamaker/reports/`](../src/wanamaker/reports/); banned LLM SDKs (`openai`, `anthropic`) in static rule above | Same `BANNED_MODULES` set in [`tests/test_no_network_in_core.py`](../tests/test_no_network_in_core.py) | **Gated in CI** (architectural — no LLM SDK is importable from core code). |
| 3. No tree-based modeling for ROI / saturation / adstock | `xgboost` walled off in [`src/wanamaker/_xgboost_aux/`](../src/wanamaker/_xgboost_aux/); engine code uses PyMC only | Engine choice enforced by code structure; no automated guard on the import boundary today. | **Not currently gated** by automated test — relies on review and code-structure conventions. |
| 4. Statistical code is load-bearing (citations + worked-example tests) | Each transform docstring cites a canonical source; see [`transforms/adstock.py`](../src/wanamaker/transforms/adstock.py), [`transforms/saturation.py`](../src/wanamaker/transforms/saturation.py), [`forecast/ramp.py`](../src/wanamaker/forecast/ramp.py), [`model/priors.py`](../src/wanamaker/model/priors.py) | Worked-example unit tests in [`tests/unit/test_adstock.py`](../tests/unit/test_adstock.py), [`tests/unit/test_saturation.py`](../tests/unit/test_saturation.py), [`tests/unit/test_priors.py`](../tests/unit/test_priors.py); doctests in each module run under `pytest --doctest-modules`. | **Gated in CI**. |
| 5. Reproducibility is a contract | Single seed plumbed through [`src/wanamaker/seeding.py`](../src/wanamaker/seeding.py) and [`engine/pymc.py`](../src/wanamaker/engine/pymc.py); no library-internal `np.random.seed()` calls | [`tests/test_reproducibility.py::test_pymc_engine_reproducible_on_synthetic_ground_truth`](../tests/test_reproducibility.py) (engine-gated) asserts numerical agreement at `RTOL=1e-6` on every float field of the posterior summary | **Engine-gated.** The criterion is `RTOL=1e-6`, not literal bit-for-bit identity — see PRD NFR-2 (v0.5). |
| 6. CSV interface is the user contract | [`src/wanamaker/data/io.py`](../src/wanamaker/data/io.py), [`src/wanamaker/config.py`](../src/wanamaker/config.py) `DataConfig` | [`tests/unit/test_diagnose_command.py`](../tests/unit/test_diagnose_command.py), [`tests/unit/test_fit_command.py`](../tests/unit/test_fit_command.py) | **Gated in CI**. |

---

## 2. Functional Requirements

### 2.1 Data Input (FR-1.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-1.1 | CSV input format with required date and target columns | [`src/wanamaker/data/io.py::load_input_csv`](../src/wanamaker/data/io.py) and [`config.py::DataConfig`](../src/wanamaker/config.py) | [`tests/unit/test_diagnose_command.py`](../tests/unit/test_diagnose_command.py), [`tests/unit/test_fit_command.py`](../tests/unit/test_fit_command.py) | Gated in CI for schema; the BRD acceptance criterion ("three real-world data shapes") is not separately gated. |
| FR-1.2 | 10-category channel taxonomy with per-category default priors | [`src/wanamaker/data/taxonomy.py::DEFAULT_CHANNEL_CATEGORIES`](../src/wanamaker/data/taxonomy.py); [`src/wanamaker/model/priors.py`](../src/wanamaker/model/priors.py) (citations to Jin et al. 2017, Broadbent 1979, Robyn) | [`tests/unit/test_priors.py`](../tests/unit/test_priors.py), [`tests/unit/test_model_spec.py`](../tests/unit/test_model_spec.py) | Gated in CI for prior values; empirical-source mapping in [`docs/references/adstock_and_saturation.md`](references/adstock_and_saturation.md) § 4 is documentation, not a test. |
| FR-1.3 | Lift-test calibration CSV (three accepted schemas + multi-row pooling) | [`src/wanamaker/data/io.py::load_lift_test_csv`](../src/wanamaker/data/io.py); [`src/wanamaker/model/spec.py::LiftPrior`](../src/wanamaker/model/spec.py); precision-weighted pooling in [`src/wanamaker/model/builder.py`](../src/wanamaker/model/builder.py) | [`tests/unit/test_lift_test_calibration.py`](../tests/unit/test_lift_test_calibration.py); end-to-end posterior shift in [`tests/benchmarks/test_calibration_behavior.py`](../tests/benchmarks/test_calibration_behavior.py) (engine-gated) | Schema and pooling math: gated in CI. Posterior-shift acceptance: engine-gated. |
| FR-1.4 | Additive control variables (price, promotions, holidays, macro) | [`config.py::DataConfig::control_columns`](../src/wanamaker/config.py); flows to [`model/spec.py::ModelSpec`](../src/wanamaker/model/spec.py) and [`engine/pymc.py`](../src/wanamaker/engine/pymc.py) | [`tests/unit/test_fit_command.py`](../tests/unit/test_fit_command.py) for wiring | Wiring gated in CI; no direct unit test asserting strict additivity (engine-internal). |

### 2.2 Data Readiness Diagnostic (FR-2.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-2.1 | `wanamaker diagnose` runs without fitting | [`src/wanamaker/cli.py::diagnose`](../src/wanamaker/cli.py); [`src/wanamaker/diagnose/__init__.py::run_diagnostic`](../src/wanamaker/diagnose/__init__.py) | [`tests/unit/test_diagnose_command.py`](../tests/unit/test_diagnose_command.py); [`tests/unit/test_cli_readiness.py`](../tests/unit/test_cli_readiness.py) | Gated in CI. |
| FR-2.2 | Risk categories: history length, missing values, spend variation, collinearity, variable count, date regularity, target stability, target leakage, structural breaks | [`src/wanamaker/diagnose/checks.py`](../src/wanamaker/diagnose/checks.py) (one function per check) | [`tests/unit/test_diagnose_core_checks.py`](../tests/unit/test_diagnose_core_checks.py), [`tests/unit/test_diagnose_spend_aware_checks.py`](../tests/unit/test_diagnose_spend_aware_checks.py), [`tests/unit/test_structural_breaks.py`](../tests/unit/test_structural_breaks.py); benchmark acceptance in [`tests/unit/test_diagnostic_benchmarks.py`](../tests/unit/test_diagnostic_benchmarks.py) (`TestLowVariationDataset…`, `TestCollinearityDataset…`, `TestTargetLeakageDataset…`, `TestStructuralBreakDataset…`) | Each check gated in CI. **Caveat:** the BRD's structural-break "±2 weeks of true break point" tolerance is not explicitly asserted; the test verifies a break is detected, not that the detected timestamp is within ±2 weeks. |
| FR-2.3 | Four discrete readiness levels | [`src/wanamaker/diagnose/readiness.py::ReadinessLevel`](../src/wanamaker/diagnose/readiness.py) | [`tests/unit/test_cli_readiness.py`](../tests/unit/test_cli_readiness.py) (one test per level) | Gated in CI. |
| FR-2.4 | Default flow blocks fit on bad data; `--skip-validation` overrides and is recorded | [`src/wanamaker/cli.py::fit`](../src/wanamaker/cli.py) (`--skip-validation` flag, manifest field) | [`tests/unit/test_fit_command.py`](../tests/unit/test_fit_command.py); skip flag in [`tests/unit/test_artifact_schemas.py`](../tests/unit/test_artifact_schemas.py) | Gated in CI. |

### 2.3 Model Fitting (FR-3.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-3.1 | Geometric adstock, Hill saturation, additive linear, weakly-informative priors | [`src/wanamaker/model/spec.py::ModelSpec`](../src/wanamaker/model/spec.py); [`engine/pymc.py`](../src/wanamaker/engine/pymc.py); [`model/priors.py`](../src/wanamaker/model/priors.py) | [`tests/unit/test_model_spec.py`](../tests/unit/test_model_spec.py); recovery benchmarks in [`tests/benchmarks/test_synthetic_recovery.py`](../tests/benchmarks/test_synthetic_recovery.py) (15 % contribution, top-3 ranking) | Spec gated in CI. **Acceptance criterion not fully gated:** the BRD requires 95 % CI coverage between 90–99 % across simulation runs; only the contribution-recovery and ranking criteria are asserted. The CI-coverage acceptance is documented in the test docstring as "lives behind a future wrapper." |
| FR-3.2 | Spend-invariant channel handling (detect, fit prior-only, flag, no curve, no reallocation) | Detection: [`engine/pymc.py:824`](../src/wanamaker/engine/pymc.py) (`spend_invariant=bool(np.std(spend)==0.0)`); [`diagnose/checks.py::check_spend_variation`](../src/wanamaker/diagnose/checks.py) (warning-level CV check). Surfaced via [`engine/summary.py::ChannelContributionSummary.spend_invariant`](../src/wanamaker/engine/summary.py). Reallocation block: [`forecast/scenarios.py::_interpretation_sentence`](../src/wanamaker/forecast/scenarios.py); [`forecast/ramp.py::_spend_invariant_reallocations`](../src/wanamaker/forecast/ramp.py). | [`tests/unit/test_compare_scenarios.py`](../tests/unit/test_compare_scenarios.py) `TestSpendInvariantChannels::*`, `test_spend_invariant_sentence_takes_precedence`; [`tests/unit/test_ramp.py::TestSpendInvariantBlock`](../tests/unit/test_ramp.py); [`tests/unit/test_advisor_channel_flagging.py`](../tests/unit/test_advisor_channel_flagging.py) | Gated in CI across detection-flag-flow, scenario reallocation block, ramp up-front block, and advisor surfacing. **Caveat:** the engine sets `spend_invariant=True` only at strict zero standard deviation; the `check_spend_variation` diagnostic uses a CV threshold and emits a *warning* — these are two different conditions and the auditor should treat the warning as the user-facing flag and the engine flag as the FR-3.2 gate. |
| FR-3.3 | Lift-test calibration replaces ROI prior with informative prior | [`model/spec.py::ModelSpec.lift_test_priors`](../src/wanamaker/model/spec.py); [`engine/pymc.py::_coefficient_prior`](../src/wanamaker/engine/pymc.py) | [`tests/unit/test_lift_test_calibration.py`](../tests/unit/test_lift_test_calibration.py); end-to-end shift in [`tests/benchmarks/test_calibration_behavior.py`](../tests/benchmarks/test_calibration_behavior.py) (engine-gated) | Wiring gated in CI; posterior-shift acceptance engine-gated. |
| FR-3.4 | Geometric default; Weibull per-channel override; no auto-flip across refreshes | [`model/spec.py::ChannelSpec.adstock_family`](../src/wanamaker/model/spec.py); [`transforms/adstock.py`](../src/wanamaker/transforms/adstock.py) | [`tests/unit/test_adstock.py`](../tests/unit/test_adstock.py); [`tests/unit/test_model_spec.py`](../tests/unit/test_model_spec.py) | Gated in CI for transforms and config. **Caveat:** "no auto-flip across refreshes" is enforced by code structure (the family is read from YAML, never auto-changed); no test explicitly asserts a refresh keeps the configured family. |
| FR-3.5 | Three runtime tiers (Quick / Standard / Full) | [`config.py::RunConfig.runtime_mode`](../src/wanamaker/config.py); [`engine/pymc.py::PyMCEngine.fit`](../src/wanamaker/engine/pymc.py) maps mode → sampler params; [`cli.py`](../src/wanamaker/cli.py) `--runtime-mode` flag | [`tests/unit/test_cli_run_example.py`](../tests/unit/test_cli_run_example.py) (uses `runtime_mode="quick"` end-to-end through a stub engine); per-mode wiring asserted via the YAML fixtures in [`tests/unit/test_cli_forecast.py`](../tests/unit/test_cli_forecast.py), [`tests/unit/test_cli_refresh.py`](../tests/unit/test_cli_refresh.py), [`tests/unit/test_cli_report.py`](../tests/unit/test_cli_report.py); [`tests/unit/test_fit_command.py`](../tests/unit/test_fit_command.py) | Mode wiring gated in CI. **Performance SLAs not gated:** the BRD's "Standard < 30 minutes" target is not asserted by any automated test. |

### 2.4 Refresh Accountability (FR-4.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-4.1 | Versioned model runs with input data hash, config, posterior summary, timestamp | [`src/wanamaker/artifacts.py`](../src/wanamaker/artifacts.py); [`cli.py::fit`](../src/wanamaker/cli.py) writes manifest | [`tests/unit/test_artifact_schemas.py`](../tests/unit/test_artifact_schemas.py); end-to-end manifest verification in [`tests/test_network_isolation.py`](../tests/test_network_isolation.py) (engine-gated) | Schema gated in CI. |
| FR-4.2 | Refresh diff report comparing prior and current run | [`src/wanamaker/refresh/diff.py::compute_diff`](../src/wanamaker/refresh/diff.py); [`cli.py::refresh`](../src/wanamaker/cli.py) | [`tests/unit/test_refresh_diff.py`](../tests/unit/test_refresh_diff.py); [`tests/unit/test_cli_refresh.py`](../tests/unit/test_cli_refresh.py) | Gated in CI (orchestration uses a stub engine; no real-fit comparison in unit lane). |
| FR-4.3 | Movement classification (within_prior_ci / improved_holdout / unexplained / user_induced / weakly_identified) | [`src/wanamaker/refresh/classify.py`](../src/wanamaker/refresh/classify.py) | [`tests/unit/test_cli_refresh.py`](../tests/unit/test_cli_refresh.py) (classification on synthetic deltas); [`tests/benchmarks/test_refresh_stability.py`](../tests/benchmarks/test_refresh_stability.py) (engine-gated, asserts ≥ 90 % non-unexplained) | Classification logic gated in CI; NFR-5 acceptance engine-gated. |
| FR-4.4 | Light posterior anchoring (mixture prior, default `w=0.3`, named presets) | [`src/wanamaker/refresh/anchor.py`](../src/wanamaker/refresh/anchor.py); [`engine/pymc.py::_blend_lognormal`](../src/wanamaker/engine/pymc.py); [`config.py`](../src/wanamaker/config.py) `anchor_strength`; CLI `--anchor-strength` | [`tests/unit/test_anchor.py`](../tests/unit/test_anchor.py) (preset resolution, mixture math, edge cases); [`tests/unit/test_cli_refresh.py`](../tests/unit/test_cli_refresh.py) (CLI override paths) | Gated in CI. **Caveat:** `w=0.3` is a placeholder pending Phase 1 empirical tuning (BRD § FR-4.4). |
| FR-4.5 | Refresh-stability dimension in the Trust Card | [`src/wanamaker/trust_card/card.py`](../src/wanamaker/trust_card/card.py); [`trust_card/compute.py`](../src/wanamaker/trust_card/compute.py); [`reports/templates/trust_card.md.j2`](../src/wanamaker/reports/templates/trust_card.md.j2) | [`tests/unit/test_trust_card_compute.py`](../tests/unit/test_trust_card_compute.py); [`tests/unit/test_reports_render.py`](../tests/unit/test_reports_render.py) | Gated in CI. |

### 2.5 Outputs and Reports (FR-5.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-5.1 | Three decision modes (Explain / What-if / Refresh) with extrapolation visually distinguished | Mode 1: [`cli.py::report`](../src/wanamaker/cli.py); Mode 2: [`cli.py::forecast`](../src/wanamaker/cli.py); Mode 3: [`cli.py::refresh`](../src/wanamaker/cli.py). Extrapolation metadata in [`engine/summary.py`](../src/wanamaker/engine/summary.py) (`observed_spend_min/max`); rendered styling in [`reports/_charts.py`](../src/wanamaker/reports/_charts.py) | [`tests/unit/test_cli_report.py`](../tests/unit/test_cli_report.py); [`tests/unit/test_cli_forecast.py`](../tests/unit/test_cli_forecast.py); [`tests/unit/test_cli_refresh.py`](../tests/unit/test_cli_refresh.py); chart styling in [`tests/unit/test_charts.py`](../tests/unit/test_charts.py) | Gated in CI. **Caveat:** the BRD's "dashed line vs. solid" requirement for extrapolated regions is not asserted at the rendered-HTML level beyond the chart unit tests. |
| FR-5.2 | Scenario comparison ranks user-supplied plans with uncertainty + extrapolation warnings | [`src/wanamaker/forecast/scenarios.py::compare_scenarios`](../src/wanamaker/forecast/scenarios.py); [`forecast/posterior_predictive.py`](../src/wanamaker/forecast/posterior_predictive.py) | [`tests/unit/test_compare_scenarios.py`](../tests/unit/test_compare_scenarios.py) (ranking, input shapes, multi-period, extrapolation, spend-invariant) | Gated in CI. Tests use a deterministic stub engine. |
| FR-5.3 | Plain-English executive summary (deterministic Jinja2; no LLM) | [`src/wanamaker/reports/render.py`](../src/wanamaker/reports/render.py); [`reports/templates/executive_summary.md.j2`](../src/wanamaker/reports/templates/executive_summary.md.j2) | [`tests/unit/test_reports_render.py`](../tests/unit/test_reports_render.py); terminology guardrails in [`tests/unit/test_terminology_guardrails.py`](../tests/unit/test_terminology_guardrails.py) | Template branches and terminology gated in CI. **Caveat:** the BRD's user-test acceptance (≥ 10 testers, average Likert ≥ 4.0) is a manual research deliverable, not an automated test. |
| FR-5.4 | Trust Card with six v1 dimensions | [`src/wanamaker/trust_card/card.py`](../src/wanamaker/trust_card/card.py); [`trust_card/compute.py`](../src/wanamaker/trust_card/compute.py) (one builder per dimension); [`reports/trust_card_one_pager.py`](../src/wanamaker/reports/trust_card_one_pager.py) | [`tests/unit/test_trust_card_compute.py`](../tests/unit/test_trust_card_compute.py) (per-dimension classification); [`tests/unit/test_trust_card_one_pager.py`](../tests/unit/test_trust_card_one_pager.py) (rendering, jargon-free guarantees, print-CSS) | Gated in CI. |
| FR-5.5 | Experiment Advisor (high-uncertainty + significant-spend channel flagging) | [`src/wanamaker/advisor/channel_flagging.py::flag_channels`](../src/wanamaker/advisor/channel_flagging.py) | [`tests/unit/test_advisor_channel_flagging.py`](../tests/unit/test_advisor_channel_flagging.py) | Gated in CI. |
| FR-5.6 | Risk-adjusted ramps (ladder `{0.10, 0.25, 0.50, 0.75, 1.0}`; verdicts `proceed`/`stage`/`test_first`/`do_not_recommend`) | [`src/wanamaker/forecast/ramp.py`](../src/wanamaker/forecast/ramp.py) (`_RAMP_LADDER`, `recommend_ramp`); design + math docs at [`docs/internal/risk_adjusted_allocation.md`](internal/risk_adjusted_allocation.md) and [`docs/references/risk_adjusted_allocation_math.md`](references/risk_adjusted_allocation_math.md) | [`tests/unit/test_ramp.py`](../tests/unit/test_ramp.py) (every gate, every verdict, Kelly clamp, Trust Card cap, JSON round-trip); [`tests/unit/test_cli_recommend_ramp.py`](../tests/unit/test_cli_recommend_ramp.py); safety benchmark in [`tests/benchmarks/test_candidate_generation_safety.py`](../tests/benchmarks/test_candidate_generation_safety.py) | Gated in CI. |

### 2.6 Distribution / API / Layered Disclosure (FR-6.x, FR-7.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-6.1 | `pip install wanamaker` works on Linux, macOS, Windows | [`pyproject.toml`](../pyproject.toml) | Build + install on three OSes is a CI deliverable per the BRD; no in-repo test exercises this directly. | **Not currently gated** in this repo's pytest suite; expected to be a release-pipeline concern. |
| FR-6.2 | Docker container | Project-level Dockerfile (when present) | Out of pytest scope. | Not gated by pytest. |
| FR-6.3 | CLI surface (`diagnose`, `fit`, `report`, `forecast`, `compare-scenarios`, `refresh`) | [`src/wanamaker/cli.py`](../src/wanamaker/cli.py) (all `@app.command()` decorators) | [`tests/unit/test_cli.py::test_help_lists_public_commands`](../tests/unit/test_cli.py) asserts every public command appears in `--help`; per-command tests under `tests/unit/test_cli_*.py` | Gated in CI. |
| FR-6.4 | Python API mirrors the CLI | Public exports in `wanamaker.forecast`, `wanamaker.diagnose`, `wanamaker.engine`, etc. | Doctest-runnable examples in [`src/wanamaker/forecast/posterior_predictive.py`](../src/wanamaker/forecast/posterior_predictive.py), [`scenarios.py`](../src/wanamaker/forecast/scenarios.py), [`generator.py`](../src/wanamaker/forecast/generator.py), [`ramp.py`](../src/wanamaker/forecast/ramp.py) | Gated in CI via `--doctest-modules` (see [pyproject.toml](../pyproject.toml)). |
| FR-7.1–7.3 | Three-layer progressive disclosure (defaults / YAML / Python expert) | YAML schema in [`config.py`](../src/wanamaker/config.py); per-layer documentation in [`docs/api_reference.md`](api_reference.md) | Tested implicitly through quickstart end-to-end and CLI tests. | The "30 minutes to summary" acceptance is a manual onboarding deliverable, not a unit test. |

---

## 3. Privacy and Local Processing (FR-Privacy.x)

| FR | Claim | Implementation | Test | Status |
|---|---|---|---|---|
| FR-Privacy.1 | No network calls in core operations | See AGENTS.md Hard Rule 1 above | [`tests/test_no_network_in_core.py`](../tests/test_no_network_in_core.py) (static); [`tests/test_network_isolation.py`](../tests/test_network_isolation.py) (runtime, engine-gated) | Static gated in CI; runtime engine-gated. |
| FR-Privacy.2 | Local artifact storage in `.wanamaker/` (configurable, never modifies home) | [`config.py::RunConfig.artifact_dir`](../src/wanamaker/config.py) defaults to `.wanamaker`; all run-writing CLI commands respect it | [`tests/unit/test_artifact_schemas.py`](../tests/unit/test_artifact_schemas.py); end-to-end in [`tests/test_network_isolation.py`](../tests/test_network_isolation.py) | Gated in CI for the configured-path round-trip; **no automated test asserts the home directory is never touched** — that property follows from code structure (no path under the user home is ever computed). |
| FR-Privacy.3 | No telemetry by default | No telemetry SDK is importable from core code | Same `BANNED_MODULES` set in [`tests/test_no_network_in_core.py`](../tests/test_no_network_in_core.py) (`sentry_sdk`, `posthog`, `mixpanel`, `segment`) | Gated in CI. |
| FR-Privacy.4 | Documentation of data residency | [`docs/guides/privacy.md`](guides/privacy.md) | Documentation review only. | Manual. |

---

## 4. Non-Functional Requirements

| NFR | Claim | Implementation / Test | Status |
|---|---|---|---|
| NFR-1 | Standard tier under 30 minutes (150 weeks × 12 channels) | Implemented via runtime tiers (FR-3.5); no automated SLA test | **Not currently gated.** |
| NFR-2 | Reproducibility under same data + seed (`RTOL=1e-6` numerical agreement on every field of the engine-neutral posterior summary; not strict bit-for-bit, per PRD v0.5) | [`tests/test_reproducibility.py`](../tests/test_reproducibility.py) (engine-gated) | Engine-gated. Operational definition matches the PRD. |
| NFR-3 | Documentation quality (Installation / Quickstart / API ref / Analyst's Guide / CMO Guide / Privacy / Comparison) | Files in [`docs/`](.) ; mkdocs nav in [`mkdocs.yml`](../mkdocs.yml). README has the one-command end-to-end example. | Manual review; not pytest-gated. |
| NFR-4 | Honest failure modes | Hard fail on diagnostic blockers, structured `CheckResult` levels, no silent fallbacks | Tested via diagnostic and validation tests in §2.2 / §2.3 above. |
| NFR-5 | Refresh accountability targets (avg movement ≤ 10 %, max channel ≤ 25 %, ≥ 90 % classifiable) | [`tests/benchmarks/test_refresh_stability.py`](../tests/benchmarks/test_refresh_stability.py) (engine-gated; asserts the ≥ 90 % non-unexplained classification target on the medium preset) | Engine-gated. The 10 %/25 % movement caps are NFR-5 *targets*, not hard test assertions. |
| NFR-6 | Cross-platform (Linux, macOS Intel + Apple Silicon, Windows) | Release-pipeline concern; Wanamaker runs on Windows in this dev environment. | Not pytest-gated. |
| NFR-7 | Benchmark datasets shipped | [`benchmark_data/`](../benchmark_data/) directory; loaders in [`src/wanamaker/benchmarks/loaders.py`](../src/wanamaker/benchmarks/loaders.py) | [`tests/unit/test_benchmark_loaders.py`](../tests/unit/test_benchmark_loaders.py) gates schema and shapes. |

---

## 5. Verification recipes

Copy-paste reproductions for the most common audit questions.

```bash
# 1. No network library is reachable from any core module.
pytest tests/test_no_network_in_core.py -v

# 2. Worked-example math contracts (transforms, priors, ramp).
pytest src/wanamaker -q --doctest-modules

# 3. FR-3.2 spend-invariant handling end to end.
pytest tests/unit/test_compare_scenarios.py -k spend_invariant -v
pytest tests/unit/test_ramp.py::TestSpendInvariantBlock -v
pytest tests/unit/test_advisor_channel_flagging.py -k invariant -v

# 4. FR-2.2 risk categories on the named benchmark fixtures.
pytest tests/unit/test_diagnostic_benchmarks.py -v

# 5. FR-4.4 anchoring math and CLI override paths.
pytest tests/unit/test_anchor.py tests/unit/test_cli_refresh.py -v

# 6. FR-5.6 ramp ladder, verdicts, gate failures.
pytest tests/unit/test_ramp.py -v

# 7. NFR-2 reproducibility on a real PyMC fit (slow; opt-in).
WANAMAKER_RUN_ENGINE_TESTS=1 pytest tests/test_reproducibility.py -v

# 8. NFR-5 refresh stability on the benchmark dataset (slow; opt-in).
WANAMAKER_RUN_ENGINE_TESTS=1 pytest tests/benchmarks/test_refresh_stability.py -v
```

---

## 6. What is *not* currently gated in CI

This list is the most useful page in the doc for an honest reviewer. Each entry is a published claim where the implementation exists but no automated test currently asserts the published acceptance threshold.

- **FR-1.1 acceptance:** "accepts CSVs from at least three different real-world data shapes." Schema parsing is tested; "three real-world shapes" is a manual-verification deliverable.
- **FR-2.2 acceptance:** structural-break detection within ±2 weeks of the true break point. Detection is asserted; the ±2 week tolerance is not.
- **FR-3.1 acceptance:** 95 % credible-interval coverage between 90 % and 99 % across simulation runs. Listed as future work in the test file; only contribution recovery and top-3 ranking are asserted.
- **FR-3.4 invariant:** "no auto-flip of adstock family across refreshes." Enforced by code structure, not by an explicit refresh test.
- **FR-3.5 acceptance:** "Standard mode under 30 minutes on a modern laptop CPU." Mode selection is tested; runtime SLA is not.
- **FR-5.3 acceptance:** ≥ 10 testers, average Likert ≥ 4.0 on three dimensions. A research deliverable, not an automated test.
- **FR-Privacy.2 invariant:** "home directory is never modified." Implied by code structure, not asserted by a test.
- **NFR-1 / NFR-6:** Performance and cross-platform guarantees are release-pipeline concerns, not pytest gates.
- **NFR-5 caps:** the 10 % / 25 % movement caps in NFR-5 are targets, not hard assertions; only the ≥ 90 % classifiable target is gated (engine-gated).

If any of these gaps matters for your audit, the test that *would* close it is usually one short PR away. Open an issue against the corresponding FR and the project will fix the gap rather than rewrite the claim.

---

## 7. When this guide drifts from the code

This file lives in the repo, not in a planning doc, exactly so that a stale entry is a code-review concern and not just a documentation tidying task. If you find an entry that points at a moved file, a renamed test, or an out-of-date acceptance status, treat it the same way you would treat a wrong docstring: open a PR. The maintainers care more about not over-claiming than about keeping the surface elegant.
