# Wanamaker -- Architecture and Technical Specification

**Audience:** Developers and AI coding assistants working on the codebase.
**Companion documents:** [`docs/wanamaker_brd_prd.md`](wanamaker_brd_prd.md) (what and why), [`AGENTS.md`](https://github.com/mbagalman/wanamaker/blob/main/AGENTS.md) (hard rules and coding conventions).
**Status:** Active pre-1.0 development. PyMC backend, transforms, model spec,
config, artifacts, diagnose, refresh diff, forecasting, scenario comparison,
risk-adjusted ramp recommendations, Trust Card, reports, and the public CLI
workflow are implemented. Active work is on hardening benchmarks, release
packaging, and deferred supporting-role xgboost paths.

---

## 1. Design Philosophy

Three principles shape every architectural decision:

1. **Trust is the product.** The module structure, naming, and data flow are all oriented around making the model's behavior auditable and reproducible -- not convenient or clever. When in doubt, choose transparency over elegance.
2. **Engine-isolated core.** PyMC is the selected Bayesian engine, but feature code must still interface through `wanamaker.engine` and never import PyMC directly.
3. **Local-first, always.** No module in the core code path may import an HTTP client, telemetry library, or LLM SDK. This is enforced by a CI test (`tests/test_no_network_in_core.py`), not just convention.

A fourth principle governs language rather than structure: **decision-support, not optimizer-grade promises.** User-facing output uses cautious phrasing ("candidate scenarios", "risk-adjusted ramp", "largest defensible move") and avoids phrases that imply the model has produced a globally optimal answer ("optimized budget", "optimal allocation", "best budget", "guaranteed lift", "maximize ROI" as a promise). The full list lives in `AGENTS.md` (at the repository root, under "Product terminology") and is enforced by `tests/unit/test_terminology_guardrails.py` against every report template and helper module.

---

## 2. Repository Layout

```
wanamaker/
+-- src/wanamaker/          # installable package
|   +-- __init__.py         # version only
|   +-- cli.py              # typer CLI; thin dispatch layer
|   +-- config.py           # YAML loading + pydantic validation
|   +-- seeding.py          # reproducibility discipline (NFR-2)
|   +-- artifacts.py        # local artifact storage (.wanamaker/ layout)
|   +-- data/               # CSV I/O and channel taxonomy
|   +-- diagnose/           # pre-flight readiness diagnostic
|   +-- engine/             # Bayesian engine abstraction (Protocol + typed summaries)
|   +-- transforms/         # adstock and saturation canonical formulas
|   +-- model/              # engine-agnostic model specification and priors
|   +-- refresh/            # versioning, diff, anchoring, movement classification
|   +-- forecast/           # posterior predictive and scenario comparison
|   +-- trust_card/         # credibility dimension data structures
|   +-- advisor/            # experiment advisor (channel flagging)
|   +-- reports/            # Jinja2 templates and rendering
|   +-- benchmarks/         # benchmark dataset loaders (NFR-7)
|   +-- _xgboost_aux/       # xgboost preview_forecast and xgb_crosscheck only
+-- tests/
|   +-- test_no_network_in_core.py  # CI gate: import-time network guardrail
|   +-- unit/               # pure-function tests
|   +-- integration/        # end-to-end command tests
|   +-- benchmarks/         # benchmark-driven acceptance tests (slow)
+-- docs/
|   +-- wanamaker_brd_prd.md
|   +-- architecture.md     # this file
|   +-- decisions/          # architectural decision records
|   |   +-- 0001-bayesian-engine-selection.md
|   +-- references/
|       +-- adstock_and_saturation.md   # canonical transform reference
+-- benchmark_data/         # synthetic + public datasets (Phase 0+)
+-- examples/               # runnable quickstart examples (Phase 2+)
+-- pyproject.toml
+-- AGENTS.md
+-- LICENSE
```

---

## 3. Module Map and Responsibilities

### 3.1 `cli.py` -- Entry Point

The CLI is intentionally thin. Each command resolves configuration, then delegates immediately to the appropriate subpackage. No business logic lives here.

| Command | Delegates to |
|---|---|
| `wanamaker diagnose <data.csv> [--config]` | `wanamaker.diagnose` |
| `wanamaker fit --config <config.yaml>` | `wanamaker.model`, `wanamaker.engine` |
| `wanamaker report --run-id <id>` | `wanamaker.reports` |
| `wanamaker run --example public_benchmark` | `wanamaker.diagnose`, `wanamaker.model`, `wanamaker.reports` |
| `wanamaker forecast --run-id <id> --plan <plan.csv>` | `wanamaker.forecast` |
| `wanamaker compare-scenarios --run-id <id> --plans ...` | `wanamaker.forecast` |
| `wanamaker recommend-ramp --run-id <id> --baseline <base.csv> --target <target.csv>` | `wanamaker.forecast` |
| `wanamaker refresh --config <config.yaml>` | `wanamaker.refresh`, `wanamaker.engine` |

**`diagnose` accepts an optional `--config`:** Without it, only checks that do not require column knowledge run (history length, date regularity, structural breaks), with a warning. With it, all checks run including spend variation and collinearity. This matches the progressive disclosure model: a user exploring a new CSV does not need a config yet.

**`--skip-validation` on `fit`:** An expert-only flag, hidden from help output. When set, the validation step is bypassed, and the skip is recorded in the run manifest and flagged on the Trust Card. Not intended for production analyses; exists for automated pipelines and test harnesses.

**`--anchor-strength` on `refresh`:** Accepts named presets (`none`, `light`, `medium`, `heavy`) or a float; resolution lives in `refresh.anchor.resolve_anchor_weight`.

Built with [typer](https://typer.tiangolo.com/).

### 3.2 `config.py` -- Configuration Contract

All user-facing configuration is validated here via pydantic before any other module sees it. The schema is the Layer 2 interface (guided YAML configuration) in the three-layer progressive disclosure architecture.

**Key types:**

```
WanamakerConfig
+-- DataConfig          csv_path, date_column, target_column,
|                       spend_columns, control_columns, lift_test_csv
+-- ChannelConfig[]     name, category, adstock_family per channel
+-- RefreshConfig       anchor_strength (preset or float)
+-- RunConfig           seed, runtime_mode, artifact_dir
```

`extra="forbid"` is set on all models -- unknown YAML keys are a validation error, not silently ignored. This prevents config drift.

**Runtime modes** (`quick` / `standard` / `full`) control Bayesian sampler effort only; the model specification is identical across all three tiers. Note: "quick mode" refers exclusively to sampler effort. The xgboost preview and cross-check paths use the labels `xgb_preview` and `xgb_crosscheck` -- these are distinct concepts; see Section 3.16.

### 3.3 `seeding.py` -- Reproducibility Discipline

The single source of truth for all random state. Two functions:

- `make_rng(seed: int) -> np.random.Generator` -- creates a fresh generator, never touching global `np.random` state.
- `derive_seed(parent_seed, label) -> int` -- produces a deterministic child seed via BLAKE2b hash. Labels in use: `"xgb_preview"`, `"xgb_crosscheck"`, `"posterior_predictive"`.

**Rule:** no module may call `np.random.seed()`, `random.seed()`, or read global random state. All randomness flows from the top-level `config.run.seed`, passed explicitly down the call stack.

### 3.4 `artifacts.py` -- Local Artifact Storage

Owns the `.wanamaker/` directory layout. Every model fit writes a versioned run directory:

```
.wanamaker/
+-- runs/
    +-- <run_id>/
        +-- manifest.json       run_fingerprint, seed, engine, schema versions, skip_validation flag
        +-- config.yaml         snapshot of the run config
        +-- data_hash.txt       SHA-256 of the input CSV
        +-- posterior.nc        posterior draws (engine-native format, e.g. NetCDF via ArviZ)
        +-- summary.json        PosteriorSummary (versioned envelope)
        +-- trust_card.json     TrustCard (versioned envelope; persisted for auditability)
        +-- refresh_diff.json   RefreshDiff vs. prior run (present only on refresh; versioned envelope)
        +-- timestamp.txt       ISO-8601 UTC fit timestamp
        +-- engine.txt          engine name + version
```

**Versioned artifact envelopes:** Every JSON artifact (`summary.json`, `trust_card.json`, `refresh_diff.json`) uses the envelope format `{"schema_version": N, "payload": {...}}`. Schema version constants (`SUMMARY_SCHEMA_VERSION`, `TRUST_CARD_SCHEMA_VERSION`, `REFRESH_DIFF_SCHEMA_VERSION`, `MANIFEST_SCHEMA_VERSION`) live in `artifacts.py`. The deserializers (`deserialize_summary`, `deserialize_trust_card`, `deserialize_refresh_diff`, `load_manifest`) validate the version before parsing and raise `ValueError` with a clear message on mismatch. Bump the constant and update the deserializer when a format change is backward-incompatible.

**Two identity concepts:**

- **`run_fingerprint`:** A deterministic BLAKE2b hash of `(data_hash, config_hash, package_version, engine_name, engine_version, seed)`. Two runs with the same fingerprint used identical analytical inputs. Used by refresh/diff to detect "same analytical setup." Stored in `manifest.json`.
- **`run_id`:** The storage key. Derived from `run_fingerprint` plus a UTC timestamp. Unique per execution. Repeated runs of the same `(data, config, seed)` produce distinct run_ids but identical run_fingerprints -- they are separate artifacts but analytically equivalent.

This separation means the refresh diff can reliably answer "did the analytical setup change between these two runs?" without ambiguity.

`artifacts.py` does not own the contents of `posterior.nc` or `summary.json` -- those formats belong to `engine` and `refresh` respectively.

### 3.5 `data/` -- CSV Interface and Channel Taxonomy

| File | Responsibility |
|---|---|
| `io.py` | Loads and validates the input CSV; parses dates; raises `ValueError` with actionable messages on bad input. No imputation -- decisions about missing data belong to the diagnostic step. |
| `taxonomy.py` | Defines the 10 default channel categories. Category-driven default priors live in `model/priors.py`. |

**The CSV contract is a user-facing API.** Do not change input/output formats without revisiting BRD/PRD Section 5.1.

### 3.6 `diagnose/` -- Pre-Flight Readiness Diagnostic

Implements `wanamaker diagnose` (FR-2). Runs before fitting; cannot be silently skipped (see `--skip-validation` above for the explicit expert escape hatch).

| File | Responsibility |
|---|---|
| `readiness.py` | Data structures: `ReadinessLevel` enum (4 values), `CheckSeverity` enum, `CheckResult` dataclass, `ReadinessReport` dataclass. |
| `checks.py` | Pure functions, one per check. Each takes a DataFrame and returns `CheckResult` or `list[CheckResult]`. |

**Checks to implement (FR-2.2):**

| Check | Needs config? | Trigger | Severity |
|---|---|---|---|
| `history_length` | No | < 52 weeks warning, < 26 weeks blocker | WARNING / BLOCKER |
| `date_regularity` | No | gaps or duplicate dates | WARNING / BLOCKER |
| `structural_breaks` | No | change-point detection on residuals | INFO / WARNING |
| `target_stability` | No | extreme outliers in target | WARNING |
| `missing_values` | Yes | any NaN in target or spend columns | WARNING / BLOCKER |
| `spend_variation` | Yes | coefficient of variation below threshold | WARNING |
| `collinearity` | Yes | paid-vs-paid or paid-vs-control correlation | WARNING |
| `variable_count` | Yes | 1:10 rule (predictors vs. observations) | WARNING |
| `target_leakage` | Yes | control correlation to target >= 0.95 | WARNING |

**Readiness levels** (never a numeric score):

| Level | Meaning |
|---|---|
| `ready` | No blockers, no significant warnings |
| `usable_with_warnings` | Model can fit; specific outputs should be treated with caution |
| `diagnostic_only` | Model can fit but should not drive decisions |
| `not_recommended` | Fundamental data problems; specific issues identified |

### 3.7 `engine/` -- Bayesian Engine Abstraction

The most important architectural boundary in the codebase. All feature code must import from `wanamaker.engine`, never from `pymc`, `numpyro`, or `cmdstanpy` directly.

**Current state:** Protocol, typed summaries, and PyMC backend all implemented. See `docs/decisions/0001-bayesian-engine-selection.md` for the engine selection rationale.

| File | Responsibility |
|---|---|
| `base.py` | `Posterior`, `FitResult` dataclasses; `Engine` Protocol. |
| `summary.py` | Typed, engine-neutral posterior summary types (see below). |

**`Engine` Protocol interface:**

```python
class Engine(Protocol):
    name: str  # "pymc"

    def fit(self, model_spec: ModelSpec, data: DataFrame,
            seed: int, runtime_mode: str) -> FitResult: ...

    def posterior_predictive(self, posterior: Posterior,
                              new_data: Any, seed: int) -> Any: ...
```

**`Posterior`** wraps the engine-native object in a stable, opaque container. Its `raw` field holds the engine-native object (e.g., ArviZ `InferenceData` for PyMC) for expert access. Core modules must not reach into `Posterior.raw`; they consume `PosteriorSummary` instead.

**`FitResult`** carries the `Posterior` plus a `diagnostics` dict (R-hat, effective sample size, divergences). The backend is responsible for also producing a `PosteriorSummary` and writing it to `summary.json`.

**Typed posterior summary layer (`engine/summary.py`):**

All downstream modules (refresh/diff, trust_card, reports, forecast, advisor) depend on these typed summaries rather than the opaque `Posterior.raw`. This is the stable contract that lets the engine backend be swapped without rewriting downstream code.

| Type | Used by |
|---|---|
| `ParameterSummary` | refresh/diff, trust_card (convergence), reports |
| `ChannelContributionSummary` | reports (waterfall), forecast, advisor |
| `PredictiveSummary` | forecast, trust_card (holdout accuracy) |
| `ConvergenceSummary` | trust_card (convergence dimension) |
| `PosteriorSummary` | top-level container; written to `summary.json` |

**Selected backend:** PyMC. The backend lives at `engine/pymc.py` and implements the `Engine` Protocol. NumPyro/JAX and Stan were rejected for the primary modeling role because their installation and toolchain risks are less compatible with Wanamaker's target persona.

### 3.8 `transforms/` -- Canonical Mathematical Formulas

These modules own the **canonical mathematical formulas** for adstock and saturation -- not the execution of those transforms inside a model fit. This distinction matters:

- In a Bayesian model, `decay`, `ec50`, and `slope` are **sampled parameters** -- they have priors and posteriors. The transforms must be applied *inside* the engine backend's probabilistic program graph (where the sampler can differentiate through them), not as a fixed preprocessing step on the input data.
- The functions in `transforms/` are canonical implementations used for: unit tests with known inputs/outputs, documentation, benchmark comparisons, fixed-parameter previews (e.g., prior predictive checks), and backend validation.
- Each engine backend (e.g., `engine/pymc.py`) consumes these formulas as the definition of correctness and reimplements them in the backend's native tensor operations.

| File | Function | Status |
|---|---|---|
| `adstock.py` | `geometric_adstock(spend, decay)` | **Implemented** |
| `adstock.py` | `weibull_adstock(spend, shape, scale)` | **Implemented** |
| `saturation.py` | `hill_saturation(spend, ec50, slope)` | **Implemented** |

**Geometric adstock:** `A_t = X_t + decay * A_{t-1}`, `A_{-1} = 0`. Decay in [0, 1). Reference: Hanssens, Parsons & Schultz, *Market Response Models* (2nd ed., 2001), ch. 10.

**Weibull adstock:** Per Jin et al. (Google, 2017) sec. 3. Per-channel override path only; does not auto-flip across refreshes (FR-3.4).

**Hill saturation:** `f(x) = x^slope / (x^slope + ec50^slope)`. EC50 = spend at half-maximum response. Reference: Jin et al. (2017) sec. 4.

Before implementing or modifying any transform, read [`docs/references/adstock_and_saturation.md`](references/adstock_and_saturation.md).

### 3.9 `model/` -- Engine-Agnostic Model Specification

Produces a `ModelSpec` -- a pure data description of the model that any backend can consume. No sampling, no engine imports here.

| File | Responsibility |
|---|---|
| `spec.py` | `ChannelSpec` (name, category, adstock_family) and `ModelSpec` (channels list, control_columns). Frozen dataclasses. |
| `priors.py` | `default_priors_for_category(category)` -- returns prior shapes keyed by category. To be populated in Phase 0 with PRD-cited values. |

**`ModelSpec` schema** (all fields implemented):

| Field | Purpose |
|---|---|
| `channels: list[ChannelSpec]` | Per-channel name, category, adstock family |
| `control_columns: list[str]` | Non-media predictors |
| `date_column: str` | Name of the date column |
| `target_column: str` | Name of the target metric column |
| `frequency: str` | `"weekly"` (v1 only) |
| `channel_priors: dict[str, PriorSpec]` | Resolved prior objects per channel (overrides defaults) |
| `lift_test_priors: dict[str, LiftPrior]` | Informative priors derived from lift test CSVs |
| `holdout_config: HoldoutConfig | None` | Hold-out period for Trust Card accuracy check |
| `seasonality: SeasonalitySpec | None` | Trend and seasonality treatment |
| `anchor_priors: dict[str, AnchoredPrior] | None` | Mixture priors from previous run (refresh path) |
| `spend_invariant_channels: set[str]` | Channels where saturation is fixed at prior median |
| `runtime_mode: str` | Passed through to engine for sampler effort |

This schema should be designed before the backend spike so the first backend implementation doesn't harden around an accidental minimum.

**Separation of concerns:** `ModelSpec` is data; the engine translates it into a concrete probabilistic program. This means the same model description can be rendered by any backend without leakage.

`build_model_spec(config)` in `model/builder.py` translates a `WanamakerConfig` into a `ModelSpec`, resolving channel priors from the taxonomy defaults. This is the function `cli.py` calls before handing off to the engine.

### 3.10 `refresh/` -- Refresh Accountability (Headline Feature)

Three components implementing FR-4:

| File | Responsibility |
|---|---|
| `anchor.py` | Posterior anchoring as a mixture prior. `ANCHOR_PRESETS` dict, `resolve_anchor_weight(value)`. |
| `diff.py` | `ParameterMovement` (with `movement_class`), `RefreshDiff` dataclasses; `compute_diff(previous_summary, current_summary, prev_run_id, curr_run_id) -> RefreshDiff`. Both summary arguments are `PosteriorSummary` from `engine/summary.py`. Diffs both scalar parameters and channel contributions (named `channel.<name>.contribution`). |
| `classify.py` | `MovementClass` enum; `classify_movement(prev_hdi, curr_mean, curr_hdi)` — priority order: WEAKLY_IDENTIFIED → WITHIN_PRIOR_CI → UNEXPLAINED; `unexplained_fraction(movements)` — headline refresh stability metric. |

**Anchoring math (FR-4.4):**
```
Prior_new(theta) = (1 - w) * Prior_default(theta) + w * Posterior_previous(theta)
```
Applied to *marginal* posteriors of channel-level parameters only (ROI, adstock half-life, saturation slope, saturation EC50). Control variable coefficients and baseline parameters re-estimate freely. Default `w = 0.3` (placeholder; to be tuned against the refresh stability benchmark in Phase 1).

**Named presets:**

| Preset | w | When to use |
|---|---|---|
| `none` | 0.0 | Pure re-estimation; full diff transparency only |
| `light` | 0.2 | Business has changed recently |
| `medium` | 0.3 | Default; balanced starting point |
| `heavy` | 0.5 | Stable business; maximum stabilization |

**Movement classification (FR-4.3):**

| Class | Meaning |
|---|---|
| `within_prior_ci` | Small movement; expected variation |
| `improved_holdout` | Large movement but new model fits better; trustworthy update |
| `unexplained` | Large movement with no diagnostic explanation; trust risk |
| `user_induced` | Driven by user config/prior change; not a model failure |
| `weakly_identified` | Channel flagged by Trust Card as poorly identified; expected instability |

The fraction classified as `unexplained` is the headline Trust Card refresh stability metric. NFR-5 target: >= 90% classified as non-`unexplained` on the refresh stability benchmark dataset.

### 3.11 `forecast/` -- Posterior Predictive and Scenario Comparison

| File | Responsibility |
|---|---|
| `posterior_predictive.py` | `forecast(posterior, future_spend, seed)` -- forward simulation given a budget plan. Returns `PredictiveSummary`. |
| `scenarios.py` | `compare_scenarios(posterior, plans, seed)` -- ranks 2-3 user-supplied plans with uncertainty. Flags extrapolation beyond historical observed spend. |

All forecast and scenario logic must consume `PosteriorSummary` (not a bare `Posterior`). `ChannelContributionSummary.observed_spend_min` and `observed_spend_max` provide the historical spend range needed to detect extrapolation. `spend_invariant=True` channels are excluded from reallocation recommendations (FR-3.2).

v1 ships scenario comparison (user-driven). Constrained inverse optimization ("how do I hit X?") is deferred to v1.1.

### 3.12 `trust_card/` -- Credibility Assessment

| File | Responsibility |
|---|---|
| `card.py` | `TrustStatus` enum (pass/moderate/weak), `TrustDimension` dataclass, `TrustCard` dataclass. |

**v1 dimensions (FR-5.4):**

| Dimension | Data source |
|---|---|
| `convergence` | `ConvergenceSummary` from `FitResult` |
| `holdout_accuracy` | `PredictiveSummary` on held-out period vs. actuals |
| `refresh_stability` | Fraction of `unexplained` movements in `RefreshDiff` (present only when there is a prior run) |
| `prior_sensitivity` | Posterior shift under perturbed priors |
| `saturation_identifiability` | Per-channel; flags spend-invariant channels |
| `lift_test_consistency` | Posterior vs. lift-test estimate alignment (present only when calibration data provided) |

The Trust Card is the connective tissue between model output and recommendations: `weak` on any dimension feeds specific hedged language into the executive summary templates and specific warnings into scenario comparison.

### 3.13 `advisor/` -- Experiment Advisor

Minimal v1: identifies channels where posterior credible interval is wide and spend is large enough that better information would meaningfully change recommendations. Output: a prioritized list of `ChannelFlag` objects with rationale.

v1.1 will add experiment design (geo holdout vs. budget split, sample size, duration, geo selection).

### 3.14 `reports/` -- Rendering

| File | Responsibility |
|---|---|
| `render.py` | `render_executive_summary(context)`, `render_trust_card(context)` via Jinja2 `PackageLoader`. Templates ship inside the wheel. |
| `templates/executive_summary.md.j2` | Plain-English executive summary (FR-5.3). Language adjusts by confidence level via Jinja2 conditionals. |
| `templates/trust_card.md.j2` | Trust card template. |

**Hard Rule:** No LLM calls for output generation. If a template feels limiting, expand the template logic. Every word in every report is deterministic given the same `PosteriorSummary`.

**Visualization ownership:** `reports/` owns the chart contracts. Response curve extrapolation warnings (the solid/dashed boundary per FR-5.1) require a `SpendRange` struct carrying `(observed_min, observed_max)` per channel -- this should be added to `ChannelContributionSummary` or as a separate output of the fit. Static charts use matplotlib/seaborn; interactive HTML is a Phase 2 decision pending user testing feedback.

### 3.15 `benchmarks/` -- Benchmark Dataset Loaders

Loaders for the eight named benchmark datasets (NFR-7). The datasets live in `benchmark_data/` at the repo root. Two artifact types:

1. **Committed CSVs + ground-truth JSON** — versioned in the repo, loaded by these functions.
2. **Generation scripts** — `benchmark_data/generate_synthetic_ground_truth.py` is the canonical example; regenerate with `python benchmark_data/generate_synthetic_ground_truth.py` from the repo root.

The synthetic ground-truth dataset (12 channels, 150 weeks) is committed and ready. `load_synthetic_ground_truth()` returns `(DataFrame, ground_truth_dict)` where the dict carries `date_column`, `target_column`, `spend_columns`, per-channel parameters, and `top_3_channels`.

| Dataset | Used to validate |
|---|---|
| `synthetic_ground_truth` | Model recovery (FR-3.1 acceptance) |
| `public_example` | Quickstart, tutorials |
| `refresh_stability` | Refresh accountability (NFR-5) |
| `low_variation_channel` | Spend-invariant handling (FR-3.2) |
| `collinearity` | Diagnostic collinearity warning (FR-2.2) |
| `lift_test_calibration` | Lift-test calibration math (FR-1.3, FR-3.3) |
| `target_leakage` | Leakage detection (FR-2.2) |
| `structural_break` | Break detection (FR-2.2) |

### 3.16 `_xgboost_aux/` -- xgboost in a Supporting Role

Leading underscore signals internal support, not a user-facing API. xgboost has two and only two uses, identified by their internal labels (also used as `seeding.derive_seed` labels):

| Label | Purpose |
|---|---|
| `xgb_preview` | Fast tree-based forecast with conformal intervals. Explicit label: "forecast preview." Not ROI. |
| `xgb_crosscheck` | Independent tree fit. If Bayesian and tree forecasts disagree substantially, Trust Card raises a flag. |

**Naming note:** These paths must never be called "quick mode." `runtime_mode="quick"` refers exclusively to Bayesian sampler effort. These are distinct concepts that happen to both be fast.

xgboost does **not** produce ROI estimates, saturation curves, adstock parameters, or channel-level inference of any kind (AGENTS.md Hard Rule 3).

---

## 4. Data Flow

The full workflow from CSV to executive summary:

```
User's CSV + YAML config
        |
        v
+--------------+
|  config.py   |  Validate YAML -> WanamakerConfig
+------+-------+
       |
       v
+--------------+
|   data/io    |  Load CSV, parse dates, sort -> DataFrame
+------+-------+
       |
       v
+--------------+
|  diagnose/   |  Battery of checks -> ReadinessReport
|              |  (required before fit; --skip-validation records in manifest)
+------+-------+
       |  ReadinessReport.level != NOT_RECOMMENDED
       v
+------------------+
|   model/         |  WanamakerConfig + taxonomy -> ModelSpec
|                  |  (channels, priors, adstock families, lift-test priors,
|                  |   anchor_priors if prior run exists)
+------+-----------+
       |
       v
+------------------+      +------------------------+
|  refresh/        |----->|  Prior_new = mixture   |  (if prior run exists)
|  anchor.py       |      |  of default + previous |
+------+-----------+      +------------------------+
       |
       v
+------------------+
|  engine/         |  Engine.fit(model_spec, data, seed, runtime_mode)
|  (backend)       |  Transforms applied INSIDE the probabilistic program
|                  |  (decay, ec50, slope are sampled parameters)
|                  |  -> FitResult(Posterior, diagnostics)
|                  |  -> PosteriorSummary (written to summary.json)
+------+-----------+
       |
       +------------------------------------------+
       |                                          |
       v                                          v
+--------------+                       +------------------+
|  artifacts/  |  persist run to       |  refresh/diff    |
|              |  .wanamaker/          |  (if prior run)  |
|              |  manifest.json has    |  PosteriorSummary|
|              |  run_fingerprint +    |  -> RefreshDiff  |
|              |  run_id               +--------+---------+
+--------------+                                |
       |                                        |
       +--------------------+-------------------+
                            |
                            v
                   +--------------+
                   |  trust_card/ |  PosteriorSummary + RefreshDiff -> TrustCard
                   +------+-------+
                          |
               +----------+----------+
               |                     |
               v                     v
       +--------------+      +--------------+
       |  forecast/   |      |  advisor/    |
       |  scenarios   |      |  flagging    |
       +------+-------+      +------+-------+
              |                     |
              +----------+----------+
                         |
                         v
                 +--------------+
                 |  reports/    |  TrustCard + PosteriorSummary + RefreshDiff
                 |              |  -> Jinja2 templates
                 |              |  -> executive_summary.md
                 |              |  -> trust_card.md
                 +--------------+
```

**Key point on transforms:** The `transforms/` module is NOT shown in the fit path above because adstock and saturation are applied *inside* the engine backend as part of the probabilistic program. The transforms module provides the canonical formula implementations for tests and documentation, not a preprocessing step.

**Seeding flow:** `config.run.seed` is the single root. `seeding.derive_seed(parent, "xgb_preview")`, `seeding.derive_seed(parent, "xgb_crosscheck")`, `seeding.derive_seed(parent, "posterior_predictive")` give independent components their own deterministic streams.

---

## 5. Three-Layer Progressive Disclosure

The user-facing API has three layers (FR-7), each providing full access:

| Layer | Interface | What it exposes |
|---|---|---|
| 1 -- Defaults only | CLI with a CSV and a minimal YAML | Date column, target column, nothing else required |
| 2 -- Guided config | YAML with full `WanamakerConfig` schema | Channel categories, business constraints, lift test path, runtime mode, anchor strength |
| 3 -- Expert overrides | Python API | Direct `ModelSpec`, prior overrides, custom transforms, raw `Posterior` objects, sampler tuning |

A new user should be able to go from source checkout today, and eventually
`pip install wanamaker` after release, to a complete executive summary in under
30 minutes following only the quickstart documentation (FR-7.1 acceptance
criterion).

---

## 6. Artifact Storage and Privacy

All outputs go to `.wanamaker/` in the project directory (FR-Privacy.2). Nothing is written to the user's home directory or any system path by default. The `artifact_dir` can be overridden via `config.run.artifact_dir` or a CLI flag.

**No network calls in any core code path.** Enforced by `tests/test_no_network_in_core.py`, which walks the import graph of all core subpackages at import time and fails the build if any banned module appears. This is an **import-time guardrail** -- it catches modules imported at module load but does not catch optional or lazy imports inside functions until those functions execute. The companion runtime gate, `tests/test_network_isolation.py`, exercises the primary core workflow in a network-isolated environment when engine tests are enabled.

**`manifest.json`** is the run manifest written to every run directory. It records:
- `run_id` (unique per execution)
- `run_fingerprint` (deterministic hash; see Section 3.4)
- `seed`
- `engine_name` and `engine_version`
- `package_version`
- `summary_schema_version` (for forward-compatibility)
- `skip_validation` (boolean; true if `--skip-validation` was passed)
- `readiness_level` (the diagnostic result, or `"skipped"`)

---

## 7. What Is Built vs. Deferred

| Module | Status | Phase |
|---|---|---|
| `cli.py` | **Implemented** -- `diagnose`, `fit`, `report`, `run`, `forecast`, `compare-scenarios`, `recommend-ramp`, and `refresh` | Done |
| `config.py` | **Implemented** -- strict pydantic schema | Done |
| `seeding.py` | **Implemented** -- `make_rng` and `derive_seed` | Done |
| `artifacts.py` | **Implemented** -- directory layout, fingerprint/run_id, versioned envelopes for summary, Trust Card, refresh diff, and ramp recommendation artifacts, `list_runs`, `load_manifest` | Done |
| `data/io.py` | **Implemented** -- input CSV and lift-test CSV loading/validation | Done |
| `data/taxonomy.py` | **Implemented** -- channel category names | Done |
| `diagnose/readiness.py` | **Implemented** -- all data structures | Done |
| `diagnose/checks.py` | **Implemented** -- all 9 checks including structural breaks, collinearity, spend variation, target leakage | Done |
| `engine/base.py` | **Implemented** -- Protocol, `Posterior`, `FitResult` | Done |
| `engine/summary.py` | **Implemented** -- typed summary objects, channel summaries, posterior predictive draws | Done |
| `engine/pymc.py` | **Implemented** -- PyMC backend; HDI, data-adaptive priors, period labels, convergence, posterior predictive | Done |
| `transforms/adstock.py` | **Implemented** -- `geometric_adstock` (validated), `weibull_adstock`, `half_life_to_decay` | Done |
| `transforms/saturation.py` | **Implemented** -- `hill_saturation` (validated, log-space stable) | Done |
| `model/spec.py` | **Implemented** -- full schema: `ChannelSpec`, `LiftPrior`, `HoldoutConfig`, `SeasonalitySpec`, `AnchoredPrior`, `ModelSpec` | Done |
| `model/priors.py` | **Implemented** -- `ChannelPriors`, `default_priors_for_category` for all 10 categories | Done |
| `model/builder.py` | **Implemented** -- `build_model_spec(config)` translates `WanamakerConfig` → `ModelSpec` | Done |
| `refresh/anchor.py` | **Implemented** -- presets, `resolve_anchor_weight` | Done |
| `refresh/classify.py` | **Implemented** -- `MovementClass` enum, `classify_movement`, `unexplained_fraction` | Done |
| `refresh/diff.py` | **Implemented** -- `ParameterMovement` (with `movement_class`), `RefreshDiff`, `compute_diff` | Done |
| `forecast/posterior_predictive.py` | **Implemented** -- forecast plan loading, spend-range warnings, posterior predictive contract | Done |
| `forecast/scenarios.py` | **Implemented** -- conservative scenario ranking with caveats | Done |
| `forecast/ramp.py` | **Implemented** -- risk-adjusted ramp recommendation core | Done |
| `trust_card/card.py` | **Implemented** -- `TrustStatus`, `TrustDimension`, `TrustCard` frozen dataclasses | Done |
| `trust_card/compute.py` | **Implemented** -- Trust Card computation from diagnostics and posterior summaries | Done |
| `advisor/channel_flagging.py` | **Implemented** -- minimal v1 experiment flagging | Done |
| `reports/render.py` | **Implemented** -- deterministic Jinja2 Markdown rendering for executive summary, Trust Card, and ramp recommendation | Done |
| `benchmarks/loaders.py` | **Implemented** -- synthetic and public benchmark loaders | Done |
| `_xgboost_aux/` | Deferred supporting-role module only; no production xgboost paths yet | Post-v1 hardening |

---

## 8. Key Invariants

These are not guidelines -- violating them means rebuilding the project's credibility from scratch.

| Invariant | Enforcement |
|---|---|
| No HTTP / telemetry in core code paths | CI test `test_no_network_in_core.py` (import-time); runtime gate `test_network_isolation.py` when engine tests are enabled |
| No LLM calls for output generation | Code review + no LLM SDK in dependencies |
| No xgboost for ROI / saturation / adstock | AGENTS.md Hard Rule 3; `_xgboost_aux` module isolation |
| Numerically close reproducibility given same seed (RTOL=1e-6) | NFR-2; `tests/test_reproducibility.py` (enabled with `WANAMAKER_RUN_ENGINE_TESTS=1`) |
| Transforms have unit tests against known outputs | AGENTS.md Hard Rule 4; test coverage requirement |
| Engine abstraction is not bypassed | No direct `pymc`/`numpyro` imports in feature code |
| Downstream modules consume `PosteriorSummary`, not `Posterior.raw` | Code review; `Posterior.raw` typed as `Any` to discourage casual use |
| CSV interface is not redesigned without PRD review | AGENTS.md Hard Rule 6 |
| `runtime_mode` and xgboost paths use distinct labels | `xgb_preview` / `xgb_crosscheck`; "quick mode" reserved for sampler tier |

---

## 9. Dependency Decisions

**Production dependencies** (currently):

| Package | Role |
|---|---|
| `numpy >= 1.26` | Numerical arrays; all transform functions operate on `NDArray[float64]` |
| `pandas >= 2.2` | CSV I/O and data manipulation |
| `pydantic >= 2.6` | Config validation with strict extra-forbid schemas |
| `pyyaml >= 6.0` | YAML config loading |
| `typer >= 0.12` | CLI |
| `jinja2 >= 3.1` | Report template rendering |
| `matplotlib >= 3.8`, `seaborn >= 0.13` | Static report charts |
| `pymc >= 5.0` | Selected Bayesian modeling engine |

**Intentionally omitted:** `numpyro`, `jax`, `cmdstanpy` -- rejected for the primary modeling role unless the engine decision is revisited.

**Optional extras:**
- `.[xgboost]` -- adds `xgboost >= 2.0` for `xgb_preview` and `xgb_crosscheck` only
- `.[dev]` -- adds `pytest`, `pytest-cov`, `ruff`, `mypy`
- `.[docs]` -- adds `mkdocs`, `mkdocs-material`

**Explicitly excluded from production dependencies:** any HTTP client, LLM SDK, telemetry library, or crash reporter.

---

## 10. Testing Strategy

**Unit tests** (`tests/unit/`): Pure-function tests. Statistical functions require at least one worked example with known input -> known output. Fast; no engine required.

**Integration tests** (`tests/integration/`): End-to-end command tests against fixture data. Require a configured engine; marked `@pytest.mark.integration` and skipped pre-Phase 0.

**Benchmark tests** (`tests/benchmarks/`): Acceptance criteria from the BRD/PRD -- e.g., model recovers known contributions within 15%, refresh stability meets NFR-5 targets. Slow; marked `@pytest.mark.benchmark`.

**CI gates:**
- `test_no_network_in_core.py` -- runs on every PR; walks core import graph at import time (fast guardrail)
- `tests/test_reproducibility.py` -- enabled with `WANAMAKER_RUN_ENGINE_TESTS=1`; fits same benchmark twice with same seed, compares all summary floats within RTOL=1e-6 (numerically close, not necessarily bit-for-bit identical across platforms)
- Network-isolated integration test -- runs the primary core workflow without network access when engine tests are enabled
- `.github/workflows/install.yml` -- runs on push to master and on release tags; performs a clean-environment `pip install .` and CLI smoke tests on Linux, macOS (Intel + ARM), and Windows
- `.github/workflows/docker.yml` -- runs on push to master and on release tags; builds the image, asserts the FR-6.2 size ceiling, and runs the bundled one-command demo inside the container

---

## 11. Open Decisions

| Decision | Options | Resolution path |
|---|---|---|
| **Default anchoring weight** | Currently 0.3 (placeholder) | Phase 1: calibration against refresh stability benchmark (#18) |
| **Primary recommended install path** | pip vs. Docker | Validate PyMC install in cross-platform CI; pip remains the target path unless CI proves otherwise |
| **Interactive charts** | matplotlib/seaborn (static) vs. plotly (interactive HTML) | Phase 2: user testing feedback |
| **Cross-platform reproducibility** | Resolved: numerically close (RTOL=1e-6), not bit-for-bit identical | Documented in `tests/test_reproducibility.py` |

---

*Last updated: 2026-05-01. Update this document when module responsibilities change, interfaces are finalized, or key decisions land.*
