# Wanamaker — Architecture and Technical Specification

**Audience:** Developers and AI coding assistants working on the codebase.
**Companion documents:** [`docs/wanamaker_brd_prd.md`](wanamaker_brd_prd.md) (what and why), [`AGENTS.md`](../AGENTS.md) (hard rules and coding conventions).
**Status:** Phase -1 (engine decision spike). Most modules are scaffolded but not yet implemented.

---

## 1. Design Philosophy

Three principles shape every architectural decision:

1. **Trust is the product.** The module structure, naming, and data flow are all oriented around making the model's behavior auditable and reproducible — not convenient or clever. When in doubt, choose transparency over elegance.
2. **Engine-agnostic core.** The Bayesian engine (PyMC / NumPyro / Stan) is not yet decided. All feature code must interface through `wanamaker.engine` and never import a specific library directly.
3. **Local-first, always.** No module in the core code path may import an HTTP client, telemetry library, or LLM SDK. This is enforced by a CI test (`tests/test_no_network_in_core.py`), not just convention.

---

## 2. Repository Layout

```
wanamaker/
├── src/wanamaker/          # installable package
│   ├── __init__.py         # version only
│   ├── cli.py              # typer CLI; thin dispatch layer
│   ├── config.py           # YAML loading + pydantic validation
│   ├── seeding.py          # reproducibility discipline (NFR-2)
│   ├── artifacts.py        # local artifact storage (.wanamaker/ layout)
│   ├── data/               # CSV I/O and channel taxonomy
│   ├── diagnose/           # pre-flight readiness diagnostic
│   ├── engine/             # Bayesian engine abstraction (Protocol + types)
│   ├── transforms/         # adstock and saturation (load-bearing math)
│   ├── model/              # engine-agnostic model specification and priors
│   ├── refresh/            # versioning, diff, anchoring, movement classification
│   ├── forecast/           # posterior predictive and scenario comparison
│   ├── trust_card/         # credibility dimension data structures
│   ├── advisor/            # experiment advisor (channel flagging)
│   ├── reports/            # Jinja2 templates and rendering
│   ├── benchmarks/         # benchmark dataset loaders (NFR-7)
│   └── _xgboost_aux/       # xgboost in supporting role only (see §7)
├── tests/
│   ├── test_no_network_in_core.py  # CI gate: no HTTP in core modules
│   ├── unit/               # pure-function tests
│   ├── integration/        # end-to-end command tests
│   └── benchmarks/         # benchmark-driven acceptance tests (slow)
├── docs/
│   ├── wanamaker_brd_prd.md
│   └── architecture.md     # this file
├── benchmark_data/         # synthetic + public datasets (Phase 0+)
├── examples/               # runnable quickstart examples (Phase 2+)
├── pyproject.toml
├── AGENTS.md
└── LICENSE
```

---

## 3. Module Map and Responsibilities

### 3.1 `cli.py` — Entry Point

The CLI is intentionally thin. Each command resolves configuration, then delegates immediately to the appropriate subpackage. No business logic lives here.

| Command | Delegates to |
|---|---|
| `wanamaker diagnose <data.csv>` | `wanamaker.diagnose` |
| `wanamaker fit --config <config.yaml>` | `wanamaker.model`, `wanamaker.engine` |
| `wanamaker report --run-id <id>` | `wanamaker.reports` |
| `wanamaker forecast --run-id <id> --plan <plan.csv>` | `wanamaker.forecast` |
| `wanamaker compare-scenarios --run-id <id> --plans ...` | `wanamaker.forecast` |
| `wanamaker refresh --config <config.yaml>` | `wanamaker.refresh`, `wanamaker.engine` |

Built with [typer](https://typer.tiangolo.com/). The `--anchor-strength` flag on `refresh` accepts named presets (`none`, `light`, `medium`, `heavy`) or a float; resolution lives in `refresh.anchor.resolve_anchor_weight`.

### 3.2 `config.py` — Configuration Contract

All user-facing configuration is validated here via pydantic before any other module sees it. The schema is the Layer 2 interface (guided YAML configuration) in the three-layer progressive disclosure architecture.

**Key types:**

```
WanamakerConfig
  └── DataConfig          csv_path, date_column, target_column,
  │                       spend_columns, control_columns, lift_test_csv
  ├── ChannelConfig[]     name, category, adstock_family per channel
  ├── RefreshConfig       anchor_strength (preset or float)
  └── RunConfig           seed, runtime_mode, artifact_dir
```

`extra="forbid"` is set on all models — unknown YAML keys are a validation error, not silently ignored. This prevents config drift.

**Runtime modes** (`quick` / `standard` / `full`) control sampler effort only; the model specification is identical across all three tiers.

### 3.3 `seeding.py` — Reproducibility Discipline

The single source of truth for all random state. Two functions:

- `make_rng(seed: int) → np.random.Generator` — creates a fresh generator, never touching global `np.random` state.
- `derive_seed(parent_seed, label) → int` — produces a deterministic child seed via BLAKE2b hash, so independent components (e.g., the xgboost cross-check) get their own reproducible streams without colliding with the main sampler.

**Rule:** no module may call `np.random.seed()`, `random.seed()`, or read global random state. All randomness flows from the top-level `config.run.seed`, passed explicitly down the call stack.

### 3.4 `artifacts.py` — Local Artifact Storage

Owns the `.wanamaker/` directory layout. Every model fit writes a versioned run directory:

```
.wanamaker/
└── runs/
    └── <run_id>/
        ├── config.yaml       snapshot of the run config
        ├── data_hash.txt     SHA-256 of the input CSV
        ├── posterior.nc      posterior draws (engine-native format, e.g. NetCDF via ArviZ)
        ├── summary.json      marginal summaries used for refresh diffs
        ├── timestamp.txt     ISO-8601 UTC fit timestamp
        └── engine.txt        engine name + version
```

**Run ID generation:** content-addressed from the data hash, config, and timestamp — stable for a given `(data, config)` pair within the same minute, so identical re-runs don't proliferate directories.

`artifacts.py` does not own the *contents* of `posterior.nc` or `summary.json` — those formats belong to `engine` and `refresh` respectively.

### 3.5 `data/` — CSV Interface and Channel Taxonomy

| File | Responsibility |
|---|---|
| `io.py` | Loads and validates the input CSV; parses dates; raises `ValueError` with actionable messages on bad input. No imputation — decisions about missing data belong to the diagnostic step. |
| `taxonomy.py` | Defines the 10 default channel categories (`paid_search`, `paid_social`, `video`, `linear_tv`, `ctv`, `audio_podcast`, `display_programmatic`, `affiliate`, `email_crm`, `promotions_discounting`). Category-driven default priors live in `model/priors.py`. |

**The CSV contract is a user-facing API.** Do not change input/output formats without revisiting BRD/PRD §5.1.

### 3.6 `diagnose/` — Pre-Flight Readiness Diagnostic

Implements `wanamaker diagnose` (FR-2). Runs before fitting; cannot be silently skipped.

| File | Responsibility |
|---|---|
| `readiness.py` | Data structures: `ReadinessLevel` enum (4 values), `CheckSeverity` enum, `CheckResult` dataclass, `ReadinessReport` dataclass. |
| `checks.py` | Pure functions, one per check. Each takes a DataFrame and returns `CheckResult` or `list[CheckResult]`. |

**Checks to implement (FR-2.2):**

| Check | Trigger | Severity |
|---|---|---|
| `history_length` | < 52 weeks warning, < 26 weeks blocker | WARNING / BLOCKER |
| `missing_values` | any NaN in target or spend columns | WARNING / BLOCKER |
| `spend_variation` | coefficient of variation below threshold | WARNING |
| `collinearity` | paid-vs-paid or paid-vs-control correlation | WARNING |
| `variable_count` | 1:10 rule (predictors vs. observations) | WARNING |
| `date_regularity` | gaps or duplicate dates | WARNING / BLOCKER |
| `target_stability` | extreme outliers | WARNING |
| `target_leakage` | control correlation to target ≥ 0.95 | WARNING |
| `structural_breaks` | change-point detection on residuals | INFO / WARNING |

**Readiness levels** (never a numeric score):

| Level | Meaning |
|---|---|
| `ready` | No blockers, no significant warnings |
| `usable_with_warnings` | Model can fit; specific outputs should be treated with caution |
| `diagnostic_only` | Model can fit but should not drive decisions |
| `not_recommended` | Fundamental data problems; specific issues identified |

### 3.7 `engine/` — Bayesian Engine Abstraction

The most important architectural boundary in the codebase. All feature code must import from `wanamaker.engine`, never from `pymc`, `numpyro`, or `cmdstanpy` directly.

**Current state:** Protocol defined; no backend implemented. Phase -1 will produce the concrete implementation.

| File | Responsibility |
|---|---|
| `base.py` | `Posterior`, `FitResult` dataclasses; `Engine` Protocol defining the two required methods. |

**`Engine` Protocol interface:**

```python
class Engine(Protocol):
    name: str  # "pymc" | "numpyro" | "stan"

    def fit(self, model_spec: ModelSpec, data: DataFrame,
            seed: int, runtime_mode: str) -> FitResult: ...

    def posterior_predictive(self, posterior: Posterior,
                              new_data: Any, seed: int) -> Any: ...
```

**`Posterior`** wraps the engine-native object (e.g., ArviZ `InferenceData` for PyMC) in a stable, opaque container. Code needing marginal summaries should call helpers rather than reaching into `Posterior.raw`.

**`FitResult`** carries the `Posterior` plus a `diagnostics` dict (R-hat, effective sample size, divergences).

**Phase -1 engine candidates** (ordered by current presumed fit):
1. **PyMC** — leading candidate; pip-installable on all platforms; mature
2. **NumPyro/JAX** — technically excellent; installation friction on Windows is the risk
3. **Stan via cmdstanpy** — highest library quality; C++ compile step adds friction

Decision criterion: *can the target user install it on Windows in 5 minutes and get a result in under 30 minutes on a laptop CPU?*

Once chosen, the backend lives at e.g. `engine/pymc.py` — a concrete class implementing `Engine`.

### 3.8 `transforms/` — Load-Bearing Statistical Primitives

These are the core of the model. Per AGENTS.md Hard Rule 4: bugs here produce silently wrong ROI numbers. Every function must have a docstring citing the canonical source, and unit tests against worked examples with known outputs.

| File | Function | Status |
|---|---|---|
| `adstock.py` | `geometric_adstock(spend, decay)` | Scaffolded, not yet implemented |
| `adstock.py` | `weibull_adstock(spend, shape, scale)` | Scaffolded, not yet implemented |
| `saturation.py` | `hill_saturation(spend, ec50, slope)` | Scaffolded, not yet implemented |

**Geometric adstock:** `y_t = x_t + decay × y_{t-1}`, `y_{-1} = 0`. Decay ∈ [0, 1). Reference: Hanssens, Parsons & Schultz, *Market Response Models* (2nd ed., 2001), §10.3.

**Weibull adstock:** Per Jin et al. (Google, 2017) §3. Per-channel override path only; does not auto-flip across refreshes (FR-3.4).

**Hill saturation:** `f(x) = x^slope / (x^slope + ec50^slope)`. EC50 = spend at half-maximum response. Reference: Jin et al. (2017) §4.

Before implementing or modifying any transform, read [`adstock_and_saturation.md`](adstock_and_saturation.md) — it contains canonical formulas, default priors per channel category, estimation tradeoffs, and diagnostic patterns.

### 3.9 `model/` — Engine-Agnostic Model Specification

Produces a `ModelSpec` — a pure data description of the model that any backend can consume. No sampling, no engine imports here.

| File | Responsibility |
|---|---|
| `spec.py` | `ChannelSpec` (name, category, adstock_family) and `ModelSpec` (channels list, control_columns). Frozen dataclasses. |
| `priors.py` | `default_priors_for_category(category)` — returns prior shapes keyed by category. To be populated in Phase 0 with PRD-cited values. |

**Separation of concerns:** `ModelSpec` is data; the engine translates it into a concrete probabilistic program (PyMC model, NumPyro plates, Stan blocks, etc.). This means the same model description can be rendered by any backend.

### 3.10 `refresh/` — Refresh Accountability (Headline Feature)

Three components implementing FR-4:

| File | Responsibility |
|---|---|
| `anchor.py` | Posterior anchoring as a mixture prior. `ANCHOR_PRESETS` dict, `resolve_anchor_weight(value)`. |
| `diff.py` | `ParameterMovement`, `RefreshDiff` dataclasses; `compute_diff(previous_summary, current_summary) → RefreshDiff`. |
| `classify.py` | `MovementClass` enum with 5 values. |

**Anchoring math (FR-4.4):**
```
Prior_new(θ) = (1 - w) · Prior_default(θ) + w · Posterior_previous(θ)
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

The fraction classified as `unexplained` is the headline Trust Card refresh stability metric. NFR-5 target: ≥ 90% classified as non-`unexplained` on the refresh stability benchmark dataset.

### 3.11 `forecast/` — Posterior Predictive and Scenario Comparison

| File | Responsibility |
|---|---|
| `posterior_predictive.py` | `forecast(posterior, future_spend, seed)` — forward simulation given a budget plan. |
| `scenarios.py` | `compare_scenarios(posterior, plans, seed)` — ranks 2–3 user-supplied plans with uncertainty. Flags extrapolation beyond historical observed spend. |

v1 ships scenario comparison (user-driven). Constrained inverse optimization ("how do I hit X?") is deferred to v1.1. Spend-invariant channels are excluded from reallocation recommendations (FR-3.2).

### 3.12 `trust_card/` — Credibility Assessment

| File | Responsibility |
|---|---|
| `card.py` | `TrustStatus` enum (pass/moderate/weak), `TrustDimension` dataclass, `TrustCard` dataclass. |

**v1 dimensions (FR-5.4):**

| Dimension | Data source |
|---|---|
| `convergence` | R-hat, effective sample size from `FitResult.diagnostics` |
| `holdout_accuracy` | Out-of-sample prediction error |
| `refresh_stability` | Fraction of `unexplained` movements (present only when there is a prior run) |
| `prior_sensitivity` | Posterior shift under perturbed priors |
| `saturation_identifiability` | Per-channel; flags spend-invariant channels |
| `lift_test_consistency` | Posterior vs. lift-test estimate alignment (present only when calibration data provided) |

The Trust Card is the connective tissue between model output and recommendations: `weak` on any dimension feeds specific hedged language into the executive summary templates and specific warnings into scenario comparison.

### 3.13 `advisor/` — Experiment Advisor

Minimal v1 implementation: identifies channels where posterior credible interval is wide and spend is large enough that better information would meaningfully change recommendations. Output: a prioritized list of `ChannelFlag` objects with rationale.

v1.1 will add experiment design (geo holdout vs. budget split, sample size, duration, geo selection).

### 3.14 `reports/` — Rendering

| File | Responsibility |
|---|---|
| `render.py` | `render_executive_summary(context)`, `render_trust_card(context)` via Jinja2 `PackageLoader`. Templates ship inside the wheel. |
| `templates/executive_summary.md.j2` | Plain-English executive summary (FR-5.3). Language adjusts by confidence level via Jinja2 conditionals — `weak` Trust Card dimensions produce hedged language; `pass` dimensions produce definitive statements. |
| `templates/trust_card.md.j2` | Trust card template. |

**Hard Rule:** No LLM calls for output generation. If a template feels limiting, expand the template logic. Every word in every report is deterministic given the same posterior statistics.

### 3.15 `benchmarks/` — Benchmark Dataset Loaders

Loaders for the eight named benchmark datasets (NFR-7). The datasets themselves live in `benchmark_data/` at the repo root and are generated/curated in Phase 0. These loaders are the stable API; the dataset files may evolve.

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

### 3.16 `_xgboost_aux/` — xgboost in a Supporting Role

Leading underscore signals internal support, not a user-facing API. xgboost has two and only two uses in this project:

1. **Quick-mode forecast preview** — a fast tree-based forecast with conformal intervals as a sanity check before the full Bayesian fit. Output is labeled as a preview, not as ROI.
2. **Validation cross-check** — an independent tree-based fit. If Bayesian and tree forecasts disagree substantially, the Trust Card raises a flag.

xgboost does **not** produce ROI estimates, saturation curves, adstock parameters, or channel-level inference of any kind (AGENTS.md Hard Rule 3).

---

## 4. Data Flow

The full workflow from CSV to executive summary:

```
User's CSV + YAML config
        │
        ▼
┌──────────────┐
│  config.py   │  Validate YAML → WanamakerConfig
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   data/io    │  Load CSV, parse dates, sort → DataFrame
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  diagnose/   │  Battery of checks → ReadinessReport
│              │  (required before fit in default flow)
└──────┬───────┘
       │  ReadinessReport.level ≠ NOT_RECOMMENDED (or --skip-validation)
       ▼
┌──────────────┐
│   model/     │  WanamakerConfig + taxonomy → ModelSpec
│              │  (channels, priors, adstock families)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  transforms/     │  apply geometric/Weibull adstock per channel
│                  │  apply Hill saturation per channel
└──────┬───────────┘
       │  transformed media arrays
       ▼
┌──────────────┐      ┌────────────────────────┐
│  refresh/    │─────▶│  Prior_new = mixture   │  (if prior run exists)
│  anchor.py   │      │  of default + previous │
└──────┬───────┘      └────────────────────────┘
       │
       ▼
┌──────────────┐
│  engine/     │  Engine.fit(model_spec, data, seed, runtime_mode)
│  (backend)   │  → FitResult(Posterior, diagnostics)
└──────┬───────┘
       │
       ├──────────────────────────────────────────────────┐
       │                                                  │
       ▼                                                  ▼
┌──────────────┐                               ┌──────────────────┐
│  artifacts/  │  persist run to .wanamaker/   │  refresh/diff    │
│              │  config, data_hash, posterior,│  (if prior run)  │
│              │  summary, timestamp, engine   │  → RefreshDiff   │
└──────────────┘                               └──────────────────┘
       │
       ▼
┌──────────────┐
│  trust_card/ │  FitResult + diagnostics → TrustCard
└──────┬───────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
┌──────────────┐      ┌──────────────┐
│  forecast/   │      │  advisor/    │
│  scenarios   │      │  flagging    │
└──────┬───────┘      └──────┬───────┘
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
          ┌──────────────┐
          │  reports/    │  TrustCard + posterior stats + RefreshDiff
          │              │  → Jinja2 templates
          │              │  → executive_summary.md
          │              │  → trust_card.md
          └──────────────┘
```

**Seeding flow:** `config.run.seed` is the single root. `seeding.derive_seed(parent, "xgb_crosscheck")`, `seeding.derive_seed(parent, "posterior_predictive")`, etc. give independent components their own streams.

---

## 5. Three-Layer Progressive Disclosure

The user-facing API has three layers (FR-7), each providing full access:

| Layer | Interface | What it exposes |
|---|---|---|
| **1 — Defaults only** | CLI with a CSV and a minimal YAML | Date column, target column, nothing else required |
| **2 — Guided config** | YAML with full `WanamakerConfig` schema | Channel categories, business constraints, lift test path, runtime mode, anchor strength |
| **3 — Expert overrides** | Python API | Direct `ModelSpec`, prior overrides, custom transforms, raw `Posterior` objects, sampler tuning |

A new user should be able to go from `pip install wanamaker` to a complete executive summary in under 30 minutes following only the quickstart documentation (FR-7.1 acceptance criterion).

---

## 6. Artifact Storage and Privacy

All outputs go to `.wanamaker/` in the project directory (FR-Privacy.2). Nothing is written to the user's home directory or any system path by default. The `artifact_dir` can be overridden via `config.run.artifact_dir` or a CLI flag.

**No network calls in any core code path.** Enforced by `tests/test_no_network_in_core.py`, which walks the import graph of all core subpackages and fails the build if any banned module (requests, httpx, aiohttp, urllib3, openai, anthropic, telemetry SDKs) appears.

---

## 7. What Is Built vs. Stubbed

| Module | Status | Phase |
|---|---|---|
| `cli.py` | Scaffolded — all 6 commands raise `NotImplementedError` | Commands wired up across Phase 0–2 |
| `config.py` | **Implemented** — full pydantic schema loads and validates | Done |
| `seeding.py` | **Implemented** — `make_rng` and `derive_seed` fully working | Done |
| `artifacts.py` | **Implemented** — directory layout, file paths, `list_runs` | Done |
| `data/io.py` | **Partially implemented** — `load_input_csv` works; `load_lift_test_csv` stubbed | Phase 1 |
| `data/taxonomy.py` | **Implemented** — channel category names defined | Priors added Phase 0 |
| `diagnose/readiness.py` | **Implemented** — all data structures defined | Done |
| `diagnose/checks.py` | Scaffolded — 3 check functions stubbed | Phase 1 |
| `engine/base.py` | **Implemented** — Protocol, `Posterior`, `FitResult` defined | Backends: Phase 0 |
| `transforms/adstock.py` | Scaffolded — both functions stubbed | Phase 0 |
| `transforms/saturation.py` | Scaffolded — Hill function stubbed | Phase 0 |
| `model/spec.py` | **Implemented** — `ChannelSpec`, `ModelSpec` frozen dataclasses | Done |
| `model/priors.py` | Scaffolded — `default_priors_for_category` stubbed | Phase 0 |
| `refresh/anchor.py` | **Implemented** — presets, `resolve_anchor_weight` fully working | Done |
| `refresh/classify.py` | **Implemented** — `MovementClass` enum defined | Done |
| `refresh/diff.py` | Scaffolded — data structures defined; `compute_diff` stubbed | Phase 1 |
| `forecast/posterior_predictive.py` | Stubbed | Phase 1 |
| `forecast/scenarios.py` | Stubbed | Phase 1 |
| `trust_card/card.py` | **Implemented** — all data structures defined | Done |
| `advisor/channel_flagging.py` | Scaffolded — `ChannelFlag` defined; `flag_channels` stubbed | Phase 2 |
| `reports/render.py` | **Implemented** — Jinja2 environment wired; templates are empty shells | Templates: Phase 2 |
| `benchmarks/loaders.py` | Stubbed | Phase 0–1 |
| `_xgboost_aux/` | Empty | Phase 1 |

---

## 8. Key Invariants (Enforced or Load-Bearing)

These are not guidelines — violating them means rebuilding the project's credibility from scratch.

| Invariant | Enforcement |
|---|---|
| No HTTP / telemetry in core code paths | CI test `test_no_network_in_core.py` |
| No LLM calls for output generation | Code review + no LLM SDK in dependencies |
| No xgboost for ROI / saturation / adstock | AGENTS.md Hard Rule 3; `_xgboost_aux` module isolation |
| Bit-for-bit reproducibility given same seed | NFR-2; planned CI test running same fit twice |
| Transforms have unit tests against known outputs | AGENTS.md Hard Rule 4; test coverage requirement |
| Engine abstraction is not bypassed | No direct `pymc`/`numpyro` imports in feature code |
| CSV interface is not redesigned without PRD review | AGENTS.md Hard Rule 6 |

---

## 9. Dependency Decisions

**Production dependencies** (currently):

| Package | Role |
|---|---|
| `numpy ≥ 1.26` | Numerical arrays; all transform functions operate on `NDArray[float64]` |
| `pandas ≥ 2.2` | CSV I/O and data manipulation |
| `pydantic ≥ 2.6` | Config validation with strict extra-forbid schemas |
| `pyyaml ≥ 6.0` | YAML config loading |
| `typer ≥ 0.12` | CLI |
| `jinja2 ≥ 3.1` | Report template rendering |
| `matplotlib ≥ 3.8`, `seaborn ≥ 0.13` | Static report charts |

**Intentionally omitted (pending Phase -1):** `pymc`, `numpyro`, `jax`, `cmdstanpy` — not added until the engine decision is made.

**Optional extras:**
- `.[xgboost]` — adds `xgboost ≥ 2.0` for quick-mode preview and validation cross-check only
- `.[dev]` — adds `pytest`, `pytest-cov`, `ruff`, `mypy`
- `.[docs]` — adds `mkdocs`, `mkdocs-material`

**Explicitly excluded from production dependencies:** any HTTP client, LLM SDK, telemetry library, or crash reporter.

---

## 10. Testing Strategy

**Unit tests** (`tests/unit/`): Pure-function tests. Statistical functions require at least one worked example with known input → known output. Fast; no engine required.

**Integration tests** (`tests/integration/`): End-to-end command tests against fixture data. Require a configured engine; marked `@pytest.mark.integration` and skipped pre-Phase 0.

**Benchmark tests** (`tests/benchmarks/`): Acceptance criteria from the BRD/PRD — e.g., model recovers known contributions within 15%, refresh stability meets NFR-5 targets. Slow; marked `@pytest.mark.benchmark`.

**CI gates:**
- `test_no_network_in_core.py` — runs on every PR; walks core import graph
- Reproducibility test (planned Phase 0) — runs same fit twice, compares posteriors bit-for-bit
- Cross-platform install test (planned Phase 2) — clean-environment pip install on Linux, macOS (Intel + ARM), Windows

---

## 11. Open Decisions (Phase -1 Outputs)

| Decision | Options | Resolution path |
|---|---|---|
| **Bayesian engine** | PyMC (leading), NumPyro/JAX, Stan | Phase -1 spike: install test on all 3 platforms + runtime benchmark |
| **Default anchoring weight** | Currently 0.3 (placeholder) | Phase 1: calibration against refresh stability benchmark |
| **Primary recommended install path** | pip vs. Docker | Follows from engine decision (JAX friction → Docker first) |
| **polars vs. pandas internal** | pandas currently; polars if perf becomes a concern | Defer until profiling shows need |
| **Interactive plots** | matplotlib/seaborn (static) or plotly (interactive HTML) | Phase 2: user testing feedback |

---

*Last updated: 2026-04-29. Update this document when module responsibilities change, interfaces are finalized, or the engine decision lands.*
