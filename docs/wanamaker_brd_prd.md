# Wanamaker

*Knowing which half.*

An open-source marketing mix model for teams that need credible measurement without a PhD program.

---

**Document type:** Combined Business Requirements Document (BRD) and Product Requirements Document (PRD)
**Version:** 0.4 (locked — final pre-build version)
**Date:** April 28, 2026
**Status:** Locked. Next revision driven by Phase -1 outputs.
**License (target):** MIT

**Changes from v0.3:**
- Added `--anchor-strength` CLI flag with named presets (light/medium/heavy) mapping to underlying `w` values (FR-4.4)
- Added memory estimates to runtime tiers (FR-3.5)
- Added explicit extrapolation visual warnings for response curves (FR-5.1)
- Added "Comparison to Robyn / Meridian / Recast / PyMC-Marketing" as a required documentation section (NFR-3)
- Expanded executive summary tester pool to include 2-3 agency analytics leads (FR-5.3)
- Added one-command end-to-end example as a README requirement (NFR-3)
- Added 95% credible interval coverage to FR-3.1 acceptance criteria

**Changes from v0.2 (carried forward in v0.3):**
- Added Section 6 (Privacy and Local Processing) specifying no network calls, no telemetry, local-only artifact storage
- Revised FR-5.3 with a Likert-scale acceptance criterion replacing subjective "comprehensible without explanation" language
- Expanded FR-4.4 with mathematical specification of the anchoring mixture prior and a placeholder default weight pending Phase 1 empirical work
- Added target leakage and structural break checks to FR-2.2
- Added explicit note in Section 9 that engine choice is deferred to Phase -1; current candidate hypotheses are PyMC as the modeling engine and xgboost in a supporting role for quick-mode previews and validation cross-checks
- Renumbered subsequent sections to accommodate the new Section 6

**Changes from v0.1 (carried forward in v0.2):**
- Added Phase -1 (Engine Decision Spike + Persona Validation) before Phase 0
- Reframed primary differentiator from "refresh stability" to "refresh accountability" (light anchoring + total transparency)
- Cut from v1: constrained inverse optimization (kept scenario comparison), full Halo diagnostic (deferred to v1.1, renamed)
- Kept in v1: forecasting, scenario comparison, executive summary, minimal Experiment Advisor (channel flagging only)
- Added runtime tiers (Quick / Standard / Full)
- Replaced "no composite score" with discrete readiness levels in data diagnostic
- Added benchmark datasets as named v1 deliverable with quantitative acceptance criteria
- Added quantitative acceptance criteria throughout functional requirements
- Refined success metrics: added usage-quality measures, dropped Forrester/Gartner year-one target

---

## 1. Executive Summary

Wanamaker is an open-source marketing mix modeling tool designed for the gap between two unsatisfying extremes: open-source MMMs that require a deep data science bench (Meridian, Robyn, PyMC-Marketing) and commercial SaaS that costs $80K–$120K/year and abstracts the methodology behind a black box (Recast, Measured, Haus).

The product is for mid-market consumer brands and the agencies that serve them — companies with $5M–$200M in annual media spend, 1.5–3 years of weekly historical data, and one analyst rather than a data science team. These organizations need credible measurement to make budget decisions, but the existing open-source tools require expertise they don't have, and the commercial tools require budgets they can't justify.

The strategic positioning, in Blue Ocean Strategy terms, is to compete on **trustable usability** rather than on model sophistication. The current market over-indexes on statistical novelty (more samplers, more transformations, more hierarchical structures) and under-invests in the things that actually determine whether a model gets used: refresh accountability, honest diagnostics, plain-English outputs, and decision-ready recommendations.

The primary differentiator is **refresh accountability**: when new data arrives and historical estimates change, Wanamaker shows what changed, why it changed, and whether the change should alter business decisions. This is implemented as a combination of light posterior anchoring (default on, configurable) and a comprehensive refresh diff report that classifies movement. No open-source tool currently solves the refresh problem; Robyn's "refresh nightmare" is the most cited business-killing complaint in the field.

The secondary differentiators are an **automated data readiness check** that runs before any modeling, a **model trust card** that surfaces credibility risks alongside results, **decision-oriented outputs** (forecasting and scenario comparison, not just descriptive analysis), and an **experiment advisor** that flags channels needing experimental validation.

---

## 2. Background and Strategic Rationale

### 2.1 Market Context

The MMM market in 2026 is in a period of forced renaissance. The deprecation of third-party cookies, Apple's App Tracking Transparency framework, and tightening privacy regulation have made multi-touch attribution increasingly unreliable. Marketing organizations are returning to aggregate-level econometric measurement as a privacy-resilient alternative.

The four open-source / quasi-open-source tools dominating the conversation each have well-understood limitations:

- **Meridian (Google):** Statistically rigorous Bayesian framework, but heavy operational footprint, GPU-dependent, and Google-shaped (with Google Query Volume and YouTube reach/frequency as flagship features that lock the user into Google's data philosophy).
- **Robyn (Meta):** Mature and accessible relative to Meridian, but suffers from refresh instability, R/Python hybrid environment friction, and lacks native forecasting.
- **LightweightMMM (Google):** Officially deprecated January 2025, archived January 2026. Was loved for speed and Python-native design, but Google never resourced it adequately.
- **Recast:** Commercial SaaS that solves many of the operational problems but costs $80K–$120K/year and is closed-source.

There is real demand for a tool that occupies the space between these — open-source and free, statistically credible enough that a senior data person trusts it, but operationally accessible enough that a marketing analyst can run it without a Bayesian background.

### 2.2 Strategic Thesis

Most MMM products compete on model sophistication. Wanamaker competes on **trustable usability** — the proposition that the tool tells you what the model thinks, how confident it is, whether you should believe it, what decision it supports, and what experiment would make it better.

The Blue Ocean Strategy framing produces a clear ERRC (Eliminate, Reduce, Raise, Create) grid:

| Action | What |
|---|---|
| **Eliminate** | R/Python hybrid environment, GPU-first architecture, raw Bayesian configuration as the default user experience, multiple sampler/backend choices |
| **Reduce** | Number of modeling choices exposed up front, platform-specific features (Google Query Volume, YouTube reach/frequency), campaign-level modeling complexity, geo-hierarchical modeling (deferred to v2), formal multiplicative synergy modeling |
| **Raise** | Validation and diagnostics surfacing, refresh accountability, plain-English interpretation, business constraints in scenario comparison, data readiness checks before modeling, executive-readable outputs |
| **Create** | MMM Readiness diagnostic (pre-flight), Model Trust Card (post-fit), Decision Modes UX, Experiment Advisor (v1: channel flagging; v1.1: experiment design), conservative scenario comparison, three-layer progressive disclosure architecture |

### 2.3 Why v1 Must Be Decision-Oriented, Not Just Descriptive

A v1 that ships only diagnostic and explanatory outputs ("here is what happened") would fail to address the gap that distinguishes Wanamaker from Meridian and Robyn. The Robyn complaint that the tool is an "allocator, not a forecaster" is precisely the gap competitors exploit. The Recast case studies repeatedly emphasize forward-looking planning as the moment of customer value.

A descriptive-only v1 would also create an adoption problem: marketing analysts and CMOs evaluate measurement tools by asking "can this help me decide what to do next quarter?" A tool that answers "no, but it can tell you what happened last quarter" loses to tools that answer "yes." Even if the descriptive answer is more honest, the strategic positioning fails.

Therefore, v1 includes forecasting, scenario comparison, and a minimal Experiment Advisor — the decision-supporting outputs — alongside the diagnostic and trust infrastructure. What is deferred to v1.1 is *constrained inverse optimization* ("how do I hit X?"), which carries the highest model-overconfidence risk and should ship only after the model has earned trust.

---

## 3. Target Users and Personas

### 3.1 Primary Persona: The Marketing Analyst

A single analyst at a mid-market consumer brand or DTC company. Comfortable with Python at a working level (can read pandas code, can run a Jupyter notebook, can edit a YAML file). Not a statistician. Reports to a VP of Marketing or CMO who wants budget recommendations, not posterior distributions.

This persona's primary needs:

- Get a credible model running in days, not weeks
- Produce reports that the CMO will actually read
- Avoid recommending budget changes that turn out to be wrong
- Know when the model is unreliable and shouldn't be trusted

### 3.2 Secondary Persona: The Agency Analytics Lead

Runs analytics for an agency serving multiple mid-market clients. More technically sophisticated than the primary persona but optimizes for repeatability across clients. Needs the tool to work consistently across different data shapes, produce client-ready outputs, and scale to a portfolio of accounts.

This persona's primary needs:

- Standardized methodology across clients
- Defensible outputs in client meetings
- Fast onboarding of new client data
- Ability to override defaults when a client situation requires it

### 3.3 Tertiary Persona: The Senior Data Scientist

Works at a larger organization or as a specialized consultant. Has Bayesian experience. Will use Wanamaker either as a baseline tool or as scaffolding for more sophisticated work. Needs full access to priors, model internals, and posterior objects.

This persona's primary needs:

- Ability to override every default
- Direct access to the underlying PyMC/NumPyro objects
- Confidence that the math is correct and the implementation is honest

### 3.4 Persona Validation (Required Before Build)

The "competent analyst with Python literacy but no Bayesian background" persona is plausible but unproven. Before significant build investment, conduct structured interviews:

- 3 agency analytics leads
- 3 mid-market in-house marketing analysts
- 2 fractional CMOs who advise mid-market companies
- 2 independent consultants doing MMM or incrementality work

The critical question to ask each: **"Would you personally install and use this tool, or would you only think it is a good idea?"** That distinction kills many open-source projects.

The interview should also test whether the proposed persona genuinely cannot afford Recast at $80–120K/year and whether they would consider $20–40K/year for a supported version of an open-source tool (relevant to potential v2+ commercial layer).

### 3.5 Explicitly Out of Scope

- Enterprise data science teams running dozens of channels with hundreds of geos (use Meridian)
- Organizations needing a fully managed measurement service (use Recast or Measured)
- Companies with less than 18 months of historical data (the data readiness check should tell them this honestly)
- Real-time bidding or daily-cadence use cases (weekly is the design center)

---

## 4. Product Scope

### 4.1 In-Scope for v1

**Core modeling:**
- Bayesian model fit (engine to be determined by Phase -1 spike; current presumption is NumPyro NUTS)
- Geometric adstock as default; Weibull adstock available
- Hill saturation function
- Channel-category-driven default priors with user override
- National-level modeling only
- Additive contribution model
- Lift-test calibration via CSV input (experimental in v1; refined in v1.1)
- Spend-invariant channel handling that preserves model fit and reports honest limitations

**Pre-modeling:**
- `wanamaker diagnose data.csv` — structured risk assessment with discrete readiness levels
- Channel taxonomy template
- Required validation step before model fitting (cannot be skipped silently)

**Post-modeling outputs (decision-oriented, by design):**
- Three Decision Modes:
  1. *Explain what happened* — descriptive contributions, ROI, baseline decomposition
  2. *What if I do X?* — forward simulation given a user-specified budget plan, including scenario comparison across multiple plans
  3. *Refresh what I had* — refresh diff report comparing current run to previous run, with movement classification
- Forecasting via posterior predictive sampling (forward-looking by default)
- Scenario comparison (user supplies 2–3 plans; tool ranks them with uncertainty)
- Experiment Advisor (v1 minimal: identifies channels needing experimental validation; v1.1: designs the experiments)

**Reports:**
- Plain-English executive summary, deterministically generated from posterior (no LLM)
- Technical appendix for analysts
- Model Trust Card with named credibility dimensions
- Refresh diff report (whenever re-running on previously-seen periods)

**Distribution:**
- Python package (pip-installable)
- Docker container with all dependencies pre-resolved
- CLI for non-Python users
- Three-layer progressive disclosure architecture (CLI → YAML config → Python API)

### 4.2 Deferred to v1.1 (Not v2)

These are real product features that v1 users will eventually want; they're deferred only because they carry implementation or trust risk that should not block v1 launch:

- **Constrained inverse optimization** ("How do I hit X?" mode) — high overconfidence risk; ship after model has earned trust
- **Halo / channel-interaction diagnostic** — high misuse risk; needs careful framing and naming
- **Full Experiment Advisor** with experiment design (sample size, geo selection, duration) — minimal channel-flagging version is in v1
- **Refined lift-test calibration** with prior sensitivity analysis — basic CSV calibration is in v1

### 4.3 Out of Scope for v1 and v1.1 (v2 candidates)

The architecture should support clean addition in v2 but neither v1 nor v1.1 should include:

- **Geo-hierarchical modeling** — most target users won't need it; significant complexity tax
- **Formal multiplicative synergy modeling** — halo diagnostic substitutes if user demand materializes
- **Reach/frequency modeling** — Meridian-shaped feature requiring platform-specific data
- **Campaign-level modeling** — usually underpowered for typical mid-market data
- **Multi-stage attribution** — relevant for subscription businesses but not the v1 design center
- **Real-time / daily refresh** — weekly cadence is the v1 design center
- **LLM-based output generation** — deterministic templates instead
- **Multiple samplers / backends** — single engine only
- **Auto-ingestion connectors** for ad platforms — example schemas and CSV templates instead

---

## 5. Functional Requirements

### 5.1 Data Input

**FR-1.1 — CSV Input Format**
The tool accepts a single CSV file with one row per time period (default: weekly). Required columns: a date column, a target metric column (revenue, conversions, etc.). Spend columns and control variables are auto-detected from a YAML config or CLI flags.

*Acceptance:* Tool accepts CSVs from at least three different real-world data shapes (anonymized) without modification beyond the YAML config.

**FR-1.2 — Channel Taxonomy**
The tool ships with a default channel taxonomy covering the standard categories: paid search, paid social, video, linear TV, CTV, audio/podcast, display/programmatic, affiliate, email/CRM, promotions/discounting. Each category has associated default priors for adstock and saturation that reflect typical behavior for that media type.

*Acceptance:* Default priors for each channel category are documented with the empirical or theoretical basis for the chosen ranges.

**FR-1.3 — Lift Test Calibration Input (Experimental in v1)**
Users can provide a separate CSV containing prior lift test results. Required columns: channel name, test start date, test end date, lift estimate (incremental revenue or conversions), confidence interval (lower bound, upper bound). The tool converts these into informative priors automatically.

*Acceptance:* When a lift test is provided for a channel, the posterior ROI for that channel must shift toward the lift-test estimate by an amount proportional to the test's precision relative to the data signal.

**FR-1.4 — Control Variables**
The tool accepts non-media control variables: pricing, promotions, holidays, seasonality, macroeconomic indicators, organic search demand. These are modeled additively alongside media. There is no requirement for users to supply Google Query Volume or any platform-specific signal.

### 5.2 Data Readiness Diagnostic

**FR-2.1 — Pre-Flight Diagnostic Command**
The CLI command `wanamaker diagnose <data.csv>` runs without fitting a model and produces a structured risk assessment.

**FR-2.2 — Risk Categories**
The diagnostic checks for and reports on, at minimum:
- History length (target: 78+ weeks, warning below 52)
- Missing values in target or spend columns
- Spend variation per channel (flag channels with coefficient of variation below threshold)
- Collinearity between paid channels and between paid channels and controls
- Variable count vs. observation count (1:10 rule)
- Date column regularity (gaps, duplicates)
- Target stability (extreme outliers, structural breaks)
- **Target leakage:** any control variable with absolute correlation to the target above 0.95 is flagged as a likely leakage candidate (e.g., conversion rate or revenue-derived metrics accidentally included as predictors). The diagnostic explains the risk and suggests inspection before fitting.
- **Structural breaks:** detection of substantive shifts in the target series (re-platforming, major price change, COVID-era discontinuities, business model changes) using a change-point detection method on the residuals after seasonality and trend are accounted for. The diagnostic identifies the suspected break date(s) and recommends either splitting the analysis at the break or modeling the break explicitly as a control variable.

*Acceptance:* On the benchmark "low-variation channel" dataset, the diagnostic must flag the spend-invariant channel as a warning. On the "collinear channels" benchmark, it must flag the correlated pair as a warning. On the "insufficient history" benchmark, it must return the appropriate readiness level. On the "target leakage" benchmark, it must flag the leaked variable. On the "structural break" benchmark, it must identify the break within ±2 weeks of the true break point.

**FR-2.3 — Discrete Readiness Levels**
The diagnostic concludes with one of four readiness levels, not a numeric score:

- **Ready** — no blockers, no significant warnings
- **Usable with warnings** — model can fit, but specific outputs should be treated with caution; warnings explained
- **Diagnostic only** — model can fit but should not be used for decision-making; useful only for understanding the data
- **Not recommended** — fundamental data problems make MMM inappropriate; specific issues identified

Each level includes the structured list of detected risks with severity. The discrete level provides the summary judgment users need without the false precision of a composite score.

**FR-2.4 — Validation Required Before Fit**
In the default flow, the data validation step is required before model fitting. Users can override with an explicit flag (`--skip-validation`) but the default cannot be silent.

### 5.3 Model Fitting

**FR-3.1 — Default Model Specification**
Geometric adstock (per-channel half-life prior driven by channel category), Hill saturation (per-channel slope and EC50 priors driven by channel category), additive linear combination of transformed media plus controls, weakly-informative priors on all coefficients.

*Acceptance:* On a synthetic ground-truth dataset with known media contributions, the default model must (a) recover total media contribution within 15%, (b) rank the top 3 channels correctly in at least 80% of simulation runs, and (c) produce 95% credible intervals that cover the true parameter at a rate between 90% and 99% across simulation runs (a basic frequentist check on Bayesian credible interval honesty).

**FR-3.2 — Spend-Invariant Channel Handling**
When a channel's spend has insufficient variation to estimate saturation, the model must:
- Detect this condition during the validation step
- Continue to fit successfully (no crash or non-convergence)
- Use a prior-only treatment for that channel's response curve
- Flag this prominently in the Trust Card and report
- Permit the channel to be included in forward simulation but **not show a saturation curve** for that channel — replace with an explicit "saturation cannot be estimated from observed data" placeholder
- Decline to recommend reallocations involving the spend-invariant channel in scenario comparison

**FR-3.3 — Lift Test Calibration**
When a lift test CSV is provided, the corresponding channel's ROI prior is replaced with an informative prior centered on the test estimate, with width derived from the test's confidence interval.

**FR-3.4 — Adstock Family Selection**
Geometric is the default. Weibull is available as a per-channel override. The tool does **not** auto-flip between families across refreshes (this would compromise refresh accountability). Family selection is user-driven and persists across runs unless explicitly changed.

**FR-3.5 — Runtime Tiers**
The tool supports three runtime modes selectable via flag or YAML:

| Mode | Use case | Target runtime (laptop CPU, 150 weeks × 12 channels) | Target peak memory |
|---|---|---|---|
| Quick | Sanity check, iterative model development | Under 5 minutes | Under 4 GB |
| Standard | Default, normal analysis cycle | Under 30 minutes | Under 8 GB |
| Full | Final decision-support, publication-quality posterior | Under 2 hours | Under 16 GB |

Quick mode uses fewer warmup samples and chains; Standard is the default; Full uses the largest sample and most rigorous diagnostics. The same model specification is fit at all three modes; only the sampling effort differs. Memory estimates are for the v1 design center (national-level model, 12 channels, 150 weeks); larger problems will require proportionally more memory and may force users to step down a tier.

### 5.4 Refresh Accountability (Headline Feature)

**FR-4.1 — Versioned Model Runs**
Every model fit produces a versioned artifact containing: the input data hash, the configuration, the posterior summary statistics, and the timestamp.

**FR-4.2 — Refresh Diff Report**
When fitting on data that includes a previously-fitted period, the tool produces a diff report showing: which historical estimates changed, by how much, and a classification of each movement.

**FR-4.3 — Movement Classification**
Each historical estimate change is classified into one of:

- **Within prior credible interval** — small movement, expected variation
- **Improved holdout accuracy** — large movement but the new model fits historical data better; trustworthy update
- **Unexplained** — large movement with no diagnostic explanation; trust risk, flagged
- **User-induced** — change driven by configuration or prior changes by the user; not a model failure
- **Weakly identified** — movement in a channel that the Trust Card flags as having poor identifiability; expected instability

*Acceptance:* On the "refresh stability" benchmark dataset (a known-stable dataset with 4 weeks added), at least 90% of historical contribution estimates must be classified as "within prior credible interval" when the default anchoring is on.

**FR-4.4 — Light Posterior Anchoring (Default On)**
By default, refreshes use the prior posterior as a *lightly-weighted* informative prior for the new fit. The weight is calibrated so that historical estimates update freely on substantive new evidence but are not destabilized by sampling noise. This is configurable: users can disable anchoring entirely or increase its weight.

**Mathematical specification.** Anchoring is implemented as a mixture prior on the marginal posteriors of channel-level parameters (channel ROI / coefficient, adstock half-life, saturation slope, saturation EC50). For a given parameter θ:

```
Prior_new(θ) = (1 - w) · Prior_default(θ) + w · Posterior_previous(θ)
```

where:
- `w ∈ [0, 1]` is the anchoring weight
- `w = 0` corresponds to no anchoring (full re-estimation against default priors)
- `w = 1` corresponds to maximum anchoring (the previous posterior fully replaces the default prior)
- The default value of `w` is **0.3 (placeholder)**, to be empirically tuned in Phase 1 against the refresh stability benchmark and validated against the unexplained-movement classification target in NFR-5

The anchoring is applied to *marginal* posteriors of channel-level parameters, not to the joint posterior structure. Joint posterior structure (parameter correlations) is regenerated from the new data; only the marginal locations and scales of channel parameters carry forward. This preserves model expressiveness while stabilizing the channel-level outputs that drive business decisions.

Control variable coefficients, baseline parameters, and global hyperparameters are not anchored by default; they re-estimate freely. Anchoring those would risk drift in the model's foundational fit to the data.

**Named presets for non-experts.** The anchoring weight is exposed via a `--anchor-strength` CLI flag (and equivalent YAML config field) with three named presets, plus a `none` option and a numeric override path:

| Preset | Underlying `w` (placeholder pending Phase 1 tuning) | When to use |
|---|---|---|
| `none` | 0.0 | Pure re-estimation; full diff transparency only |
| `light` | 0.2 | Minimal stabilization; prefer when business has changed |
| `medium` | 0.3 (default) | Balanced; the recommended starting point |
| `heavy` | 0.5 | Maximum stabilization; prefer when business is stable |
| `<float>` | User-provided in [0, 1] | Expert override |

This avoids asking non-expert users to reason about a numeric weight while preserving fine-grained control for users who want it.

**FR-4.5 — Stability Diagnostic in Trust Card**
The Trust Card includes a "refresh stability" dimension that quantifies how much historical estimates moved between runs and what fraction of movement was classified as "unexplained" (the trust-risk category).

### 5.5 Outputs and Reports

**FR-5.1 — Three Decision Modes**

*Mode 1: Explain what happened*
Output includes channel contribution waterfall, share-of-spend vs share-of-effect chart, ROI per channel with credible intervals, baseline vs. media decomposition over time. Saturation curves with confidence bands are shown only for channels where saturation is identifiable; spend-invariant channels show the "cannot be estimated" placeholder.

When response curves are shown, the portion of the curve that extrapolates beyond the historical observed spend range must be visually distinguished from the in-range portion (e.g., dashed line vs. solid, lighter color, with an explicit annotation: "extrapolation — model has not observed spend in this range"). This applies to both saturation curves in Explain mode and forecast curves in scenario comparison.

*Mode 2: What if I do X?*
User specifies one or more budget plans as a CSV (channel × period, future spend). Output includes forecasted target metric (point estimate and credible interval) for each plan, per-channel contribution forecast, and a side-by-side comparison ranking the plans with uncertainty.

*Mode 3: Refresh what I had*
Re-runs the model on updated data and produces the refresh diff report.

**FR-5.2 — Scenario Comparison (Not Constrained Optimization)**
v1 supports user-driven scenario comparison: the user supplies budget plans and the tool ranks them. This keeps the human in control of the strategic decision and avoids the model recommending overconfident reallocations.

Each compared plan reports: expected outcome, credible interval, channels driving the difference, and explicit warnings when the plan extrapolates beyond historical observed spend ranges.

Constrained inverse optimization ("How do I hit X?") is deferred to v1.1.

**FR-5.3 — Plain-English Executive Summary**
A deterministic, template-driven narrative report summarizing the key findings. The template adjusts language based on confidence level (high-confidence channels get definitive statements; low-confidence channels get hedged language with explicit uncertainty framing). No LLM involvement; the templates are code.

*Acceptance:* The executive summary is validated through a structured user test with at least 10 testers drawn from the primary, secondary, and CMO personas (no statisticians). The tester pool must include at least 4 marketing analysts, 2-3 agency analytics leads, and 2-3 CMOs or VP-level marketing decision-makers. Each tester reads the summary independently without clarifying questions, then completes a 5-point Likert assessment on three dimensions:

- *Comprehension:* "I understood what this report said about my marketing spend."
- *Decision-readiness:* "I have a clear sense of what action this report suggests."
- *Trust calibration:* "Where the report expressed uncertainty, I understood why."

Acceptance threshold: average Likert score ≥ 4.0 on each dimension across the tester pool, with no single dimension scoring below 3.5 from any individual tester. Testers also identify any sentences they found confusing or unclear; sentences flagged by ≥ 30% of testers must be revised before v1 release.

**FR-5.4 — Model Trust Card**
A single-page summary with named credibility dimensions. Each dimension is reported with a status (pass / moderate / weak) and a one-line explanation. v1 dimensions:

- Convergence (R-hat, effective sample size)
- Holdout accuracy
- Refresh stability (when applicable)
- Prior sensitivity
- Saturation identifiability per channel
- Lift-test consistency (if calibration data provided)

The Trust Card is the connective tissue between the model output and the recommendation: weak status on a dimension translates into specific hedged language in the executive summary and specific warnings in scenario comparison.

**FR-5.5 — Experiment Advisor (Minimal v1)**
Generates a prioritized list of channels that would most benefit from experimental validation. The v1 version identifies channels where the posterior credible interval is wide and the spend is large enough that better information would meaningfully change recommendations. Output is in the form: "Channel X has high posterior uncertainty (CI: $Y to $Z) and significant spend ($W); a controlled experiment would substantially improve confidence."

The full v1.1 version will additionally recommend experiment designs (geo holdout vs. budget split, sample size, duration, geo selection).

### 5.6 Distribution and Installation

**FR-6.1 — pip Installable**
`pip install wanamaker` should produce a working installation on Linux, macOS (Intel and Apple Silicon), and Windows without requiring R, GPU drivers, or manual jaxlib downloads.

*Acceptance:* Clean-environment installation succeeds on all three OS platforms in CI testing for every release.

**FR-6.2 — Docker Container**
An official Docker image is published with all dependencies pre-resolved. This is the recommended path for users who don't want to manage Python environments.

**FR-6.3 — CLI**
The CLI provides the core workflow: `wanamaker diagnose`, `wanamaker fit`, `wanamaker report`, `wanamaker forecast`, `wanamaker compare-scenarios`, `wanamaker refresh`. All commands accept a YAML config file as input.

**FR-6.4 — Python API**
A Python API mirrors the CLI for users who want programmatic access. The high-level API hides priors and sampler internals; a lower-level API exposes the underlying model objects for expert users.

### 5.7 Three-Layer Progressive Disclosure

**FR-7.1 — Layer 1: Defaults Only**
A user with a CSV and a YAML naming the date and target columns should be able to run a complete analysis end-to-end without specifying any priors, transformations, or sampler arguments. The defaults must produce a sensible result for the typical mid-market case.

*Acceptance:* On the public example dataset, a new user can go from `pip install wanamaker` to a complete executive summary in under 30 minutes, following only the quickstart documentation.

**FR-7.2 — Layer 2: Guided Configuration**
The YAML config supports configuration of: channel category mappings, business constraints (min/max spend per channel), control variables, lift test CSV path, runtime mode, anchoring weight. Documentation provides example configs for common situations.

**FR-7.3 — Layer 3: Expert Overrides**
The Python API exposes: prior specification per channel, custom adstock and saturation functions, sampler tuning parameters, alternative likelihoods, full posterior objects. The expert layer is documented but is not the path users start on.

---

## 6. Privacy and Local Processing

Wanamaker is a local-first tool. The data being modeled — marketing spend, revenue, conversions — is commercially sensitive at mid-market brands and is subject to client confidentiality obligations at agencies. Many target users will be unable to adopt a tool that transmits this data anywhere, regardless of methodological merit. This is not a compliance footnote; it is a foundational design constraint.

**FR-Privacy.1 — No Network Calls in Core Operations**
The core CLI commands (`wanamaker diagnose`, `wanamaker fit`, `wanamaker report`, `wanamaker forecast`, `wanamaker compare-scenarios`, `wanamaker refresh`) must not make any outbound network calls. This is enforced architecturally — the relevant code paths must not import or invoke any HTTP client, telemetry library, or remote service.

*Acceptance:* CI test verifies that running each core command in a network-isolated container produces identical output to running with network access. Any network call is a build failure.

**FR-Privacy.2 — Local Artifact Storage**
All model artifacts — input data hashes, posterior summaries, configuration snapshots, refresh diff records, generated reports — are stored in a project-local `.wanamaker/` directory. The user's home directory and global configuration are not modified by default. Users can configure an alternate artifact directory via YAML or CLI flag.

**FR-Privacy.3 — No Telemetry by Default**
Wanamaker collects no usage analytics, crash reports, model metadata, or any other information about how the tool is used. This includes no anonymized telemetry. If a future version offers optional telemetry to support development, it must be:

- Disabled by default
- Opt-in only via explicit user action (not opt-out)
- Documented transparently with a clear list of what is collected
- Easy to disable per-run and per-installation
- Subject to explicit version-bumped consent if the data collected ever changes

This is a permanent design constraint, not a v1 limitation.

**FR-Privacy.4 — Documentation of Data Residency**
The documentation includes an explicit "Privacy and Data Handling" section addressing the concerns of agency users with client confidentiality obligations. The section must clearly state that no data leaves the user's machine under any normal operation, identify the specific files written to disk and where, and explain how to use the tool in air-gapped environments.

**Strategic note.** Local-first processing is not just compliance; it is positioning. Every commercial MMM tool is cloud-based and therefore involves some level of trust about data handling. "Wanamaker runs on your laptop and never phones home" is a real differentiator in client meetings, in legal review, and in any environment where data residency matters. The project should lean into this rather than treat it as a constraint.

---

## 7. Non-Functional Requirements

**NFR-1 — Runtime Performance**
See FR-3.5 for the runtime tier specifications. The Standard tier (under 30 minutes on a modern laptop CPU for 150 weeks × 12 channels at the national level) is the default and the tier against which adoption will be evaluated.

**NFR-2 — Reproducibility**
Given the same input data, the same configuration, and the same random seed, model fits must be bit-for-bit reproducible.

**NFR-3 — Documentation Quality**
Documentation is a v1 deliverable, not a follow-up. The documentation site must include the following sections at minimum:

- Installation guide (covering pip install and Docker paths)
- Quickstart tutorial using the public benchmark dataset
- Full API reference
- An "Analyst's Guide" covering MMM concepts at a working level
- A "What Wanamaker Tells Your CMO" section explaining outputs in business terms
- A "Privacy and Data Handling" section addressing local-first processing and agency confidentiality concerns (per Section 6)
- A **"Comparison to Other Tools"** page covering Wanamaker vs. Robyn, Meridian, Recast, and PyMC-Marketing — clearly identifying where each tool fits best, what Wanamaker does differently, and where users should pick a different tool. Transparency builds credibility better than self-praise; we describe competitor strengths honestly.

The README must include a **one-command end-to-end example** using the public benchmark dataset, runnable in under 5 minutes from `pip install`. The example produces a complete executive summary and demonstrates the core value proposition without requiring any configuration beyond the dataset.

The Robyn project's documentation quality is the bar.

**NFR-4 — Honest Failure Modes**
When the tool cannot produce a credible answer, it must say so clearly. Silent failure is the single largest risk to user trust and is unacceptable.

**NFR-5 — Refresh Accountability Targets**
Across a typical refresh (4 additional weeks of data added to a 150-week history) on the benchmark refresh dataset:
- Average historical contribution estimate movement should not exceed 10%
- No single channel's historical estimate should move by more than 25%
- At least 90% of movements should be classifiable into a non-"unexplained" category

These are targets, not hard constraints; they are the metrics we hold ourselves accountable to.

**NFR-6 — Cross-Platform Support**
Linux, macOS (Intel and Apple Silicon), and Windows must all be first-class supported platforms. JAX/dependency hell is the single biggest reason LightweightMMM lost adoption; the engine choice in Phase -1 must demonstrate this is solvable.

**NFR-7 — Benchmark Datasets**
The following benchmark datasets are v1 deliverables, used both for acceptance testing and for documentation:

- **Synthetic ground-truth dataset** — known media contributions, used to validate model recovery
- **Public example dataset** — realistic, anonymized, used in quickstart and tutorials
- **Refresh stability benchmark** — known-stable dataset with 4 additional weeks, used to validate refresh accountability
- **Low-variation channel benchmark** — includes a spend-invariant channel, used to validate spend-invariant handling
- **Collinearity benchmark** — includes correlated channels, used to validate diagnostic warnings
- **Lift-test calibration benchmark** — synthetic data with known lift, used to validate calibration math
- **Target leakage benchmark** — includes a variable derived from the target, used to validate leakage detection
- **Structural break benchmark** — includes a known break point in the target series, used to validate break detection

---

## 8. Success Metrics

### 8.1 Usage-Quality Metrics (Primary)

These measure actual value delivered, not just attention:

- **Completed diagnostic runs** — users get past the data hurdle
- **Completed model fits** — installation and runtime work
- **Generated executive summaries** — end-to-end value delivered
- **Repeat refresh runs by the same user** — recurring usage, not one-shot trial
- **GitHub issues asking about output interpretation** — users are actually trying to make decisions from the output (proxy for serious adoption)
- **External notebooks / blog posts demonstrating use** — community learning beyond the maintainer

Targets are placeholders pending v1 launch and should be set against benchmark data once available.

### 8.2 Adoption Metrics (Secondary)

These measure community signal:

- GitHub stars (target: 500+ in year one; Robyn has roughly 1,500 in five years, LMMM had 1,000 at archival)
- Monthly PyPI downloads (target: 1,000+ steady-state by month 12)
- Active GitHub Discussions (target: comparable to early Robyn — 5+ active threads per week)
- Number of public case studies or blog posts by users not affiliated with the project (target: 10+ in year one)

### 8.3 Strategic Metrics

- Citation in industry comparison articles alongside Meridian, Robyn, PyMC-Marketing
- Adoption by at least one mid-market agency for live client work
- Inclusion in at least one major analyst report (Forrester / Gartner / IDC) — aspirational, not a year-one commitment

### 8.4 Quality Metrics

- Refresh stability: meets NFR-5 targets on benchmark datasets
- Time-to-first-result: <30 minutes from `pip install` to first executive summary on the public example dataset
- Documentation completeness: 100% public API coverage, all CLI commands documented with examples
- Cross-platform CI: green on Linux, macOS (Intel + Apple Silicon), Windows for every release

---

## 9. Technical Architecture (Preliminary)

This section sketches the technical approach at a high level. Detailed technical design is a separate document. Engine selection is the output of Phase -1 (see Section 11).

**Language:** Python 3.11+

**Statistical engine:** To be determined by Phase -1 spike. The decision criterion is *"can the target user install it on Windows in five minutes and get a credible result on a laptop CPU,"* not statistical elegance.

Current candidate hypotheses, in approximate order of presumed fit:

- **PyMC** — leading candidate. Installs cleanly via pip on all major platforms following the move away from the legacy theano/aesara stack. Marginally slower than NumPyro for very large models but well within tolerance for the v1 design center (150 weeks × 12 channels at the national level). Mature and widely understood in the data science community, which is a meaningful adoption signal for the senior-data-scientist persona who will vet the tool.
- **NumPyro / JAX** — technically excellent, the original v0.1 presumption. Demoted because of installation friction history (see LightweightMMM). Worth re-evaluating in Phase -1 in case the JAX ecosystem has stabilized sufficiently on Windows and Apple Silicon to be the right answer.
- **Stan via cmdstanpy** — viable. Compiles a small C++ toolchain on first install, which adds friction for the target persona. Library quality is unmatched and the stability story is strong.

A non-Bayesian engine such as xgboost was considered for the modeling role and rejected. Tree-boosting models do not produce posterior distributions over channel-level parameters (ROI, saturation parameters, adstock half-life), which means they cannot deliver the credible intervals, lift-test calibration, or refresh accountability that constitute Wanamaker's value proposition. Conformal prediction can produce honest forecast intervals from a tree model, but conformal intervals are about predictions, not parameters — and MMM is fundamentally a parameter-estimation problem dressed as a prediction problem. The detailed reasoning is preserved in the project decision log.

**xgboost in a supporting role.** xgboost is a candidate for two non-modeling uses where its strengths (speed, ease of installation, well-understood behavior) help and its lack of native parameter inference is irrelevant:

- **Quick mode preview.** A fast tree-based forecast with conformal intervals as a sanity check before the full Bayesian fit. Output is explicitly labeled as a forecast preview, not as ROI estimates.
- **Validation cross-check.** An independent tree-based fit used to detect specification problems in the Bayesian model. If the Bayesian and tree forecasts disagree substantially, the Trust Card raises a flag.

These uses are candidates pending Phase -1 evaluation, not commitments.

**Data handling:** pandas for I/O, polars internally if performance becomes a concern.
**Plotting:** matplotlib + seaborn for static reports, optionally plotly for interactive HTML.
**Reporting:** Jinja2 templates rendering to HTML and Markdown.
**Configuration:** YAML, validated with pydantic.
**CLI:** typer or click.
**Distribution:** PyPI (pip-installable wheel) and Docker Hub (official image).
**Documentation:** mkdocs-material (matches Robyn's documentation pattern, which users praise).

The dependency installation question is the single biggest open architectural risk. Phase -1 will resolve it.

---

## 10. Risks and Open Questions

### 10.1 Technical Risks

**R-1 — Statistical engine installation friction.** Mitigation: Phase -1 engine spike before architecture lock-in; Docker as a primary recommended distribution path; willingness to choose the less-elegant-but-more-installable engine.

**R-2 — Refresh anchoring weight is hard to tune.** A model anchored too lightly will oscillate; anchored too heavily will fail to update on real evidence. Mitigation: prototype anchoring early on the refresh benchmark dataset; iterate on the default weight before locking the design; ship with the anchoring weight user-configurable.

**R-3 — Default priors per channel category may be wrong for some industries.** A subscription business has different adstock characteristics than a CPG brand. Mitigation: start with a single set of generic defaults documented with the empirical/theoretical basis; collect community feedback in v1; consider vertical-specific defaults in v2.

**R-4 — Spend-invariant channel handling could mask real problems.** If a channel is spend-invariant because the analyst forgot to include real variation, our "graceful handling" hides the issue. Mitigation: the diagnostic step warns prominently before model fitting; the Trust Card flags it; the report explicitly says the channel's contribution is prior-driven.

### 10.2 Adoption Risks

**R-5 — The "yet another MMM" problem.** The market is crowded. Differentiation through positioning ("knowing which half") and through actual unmet needs (refresh accountability, data readiness, decision-oriented outputs) is essential. Mitigation: launch with a clear narrative blog post explaining what's different; don't launch with a feature list.

**R-6 — Documentation quality.** LMMM died partly because Google didn't resource the docs. Mitigation: documentation as a v1 deliverable with full coverage; "What Wanamaker Tells Your CMO" section as a distinguishing piece of writing.

**R-7 — The "competent analyst" persona may not exist in sufficient numbers.** Mitigation: Phase -1 persona validation interviews before significant build investment.

### 10.3 Strategic Risks

**R-8 — A major MMM vendor open-sources their tool.** Recast, Measured, or Haus could shift strategy. Mitigation: open-source ≠ free for the tool's vendor; even if a competitor opens source, the operational support layer is still a moat for them and our positioning still holds for unsupported users.

**R-9 — A new well-funded entrant.** Anthropic, OpenAI, or another player ships a competing tool. Mitigation: speed to launch matters; the community-building head start is real; our specific differentiator (refresh accountability) is hard to retrofit.

### 10.4 Open Questions for Resolution

- **Engine choice:** NumPyro, PyMC, Stan, or hybrid? Resolution: Phase -1 spike.
- **Default anchoring weight:** What's the right calibration? Resolution: prototype on benchmark data in Phase 1.
- **Channel taxonomy completeness:** Are there industry verticals where the default taxonomy fails? Resolution: persona interviews in Phase -1.
- **Docker vs. pip as the primary recommended path:** Which gets featured in the README quickstart? Resolution: depends on Phase -1 engine outcome.
- **Lift-test calibration depth in v1 vs. v1.1:** How much calibration sophistication ships in v1 vs. waits? Resolution: time-box the v1 implementation; whatever is solid by Phase 2 ships, the rest goes to v1.1.
- **Commercial layer:** Is there a v2+ supported version with paid support? Resolution: defer until v1 adoption is real; persona interviews include the affordability question.

---

## 11. Phased Delivery Plan

### Phase -1: Engine Decision and Persona Validation (months 1-2)

Two parallel workstreams that gate Phase 0.

**Workstream A: Statistical engine spike.** Implement a minimal MMM (adstock + saturation + Bayesian fit) in each candidate engine: NumPyro/JAX, PyMC, and Stan via cmdstanpy. Test installation on Linux, macOS (Intel + Apple Silicon), and Windows in clean VMs. Test runtime against the synthetic ground-truth benchmark. Decision criterion: *can the target user install it on Windows in five minutes and get a result in under 30 minutes on a laptop CPU?* Output: a decision memo with the engine choice and the empirical basis.

**Workstream B: Persona validation.** Conduct structured interviews per Section 3.4. Critical question per interviewee: "Would you personally install and use this, or would you only think it's a good idea?" Secondary questions: data shape they have, willingness to use a CLI, alternative tools they currently use, what they would pay for support. Output: validated persona profile or revised target persona.

**Gate:** Both workstreams complete. Engine chosen. Persona validated or scope revised.

### Phase 0: Internal Prototype (months 3-4)

- Core Bayesian model with chosen engine
- Adstock and saturation transformations
- Synthetic test datasets (the benchmark suite)
- Basic posterior summary outputs
- Reproducibility infrastructure

**Gate:** Model fits and produces statistically valid output on synthetic ground-truth data; recovers known contributions within FR-3.1 acceptance criteria.

### Phase 1: Alpha (months 5-6)

- Data Readiness diagnostic with discrete levels
- Three Decision Modes (Explain, What-if, Refresh)
- Scenario comparison
- Forecasting via posterior predictive
- Trust Card v1
- Refresh accountability infrastructure (versioning, diff report, classification)
- Light anchoring prototype with iteration on default weight
- Documentation skeleton
- Benchmark dataset suite complete

**Gate:** End-to-end workflow runs on real-world dataset (from a friendly persona-validation interviewee); refresh stability targets met on benchmark; Trust Card surfaces all v1 dimensions correctly.

### Phase 2: Beta (months 7-8)

- Plain-English executive summary with template-based language adjustment
- Experiment Advisor (minimal channel-flagging version)
- Cross-platform installation hardening
- Docker image with CI testing
- Documentation completion (Analyst's Guide, "What Wanamaker Tells Your CMO," API reference)
- Selected friendly users complete real analyses end-to-end

**Gate:** Ready for public GitHub launch; documentation passes a "new user from quickstart to executive summary in 30 minutes" test.

### Phase 3: v1.0 Public Release (month 9)

- Public GitHub launch
- Launch blog post explaining the "trustable usability" thesis and refresh accountability
- HackerNews / r/datascience / LinkedIn announcements
- Active community management for first 30 days
- Issue triage and rapid bug-fix response

### Phase 4: Post-Launch and v1.1 (months 10-14)

- Bug fixes from community feedback
- Documentation improvements based on real user questions
- v1.1 features: constrained inverse optimization, Halo / channel-interaction diagnostic (with careful naming), full Experiment Advisor with experiment design, refined lift-test calibration
- v2 scoping based on real user demand signals

---

## 12. Appendix

### A. Naming and Voice

**Project name:** Wanamaker
**Tagline:** Knowing which half.
**Origin:** The name references the apocryphal John Wanamaker quote "Half the money I spend on advertising is wasted; the trouble is I don't know which half." The quote is widely attributed to him but there is no contemporary documentary evidence that he actually said it. The misattribution itself is appropriate — a quote about measurement uncertainty whose own provenance is uncertain — and the README will explain this honestly.

**Voice:** Dry, honest, slightly self-aware. Not "AI sparkle." Not breathless marketing. The product respects the user's intelligence and treats statistical honesty as the differentiator.

### B. Reference Documents

This BRD/PRD draws on four research reports analyzing existing MMM solutions:
- Meridian (Google) — feedback analysis
- Robyn (Meta) — feedback and complaints
- LightweightMMM (Google) — user feedback review
- Recast — research report

It also synthesizes Blue Ocean Strategy analyses produced by four independent AI systems (Claude, ChatGPT, Gemini, Grok) reviewing the same underlying research, plus three rounds of critical review of successive document versions: a Round 1 critical review of v0.1 (driving v0.2 changes), a Round 2 critical review of v0.2 (driving v0.3 changes), and a Round 3 critical review of v0.3 (driving v0.4 changes).

### C. Glossary

- **Adstock:** The carryover effect of marketing spend across time periods. A purchase today may be influenced by an ad seen last week.
- **Saturation:** The diminishing returns property of marketing spend. Each additional dollar produces less incremental impact than the last.
- **Hill function:** A specific mathematical form for saturation curves, parameterized by an EC50 (the spend level at half-maximum response) and a slope.
- **Posterior distribution:** In Bayesian statistics, the distribution of parameter values given the observed data and prior beliefs. The output of a Bayesian model fit.
- **Credible interval:** The Bayesian analog of a confidence interval. A 95% credible interval contains 95% of the posterior probability.
- **Posterior anchoring:** The technique of using a previous run's posterior as an informative prior for a new run, with a configurable weight that controls how much new data can update prior beliefs.
- **NUTS (No-U-Turn Sampler):** An efficient MCMC sampler well-suited to high-dimensional Bayesian models.
- **MCMC (Markov Chain Monte Carlo):** A family of algorithms for sampling from probability distributions. The methodology underlying NumPyro, PyMC, and Stan.
- **Lift test:** A controlled experiment (often geographic) measuring the incremental effect of a marketing intervention by comparing exposed and unexposed groups.
- **Halo effect:** The hypothesized phenomenon in which upper-funnel marketing (brand, TV) increases the efficiency of lower-funnel marketing (paid search, performance social). v1.1 will include a diagnostic for this; v1 does not.
