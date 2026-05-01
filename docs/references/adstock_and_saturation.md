# Adstock and Saturation: Technical Reference

Reference document for engineers and AI agents implementing or modifying adstock (carryover) and saturation transformations in Wanamaker.

This file is required reading before touching any code in `src/wanamaker/transforms/` or any default-prior code in `src/wanamaker/model/`. It captures the canonical formulas, the empirically-grounded parameter ranges per channel category, the estimation method tradeoffs, and the diagnostic patterns that distinguish a correctly-specified transformation from a mis-specified one.

Wanamaker uses **geometric adstock as the default carryover** and **Hill saturation as the default saturation**. Weibull adstock is available as a per-channel override. Other functions are not in v1 scope.

---

## 1. Concepts

**Adstock** (also called carryover) refers to the lagged effect of advertising on consumer behavior — a portion of an ad's impact persists and decays over subsequent time periods. Adstock is a *temporal* transformation applied to a media variable.

**Saturation** refers to the diminishing returns property of marketing spend — each incremental dollar produces less incremental impact than the last. Saturation is a *non-temporal* transformation applied to a media variable.

In Wanamaker, the two transformations are applied **in sequence**: first adstock (temporal smoothing), then saturation (non-linear response). This ordering matches Robyn, LightweightMMM, PyMC-Marketing, and the underlying Bayesian MMM literature (Jin et al., Google Research, "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects").

---

## 2. Adstock Functions

### 2.1 Geometric (Exponential) Decay — Default

The simplest and most widely used carryover function. One parameter, `theta`, representing the fraction of an ad's impact that carries over to the next time period.

**Formula (recursive):**

```
A_t = X_t + theta * A_{t-1}
```

with `A_0 = 0`, where `X_t` is the spend at time `t` and `A_t` is the adstocked value.

**Equivalent closed form:**

```
A_t = sum over k=0 to t-1 of: theta^k * X_{t-k}
```

**Parameter:** `theta ∈ [0, 1)`. `theta = 0` means no carryover; `theta` close to 1 means very long-lived effects.

**Half-life conversion:** if a channel has half-life `h` in periods, then `theta = 0.5^(1/h)`. So:
- 1 period half-life → theta = 0.5
- 2 period half-life → theta ≈ 0.707
- 4 period half-life → theta ≈ 0.841
- 8 period half-life → theta ≈ 0.917

Wanamaker priors are specified in **half-life space** (more interpretable for users) and converted to theta internally.

### 2.2 Weibull Decay — Optional, Per-Channel Override

A two-parameter flexible alternative that can capture a delayed peak or non-exponential decay shapes. Useful for channels with slow build-up (brand TV, sponsorships) or platform behavior that doesn't fit a constant decay rate.

**Two variants:**

- **Weibull CDF adstock**: produces an S-shaped cumulative carryover. With shape = 1, this reduces to geometric decay. With shape > 1, decay accelerates over time.
- **Weibull PDF adstock**: produces a "lagged peak" — impact builds, peaks, then decays. Useful when ad effect requires multiple exposures before peak response.

**Parameters:** `shape > 0`, `scale > 0`. Wanamaker default: Weibull CDF with `shape = 2`, `scale = 2` as a reasonable starting point if a user enables Weibull on a channel.

**When to use Weibull over geometric:**
- Channel exhibits delayed-peak behavior (TV brand campaigns, podcast sponsorships)
- Geometric fits poorly in residual diagnostics (see Section 6)
- User has prior knowledge that decay isn't constant

**When NOT to use Weibull:**
- Default case — geometric is sufficient for most channels
- Sparse data — Weibull's two parameters need data to identify
- Stakeholder communication is primary concern (half-life is intuitive; shape/scale is not)

The Robyn project documents that Weibull adstock requires significantly more iterations to converge than geometric. This matches our engine performance expectations.

### 2.3 Functions NOT in v1 Scope

- **Distributed lag with custom weights** (LightweightMMM's "carryover" model): too flexible, hard to identify
- **Negative binomial / Gamma decay**: niche; equivalent to choosing a different parametric form
- **Custom convolutions**: out of scope for v1

If a future user need emerges, these can be added in v2. v1 ships with geometric (default) and Weibull (per-channel override) only.

---

## 3. Saturation Functions

### 3.1 Hill Function — Default

An S-shaped saturation curve with two parameters. Models diminishing returns and can capture threshold effects (low spend has minimal impact, then rapid response, then plateau).

**Formula:**

```
S(x) = x^alpha / (x^alpha + gamma^alpha)
```

where:
- `alpha` (shape) controls curvature. `alpha > 1` produces a more pronounced S-shape; `alpha < 1` produces a concave (no threshold) curve.
- `gamma` (half-saturation point) is the spend level at which response reaches half its asymptotic maximum.

**Parameter ranges (Wanamaker default priors):**
- `alpha`: weakly-informative prior centered around 1.5, range roughly [0.5, 3.0]
- `gamma`: scaled to the channel's observed spend range; default centered around the median spend, ranging across the observed distribution

**Important:** Hill parameters are not identifiable via ordinary linear regression. They require Bayesian inference, non-linear least squares, or grid search. Wanamaker uses Bayesian inference via the engine layer.

### 3.2 Functions NOT in v1 Scope

- **Logistic saturation**: similar shape to Hill but parameterized differently; Hill is more flexible
- **Power transformation** (`y = x^beta` with `beta < 1`): simpler but no threshold capability
- **Logarithm** (`y = log(1 + x)`): simpler but rigid concave shape

Hill subsumes the use cases of these alternatives via parameter choices.

---

## 4. Default Priors by Channel Category

The priors below are placeholders pending Phase 1 empirical tuning against benchmark datasets. They reflect the consensus from the MMM literature (Robyn documentation, Jin et al., Recast case studies, industry surveys) and are designed to be weakly informative — they nudge the model toward reasonable values without dominating the data.

### 4.1 Adstock Priors (Half-Life, in Weeks)

| Channel category | Half-life prior median | Rationale |
|---|---|---|
| Paid search (brand) | 0.5–1 week | Direct response, intent-driven, very short carryover |
| Paid search (non-brand) | 1 week | Direct response with some retargeting tail |
| Paid social (performance) | 1–2 weeks | Short-lived attention; some retargeting |
| Paid social (brand) | 2–3 weeks | Slightly longer for awareness-oriented social |
| Display / programmatic | 1–2 weeks | Short impressions effect |
| Online video (YouTube etc.) | 2–4 weeks | Mix of awareness and response |
| Connected TV (CTV) | 3–5 weeks | Awareness-leaning, longer carryover than digital video |
| Linear TV | 4–8 weeks | Brand-building, long memory |
| Audio / podcast | 3–6 weeks | Episodic listening, moderate carryover |
| Affiliate | 1 week | Conversion-driven, short |
| Email / CRM | 0.5–1 week | Immediate engagement |
| Promotions / discounting | 0–1 week | Often immediate; depends on promotion type |
| Out-of-home (OOH) | 4–8 weeks | Awareness, persistent ambient exposure |
| Direct mail / print | 2–4 weeks | Slower reach, moderate carryover |

**Conversion to theta:** Wanamaker stores priors in half-life space and converts to theta. Users see "median half-life of TV is 6 weeks"; the model uses theta ≈ 0.891.

### 4.2 Saturation Priors (Hill alpha and gamma)

For all channels, default Hill priors are weakly informative:

- **alpha**: prior centered around 1.5 (slight S-shape), range [0.5, 3.0]. Wider range allowed only if user explicitly overrides.
- **gamma**: prior centered around the median observed spend per period for that channel, with a scale matching the spread of observed spends. This is data-driven (an "empirical Bayes" element); users can override with explicit values if they have prior knowledge.

The Hill function alone (without channel-specific data) is hard to specify priors for. Wanamaker's approach: scale gamma to the channel's observed spend range, then apply weak priors centered on that scale. Document this clearly in the executive summary so users understand that saturation curves are partially data-driven.

### 4.3 Notes on Priors

- These are **weakly informative**, not rigid. They influence the posterior in low-data regimes but allow the data to dominate when there's signal.
- Users can override any prior per channel via the YAML config.
- The lift-test calibration (FR-1.3) replaces the channel's ROI prior with an informative prior derived from the test result. This does *not* override the adstock or saturation priors — those remain as configured.
- Spend-invariant channels (FR-3.2) use the default priors only; the data cannot update them, and outputs label the result as prior-driven.

---

## 5. Estimation Approaches

Wanamaker uses **Bayesian estimation** via the engine layer, with PyMC as the
current production backend. The adstock and saturation parameters are treated
as model parameters with priors and are estimated jointly with the channel
coefficients.

For reference, here are the alternatives that are *not* used:

### 5.1 Grid Search (Not Used)

Iterate `theta` from 0 to 1 in increments, fit the linear model at each, pick the best by validation error. Simple but doesn't scale to multi-parameter saturation, doesn't quantify uncertainty, and doesn't propagate parameter uncertainty into ROI estimates.

### 5.2 Heuristic Optimization (Not Used)

Robyn's approach: Nevergrad evolutionary optimization over decay and saturation parameters. Effective but produces point estimates without uncertainty, and contributes to refresh instability (small data changes produce different optima).

### 5.3 Non-linear Least Squares (Not Used)

Fit the entire model as one non-linear regression. Estimates everything jointly but doesn't quantify uncertainty, doesn't incorporate priors cleanly, and is sensitive to local optima.

### 5.4 Bayesian (Used in Wanamaker)

Specify priors on all parameters, sample the posterior via MCMC (NUTS). Provides:
- Joint estimation of channel coefficients, adstock, and saturation parameters
- Credible intervals for all parameters
- Natural integration with lift-test calibration (calibration becomes a prior modification)
- Convergence diagnostics (R-hat, ESS) that tell you when fitting failed

The engine layer (`src/wanamaker/engine/`) abstracts the specific Bayesian library. Transform code should be engine-agnostic where possible; engine-specific math goes in the engine module.

---

## 6. Diagnostics

When implementing, modifying, or debugging adstock and saturation code, run these checks against benchmark datasets.

### 6.1 Plot the Adstock Curve

Visualize the impulse response implied by the fitted theta (or shape/scale). For a channel with half-life prior of 4 weeks, the impulse response should drop to ~50% at week 4 and ~25% at week 8. If the fitted curve diverges substantially from the prior median in the absence of strong data signal, the prior may be too weak, the data may be informative in unexpected ways, or there may be a bug.

### 6.2 Plot the Saturation Curve

Visualize spend-vs-response for each channel. The curve should:
- Start near zero at zero spend
- Rise smoothly through the observed spend range
- Show diminishing returns past the EC50 (gamma)
- Clearly mark the boundary between observed spend (solid) and extrapolation (dashed) per FR-5.1

### 6.3 Residual Autocorrelation

After fitting, compute the autocorrelation function of the residuals. Significant autocorrelation at lags 1-4 weeks suggests carryover is mis-specified — either theta is too low (effect drops too fast) or the wrong family (geometric instead of Weibull). Plot as part of the Trust Card diagnostics.

### 6.4 Sensitivity to Decay Parameter

Vary theta in a small range around the posterior mean and refit. If the channel coefficient or contribution shifts dramatically, the parameter is poorly identified — likely a data variation problem (Section 6.5). If results are stable across reasonable theta values, the model is well-conditioned.

### 6.5 Identifiability Warnings

These should fire from the data readiness diagnostic (FR-2.2) but are worth listing here:

- **Channel with constant spend**: adstock and saturation parameters cannot be estimated from data; output is prior-driven only. Trust Card flags `weak` on saturation identifiability.
- **Highly collinear channels**: adstock vs. saturation effects can be confused with each other when correlated. Diagnostic flags collinearity above 0.85.
- **Insufficient observations**: with fewer than 78 weekly observations, the data may not support estimating two-parameter Hill saturation per channel. Diagnostic warns when variable count exceeds observations / 10.
- **Spend variance below threshold**: a channel whose spend coefficient of variation is below 0.1 is effectively constant. Treat as spend-invariant.

### 6.6 Counterfactual Sanity Check

Set channel spend to zero in the fitted model and verify the predicted target metric decays gradually (not instantly) — that's the adstock at work. Compare the decay shape to the fitted adstock curve. They should match.

### 6.7 Business Sanity Check

Half-life translations should match common sense:
- TV decay > 4 weeks: plausible
- Search decay > 2 weeks: implausible, likely overfit
- Display decay > 3 weeks: implausible, likely confounded with other channels
- Any channel with theta > 0.95: likely overfit

If the posterior median for a channel falls outside the prior range, investigate: the data may be informative in an unexpected way, or there may be a specification problem.

---

## 7. Implementation Notes

### 7.1 Numerical Stability

- Compute adstock recursively rather than via the full sum to avoid numerical issues with long histories
- For Hill saturation, work in log-space when `x` is large to avoid overflow
- Clip parameters at boundaries (theta ∈ [0, 0.999]) to avoid degenerate cases

### 7.2 Vectorization

Adstock is sequential by definition (each value depends on the previous). For per-channel parallelism, vectorize across channels but iterate within each channel's time series. The engine layer should expose a pre-compiled adstock kernel.

### 7.3 Interaction with Lift Calibration

When a channel has a lift test, the calibration modifies the *coefficient* prior, not the adstock or saturation priors. The transformations remain estimated from data with their default priors. Document this clearly in the calibration module.

### 7.4 Refresh Anchoring (FR-4.4)

Per the BRD, light posterior anchoring applies to marginal posteriors of channel-level parameters: ROI/coefficient, adstock half-life, saturation slope (alpha), saturation EC50 (gamma). Anchoring is mixture-prior form:

```
Prior_new(theta_channel) = (1 - w) * Prior_default(theta_channel) + w * Posterior_previous(theta_channel)
```

Apply this independently to each channel-level parameter. Joint structure (correlations between alpha and gamma within a channel, or between channels) is not anchored — it regenerates from the new fit.

### 7.5 Spend-Invariant Handling

When a channel's spend coefficient of variation is below 0.1:
- Skip MCMC estimation of saturation parameters for that channel — fix them at the prior median
- The channel's coefficient still gets estimated
- The output report and Trust Card flag this prominently
- Saturation curves are NOT shown for the channel (per FR-3.2 — "saturation cannot be estimated from observed data")
- Surface to the Experiment Advisor as a high-priority candidate for testing

---

## 8. References

These are the canonical sources for the formulas and parameter recommendations above. Statistical functions in Wanamaker's codebase should cite these in their docstrings.

**Primary:**

- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects*. Google Research. https://research.google.com/pubs/archive/46001.pdf — canonical paper on joint Bayesian estimation of carryover and saturation
- *An Analyst's Guide to MMM*. Robyn (Meta). https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/ — practical guidance on Weibull, Hill, and parameter ranges
- *Adstock in Marketing Mix Modeling*. Forecastegy. https://forecastegy.com/posts/adstock-in-marketing-mix-modeling/ — comparison of adstock function families
- *PyMC-Marketing MMM Example*. https://www.pymc-marketing.io/ — reference Bayesian implementation of geometric + logistic and geometric + Hill

**Historical:**

- Broadbent, S. (1979). *One Way TV Advertisements Work*. Journal of the Market Research Society — original "adstock" terminology
- Hanssens, D. M., Parsons, L. J., & Schultz, R. L. (2001). *Market Response Models: Econometric and Time Series Analysis*. Kluwer — foundational textbook for response function estimation

**Tool documentation:**

- Robyn (Meta): https://facebookexperimental.github.io/Robyn/
- LightweightMMM (Google, archived): documented in repository archives
- Meridian (Google): https://developers.google.com/meridian/
- PyMC-Marketing: https://www.pymc-marketing.io/

When implementing a transformation, the docstring should reference at least one source. When choosing a default parameter range, the rationale should reference an empirical or theoretical source.

---

## 9. Summary

For Wanamaker v1:

1. Geometric adstock is the default. Weibull is per-channel override.
2. Hill saturation is the default. No alternatives in v1.
3. Apply adstock first, then saturation.
4. Priors are specified in half-life space (intuitive); converted to theta internally.
5. Channel-category-driven default priors per Section 4.
6. Bayesian estimation via the engine layer; no grid search, no Nevergrad, no NLS.
7. Spend-invariant channels: fix transformations at prior median, do not show saturation curves, flag in Trust Card and report.
8. Refresh anchoring applies to marginal posteriors of channel-level transformation parameters.

When in doubt, prefer the simpler functional form (geometric over Weibull) and the wider prior (less informative). Add complexity only when the data demands it and the diagnostic checks confirm the more complex form is justified.
