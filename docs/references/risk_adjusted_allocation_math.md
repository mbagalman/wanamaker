# Risk-Adjusted Allocation: Math Reference

Reference document for engineers and AI agents implementing or modifying the risk-adjusted ramp recommender in Wanamaker.

This file is the math companion to the design note in [`docs/internal/risk_adjusted_allocation.md`](../internal/risk_adjusted_allocation.md). The design note covers *why* the feature exists and how the verdicts (`proceed`, `stage`, `test_first`, `do_not_recommend`) read in product language. This file covers *what each formula is, where it comes from, and how the constants were chosen*. It is required reading before touching anything in [`src/wanamaker/forecast/ramp.py`](https://github.com/mbagalman/wanamaker/blob/master/src/wanamaker/forecast/ramp.py).

Wanamaker's ramp module is engine-neutral: the math runs entirely on the per-draw posterior predictive matrix `(n_draws, n_periods)` exposed via `PredictiveSummary.draws`. No PyMC types appear here.

---

## 1. Concepts

**Risk-adjusted allocation ramp.** Given a baseline plan `x0` (the user's current allocation) and a model-favored target plan `x_star` (typically the top-ranked entry from `compare_scenarios`), recommend the largest fraction `f` of the way from `x0` toward `x_star` that the posterior, the historical spend ranges, and the Trust Card all support.

**Not optimization.** The user supplies both plans. The recommender's job is to size a *partial move*, not to search for the best plan. v1 stays user-driven by design; constrained inverse optimization is BRD/PRD § 4.2 deferred work.

**Discrete verdicts, not a continuous score.** The output is one of four labelled categories. The numeric `recommended_fraction` is meaningful only as a member of the ladder `{0.10, 0.25, 0.50, 0.75, 1.0}`.

---

## 2. Plan Interpolation

For each channel `c` and period `t`, the ramped plan is a linear interpolation:

```text
x(f)[c, t] = x0[c, t] + f * (x_star[c, t] - x0[c, t])
```

Evaluated at:

```text
f in {0.10, 0.25, 0.50, 0.75, 1.0}
```

The ladder deliberately omits `f = 0` (the do-nothing fallback the recommender drops to when no positive fraction passes) and excludes intermediate values; the design note explains the trade-off between ladder granularity and audit clarity.

Misaligned periods or channel columns between `x0` and `x_star` raise before any engine call. The implementation lives in `_interpolate_plan` and `_validate_plans_align`.

---

## 3. Posterior Risk Metrics

Let `S` index posterior predictive draws (shape: one outcome trajectory per draw across the plan periods). Define the per-draw incremental outcome:

```text
delta_s(f) = sum_t outcome_s(x(f), t) - sum_t outcome_s(x0, t)
```

Crucially, both terms use the **same seed**, so `delta_s(f)` is paired by draw — within-engine sampling noise cancels, and the metrics below characterise model uncertainty about the *increment*, not about either plan in isolation.

### 3.1 Expected Increment

```text
E_delta(f) = (1/S) * sum_s delta_s(f)
```

The posterior mean improvement vs. the baseline plan. Reported in target units (revenue, conversions, ...).

### 3.2 Probability of Improvement

```text
P_positive(f) = (1/S) * sum_s 1{delta_s(f) > 0}
```

Default gate: pass if `P_positive(f) >= 0.75`.

### 3.3 Probability of Material Loss

The "material" threshold is a fraction of baseline outcome with a hard absolute floor:

```text
loss_threshold = max(min_absolute_loss, baseline_outcome_mean * min_relative_loss)
P_material_loss(f) = (1/S) * sum_s 1{delta_s(f) < -loss_threshold}
```

Defaults: `min_relative_loss = 0.05`, `min_absolute_loss = 0.0`. Default gate: pass if `P_material_loss(f) <= 0.10`.

### 3.4 Lower-Tail Risk: q05 and CVaR_5

The 5 % quantile of the increment distribution and the conditional mean *given* you're in the worst 5 % tail:

```text
q05(f)   = 5th percentile of delta_s(f)
CVaR_5(f) = mean(delta_s(f) | delta_s(f) <= q05(f))
```

Both are signed deltas: negative values indicate losses. The gate compares against a negative tolerance:

```text
cvar_tolerance = -cvar_relative_tolerance * loss_threshold
```

Default `cvar_relative_tolerance = 2.0`, so the gate fails when the average outcome conditional on the worst 5 % tail is more than `2 * loss_threshold` below baseline.

CVaR (also called expected shortfall) is preferred to a raw quantile because it answers *"if the bad tail happens, how bad is it on average?"* instead of just *"where is the cliff edge?"*. CVaR is a coherent risk measure in the sense of Artzner et al. (1999); plain VaR (a quantile) is not.

**Reference:** Rockafellar & Uryasev (2000); see § 7.

### 3.5 Largest Move Share

A reported (not gated) signal that flags one-channel-driven recommendations:

```text
moved[c]            = |sum_t x(f)[c, t] - sum_t x0[c, t]|
largest_move_share  = max_c moved[c] / sum_c moved[c]
```

`largest_move_share = 1.0` means the entire move is concentrated in one channel. This number is read alongside the Trust Card; v1 does not gate on it because the right cutoff interacts with per-channel weakness in non-obvious ways. See the design note's "Concentration of the Recommendation" section.

### 3.6 Extrapolation Severity

For each plan cell where `x(f)[c, t]` falls outside the channel's observed historical spend range `[observed_spend_min[c], observed_spend_max[c]]`, compute the relative overshoot:

```text
overshoot_above = (planned - observed_spend_max) / observed_spend_max
overshoot_below = (observed_spend_min - planned) / observed_spend_min
```

The candidate fails the extrapolation gate if any cell exceeds `extrapolation_severity_cap` (default `0.25`, i.e. 25 %) on either side.

This is a hard gate on *severity*, not on *presence*. Mild extrapolation surfaces as a warning flag without failing the gate; severe extrapolation routes the verdict toward `stage` or `test_first`.

---

## 4. Kelly-Inspired Sizing

The fractional-Kelly cap is the link between posterior risk and ramp size. It is *Kelly-inspired*, not raw Kelly: the design note explains why MMM allocations don't satisfy the assumptions of the original Kelly derivation (independent bets, known odds, no model-form uncertainty).

### 4.1 The Continuous-Time Kelly Approximation

Kelly (1956) derived the optimal fraction for a binary bet:

```text
f*_kelly = (b * p - q) / b
```

with win probability `p`, loss probability `q = 1 - p`, and odds `b`. For continuous returns, a Taylor expansion of expected log-utility yields the more useful approximation:

```text
f*_kelly ≈ mean(r) / variance(r)
```

where `r` is the per-period return ratio. This is the form derived in Thorp (2006) and used throughout quantitative finance.

**Reference:** Kelly (1956); Thorp (2006); see § 7.

### 4.2 Wanamaker's Per-Draw Estimator

In `_fractional_kelly`, the relative return per posterior draw is:

```text
r_s = delta_full_s / max(baseline_outcome_s, 1.0)
```

where `delta_full_s` is the per-draw increment for the *full* move (`f = 1.0`) and `max(..., 1.0)` is a numerical floor that prevents division by near-zero baseline outcomes (a defensive measure for unusual datasets; a real baseline outcome is always orders of magnitude above 1).

The raw Kelly fraction is the sample-variance estimator:

```text
kelly_raw = mean(r_s) / var(r_s, ddof=1)     if var(r_s) > 0
          = 0                                 otherwise
```

### 4.3 Clamping

`kelly_raw` is unbounded above. With a tight posterior — small variance, modest positive mean — it can easily push into the tens, at which point the `f <= fractional_kelly` gate becomes decorative. We clamp to `[0, 1]`:

```text
kelly_clamped = max(0, min(1, kelly_raw))
```

This keeps the Kelly term on the same scale as the candidate ramp fractions.

### 4.4 Fractional-Kelly Multiplier

The clamped Kelly is then multiplied by a Trust-Card-derived factor:

```text
fractional_kelly = kelly_clamped * model_confidence_multiplier
```

Multipliers:

| Trust Card status | Multiplier |
|---|---:|
| Any weak dimension | 0.10 |
| Any moderate (no weak) | 0.25 |
| Clean (or `None`) | 0.50 |

Scaling Kelly down ("fractional Kelly") trades long-run growth rate for lower drawdown probability. MacLean, Ziemba & Blazenko (1992) characterise the trade-off explicitly: half-Kelly (`0.5`) was chosen as the v1 ceiling for clean Trust Cards because it produces near-optimal growth with materially lower drawdown frequency than full Kelly. The lower multipliers (`0.25`, `0.10`) extend that conservatism when the Trust Card flags evidence-quality concerns.

**Reference:** MacLean, Ziemba & Blazenko (1992); see § 7.

### 4.5 Naming Discipline (Product Language)

The term "Kelly" does not appear in user-facing reports. The design note ("Kelly-Inspired Sizing") explains why: invoking gambling math invites either false precision or rejection-by-association. Internally, the gate is `fractional_kelly`; in reports it surfaces as a "sizing cap" justified by the risk metrics above.

---

## 5. Trust Card Cap

Independent of Kelly, the Trust Card directly caps the ramp:

| Trust Card status | Ramp cap |
|---:|---:|
| Any weak dimension | 0.10 |
| Any moderate (no weak) | 0.50 |
| Clean (or `None`) | 1.00 |

A candidate `f > ramp_cap` fails the `trust_card` gate. This gate operates on top of the Kelly cap, so the *binding* cap on any given run is `min(fractional_kelly, ramp_cap)`.

The two caps overlap deliberately. Kelly characterises uncertainty *of the increment*; the Trust Card characterises uncertainty *of the underlying model*. Both can fail a candidate; both belong in the gate set.

---

## 6. Decision Rule

Let `passing = {f : every gate passes for f}`. Let `evidence_gates = {trust_card, extrapolation, fractional_kelly}`.

### 6.1 If `passing` is non-empty

Choose `f_star = max(passing)`. Examine the candidates with `f > f_star` (those that failed):

- If every higher-fraction failure list is a subset of `evidence_gates`: **`stage`**. A future refresh or experiment could lift the binding cap; recommend `f_star` and prepare to revisit.
- Otherwise (any higher fraction failed on a value/downside gate — `p_positive`, `p_material_loss`, `cvar_5`): **`proceed`**. The model genuinely doesn't believe in larger moves; `f_star` is the verdict.

### 6.2 If `passing` is empty

Inspect the smallest candidate's failures:

- If its failure list is a subset of `evidence_gates`: **`test_first`**. The block is about evidence quality, not expected value. The next useful action is a controlled experiment or a refresh after more data, not a smaller move.
- Otherwise: **`do_not_recommend`**. The block is about expected value or downside risk. The model does not currently support this reallocation in any form.

### 6.3 Up-Front Blocks

Two conditions short-circuit the recommendation before any candidate metric is computed:

- **Spend-invariant reallocation:** if the target plan moves any channel marked `spend_invariant` in the posterior summary (FR-3.2), the verdict is immediately `do_not_recommend` with `blocking_reason = "spend_invariant_reallocation"`. Saturation cannot be estimated for those channels, so the model is in no position to endorse moving spend into or out of them.
- **Plan misalignment:** if `x0` and `x_star` disagree on periods or channel columns, the call raises `ValueError` before evaluating any candidate.

---

## 7. References

These are the canonical sources for the math above. Statistical functions in `src/wanamaker/forecast/ramp.py` should cite at least one of these in their docstring when implementing or modifying the corresponding formula.

**Primary (formulas the code uses directly):**

- Kelly, J. L. (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal, 35(4): 917–926. — original Kelly criterion; the position-sizing intuition.
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*. In Zenios, S. A., & Ziemba, W. T. (Eds.), *Handbook of Asset and Liability Management*, Volume 1: Theory and Methodology. North-Holland. — accessible derivation of the continuous-time `f* ≈ μ / σ²` form used in `_fractional_kelly`.
- MacLean, L. C., Ziemba, W. T., & Blazenko, G. (1992). *Growth versus Security in Dynamic Investment Analysis*. Management Science, 38(11): 1562–1585. — characterises the growth-vs-drawdown trade-off that justifies the `0.5 / 0.25 / 0.10` Trust-Card-conditional multipliers.
- Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3): 21–41. — canonical formulation of CVaR (a.k.a. expected shortfall) as a tractable, optimisable risk measure.

**Background (cited for context, not directly implemented):**

- Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999). *Coherent Measures of Risk*. Mathematical Finance, 9(3): 203–228. — defines the coherence axioms (sub-additivity, monotonicity, positive homogeneity, translation invariance) under which CVaR is coherent and plain VaR is not.

**Wanamaker-internal:**

- [`docs/internal/risk_adjusted_allocation.md`](../internal/risk_adjusted_allocation.md) — design note: rationale, language, Wanamaker-specific gate set, and the v1 verdict categories.
- [`src/wanamaker/forecast/ramp.py`](https://github.com/mbagalman/wanamaker/blob/master/src/wanamaker/forecast/ramp.py) — implementation; module docstring's `References` block points back to this file and to the primary sources above.

---

## 8. Summary

For Wanamaker v1:

1. The ladder is `{0.10, 0.25, 0.50, 0.75, 1.0}`. `f = 0` is the implicit do-nothing fallback.
2. All risk metrics are computed paired-by-draw against the baseline outcome trajectory.
3. The fractional Kelly is the continuous-time approximation (`mean(r) / var(r)`), clamped to `[0, 1]`, then scaled by a Trust-Card-derived multiplier (`0.5 / 0.25 / 0.10` for clean / moderate / weak).
4. CVaR_5 is signed and gated against a negative tolerance (`-cvar_relative_tolerance * loss_threshold`).
5. Spend-invariant channel reallocation is an up-front block; it never produces a positive `recommended_fraction`.
6. The decision rule distinguishes evidence-quality failures from value/downside failures: the former route to `stage` or `test_first`, the latter to `proceed` (at the largest passing fraction) or `do_not_recommend`.
7. The term "Kelly" does not appear in user-facing reports; product language stays at "risk-adjusted ramp" and "sizing cap."

When in doubt, prefer the more conservative gate. The product position is "the model can advise on direction; the user owns the decision to commit the full budget."
