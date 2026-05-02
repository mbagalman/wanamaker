# Risk-Adjusted Allocation Ramps

> **Note.** This is the **design note** — rationale, language, and the
> v1 verdict categories. The formulas, the published sources for each
> formula, and the per-call-site citation map live in
> [`docs/references/risk_adjusted_allocation_math.md`](../references/risk_adjusted_allocation_math.md).
> Read this doc for *why*, that one for *what each line of math is and
> where it comes from*.

## Purpose

Wanamaker should not tell a user to make a large, unfamiliar marketing shift just
because one fitted model says the new allocation has the highest expected
outcome. That is exactly the kind of model overconfidence the product is meant
to avoid.

This note proposes a future feature for deciding how much of a model-favored
allocation change to adopt. The recommended framing is not "optimized budget."
It is:

> Given a current plan and a model-favored plan, how far toward the new plan is
> analytically justified right now?

The answer should be a staged ramp fraction with explicit risk language. This
fits Wanamaker's existing direction: scenario comparison, Trust Card warnings,
forecast uncertainty, and no constrained inverse optimization in v1.

## Prerequisites

Risk-adjusted ramps need **per-draw posterior predictive outcomes**, not just
the mean and HDI bounds we currently expose through `PredictiveSummary`. Every
probability metric below — `P_positive`, `P_material_loss`, `q05`, `CVaR_5` —
is computed on the matrix of `(n_draws, n_periods)` outcome samples for each
candidate plan. The PyMC engine already produces this matrix internally before
collapsing it to summary statistics; the foundational work is to expose it
through the engine Protocol.

This is a hard prerequisite. Without raw draws, the feature has to fall back
to point estimates and interval widths, which loses the lower-tail story this
plan exists to tell. See ticket 1 below.

## Recommendation

Use a **Bayesian fractional Kelly-inspired ramp rule**, not raw Kelly.

The Kelly Criterion is useful as an intuition: bet more when the expected edge is
large relative to uncertainty, and bet less when the edge is small or noisy. But
raw Kelly is not appropriate for MMM allocations because:

- The payoff distribution is model-estimated, not known.
- MMM uncertainty is not only posterior sampling uncertainty. It also includes
  model form error, measurement error, creative wearout, competitive response,
  execution risk, and unobserved confounding.
- Marketing allocations are not independent repeated bets with clean odds.
- Channel response outside historical spend ranges is extrapolation, not a
  directly observed edge.
- A plan can look optimal in expectation while still having unacceptable
  downside in the lower tail.

So Wanamaker should borrow the **position-sizing discipline** of Kelly, but
implement it conservatively as a posterior risk calculation.

## Core Idea

Let:

- `x0` be the user-supplied baseline plan. v1 takes the baseline explicitly
  from the user; auto-deriving "current" spend from a recent historical window
  is deferred to v1.1 because the right window length is itself a product
  decision worth its own discussion.
- `x_star` be the user-supplied target plan, typically the preferred entry
  from `wanamaker compare-scenarios`. A future optimizer output could replace
  this in v1.1.
- `f` be the adoption fraction, where `0 <= f <= 1`.

The ramped plan is:

```text
x(f) = x0 + f * (x_star - x0)
```

Instead of asking whether `x_star` is better than `x0`, evaluate a ladder of
partial moves:

```text
f in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}
```

For each `f`, use posterior predictive draws to estimate incremental outcome:

```text
delta_s(f) = outcome_s(x(f)) - outcome_s(x0)
```

where `s` indexes posterior predictive draws.

The output is the largest ramp fraction that passes risk criteria.

## Why Not Just Cap Movement?

A hard cap such as "never move more than 20%" is easy to understand but weak
analytically. It treats all recommendations as equally risky.

Examples:

- A 40% move within historical spend ranges, backed by narrow posterior
  intervals and lift-test calibration, may be safer than a 10% move into an
  unobserved spend range.
- A 15% move away from a spend-invariant channel may be less defensible than a
  30% move between well-identified channels.
- Some large shifts are operationally obvious, such as cutting a channel with
  strong evidence of low return and reallocating to a channel repeatedly tested
  in-market.

The rule should therefore be based on uncertainty, downside, and extrapolation,
not only distance.

## Metrics to Compute

For each candidate ramp fraction `f`, compute:

### Expected Increment

```text
E_delta(f) = mean(delta_s(f))
```

This is the expected gain versus the baseline plan.

### Probability of Improvement

```text
P_positive(f) = P(delta_s(f) > 0)
```

This is easier to explain than expected utility and should appear in user-facing
output.

### Downside Probability

Choose a material loss threshold, either absolute or relative:

```text
loss_threshold = max(min_absolute_loss, baseline_outcome * min_relative_loss)
```

Then compute:

```text
P_material_loss(f) = P(delta_s(f) < -loss_threshold)
```

This prevents a plan with a good mean but a fat left tail from being treated as
safe.

### Lower-Tail Downside

Use a lower quantile and, preferably, conditional value at risk:

```text
q05_delta(f) = 5th percentile of delta_s(f)
CVaR_5(f) = mean(delta_s(f) where delta_s(f) <= q05_delta(f))
```

CVaR is useful because it answers: "If the bad tail happens, how bad is it on
average?"

### Historical Support

For each channel and period, compare planned spend to the fitted historical
range:

```text
observed_spend_min[channel] <= spend[channel, period] <= observed_spend_max[channel]
```

Summarize:

- number of extrapolated channel-period cells
- maximum percent above historical max
- maximum percent below historical min
- share of moved budget involving extrapolated cells

The ramp should become more conservative as historical support weakens.

### Identification Quality

Use the Trust Card and channel flags:

- weak convergence
- weak holdout accuracy
- weak prior sensitivity
- weak saturation identifiability
- lift-test inconsistency
- spend-invariant channels
- refresh instability

The ramp rule should not allow a large move when the recommendation depends on a
weak dimension.

### Concentration of the Recommendation (v1.1 gate)

Large recommendations driven by one uncertain channel are riskier than broad
recommendations supported across several channels.

Compute (and surface in the candidate report):

```text
moved_budget_by_channel = abs(x(f) - x0) by channel
largest_move_share = max(moved_budget_by_channel) / sum(moved_budget_by_channel)
```

This is **computed and reported in v1**, but **not gated**. The threshold
above which a single channel is "too concentrated" interacts with Trust Card
weakness on that specific channel in non-obvious ways, and we'd rather ship
the field as an observable signal first and learn the right cutoff from
benchmark cases than guess at it for the v1 gate set.

## Kelly-Inspired Sizing

Raw Kelly for a simple bet can be summarized as:

```text
fraction ~= edge / variance
```

For Wanamaker, a safer approximation is:

```text
kelly_fraction_raw = E_delta(1.0) / variance(delta_s(1.0))
```

This is dimensionally incomplete unless outcome units are normalized. A more
usable version normalizes by baseline outcome or available marketing budget:

```text
r_s = delta_s(1.0) / baseline_outcome_s
kelly_fraction_raw = mean(r_s) / variance(r_s)
```

Clamp to `[0, 1]` before applying the multiplier — without the clamp,
``kelly_fraction_raw`` is unbounded above (a tight posterior with a small
positive mean can easily push the raw value into the tens), and the cap
gate then does no work. The clamp keeps the Kelly term in the same scale
as the candidate ramp fractions:

```text
kelly_fraction = min(max(kelly_fraction_raw, 0.0), 1.0)
fractional_kelly = kelly_fraction * model_confidence_multiplier
```

where:

```text
model_confidence_multiplier in {0.0, 0.1, 0.25, 0.5}
```

Suggested defaults:

| Condition | Multiplier |
|---|---:|
| Any weak Trust Card dimension materially related to the move | 0.0 to 0.1 |
| Moderate Trust Card, some extrapolation, no lift-test support | 0.25 |
| Mostly pass, in-range move, stable refresh history | 0.5 |
| Strong repeated experimental support | 0.5 max for v1.1 |

Do not expose the term "Kelly" prominently to non-expert users. It invites false
precision and gambling analogies. Internally this can be described as
Kelly-inspired, but product language should say "risk-adjusted ramp."

## Recommended Decision Rule

Use the Kelly-inspired score as one input, but make the final recommendation a
discrete ramp choice.

Candidate ramp fractions:

```text
0%, 10%, 25%, 50%, 75%, 100%
```

For each candidate, compute the metrics above. Recommend the largest `f` that
passes all gates.

Suggested initial gates:

| Gate | Pass Threshold |
|---|---:|
| Probability of improvement | `P_positive(f) >= 0.75` |
| Probability of material loss | `P_material_loss(f) <= 0.10` |
| Lower tail | `CVaR_5(f)` no worse than user tolerance |
| Extrapolation | no severe extrapolation, or ramp reduced until extrapolation is mild |
| Trust Card | no weak dimension directly driving the move |
| Spend-invariant channels | no reallocation recommendation involving those channels |
| Fractional Kelly | `f <= fractional_kelly_cap` |

The thresholds should be configurable later, but not in the first version. The
first version should use conservative defaults and report them plainly.

## Output Categories

The feature should produce one of four plain-English outcomes:

### Proceed

Conditions:

- High probability of improvement
- Limited downside
- In historical range
- Trust Card mostly pass

Example:

```text
The model supports moving 50% toward Plan B. The expected gain is $420k over
the plan period, with a 79% probability of outperforming the current plan.
The full move has more downside risk, so we recommend staging the change.
```

### Stage

Conditions:

- Positive expected value
- Some uncertainty or mild extrapolation
- Partial move passes gates, full move does not

Example:

```text
Move 25% toward Plan B, then refresh after 4-8 weeks. The full shift is outside
the model's observed spend range for paid social, but a 25% move stays close to
observed history while preserving most of the expected upside.
```

### Test First

Conditions:

- Recommendation depends on uncertain channel response
- High upside but weak identification
- Experiment would materially change the decision

Example:

```text
Do not reallocate the full budget yet. The opportunity is large, but the result
depends on a channel with weak saturation identifiability. Run a lift test or a
smaller geo holdout before making this move.
```

### Do Not Recommend

Conditions:

- Downside risk too high
- Improvement probability too low
- Extrapolation severe
- Trust Card weak in a material way

Example:

```text
Wanamaker does not recommend this reallocation. The expected outcome is higher,
but the lower-tail risk is too large and most of the move is outside observed
spend ranges.
```

## Reading the Decision Ladder

The Markdown report leads with a six-column ladder, one row per ramp
fraction:

| Ramp | Expected lift | Downside risk | Historical support | Trust Card gate | Verdict |
|---:|---:|---:|---|---|---|
| 10% | +$54,000 | 4% | in range | pass | pass (selected) |
| 25% | +$135,000 | 9% | in range | pass | pass |
| 50% | +$255,000 | 17% | mild | pass | fail |
| 75% | +$340,000 | 24% | severe | pass | fail |
| 100% | +$405,000 | 31% | severe | pass | fail |

How to read each column:

- **Ramp** — the partial move evaluated, in 10/25/50/75/100% of the
  baseline → target delta.
- **Expected lift** — posterior mean increment vs. the baseline plan.
  Signed integer, rounded to whole units of the target metric.
- **Downside risk** — posterior probability of a "material loss" relative
  to baseline (the threshold defaults to 5% of baseline outcome). A
  one-number summary; the analyst-detail table at the bottom of the
  report exposes the underlying CVaR and q05.
- **Historical support** — three buckets: ``in range`` (every channel
  stays inside its training-window spend range), ``mild`` (some flagged
  cells but the extrapolation gate did not trigger), ``severe`` (the
  extrapolation gate failed at this fraction).
- **Trust Card gate** — ``pass`` unless a weak Trust Card dimension
  caps this fraction below the requested move.
- **Verdict** — ``pass`` or ``fail``. The recommended fraction (the
  largest passing one) is marked ``(selected)``.

Below the ladder the report lists, for each fraction that failed,
plain-English bullets explaining *which* gates failed and *why*. That
part is the part to forward when a stakeholder asks "why didn't we go
to 50%?"

The bottom of the report includes a "Sizing detail (for analysts)"
table that preserves the Kelly-style sizing math, q05, CVaR, and the
raw failed-gate names for analysts who want to audit the verdict.

## Integration With Existing Wanamaker Concepts

### Scenario Comparison

This should sit on top of scenario comparison. Users can provide a current plan
and one or more candidate plans. Wanamaker ranks the plans, then evaluates how
far to move toward the preferred one.

This is safer than constrained inverse optimization because the user still
controls the candidate direction.

### Trust Card

The Trust Card should affect the ramp multiplier and language:

- `pass`: allow normal ramp evaluation
- `moderate`: cap ramp at 25% or 50%, depending on downside
- `weak`: cap ramp at 0% or 10% unless lift-test evidence strongly supports the
  move

Weak dimensions should not be hidden in a footnote. They should change the
recommendation.

### Experiment Advisor

If the recommended ramp is low because uncertainty is high, that should feed the
Experiment Advisor:

```text
This reallocation is not blocked because expected value is low. It is blocked
because uncertainty is high. A test on paid social would most improve the
decision.
```

This gives the user a productive next action instead of simply saying "too
risky."

### Refresh Accountability

After a staged move, Wanamaker should encourage a refresh window:

```text
Adopt 25% of the move, run for 4-8 weeks, then refresh. Treat a stable or
improved Trust Card as permission to consider the next ramp step.
```

This connects future decision-making back to the refresh stability thesis.

## Proposed Data Structures

These are deliberately sketch-level and should be refined during ticketing.

```python
@dataclass(frozen=True)
class RampCandidate:
    fraction: float
    plan: pd.DataFrame
    expected_increment: float
    probability_positive: float
    probability_material_loss: float
    q05_increment: float
    cvar_5: float
    extrapolation_flags: list[ExtrapolationFlag]
    largest_move_share: float
    passes: bool
    failed_gates: list[str]


@dataclass(frozen=True)
class RampRecommendation:
    baseline_plan_name: str
    target_plan_name: str
    recommended_fraction: float
    status: Literal["proceed", "stage", "test_first", "do_not_recommend"]
    candidates: list[RampCandidate]
    explanation: str
```

The output should be serializable to JSON and renderable in reports.

## Proposed API

```python
def recommend_ramp(
    posterior_summary: PosteriorSummary,
    baseline_plan: str | Path | pd.DataFrame,
    target_plan: str | Path | pd.DataFrame,
    seed: int,
    engine: PosteriorPredictiveEngine,
    trust_card: TrustCard | None = None,
    risk_tolerance: RiskTolerance | None = None,
) -> RampRecommendation:
    ...
```

`risk_tolerance` can be deferred. The first version can use a fixed conservative
profile.

## Ticket Breakdown for v1.0

Four tickets, ordered by dependency. Each is meaningfully scoped (about half
a day to a day and a half of focused work) and self-contained enough that a
later ticket's tests will also exercise the earlier tickets' code paths.

### Ticket 1 · Expose raw posterior predictive draws from the engine (foundational)

Extend the `PosteriorPredictiveEngine` Protocol so callers can ask for the
per-draw `(n_draws, n_periods)` outcome matrix in addition to the existing
`PredictiveSummary`. Update `PyMCEngine.posterior_predictive` to return the
matrix (the existing implementation already computes it internally before
collapsing). Existing `forecast()` and `compare_scenarios()` callers keep
their current behaviour.

Acceptance:
- A new `PosteriorPredictiveDraws` shape (or equivalent) is part of the
  engine contract.
- An engine-marked test fits a real PyMC model and asserts the draws have
  the expected `(n_draws, n_periods)` shape.
- The lazy-import contract still holds (no PyMC pulled in at module import).

### Ticket 2 · `recommend_ramp()` core: math, gates, decision rule

In a new `wanamaker.forecast.ramp` module, implement:

- `RampCandidate` and `RampRecommendation` dataclasses, JSON-serialisable
  through the existing artifact envelope pattern.
- Plan interpolation `x(f) = x0 + f * (x_star - x0)` over the ladder
  `{0, 0.10, 0.25, 0.50, 0.75, 1.0}`, with channel/period alignment
  validation.
- Per-candidate posterior risk metrics: `expected_increment`,
  `probability_positive`, `probability_material_loss`, `q05_increment`,
  `cvar_5`, `largest_move_share`, `extrapolation_flags`,
  `fractional_kelly` (clamped per the note above).
- The decision rule: pick the largest `f` that passes all v1 gates;
  classify the result into `proceed`, `stage`, `test_first`, or
  `do_not_recommend`.
- Trust Card gating: weak/moderate dimensions cap ramp; spend-invariant
  channels block any ramp that would reallocate them.

No CLI yet. Tests cover each metric on synthetic draws, both passing and
failing cases for each gate, and the four output statuses.

### Ticket 3 · `wanamaker recommend-ramp` CLI command

Wire up the command:

```text
wanamaker recommend-ramp --run-id <id> --baseline <plan> --target <plan>
```

Reuses the engine adapter from `wanamaker forecast` so the run's saved
posterior is rebuilt and bound automatically. Saves a deterministic Jinja2
Markdown report to `<run_dir>/ramp_<baseline>_to_<target>.md` with the
selected fraction, the four-status verdict, the per-candidate risk-metric
table, and an Experiment Advisor handoff sentence when uncertainty is the
binding constraint.

Tests stub the engine (mirroring `test_cli_forecast.py`) and assert the
command runs end-to-end, the report file is written, and the status text
matches the expected category.

### Ticket 4 · Documentation

Surface the feature in user-facing docs without rewriting them:

- `docs/guides/analyst_guide.md`: a "Risk-adjusted ramps" section explaining the
  four output categories and when to read them as decision guidance vs
  evidence prompts.
- `docs/guides/cmo_guide.md`: a paragraph on ramp recommendations as the
  conservative alternative to "the model says move 100% to Plan B."
- `docs/quickstart.md`: an optional Step 7 showing the ramp command on the
  bundled `public_benchmark` dataset.
- Cross-link to this design note for engineers and reviewers.

The Phase 2 templates (`executive_summary.md.j2`, `trust_card.md.j2`)
already point to the Trust Card and forecast outputs; deeper integration
(executive summary embedding the ramp verdict directly) is v1.1 work.

## Open Questions

- Should the material loss threshold default to a revenue amount, a percentage
  of forecast revenue, or a percentage of marketing budget?
- Should downside be measured in revenue, contribution, profit, or gross margin?
  Profit is preferable, but not every user will have margin data.
- Should the ramp horizon be one planning period, a quarter, or whatever horizon
  appears in the supplied plan?
- How should interactions with fixed contractual commitments be represented?
- Should the first version allow only target plans supplied by the user, or also
  consume a future optimizer output?

## Bottom Line

Wanamaker should not present large allocation changes as direct instructions.
It should present them as evidence-weighted ramps.

The model can say:

```text
Plan B looks better than Plan A.
```

The risk-adjusted ramp layer says:

```text
Move 25% toward Plan B now, because that is the largest step supported by the
posterior, downside risk, historical spend range, and Trust Card.
```

That distinction is central to the product's credibility.
