# Analyst's Guide

This guide is for marketing analysts who know the business, can work with CSVs
and Python tools, and want a useful MMM without becoming Bayesian specialists.

Wanamaker is built around one idea: the model should help you make better
marketing decisions while being honest about what the data can and cannot tell
you.

## What MMM Is

Marketing mix modeling estimates how aggregate marketing activity relates to
aggregate business outcomes over time.

A typical Wanamaker dataset has one row per week and columns like:

- revenue or another target metric
- spend by marketing channel
- controls such as promotions, holidays, price, or macro indicators

The model learns a relationship between those inputs and the target. The useful
outputs are not just fitted lines. The useful outputs are questions like:

- Which channels appear to contribute the most?
- Which channels have uncertain estimates?
- What happens if I compare one future spend plan to another?
- Did the model's historical estimates change after a refresh?
- Is the model trustworthy enough for the decision in front of me?

MMM works best when marketing activity changes over time. If a channel spends
the same amount every week, the model has little evidence about how response
changes with spend. Wanamaker will still fit where possible, but it should not
pretend that flat spend can reveal a full response curve.

## What MMM Is Not

MMM is not a receipt-level attribution system. It does not follow individual
people across channels. It uses aggregate data.

MMM is also not proof that one channel caused every sale assigned to it. It is
a statistical estimate based on historical variation, controls, and assumptions.
Good MMM narrows uncertainty. It does not eliminate it.

Wanamaker intentionally avoids language like "the model knows" or "the model
proved." The right question is usually:

> Is this estimate strong enough to support this decision?

## The Basic Workflow

The core workflow is:

1. Prepare a weekly CSV and YAML config.
2. Run `wanamaker diagnose`.
3. Fix blockers or understand warnings.
4. Run `wanamaker fit`.
5. Run `wanamaker report`.
6. Use `forecast` or `compare-scenarios` for future plans.
7. When new data arrives, run `refresh` and read the diff.

The readiness diagnostic is not paperwork. It is part of the modeling workflow.
Most bad MMM decisions start before sampling begins, with data that cannot
support the question being asked.

## Adstock

Advertising often has carryover. A TV campaign, video campaign, or podcast buy
can affect customers after the week when the money was spent. Adstock is the
model's way of representing that carryover.

For example, if a channel has short carryover, this week's spend mostly affects
this week's outcome. If it has longer carryover, some effect remains in later
weeks.

Wanamaker supports:

- **Geometric adstock**, the default. It is simple and stable.
- **Weibull adstock**, available as a per-channel override when the channel's
  carryover shape needs more flexibility.

Do not interpret adstock half-life too literally. It is useful as a summary of
carryover, but it is estimated from noisy aggregate data. Treat it as a model
parameter with uncertainty, not as a physical law.

## Saturation

Saturation means diminishing returns. The first dollars in a channel may be
more productive than later dollars. At some point, more spend may still help,
but each additional dollar adds less.

Wanamaker uses Hill saturation curves to model this behavior. The report should
distinguish the part of a response curve supported by observed spend from the
part that extrapolates beyond history.

This distinction matters. A model can draw a curve beyond the spend levels it
has seen, but that does not mean the business has tested that spend range.

When a channel has nearly constant spend, Wanamaker flags saturation
identifiability risk. In plain English:

> The model may estimate that the channel contributes to revenue, but it cannot
> learn much about what would happen at meaningfully different spend levels.

## Credible Intervals

Wanamaker reports credible intervals for estimates such as ROI, contribution,
and forecasts.

A 95% credible interval means that, given the model and data, most posterior
mass lies in that range. It is not a guarantee that next quarter's outcome will
fall there. It is also not a promise that the model includes every possible
source of business risk.

Use intervals to calibrate confidence:

- A narrow interval entirely above zero is stronger evidence.
- A wide interval means the estimate could change materially.
- A channel with high expected ROI and very wide uncertainty may be a good
  experiment candidate, not an automatic budget winner.

The mean is only one part of the decision. The interval tells you how much faith
to put in that mean.

## The Trust Card

The Trust Card is Wanamaker's summary of model credibility. It does not say
"good model" or "bad model" as a single score. It names the dimensions that
matter.

The v1 Trust Card dimensions are:

| Dimension | What It Checks | Why It Matters |
|---|---|---|
| Convergence | Whether sampling diagnostics look stable | Bad sampling can make estimates unreliable |
| Holdout accuracy | Whether predictions matched held-out data | A model that cannot predict withheld data deserves caution |
| Refresh stability | Whether historical estimates changed for explainable reasons | Refreshes should not silently rewrite history |
| Prior sensitivity | Whether results move too much when priors are perturbed | Prior-driven results need careful language |
| Saturation identifiability | Whether channel spend variation can support response curves | Flat spend cannot identify saturation |
| Lift-test consistency | Whether posterior ROI agrees with calibration evidence | Experiments should inform estimates |

Status values are:

- `pass`: no major concern found
- `moderate`: use the result, but read the explanation
- `weak`: do not rely on the affected output without additional evidence

Weak status should change how you act. For example, a weak saturation
identifiability flag should make you cautious about reallocating budget into or
out of that channel based on curve shape.

## Reading the Executive Summary

The executive summary is deterministic text generated from model statistics.
There are no LLM calls in product output.

Read the summary in this order:

1. What are the strongest findings?
2. Which findings are explicitly hedged?
3. Which Trust Card dimensions explain the hedges?
4. What decision is supported now?
5. What decision needs more evidence?

If the summary says a channel is "likely" contributing, that is different from
saying the channel's exact ROI is known. If the summary says Wanamaker cannot
estimate saturation for a channel, do not use that channel's curve as a budget
allocation tool.

## Forecasts and Scenario Comparison

Forecasting asks:

> What does the model expect if I use this future spend plan?

Scenario comparison asks:

> Among the plans I supplied, which looks better, and how uncertain is that
> ranking?

Wanamaker's v1 scenario comparison is user-driven. You provide plans. The tool
ranks them with uncertainty and flags extrapolation beyond historical spend
ranges.

This is deliberately more conservative than automatic budget optimization.
The user remains in control of the strategic options being evaluated.

## Refresh Workflow

Refresh is one of Wanamaker's headline features. When new data arrives, you can
re-run the model and compare the new estimates to the prior run.

A refresh diff should answer:

- Which historical estimates changed?
- How much did they change?
- Are the movements explainable?
- Did the change affect the decision?

Movement classes include:

| Class | Meaning |
|---|---|
| Within prior credible interval | Expected movement; usually not a concern |
| Improved holdout accuracy | Larger movement, but new fit appears better |
| Unexplained | Trust risk; investigate before acting |
| User-induced | Driven by config or prior changes |
| Weakly identified | Expected instability in a weak channel |

The point is not to freeze history. New data should be allowed to update the
model. The point is to prevent historical estimates from silently changing with
no explanation.

## Common Pitfalls

### Too Little History

MMM needs enough variation over enough time. Fewer than 52 weekly observations
is usually fragile, and fewer than 26 is usually not recommended.

### Flat Spend

If a channel spends almost the same amount every week, the model cannot learn
how response changes at different spend levels.

### Collinear Channels

If two channels always move together, the model may struggle to separate their
effects. Wanamaker can still fit, but individual channel estimates may be less
stable.

### Target Leakage

Do not include controls that are mechanically derived from the target. A
conversion rate, revenue-derived metric, or post-outcome operational measure can
make the model look better while making the marketing estimates worse.

### Overreading ROI Rankings

A channel with the highest mean ROI is not always the best place to move budget.
Look at the interval, saturation identifiability, historical spend range, and
operational reality.

### Treating Extrapolation as Evidence

If a future plan spends far above anything in history, the forecast is partly an
extrapolation. Wanamaker should flag this clearly. Treat it as a reason to stage
or test, not as proof.

## Practical Decision Rules

Use these defaults unless there is a strong reason not to:

- Fix blockers before fitting.
- Treat warnings as report language, not as noise.
- Do not make large reallocations involving spend-invariant channels.
- Prefer scenario comparison over automatic optimization.
- If a recommended plan is far from historical behavior, stage the move and
  refresh after new data arrives.
- Use experiments when a high-value decision depends on a wide interval.

## What To Bring To A Review Meeting

For a stakeholder meeting, bring:

- the executive summary
- the Trust Card
- top channel contributions and ROI intervals
- any scenario comparison output
- any refresh diff if this is not the first run
- a short list of decisions the model supports now
- a short list of decisions that need more evidence

The best Wanamaker meeting should not end with "the model said so." It should
end with a clear decision, a clear uncertainty statement, and a clear next
measurement step.
