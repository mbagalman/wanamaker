# What Wanamaker Tells Your CMO

This guide explains Wanamaker outputs for marketing executives and budget
decision-makers.

The short version: Wanamaker is meant to support budget decisions, not settle
arguments with false precision. It tells you what the model estimates, how
uncertain that estimate is, which parts deserve caution, and what evidence would
make the next decision easier.

## How To Read Wanamaker Outputs

Start with four questions:

1. What decision are we trying to make?
2. Which outputs support that decision?
3. What does the Trust Card say about those outputs?
4. Is the recommended action inside or outside our historical experience?

If the answer to the fourth question is "outside," treat the result as a test
candidate, not as permission to make a large immediate move.

## Readiness Diagnostic

The readiness diagnostic runs before modeling. It answers a basic question:

> Is this dataset suitable for the decision we want the model to support?

The diagnostic checks the practical failure modes that often undermine MMM:
too little history, missing values, channels that never changed spend, channels
that moved together, too many predictors, irregular dates, structural breaks,
and possible target leakage.

The readiness levels are plain:

| Level | Business meaning |
|---|---|
| `ready` | The data passed the checks Wanamaker can run before fitting. |
| `usable_with_warnings` | You can fit the model, but the warnings should shape the conversation. |
| `diagnostic_only` | The dataset can teach you what is wrong, but should not drive budget decisions. |
| `not_recommended` | Fix the data problem before using the model. |

Do not treat warnings as housekeeping. A warning about flat spend or collinear
channels may be the reason the model cannot separate two budget options.

## Executive Summary

The executive summary is the first page for leadership. It answers:

- how much paid media appears to have contributed
- which channels appear largest
- how uncertain those estimates are
- which Trust Card weaknesses affect interpretation
- what actions the model supports now

The summary is generated from deterministic templates. It is not written by an
LLM and it does not improvise a story.

How to act on it:

- If the summary names a clear finding and the Trust Card is clean, use it as
  input to planning.
- If the summary uses cautious language, read the reason before debating the
  recommendation.
- If the summary says a channel is weakly identified, do not make that channel
  the center of a major reallocation.

The executive summary should not be the only thing reviewed. It is the front
door to the evidence.

## Channel Contributions

Channel contribution estimates answer:

> How much of the target metric does the model attribute to each marketing
> channel during the historical period?

This is useful for understanding what appears to have mattered. It is not the
same as saying the channel would produce the same result next quarter at any
spend level.

Read contribution estimates with their intervals. A channel with a large point
estimate and a wide interval may be important, but uncertain. That is different
from a channel with a large estimate and a tight interval.

How to act on it:

- Use stable, material contribution estimates to inform channel importance.
- Use uncertain contribution estimates to identify where better evidence is
  needed.
- Do not treat contribution shares as a complete budget allocation plan.

## ROI Summary

ROI estimates answer:

> For each dollar spent, what return does the model estimate for the channel?

ROI is often the most tempting table in an MMM report. It is also one of the
easiest to overread.

The mean ROI is only the middle of the estimate. The credible interval tells you
how much faith to put in it. A channel with the highest mean ROI is not
automatically the best place to move budget if its interval is wide, its spend
was flat, or the proposed plan is far outside history.

How to act on it:

- Prefer channels with strong ROI and reasonable uncertainty.
- Treat high-ROI, high-uncertainty channels as experiment candidates.
- Do not cut a channel only because its point estimate is lower than another
  channel's point estimate.
- Do not use ROI for spend-invariant channels as if Wanamaker learned a response
  curve from data.

## Model Trust Card

The Trust Card is Wanamaker's credibility audit. It does not say "trust the
model" or "do not trust the model" as a single score. It names the specific
places where the model does or does not deserve confidence.

The v1 dimensions are:

| Dimension | Plain-English question |
|---|---|
| `convergence` | Did the Bayesian sampler settle enough to trust the estimates? |
| `holdout_accuracy` | Did the model predict held-out periods reasonably well? |
| `prior_sensitivity` | Did results change too much when prior assumptions were perturbed? |
| `saturation_identifiability` | Did spend vary enough to learn diminishing returns? |
| `refresh_stability` | When new data arrived, did old estimates move for explainable reasons? |
| `lift_test_consistency` | Do model estimates agree with any lift-test evidence supplied? |

Status values are:

| Status | How to use it |
|---|---|
| `pass` | No major concern was found for that dimension. |
| `moderate` | Use the output, but keep the caveat in the decision. |
| `weak` | Do not rely on the affected output without more evidence. |

A weak Trust Card dimension should change behavior. It may mean using the model
for directional planning, staging the budget move, or running an experiment
before committing to a larger shift.

## Forecast

Forecasting answers:

> If we run this future spend plan, what outcome range does the model expect?

The forecast report shows period-by-period expected outcomes and credible
intervals. It also flags plan cells that fall outside the historical spend range.

How to act on it:

- Use the mean as the central expectation, not as a promise.
- Use the interval as the range of model uncertainty.
- Treat extrapolation warnings seriously. They mean the model is being asked to
  predict a spend level it has not seen.
- Compare forecasts against operational reality: inventory, sales capacity,
  seasonality, promotions, and finance constraints still matter.

## Scenario Comparison

Scenario comparison answers:

> Of the budget plans we supplied, which one looks better, and how uncertain is
> that ranking?

This is different from automatic optimization. Wanamaker does not invent the
plan. You provide reasonable options. Wanamaker ranks them and shows uncertainty.

How to act on it:

- Use scenario comparison for real planning choices, not abstract channel
  debates.
- Prefer options that are both strong and inside the range of historical
  evidence.
- If the top-ranked scenario is far from anything the business has done before,
  stage the move or test it.
- If two scenarios have overlapping intervals, the model may not be able to
  distinguish them well enough to justify a hard recommendation.

This conservative framing is intentional. A model should not make a large budget
change look safer than it is.

## Risk-Adjusted Ramps

Once a preferred plan is identified, the next question is rarely "do we
adopt all of it." It is usually:

> How much of the way toward the new plan is supported by the evidence we
> have today, and what does the model want us to do with the rest?

Wanamaker answers this with a discrete ramp recommendation. You supply
the current plan and the candidate plan; the tool evaluates partial moves
(10%, 25%, 50%, 75%, 100%) against probability of improvement, downside
probability of material loss, lower-tail risk, extrapolation severity,
and the Trust Card. The output is one of four plain-English categories:

| Verdict | What it means for the decision |
|---|---|
| **Proceed** | The model supports moving the recommended fraction. The chosen ramp is the largest the evidence justifies right now; higher fractions failed because the model genuinely doesn't believe in them or because we are at 100%. |
| **Stage** | Adopt the recommended fraction, refresh after new data, and revisit. Higher fractions failed on evidence-quality grounds (extrapolation beyond historical spend, or a Trust Card weakness). The cap may lift on the next run. |
| **Test first** | The model cannot defend any positive ramp on the current evidence, but the failures point at uncertainty rather than expected value. Run a controlled test or wait for a refresh; don't commit budget yet. |
| **Do not recommend** | The model does not support this reallocation in any size. The expected value or downside risk is the binding constraint, and more time alone won't fix it. |

How to use this in practice:

- A `proceed` or `stage` verdict gives the executive a number to commit
  to *now* and a clear story for the rest of the move.
- A `test first` verdict is a productive answer, not a refusal — it tells
  you which channel to test and how the result would change the
  recommendation.
- A `do not recommend` verdict is the one place where the tool says
  "no" outright, and it should be read alongside the underlying
  scenario-comparison output: if a candidate plan looks better in
  expectation but the ramp output blocks it, the downside profile is
  the reason.

The full design rationale and the math live in
[`docs/internal/risk_adjusted_allocation.md`](../internal/risk_adjusted_allocation.md). For
v1 the ramp will rarely recommend a full 100% move even on a clean run —
that is intentional. The product's position is "the model can advise on
direction; the user owns the decision to commit the full budget."

## Refresh Accountability

Refresh accountability answers:

> When we add new data, did the model rewrite history, and if so, why?

Every MMM changes when new data arrives. That is not a failure. The problem is
when last quarter's estimates change silently and no one can explain whether the
change matters.

Wanamaker's refresh diff classifies movement in historical estimates:

| Movement class | Business meaning |
|---|---|
| Within prior credible interval | Normal movement. Usually not a concern. |
| Improved holdout accuracy | The estimate moved, but the new model appears to predict better. |
| User-induced | The change came from a config, prior, or input change. |
| Weakly identified | The channel was already hard to estimate, so instability is expected. |
| Unexplained | Investigate before acting on the changed estimate. |

How to act on it:

- Do not panic when estimates move. Read the movement class.
- Treat unexplained movement as a governance issue.
- Ask whether the movement changes the decision, not just whether a number
  changed.
- Keep refresh diffs with planning materials so the history of the decision is
  visible.

## Experiment Advisor

The Experiment Advisor answers:

> Which channels would benefit most from a controlled test?

In v1, the advisor is deliberately narrow. It flags channels where experimental
evidence would most improve decision confidence. The common reasons are:

- spend was effectively constant, so saturation cannot be learned from history
- the channel appears important but the posterior interval is too wide

How to act on it:

- Treat advisor recommendations as evidence priorities, not campaign
  instructions.
- Use them to decide where a lift test or planned spend variation would reduce
  risk.
- Prioritize tests where the budget decision is material enough to justify the
  effort.

The advisor does not yet design the experiment. It tells you where better
evidence is likely to matter.

## Saved Artifacts

Each fit writes local artifacts under `.wanamaker/`. These files are not for
daily executive review, but they matter for governance.

Useful artifacts include:

| Artifact | Business purpose |
|---|---|
| `manifest.json` | Shows the run ID, seed, engine, and whether validation was skipped. |
| `config.yaml` | Preserves the exact settings used for the run. |
| `data_hash.txt` | Proves which input data was used without exposing the whole dataset. |
| `summary.json` | Stores the engine-neutral posterior summary used by reports. |
| `trust_card.json` | Stores the credibility audit. |
| `refresh_diff.json` | Stores the movement explanation for a refresh run. |
| `report.md` | Markdown executive summary + Trust Card — analyst-facing review. |
| `showcase.html` | Single-file HTML with charts, response curves, and the Trust Card — the "show your CMO" artifact. |
| `trust_card.html` | One-page plain-English Trust Card — forward when "do I trust this MMM?" comes up. |
| `summary.xlsx` | Structured tables (channels, ROI, parameters, scenarios) for analysts who want to pivot. |

These artifacts make it possible to answer "what did we know when we made this
decision?"

## What Wanamaker Cannot Tell You

Wanamaker cannot tell you everything a marketing leader wants to know.

It cannot prove individual customer journeys. It uses aggregate data, not
person-level tracking.

It cannot estimate a reliable response curve for a channel whose spend barely
changed. In that case, the curve comes mostly from assumptions and prior
knowledge.

It cannot guarantee the future. Forecasts are conditional on the supplied plan,
the historical data, and the model assumptions.

It cannot fully separate channels that always moved together. If paid social and
online video rose and fell in the same weeks, the model may know the pair
mattered without knowing the split precisely.

It cannot make a risky plan safe. If the best-looking plan is much more
aggressive than anything the business has tried, Wanamaker should make that risk
visible, not hide it.

It cannot replace business judgment. Stockouts, margin pressure, creative
quality, competitive action, channel capacity, and brand strategy still belong
in the decision.

## Executive Review Checklist

Before approving a budget change based on Wanamaker, ask:

- What decision are we making?
- Which scenario or forecast supports it?
- What does the ramp recommendation say — proceed, stage, test first, or
  do not recommend? Which gates were binding?
- What does the Trust Card mark as weak or moderate?
- Are we staying inside historical spend ranges?
- If not, how are we staging or testing the move?
- Did the latest refresh change any historical conclusions?
- Which recommendation would change if an uncertain channel estimate moved
  within its credible interval?

If the team cannot answer those questions, the next step is not a larger budget
move. The next step is more analysis, a smaller staged change, or a controlled
test.
