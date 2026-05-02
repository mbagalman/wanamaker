# Example: a DTC brand reallocates linear TV to CTV

A direct-to-consumer apparel brand has been running roughly $30K/week in
linear TV and $15K/week in CTV. The CMO wants to shift $5K/week from
linear TV to CTV to chase younger viewers. The CFO wants to see the
math before approving the move.

This walk-through uses Wanamaker to:

1. Estimate channel ROIs with credible intervals from history.
2. Score the proposed move against the current plan.
3. Decide whether to ramp the move all at once, partially, or test first.

It runs end-to-end on the bundled `public_benchmark` dataset — no
private data needed. About 5–8 minutes on a laptop in `quick` mode.

## What you need

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
pip install -e ".[dev]"
```

## Step 1 — Fit the baseline model

```bash
wanamaker run --example public_benchmark
```

Note the run ID printed at the end (e.g. `1a2b3c4d_20260501T120000Z`).
Save it:

```bash
RUN_ID=$(ls -t .wanamaker/runs | head -1)
echo "$RUN_ID"
```

The Markdown report is at `.wanamaker/runs/$RUN_ID/report.md` and the
HTML showcase at `.wanamaker/runs/$RUN_ID/showcase.html`. Open the
showcase in a browser before continuing — the channel ROI table is the
context for everything below.

The dataset is engineered so `linear_tv` has a higher absolute
contribution but `ctv` has a higher per-dollar ROI. That is exactly the
shape of the question the CMO is asking, which is why this dataset is
useful for the example.

## Step 2 — Write the two plans

`baseline.csv` keeps the current mix; `tv_to_ctv.csv` shifts $5K/week
from linear TV to CTV. Both run for the next eight weeks.

```bash
cat > baseline.csv <<'EOF'
period,paid_search,paid_social,linear_tv,ctv,online_video,display,affiliate,email
2025-01-06,9700,8500,20800,12100,9700,4700,3300,2500
2025-01-13,9700,8500,20800,12100,9700,4700,3300,2500
2025-01-20,9700,8500,20800,12100,9700,4700,3300,2500
2025-01-27,9700,8500,20800,12100,9700,4700,3300,2500
2025-02-03,9700,8500,20800,12100,9700,4700,3300,2500
2025-02-10,9700,8500,20800,12100,9700,4700,3300,2500
2025-02-17,9700,8500,20800,12100,9700,4700,3300,2500
2025-02-24,9700,8500,20800,12100,9700,4700,3300,2500
EOF

cat > tv_to_ctv.csv <<'EOF'
period,paid_search,paid_social,linear_tv,ctv,online_video,display,affiliate,email
2025-01-06,9700,8500,15800,17100,9700,4700,3300,2500
2025-01-13,9700,8500,15800,17100,9700,4700,3300,2500
2025-01-20,9700,8500,15800,17100,9700,4700,3300,2500
2025-01-27,9700,8500,15800,17100,9700,4700,3300,2500
2025-02-03,9700,8500,15800,17100,9700,4700,3300,2500
2025-02-10,9700,8500,15800,17100,9700,4700,3300,2500
2025-02-17,9700,8500,15800,17100,9700,4700,3300,2500
2025-02-24,9700,8500,15800,17100,9700,4700,3300,2500
EOF
```

The amounts roughly match the dataset's historical observed ranges so
the model has direct evidence for both plans. Wanamaker will warn when a
plan exceeds the historical range; the warning still applies even on
synthetic data.

## Step 3 — Compare the two plans

```bash
wanamaker compare-scenarios --run-id "$RUN_ID" --plans baseline.csv tv_to_ctv.csv
```

The output ranks the plans by expected outcome with credible intervals.
Expect `tv_to_ctv.csv` to come out narrowly ahead on the mean — the
dataset has CTV's ROI engineered higher than linear TV's. Whether the
credible intervals overlap is the more important number than the point
ranking.

## Step 4 — Ask how far to move

`compare-scenarios` says which plan looks better. It does not say how
much of the way to move toward it. That is the job of `recommend-ramp`:

```bash
wanamaker recommend-ramp --run-id "$RUN_ID" --baseline baseline.csv --target tv_to_ctv.csv
```

The output is one of four verdicts:

| Verdict | What it means |
|---|---|
| `proceed` | Defensible at 100%. Rare on first runs by design (v1 caps the Kelly multiplier at 0.5). |
| `stage` | Defensible at 25%–50%. The expected outcome at this fraction. |
| `test_first` | The move is plausible, but evidence is too thin — design an experiment first. |
| `do_not_recommend` | Either the math is unfavorable or the Trust Card flags a blocker. |

For this dataset, expect a `stage` verdict — the data supports the
direction of the move but the credible intervals are wide enough that
reallocating 100% on the first decision would be reckless. The full
report at `.wanamaker/runs/$RUN_ID/ramp_baseline_to_tv_to_ctv.md` shows
the per-fraction gate table: which fractions failed and why
(extrapolation, Trust Card weakness, expected value, downside risk).

## Step 5 — Read the verdict alongside the showcase

The ramp output is a defensible answer to "how much to shift now". The
showcase from Step 1 is the answer to "why do we believe it". Open both
side by side before booking the change in your media plan.

## What this example demonstrates

- **`compare-scenarios`** is descriptive: which plan does the model
  expect to do better?
- **`recommend-ramp`** is prescriptive: how much of the move can we
  defend with the evidence we have right now?
- The two answers can disagree. A plan can rank first on expected
  outcome but still be `test_first` because the evidence base is too
  thin to bet much on it.

## Next reading

- [Risk-Adjusted Allocation Ramps](../internal/risk_adjusted_allocation.md) —
  the design rationale for the four-status verdict
- [Refresh after a noisy quarter](refresh_after_noisy_quarter.md) —
  the next workflow you'll need
- [Lift-test calibration changes the verdict](lift_test_changes_verdict.md) —
  what happens when an experiment contradicts the model
