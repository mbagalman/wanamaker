# Example: B2B refresh after a noisy quarter

A B2B SaaS company runs MMM quarterly. After Q4 — with a heavy holiday
promotion calendar and a one-off PR spike — the analyst refreshes the
model and finds that the historical paid-search ROI estimate has dropped
about 15%. The CMO wants to know whether to revise last quarter's
budget plan based on the new estimate.

This is the "refresh nightmare" Robyn users complain about: re-running
on new data quietly rewrites history, and nobody can tell whether the
change is a real signal or a sampling artifact. Wanamaker's refresh
workflow is built specifically to answer that question.

This walk-through uses the bundled `public_benchmark` dataset, split
into an "initial" window and a "refresh" window. About 8–12 minutes on
a laptop in `quick` mode.

## Setup

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
pip install -e ".[dev]"
```

Make a working copy of the example so you can swap the CSV under it
without modifying the bundled file:

```bash
mkdir -p tutorial_data
cp benchmark_data/public_example.csv tutorial_data/data.csv
cat > tutorial_data/config.yaml <<'EOF'
data:
  csv_path: tutorial_data/data.csv
  date_column: week
  target_column: revenue
  spend_columns:
    - paid_search
    - paid_social
    - linear_tv
    - ctv
    - online_video
    - display
    - affiliate
    - email
  control_columns:
    - price_index
    - promo_flag
    - holiday_flag
    - macro_index

channels:
  - name: paid_search
    category: paid_search
  - name: paid_social
    category: paid_social
  - name: linear_tv
    category: linear_tv
  - name: ctv
    category: ctv
  - name: online_video
    category: video
  - name: display
    category: display_programmatic
  - name: affiliate
    category: affiliate
  - name: email
    category: email_crm

refresh:
  anchor_strength: medium

run:
  seed: 20260430
  runtime_mode: quick
  artifact_dir: .wanamaker
EOF
```

## Step 1 — Fit on the initial window

Pretend you are at the end of Q3 with 130 weeks of history:

```bash
head -n 131 benchmark_data/public_example.csv > tutorial_data/data.csv
wc -l tutorial_data/data.csv  # 131 = 1 header + 130 weeks
wanamaker fit --config tutorial_data/config.yaml
RUN_INITIAL=$(ls -t .wanamaker/runs | head -1)
wanamaker showcase --run-id "$RUN_INITIAL"
```

Open
`.wanamaker/runs/$RUN_INITIAL/showcase.html` and skim the channel ROI
table. This is the picture the CMO has from the previous refresh.

## Step 2 — Q4 happens; refresh on the full window

Now extend the data with the remaining 26 weeks (Q4 + early Q1):

```bash
cp benchmark_data/public_example.csv tutorial_data/data.csv
wc -l tutorial_data/data.csv  # 157 = 1 header + 156 weeks
wanamaker refresh --config tutorial_data/config.yaml
RUN_REFRESH=$(ls -t .wanamaker/runs | head -1)
```

This is the key command. It does three things at once:

1. **Fits a new model** on the full 156-week window.
2. **Anchors lightly to the previous posterior** (default
   `anchor_strength: medium`) so estimates cannot drift unboundedly on
   one quarter of new data.
3. **Writes a refresh diff** comparing the old and new posteriors,
   classifying every parameter movement.

The new run ID is now stored in `RUN_REFRESH`.

## Step 3 — Read the refresh diff

```bash
ls -la .wanamaker/runs/$RUN_REFRESH/
```

You should see `refresh_diff.json` alongside the usual artifacts. The
plain-English summary lives in `report.md`:

```bash
wanamaker report --run-id $RUN_REFRESH
less .wanamaker/runs/$RUN_REFRESH/report.md
```

Look for the **Refresh notes** section. Each parameter movement is
classified as one of:

| Class | What it means |
|---|---|
| `within_prior_ci` | The new estimate is inside the previous credible interval. The "movement" is sampling noise, not a real shift. |
| `improved_holdout` | The new estimate moved AND the new model predicts recent weeks better than the old one. The shift is real and supported. |
| `user_induced` | The user changed something (config, anchor strength, prior). The movement is a deliberate decision, not new data. |
| `weakly_identified` | The parameter has a wide posterior — small differences between runs aren't meaningful. |
| `unexplained` | The estimate moved, holdout did not improve, and nothing on the user side changed. **This is the only class that warrants real investigation.** |

## Step 4 — Decide what to do

The refresh diff turns "the number changed" into a structured triage
question:

- **Mostly `within_prior_ci`?** The model is stable; report this to the
  CMO as "no material change" and continue with the previous plan.
- **`improved_holdout` for the channel that moved?** This is a real
  shift. Update plans accordingly and explain the shift in stakeholder
  communication.
- **`unexplained` movements above ~10% of all movements?** Pause
  recommendations and investigate. Possible causes: a structural break
  in the data (a competitor exit, a media buy that wasn't logged), a
  config change you didn't realize you made, or a sampling artifact
  that needs more chains. Open the previous and current `summary.json`
  and look at convergence statistics for the affected parameters.

The Trust Card has a `refresh_stability` dimension that summarises this
automatically — green if mostly explained, weak if the unexplained
fraction is high.

## What this example demonstrates

- Refresh is **not just re-fit** — it's a structured comparison with a
  defensible classification of every change.
- The **unexplained fraction** is the only number that should make you
  nervous, and it is surfaced explicitly rather than buried in a summary
  statistic.
- **Light posterior anchoring** prevents the "refresh nightmare" failure
  mode where one noisy quarter causes a wholesale re-estimate of
  everything.

## Next reading

- [DTC reallocates linear TV to CTV](tv_to_ctv.md) — using a stable run
  to score a proposed budget change
- [Lift-test calibration changes the verdict](lift_test_changes_verdict.md) —
  when a real-world experiment contradicts the model
- [Risk-Adjusted Allocation Ramps](../internal/risk_adjusted_allocation.md) —
  how the Trust Card and refresh diff feed into ramp gating
