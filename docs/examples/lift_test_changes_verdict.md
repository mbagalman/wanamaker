# Example: a lift test changes the verdict

A retailer ran a 6-week geo-holdout test on paid social and got a clean
result: incremental lift was about 12% lower than the MMM had been
estimating. The CFO wants to know whether to keep planning around the
old MMM estimate or revise the budget.

This example shows how to feed a real-world experiment back into
Wanamaker as a calibration prior, and how the recommendations change
when the model has to reconcile its history with an actual experiment.

It uses the bundled `public_benchmark` dataset plus a synthetic
lift-test result. About 10–15 minutes on a laptop in `quick` mode (two
fits).

## Setup

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
pip install -e ".[dev]"
mkdir -p tutorial_data
cp benchmark_data/public_example.csv tutorial_data/data.csv
cat > tutorial_data/config_uncalibrated.yaml <<'EOF'
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

run:
  seed: 20260430
  runtime_mode: quick
  artifact_dir: .wanamaker
EOF
```

## Step 1 — Fit without calibration

```bash
wanamaker fit --config tutorial_data/config_uncalibrated.yaml
RUN_UNCAL=$(ls -t .wanamaker/runs | head -1)
wanamaker showcase --run-id "$RUN_UNCAL"
```

Open `.wanamaker/runs/$RUN_UNCAL/showcase.html`. Note the paid-social
ROI mean and the 95% credible interval. This is the model's view based
purely on observational history.

The Trust Card panel will show a `lift_test_consistency` dimension as
**not present** — there is no calibration data to compare against.

## Step 2 — Add the lift-test result

Suppose your geo-holdout produced a 12% lower point estimate than the
historical model has been showing — say a measured lift of 0.38 with a
95% CI of 0.30–0.46 (vs. an MMM-implied lift around 0.43). Encode this
as a one-row CSV:

```bash
cat > tutorial_data/lift_tests.csv <<'EOF'
channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper
paid_social,2024-09-01,2024-10-15,0.38,0.30,0.46
EOF
```

The required columns are `channel`, `test_start`, `test_end`,
`roi_estimate`, `roi_ci_lower`, `roi_ci_upper`. The CI bounds are converted
into a Gaussian prior on the channel's effect; tighter bounds put more
weight on the experiment.

If you have multiple tests for the same channel (different geos or different
time windows), put each on its own row. Wanamaker combines them via
precision-weighted pooling and reports the total in the Trust Card. This
assumes the tests are *independent* — overlapping market/time/audience
windows trigger a warning, in which case combine the tests externally or
widen one interval to absorb the correlation.

## Step 3 — Refit with calibration

Write a calibrated config that points at the lift-test CSV. Use the
same fields as the uncalibrated config plus a `calibration.lift_tests`
section:

```bash
cat > tutorial_data/config_calibrated.yaml <<'EOF'
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

calibration:
  lift_tests:
    path: tutorial_data/lift_tests.csv

run:
  seed: 20260430
  runtime_mode: quick
  artifact_dir: .wanamaker
EOF

wanamaker fit --config tutorial_data/config_calibrated.yaml
RUN_CAL=$(ls -t .wanamaker/runs | head -1)
wanamaker showcase --run-id "$RUN_CAL"
```

> **Tip:** Make sure the CSV path in the calibrated config is correct —
> the easiest mistake here is leaving a relative path that resolves
> differently from the run directory. Wanamaker validates the file
> exists at fit time and fails fast if it doesn't.

## Step 4 — Compare the two showcases

Open the two `showcase.html` files side by side. Three things will have
moved:

1. **Paid-social ROI mean** in the calibrated run will be pulled toward
   the lift test (lower than the uncalibrated estimate but not all the
   way down to it — Bayesian updating compromises between history and
   experiment in proportion to their relative precisions).
2. **Paid-social ROI credible interval** will be tighter, because the
   experiment supplied independent information.
3. **Trust Card** will now show a `lift_test_consistency` dimension. If
   the calibrated and uncalibrated estimates were close, it shows
   **pass**. If they disagree more than the joint uncertainty allows,
   it shows **weak** and explains what the disagreement implies for
   downstream decisions.

## Step 5 — See whether the verdict changes

If you have a budget plan that increases paid-social spend, run the
ramp recommendation against both runs:

```bash
cat > baseline.csv <<'EOF'
period,paid_search,paid_social,linear_tv,ctv,online_video,display,affiliate,email
2025-01-06,9700,8500,20800,12100,9700,4700,3300,2500
2025-01-13,9700,8500,20800,12100,9700,4700,3300,2500
2025-01-20,9700,8500,20800,12100,9700,4700,3300,2500
2025-01-27,9700,8500,20800,12100,9700,4700,3300,2500
EOF

cat > scale_paid_social.csv <<'EOF'
period,paid_search,paid_social,linear_tv,ctv,online_video,display,affiliate,email
2025-01-06,9700,12500,20800,12100,9700,4700,3300,2500
2025-01-13,9700,12500,20800,12100,9700,4700,3300,2500
2025-01-20,9700,12500,20800,12100,9700,4700,3300,2500
2025-01-27,9700,12500,20800,12100,9700,4700,3300,2500
EOF

wanamaker recommend-ramp --run-id "$RUN_UNCAL" \
    --baseline baseline.csv --target scale_paid_social.csv \
    --output uncalibrated_ramp.md

wanamaker recommend-ramp --run-id "$RUN_CAL" \
    --baseline baseline.csv --target scale_paid_social.csv \
    --output calibrated_ramp.md
```

Compare `uncalibrated_ramp.md` and `calibrated_ramp.md`. The expected
behavior:

- **Uncalibrated:** the ramp may suggest a more aggressive fraction
  because the model thinks paid-social is more productive than the
  experiment proved.
- **Calibrated:** the ramp is more conservative — lower fraction, or a
  `test_first` verdict. The lift-test prior pulled the expected return
  down, and the math now flags less headroom for the increase.

This is the calibration mechanism doing exactly what it should: an
expensive real-world experiment overrides historical correlations.

## What this example demonstrates

- Lift tests are **first-class inputs** to Wanamaker, not a separate
  reconciliation step you run by hand.
- The `lift_test_consistency` dimension on the Trust Card surfaces
  disagreement between MMM and experiments before it becomes a budget
  argument.
- A lift test can change the **direction** of a recommendation, not
  just the magnitude. The calibrated run is the one to act on; the
  uncalibrated run was the best the model could do without the
  experiment.

## Cost-benefit framing

Geo-holdouts are expensive — a 6-week test can mean low-six-figures of
deferred spend. This example exists because that cost is justified
when:

- The MMM's credible interval is wide enough that bigger budget moves
  feel risky, and
- The channel is large enough that a 10–20% recalibration changes
  meaningful dollars.

For small, well-identified channels, the test isn't worth running. The
Experiment Advisor in the showcase suggests which channels are the
best candidates for a calibration test based on width-of-CI and
spend-share.

## Next reading

- [DTC reallocates linear TV to CTV](tv_to_ctv.md)
- [Refresh after a noisy quarter](refresh_after_noisy_quarter.md)
- [What Wanamaker tells your CMO](../cmo_guide.md) — how to communicate
  a lift-test-driven plan change to non-technical stakeholders
