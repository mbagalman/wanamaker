# Quickstart

This tutorial walks through a first Wanamaker analysis using the public example
dataset included in the repository.

The goal is to go from installation to a first Markdown report. The report is
not a final business recommendation by itself. It is the starting point for
reading channel estimates, uncertainty, and the Trust Card.

## What You Need

- Python 3.11 or newer
- A local checkout of the Wanamaker repository
- A terminal in the repository root

The public example uses synthetic, anonymized weekly marketing data. It is safe
to inspect and commit publicly.

## Install

For development or source checkout usage:

```bash
pip install -e ".[dev]"
```

If you only want to build documentation:

```bash
pip install -e ".[docs]"
```

The Bayesian engine dependencies can take longer to install than the rest of the
package. That is normal.

## Inspect The Example Files

The quickstart uses:

```text
benchmark_data/public_example.csv
benchmark_data/public_example_metadata.json
benchmark_data/public_example.yaml
```

The CSV has one row per week. The key columns are:

| Type | Columns |
|---|---|
| Date | `week` |
| Target | `revenue` |
| Media spend | `paid_search`, `paid_social`, `linear_tv`, `ctv`, `online_video`, `display`, `affiliate`, `email` |
| Controls | `price_index`, `promo_flag`, `holiday_flag`, `macro_index` |

The YAML config tells Wanamaker which columns are media channels, which columns
are controls, and which runtime tier to use.

## Step 1: Run The Readiness Diagnostic

Start by checking whether the data is suitable for MMM:

```bash
wanamaker diagnose benchmark_data/public_example.csv --config benchmark_data/public_example.yaml
```

The diagnostic looks for common problems such as:

- insufficient history
- missing values
- irregular dates
- low spend variation
- collinearity
- too many predictors for the amount of history
- target leakage
- structural breaks

Expected result for the public example: the command should finish without
blockers. Warnings are possible as checks evolve; read them before fitting.

Readiness levels are:

| Level | Meaning |
|---|---|
| `ready` | No blockers or major warnings found |
| `usable_with_warnings` | Fit is possible, but some outputs need caution |
| `diagnostic_only` | Useful for understanding data, not for decisions |
| `not_recommended` | Do not fit until the data problem is fixed |

## Step 2: Fit The Model

Run a quick fit:

```bash
wanamaker fit --config benchmark_data/public_example.yaml
```

The config uses `runtime_mode: quick` so the tutorial is practical on a laptop.
For real decision support, use the default or full runtime after the workflow is
working.

At the end, Wanamaker prints a run ID and an artifact directory, for example:

```text
Done. Run ID: 1a2b3c4d_20260430T120000Z
  .wanamaker/runs/1a2b3c4d_20260430T120000Z
```

Copy the run ID for the next step.

The run directory contains files such as:

| File | Purpose |
|---|---|
| `manifest.json` | Run ID, fingerprint, seed, engine, readiness level |
| `config.yaml` | Snapshot of the config used for the run |
| `data_hash.txt` | Hash of the input CSV |
| `posterior.nc` | Engine-native posterior draws |
| `summary.json` | Engine-neutral posterior summary |
| `timestamp.txt` | Fit timestamp |
| `engine.txt` | Engine name and version |

## Step 3: Render The Report

Use the run ID from the fit command:

```bash
wanamaker report --run-id <run_id>
```

By default, this writes:

```text
.wanamaker/runs/<run_id>/report.md
```

The report includes:

- an executive summary
- channel contribution and ROI language
- uncertainty-aware wording
- the Model Trust Card

The executive summary is deterministic template output. Wanamaker does not use
LLM calls to generate product reports.

## Step 4: Read The Trust Card

Before using the results, read the Trust Card.

Pay special attention to any dimension marked `weak`:

- convergence
- holdout accuracy
- refresh stability
- prior sensitivity
- saturation identifiability
- lift-test consistency

A weak dimension does not always mean the whole model is useless. It means the
affected output needs caution. For example, weak saturation identifiability
means you should not make strong reallocation decisions based on that channel's
response curve.

## Step 5: Compare Future Plans

Once you have a fitted run, you can compare user-supplied future spend plans.
Each plan should use the same channel names as the training data.

Example wide-format plan:

```csv
period,paid_search,paid_social,linear_tv,ctv,online_video,display,affiliate,email
2026-01-05,21000,16000,30000,15000,12000,8500,6000,2800
2026-01-12,21000,16000,30000,15000,12000,8500,6000,2800
```

Forecast one plan:

```bash
wanamaker forecast --run-id <run_id> --plan path/to/plan.csv
```

Compare multiple plans:

```bash
wanamaker compare-scenarios --run-id <run_id> --plans path/to/base.csv path/to/test.csv
```

Scenario comparison is not automatic optimization. You provide the candidate
plans. Wanamaker ranks them with uncertainty and flags extrapolation beyond the
historical spend ranges seen in the training data.

## Step 6: Refresh When New Data Arrives

After more weeks of data are available, update the CSV and run:

```bash
wanamaker refresh --config benchmark_data/public_example.yaml
```

Refresh uses light posterior anchoring and writes a diff report. The diff shows
which historical estimates moved and whether the movement looks expected,
user-induced, weakly identified, or unexplained.

Use refresh to avoid silently rewriting history. A changed estimate is not
automatically bad, but it should be visible and explainable.

## Troubleshooting

### The diagnostic says the data is not recommended

Do not jump straight to `--skip-validation`. Read the blockers first. Common
causes are too little history, missing target values, duplicate dates, or a
structural break.

### Fit is slow

Start with `runtime_mode: quick` in the YAML config. Once the workflow works,
move to `standard` for normal analysis and `full` for final decision support.

### A channel has weak saturation identifiability

Check whether spend changed enough in history. Flat spend can support a rough
contribution estimate, but it cannot support a confident response curve.

### Scenario comparison flags extrapolation

The plan is outside the spend range the model observed. Treat the result as a
staged test candidate, not as proof that the full move is safe.

### The report language is cautious

That is intentional. Wanamaker changes wording when uncertainty is high or the
Trust Card flags a risk. Cautious language is a feature, not a failure.

## Next Reading

- [Analyst's Guide](analyst_guide.md)
- [What Wanamaker Tells Your CMO](cmo_guide.md)
- [Privacy and Data Handling](privacy.md)
- [Technical Architecture](architecture.md)
