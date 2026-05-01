# API Reference

This page documents the public command-line, YAML, and Python API surface.

Generated Python sections use `mkdocstrings` and the Google-style docstrings in
`src/wanamaker`. Public names are shown; private helpers are omitted.

## CLI Reference

All commands run locally. The core commands do not make network calls.

### `wanamaker diagnose`

Run the pre-flight data readiness diagnostic.

```bash
wanamaker diagnose benchmark_data/public_example.csv --config benchmark_data/public_example.yaml
```

| Argument or option | Type | Default | Description |
|---|---|---|---|
| `data` | path | required | CSV file to inspect. |
| `--config`, `-c` | path | none | Optional YAML config. When supplied, spend and control column checks run. |
| `--date-column`, `-d` | string | inferred | Date column name when no config is supplied. |
| `--target-column`, `-t` | string | inferred | Numeric target column name when no config is supplied. |

Exit code is non-zero when the readiness level is `diagnostic_only` or
`not_recommended`.

### `wanamaker fit`

Fit the Bayesian model and write a versioned run directory.

```bash
wanamaker fit --config benchmark_data/public_example.yaml
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--config`, `-c` | path | required | YAML config file. |
| `--skip-validation` | flag | false | Hidden expert flag. Bypasses readiness checks and records the skip in artifacts. |

The command prints a run ID. Use that run ID with `report`, `forecast`, and
`compare-scenarios`.

### `wanamaker report`

Render the Markdown executive summary and Model Trust Card for a completed run.

```bash
wanamaker report --run-id <run_id>
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing run ID under the artifact directory. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--output` | path | `<run_dir>/report.md` | Markdown report path. |

### `wanamaker showcase`

Render a single self-contained HTML showcase suitable for forwarding to
stakeholders. Bundles channel charts, response curves, ROI, the contribution
waterfall, the Trust Card, and an optional side-by-side scenario comparison.

```bash
wanamaker showcase --run-id <run_id>
wanamaker showcase --run-id <run_id> --scenario base.csv --scenario alt.csv
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing run ID under the artifact directory. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--output` | path | `<run_dir>/showcase.html` | Output HTML path. |
| `--title` | string | `"Wanamaker MMM — <run_id short>"` | Override the showcase title. |
| `--scenario` | path, repeatable | none | Plan CSV to forecast and overlay. Pass multiple times for side-by-side comparison; the first plan is the baseline for the delta column. |
| `--open` | flag | false | Open the rendered showcase in the default browser. |

### `wanamaker trust-card`

Render the executive-facing Trust Card one-pager. Single self-contained HTML
file designed for forwarding to non-technical readers and printing to one
physical page.

```bash
wanamaker trust-card --run-id <run_id>
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing run ID under the artifact directory. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--output` | path | `<run_dir>/trust_card.html` | Output HTML path. |
| `--title` | string | `"Wanamaker MMM — <run_id short>"` | Override the title. |
| `--open` | flag | false | Open the rendered one-pager in the default browser. |

### `wanamaker export`

Export a run's structured tables to an analyst-friendly Excel workbook.
Sheets: Summary, Channels, Trust Card, Parameters, optional Refresh diff,
optional Scenarios. Numbers are stored as numbers (not formatted strings) so
formulas and pivots work directly.

```bash
wanamaker export --run-id <run_id> --format excel
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing run ID under the artifact directory. |
| `--format` | string | `excel` | Export format. Currently only `excel` (.xlsx) is implemented. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--output` | path | `<run_dir>/summary.xlsx` | Output workbook path. |
| `--scenario` | path, repeatable | none | Plan CSV to forecast and include as a row in the Scenarios sheet. |

### `wanamaker forecast`

Forecast the target metric under one future spend plan.

```bash
wanamaker forecast --run-id <run_id> --plan path/to/plan.csv
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing fitted run ID. |
| `--plan` | path | required | CSV future spend plan. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--seed` | integer | run seed | Override posterior-predictive sampling seed. |
| `--output` | path | `<run_dir>/forecast_<plan>.md` | Markdown forecast report path. |

### `wanamaker compare-scenarios`

Rank one or more future spend plans with uncertainty.

```bash
wanamaker compare-scenarios --run-id <run_id> --plans path/to/base.csv --plans path/to/test.csv
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing fitted run ID. |
| `--plans` | path, repeatable | required | Future spend plan CSV. Repeat once per scenario. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--seed` | integer | run seed | Override posterior-predictive sampling seed. |
| `--output` | path | `<run_dir>/scenario_comparison.md` | Markdown scenario report path. |

### `wanamaker recommend-ramp`

Recommend a risk-adjusted staged move from a baseline plan toward a target plan.

```bash
wanamaker recommend-ramp --run-id <run_id> --baseline path/to/base.csv --target path/to/alt.csv
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | required | Existing fitted run ID. |
| `--baseline` | path | required | Current or status-quo future spend plan CSV. |
| `--target` | path | required | Candidate future spend plan CSV. |
| `--artifact-dir` | path | `.wanamaker` | Root artifact directory. |
| `--seed` | integer | run seed | Override posterior-predictive sampling seed. |
| `--output` | path | `<run_dir>/ramp_<baseline>_to_<target>.md` | Markdown ramp report path. |

### `wanamaker refresh`

Re-run the model with posterior anchoring against the previous compatible run.

```bash
wanamaker refresh --config benchmark_data/public_example.yaml --anchor-strength light
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--config`, `-c` | path | required | YAML config file. |
| `--anchor-strength` | `none`, `light`, `medium`, `heavy`, or float | config value | Override refresh anchoring strength for this run. |
| `--skip-validation` | flag | false | Hidden expert flag. Bypasses readiness checks and records the skip in artifacts. |

## YAML Config Reference

The YAML config is validated by `wanamaker.config.WanamakerConfig`.
Unknown fields are rejected.

### Top Level

| Field | Type | Default | Description |
|---|---|---|---|
| `data` | `DataConfig` | required | Input CSV and column mapping. |
| `channels` | list of `ChannelConfig` | `[]` | Per-channel category and adstock settings. |
| `refresh` | `RefreshConfig` | `{anchor_strength: medium}` | Refresh accountability settings. |
| `run` | `RunConfig` | `{seed: 0, runtime_mode: standard, artifact_dir: .wanamaker}` | Reproducibility and artifact settings. |

### `data`

| Field | Type | Default | Description |
|---|---|---|---|
| `csv_path` | path | required | Weekly aggregate input CSV. |
| `date_column` | string | required | Date column in the CSV. |
| `target_column` | string | required | Target metric column, such as revenue or conversions. |
| `spend_columns` | list of strings | `[]` | Paid media spend columns. |
| `control_columns` | list of strings | `[]` | Non-media controls such as price, promotions, holidays, or macro indicators. |
| `lift_test_csv` | path or null | `null` | Optional lift-test calibration CSV. |

### `channels`

Each entry describes one spend column.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Spend column name. |
| `category` | string | required | Channel category used to look up default priors. |
| `adstock_family` | `geometric` or `weibull` | `geometric` | Carryover transform family for the channel. |

### `refresh`

| Field | Type | Default | Description |
|---|---|---|---|
| `anchor_strength` | `none`, `light`, `medium`, `heavy`, or float | `medium` | Posterior anchoring preset or numeric weight in `[0, 1]`. |

### `run`

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | integer | `0` | Top-level seed passed down explicitly for reproducibility. |
| `runtime_mode` | `quick`, `standard`, or `full` | `standard` | Bayesian sampler effort tier. Model structure is unchanged. |
| `artifact_dir` | path | `.wanamaker` | Local artifact root. |

### Minimal Example

```yaml
data:
  csv_path: benchmark_data/public_example.csv
  date_column: week
  target_column: revenue
  spend_columns:
    - paid_search
    - paid_social
  control_columns:
    - promo_flag

channels:
  - name: paid_search
    category: paid_search
  - name: paid_social
    category: paid_social
    adstock_family: weibull

refresh:
  anchor_strength: medium

run:
  seed: 20260430
  runtime_mode: standard
  artifact_dir: .wanamaker
```

## Python API Reference

### Package Metadata

::: wanamaker

### CLI Module

::: wanamaker.cli

### Configuration

::: wanamaker.config

### Data

::: wanamaker.data.io

::: wanamaker.data.taxonomy

### Diagnostics

::: wanamaker.diagnose.readiness

::: wanamaker.diagnose.checks

### Transforms

::: wanamaker.transforms.adstock

::: wanamaker.transforms.saturation

### Model

::: wanamaker.model.spec

::: wanamaker.model.priors

::: wanamaker.model.builder

### Engine

::: wanamaker.engine.base

::: wanamaker.engine.summary

::: wanamaker.engine.pymc

### Artifacts

::: wanamaker.artifacts

### Refresh

::: wanamaker.refresh.anchor

::: wanamaker.refresh.classify

::: wanamaker.refresh.diff

### Forecast

::: wanamaker.forecast.posterior_predictive

::: wanamaker.forecast.ramp

::: wanamaker.forecast.scenarios

### Trust Card

::: wanamaker.trust_card.card

::: wanamaker.trust_card.compute

### Reports

::: wanamaker.reports.render

### Experiment Advisor

::: wanamaker.advisor.channel_flagging

### Benchmarks

::: wanamaker.benchmarks.loaders

### Reproducibility

::: wanamaker.seeding
