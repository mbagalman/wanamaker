# Quickstart

This page will become the end-to-end tutorial for a first Wanamaker analysis.

## Target Outcome

A new user should be able to install Wanamaker, run the public example dataset, and produce
an executive summary in under 30 minutes.

## Planned Workflow

```bash
wanamaker diagnose benchmark_data/public_example.csv --config benchmark_data/public_example.yaml
wanamaker fit --config benchmark_data/public_example.yaml
wanamaker report --run-id <run_id>
```

## What This Tutorial Will Cover

- Preparing a weekly aggregate marketing CSV
- Running the readiness diagnostic
- Reading warnings before fitting
- Fitting the Bayesian model
- Rendering the executive summary and Trust Card
- Finding artifacts under `.wanamaker/`

## To Be Completed

- Public example dataset
- Example YAML config
- Expected command output
- Troubleshooting notes for common data issues

