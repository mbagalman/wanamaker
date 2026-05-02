# Privacy and Data Handling

Wanamaker is designed from the ground up as a local-first tool. Core operations must not send data to remote services and must not include telemetry.

## Explicit Statement on Data Handling

**No data leaves your machine under any normal operation.**
When you run Wanamaker commands, all data loading, Bayesian inference, forecasting, and reporting are executed entirely on your local machine or your organization's servers. Wanamaker does not phone home, does not upload your CSVs to a cloud service, and does not use API calls to external LLMs or third-party platforms.

## Files Written to Disk

All model artifacts and outputs are strictly stored locally in a project-local `.wanamaker/` directory by default. Wanamaker writes no files to your home directory or other system paths.

The directory layout for run artifacts is as follows:

```text
.wanamaker/
+-- runs/
    +-- <run_id>/
        +-- manifest.json       # run_fingerprint, seed, engine, schema versions, skip_validation flag
        +-- config.yaml         # snapshot of the run config
        +-- data_hash.txt       # SHA-256 of the input CSV
        +-- posterior.nc        # posterior draws (engine-native format, e.g. NetCDF via ArviZ)
        +-- summary.json        # PosteriorSummary (versioned envelope)
        +-- trust_card.json     # TrustCard (versioned envelope; persisted for auditability)
        +-- refresh_diff.json   # RefreshDiff vs. prior run (present only on refresh; versioned envelope)
        +-- timestamp.txt       # ISO-8601 UTC fit timestamp
        +-- engine.txt          # engine name + version
```

### Configuring an Alternate Artifact Directory

If you need to store artifacts elsewhere (for example, on a different drive or within an agency-specific client folder), you can configure an alternate directory. This can be done by specifying `artifact_dir` in the run configuration (`config.run.artifact_dir`) or by passing the appropriate command-line interface (CLI) flag when executing a run.

## Agency Client Confidentiality Obligations

Wanamaker's architecture ensures that marketing agencies can safely use the software while meeting strict client confidentiality and non-disclosure obligations:
- **Strictly Local Storage:** Because all analysis is local, client spend data, taxonomies, and performance outputs remain securely within the agency's controlled infrastructure.
- **No Third-Party Access:** The absence of remote dependencies means no third-party vendor ever has access to your client's data.
- **Isolated Artifacts:** By configuring the artifact directory (as described above), an agency can easily silo different clients' model artifacts into isolated, access-controlled network drives or encrypted volumes.

## Air-Gapped Environments

Wanamaker is fully functional in air-gapped or network-isolated environments.

To use Wanamaker in an air-gapped setup:
1. **Docker (Recommended):** Use the official Wanamaker Docker image (`docker run wanamaker/wanamaker`). The container comes with all dependencies pre-resolved. You can mount your local client CSVs and configuration files into the container at runtime.
2. **Local Python Install:** If installing via `pip`, you can download the wheel and its dependencies on an internet-connected machine, transfer them to the air-gapped machine, and install them offline.
Once installed, no network connection is required to run the core operations (`diagnose`, `fit`, `report`, `forecast`, `compare-scenarios`, `refresh`). Wanamaker's core is continuously verified against a CI network-isolation test.

## Telemetry Policy

There is **no telemetry, ever**, included in the core Wanamaker codebase.

If any usage telemetry features are considered in the future, they will adhere to the following strict policy:
- **Opt-in Only:** Any telemetry must be explicitly opted into by the user. It will never be enabled by default.
- **Documented:** The exact data collected will be transparently documented.
- **Easy to Disable:** The mechanism to disable it will be trivial and permanently respected.