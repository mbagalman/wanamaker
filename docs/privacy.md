# Privacy and Data Handling

Wanamaker is designed as a local-first tool. Core operations must not send data to remote
services and must not include telemetry.

## Local Processing

The core commands run on the user's machine:

- `diagnose`
- `fit`
- `report`
- `forecast`
- `compare-scenarios`
- `refresh`

## Files Written To Disk

Model artifacts are stored in a project-local `.wanamaker/` directory by default.

Planned run layout:

```text
.wanamaker/
+-- runs/
    +-- <run_id>/
        +-- manifest.json
        +-- config.yaml
        +-- data_hash.txt
        +-- posterior.nc
        +-- summary.json
        +-- timestamp.txt
        +-- engine.txt
```

## Air-Gapped Use

After dependencies are installed, core commands should work without network access. A
network-isolated integration test is planned before release.

## Telemetry

There is no telemetry. Any future telemetry system would need to be explicit opt-in,
disabled by default, documented, and easy to disable.

## To Be Completed

- Exact artifact list after implementation
- Docker instructions for air-gapped environments
- Agency confidentiality review

