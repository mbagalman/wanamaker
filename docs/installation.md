# Installation

This page will cover the supported installation paths for Wanamaker.

## Status

Wanamaker is in Phase -1. The Bayesian engine has not been selected yet, so installation
instructions are not final.

## Planned Paths

### pip

The target path for users comfortable with Python environments:

```bash
pip install wanamaker
```

For local development:

```bash
pip install -e ".[dev,docs]"
```

### Docker

The target path for users who want all dependencies pre-resolved.

#### Build from source (available today)

A `Dockerfile` ships in the repository root. Until the official
`wanamaker/wanamaker` image is published to Docker Hub at v1.0, build
the image locally:

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
docker build -t wanamaker .
```

Run the bundled one-command demo inside the image:

```bash
docker run --rm wanamaker run --example public_benchmark
```

Run any subcommand against your own data by mounting the file:

```bash
docker run --rm -v "$PWD":/workspace wanamaker diagnose data.csv --config config.yaml
```

The image leaves run artifacts in the container's `/workspace/.wanamaker/`
directory; mounting your project root lets `wanamaker fit` write them
back to the host filesystem so they persist between runs.

#### Pulled image (after v1.0)

Once the image is published, the same flow becomes:

```bash
docker run --rm wanamaker/wanamaker run --example public_benchmark
```

## Requirements

- Python 3.11+
- No R runtime
- No GPU requirement for v1
- No network access required for core commands after installation (see [Privacy and Data Handling](privacy.md) for details on our local-first guarantees)

## To Be Completed

- Final engine-specific dependency notes
- Windows, macOS, and Linux installation checks
- Docker mount examples for local CSV files

