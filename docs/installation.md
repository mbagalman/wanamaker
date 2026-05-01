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

The target path for users who want all dependencies pre-resolved:

```bash
docker run wanamaker/wanamaker wanamaker --help
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

