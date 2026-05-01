# Installation

Wanamaker is in active pre-1.0 development. The package is not published to
PyPI yet, so the supported path today is a source checkout.

## Requirements

- Python 3.11+
- No R runtime
- No GPU requirement for v1
- No network access required for core commands after installation

The selected Bayesian engine is PyMC. Install time can be longer than a small
pure-Python package because the scientific Python stack is part of the runtime.

## Source Checkout

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
pip install -e ".[dev]"
```

For documentation work:

```bash
pip install -e ".[docs]"
```

For both development and docs:

```bash
pip install -e ".[dev,docs]"
```

Verify the CLI is available:

```bash
wanamaker --help
wanamaker run --example public_benchmark
```

The example writes artifacts under `.wanamaker/runs/<run_id>/`.

## Docker

A `Dockerfile` ships in the repository root. Until an official image is
published, build it locally:

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
docker build -t wanamaker .
```

Run the bundled example:

```bash
docker run --rm wanamaker run --example public_benchmark
```

Run a command against local files by mounting your project directory:

```bash
docker run --rm -v "$PWD":/workspace wanamaker diagnose /workspace/data.csv --config /workspace/config.yaml
```

Run artifacts written under `/workspace/.wanamaker/` persist on the host when
the project directory is mounted.

## Not Yet Published

The future user-facing install paths are still release work:

- `pip install wanamaker` after the PyPI package is published
- `docker run --rm wanamaker/wanamaker ...` after the official image is published

Until then, use the source checkout or locally built Docker image.
