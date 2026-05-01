# Wanamaker Docker image (FR-6.2).
#
# Single-stage on python:3.11-slim. The build toolchain stays in the
# runtime image because PyTensor (PyMC's tensor library) compiles
# generated C code lazily on first model build. Without gcc / g++ in
# the runtime layer, ``wanamaker fit`` would fail at sample-time when
# PyTensor tries to compile its scan/op graph.
#
# Image size target (FR-6.2): under 3 GB.
#
# Usage:
#   # Build:
#   docker build -t wanamaker .
#
#   # Run the bundled one-command demo:
#   docker run --rm wanamaker run --example public_benchmark
#
#   # Run any wanamaker subcommand against a mounted dataset:
#   docker run --rm -v "$PWD":/workspace wanamaker diagnose data.csv

FROM python:3.11-slim AS runtime

# System packages PyTensor needs at runtime to JIT-compile its graph.
# ``--no-install-recommends`` and removing the apt cache keep the
# image as small as possible while still leaving the toolchain in
# place for the lazy compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Working directory for user-supplied data. Users mount their CSVs and
# configs here with ``-v $PWD:/workspace``; ``.wanamaker/`` artifacts
# also land here so the host filesystem keeps the run output.
WORKDIR /workspace

# Copy only what the package install needs. ``.dockerignore`` filters
# the rest (tests, docs, .git, etc.) so the build context stays small.
COPY pyproject.toml README.md LICENSE /opt/wanamaker/
COPY src/ /opt/wanamaker/src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /opt/wanamaker

# ``wanamaker`` is on PATH (via the project.scripts entry-point), so
# ENTRYPOINT lets users say ``docker run wanamaker diagnose ...``
# without retyping ``wanamaker``. CMD prints help when no subcommand
# is supplied.
ENTRYPOINT ["wanamaker"]
CMD ["--help"]
