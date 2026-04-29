"""Local-only artifact storage (FR-Privacy.2, FR-4.1).

Every model fit produces a versioned artifact under the project-local
``.wanamaker/`` directory:

    .wanamaker/
    └── runs/
        └── <run_id>/
            ├── config.yaml          # snapshot of the run config
            ├── data_hash.txt        # sha256 of the input CSV
            ├── posterior.nc         # posterior draws (engine-native format)
            ├── summary.json         # marginal summaries used for refresh diffs
            ├── timestamp.txt        # ISO-8601 UTC fit timestamp
            └── engine.txt           # name + version of the Bayesian engine

Run IDs are short content-addressed identifiers derived from the data hash,
the config, and the timestamp — they are stable for a given (data, config)
pair within the same minute, so identical re-runs don't proliferate
directories.

This module owns directory layout, file naming, and reproducible run IDs.
It does not own the contents of ``posterior.nc`` or ``summary.json``;
those live in the engine and refresh modules respectively.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Resolved filesystem paths for a single run's artifacts."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def data_hash(self) -> Path:
        return self.root / "data_hash.txt"

    @property
    def posterior(self) -> Path:
        return self.root / "posterior.nc"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def timestamp(self) -> Path:
        return self.root / "timestamp.txt"

    @property
    def engine(self) -> Path:
        return self.root / "engine.txt"


def hash_file(path: Path, chunk_size: int = 1 << 16) -> str:
    """Return the lowercase hex sha256 of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def run_paths(artifact_dir: Path, run_id: str) -> RunPaths:
    """Return the ``RunPaths`` for a run, creating the directory if needed."""
    root = artifact_dir / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root)


def list_runs(artifact_dir: Path) -> list[str]:
    """List existing run IDs in ``artifact_dir``, oldest first."""
    runs_dir = artifact_dir / "runs"
    if not runs_dir.exists():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
