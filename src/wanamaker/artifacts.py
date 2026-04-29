"""Local-only artifact storage (FR-Privacy.2, FR-4.1).

Every model fit produces a versioned artifact under the project-local
``.wanamaker/`` directory:

    .wanamaker/
    └── runs/
        └── <run_id>/
            ├── manifest.json        # run fingerprint, seed, engine, schema version
            ├── config.yaml          # snapshot of the run config
            ├── data_hash.txt        # sha256 of the input CSV
            ├── posterior.nc         # posterior draws (engine-native format)
            ├── summary.json         # PosteriorSummary (used by refresh diffs and reports)
            ├── trust_card.json      # computed TrustCard (persisted for auditability)
            ├── refresh_diff.json    # RefreshDiff vs. prior run (present only on refresh)
            ├── timestamp.txt        # ISO-8601 UTC fit timestamp
            └── engine.txt           # name + version of the Bayesian engine

Two identity concepts are kept separate:

- **run_fingerprint**: a deterministic hash of (data_hash, normalized config,
  package version, engine name/version, seed). Two runs with the same
  fingerprint produced identical analytical inputs and should produce
  identical results. Used by refresh/diff to detect "same analytical setup."

- **run_id**: the storage key, derived from run_fingerprint plus a UTC
  timestamp. Unique per execution. Repeated runs of the same (data, config,
  seed) produce separate run_ids pointing to the same run_fingerprint --
  they are distinct artifacts but analytically equivalent.

This module owns directory layout, file naming, and these two identities.
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
    def manifest(self) -> Path:
        return self.root / "manifest.json"

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
    def trust_card(self) -> Path:
        return self.root / "trust_card.json"

    @property
    def refresh_diff(self) -> Path:
        return self.root / "refresh_diff.json"

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


def make_run_fingerprint(
    data_hash: str,
    config_hash: str,
    package_version: str,
    engine_name: str,
    engine_version: str,
    seed: int,
) -> str:
    """Return a deterministic fingerprint for an analytical setup.

    Two runs with the same fingerprint used identical inputs and should
    produce identical results (NFR-2). This is used by the refresh diff
    to identify "same analytical setup" regardless of when it ran.

    Args:
        data_hash: SHA-256 of the input CSV.
        config_hash: SHA-256 of the normalized config YAML.
        package_version: ``wanamaker.__version__``.
        engine_name: Stable engine identifier (e.g. ``"pymc"``).
        engine_version: Engine library version string.
        seed: The top-level random seed from config.

    Returns:
        Lowercase hex BLAKE2b fingerprint (16 bytes, 32 hex chars).
    """
    payload = "|".join([
        data_hash, config_hash, package_version,
        engine_name, engine_version, str(seed),
    ])
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


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
