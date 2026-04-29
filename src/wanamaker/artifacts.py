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

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = 1
"""Bump when the manifest.json structure changes in a backward-incompatible way."""

SUMMARY_SCHEMA_VERSION = 1
"""Bump when the summary.json structure changes in a backward-incompatible way."""

TRUST_CARD_SCHEMA_VERSION = 1
"""Bump when the trust_card.json structure changes in a backward-incompatible way."""

REFRESH_DIFF_SCHEMA_VERSION = 1
"""Bump when the refresh_diff.json structure changes in a backward-incompatible way."""


def _wrap(payload: dict[str, Any], schema_version: int) -> str:
    """Wrap a serialized payload in a versioned envelope and return JSON.

    Every persisted artifact uses the same envelope so that any deserializer
    can detect and reject incompatible versions before attempting to parse
    the payload:

        { "schema_version": <int>, "payload": { ... } }

    Args:
        payload: The artifact dict produced by ``dataclasses.asdict``.
        schema_version: The current schema version constant for this artifact type.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps({"schema_version": schema_version, "payload": payload}, indent=2)


def _unwrap(json_str: str, expected_version: int, artifact_name: str) -> dict[str, Any]:
    """Parse a versioned envelope, validating the schema version.

    Args:
        json_str: Raw JSON string from disk.
        expected_version: The current schema version constant for this type.
        artifact_name: Human-readable name used in the error message.

    Returns:
        The ``payload`` dict extracted from the envelope.

    Raises:
        ValueError: If ``schema_version`` is missing or does not match
            ``expected_version``.  The error message names the artifact, the
            found version, and the expected version so the user knows what to do.
        json.JSONDecodeError: If ``json_str`` is not valid JSON.
    """
    d: dict[str, Any] = json.loads(json_str)
    found = d.get("schema_version")
    if found != expected_version:
        raise ValueError(
            f"Incompatible {artifact_name} schema version {found!r}. "
            f"Expected {expected_version}. "
            "Re-run 'wanamaker fit' to regenerate this artifact."
        )
    return dict(d["payload"])


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


def hash_config(config_dict: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hash of a config dict.

    The dict is serialised to JSON with sorted keys so that field ordering
    differences do not produce different hashes.  Only analytical fields
    should be included (exclude ``artifact_dir`` and other storage paths).
    """
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def serialize_summary(summary: Any) -> str:
    """Serialize a ``PosteriorSummary`` frozen dataclass to a versioned JSON string.

    Wraps the payload in ``{ "schema_version": SUMMARY_SCHEMA_VERSION, "payload": ... }``
    so that ``deserialize_summary`` can detect and reject incompatible artifacts.
    Uses ``dataclasses.asdict`` recursively to convert all nested frozen dataclasses.
    """
    return _wrap(dataclasses.asdict(summary), SUMMARY_SCHEMA_VERSION)


def write_manifest(
    paths: RunPaths,
    *,
    run_id: str,
    run_fingerprint: str,
    timestamp: str,
    seed: int,
    engine_name: str,
    engine_version: str,
    wanamaker_version: str,
    skip_validation: bool,
    readiness_level: str | None,
    summary_schema_version: int = SUMMARY_SCHEMA_VERSION,
) -> None:
    """Write ``manifest.json`` to the run artifact directory."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary_schema_version": summary_schema_version,
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "timestamp": timestamp,
        "seed": seed,
        "engine": {"name": engine_name, "version": engine_version},
        "wanamaker_version": wanamaker_version,
        "skip_validation": skip_validation,
        "readiness_level": readiness_level,
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2))


def load_manifest(json_str: str) -> dict[str, Any]:
    """Parse a ``manifest.json`` string, checking schema version compatibility.

    Args:
        json_str: Contents of a ``manifest.json`` file.

    Returns:
        Parsed manifest as a plain dict.

    Raises:
        ValueError: If ``schema_version`` is missing or incompatible with the
            current ``MANIFEST_SCHEMA_VERSION``. The error message includes the
            found version and the expected version so the user knows what to do.
    """
    d: dict[str, Any] = json.loads(json_str)
    found = d.get("schema_version")
    if found != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible manifest schema version {found!r}. "
            f"Expected {MANIFEST_SCHEMA_VERSION}. "
            "Re-run 'wanamaker fit' to generate a compatible artifact."
        )
    return d


def deserialize_summary(json_str: str) -> Any:
    """Reconstruct a ``PosteriorSummary`` from a ``summary.json`` string.

    This is the inverse of ``serialize_summary``. Validates the envelope
    schema version before parsing, raising ``ValueError`` for incompatible
    artifacts.  Reconstructs all nested frozen dataclasses from plain dicts.

    Args:
        json_str: Contents of a ``summary.json`` file.

    Returns:
        A fully populated ``PosteriorSummary`` instance.

    Raises:
        ValueError: If the schema version in the envelope does not match
            ``SUMMARY_SCHEMA_VERSION``.
        KeyError: If required fields are missing from the payload.
        json.JSONDecodeError: If ``json_str`` is not valid JSON.
    """
    from wanamaker.engine.summary import (
        ChannelContributionSummary,
        ConvergenceSummary,
        ParameterSummary,
        PosteriorSummary,
        PredictiveSummary,
    )

    d: dict[str, Any] = _unwrap(json_str, SUMMARY_SCHEMA_VERSION, "summary.json")

    parameters = [ParameterSummary(**p) for p in d.get("parameters", [])]
    channel_contributions = [
        ChannelContributionSummary(**c) for c in d.get("channel_contributions", [])
    ]
    convergence = (
        ConvergenceSummary(**d["convergence"]) if d.get("convergence") is not None else None
    )
    in_sample_predictive = (
        PredictiveSummary(**d["in_sample_predictive"])
        if d.get("in_sample_predictive") is not None
        else None
    )
    return PosteriorSummary(
        parameters=parameters,
        channel_contributions=channel_contributions,
        convergence=convergence,
        in_sample_predictive=in_sample_predictive,
    )


def serialize_trust_card(card: Any) -> str:
    """Serialize a ``TrustCard`` frozen dataclass to a versioned JSON string.

    Wraps the payload in a versioned envelope so that ``deserialize_trust_card``
    can detect and reject incompatible artifacts.

    Args:
        card: A ``TrustCard`` instance.

    Returns:
        Pretty-printed JSON string suitable for writing to ``trust_card.json``.
    """
    return _wrap(dataclasses.asdict(card), TRUST_CARD_SCHEMA_VERSION)


def deserialize_trust_card(json_str: str) -> Any:
    """Reconstruct a ``TrustCard`` from a ``trust_card.json`` string.

    Validates the envelope schema version before parsing.

    Args:
        json_str: Contents of a ``trust_card.json`` file.

    Returns:
        A fully populated ``TrustCard`` instance.

    Raises:
        ValueError: If the schema version does not match ``TRUST_CARD_SCHEMA_VERSION``
            or a ``TrustStatus`` value is not recognised.
        json.JSONDecodeError: If ``json_str`` is not valid JSON.
    """
    from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus

    d: dict[str, Any] = _unwrap(json_str, TRUST_CARD_SCHEMA_VERSION, "trust_card.json")
    dimensions = [
        TrustDimension(
            name=dim["name"],
            status=TrustStatus(dim["status"]),
            explanation=dim["explanation"],
        )
        for dim in d.get("dimensions", [])
    ]
    return TrustCard(dimensions=dimensions)


def serialize_refresh_diff(diff: Any) -> str:
    """Serialize a ``RefreshDiff`` frozen dataclass to a versioned JSON string.

    ``ParameterMovement.previous_ci`` and ``current_ci`` are tuples; they are
    serialised as JSON arrays and restored as tuples by ``deserialize_refresh_diff``.
    Wraps the payload in a versioned envelope.

    Args:
        diff: A ``RefreshDiff`` instance.

    Returns:
        Pretty-printed JSON string suitable for writing to ``refresh_diff.json``.
    """
    return _wrap(dataclasses.asdict(diff), REFRESH_DIFF_SCHEMA_VERSION)


def deserialize_refresh_diff(json_str: str) -> Any:
    """Reconstruct a ``RefreshDiff`` from a ``refresh_diff.json`` string.

    Validates the envelope schema version before parsing. JSON arrays for
    ``previous_ci`` and ``current_ci`` are converted back to ``tuple[float, float]``.

    Args:
        json_str: Contents of a ``refresh_diff.json`` file.

    Returns:
        A fully populated ``RefreshDiff`` instance.

    Raises:
        ValueError: If the schema version does not match ``REFRESH_DIFF_SCHEMA_VERSION``.
        json.JSONDecodeError: If ``json_str`` is not valid JSON.
    """
    from wanamaker.refresh.diff import ParameterMovement, RefreshDiff

    d: dict[str, Any] = _unwrap(json_str, REFRESH_DIFF_SCHEMA_VERSION, "refresh_diff.json")
    movements = [
        ParameterMovement(
            name=m["name"],
            previous_mean=float(m["previous_mean"]),
            current_mean=float(m["current_mean"]),
            previous_ci=(float(m["previous_ci"][0]), float(m["previous_ci"][1])),
            current_ci=(float(m["current_ci"][0]), float(m["current_ci"][1])),
        )
        for m in d.get("movements", [])
    ]
    return RefreshDiff(
        previous_run_id=d["previous_run_id"],
        current_run_id=d["current_run_id"],
        movements=movements,
    )
