"""Reproducibility checks for the Bayesian engine (NFR-2).

Two fits on the same data with the same seed must produce numerically
identical posterior summaries — meaning every float field agrees to at
least RTOL relative tolerance (default 1e-6). This allows for
inconsequential floating-point differences across library versions or
platforms while still catching any non-determinism in the sampling path.

Gate: set WANAMAKER_RUN_ENGINE_TESTS=1 to enable. The gate exists
because a full fit on the synthetic benchmark takes several minutes; it
should not block fast unit-test runs.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import pytest

from wanamaker.benchmarks.loaders import load_synthetic_ground_truth
from wanamaker.engine.pymc import PyMCEngine
from wanamaker.model.spec import ChannelSpec, ModelSpec

pytestmark = pytest.mark.engine

# Relative tolerance for float comparisons. Two values a and b are
# considered equal when |a - b| / max(|a|, |b|, 1) <= RTOL.
RTOL = 1e-6


def test_pymc_engine_reproducible_on_synthetic_ground_truth() -> None:
    """Fit the same benchmark twice; all summary floats must agree to RTOL.

    NFR-2: same data + same seed → same analytical conclusions regardless
    of when the fit runs. Exact bit-for-bit identity is not required; the
    criterion is numerical closeness at RTOL=1e-6.
    """
    if os.getenv("WANAMAKER_RUN_ENGINE_TESTS") != "1":
        pytest.skip("Set WANAMAKER_RUN_ENGINE_TESTS=1 to run backend reproducibility checks.")

    pytest.importorskip("pymc")

    data, truth = load_synthetic_ground_truth()
    model_spec = _model_spec_from_truth(truth)
    seed = int(truth.get("seed", 20260429))

    engine = PyMCEngine()

    first = engine.fit(model_spec, data, seed=seed, runtime_mode="quick")
    second = engine.fit(model_spec, data, seed=seed, runtime_mode="quick")

    failures = _compare(first.summary, second.summary, path="summary")
    if failures:
        msg = "\n".join(f"  {path}: {a!r} vs {b!r}" for path, a, b in failures)
        pytest.fail(
            f"Reproducibility check failed — {len(failures)} field(s) differ beyond RTOL={RTOL}:\n{msg}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_spec_from_truth(truth: dict[str, Any]) -> ModelSpec:
    channel_categories = {
        channel["name"]: channel["category"] for channel in truth.get("channels", [])
    }
    channels = [
        ChannelSpec(name=name, category=channel_categories.get(name, "other"))
        for name in truth["spend_columns"]
    ]
    return ModelSpec(
        channels=channels,
        target_column=truth["target_column"],
        date_column=truth["date_column"],
        control_columns=list(truth.get("control_columns", [])),
        runtime_mode="quick",
    )


def _compare(a: Any, b: Any, path: str) -> list[tuple[str, Any, Any]]:
    """Recursively compare two values; return list of (path, a, b) mismatches."""
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        failures = []
        for f in dataclasses.fields(a):
            failures.extend(_compare(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}"))
        return failures

    if isinstance(a, dict) and isinstance(b, dict):
        failures = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                failures.append((f"{path}[{k!r}]", a.get(k), b.get(k)))
            else:
                failures.extend(_compare(a[k], b[k], f"{path}[{k!r}]"))
        return failures

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(f"{path}[len]", len(a), len(b))]
        failures = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            failures.extend(_compare(ai, bi, f"{path}[{i}]"))
        return failures

    if isinstance(a, float) and isinstance(b, float):
        # Both NaN → equal; one NaN → mismatch
        import math
        if math.isnan(a) and math.isnan(b):
            return []
        if math.isnan(a) or math.isnan(b):
            return [(path, a, b)]
        denom = max(abs(a), abs(b), 1.0)
        if abs(a - b) / denom > RTOL:
            return [(path, a, b)]
        return []

    if a != b:
        return [(path, a, b)]

    return []
