"""Reproducibility checks for the Bayesian engine.

These tests are intentionally gated while the Phase 0 engine and benchmark
fixtures are still settling. CI can enable them with ``WANAMAKER_RUN_ENGINE_TESTS=1``.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import pytest

from wanamaker.benchmarks.loaders import load_synthetic_ground_truth
from wanamaker.engine.pymc import PyMCEngine
from wanamaker.model.spec import ChannelSpec, ModelSpec

pytestmark = pytest.mark.engine


def test_pymc_engine_reproducible_on_synthetic_ground_truth() -> None:
    """Fit the same benchmark twice and compare posterior summaries exactly."""
    if os.getenv("WANAMAKER_RUN_ENGINE_TESTS") != "1":
        pytest.skip("Set WANAMAKER_RUN_ENGINE_TESTS=1 to run backend reproducibility checks.")

    pytest.importorskip("pymc")

    data, truth = load_synthetic_ground_truth()
    model_spec = _model_spec_from_truth(truth)
    seed = int(truth.get("seed", 20260429))

    engine = PyMCEngine()

    first = engine.fit(model_spec, data, seed=seed, runtime_mode="quick")
    second = engine.fit(model_spec, data, seed=seed, runtime_mode="quick")

    assert _canonicalize(first.summary) == _canonicalize(second.summary)


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


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }

    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    return value
