"""Unit tests for the PyMC engine shell.

These tests intentionally avoid importing PyMC itself. The backend module uses
lazy imports so the rest of Wanamaker remains importable before the selected
engine dependency is installed in a given environment.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from wanamaker.engine.pymc import PyMCEngine, RuntimeSettings, runtime_settings
from wanamaker.model.spec import ChannelSpec, ModelSpec


def test_importing_backend_does_not_import_pymc_stack() -> None:
    assert "pymc" not in sys.modules
    assert "arviz" not in sys.modules
    assert "pytensor.tensor" not in sys.modules


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("quick", RuntimeSettings(draws=250, tune=250, chains=2, target_accept=0.9)),
        ("standard", RuntimeSettings(draws=1000, tune=1000, chains=4, target_accept=0.9)),
        ("full", RuntimeSettings(draws=2000, tune=2000, chains=4, target_accept=0.95)),
    ],
)
def test_runtime_settings(mode: str, expected: RuntimeSettings) -> None:
    assert runtime_settings(mode) == expected


def test_runtime_settings_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown runtime_mode"):
        runtime_settings("demo")


def test_fit_requires_target_column_before_importing_pymc() -> None:
    engine = PyMCEngine()
    model_spec = ModelSpec(
        channels=[ChannelSpec(name="paid_search", category="paid_search")],
    )
    data = pd.DataFrame({"paid_search": [1.0, 2.0, 3.0], "revenue": [2.0, 4.0, 6.0]})

    with pytest.raises(ValueError, match="target_column"):
        engine.fit(model_spec, data, seed=1, runtime_mode="quick")

    assert "pymc" not in sys.modules


def test_fit_validates_required_columns_before_importing_pymc() -> None:
    engine = PyMCEngine()
    model_spec = SimpleNamespace(
        target_column="revenue",
        channels=[ChannelSpec(name="paid_search", category="paid_search")],
        control_columns=["promo"],
    )
    data = pd.DataFrame({"paid_search": [1.0, 2.0, 3.0], "revenue": [2.0, 4.0, 6.0]})

    with pytest.raises(ValueError, match="promo"):
        engine.fit(model_spec, data, seed=1, runtime_mode="quick")

    assert "pymc" not in sys.modules
