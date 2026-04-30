"""Unit tests for the PyMC engine shell.

These tests intentionally avoid importing PyMC itself. The backend module uses
lazy imports so the rest of Wanamaker remains importable before the selected
engine dependency is installed in a given environment.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wanamaker.engine.base import Posterior
from wanamaker.engine.pymc import (
    PyMCEngine,
    PyMCRawPosterior,
    RuntimeSettings,
    _apply_control_centering,
    runtime_settings,
)
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


def test_modelspec_requires_target_column() -> None:
    """ModelSpec now requires target_column at construction time."""
    with pytest.raises(TypeError):
        ModelSpec(  # type: ignore[call-arg]
            channels=[ChannelSpec(name="paid_search", category="paid_search")],
            date_column="week",
        )


def test_fit_validates_target_column_before_importing_pymc() -> None:
    """Even a SimpleNamespace with no target_column raises before PyMC import."""
    engine = PyMCEngine()
    model_spec = SimpleNamespace(
        target_column="",
        date_column="week",
        channels=[ChannelSpec(name="paid_search", category="paid_search")],
        control_columns=[],
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


# ---------------------------------------------------------------------------
# Control centering helper (pure-numpy, no engine needed)
# ---------------------------------------------------------------------------


class TestApplyControlCentering:
    def test_zero_centered_unit_scaled_when_std_positive(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        out = _apply_control_centering(values, mean=2.5, std=1.0)
        assert out.tolist() == pytest.approx([-1.5, -0.5, 0.5, 1.5])

    def test_only_centered_when_std_is_zero(self) -> None:
        values = np.array([1.0, 1.0, 1.0])
        out = _apply_control_centering(values, mean=1.0, std=0.0)
        assert out.tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_does_not_mutate_input(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        original = values.copy()
        _apply_control_centering(values, mean=2.0, std=1.0)
        assert (values == original).all()


# ---------------------------------------------------------------------------
# posterior_predictive(new_data=...) validation — runs *before* PyMC import
# ---------------------------------------------------------------------------


def _stub_raw(*, channels: list[str], controls: list[str]) -> PyMCRawPosterior:
    """Construct a PyMCRawPosterior with no real PyMC objects.

    The validation path runs before any engine code, so a stub with the
    minimum surface needed by ``_validate_new_data`` is enough to exercise
    every error message.
    """
    model_spec = SimpleNamespace(
        target_column="revenue",
        channels=[ChannelSpec(name=name, category="paid_search") for name in channels],
        control_columns=list(controls),
    )
    return PyMCRawPosterior(
        idata=None,
        model=None,
        model_spec=model_spec,
        date_column=None,
        target_column="revenue",
    )


class TestPosteriorPredictiveNewDataValidation:
    def test_missing_channel_column_raises(self) -> None:
        raw = _stub_raw(channels=["paid_search", "tv"], controls=[])
        new_data = pd.DataFrame({"paid_search": [1.0, 2.0]})  # tv missing

        with pytest.raises(ValueError, match="missing columns"):
            PyMCEngine().posterior_predictive(Posterior(raw=raw), new_data, seed=0)

        assert "pymc" not in sys.modules

    def test_missing_control_column_raises(self) -> None:
        raw = _stub_raw(channels=["paid_search"], controls=["promo"])
        new_data = pd.DataFrame({"paid_search": [1.0, 2.0]})  # promo missing

        with pytest.raises(ValueError, match="missing columns.*promo"):
            PyMCEngine().posterior_predictive(Posterior(raw=raw), new_data, seed=0)

        assert "pymc" not in sys.modules

    def test_negative_spend_raises(self) -> None:
        raw = _stub_raw(channels=["paid_search"], controls=[])
        new_data = pd.DataFrame({"paid_search": [1.0, -2.0, 3.0]})

        with pytest.raises(ValueError, match="negative spend.*paid_search"):
            PyMCEngine().posterior_predictive(Posterior(raw=raw), new_data, seed=0)

        assert "pymc" not in sys.modules

    def test_nan_in_required_column_raises(self) -> None:
        raw = _stub_raw(channels=["paid_search"], controls=["promo"])
        new_data = pd.DataFrame(
            {"paid_search": [1.0, 2.0, 3.0], "promo": [0.0, float("nan"), 0.0]}
        )

        with pytest.raises(ValueError, match="missing values.*promo"):
            PyMCEngine().posterior_predictive(Posterior(raw=raw), new_data, seed=0)

        assert "pymc" not in sys.modules

    def test_empty_dataframe_raises(self) -> None:
        raw = _stub_raw(channels=["paid_search"], controls=[])
        new_data = pd.DataFrame({"paid_search": []})

        with pytest.raises(ValueError, match="empty"):
            PyMCEngine().posterior_predictive(Posterior(raw=raw), new_data, seed=0)

        assert "pymc" not in sys.modules

    def test_non_pymc_raw_posterior_raises_type_error(self) -> None:
        bogus = Posterior(raw=object())
        new_data = pd.DataFrame({"paid_search": [1.0, 2.0]})

        with pytest.raises(TypeError, match="PyMCRawPosterior"):
            PyMCEngine().posterior_predictive(bogus, new_data, seed=0)

        assert "pymc" not in sys.modules
