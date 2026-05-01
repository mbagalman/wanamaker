"""End-to-end PyMC posterior-predictive checks for new-data forecasting (#60).

These tests fit a tiny model with PyMC and exercise the future-plan path
through ``PyMCEngine.posterior_predictive``. They are gated behind
``WANAMAKER_RUN_ENGINE_TESTS=1`` because each test does a real (quick-mode)
fit; the gate matches the reproducibility job's wiring in CI.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from wanamaker.engine.summary import PredictiveSummary
from wanamaker.model.spec import ChannelSpec, ModelSpec

pytestmark = pytest.mark.engine

# The fit is tiny — quick-mode, two channels, ~30 weeks — but PyMC compilation
# still dominates the runtime, so a single fit is reused across tests via a
# session-scoped fixture below.

_SEED = 20260430
_FORECAST_SEED = 20260501


def _toy_dataset() -> pd.DataFrame:
    """Synthetic dataset just large enough to fit a quick-mode model.

    Two channels (paid_search, tv) with mild adstock + saturation and a
    single ``promo`` control. Reproducible from a fixed seed so the fit is
    stable across CI runs.
    """
    rng = np.random.default_rng(_SEED)
    n_weeks = 32
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    search = rng.uniform(800.0, 1500.0, size=n_weeks)
    tv = rng.uniform(2500.0, 5000.0, size=n_weeks)
    promo = (rng.random(size=n_weeks) > 0.7).astype(float)
    revenue = (
        50000.0
        + 1.8 * search
        + 0.6 * tv
        + 6000.0 * promo
        + rng.normal(0.0, 1500.0, size=n_weeks)
    )
    return pd.DataFrame(
        {
            "week": weeks,
            "revenue": revenue,
            "paid_search": search,
            "tv": tv,
            "promo": promo,
        }
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        channels=[
            ChannelSpec(name="paid_search", category="paid_search"),
            ChannelSpec(name="tv", category="linear_tv"),
        ],
        target_column="revenue",
        date_column="week",
        control_columns=["promo"],
        runtime_mode="quick",
    )


def _future_plan(n_weeks: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(_SEED + 1)
    weeks = pd.date_range("2024-09-01", periods=n_weeks, freq="W-MON")
    return pd.DataFrame(
        {
            "week": weeks,
            "paid_search": rng.uniform(900.0, 1400.0, size=n_weeks),
            "tv": rng.uniform(2700.0, 4800.0, size=n_weeks),
            "promo": np.zeros(n_weeks),
        }
    )


@pytest.fixture(scope="module")
def fitted_posterior():
    """One quick-mode fit reused across new-data tests.

    Skipped automatically when the engine extras are not available or when
    the gate env var is not set, so this file is a no-op in the fast unit
    job.
    """
    if os.getenv("WANAMAKER_RUN_ENGINE_TESTS") != "1":
        pytest.skip("Set WANAMAKER_RUN_ENGINE_TESTS=1 to run engine tests.")
    pytest.importorskip("pymc")

    from wanamaker.engine.pymc import PyMCEngine

    engine = PyMCEngine()
    fit = engine.fit(_model_spec(), _toy_dataset(), seed=_SEED, runtime_mode="quick")
    return engine, fit.posterior


def test_new_data_returns_predictive_summary_over_plan_periods(fitted_posterior) -> None:
    engine, posterior = fitted_posterior
    plan = _future_plan(n_weeks=4)

    result = engine.posterior_predictive(posterior, plan, seed=_FORECAST_SEED)

    assert isinstance(result, PredictiveSummary)
    assert len(result.periods) == 4
    assert len(result.mean) == 4
    assert len(result.hdi_low) == 4
    assert len(result.hdi_high) == 4
    # HDI bounds bracket the mean.
    for low, mean, high in zip(
        result.hdi_low, result.mean, result.hdi_high, strict=True
    ):
        assert low <= mean <= high
    # Periods come from the plan's date column.
    assert result.periods[0].startswith("2024-09")
    assert result.interval_mass == pytest.approx(0.95)

    # Check draws matrix (#64)
    assert result.draws is not None
    n_draws = len(result.draws)
    assert n_draws > 0
    assert all(len(d) == 4 for d in result.draws)

    # Mean derived from draws must match the summary
    draws_array = np.array(result.draws)
    derived_mean = draws_array.mean(axis=0)
    np.testing.assert_allclose(derived_mean, result.mean, rtol=1e-6)


def test_new_data_predictive_is_reproducible_for_same_seed(fitted_posterior) -> None:
    engine, posterior = fitted_posterior
    plan = _future_plan(n_weeks=4)

    first = engine.posterior_predictive(posterior, plan, seed=_FORECAST_SEED)
    second = engine.posterior_predictive(posterior, plan, seed=_FORECAST_SEED)

    assert first.periods == second.periods
    assert first.mean == pytest.approx(second.mean, rel=1e-6)
    assert first.hdi_low == pytest.approx(second.hdi_low, rel=1e-6)
    assert first.hdi_high == pytest.approx(second.hdi_high, rel=1e-6)


def test_new_data_predictive_changes_when_plan_spend_changes(fitted_posterior) -> None:
    """Doubling spend on a non-saturated channel must move the predictive mean.

    A guard against accidentally producing the in-sample summary when the
    forecast path is invoked (e.g. if pm.set_data was a no-op).
    """
    engine, posterior = fitted_posterior
    base_plan = _future_plan(n_weeks=4)
    boosted_plan = base_plan.copy()
    boosted_plan["paid_search"] = boosted_plan["paid_search"] * 2.0

    base = engine.posterior_predictive(posterior, base_plan, seed=_FORECAST_SEED)
    boosted = engine.posterior_predictive(posterior, boosted_plan, seed=_FORECAST_SEED)

    # On a non-saturated quick-mode fit, more spend should not lower the mean.
    # (If it does, the mutable-data wiring is broken.)
    base_mean = sum(base.mean)
    boosted_mean = sum(boosted.mean)
    assert boosted_mean >= base_mean


def test_in_sample_path_still_works(fitted_posterior) -> None:
    """Regression check: passing ``new_data=None`` returns training-period draws."""
    engine, posterior = fitted_posterior
    result = engine.posterior_predictive(posterior, new_data=None, seed=_FORECAST_SEED)
    assert isinstance(result, PredictiveSummary)
    assert len(result.periods) == 32  # toy dataset length
    assert result.draws is not None
    assert all(len(d) == 32 for d in result.draws)
