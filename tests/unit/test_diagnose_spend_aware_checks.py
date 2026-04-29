"""Unit tests for spend-aware diagnostic checks from issue #13."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanamaker.diagnose.checks import (
    check_collinearity,
    check_spend_variation,
    check_target_leakage,
    check_variable_count,
)
from wanamaker.diagnose.readiness import CheckSeverity


def test_spend_variation_flags_constant_channel() -> None:
    df = pd.DataFrame({"tv": [100.0] * 12, "search": np.linspace(10.0, 100.0, 12)})

    results = check_spend_variation(df, ["tv", "search"])

    assert len(results) == 1
    assert results[0].name == "spend_variation"
    assert results[0].severity == CheckSeverity.WARNING
    assert "'tv'" in results[0].message


def test_spend_variation_flags_low_cv_channel() -> None:
    df = pd.DataFrame({"display": [100.0, 101.0, 99.0, 100.5, 99.5]})

    results = check_spend_variation(df, ["display"], threshold=0.10)

    assert len(results) == 1
    assert "coefficient of variation" in results[0].message


def test_spend_variation_ignores_variable_channel() -> None:
    df = pd.DataFrame({"search": [10.0, 50.0, 90.0, 130.0, 170.0]})

    results = check_spend_variation(df, ["search"])

    assert results == []


def test_collinearity_flags_paid_channel_pair() -> None:
    df = pd.DataFrame(
        {
            "tv": [10.0, 20.0, 30.0, 40.0, 50.0],
            "ctv": [11.0, 21.0, 31.0, 41.0, 51.0],
            "search": [5.0, 30.0, 15.0, 45.0, 25.0],
        }
    )

    results = check_collinearity(df, ["tv", "ctv", "search"], [])

    assert len(results) == 1
    assert results[0].name == "collinearity"
    assert results[0].severity == CheckSeverity.WARNING
    assert "'tv'" in results[0].message
    assert "'ctv'" in results[0].message


def test_collinearity_flags_paid_control_pair() -> None:
    df = pd.DataFrame(
        {
            "social": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo_index": [50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )

    results = check_collinearity(df, ["social"], ["promo_index"])

    assert len(results) == 1
    assert "'promo_index'" in results[0].message


def test_collinearity_ignores_uncorrelated_pairs() -> None:
    df = pd.DataFrame(
        {
            "tv": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "search": [5.0, 5.0, 30.0, 30.0, 55.0, 55.0],
        }
    )

    results = check_collinearity(df, ["tv", "search"], [])

    assert results == []


def test_variable_count_ready_when_observations_cover_predictors() -> None:
    df = pd.DataFrame({"target": range(40)})

    result = check_variable_count(df, ["tv", "search"], ["promo"], observations_per_variable=10)

    assert result.name == "variable_count"
    assert result.severity == CheckSeverity.INFO


def test_variable_count_warns_when_predictor_count_is_too_high() -> None:
    df = pd.DataFrame({"target": range(20)})

    result = check_variable_count(df, ["tv", "search"], ["promo"], observations_per_variable=10)

    assert result.severity == CheckSeverity.WARNING
    assert "at least 30" in result.message


def test_target_leakage_flags_control_correlated_with_target() -> None:
    df = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
            "conversion_rate": [10.0, 20.0, 30.0, 40.0, 50.0],
            "holiday": [0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    results = check_target_leakage(df, "revenue", ["conversion_rate", "holiday"])

    assert len(results) == 1
    assert results[0].name == "target_leakage"
    assert results[0].severity == CheckSeverity.WARNING
    assert "'conversion_rate'" in results[0].message


def test_target_leakage_ignores_normal_controls() -> None:
    df = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "holiday": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    results = check_target_leakage(df, "revenue", ["holiday"])

    assert results == []


def test_invalid_threshold_raises() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})

    with pytest.raises(ValueError, match="threshold"):
        check_collinearity(df, ["x"], ["y"], threshold=1.5)
