"""Unit tests for CSV-only diagnostic checks from issue #12."""

from __future__ import annotations

import pandas as pd

from wanamaker.diagnose.checks import (
    check_date_regularity,
    check_history_length,
    check_missing_values,
    check_target_stability,
)
from wanamaker.diagnose.readiness import CheckSeverity, ReadinessLevel


def _weekly_df(periods: int, start: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": pd.date_range(start, periods=periods, freq="W-MON"),
            "revenue": [100.0] * periods,
        }
    )


def _readiness_from_severities(severities: list[CheckSeverity]) -> ReadinessLevel:
    if CheckSeverity.BLOCKER in severities:
        return ReadinessLevel.NOT_RECOMMENDED
    if CheckSeverity.WARNING in severities:
        return ReadinessLevel.USABLE_WITH_WARNINGS
    return ReadinessLevel.READY


def test_history_length_ready_at_52_weeks() -> None:
    result = check_history_length(_weekly_df(52), "week")

    assert result.name == "history_length"
    assert result.severity == CheckSeverity.INFO


def test_history_length_warns_below_52_weeks() -> None:
    result = check_history_length(_weekly_df(40), "week")

    assert result.severity == CheckSeverity.WARNING
    assert "40 unique periods" in result.message


def test_history_length_blocks_below_26_weeks() -> None:
    result = check_history_length(_weekly_df(25), "week")

    assert result.severity == CheckSeverity.BLOCKER
    assert "25 unique periods" in result.message


def test_insufficient_history_implies_not_recommended_readiness() -> None:
    result = check_history_length(_weekly_df(25), "week")

    readiness = _readiness_from_severities([result.severity])

    assert readiness == ReadinessLevel.NOT_RECOMMENDED


def test_date_regularity_detects_duplicate_dates() -> None:
    df = _weekly_df(4)
    df.loc[3, "week"] = df.loc[2, "week"]

    results = check_date_regularity(df, "week")

    assert len(results) == 1
    assert results[0].name == "date_regularity"
    assert results[0].severity == CheckSeverity.BLOCKER
    assert "duplicate date" in results[0].message


def test_date_regularity_detects_gaps() -> None:
    df = pd.DataFrame(
        {
            "week": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-29"]),
            "revenue": [100.0, 110.0, 120.0],
        }
    )

    results = check_date_regularity(df, "week")

    assert len(results) == 1
    assert results[0].severity == CheckSeverity.WARNING
    assert "gap" in results[0].message


def test_date_regularity_returns_empty_for_regular_dates() -> None:
    results = check_date_regularity(_weekly_df(8), "week")

    assert results == []


def test_missing_values_reports_each_column_with_nan() -> None:
    df = pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=3, freq="W-MON"),
            "revenue": [100.0, None, 120.0],
            "tv": [1.0, 2.0, None],
        }
    )

    results = check_missing_values(df)

    assert [result.severity for result in results] == [
        CheckSeverity.WARNING,
        CheckSeverity.WARNING,
    ]
    assert [result.message for result in results] == [
        "Column 'revenue' has 1 missing value(s).",
        "Column 'tv' has 1 missing value(s).",
    ]


def test_missing_values_returns_empty_when_complete() -> None:
    results = check_missing_values(_weekly_df(4))

    assert results == []


def test_target_stability_detects_extreme_outlier() -> None:
    df = pd.DataFrame({"revenue": [100.0, 102.0, 98.0, 101.0, 99.0, 1000.0]})

    results = check_target_stability(df, "revenue")

    assert len(results) == 1
    assert results[0].name == "target_stability"
    assert results[0].severity == CheckSeverity.WARNING
    assert "1 extreme outlier" in results[0].message


def test_target_stability_ignores_stable_target() -> None:
    df = pd.DataFrame({"revenue": [100.0, 102.0, 98.0, 101.0, 99.0, 103.0]})

    results = check_target_stability(df, "revenue")

    assert results == []
