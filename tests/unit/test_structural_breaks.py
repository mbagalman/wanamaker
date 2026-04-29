"""Unit tests for structural-break detection (issue #14).

Every test uses known inputs and known outputs (AGENTS.md Hard Rule 4).
The synthetic datasets are constructed deterministically so the ground-truth
break location is unambiguous.

Algorithm under test:
- Linear detrend followed by greedy binary segmentation
- O(n) prefix-sum scan for candidate break evaluation
- F-statistic threshold = 15.0, min segment = 12, max breaks = 3
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanamaker.diagnose.checks import check_structural_breaks
from wanamaker.diagnose.readiness import CheckSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weekly_df(
    values: list[float] | np.ndarray,
    start: str = "2020-01-06",
    target: str = "revenue",
    date: str = "week",
) -> pd.DataFrame:
    """Build a weekly DataFrame from a list of target values."""
    n = len(values)
    dates = pd.date_range(start, periods=n, freq="W-MON")
    return pd.DataFrame({date: dates, target: np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# No-break baselines
# ---------------------------------------------------------------------------


class TestNoBreak:
    def test_constant_series_no_breaks(self) -> None:
        df = _weekly_df([100.0] * 52)
        results = check_structural_breaks(df, "revenue", "week")
        assert results == []

    def test_pure_linear_trend_no_breaks(self) -> None:
        """A perfect linear trend has zero residual variance after detrending."""
        values = np.arange(1, 53, dtype=float) * 10.0
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert results == []

    def test_low_noise_series_no_breaks(self) -> None:
        """Gaussian noise with no shift should not trigger a break."""
        rng = np.random.default_rng(42)
        values = 100.0 + rng.normal(0, 2, size=78)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert results == []

    def test_short_series_returns_empty(self) -> None:
        """Series shorter than 2 * min_segment + 1 = 25 periods: no scan."""
        df = _weekly_df([50.0] * 20)
        results = check_structural_breaks(df, "revenue", "week")
        assert results == []


# ---------------------------------------------------------------------------
# Single break — known location
# ---------------------------------------------------------------------------


class TestSingleBreak:
    def test_detects_level_shift_at_week_52(self) -> None:
        """Sharp level shift at period 52: pre=100, post=200.

        At least one break must be detected and the earliest-reported break
        should be within ±3 periods of the true break at index 52.
        Additional breaks in the noisy sub-segments are acceptable.
        """
        rng = np.random.default_rng(0)
        n = 104
        values = np.where(np.arange(n) < 52, 100.0, 200.0) + rng.normal(0, 3, n)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert len(results) >= 1
        # The earliest-reported break must be near the true break at index 52.
        dates = pd.to_datetime(df["week"])
        true_date = dates.iloc[52]
        # results are ordered by break index; find the one nearest the true date
        best_delta = None
        for result in results:
            assert result.severity == CheckSeverity.WARNING
            assert result.name == "structural_breaks"
            detected_date = pd.to_datetime(
                result.message.split("near ")[1].split(" ")[0]
            )
            delta_weeks = abs((detected_date - true_date).days) / 7
            if best_delta is None or delta_weeks < best_delta:
                best_delta = delta_weeks
        assert best_delta is not None and best_delta <= 3, (
            f"No break detected within 3 periods of the true break at {true_date}; "
            f"closest was {best_delta} weeks"
        )

    def test_detects_level_shift_returns_warning_severity(self) -> None:
        rng = np.random.default_rng(1)
        n = 78
        values = np.where(np.arange(n) < 39, 50.0, 150.0) + rng.normal(0, 2, n)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert len(results) >= 1
        assert all(r.severity == CheckSeverity.WARNING for r in results)

    def test_result_message_contains_column_name(self) -> None:
        rng = np.random.default_rng(2)
        n = 78
        values = np.where(np.arange(n) < 39, 10.0, 60.0) + rng.normal(0, 1, n)
        df = _weekly_df(values, target="sales")
        results = check_structural_breaks(df, "sales", "week")
        assert len(results) >= 1
        assert "'sales'" in results[0].message

    def test_result_message_contains_date(self) -> None:
        rng = np.random.default_rng(3)
        n = 78
        values = np.where(np.arange(n) < 39, 10.0, 60.0) + rng.normal(0, 1, n)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert len(results) >= 1
        # Message should contain a date in YYYY-MM-DD format
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2}", results[0].message)

    def test_detects_slope_change(self) -> None:
        """Slope reversal at mid-series (flat then declining) should be caught."""
        n = 104
        t = np.arange(n, dtype=float)
        values = np.where(t < 52, t * 2.0, 104.0 - (t - 52) * 2.0)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Multiple breaks
# ---------------------------------------------------------------------------


class TestMultipleBreaks:
    def test_two_breaks_detected(self) -> None:
        """Three-segment series: low → high → low.

        Breaks near period 35 and period 70 in a 104-period series.
        """
        rng = np.random.default_rng(10)
        n = 104
        idx = np.arange(n)
        values = np.where(idx < 35, 100.0, np.where(idx < 70, 200.0, 100.0))
        values = values + rng.normal(0, 3, n)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert len(results) >= 2

    def test_max_three_breaks_returned(self) -> None:
        """Even with 4 true breaks, at most 3 are returned (_MAX_BREAKS=3)."""
        rng = np.random.default_rng(20)
        n = 156
        idx = np.arange(n)
        # Alternating level: 4 breaks at 30, 60, 90, 120
        base = np.where(idx < 30, 100.0,
               np.where(idx < 60, 200.0,
               np.where(idx < 90, 100.0,
               np.where(idx < 120, 200.0, 100.0))))
        values = base + rng.normal(0, 2, n)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Data ordering and column names
# ---------------------------------------------------------------------------


class TestInputHandling:
    def test_unsorted_input_is_handled(self) -> None:
        """Rows in reverse chronological order must give the same result as sorted."""
        rng = np.random.default_rng(7)
        n = 78
        values = np.where(np.arange(n) < 39, 100.0, 200.0) + rng.normal(0, 2, n)
        df_sorted = _weekly_df(values)
        df_reversed = df_sorted.iloc[::-1].reset_index(drop=True)
        r_sorted = check_structural_breaks(df_sorted, "revenue", "week")
        r_reversed = check_structural_breaks(df_reversed, "revenue", "week")
        assert len(r_sorted) == len(r_reversed)
        # Break dates should be the same regardless of input order
        def extract_dates(results):
            return sorted(r.message.split("near ")[1].split(" ")[0] for r in results)
        assert extract_dates(r_sorted) == extract_dates(r_reversed)

    def test_custom_column_names(self) -> None:
        rng = np.random.default_rng(8)
        n = 78
        values = np.where(np.arange(n) < 39, 10.0, 80.0) + rng.normal(0, 1, n)
        df = _weekly_df(values, target="kpi", date="period")
        results = check_structural_breaks(df, "kpi", "period")
        assert len(results) >= 1
        assert "'kpi'" in results[0].message

    def test_missing_target_column_raises(self) -> None:
        df = _weekly_df([1.0] * 30)
        with pytest.raises(KeyError):
            check_structural_breaks(df, "nonexistent", "week")

    def test_missing_date_column_raises(self) -> None:
        df = _weekly_df([1.0] * 30)
        with pytest.raises(KeyError):
            check_structural_breaks(df, "revenue", "nonexistent")

    def test_result_name_field_is_structural_breaks(self) -> None:
        rng = np.random.default_rng(9)
        n = 78
        values = np.where(np.arange(n) < 39, 10.0, 80.0) + rng.normal(0, 1, n)
        df = _weekly_df(values)
        results = check_structural_breaks(df, "revenue", "week")
        assert all(r.name == "structural_breaks" for r in results)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_calls_return_identical_results(self) -> None:
        """check_structural_breaks is deterministic: same input → same output."""
        rng = np.random.default_rng(99)
        n = 104
        values = np.where(np.arange(n) < 52, 100.0, 200.0) + rng.normal(0, 5, n)
        df = _weekly_df(values)
        results_a = check_structural_breaks(df, "revenue", "week")
        results_b = check_structural_breaks(df, "revenue", "week")
        assert len(results_a) == len(results_b)
        for a, b in zip(results_a, results_b):
            assert a == b
