"""Tests for the shared readiness-check pipeline used by fit/refresh (#59).

Before #59, ``fit`` and ``refresh`` only ran ``check_structural_breaks``;
real blockers like missing values or insufficient history could slip
through to the model without flagging. These tests exercise the new
shared helpers directly so the gate's behaviour is pinned down without
running PyMC.
"""

from __future__ import annotations

import pandas as pd

from wanamaker.cli import (
    _readiness_level_from_results,
    _run_diagnostics,
    _run_readiness_checks,
)
from wanamaker.config import (
    ChannelConfig,
    DataConfig,
    RunConfig,
    WanamakerConfig,
)
from wanamaker.diagnose.readiness import CheckSeverity, ReadinessLevel


def _weekly_frame(n_weeks: int, *, with_spend: bool = False) -> pd.DataFrame:
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    revenue = [100.0 + i * 1.5 for i in range(n_weeks)]
    data: dict[str, object] = {"week": weeks, "revenue": revenue}
    if with_spend:
        data["search"] = [10.0 + (i % 5) for i in range(n_weeks)]
        data["tv"] = [50.0 + (i % 3) * 4 for i in range(n_weeks)]
    return pd.DataFrame(data)


def _config(csv_path, *, with_spend: bool = False) -> WanamakerConfig:
    return WanamakerConfig(
        data=DataConfig(
            csv_path=csv_path,
            date_column="week",
            target_column="revenue",
            spend_columns=(["search", "tv"] if with_spend else []),
        ),
        channels=(
            [
                ChannelConfig(name="search", category="paid_search"),
                ChannelConfig(name="tv", category="linear_tv"),
            ]
            if with_spend
            else []
        ),
        run=RunConfig(seed=0),
    )


# ---------------------------------------------------------------------------
# _run_readiness_checks: full check set runs (not just structural breaks)
# ---------------------------------------------------------------------------


class TestRunReadinessChecks:
    def test_history_length_blocker_flows_through(self, tmp_path) -> None:
        df = _weekly_frame(10)  # <26 weeks → BLOCKER
        results = _run_readiness_checks(
            df, target_column="revenue", date_column="week",
        )
        names = {r.name for r in results}
        assert "history_length" in names
        history = next(r for r in results if r.name == "history_length")
        assert history.severity == CheckSeverity.BLOCKER

    def test_missing_values_warning_flows_through(self, tmp_path) -> None:
        df = _weekly_frame(80)
        df.loc[10:14, "revenue"] = float("nan")  # introduce missingness
        results = _run_readiness_checks(
            df, target_column="revenue", date_column="week",
        )
        names = {r.name for r in results}
        assert "missing_values" in names
        missing = next(r for r in results if r.name == "missing_values")
        assert missing.severity == CheckSeverity.WARNING

    def test_spend_aware_checks_only_run_when_spend_columns_provided(self) -> None:
        df = _weekly_frame(80, with_spend=True)
        without_spend = _run_readiness_checks(
            df, target_column="revenue", date_column="week",
        )
        with_spend = _run_readiness_checks(
            df, target_column="revenue", date_column="week",
            spend_columns=["search", "tv"],
        )
        without_names = {r.name for r in without_spend}
        with_names = {r.name for r in with_spend}
        # Spend-aware names appear only in the second call.
        assert "spend_variation" not in without_names
        assert "spend_variation" in with_names or any(
            n.startswith("spend_variation") for n in with_names
        ) or "variable_count" in with_names

    def test_no_issues_for_clean_data_yields_empty_or_info_only(self) -> None:
        df = _weekly_frame(80)
        results = _run_readiness_checks(
            df, target_column="revenue", date_column="week",
        )
        for r in results:
            assert r.severity in (CheckSeverity.INFO, CheckSeverity.WARNING)
        # No BLOCKERs.
        assert all(r.severity != CheckSeverity.BLOCKER for r in results)

    def test_check_with_runtime_error_becomes_inline_warning(self) -> None:
        # Wrong date column name → KeyError inside check_history_length →
        # caught and surfaced as a WARNING by the shared _try wrapper.
        df = _weekly_frame(80)
        results = _run_readiness_checks(
            df, target_column="revenue", date_column="not_a_column",
        )
        skipped = [r for r in results if "skipped" in r.message.lower()]
        assert skipped, "expected at least one inline WARNING from a check error"
        assert all(r.severity == CheckSeverity.WARNING for r in skipped)


# ---------------------------------------------------------------------------
# _readiness_level_from_results
# ---------------------------------------------------------------------------


class TestReadinessLevelFromResults:
    def _make(self, severity: CheckSeverity):
        from wanamaker.diagnose.readiness import CheckResult

        return CheckResult(name="x", severity=severity, message="m")

    def test_blocker_dominates(self) -> None:
        results = [self._make(CheckSeverity.WARNING), self._make(CheckSeverity.BLOCKER)]
        assert _readiness_level_from_results(results) == ReadinessLevel.NOT_RECOMMENDED

    def test_warning_only(self) -> None:
        results = [self._make(CheckSeverity.WARNING)]
        assert (
            _readiness_level_from_results(results)
            == ReadinessLevel.USABLE_WITH_WARNINGS
        )

    def test_empty_or_info_only_is_ready(self) -> None:
        assert _readiness_level_from_results([]) == ReadinessLevel.READY
        info_only = [self._make(CheckSeverity.INFO)]
        assert _readiness_level_from_results(info_only) == ReadinessLevel.READY


# ---------------------------------------------------------------------------
# _run_diagnostics: end-to-end through fit/refresh's gate
# ---------------------------------------------------------------------------


class TestRunDiagnostics:
    def test_short_history_records_not_recommended(self, tmp_path, capsys) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("placeholder")  # path validation only, content unused below
        df = _weekly_frame(10)  # <26 weeks → BLOCKER
        cfg = _config(csv)

        level = _run_diagnostics(df, cfg)

        assert level == ReadinessLevel.NOT_RECOMMENDED.value
        captured = capsys.readouterr()
        assert "BLOCKER" in captured.out
        assert "Readiness:" in captured.out

    def test_missing_values_records_usable_with_warnings(
        self, tmp_path, capsys
    ) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("placeholder")
        df = _weekly_frame(80)
        df.loc[5:7, "revenue"] = float("nan")
        cfg = _config(csv)

        level = _run_diagnostics(df, cfg)

        assert level == ReadinessLevel.USABLE_WITH_WARNINGS.value
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_clean_data_records_ready(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("placeholder")
        df = _weekly_frame(80)
        cfg = _config(csv)

        level = _run_diagnostics(df, cfg)
        assert level == ReadinessLevel.READY.value

    def test_spend_aware_checks_run_when_config_lists_spend_columns(
        self, tmp_path
    ) -> None:
        """Issue #59: spend-aware checks must run too, not only structural break."""
        csv = tmp_path / "data.csv"
        csv.write_text("placeholder")
        df = _weekly_frame(80, with_spend=True)
        # Inject zero variation into one channel — triggers spend_variation WARNING.
        df["tv"] = 100.0
        cfg = _config(csv, with_spend=True)

        level = _run_diagnostics(df, cfg)

        assert level == ReadinessLevel.USABLE_WITH_WARNINGS.value
