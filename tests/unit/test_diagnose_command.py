"""Unit tests for the ``wanamaker diagnose`` CLI command (issues #15, #13).

Tests cover:
- Config-independent path (--date-column + --target-column)
- Config-provided path (column names from YAML)
- Missing required flags exits with code 2
- READY → exit 0, USABLE_WITH_WARNINGS → exit 0, NOT_RECOMMENDED → exit 1
- Output contains expected severity labels
- Unknown column names produce a skipped-check WARNING
- Spend-aware checks run when --config supplies spend/control columns
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from wanamaker.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------


def _write_clean_csv(tmp: Path, periods: int = 80) -> Path:
    """Write a clean weekly CSV with no issues (well-named columns for inference)."""
    p = tmp / "data.csv"
    dates = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    df = pd.DataFrame({"week": dates.strftime("%Y-%m-%d"), "revenue": range(periods)})
    df.to_csv(p, index=False)
    return p


def _write_ambiguous_csv(tmp: Path, periods: int = 80) -> Path:
    """Write a CSV with non-inferrable column names (col_a, col_b)."""
    p = tmp / "ambiguous.csv"
    df = pd.DataFrame({"col_a": range(periods), "col_b": range(periods)})
    df.to_csv(p, index=False)
    return p


def _write_short_csv(tmp: Path, periods: int = 10) -> Path:
    """Write a very short CSV that will trigger BLOCKER history length."""
    p = tmp / "short.csv"
    dates = pd.date_range("2024-01-01", periods=periods, freq="W-MON")
    df = pd.DataFrame({"week": dates.strftime("%Y-%m-%d"), "revenue": [100.0] * periods})
    df.to_csv(p, index=False)
    return p


def _write_csv_with_warnings(tmp: Path) -> Path:
    """Write a 60-row CSV (warning range) with a missing value."""
    p = tmp / "warn.csv"
    dates = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    revenue = [float(i) for i in range(60)]
    revenue[5] = float("nan")
    df = pd.DataFrame({"week": dates.strftime("%Y-%m-%d"), "revenue": revenue})
    df.to_csv(p, index=False)
    return p


def _write_minimal_config(tmp: Path, csv_path: Path) -> Path:
    """Write a minimal wanamaker YAML config."""
    p = tmp / "config.yaml"
    p.write_text(
        f"data:\n"
        f"  csv_path: {csv_path}\n"
        f"  date_column: week\n"
        f"  target_column: revenue\n"
        f"  spend_columns: []\n"
        f"channels: []\n"
    )
    return p


def _write_config_with_spend(tmp: Path, csv_path: Path, spend_cols: list, control_cols: list = []) -> Path:
    """Write a config that includes spend and control columns."""
    p = tmp / "config_spend.yaml"
    spend_yaml = "[" + ", ".join(spend_cols) + "]"
    control_yaml = "[" + ", ".join(control_cols) + "]"
    channels_yaml = "\n".join(
        f"  - name: {c}\n    category: paid_search" for c in spend_cols
    )
    p.write_text(
        f"data:\n"
        f"  csv_path: {csv_path}\n"
        f"  date_column: week\n"
        f"  target_column: revenue\n"
        f"  spend_columns: {spend_yaml}\n"
        f"  control_columns: {control_yaml}\n"
        f"channels:\n{channels_yaml}\n"
    )
    return p


def _write_collinear_csv(tmp: Path, periods: int = 80) -> Path:
    """Write a CSV where two spend columns are nearly perfectly correlated."""
    p = tmp / "collinear.csv"
    dates = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    base = list(range(periods))
    df = pd.DataFrame({
        "week": dates.strftime("%Y-%m-%d"),
        "revenue": [float(i * 100) for i in range(periods)],
        "search": [float(i * 10) for i in base],
        "branded_search": [float(i * 10 + 1) for i in base],  # nearly identical
    })
    df.to_csv(p, index=False)
    return p


def _write_low_variation_csv(tmp: Path, periods: int = 80) -> Path:
    """Write a CSV with a constant spend column."""
    p = tmp / "low_var.csv"
    dates = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    df = pd.DataFrame({
        "week": dates.strftime("%Y-%m-%d"),
        "revenue": list(range(periods)),
        "tv": [5000.0] * periods,        # constant — low variation
        "search": [float(i * 100) for i in range(periods)],
    })
    df.to_csv(p, index=False)
    return p


def _write_leakage_csv(tmp: Path, periods: int = 80) -> Path:
    """Write a CSV where a control column is derived from the target."""
    p = tmp / "leakage.csv"
    dates = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    revenue = list(range(100, 100 + periods))
    df = pd.DataFrame({
        "week": dates.strftime("%Y-%m-%d"),
        "revenue": revenue,
        "search": [float(i * 10) for i in range(periods)],
        "revenue_lag": [float(v) * 0.99 for v in revenue],  # nearly = revenue → leakage
    })
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Config-independent path (--date-column + --target-column required)
# ---------------------------------------------------------------------------


class TestDiagnoseWithoutConfig:
    def test_exits_0_for_clean_data(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert result.exit_code == 0

    def test_ready_in_output_for_clean_data(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert "ready" in result.output.lower()

    def test_exits_1_for_blocker(self, tmp_path: Path) -> None:
        csv = _write_short_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert result.exit_code == 1

    def test_blocker_label_in_output(self, tmp_path: Path) -> None:
        csv = _write_short_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert "BLOCKER" in result.output
        assert "not_recommended" in result.output.lower()

    def test_exits_0_for_warnings_only(self, tmp_path: Path) -> None:
        csv = _write_csv_with_warnings(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert result.exit_code == 0

    def test_warning_label_in_output(self, tmp_path: Path) -> None:
        csv = _write_csv_with_warnings(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert "WARNING" in result.output
        assert "usable_with_warnings" in result.output.lower()

    def test_no_flags_infers_columns_from_well_named_csv(self, tmp_path: Path) -> None:
        """wanamaker diagnose data.csv with no flags should auto-detect columns."""
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(app, ["diagnose", str(csv)])
        assert result.exit_code == 0

    def test_inferred_column_names_echoed(self, tmp_path: Path) -> None:
        """The inferred column names should appear in output so users can verify."""
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(app, ["diagnose", str(csv)])
        assert "week" in result.output
        assert "revenue" in result.output

    def test_ambiguous_csv_exits_2(self, tmp_path: Path) -> None:
        """A CSV with no date-like or target-like columns should exit 2."""
        csv = _write_ambiguous_csv(tmp_path)
        result = runner.invoke(app, ["diagnose", str(csv)])
        assert result.exit_code == 2

    def test_ambiguous_csv_error_lists_available_columns(self, tmp_path: Path) -> None:
        """The error for an ambiguous CSV should show the column names."""
        csv = _write_ambiguous_csv(tmp_path)
        result = runner.invoke(app, ["diagnose", str(csv)])
        combined = result.output + (result.stderr or "")
        assert "col_a" in combined or "col_b" in combined

    def test_explicit_flags_override_inference(self, tmp_path: Path) -> None:
        """Explicit --date-column / --target-column should be used as-is."""
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert result.exit_code == 0

    def test_unknown_date_column_produces_skipped_warning(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "no_such_col", "--target-column", "revenue"],
        )
        # Should not crash — falls back to WARNING for skipped check
        assert result.exit_code in (0, 1)
        assert "skipped" in result.output.lower() or "WARNING" in result.output

    def test_row_count_echoed(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path, periods=80)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert "80" in result.output


# ---------------------------------------------------------------------------
# Config-provided path
# ---------------------------------------------------------------------------


class TestDiagnoseWithConfig:
    def test_exits_0_with_config_clean_data(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        cfg = _write_minimal_config(tmp_path, csv)
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert result.exit_code == 0

    def test_column_names_from_config(self, tmp_path: Path) -> None:
        """Config column names are used even without explicit CLI flags."""
        csv = _write_clean_csv(tmp_path)
        cfg = _write_minimal_config(tmp_path, csv)
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert result.exit_code == 0
        assert "ready" in result.output.lower()

    def test_blocker_with_config(self, tmp_path: Path) -> None:
        csv = _write_short_csv(tmp_path)
        cfg = _write_minimal_config(tmp_path, csv)
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert result.exit_code == 1
        assert "BLOCKER" in result.output


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestDiagnoseOutputFormat:
    def test_info_results_shown(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        # history_length check produces INFO for 80+ weeks
        assert "INFO" in result.output

    def test_no_issues_message_when_no_checks_return_anything(self, tmp_path: Path) -> None:
        """A series that passes every check should confirm no issues (or show INFOs)."""
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        # Either "No issues found." or some INFO results — either is acceptable
        assert "no issues" in result.output.lower() or "INFO" in result.output

    def test_readiness_line_always_present(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        assert "Readiness:" in result.output


# ---------------------------------------------------------------------------
# Spend-aware checks via --config (issue #13)
# ---------------------------------------------------------------------------


class TestDiagnoseSpendAwareViaConfig:
    def test_spend_variation_warning_for_constant_channel(self, tmp_path: Path) -> None:
        csv = _write_low_variation_csv(tmp_path)
        cfg = _write_config_with_spend(tmp_path, csv, ["tv", "search"])
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert "spend_variation" in result.output.lower() or "WARNING" in result.output
        assert "tv" in result.output

    def test_collinearity_warning_for_correlated_channels(self, tmp_path: Path) -> None:
        csv = _write_collinear_csv(tmp_path)
        cfg = _write_config_with_spend(tmp_path, csv, ["search", "branded_search"])
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert "WARNING" in result.output
        assert "collinearity" in result.output.lower() or "correlated" in result.output.lower()

    def test_target_leakage_warning(self, tmp_path: Path) -> None:
        csv = _write_leakage_csv(tmp_path)
        cfg = _write_config_with_spend(tmp_path, csv, ["search"], ["revenue_lag"])
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert "WARNING" in result.output
        assert "revenue_lag" in result.output

    def test_spend_aware_checks_not_run_without_config(self, tmp_path: Path) -> None:
        """Without a config, spend-aware checks must not run (no spend cols known)."""
        csv = _write_low_variation_csv(tmp_path)
        result = runner.invoke(
            app,
            ["diagnose", str(csv), "--date-column", "week", "--target-column", "revenue"],
        )
        # tv is constant but without config the spend_variation check shouldn't fire
        assert "spend_variation" not in result.output.lower()

    def test_variable_count_info_shown_with_config(self, tmp_path: Path) -> None:
        csv = _write_clean_csv(tmp_path)
        cfg = _write_config_with_spend(tmp_path, csv, ["search"])
        result = runner.invoke(app, ["diagnose", str(csv), "--config", str(cfg)])
        assert result.exit_code == 0
        assert "variable_count" in result.output.lower() or "INFO" in result.output
