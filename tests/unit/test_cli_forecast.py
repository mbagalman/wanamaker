"""Tests for the wanamaker forecast and compare-scenarios commands (issue #23).

Both commands orchestrate run-loading, posterior reconstruction, and the
forecast layer call. These tests stub ``PyMCEngine.load_posterior`` and
``PyMCEngine.posterior_predictive`` so the CLI orchestration can be exercised
end-to-end without invoking PyMC.

Coverage includes: missing run id, missing posterior.nc / summary.json,
forecast happy path with HDI table written to disk, compare-scenarios
happy path with ranked table written to disk, ``--seed`` override, and
extrapolation / spend-invariant warnings appearing in the output.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from wanamaker.artifacts import serialize_summary
from wanamaker.engine.base import Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    PosteriorSummary,
    PredictiveSummary,
)

runner = CliRunner()

CSV_BODY = textwrap.dedent(
    """\
    week,revenue,search,tv
    2024-01-01,100,10,50
    2024-01-08,110,12,55
    2024-01-15,120,15,60
    2024-01-22,118,14,58
    """
)


def _write_data_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    path = tmp_path / name
    path.write_text(CSV_BODY)
    return path


def _write_plan_csv(
    tmp_path: Path,
    name: str = "plan.csv",
    *,
    search: list[float] | None = None,
    tv: list[float] | None = None,
    periods: list[str] | None = None,
) -> Path:
    search = search or [20.0, 25.0]
    tv = tv or [60.0, 65.0]
    periods = periods or ["2024-02-05", "2024-02-12"]
    rows = "\n".join(
        f"{p},{s},{t}" for p, s, t in zip(periods, search, tv, strict=True)
    )
    path = tmp_path / name
    path.write_text(f"period,search,tv\n{rows}\n")
    return path


def _write_run(
    tmp_path: Path,
    *,
    csv_path: Path,
    artifact_dir: Path,
    run_id: str = "00000000_20260101T000000Z",
    summary: PosteriorSummary | None = None,
    posterior_present: bool = True,
) -> Path:
    """Create a minimal but valid run directory.

    The CLI doesn't open posterior.nc itself — that is the engine's job,
    and the engine is stubbed in these tests — but the file must exist so
    the CLI does not error before reaching the engine.
    """
    run_dir = artifact_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(
        textwrap.dedent(
            f"""\
            data:
              csv_path: {csv_path.as_posix()}
              date_column: week
              target_column: revenue
              spend_columns: [search, tv]
            channels:
              - {{name: search, category: paid_search}}
              - {{name: tv, category: linear_tv}}
            run:
              seed: 7
              runtime_mode: quick
            """
        )
    )

    if summary is None:
        summary = PosteriorSummary(
            parameters=[],
            channel_contributions=[
                ChannelContributionSummary(
                    channel="search",
                    mean_contribution=300.0, hdi_low=200.0, hdi_high=400.0,
                    observed_spend_min=10.0, observed_spend_max=15.0,
                ),
                ChannelContributionSummary(
                    channel="tv",
                    mean_contribution=600.0, hdi_low=400.0, hdi_high=800.0,
                    observed_spend_min=50.0, observed_spend_max=60.0,
                ),
            ],
            convergence=ConvergenceSummary(
                max_r_hat=1.01, min_ess_bulk=400.0,
                n_divergences=0, n_chains=4, n_draws=1000,
            ),
        )
    (run_dir / "summary.json").write_text(serialize_summary(summary))
    if posterior_present:
        (run_dir / "posterior.nc").write_bytes(b"fake")
    return run_dir


@pytest.fixture()
def stub_engine(monkeypatch: pytest.MonkeyPatch):
    """Replace PyMCEngine.load_posterior + posterior_predictive with stubs.

    The stub returns a deterministic ``PredictiveSummary`` whose mean equals
    ``2 * search + 0.5 * tv`` per row, matching the test's analytic
    expectations.
    """
    captured: dict[str, Any] = {}

    def _load_posterior(self: Any, run_dir: Any, model_spec: Any, data: Any) -> Any:
        captured["model_spec"] = model_spec
        captured["data_rows"] = len(data)
        return Posterior(raw=object())

    def _posterior_predictive(
        self: Any, posterior: Any, new_data: Any, seed: int  # noqa: ARG001
    ) -> PredictiveSummary:
        captured.setdefault("predictive_calls", []).append(seed)
        mean = (
            new_data["search"].astype(float) * 2.0
            + new_data["tv"].astype(float) * 0.5
        )
        return PredictiveSummary(
            periods=new_data["period"].astype(str).tolist(),
            mean=mean.tolist(),
            hdi_low=(mean * 0.8).tolist(),
            hdi_high=(mean * 1.2).tolist(),
        )

    monkeypatch.setattr(
        "wanamaker.engine.pymc.PyMCEngine.load_posterior", _load_posterior
    )
    monkeypatch.setattr(
        "wanamaker.engine.pymc.PyMCEngine.posterior_predictive", _posterior_predictive
    )
    return captured


# ---------------------------------------------------------------------------
# Error paths (no engine setup needed)
# ---------------------------------------------------------------------------


class TestForecastErrors:
    def test_missing_run_id_errors_cleanly(self, tmp_path: Path) -> None:
        plan = _write_plan_csv(tmp_path)
        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "forecast",
                "--run-id", "does_not_exist",
                "--plan", str(plan),
                "--artifact-dir", str(tmp_path / ".wanamaker"),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_missing_posterior_nc_errors_cleanly(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir, posterior_present=False)
        plan = _write_plan_csv(tmp_path)

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "forecast",
                "--run-id", "00000000_20260101T000000Z",
                "--plan", str(plan),
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "posterior.nc" in result.output.lower()


# ---------------------------------------------------------------------------
# Forecast happy path
# ---------------------------------------------------------------------------


class TestForecastHappyPath:
    def test_writes_markdown_report_and_predictive_table(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        run_dir = _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        plan = _write_plan_csv(
            tmp_path, name="aggressive.csv",
            search=[20.0, 25.0], tv=[60.0, 65.0],
        )

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "forecast",
                "--run-id", "00000000_20260101T000000Z",
                "--plan", str(plan),
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        # Predictive draws appear in stdout.
        assert "Posterior predictive forecast" in result.output
        assert "2024-02-05" in result.output
        # 2*20 + 0.5*60 = 70.0 → expect 70.00 to appear
        assert "70.00" in result.output

        # Markdown report saved next to the run.
        report = run_dir / "forecast_aggressive.md"
        assert report.exists()
        body = report.read_text()
        assert "# Forecast: aggressive.csv" in body
        assert "## Predictive draws" in body
        assert "2024-02-05" in body
        assert "70.00" in body
        # Total row.
        # 2024-02-05: 70.0; 2024-02-12: 2*25+0.5*65 = 82.5 → total 152.50.
        assert "**152.50**" in body

    def test_seed_override_propagates_to_engine(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        plan = _write_plan_csv(tmp_path)

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "forecast",
                "--run-id", "00000000_20260101T000000Z",
                "--plan", str(plan),
                "--artifact-dir", str(artifact_dir),
                "--seed", "1234",
            ],
        )
        assert result.exit_code == 0, result.output
        assert 1234 in stub_engine["predictive_calls"]

    def test_extrapolation_warning_printed_and_in_report(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        # Default summary's search.observed_spend_max=15.0; planning 30 triggers
        # an above_historical_max flag.
        run_dir = _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        plan = _write_plan_csv(
            tmp_path, name="overshoot.csv",
            search=[30.0], tv=[55.0], periods=["2024-02-05"],
        )

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "forecast",
                "--run-id", "00000000_20260101T000000Z",
                "--plan", str(plan),
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "extrapolation" in result.output.lower()
        body = (run_dir / "forecast_overshoot.md").read_text()
        assert "## Extrapolation warnings" in body
        assert "above historical max" in body

    def test_spend_invariant_channel_noted(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        invariant_summary = PosteriorSummary(
            parameters=[],
            channel_contributions=[
                ChannelContributionSummary(
                    channel="search",
                    mean_contribution=300.0, hdi_low=200.0, hdi_high=400.0,
                    observed_spend_min=10.0, observed_spend_max=15.0,
                ),
                ChannelContributionSummary(
                    channel="tv",
                    mean_contribution=600.0, hdi_low=400.0, hdi_high=800.0,
                    observed_spend_min=55.0, observed_spend_max=55.0,
                    spend_invariant=True,
                ),
            ],
        )
        run_dir = _write_run(
            tmp_path, csv_path=csv, artifact_dir=artifact_dir,
            summary=invariant_summary,
        )
        plan = _write_plan_csv(tmp_path)

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "forecast",
                "--run-id", "00000000_20260101T000000Z",
                "--plan", str(plan),
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "spend-invariant" in result.output.lower()
        assert "tv" in result.output
        body = (run_dir / "forecast_plan.md").read_text()
        assert "## Spend-invariant channels" in body


# ---------------------------------------------------------------------------
# Compare-scenarios happy path
# ---------------------------------------------------------------------------


class TestCompareScenariosHappyPath:
    def test_ranks_plans_and_writes_markdown_report(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        run_dir = _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        plan_a = _write_plan_csv(
            tmp_path, name="conservative.csv",
            search=[10.0], tv=[55.0], periods=["2024-02-05"],
        )
        plan_b = _write_plan_csv(
            tmp_path, name="aggressive.csv",
            search=[15.0], tv=[55.0], periods=["2024-02-05"],
        )

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "compare-scenarios",
                "--run-id", "00000000_20260101T000000Z",
                "--plans", str(plan_a),
                "--plans", str(plan_b),
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        # Aggressive (higher search) ranks first.
        assert result.output.index("aggressive") < result.output.index("conservative")

        report = run_dir / "scenario_comparison.md"
        assert report.exists()
        body = report.read_text()
        assert "# Scenario comparison" in body
        assert "## Ranking" in body
        assert "aggressive" in body
        assert "conservative" in body
        # Higher-mean plan listed before lower-mean plan in the ranking section.
        assert body.index("aggressive") < body.index("conservative")
        # Total spend section lists both channels.
        assert "## Total spend by channel" in body
        assert "`search`" in body and "`tv`" in body

    def test_caveats_section_lists_extrapolation_and_invariant_channels(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        invariant_summary = PosteriorSummary(
            parameters=[],
            channel_contributions=[
                ChannelContributionSummary(
                    channel="search",
                    mean_contribution=300.0, hdi_low=200.0, hdi_high=400.0,
                    observed_spend_min=10.0, observed_spend_max=15.0,
                ),
                ChannelContributionSummary(
                    channel="tv",
                    mean_contribution=600.0, hdi_low=400.0, hdi_high=800.0,
                    observed_spend_min=55.0, observed_spend_max=55.0,
                    spend_invariant=True,
                ),
            ],
        )
        run_dir = _write_run(
            tmp_path, csv_path=csv, artifact_dir=artifact_dir,
            summary=invariant_summary,
        )
        # Plan A has spend within range; Plan B exceeds search max.
        plan_a = _write_plan_csv(
            tmp_path, name="ok.csv",
            search=[12.0], tv=[55.0], periods=["2024-02-05"],
        )
        plan_b = _write_plan_csv(
            tmp_path, name="overshoot.csv",
            search=[40.0], tv=[55.0], periods=["2024-02-05"],
        )

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "compare-scenarios",
                "--run-id", "00000000_20260101T000000Z",
                "--plans", str(plan_a),
                "--plans", str(plan_b),
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        # Stdout flags extrapolation for the overshoot plan.
        assert "extrapolation" in result.output.lower()
        # Stdout flags spend-invariant for both plans.
        assert "spend-invariant" in result.output.lower()
        body = (run_dir / "scenario_comparison.md").read_text()
        assert "## Caveats" in body
        assert "overshoot" in body
        assert "above historical max" in body

    def test_output_override_writes_to_custom_path(
        self, tmp_path: Path, stub_engine
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        plan = _write_plan_csv(tmp_path)
        custom_output = tmp_path / "custom_report.md"

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "compare-scenarios",
                "--run-id", "00000000_20260101T000000Z",
                "--plans", str(plan),
                "--artifact-dir", str(artifact_dir),
                "--output", str(custom_output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert custom_output.exists()
        assert "# Scenario comparison" in custom_output.read_text()
