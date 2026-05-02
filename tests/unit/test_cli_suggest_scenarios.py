"""Tests for the ``wanamaker suggest-scenarios`` CLI command (issue #85).

The CLI orchestration mirrors ``forecast`` and ``compare-scenarios``: it
reconstructs the run, binds a saved posterior to the engine adapter,
calls the engine-neutral generator core, writes candidate CSVs, and
saves a deterministic Markdown report. These tests stub
``PyMCEngine.load_posterior`` and ``PyMCEngine.posterior_predictive``
so no PyMC sampling is invoked.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

DATA_CSV = textwrap.dedent(
    """\
    week,revenue,search,tv,affiliate
    2024-01-01,100,10,50,5
    2024-01-08,110,12,55,6
    2024-01-15,120,15,60,7
    2024-01-22,118,10,58,5
    """
)


def _write_data_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(DATA_CSV)
    return path


def _write_baseline_csv(tmp_path: Path, *, name: str = "baseline.csv") -> Path:
    path = tmp_path / name
    path.write_text(
        "period,search,tv,affiliate\n"
        "2026-W01,25,25,25\n"
        "2026-W02,25,25,25\n"
        "2026-W03,25,25,25\n"
        "2026-W04,25,25,25\n"
    )
    return path


def _summary(
    *, spend_invariant: dict[str, bool] | None = None
) -> PosteriorSummary:
    spend_invariant = spend_invariant or {}
    return PosteriorSummary(
        parameters=[],
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=500.0, hdi_low=400.0, hdi_high=600.0,
                roi_mean=3.0, roi_hdi_low=2.0, roi_hdi_high=4.0,
                observed_spend_min=10.0, observed_spend_max=200.0,
                spend_invariant=spend_invariant.get("search", False),
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=200.0, hdi_low=150.0, hdi_high=250.0,
                roi_mean=1.0, roi_hdi_low=0.5, roi_hdi_high=1.5,
                observed_spend_min=10.0, observed_spend_max=200.0,
                spend_invariant=spend_invariant.get("tv", False),
            ),
            ChannelContributionSummary(
                channel="affiliate",
                mean_contribution=80.0, hdi_low=60.0, hdi_high=100.0,
                roi_mean=0.3, roi_hdi_low=0.1, roi_hdi_high=0.5,
                observed_spend_min=10.0, observed_spend_max=200.0,
                spend_invariant=spend_invariant.get("affiliate", False),
            ),
        ],
        convergence=ConvergenceSummary(
            max_r_hat=1.001,
            min_ess_bulk=800.0,
            n_divergences=0,
            n_chains=4,
            n_draws=500,
        ),
    )


def _write_run(
    tmp_path: Path,
    *,
    csv_path: Path,
    artifact_dir: Path,
    summary: PosteriorSummary | None = None,
    scenario_generation_yaml: str | None = None,
    run_id: str = "00000000_20260101T000000Z",
) -> Path:
    run_dir = artifact_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_text = textwrap.dedent(
        f"""\
        data:
          csv_path: {csv_path.as_posix()}
          date_column: week
          target_column: revenue
          spend_columns: [search, tv, affiliate]
        channels:
          - {{name: search, category: paid_search}}
          - {{name: tv, category: linear_tv}}
          - {{name: affiliate, category: affiliate}}
        run:
          seed: 7
          runtime_mode: quick
        """
    )
    if scenario_generation_yaml is not None:
        config_text += scenario_generation_yaml
    (run_dir / "config.yaml").write_text(config_text)
    (run_dir / "summary.json").write_text(serialize_summary(summary or _summary()))
    (run_dir / "posterior.nc").write_bytes(b"fake")
    return run_dir


def _install_stub_engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Linear stub: outcome = 3*search + 1*tv + 0.3*affiliate."""
    captured: dict[str, Any] = {"seeds": []}

    def _load_posterior(self: Any, run_dir: Any, model_spec: Any, data: Any) -> Any:
        captured["run_dir"] = run_dir
        captured["model_spec"] = model_spec
        captured["data_rows"] = len(data)
        return Posterior(raw=object())

    def _posterior_predictive(
        self: Any, posterior: Any, new_data: Any, seed: int  # noqa: ARG001
    ) -> PredictiveSummary:
        captured["seeds"].append(seed)
        df = pd.DataFrame(new_data)
        means = (
            df["search"].astype(float).to_numpy() * 3.0
            + df["tv"].astype(float).to_numpy() * 1.0
            + df["affiliate"].astype(float).to_numpy() * 0.3
        )
        # 50 draws with a small symmetric perturbation around the mean.
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.05, size=(50, len(df)))
        draws = means[None, :] * (1.0 + noise)
        return PredictiveSummary(
            periods=df["period"].astype(str).tolist(),
            mean=means.tolist(),
            hdi_low=np.quantile(draws, 0.025, axis=0).tolist(),
            hdi_high=np.quantile(draws, 0.975, axis=0).tolist(),
            draws=draws.tolist(),
        )

    monkeypatch.setattr(
        "wanamaker.engine.pymc.PyMCEngine.load_posterior",
        _load_posterior,
    )
    monkeypatch.setattr(
        "wanamaker.engine.pymc.PyMCEngine.posterior_predictive",
        _posterior_predictive,
    )
    return captured


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_args: tuple[str, ...] = (),
    summary: PosteriorSummary | None = None,
    scenario_generation_yaml: str | None = None,
) -> tuple[Any, Path, Path]:
    artifact_dir = tmp_path / ".wanamaker"
    csv = _write_data_csv(tmp_path)
    run_dir = _write_run(
        tmp_path,
        csv_path=csv,
        artifact_dir=artifact_dir,
        summary=summary,
        scenario_generation_yaml=scenario_generation_yaml,
    )
    baseline = _write_baseline_csv(tmp_path)
    _install_stub_engine(monkeypatch)

    from wanamaker.cli import app

    args = [
        "suggest-scenarios",
        "--run-id",
        "00000000_20260101T000000Z",
        "--baseline",
        str(baseline),
        "--artifact-dir",
        str(artifact_dir),
        *extra_args,
    ]
    result = runner.invoke(app, args)
    return result, run_dir, baseline


def test_suggest_scenarios_writes_candidate_csvs_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, run_dir, _ = _invoke(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    candidates_dir = run_dir / "candidates"
    assert candidates_dir.is_dir()
    csvs = sorted(candidates_dir.glob("*.csv"))
    assert csvs, "expected at least one candidate CSV"
    for csv in csvs:
        df = pd.read_csv(csv)
        assert set(df.columns) == {"period", "search", "tv", "affiliate"}
        assert len(df) == 4

    report = (run_dir / "scenario_suggestions.md").read_text(encoding="utf-8")
    assert "# Candidate scenarios" in report
    assert "## Constraints used" in report
    assert "## Decision ranking" in report


def test_suggest_scenarios_top_n_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, run_dir, _ = _invoke(
        tmp_path, monkeypatch, extra_args=("--top-n", "2"),
    )

    assert result.exit_code == 0, result.output
    csvs = sorted((run_dir / "candidates").glob("*.csv"))
    assert len(csvs) <= 2


def test_suggest_scenarios_reports_blocked_spend_invariant_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, run_dir, _ = _invoke(
        tmp_path, monkeypatch,
        summary=_summary(spend_invariant={"affiliate": True}),
    )

    assert result.exit_code == 0, result.output
    report = (run_dir / "scenario_suggestions.md").read_text(encoding="utf-8")
    assert "## Channels excluded from reallocation" in report
    assert "affiliate" in report
    assert "spend-invariant" in report


def test_suggest_scenarios_no_candidates_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Block all three channels → no donor/recipient pair → exit code 1."""
    result, run_dir, _ = _invoke(
        tmp_path, monkeypatch,
        summary=_summary(
            spend_invariant={"search": True, "tv": True, "affiliate": True}
        ),
    )

    assert result.exit_code == 1
    report = (run_dir / "scenario_suggestions.md").read_text(encoding="utf-8")
    assert "## No candidate plans produced" in report


def test_suggest_scenarios_reads_yaml_constraint_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locked channel from YAML survives into the generator's blocked set."""
    yaml_block = textwrap.dedent(
        """\
        scenario_generation:
          top_n: 3
          max_channel_change: 0.15
          max_total_moved_budget: 0.50
          locked_channels: [tv]
        """
    )
    result, run_dir, _ = _invoke(
        tmp_path, monkeypatch, scenario_generation_yaml=yaml_block,
    )

    assert result.exit_code == 0, result.output
    candidates_dir = run_dir / "candidates"
    csvs = sorted(candidates_dir.glob("*.csv"))
    assert csvs

    # Every candidate must leave tv unchanged at the per-channel total.
    expected_tv_total = 100.0
    for csv in csvs:
        df = pd.read_csv(csv)
        assert df["tv"].sum() == pytest.approx(expected_tv_total, rel=1e-6)

    report = (run_dir / "scenario_suggestions.md").read_text(encoding="utf-8")
    assert "Locked channels: `tv`" in report
