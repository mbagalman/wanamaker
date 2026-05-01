"""Tests for the ``wanamaker recommend-ramp`` CLI command (issue #66).

The CLI orchestration mirrors ``forecast`` and ``compare-scenarios``:
it reconstructs the run, binds a saved posterior to the engine adapter,
calls the engine-neutral ramp core, and writes a deterministic Markdown
report. Tests stub ``PyMCEngine.load_posterior`` and
``PyMCEngine.posterior_predictive`` so no PyMC sampling is invoked.
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

CSV_BODY = textwrap.dedent(
    """\
    week,revenue,search,tv
    2024-01-01,100,10,50
    2024-01-08,110,12,55
    2024-01-15,120,15,60
    2024-01-22,118,10,58
    """
)


def _write_data_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(CSV_BODY)
    return path


def _write_plan_csv(
    tmp_path: Path,
    name: str,
    *,
    search: float,
    tv: float = 100.0,
) -> Path:
    path = tmp_path / name
    path.write_text(f"period,search,tv\n2024-02-05,{search},{tv}\n")
    return path


def _summary(*, spend_invariant_tv: bool = False) -> PosteriorSummary:
    return PosteriorSummary(
        parameters=[],
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=500.0,
                hdi_low=100.0,
                hdi_high=900.0,
                roi_mean=1.0,
                roi_hdi_low=-1.0,
                roi_hdi_high=3.0,
                observed_spend_min=10.0,
                observed_spend_max=50.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=200.0,
                hdi_low=150.0,
                hdi_high=250.0,
                roi_mean=0.5,
                roi_hdi_low=0.4,
                roi_hdi_high=0.6,
                observed_spend_min=80.0,
                observed_spend_max=120.0,
                spend_invariant=spend_invariant_tv,
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
    run_id: str = "00000000_20260101T000000Z",
) -> Path:
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
    (run_dir / "summary.json").write_text(serialize_summary(summary or _summary()))
    (run_dir / "posterior.nc").write_bytes(b"fake")
    return run_dir


def _install_stub_engine(monkeypatch: pytest.MonkeyPatch, mode: str) -> dict[str, Any]:
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
        search = df["search"].astype(float).to_numpy()
        draws = _draws_for_mode(mode, search, n_periods=len(df))
        return PredictiveSummary(
            periods=df["period"].astype(str).tolist(),
            mean=draws.mean(axis=0).tolist(),
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


def _draws_for_mode(mode: str, search: np.ndarray, *, n_periods: int) -> np.ndarray:
    draws = np.full((100, n_periods), 1_000.0)
    first_search = float(search[0])

    if mode == "proceed":
        fraction = max(0.0, min(1.0, (first_search - 10.0) / 10.0))
        if fraction > 0.50:
            delta = np.concatenate([
                np.full(70, 100.0 * fraction),
                np.full(30, -10.0 * fraction),
            ])
        else:
            delta = np.full(100, 100.0 * fraction)
    elif mode == "stage":
        fraction = max(0.0, min(1.0, (first_search - 20.0) / 60.0))
        delta = np.full(100, 100.0 * fraction)
    elif mode == "test_first":
        fraction = max(0.0, min(1.0, (first_search - 50.0) / 250.0))
        delta = np.full(100, 100.0 * fraction)
    elif mode == "negative":
        fraction = max(0.0, min(1.0, (first_search - 10.0) / 10.0))
        delta = np.full(100, -100.0 * fraction)
    else:
        delta = np.zeros(100)

    return draws + delta[:, None]


def _invoke_recommend_ramp(
    tmp_path: Path,
    *,
    mode: str,
    baseline_search: float,
    target_search: float,
    baseline_tv: float = 100.0,
    target_tv: float = 100.0,
    summary: PosteriorSummary | None = None,
    output: Path | None = None,
    seed: int | None = None,
) -> tuple[Any, Path, Path]:
    artifact_dir = tmp_path / ".wanamaker"
    csv = _write_data_csv(tmp_path)
    run_dir = _write_run(
        tmp_path,
        csv_path=csv,
        artifact_dir=artifact_dir,
        summary=summary,
    )
    baseline = _write_plan_csv(
        tmp_path, "base.csv", search=baseline_search, tv=baseline_tv,
    )
    target = _write_plan_csv(
        tmp_path, "alt.csv", search=target_search, tv=target_tv,
    )

    from wanamaker.cli import app

    args = [
        "recommend-ramp",
        "--run-id",
        "00000000_20260101T000000Z",
        "--baseline",
        str(baseline),
        "--target",
        str(target),
        "--artifact-dir",
        str(artifact_dir),
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if output is not None:
        args.extend(["--output", str(output)])

    result = runner.invoke(app, args)
    return result, run_dir, target


@pytest.mark.parametrize(
    ("mode", "baseline_search", "target_search", "expected_status"),
    [
        ("proceed", 10.0, 20.0, "proceed"),
        ("stage", 20.0, 80.0, "stage"),
        ("test_first", 50.0, 300.0, "test_first"),
        ("negative", 10.0, 20.0, "do_not_recommend"),
    ],
)
def test_each_ramp_status_is_reachable_through_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    baseline_search: float,
    target_search: float,
    expected_status: str,
) -> None:
    _install_stub_engine(monkeypatch, mode)
    result, run_dir, _ = _invoke_recommend_ramp(
        tmp_path,
        mode=mode,
        baseline_search=baseline_search,
        target_search=target_search,
    )
    assert result.exit_code == 0, result.output
    assert f"Verdict: {expected_status}" in result.output

    report = run_dir / "ramp_base_to_alt.md"
    assert report.exists()
    body = report.read_text()
    assert f"(`{expected_status}`)" in body
    assert "## Decision ladder" in body
    # The analyst-detail "Sizing detail" table preserves the failed_gates
    # column; verify it lands when there is at least one fail.
    if expected_status != "do_not_recommend":
        assert "Failed gates" in body


def test_report_contains_risk_table_failed_gates_and_advisor_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_engine(monkeypatch, "test_first")
    result, run_dir, _ = _invoke_recommend_ramp(
        tmp_path,
        mode="test_first",
        baseline_search=50.0,
        target_search=300.0,
    )
    assert result.exit_code == 0, result.output

    body = (run_dir / "ramp_base_to_alt.md").read_text()
    # Decision ladder is the lead executive-facing table.
    assert (
        "| Ramp | Expected lift | Downside risk | Historical support | "
        "Trust Card gate | Verdict |"
    ) in body
    # Analyst detail table is preserved at the bottom.
    assert "## Sizing detail (for analysts)" in body
    assert "extrapolation" in body
    assert "A test on `search` would most reduce the binding gate." in body


def test_spend_invariant_target_reallocation_blocks_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_engine(monkeypatch, "stage")
    result, run_dir, _ = _invoke_recommend_ramp(
        tmp_path,
        mode="stage",
        baseline_search=20.0,
        target_search=20.0,
        baseline_tv=100.0,
        target_tv=120.0,
        summary=_summary(spend_invariant_tv=True),
    )
    assert result.exit_code == 0, result.output
    assert "do_not_recommend" in result.output
    assert "spend_invariant_reallocation" in result.output

    body = (run_dir / "ramp_base_to_alt.md").read_text()
    assert "Do not recommend" in body
    assert "spend-invariant channel(s)" in body
    assert "Blocking reason: `spend_invariant_reallocation`" in body


def test_seed_and_output_overrides_are_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_stub_engine(monkeypatch, "stage")
    custom_output = tmp_path / "custom_ramp.md"
    result, _, _ = _invoke_recommend_ramp(
        tmp_path,
        mode="stage",
        baseline_search=20.0,
        target_search=80.0,
        output=custom_output,
        seed=1234,
    )
    assert result.exit_code == 0, result.output
    assert custom_output.exists()
    assert "# Risk-adjusted ramp recommendation" in custom_output.read_text()
    assert captured["seeds"]
    assert set(captured["seeds"]) == {1234}
