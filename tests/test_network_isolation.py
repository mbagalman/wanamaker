"""End-to-end network-isolation gate for the primary core CLI workflow (#33).

Per FR-Privacy.1 / AGENTS.md Hard Rule 1, none of ``diagnose``, ``fit``,
``report``, ``forecast``, ``compare-scenarios``, or ``refresh`` may make
outbound network calls. The static guardrail in
``test_no_network_in_core.py`` walks the import graph and rejects banned
network/telemetry imports; this test catches the *runtime* gap — a dep
that's not itself a banned import but reaches out at execution time.

The matching CI workflow runs this test inside ``sudo unshare --net`` so
the kernel itself denies any non-loopback traffic, including from
subprocess workers PyMC may spawn for parallel sampling. As a clearer-
error belt-and-suspenders, the test also patches
``socket.socket.connect`` in the parent process so any in-process
connect names the offending address in the failure.

Engine-marked: the ``fit`` step is a real PyMC quick-mode sample on a
small synthetic dataset. Skipped automatically when
``WANAMAKER_RUN_ENGINE_TESTS`` is unset or pymc is not installed.
"""

from __future__ import annotations

import os
import socket
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.engine

runner = CliRunner()


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def block_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``socket.socket.connect`` to refuse non-loopback addresses.

    Belt-and-suspenders alongside the OS-level network namespace in the
    CI workflow. If any in-process code path inside the CLI calls
    ``connect()`` to an external host, the test fails immediately with
    a message naming the offending address.

    Loopback (``127.0.0.1``, ``::1``, ``localhost``) is allowed because
    PyMC's parallel sampler may use loopback for IPC between workers and
    the main process; that is local-only and not a privacy concern.
    """
    real_connect = socket.socket.connect

    def _blocked_connect(self: socket.socket, address: object) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="replace")
        if str(host) in _LOOPBACK_HOSTS:
            return real_connect(self, address)
        raise OSError(
            f"Network access blocked: socket.connect({address!r}). "
            "Core CLI commands must not make outbound calls "
            "(AGENTS.md Hard Rule 1 / FR-Privacy.1)."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _toy_dataset(n_weeks: int = 60) -> pd.DataFrame:
    """Synthetic dataset just large enough to clear the diagnose gate.

    60 weeks puts history above the 52-week warning threshold, so the
    diagnose command exits 0 with READY (or USABLE_WITH_WARNINGS). The
    schema matches the CLI's expectations: ``week`` (date column),
    ``revenue`` (target), ``search``/``tv`` (channels), ``promo`` (control).
    """
    rng = np.random.default_rng(seed=20260601)
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    search = rng.uniform(800.0, 1500.0, size=n_weeks)
    tv = rng.uniform(2500.0, 5000.0, size=n_weeks)
    promo = (rng.random(n_weeks) > 0.7).astype(float)
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
            "search": search,
            "tv": tv,
            "promo": promo,
        }
    )


def _write_config(tmp_path: Path, csv_path: Path, artifact_dir: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""\
            data:
              csv_path: {csv_path.as_posix()}
              date_column: week
              target_column: revenue
              spend_columns: [search, tv]
              control_columns: [promo]
            channels:
              - {{name: search, category: paid_search}}
              - {{name: tv, category: linear_tv}}
            run:
              seed: 42
              runtime_mode: quick
              artifact_dir: {artifact_dir.as_posix()}
            """
        )
    )
    return cfg_path


def _write_plan(
    tmp_path: Path, name: str, *, search: float, tv: float,
) -> Path:
    """A two-period future plan with the channels and control the model needs."""
    path = tmp_path / name
    path.write_text(
        textwrap.dedent(
            f"""\
            period,search,tv,promo
            2024-08-05,{search},{tv},0
            2024-08-12,{search * 1.05},{tv},0
            """
        )
    )
    return path


def _extract_run_id(output: str) -> str:
    """Pull the run_id out of fit's stdout."""
    for line in output.splitlines():
        if line.startswith("Run ID: "):
            return line.removeprefix("Run ID: ").strip()
    raise AssertionError(f"Run ID not found in fit output:\n{output}")


# ---------------------------------------------------------------------------
# The six-command flow
# ---------------------------------------------------------------------------


def test_primary_core_workflow_in_network_isolation(tmp_path: Path) -> None:
    """Run ``diagnose`` → ``fit`` → ``report`` → ``forecast`` →
    ``compare-scenarios`` → ``refresh`` end-to-end with outbound network
    blocked, asserting each command exits cleanly and writes its
    expected artifact."""
    if os.getenv("WANAMAKER_RUN_ENGINE_TESTS") != "1":
        pytest.skip("Set WANAMAKER_RUN_ENGINE_TESTS=1 to run engine tests.")
    pytest.importorskip("pymc")

    from wanamaker.cli import app

    artifact_dir = tmp_path / ".wanamaker"
    csv_path = tmp_path / "data.csv"
    _toy_dataset().to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path, artifact_dir)

    # 1. diagnose
    result = runner.invoke(
        app, ["diagnose", str(csv_path), "--config", str(config)]
    )
    assert result.exit_code == 0, f"diagnose failed:\n{result.output}"
    assert "Readiness:" in result.output

    # 2. fit
    result = runner.invoke(app, ["fit", "--config", str(config)])
    assert result.exit_code == 0, f"fit failed:\n{result.output}"
    run_id = _extract_run_id(result.output)
    run_dir = artifact_dir / "runs" / run_id
    for name in (
        "manifest.json", "config.yaml", "summary.json",
        "posterior.nc", "data_hash.txt", "engine.txt",
    ):
        assert (run_dir / name).exists(), f"fit did not write {name}"

    # 3. report
    result = runner.invoke(
        app,
        [
            "report", "--run-id", run_id,
            "--artifact-dir", str(artifact_dir),
        ],
    )
    assert result.exit_code == 0, f"report failed:\n{result.output}"
    report_path = run_dir / "report.md"
    assert report_path.exists(), "report did not write report.md"
    body = report_path.read_text(encoding="utf-8")
    assert "# Executive summary" in body
    assert "# Model Trust Card" in body

    # 4. forecast
    plan_a = _write_plan(tmp_path, "plan_a.csv", search=1000.0, tv=3000.0)
    result = runner.invoke(
        app,
        [
            "forecast", "--run-id", run_id, "--plan", str(plan_a),
            "--artifact-dir", str(artifact_dir),
        ],
    )
    assert result.exit_code == 0, f"forecast failed:\n{result.output}"
    forecast_md = run_dir / "forecast_plan_a.md"
    assert forecast_md.exists(), "forecast did not write forecast_plan_a.md"

    # 5. compare-scenarios
    plan_b = _write_plan(tmp_path, "plan_b.csv", search=1500.0, tv=3000.0)
    result = runner.invoke(
        app,
        [
            "compare-scenarios", "--run-id", run_id,
            "--plans", str(plan_a), "--plans", str(plan_b),
            "--artifact-dir", str(artifact_dir),
        ],
    )
    assert result.exit_code == 0, (
        f"compare-scenarios failed:\n{result.output}"
    )
    assert (run_dir / "scenario_comparison.md").exists()

    # 6. refresh — uses the fit's run as the prior; produces a new run dir.
    result = runner.invoke(app, ["refresh", "--config", str(config)])
    assert result.exit_code == 0, f"refresh failed:\n{result.output}"
    new_runs = [
        p for p in (artifact_dir / "runs").iterdir() if p.name != run_id
    ]
    assert len(new_runs) == 1, (
        f"refresh did not create a new run dir; existing dirs: "
        f"{[p.name for p in (artifact_dir / 'runs').iterdir()]}"
    )
    assert (new_runs[0] / "refresh_diff.json").exists()
