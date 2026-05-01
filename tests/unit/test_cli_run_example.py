"""Tests for the wanamaker run --example one-command demo (#42 / FR-7.1).

Verifies the bundled-example wiring: the ``run`` subcommand resolves the
in-package config + CSV, materialises a config snapshot, and chains
diagnose → fit → report. The fit step is stubbed at the engine level so
the test runs in milliseconds without invoking PyMC; the engine-marked
end-to-end test in ``tests/test_network_isolation.py`` already covers
the live PyMC path on CI.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from wanamaker.engine.base import FitResult, Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    PosteriorSummary,
    PredictiveSummary,
)

runner = CliRunner()


@pytest.fixture()
def stub_pymc_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``PyMCEngine.fit`` and ``load_posterior`` with stubs.

    The stub keeps the channel structure honest (one
    ``ChannelContributionSummary`` per channel in the model spec) so
    downstream report rendering exercises a realistic shape.
    """

    def _fit(
        self: Any, *, model_spec: Any, data: Any, seed: int, runtime_mode: str,  # noqa: ARG001
    ) -> FitResult:
        contributions = [
            ChannelContributionSummary(
                channel=channel.name,
                mean_contribution=1000.0 + 100.0 * idx,
                hdi_low=900.0,
                hdi_high=1100.0,
                roi_mean=1.5,
                roi_hdi_low=1.4,
                roi_hdi_high=1.6,
                observed_spend_min=10.0,
                observed_spend_max=50.0,
            )
            for idx, channel in enumerate(model_spec.channels)
        ]
        summary = PosteriorSummary(
            channel_contributions=contributions,
            convergence=ConvergenceSummary(
                max_r_hat=1.005, min_ess_bulk=500.0,
                n_divergences=0, n_chains=4, n_draws=1000,
            ),
            in_sample_predictive=PredictiveSummary(
                periods=data["week"].astype(str).tolist(),
                mean=[100.0] * len(data),
                hdi_low=[90.0] * len(data),
                hdi_high=[110.0] * len(data),
            ),
        )
        return FitResult(
            posterior=Posterior(raw=object()),
            summary=summary,
            diagnostics={"max_r_hat": 1.005},
        )

    def _load_posterior(self: Any, run_dir: Any, model_spec: Any, data: Any) -> Posterior:  # noqa: ARG001
        return Posterior(raw=object())

    monkeypatch.setattr("wanamaker.engine.pymc.PyMCEngine.fit", _fit)
    monkeypatch.setattr(
        "wanamaker.engine.pymc.PyMCEngine.load_posterior", _load_posterior
    )


# ---------------------------------------------------------------------------
# Bundled-data sanity
# ---------------------------------------------------------------------------


class TestBundledExampleData:
    def test_public_benchmark_csv_is_in_package(self) -> None:
        root = importlib.resources.files("wanamaker._examples.public_benchmark")
        assert root.joinpath("public_example.csv").is_file()
        assert root.joinpath("public_example.yaml").is_file()
        assert root.joinpath("public_example_metadata.json").is_file()

    def test_bundled_yaml_has_relative_csv_path(self) -> None:
        """The bundled config must use a bare csv filename so the runtime
        resolution against the package directory works regardless of cwd."""
        import yaml

        root = importlib.resources.files("wanamaker._examples.public_benchmark")
        cfg = yaml.safe_load(root.joinpath("public_example.yaml").read_text())
        assert cfg["data"]["csv_path"] == "public_example.csv"


# ---------------------------------------------------------------------------
# CLI: error paths
# ---------------------------------------------------------------------------


class TestRunExampleErrors:
    def test_unknown_example_errors_cleanly(self, tmp_path: Path) -> None:
        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "run",
                "--example", "not_a_thing",
                "--artifact-dir", str(tmp_path / ".wanamaker"),
            ],
        )
        assert result.exit_code == 2
        assert "unknown example" in result.output.lower()
        assert "public_benchmark" in result.output


# ---------------------------------------------------------------------------
# CLI: happy path
# ---------------------------------------------------------------------------


class TestRunExampleHappyPath:
    def test_chains_diagnose_fit_report_and_writes_report_md(
        self, tmp_path: Path, stub_pymc_engine: None,
    ) -> None:
        from wanamaker.cli import app

        artifact_dir = tmp_path / ".wanamaker"

        result = runner.invoke(
            app,
            [
                "run",
                "--example", "public_benchmark",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        # All three steps echoed their headings.
        out = result.output
        assert "Step 1/3" in out and "readiness diagnostic" in out
        assert "Step 2/3" in out and "fitting model" in out
        assert "Step 3/3" in out and "executive summary" in out

        # Exactly one new run dir, with a complete artifact set.
        runs = list((artifact_dir / "runs").iterdir())
        assert len(runs) == 1
        run_dir = runs[0]
        for name in (
            "manifest.json", "config.yaml", "summary.json",
            "data_hash.txt", "engine.txt", "report.md",
        ):
            assert (run_dir / name).exists(), f"missing {name}"

        # Materialised example config sits next to the runs dir, with
        # csv_path resolved to the bundled file.
        materialised = artifact_dir / "_example_public_benchmark_config.yaml"
        assert materialised.exists()
        body = materialised.read_text()
        assert "wanamaker/_examples/public_benchmark/public_example.csv" in body \
            or "wanamaker\\_examples\\public_benchmark\\public_example.csv" in body

        # The report content is printed to stdout at the end.
        report_body = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "# Executive summary" in report_body
        assert "# Model Trust Card" in report_body
        # Every spend channel from the bundled config appears in the report.
        for channel in (
            "paid_search", "paid_social", "linear_tv", "ctv",
            "online_video", "display", "affiliate", "email",
        ):
            assert f"`{channel}`" in report_body
