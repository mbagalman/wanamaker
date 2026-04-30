"""Tests for the wanamaker refresh command (issue #22).

These tests stub out ``PyMCEngine.fit`` so the orchestration logic can be
exercised without invoking PyMC. They cover:

- The "no previous run" error path
- Selection of the most recent prior run matching the configured data file
- Skipping prior runs whose snapshotted config used a different data file
- ``--anchor-strength`` overriding the config default and accepting both
  preset names and numeric weights
- The refresh artifact (``refresh_diff.json``) being persisted with
  movement classifications per FR-4.3
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from wanamaker.artifacts import serialize_summary
from wanamaker.engine.base import FitResult, Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


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


def _write_config(
    tmp_path: Path,
    *,
    csv_path: Path,
    artifact_dir: Path,
    anchor_strength: str | float = "medium",
) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
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
            refresh:
              anchor_strength: {anchor_strength}
            run:
              seed: 0
              runtime_mode: quick
              artifact_dir: {artifact_dir.as_posix()}
            """
        )
    )
    return cfg_path


def _previous_summary() -> PosteriorSummary:
    return PosteriorSummary(
        parameters=[
            ParameterSummary(
                name="channel.search.half_life",
                mean=1.2, sd=0.3, hdi_low=0.8, hdi_high=1.8,
            ),
            ParameterSummary(
                name="channel.search.coefficient",
                mean=2.0, sd=0.4, hdi_low=1.5, hdi_high=2.5,
            ),
        ],
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=5000.0, hdi_low=3000.0, hdi_high=7000.0,
                observed_spend_min=10.0, observed_spend_max=15.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=10000.0, hdi_low=8000.0, hdi_high=12000.0,
                observed_spend_min=50.0, observed_spend_max=60.0,
            ),
        ],
        convergence=ConvergenceSummary(
            max_r_hat=1.01, min_ess_bulk=400.0,
            n_divergences=0, n_chains=4, n_draws=1000,
        ),
    )


def _seed_prior_run(
    artifact_dir: Path,
    *,
    run_id: str,
    csv_path: Path,
    summary: PosteriorSummary,
) -> Path:
    """Write a minimal but valid prior run artifact directory.

    Only the files refresh consults are written: ``config.yaml`` (so the
    prior data path can be matched) and ``summary.json`` (the posterior
    summary used to anchor and to diff against).
    """
    run_dir = artifact_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config: same csv_path, otherwise minimal.
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
              seed: 0
              runtime_mode: quick
            """
        )
    )
    (run_dir / "summary.json").write_text(serialize_summary(summary))
    return run_dir


@dataclass
class _FakeRawPosterior:
    """Stand-in for ``PyMCRawPosterior`` so ``_write_posterior`` exits via
    the unrecognised-raw branch (warning, no NetCDF write)."""

    note: str = "fake"


def _current_summary_within_ci() -> PosteriorSummary:
    """A current-run summary whose movements all sit inside the previous HDI."""
    return PosteriorSummary(
        parameters=[
            ParameterSummary(
                name="channel.search.half_life",
                mean=1.25, sd=0.3, hdi_low=0.85, hdi_high=1.85,
            ),
            ParameterSummary(
                name="channel.search.coefficient",
                mean=2.05, sd=0.4, hdi_low=1.6, hdi_high=2.5,
            ),
        ],
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=5050.0, hdi_low=3100.0, hdi_high=7050.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=10100.0, hdi_low=8100.0, hdi_high=12100.0,
            ),
        ],
        convergence=ConvergenceSummary(
            max_r_hat=1.01, min_ess_bulk=400.0,
            n_divergences=0, n_chains=4, n_draws=1000,
        ),
    )


def _current_summary_unexplained() -> PosteriorSummary:
    """A current-run summary with one parameter that moves well outside its
    previous 95 % HDI and has a tight current HDI — i.e. UNEXPLAINED."""
    return PosteriorSummary(
        parameters=[
            ParameterSummary(
                name="channel.search.coefficient",
                mean=10.0, sd=0.1, hdi_low=9.8, hdi_high=10.2,
            ),
        ],
        channel_contributions=[],
        convergence=ConvergenceSummary(
            max_r_hat=1.01, min_ess_bulk=400.0,
            n_divergences=0, n_chains=4, n_draws=1000,
        ),
    )


@pytest.fixture()
def fake_engine_fit(monkeypatch: pytest.MonkeyPatch):
    """Patch ``PyMCEngine.fit`` to return a caller-supplied summary.

    Yields a setter that tests use to install the desired summary; also
    captures the ``ModelSpec`` actually passed to the engine so tests can
    assert on the anchor priors that refresh built.
    """
    captured: dict[str, Any] = {}

    def install(summary: PosteriorSummary) -> None:
        def _fit(
            self: Any,
            *,
            model_spec: Any,
            data: Any,  # noqa: ARG001
            seed: int,
            runtime_mode: str,
        ) -> FitResult:
            captured["model_spec"] = model_spec
            captured["seed"] = seed
            captured["runtime_mode"] = runtime_mode
            return FitResult(
                posterior=Posterior(raw=_FakeRawPosterior()),
                summary=summary,
                diagnostics={"max_r_hat": 1.01},
            )

        monkeypatch.setattr(
            "wanamaker.engine.pymc.PyMCEngine.fit", _fit, raising=True
        )

    install._captured = captured  # type: ignore[attr-defined]
    return install


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRefreshErrorPaths:
    def test_errors_when_no_previous_run_exists(self, tmp_path: Path) -> None:
        csv = _write_data_csv(tmp_path)
        cfg = _write_config(tmp_path, csv_path=csv, artifact_dir=tmp_path / ".wanamaker")

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "no previous run found" in result.output.lower()

    def test_errors_when_prior_run_used_different_data_file(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        # Two CSVs in the same directory; the prior run used 'other.csv'.
        csv = _write_data_csv(tmp_path, "data.csv")
        other_csv = _write_data_csv(tmp_path, "other.csv")
        artifact_dir = tmp_path / ".wanamaker"

        _seed_prior_run(
            artifact_dir,
            run_id="00000000_20260101T000000Z",
            csv_path=other_csv,
            summary=_previous_summary(),
        )

        cfg = _write_config(tmp_path, csv_path=csv, artifact_dir=artifact_dir)

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "no previous run" in result.output.lower()

    def test_invalid_anchor_strength_string_errors(self, tmp_path: Path) -> None:
        csv = _write_data_csv(tmp_path)
        cfg = _write_config(tmp_path, csv_path=csv, artifact_dir=tmp_path / ".wanamaker")

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            ["refresh", "--config", str(cfg), "--anchor-strength", "extra-heavy"],
        )
        # Either typer surfaces a non-zero exit, or the underlying ValueError
        # propagates — the contract is that this does not silently succeed.
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Happy path: artifact persistence and run-selection
# ---------------------------------------------------------------------------


class TestRefreshHappyPath:
    def test_persists_refresh_diff_and_run_artifacts(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        csv = _write_data_csv(tmp_path)
        artifact_dir = tmp_path / ".wanamaker"
        prev_run_id = "00000000_20260101T000000Z"
        _seed_prior_run(
            artifact_dir, run_id=prev_run_id, csv_path=csv, summary=_previous_summary()
        )
        cfg = _write_config(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        fake_engine_fit(_current_summary_within_ci())

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 0, result.output

        # A new run directory exists and contains the expected files.
        runs_dir = artifact_dir / "runs"
        new_runs = [p for p in runs_dir.iterdir() if p.name != prev_run_id]
        assert len(new_runs) == 1
        new_run = new_runs[0]
        for name in ("manifest.json", "config.yaml", "summary.json",
                     "data_hash.txt", "timestamp.txt", "engine.txt",
                     "refresh_diff.json"):
            assert (new_run / name).exists(), f"missing {name}"

        # Refresh diff payload references both runs.
        diff_envelope = json.loads((new_run / "refresh_diff.json").read_text())
        payload = diff_envelope["payload"]
        assert payload["previous_run_id"] == prev_run_id
        assert payload["current_run_id"] == new_run.name
        assert payload["movements"], "expected at least one movement"

    def test_classifies_movements_per_fr_4_3(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        csv = _write_data_csv(tmp_path)
        artifact_dir = tmp_path / ".wanamaker"
        _seed_prior_run(
            artifact_dir,
            run_id="00000000_20260101T000000Z",
            csv_path=csv,
            summary=_previous_summary(),
        )
        cfg = _write_config(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        fake_engine_fit(_current_summary_unexplained())

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 0, result.output

        new_run = next(
            p for p in (artifact_dir / "runs").iterdir()
            if p.name != "00000000_20260101T000000Z"
        )
        payload = json.loads((new_run / "refresh_diff.json").read_text())["payload"]
        coef = next(
            m for m in payload["movements"]
            if m["name"] == "channel.search.coefficient"
        )
        assert coef["movement_class"] == "unexplained"

    def test_picks_most_recent_matching_run(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        csv = _write_data_csv(tmp_path)
        artifact_dir = tmp_path / ".wanamaker"
        # Older run we should ignore.
        old_summary = PosteriorSummary(
            parameters=[ParameterSummary(
                name="channel.search.coefficient",
                mean=99.0, sd=10.0, hdi_low=80.0, hdi_high=120.0,
            )]
        )
        _seed_prior_run(
            artifact_dir, run_id="aaaaaaaa_20250101T000000Z",
            csv_path=csv, summary=old_summary,
        )
        # Newer run we should pick.
        newer = _previous_summary()
        newer_run_id = "bbbbbbbb_20260101T000000Z"
        _seed_prior_run(
            artifact_dir, run_id=newer_run_id, csv_path=csv, summary=newer,
        )

        cfg = _write_config(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        fake_engine_fit(_current_summary_within_ci())

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 0, result.output

        new_run = next(
            p for p in (artifact_dir / "runs").iterdir()
            if p.name not in {"aaaaaaaa_20250101T000000Z", newer_run_id}
        )
        payload = json.loads((new_run / "refresh_diff.json").read_text())["payload"]
        assert payload["previous_run_id"] == newer_run_id


# ---------------------------------------------------------------------------
# --anchor-strength resolution
# ---------------------------------------------------------------------------


class TestAnchorStrengthResolution:
    def _setup(self, tmp_path: Path, anchor_strength_yaml: str | float = "medium"):
        csv = _write_data_csv(tmp_path)
        artifact_dir = tmp_path / ".wanamaker"
        _seed_prior_run(
            artifact_dir,
            run_id="00000000_20260101T000000Z",
            csv_path=csv,
            summary=_previous_summary(),
        )
        cfg = _write_config(
            tmp_path, csv_path=csv, artifact_dir=artifact_dir,
            anchor_strength=anchor_strength_yaml,
        )
        return cfg

    def test_default_uses_config_value(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        cfg = self._setup(tmp_path, anchor_strength_yaml="light")
        fake_engine_fit(_current_summary_within_ci())

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        # 'light' preset is 0.2.
        captured = fake_engine_fit._captured  # type: ignore[attr-defined]
        anchor_priors = captured["model_spec"].anchor_priors
        weights = [ap.weight for ap in anchor_priors.values()]
        assert weights == pytest.approx([0.2] * len(weights))

    def test_cli_flag_overrides_config_default(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        cfg = self._setup(tmp_path, anchor_strength_yaml="medium")
        fake_engine_fit(_current_summary_within_ci())

        from wanamaker.cli import app

        result = runner.invoke(
            app, ["refresh", "--config", str(cfg), "--anchor-strength", "heavy"],
        )
        assert result.exit_code == 0, result.output
        # 'heavy' preset is 0.5 (CLI wins over config's 'medium' = 0.3).
        captured = fake_engine_fit._captured  # type: ignore[attr-defined]
        weights = [
            ap.weight for ap in captured["model_spec"].anchor_priors.values()
        ]
        assert weights == pytest.approx([0.5] * len(weights))

    def test_cli_flag_accepts_float(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        cfg = self._setup(tmp_path)
        fake_engine_fit(_current_summary_within_ci())

        from wanamaker.cli import app

        result = runner.invoke(
            app, ["refresh", "--config", str(cfg), "--anchor-strength", "0.42"],
        )
        assert result.exit_code == 0, result.output
        captured = fake_engine_fit._captured  # type: ignore[attr-defined]
        weights = [
            ap.weight for ap in captured["model_spec"].anchor_priors.values()
        ]
        assert weights == pytest.approx([0.42] * len(weights))

    def test_anchor_priors_keyed_by_parameter_name(
        self, tmp_path: Path, fake_engine_fit
    ) -> None:
        cfg = self._setup(tmp_path)
        fake_engine_fit(_current_summary_within_ci())

        from wanamaker.cli import app

        result = runner.invoke(app, ["refresh", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        captured = fake_engine_fit._captured  # type: ignore[attr-defined]
        anchor_priors = captured["model_spec"].anchor_priors
        # Both parameter names from the previous summary appear as keys.
        assert "channel.search.half_life" in anchor_priors
        assert "channel.search.coefficient" in anchor_priors
        # Mean/sd are carried through from ParameterSummary.
        assert anchor_priors["channel.search.coefficient"].mean == pytest.approx(2.0)
        assert anchor_priors["channel.search.coefficient"].sd == pytest.approx(0.4)
