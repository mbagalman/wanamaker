"""Tests for the wanamaker report command (#30)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from wanamaker.artifacts import serialize_refresh_diff, serialize_summary
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.refresh.classify import MovementClass
from wanamaker.refresh.diff import ParameterMovement, RefreshDiff

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


def _write_data_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(CSV_BODY)
    return path


def _summary() -> PosteriorSummary:
    return PosteriorSummary(
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=300.0, hdi_low=270.0, hdi_high=330.0,
                roi_mean=2.0, roi_hdi_low=1.85, roi_hdi_high=2.15,
                observed_spend_min=10.0, observed_spend_max=15.0,
            ),
            ChannelContributionSummary(
                channel="tv",
                mean_contribution=600.0, hdi_low=400.0, hdi_high=800.0,
                roi_mean=0.6, roi_hdi_low=0.4, roi_hdi_high=0.8,
                observed_spend_min=50.0, observed_spend_max=60.0,
            ),
        ],
        convergence=ConvergenceSummary(
            max_r_hat=1.005, min_ess_bulk=500.0,
            n_divergences=0, n_chains=4, n_draws=1000,
        ),
        in_sample_predictive=PredictiveSummary(
            periods=["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            mean=[100, 110, 120, 118],
            hdi_low=[90, 100, 110, 108],
            hdi_high=[110, 120, 130, 128],
        ),
    )


def _write_run(
    tmp_path: Path,
    *,
    csv_path: Path,
    artifact_dir: Path,
    run_id: str = "00000000_20260101T000000Z",
    summary: PosteriorSummary | None = None,
    refresh_diff: RefreshDiff | None = None,
    summary_present: bool = True,
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
    if summary_present:
        (run_dir / "summary.json").write_text(serialize_summary(summary or _summary()))
    if refresh_diff is not None:
        (run_dir / "refresh_diff.json").write_text(serialize_refresh_diff(refresh_diff))
    return run_dir


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestReportErrors:
    def test_missing_run_id_errors_cleanly(self, tmp_path: Path) -> None:
        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "report",
                "--run-id", "does_not_exist",
                "--artifact-dir", str(tmp_path / ".wanamaker"),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_missing_summary_errors_cleanly(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        _write_run(
            tmp_path, csv_path=csv, artifact_dir=artifact_dir,
            summary_present=False,
        )
        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "report",
                "--run-id", "00000000_20260101T000000Z",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "summary.json" in result.output.lower()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestReportHappyPath:
    def test_writes_self_contained_report_with_both_sections(
        self, tmp_path: Path
    ) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        run_dir = _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "report",
                "--run-id", "00000000_20260101T000000Z",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        report = run_dir / "report.md"
        assert report.exists()
        body = report.read_text()
        assert "# Executive summary" in body
        assert "# Model Trust Card" in body
        # Both channels appear.
        assert "`search`" in body and "`tv`" in body
        # Anchor link is in-document, not a separate file.
        assert "[Model Trust Card](#model-trust-card)" in body
        # Sections appear in the expected order.
        assert body.index("# Executive summary") < body.index("# Model Trust Card")

    def test_refresh_diff_appears_in_report_when_present(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        diff = RefreshDiff(
            previous_run_id="aaaa1111_20250101T000000Z",
            current_run_id="bbbb2222_20260101T000000Z",
            movements=[
                ParameterMovement(
                    name="channel.search.coefficient",
                    previous_mean=2.0, current_mean=2.05,
                    previous_ci=(1.8, 2.2), current_ci=(1.85, 2.25),
                    movement_class=MovementClass.WITHIN_PRIOR_CI,
                ),
            ],
        )
        run_dir = _write_run(
            tmp_path, csv_path=csv, artifact_dir=artifact_dir, refresh_diff=diff,
        )

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "report",
                "--run-id", "00000000_20260101T000000Z",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        body = (run_dir / "report.md").read_text()
        assert "## Refresh notes" in body
        assert "aaaa1111_20250101T000000Z" in body

    def test_output_override_writes_to_custom_path(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        _write_run(tmp_path, csv_path=csv, artifact_dir=artifact_dir)
        custom = tmp_path / "custom_report.md"

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "report",
                "--run-id", "00000000_20260101T000000Z",
                "--artifact-dir", str(artifact_dir),
                "--output", str(custom),
            ],
        )
        assert result.exit_code == 0, result.output
        assert custom.exists()
        body = custom.read_text()
        assert "# Executive summary" in body and "# Model Trust Card" in body

    def test_advisor_rationale_appears_for_wide_posterior_channel(
        self, tmp_path: Path,
    ) -> None:
        """A high-share, wide-HDI channel must surface the advisor's
        rationale in the Recommended actions section, with real spend
        from the training data quoted in the bullet."""
        artifact_dir = tmp_path / ".wanamaker"
        csv = _write_data_csv(tmp_path)
        weak_summary = PosteriorSummary(
            channel_contributions=[
                ChannelContributionSummary(
                    channel="search",
                    mean_contribution=300.0, hdi_low=100.0, hdi_high=500.0,
                    roi_mean=2.0, roi_hdi_low=0.5, roi_hdi_high=3.5,
                    observed_spend_min=10.0, observed_spend_max=15.0,
                ),
            ],
            convergence=ConvergenceSummary(
                max_r_hat=1.005, min_ess_bulk=500.0,
                n_divergences=0, n_chains=4, n_draws=1000,
            ),
        )
        run_dir = _write_run(
            tmp_path, csv_path=csv, artifact_dir=artifact_dir, summary=weak_summary,
        )

        from wanamaker.cli import app

        result = runner.invoke(
            app,
            [
                "report",
                "--run-id", "00000000_20260101T000000Z",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        body = (run_dir / "report.md").read_text()
        # The advisor's rationale (issue #29) replaces the template fallback.
        assert "Channel `search` has high posterior uncertainty" in body
        assert "controlled experiment" in body
        # Spend phrase quotes real spend from the training data
        # (search column sums to 51 in the toy CSV).
        assert "significant spend (51)" in body
