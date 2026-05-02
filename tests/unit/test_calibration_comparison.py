"""Tests for the calibrated-vs-uncalibrated comparison report (issue #80).

The module under test is purely functional: it consumes two
``PosteriorSummary`` objects and produces a typed comparison plus a
deterministic Markdown report. These tests verify the four classifier
buckets, the summary-sentence selection logic, the ChannelSetMismatch
error, and that the rendered Markdown contains the right structure.

CLI-level mismatch errors (data_hash mismatch, both-calibrated, neither-
calibrated, missing summary, etc.) are exercised via the CLI smoke
test plus a focused subset here using the public typer.testing.CliRunner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
)
from wanamaker.reports import (
    ChannelSetMismatchError,
    build_calibration_comparison_context,
    compare_calibration,
    render_calibration_comparison,
)


def _channel(
    name: str,
    *,
    contribution: float = 1000.0,
    roi_mean: float = 2.0,
    roi_low: float = 1.6,
    roi_high: float = 2.4,
) -> ChannelContributionSummary:
    return ChannelContributionSummary(
        channel=name,
        mean_contribution=contribution,
        hdi_low=contribution * 0.8,
        hdi_high=contribution * 1.2,
        roi_mean=roi_mean,
        roi_hdi_low=roi_low,
        roi_hdi_high=roi_high,
        observed_spend_min=10.0,
        observed_spend_max=50.0,
    )


def _summary(*channels: ChannelContributionSummary) -> PosteriorSummary:
    return PosteriorSummary(channel_contributions=list(channels))


# ---------------------------------------------------------------------------
# Channel-set validation
# ---------------------------------------------------------------------------


class TestChannelSetMismatch:
    def test_missing_channel_in_calibrated_raises(self) -> None:
        uncal = _summary(_channel("search"), _channel("tv"))
        cal = _summary(_channel("search"))
        with pytest.raises(ChannelSetMismatchError, match="Only in uncalibrated"):
            compare_calibration(
                uncal, cal,
                uncalibrated_run_id="u",
                calibrated_run_id="c",
                calibrated_channels=["search"],
            )

    def test_extra_channel_in_calibrated_raises(self) -> None:
        uncal = _summary(_channel("search"))
        cal = _summary(_channel("search"), _channel("tv"))
        with pytest.raises(ChannelSetMismatchError, match="only in calibrated"):
            compare_calibration(
                uncal, cal,
                uncalibrated_run_id="u",
                calibrated_run_id="c",
                calibrated_channels=["search"],
            )


# ---------------------------------------------------------------------------
# Per-channel classifier (the four directly-calibrated buckets + the
# secondary-shift bucket for non-calibrated channels)
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_no_material_change_when_means_inside_each_others_hdi(self) -> None:
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.5, roi_high=2.5))
        cal = _summary(_channel("search", roi_mean=2.05, roi_low=1.6, roi_high=2.5))
        result = compare_calibration(
            uncal, cal,
            uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        ch = result.channels[0]
        assert ch.classification == "history-dominant"
        # Wider HDI → no shrinkage; matching means → no material change.
        assert not ch.is_material_change

    def test_experiment_dominant_when_calibrated_hdi_much_tighter(self) -> None:
        # Uncalibrated HDI is wide; calibrated HDI is much narrower.
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.0, roi_high=3.0))
        cal = _summary(_channel("search", roi_mean=2.0, roi_low=1.85, roi_high=2.15))
        result = compare_calibration(
            uncal, cal,
            uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        assert result.channels[0].classification == "experiment-dominant"

    def test_directional_shift_when_mean_moves_outside_uncal_hdi(self) -> None:
        # cal mean (1.0) is below uncalibrated HDI (1.6-2.4); calibrated
        # HDI (0.7-1.3) is similar in width to uncalibrated (0.8 wide
        # each), so it doesn't trip the experiment-dominant ratio gate.
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.6, roi_high=2.4))
        cal = _summary(_channel("search", roi_mean=1.0, roi_low=0.7, roi_high=1.3))
        result = compare_calibration(
            uncal, cal,
            uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        ch = result.channels[0]
        assert ch.classification == "directional-shift"
        assert ch.is_material_change

    def test_history_dominant_when_calibration_did_not_move_posterior(self) -> None:
        # Same uncalibrated and calibrated HDIs, same mean → had a lift
        # test (so the channel is in calibrated_channels) but the
        # posterior didn't move materially and the HDI didn't shrink.
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.5, roi_high=2.5))
        cal = _summary(_channel("search", roi_mean=2.02, roi_low=1.51, roi_high=2.5))
        result = compare_calibration(
            uncal, cal,
            uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        assert result.channels[0].classification == "history-dominant"

    def test_secondary_shift_for_non_calibrated_channel_with_material_move(self) -> None:
        # `tv` was not directly calibrated, but `search`'s calibration
        # reshaped the contribution mix and `tv`'s ROI shifted outside
        # its uncalibrated HDI as a side effect.
        uncal = _summary(
            _channel("search", roi_mean=2.0),
            _channel("tv", roi_mean=1.5, roi_low=1.3, roi_high=1.7),
        )
        cal = _summary(
            _channel("search", roi_mean=1.5, roi_low=1.4, roi_high=1.6),
            _channel("tv", roi_mean=1.0, roi_low=0.8, roi_high=1.2),
        )
        result = compare_calibration(
            uncal, cal,
            uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        tv_row = next(c for c in result.channels if c.channel == "tv")
        assert tv_row.classification == "secondary-shift"
        assert not tv_row.is_calibrated

    def test_no_material_change_for_non_calibrated_consistent_channel(self) -> None:
        uncal = _summary(
            _channel("search", roi_mean=2.0, roi_low=1.5, roi_high=2.5),
            _channel("tv", roi_mean=1.5, roi_low=1.3, roi_high=1.7),
        )
        cal = _summary(
            _channel("search", roi_mean=1.95, roi_low=1.85, roi_high=2.05),
            _channel("tv", roi_mean=1.55, roi_low=1.35, roi_high=1.75),
        )
        result = compare_calibration(
            uncal, cal,
            uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        tv_row = next(c for c in result.channels if c.channel == "tv")
        assert tv_row.classification == "no-material-change"


# ---------------------------------------------------------------------------
# Summary-sentence priority — experiment-dominant > directional-shift >
# secondary-shift > history-dominant > no-material-change
# ---------------------------------------------------------------------------


class TestSummarySentence:
    def test_leads_with_experiment_dominant_when_present(self) -> None:
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.0, roi_high=3.0))
        cal = _summary(_channel("search", roi_mean=2.0, roi_low=1.9, roi_high=2.1))
        result = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        assert "experiment-led" in result.summary_sentence.lower()
        assert "search" in result.summary_sentence.lower()

    def test_falls_back_to_directional_shift_message(self) -> None:
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.6, roi_high=2.4))
        cal = _summary(_channel("search", roi_mean=1.0, roi_low=0.7, roi_high=1.3))
        result = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        assert "uncalibrated credible interval" in result.summary_sentence.lower()

    def test_history_dominant_message_when_nothing_moved(self) -> None:
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.5, roi_high=2.5))
        cal = _summary(_channel("search", roi_mean=2.02, roi_low=1.51, roi_high=2.5))
        result = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        assert "did not materially move" in result.summary_sentence.lower()

    def test_no_material_change_message_when_no_lift_test_channels(self) -> None:
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.5, roi_high=2.5))
        cal = _summary(_channel("search", roi_mean=1.95, roi_low=1.85, roi_high=2.05))
        result = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=[],  # no calibrated channels
        )
        assert "no material change" in result.summary_sentence.lower()


# ---------------------------------------------------------------------------
# Ranking and contribution math
# ---------------------------------------------------------------------------


class TestStructuralProperties:
    def test_channels_ordered_by_uncalibrated_contribution_desc(self) -> None:
        uncal = _summary(
            _channel("small", contribution=100.0),
            _channel("big", contribution=10_000.0),
            _channel("medium", contribution=1_000.0),
        )
        cal = _summary(
            _channel("small", contribution=110.0),
            _channel("big", contribution=10_500.0),
            _channel("medium", contribution=900.0),
        )
        result = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=[],
        )
        assert [c.channel for c in result.channels] == ["big", "medium", "small"]

    def test_total_media_aggregates_correctly(self) -> None:
        uncal = _summary(
            _channel("a", contribution=300.0),
            _channel("b", contribution=200.0),
        )
        cal = _summary(
            _channel("a", contribution=350.0),
            _channel("b", contribution=180.0),
        )
        result = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=[],
        )
        assert result.total_media_uncal == 500.0
        assert result.total_media_cal == 530.0


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdownRender:
    def test_render_includes_summary_per_channel_table_and_status_legend(self) -> None:
        uncal = _summary(
            _channel("search", roi_mean=2.0, roi_low=1.0, roi_high=3.0),
            _channel("tv", roi_mean=1.5),
        )
        cal = _summary(
            _channel("search", roi_mean=2.0, roi_low=1.9, roi_high=2.1),
            _channel("tv", roi_mean=1.5, roi_low=1.3, roi_high=1.7),
        )
        comparison = compare_calibration(
            uncal, cal, uncalibrated_run_id="u", calibrated_run_id="c",
            calibrated_channels=["search"],
        )
        ctx = build_calibration_comparison_context(comparison)
        body = render_calibration_comparison(ctx)

        assert "# Calibration comparison" in body
        assert "## Headline" in body
        assert "experiment-led" in body
        assert "## Per-channel comparison" in body
        # Status legend must explain every classification term it uses.
        for label in (
            "no material change",
            "experiment-dominant",
            "directional shift",
            "history-dominant",
            "secondary shift",
        ):
            assert label in body
        # Both runs are named in the header.
        assert "`u`" in body
        assert "`c`" in body
        # Channels appear in both tables.
        assert body.count("`search`") >= 2
        assert body.count("`tv`") >= 2

    def test_no_banned_phrases_in_rendered_output(self) -> None:
        # The terminology guardrail scans this module's source; this
        # additional check belts-and-suspenders the rendered output for
        # a real-world configuration.
        banned = (
            "optimized budget", "optimal allocation", "best budget",
            "guaranteed lift", "maximize roi",
        )
        uncal = _summary(_channel("search", roi_mean=2.0, roi_low=1.0, roi_high=3.0))
        cal = _summary(_channel("search", roi_mean=2.0, roi_low=1.9, roi_high=2.1))
        body = render_calibration_comparison(
            build_calibration_comparison_context(
                compare_calibration(
                    uncal, cal,
                    uncalibrated_run_id="u", calibrated_run_id="c",
                    calibrated_channels=["search"],
                )
            )
        )
        lowered = body.lower()
        for phrase in banned:
            assert phrase not in lowered, f"banned phrase {phrase!r} in render"


# ---------------------------------------------------------------------------
# CLI mismatch errors — exercise the failure paths the message body
# advertises, since they only fire at the CLI layer.
# ---------------------------------------------------------------------------


def _runner() -> CliRunner:
    return CliRunner()


def _write_run(
    artifact_dir: Path,
    run_id: str,
    *,
    summary: PosteriorSummary,
    data_hash: str,
    calibration: bool,
) -> Path:
    from wanamaker.artifacts import serialize_summary

    run_dir = artifact_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(serialize_summary(summary))
    (run_dir / "data_hash.txt").write_text(data_hash)
    # Minimal valid config; calibration toggles whether lift_test_csv is
    # set to a real file (which we also write so the loader can read it).
    if calibration:
        lift_csv = run_dir / "lift.csv"
        lift_csv.write_text(
            "channel,test_start,test_end,roi_estimate,roi_ci_lower,roi_ci_upper\n"
            "search,2024-01-01,2024-01-14,2.0,1.6,2.4\n",
            encoding="utf-8",
        )
        cal_block = f"  lift_test_csv: {lift_csv.as_posix()}\n"
    else:
        cal_block = ""
    (run_dir / "config.yaml").write_text(
        "data:\n"
        "  csv_path: stub.csv\n"
        "  date_column: week\n"
        "  target_column: revenue\n"
        "  spend_columns: [search]\n"
        f"{cal_block}"
        "channels:\n"
        "  - {name: search, category: paid_search}\n"
        "run:\n"
        "  seed: 1\n"
        "  runtime_mode: quick\n"
    )
    return run_dir


class TestCliMismatchErrors:
    def test_missing_run_errors_cleanly(self, tmp_path: Path) -> None:
        from wanamaker.cli import app

        artifact_dir = tmp_path / ".wanamaker"
        artifact_dir.mkdir()
        result = _runner().invoke(
            app,
            [
                "compare-calibration",
                "--uncalibrated-run", "does_not_exist",
                "--calibrated-run", "also_missing",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_data_hash_mismatch_errors_cleanly(self, tmp_path: Path) -> None:
        from wanamaker.cli import app

        artifact_dir = tmp_path / ".wanamaker"
        s = _summary(_channel("search"))
        _write_run(artifact_dir, "uncal_run", summary=s, data_hash="aaa", calibration=False)
        _write_run(artifact_dir, "cal_run", summary=s, data_hash="bbb", calibration=True)

        result = _runner().invoke(
            app,
            [
                "compare-calibration",
                "--uncalibrated-run", "uncal_run",
                "--calibrated-run", "cal_run",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "different training data" in result.output.lower()

    def test_neither_calibrated_errors_cleanly(self, tmp_path: Path) -> None:
        from wanamaker.cli import app

        artifact_dir = tmp_path / ".wanamaker"
        s = _summary(_channel("search"))
        _write_run(artifact_dir, "u", summary=s, data_hash="hash1", calibration=False)
        _write_run(artifact_dir, "c", summary=s, data_hash="hash1", calibration=False)

        result = _runner().invoke(
            app,
            [
                "compare-calibration",
                "--uncalibrated-run", "u",
                "--calibrated-run", "c",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "neither run has lift-test priors" in result.output.lower()

    def test_both_calibrated_errors_cleanly(self, tmp_path: Path) -> None:
        from wanamaker.cli import app

        artifact_dir = tmp_path / ".wanamaker"
        s = _summary(_channel("search"))
        _write_run(artifact_dir, "u", summary=s, data_hash="hash1", calibration=True)
        _write_run(artifact_dir, "c", summary=s, data_hash="hash1", calibration=True)

        result = _runner().invoke(
            app,
            [
                "compare-calibration",
                "--uncalibrated-run", "u",
                "--calibrated-run", "c",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "both runs have lift-test priors" in result.output.lower()

    def test_swapped_flags_errors_cleanly(self, tmp_path: Path) -> None:
        """User passed the calibrated run as --uncalibrated-run and vice versa.
        We detect that and tell them to swap."""
        from wanamaker.cli import app

        artifact_dir = tmp_path / ".wanamaker"
        s = _summary(_channel("search"))
        # The "uncalibrated" run actually has priors; the "calibrated"
        # run does not.
        _write_run(artifact_dir, "uncal_label", summary=s, data_hash="hash1", calibration=True)
        _write_run(artifact_dir, "cal_label", summary=s, data_hash="hash1", calibration=False)

        result = _runner().invoke(
            app,
            [
                "compare-calibration",
                "--uncalibrated-run", "uncal_label",
                "--calibrated-run", "cal_label",
                "--artifact-dir", str(artifact_dir),
            ],
        )
        assert result.exit_code == 1
        assert "swap the flag values" in result.output.lower()
