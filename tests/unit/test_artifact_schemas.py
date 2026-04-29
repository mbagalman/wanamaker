"""Round-trip and schema-version tests for all persisted artifact types (issue #41).

Tests cover:
- summary.json: serialize → deserialize round-trip with full and minimal payloads
- manifest.json: write → load round-trip; version mismatch raises ValueError
- trust_card.json: serialize → deserialize round-trip
- refresh_diff.json: serialize → deserialize round-trip; tuple restoration
- summary_schema_version present in manifest
- All round-trips complete in well under 1 second (pure unit tests)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wanamaker.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    REFRESH_DIFF_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    TRUST_CARD_SCHEMA_VERSION,
    deserialize_refresh_diff,
    deserialize_summary,
    deserialize_trust_card,
    load_manifest,
    run_paths,
    serialize_refresh_diff,
    serialize_summary,
    serialize_trust_card,
    write_manifest,
)
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.refresh.diff import ParameterMovement, RefreshDiff
from wanamaker.trust_card.card import TrustCard, TrustDimension, TrustStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_summary() -> PosteriorSummary:
    return PosteriorSummary(
        parameters=[
            ParameterSummary(
                name="channel.search.roi",
                mean=2.5,
                sd=0.4,
                hdi_low=1.8,
                hdi_high=3.3,
                r_hat=1.002,
                ess_bulk=620.0,
            ),
            ParameterSummary(
                name="channel.tv.half_life",
                mean=5.1,
                sd=1.2,
                hdi_low=3.0,
                hdi_high=7.5,
            ),
        ],
        channel_contributions=[
            ChannelContributionSummary(
                channel="search",
                mean_contribution=45000.0,
                hdi_low=32000.0,
                hdi_high=58000.0,
                share_of_effect=0.6,
                roi_mean=2.5,
                roi_hdi_low=1.8,
                roi_hdi_high=3.3,
                observed_spend_min=500.0,
                observed_spend_max=4000.0,
            ),
        ],
        convergence=ConvergenceSummary(
            max_r_hat=1.01,
            min_ess_bulk=450.0,
            n_divergences=0,
            n_chains=4,
            n_draws=1000,
        ),
        in_sample_predictive=PredictiveSummary(
            periods=["2024-01-01", "2024-01-08"],
            mean=[100.0, 110.0],
            hdi_low=[90.0, 98.0],
            hdi_high=[110.0, 122.0],
        ),
    )


def _empty_summary() -> PosteriorSummary:
    return PosteriorSummary()


def _minimal_trust_card() -> TrustCard:
    return TrustCard(
        dimensions=[
            TrustDimension(
                name="convergence",
                status=TrustStatus.PASS,
                explanation="R-hat < 1.01, ESS > 400.",
            ),
            TrustDimension(
                name="holdout_accuracy",
                status=TrustStatus.MODERATE,
                explanation="MAPE 12% — acceptable but watch.",
            ),
        ]
    )


def _minimal_refresh_diff() -> RefreshDiff:
    return RefreshDiff(
        previous_run_id="abc12345_20240101T000000Z",
        current_run_id="def67890_20240201T000000Z",
        movements=[
            ParameterMovement(
                name="channel.search.roi",
                previous_mean=2.1,
                current_mean=2.4,
                previous_ci=(1.6, 2.7),
                current_ci=(1.9, 3.0),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# PosteriorSummary round-trip
# ---------------------------------------------------------------------------


class TestSummaryRoundTrip:
    def test_full_round_trip(self) -> None:
        summary = _full_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored == summary

    def test_empty_round_trip(self) -> None:
        summary = _empty_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored == summary

    def test_parameters_preserved(self) -> None:
        summary = _full_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored.parameters[0].name == "channel.search.roi"
        assert restored.parameters[0].mean == pytest.approx(2.5)
        assert restored.parameters[0].r_hat == pytest.approx(1.002)

    def test_channel_contributions_preserved(self) -> None:
        summary = _full_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored.channel_contributions[0].channel == "search"
        assert restored.channel_contributions[0].share_of_effect == pytest.approx(0.6)
        assert restored.channel_contributions[0].spend_invariant is False

    def test_convergence_preserved(self) -> None:
        summary = _full_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored.convergence is not None
        assert restored.convergence.n_chains == 4
        assert restored.convergence.n_divergences == 0

    def test_in_sample_predictive_preserved(self) -> None:
        summary = _full_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored.in_sample_predictive is not None
        assert restored.in_sample_predictive.periods == ["2024-01-01", "2024-01-08"]
        assert restored.in_sample_predictive.mean == pytest.approx([100.0, 110.0])

    def test_null_convergence_preserved(self) -> None:
        summary = PosteriorSummary(convergence=None)
        restored = deserialize_summary(serialize_summary(summary))
        assert restored.convergence is None

    def test_null_predictive_preserved(self) -> None:
        summary = _full_summary()
        assert summary.in_sample_predictive is not None  # sanity check
        # Verify a summary with no predictive also round-trips
        no_pred = PosteriorSummary(
            parameters=summary.parameters,
            convergence=summary.convergence,
        )
        restored = deserialize_summary(serialize_summary(no_pred))
        assert restored.in_sample_predictive is None

    def test_interval_mass_preserved(self) -> None:
        summary = _full_summary()
        restored = deserialize_summary(serialize_summary(summary))
        assert restored.parameters[0].interval_mass == pytest.approx(0.95)

    def test_result_is_frozen(self) -> None:
        restored = deserialize_summary(serialize_summary(_full_summary()))
        with pytest.raises((AttributeError, TypeError)):
            restored.convergence = None  # type: ignore[misc]

    def test_output_is_valid_json(self) -> None:
        s = serialize_summary(_full_summary())
        parsed = json.loads(s)
        assert "schema_version" in parsed
        assert "payload" in parsed
        assert "parameters" in parsed["payload"]
        assert "channel_contributions" in parsed["payload"]
        assert "convergence" in parsed["payload"]

    def test_schema_version_in_envelope(self) -> None:
        s = serialize_summary(_full_summary())
        parsed = json.loads(s)
        assert parsed["schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_incompatible_schema_version_raises(self) -> None:
        bad = json.dumps({"schema_version": 999, "payload": {}})
        with pytest.raises(ValueError, match="summary.json"):
            deserialize_summary(bad)

    def test_missing_schema_version_raises(self) -> None:
        bad = json.dumps({"payload": {}})
        with pytest.raises(ValueError, match="summary.json"):
            deserialize_summary(bad)

    def test_error_message_includes_versions(self) -> None:
        bad = json.dumps({"schema_version": 0, "payload": {}})
        with pytest.raises(ValueError, match=str(SUMMARY_SCHEMA_VERSION)):
            deserialize_summary(bad)


# ---------------------------------------------------------------------------
# manifest.json round-trip and version checks
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    def _write_and_load(self, tmp_path: Path, **overrides: object) -> dict:
        paths = run_paths(tmp_path, "test_run")
        kwargs: dict = dict(
            run_id="test_run",
            run_fingerprint="a" * 32,
            timestamp="2026-04-29T12:00:00+00:00",
            seed=42,
            engine_name="pymc",
            engine_version="5.1.0",
            wanamaker_version="0.0.0",
            skip_validation=False,
            readiness_level="ready",
        )
        kwargs.update(overrides)
        write_manifest(paths, **kwargs)  # type: ignore[arg-type]
        return load_manifest(paths.manifest.read_text())

    def test_round_trip_basic_fields(self, tmp_path: Path) -> None:
        m = self._write_and_load(tmp_path)
        assert m["run_id"] == "test_run"
        assert m["seed"] == 42
        assert m["engine"]["name"] == "pymc"

    def test_schema_version_present(self, tmp_path: Path) -> None:
        m = self._write_and_load(tmp_path)
        assert m["schema_version"] == MANIFEST_SCHEMA_VERSION

    def test_summary_schema_version_present(self, tmp_path: Path) -> None:
        m = self._write_and_load(tmp_path)
        assert m["summary_schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_summary_schema_version_default(self, tmp_path: Path) -> None:
        """Default summary_schema_version matches the module constant."""
        m = self._write_and_load(tmp_path)
        assert m["summary_schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_custom_summary_schema_version(self, tmp_path: Path) -> None:
        m = self._write_and_load(tmp_path, summary_schema_version=99)
        assert m["summary_schema_version"] == 99

    def test_incompatible_schema_version_raises(self, tmp_path: Path) -> None:
        bad = json.dumps({"schema_version": 999, "run_id": "x"})
        with pytest.raises(ValueError, match="Incompatible manifest schema version"):
            load_manifest(bad)

    def test_error_message_includes_expected_version(self, tmp_path: Path) -> None:
        bad = json.dumps({"schema_version": 0})
        with pytest.raises(ValueError, match=str(MANIFEST_SCHEMA_VERSION)):
            load_manifest(bad)

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        bad = json.dumps({"run_id": "x"})
        with pytest.raises(ValueError, match="Incompatible manifest schema version"):
            load_manifest(bad)

    def test_readiness_level_null_preserved(self, tmp_path: Path) -> None:
        m = self._write_and_load(tmp_path, readiness_level=None)
        assert m["readiness_level"] is None

    def test_skip_validation_preserved(self, tmp_path: Path) -> None:
        m = self._write_and_load(tmp_path, skip_validation=True, readiness_level="skipped")
        assert m["skip_validation"] is True


# ---------------------------------------------------------------------------
# TrustCard round-trip
# ---------------------------------------------------------------------------


class TestTrustCardRoundTrip:
    def test_full_round_trip(self) -> None:
        card = _minimal_trust_card()
        restored = deserialize_trust_card(serialize_trust_card(card))
        assert restored == card

    def test_dimensions_preserved(self) -> None:
        card = _minimal_trust_card()
        restored = deserialize_trust_card(serialize_trust_card(card))
        assert len(restored.dimensions) == 2
        assert restored.dimensions[0].name == "convergence"
        assert restored.dimensions[0].status == TrustStatus.PASS

    def test_all_trust_statuses_round_trip(self) -> None:
        for status in TrustStatus:
            card = TrustCard(
                dimensions=[TrustDimension(name="x", status=status, explanation="")]
            )
            restored = deserialize_trust_card(serialize_trust_card(card))
            assert restored.dimensions[0].status == status

    def test_empty_dimensions_round_trip(self) -> None:
        card = TrustCard(dimensions=[])
        restored = deserialize_trust_card(serialize_trust_card(card))
        assert restored.dimensions == []

    def test_result_is_frozen(self) -> None:
        restored = deserialize_trust_card(serialize_trust_card(_minimal_trust_card()))
        with pytest.raises((AttributeError, TypeError)):
            restored.dimensions = []  # type: ignore[misc]

    def test_output_is_valid_json(self) -> None:
        s = serialize_trust_card(_minimal_trust_card())
        parsed = json.loads(s)
        assert "schema_version" in parsed
        assert "payload" in parsed
        assert "dimensions" in parsed["payload"]

    def test_schema_version_in_envelope(self) -> None:
        s = serialize_trust_card(_minimal_trust_card())
        parsed = json.loads(s)
        assert parsed["schema_version"] == TRUST_CARD_SCHEMA_VERSION

    def test_incompatible_schema_version_raises(self) -> None:
        bad = json.dumps({"schema_version": 999, "payload": {"dimensions": []}})
        with pytest.raises(ValueError, match="trust_card.json"):
            deserialize_trust_card(bad)

    def test_missing_schema_version_raises(self) -> None:
        bad = json.dumps({"payload": {"dimensions": []}})
        with pytest.raises(ValueError, match="trust_card.json"):
            deserialize_trust_card(bad)


# ---------------------------------------------------------------------------
# RefreshDiff round-trip
# ---------------------------------------------------------------------------


class TestRefreshDiffRoundTrip:
    def test_full_round_trip(self) -> None:
        diff = _minimal_refresh_diff()
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        assert restored == diff

    def test_run_ids_preserved(self) -> None:
        diff = _minimal_refresh_diff()
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        assert restored.previous_run_id == "abc12345_20240101T000000Z"
        assert restored.current_run_id == "def67890_20240201T000000Z"

    def test_movement_values_preserved(self) -> None:
        diff = _minimal_refresh_diff()
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        m = restored.movements[0]
        assert m.name == "channel.search.roi"
        assert m.previous_mean == pytest.approx(2.1)
        assert m.current_mean == pytest.approx(2.4)

    def test_ci_tuples_restored(self) -> None:
        """JSON arrays must be deserialized back to tuples."""
        diff = _minimal_refresh_diff()
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        m = restored.movements[0]
        assert isinstance(m.previous_ci, tuple)
        assert isinstance(m.current_ci, tuple)
        assert m.previous_ci == pytest.approx((1.6, 2.7))
        assert m.current_ci == pytest.approx((1.9, 3.0))

    def test_empty_movements_round_trip(self) -> None:
        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[],
        )
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        assert restored.movements == []
        assert restored.previous_run_id == "a"

    def test_multiple_movements_round_trip(self) -> None:
        diff = RefreshDiff(
            previous_run_id="p",
            current_run_id="c",
            movements=[
                ParameterMovement(
                    name=f"param.{i}",
                    previous_mean=float(i),
                    current_mean=float(i) + 0.1,
                    previous_ci=(float(i) - 0.5, float(i) + 0.5),
                    current_ci=(float(i) - 0.4, float(i) + 0.6),
                )
                for i in range(5)
            ],
        )
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        assert len(restored.movements) == 5
        assert restored.movements[3].name == "param.3"

    def test_result_is_frozen(self) -> None:
        restored = deserialize_refresh_diff(serialize_refresh_diff(_minimal_refresh_diff()))
        with pytest.raises((AttributeError, TypeError)):
            restored.previous_run_id = "x"  # type: ignore[misc]

    def test_output_is_valid_json(self) -> None:
        s = serialize_refresh_diff(_minimal_refresh_diff())
        parsed = json.loads(s)
        assert "schema_version" in parsed
        assert "payload" in parsed
        assert "previous_run_id" in parsed["payload"]
        assert "movements" in parsed["payload"]

    def test_schema_version_in_envelope(self) -> None:
        s = serialize_refresh_diff(_minimal_refresh_diff())
        parsed = json.loads(s)
        assert parsed["schema_version"] == REFRESH_DIFF_SCHEMA_VERSION

    def test_incompatible_schema_version_raises(self) -> None:
        bad = json.dumps({"schema_version": 999, "payload": {}})
        with pytest.raises(ValueError, match="refresh_diff.json"):
            deserialize_refresh_diff(bad)

    def test_missing_schema_version_raises(self) -> None:
        bad = json.dumps({"payload": {}})
        with pytest.raises(ValueError, match="refresh_diff.json"):
            deserialize_refresh_diff(bad)
