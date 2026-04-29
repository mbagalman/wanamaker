"""Unit tests for the wanamaker fit command and its helpers (issue #10).

Tests that do not require PyMC cover:
- build_model_spec: config → ModelSpec translation
- write_manifest / serialize_summary: artifact helpers
- run fingerprint determinism (same inputs → same fingerprint)
- _analytical_config_hash excludes artifact_dir

Tests that require PyMC are marked @pytest.mark.engine and skipped
unless the engine extra is installed.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest

from wanamaker.artifacts import (
    hash_config,
    make_run_fingerprint,
    run_paths,
    serialize_summary,
    write_manifest,
)
from wanamaker.config import ChannelConfig, DataConfig, RunConfig, WanamakerConfig
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
)
from wanamaker.model.builder import build_model_spec
from wanamaker.model.spec import ChannelSpec, ModelSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(csv_path: Path) -> WanamakerConfig:
    return WanamakerConfig(
        data=DataConfig(
            csv_path=csv_path,
            date_column="week",
            target_column="revenue",
            spend_columns=["search", "tv"],
        ),
        channels=[
            ChannelConfig(name="search", category="paid_search"),
            ChannelConfig(name="tv", category="linear_tv"),
        ],
    )


def _dummy_csv(tmp: Path) -> Path:
    """Write a minimal valid CSV and return its path."""
    p = tmp / "data.csv"
    p.write_text("week,revenue,search,tv\n2020-01-06,100,10,50\n2020-01-13,110,12,55\n")
    return p


# ---------------------------------------------------------------------------
# build_model_spec
# ---------------------------------------------------------------------------


class TestBuildModelSpec:
    def test_channels_map_correctly(self, tmp_path: Path) -> None:
        cfg = _minimal_config(_dummy_csv(tmp_path))
        spec = build_model_spec(cfg)
        assert len(spec.channels) == 2
        assert spec.channels[0].name == "search"
        assert spec.channels[0].category == "paid_search"
        assert spec.channels[1].name == "tv"
        assert spec.channels[1].category == "linear_tv"

    def test_target_and_date_columns(self, tmp_path: Path) -> None:
        cfg = _minimal_config(_dummy_csv(tmp_path))
        spec = build_model_spec(cfg)
        assert spec.target_column == "revenue"
        assert spec.date_column == "week"

    def test_control_columns_forwarded(self, tmp_path: Path) -> None:
        cfg = WanamakerConfig(
            data=DataConfig(
                csv_path=_dummy_csv(tmp_path),
                date_column="week",
                target_column="revenue",
                spend_columns=["search"],
                control_columns=["promo", "holiday"],
            ),
            channels=[ChannelConfig(name="search", category="paid_search")],
        )
        spec = build_model_spec(cfg)
        assert spec.control_columns == ["promo", "holiday"]

    def test_runtime_mode_forwarded(self, tmp_path: Path) -> None:
        cfg = WanamakerConfig(
            data=DataConfig(
                csv_path=_dummy_csv(tmp_path),
                date_column="week",
                target_column="revenue",
            ),
            channels=[ChannelConfig(name="search", category="paid_search")],
            run=RunConfig(seed=0, runtime_mode="quick"),
        )
        spec = build_model_spec(cfg)
        assert spec.runtime_mode == "quick"

    def test_channel_priors_resolved_from_taxonomy(self, tmp_path: Path) -> None:
        cfg = _minimal_config(_dummy_csv(tmp_path))
        spec = build_model_spec(cfg)
        assert "search" in spec.channel_priors
        assert "tv" in spec.channel_priors
        import math
        assert math.exp(spec.channel_priors["tv"].half_life_mu) == pytest.approx(6.0, rel=1e-6)

    def test_adstock_family_preserved(self, tmp_path: Path) -> None:
        cfg = WanamakerConfig(
            data=DataConfig(
                csv_path=_dummy_csv(tmp_path),
                date_column="week",
                target_column="revenue",
            ),
            channels=[
                ChannelConfig(name="tv", category="linear_tv", adstock_family="weibull"),
            ],
        )
        spec = build_model_spec(cfg)
        assert spec.channels[0].adstock_family == "weibull"

    def test_unknown_category_raises(self, tmp_path: Path) -> None:
        cfg = WanamakerConfig(
            data=DataConfig(
                csv_path=_dummy_csv(tmp_path),
                date_column="week",
                target_column="revenue",
            ),
            channels=[ChannelConfig(name="ooh", category="out_of_home")],
        )
        with pytest.raises(ValueError, match="Unknown channel categories"):
            build_model_spec(cfg)

    def test_returns_frozen_modelspec(self, tmp_path: Path) -> None:
        cfg = _minimal_config(_dummy_csv(tmp_path))
        spec = build_model_spec(cfg)
        assert isinstance(spec, ModelSpec)
        with pytest.raises((AttributeError, TypeError)):
            spec.target_column = "other"  # type: ignore[misc]

    def test_empty_channels_produces_empty_spec(self, tmp_path: Path) -> None:
        cfg = WanamakerConfig(
            data=DataConfig(
                csv_path=_dummy_csv(tmp_path),
                date_column="week",
                target_column="revenue",
            ),
            channels=[],
        )
        spec = build_model_spec(cfg)
        assert spec.channels == []
        assert spec.channel_priors == {}


# ---------------------------------------------------------------------------
# run_fingerprint determinism
# ---------------------------------------------------------------------------


class TestRunFingerprint:
    def _fp(self, seed: int = 0, data_hash: str = "abc") -> str:
        return make_run_fingerprint(
            data_hash=data_hash,
            config_hash="def",
            package_version="0.0.0",
            engine_name="pymc",
            engine_version="5.0.0",
            seed=seed,
        )

    def test_same_inputs_same_fingerprint(self) -> None:
        assert self._fp() == self._fp()

    def test_different_seed_different_fingerprint(self) -> None:
        assert self._fp(seed=0) != self._fp(seed=1)

    def test_different_data_hash_different_fingerprint(self) -> None:
        assert self._fp(data_hash="abc") != self._fp(data_hash="xyz")

    def test_fingerprint_is_32_hex_chars(self) -> None:
        fp = self._fp()
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_manifest_written_and_parseable(self, tmp_path: Path) -> None:
        paths = run_paths(tmp_path, "test_run_001")
        write_manifest(
            paths,
            run_id="test_run_001",
            run_fingerprint="abcd1234" * 4,
            timestamp="2026-04-29T12:00:00+00:00",
            seed=42,
            engine_name="pymc",
            engine_version="5.1.0",
            wanamaker_version="0.0.0",
            skip_validation=False,
            readiness_level="ready",
        )
        manifest = json.loads(paths.manifest.read_text())
        assert manifest["run_id"] == "test_run_001"
        assert manifest["seed"] == 42
        assert manifest["engine"]["name"] == "pymc"
        assert manifest["engine"]["version"] == "5.1.0"
        assert manifest["skip_validation"] is False
        assert manifest["readiness_level"] == "ready"
        assert manifest["schema_version"] == 1

    def test_skip_validation_recorded(self, tmp_path: Path) -> None:
        paths = run_paths(tmp_path, "skip_run")
        write_manifest(
            paths,
            run_id="skip_run",
            run_fingerprint="a" * 32,
            timestamp="2026-04-29T12:00:00+00:00",
            seed=0,
            engine_name="pymc",
            engine_version="5.0.0",
            wanamaker_version="0.0.0",
            skip_validation=True,
            readiness_level="skipped",
        )
        manifest = json.loads(paths.manifest.read_text())
        assert manifest["skip_validation"] is True
        assert manifest["readiness_level"] == "skipped"

    def test_null_readiness_level_written(self, tmp_path: Path) -> None:
        paths = run_paths(tmp_path, "null_run")
        write_manifest(
            paths,
            run_id="null_run",
            run_fingerprint="b" * 32,
            timestamp="2026-04-29T12:00:00+00:00",
            seed=0,
            engine_name="pymc",
            engine_version="5.0.0",
            wanamaker_version="0.0.0",
            skip_validation=False,
            readiness_level=None,
        )
        manifest = json.loads(paths.manifest.read_text())
        assert manifest["readiness_level"] is None


# ---------------------------------------------------------------------------
# serialize_summary
# ---------------------------------------------------------------------------


class TestSerializeSummary:
    def _minimal_summary(self) -> PosteriorSummary:
        return PosteriorSummary(
            parameters=[
                ParameterSummary(
                    name="channel.search.half_life",
                    mean=1.2,
                    sd=0.3,
                    hdi_low=0.8,
                    hdi_high=1.8,
                )
            ],
            channel_contributions=[
                ChannelContributionSummary(
                    channel="search",
                    mean_contribution=5000.0,
                    hdi_low=3000.0,
                    hdi_high=7000.0,
                )
            ],
            convergence=ConvergenceSummary(
                max_r_hat=1.01,
                min_ess_bulk=450.0,
                n_divergences=0,
                n_chains=4,
                n_draws=1000,
            ),
        )

    def test_returns_valid_json(self) -> None:
        summary = self._minimal_summary()
        s = serialize_summary(summary)
        parsed = json.loads(s)
        assert "parameters" in parsed
        assert "channel_contributions" in parsed

    def test_parameters_serialized(self) -> None:
        summary = self._minimal_summary()
        parsed = json.loads(serialize_summary(summary))
        assert parsed["parameters"][0]["name"] == "channel.search.half_life"
        assert parsed["parameters"][0]["mean"] == pytest.approx(1.2)

    def test_channel_contributions_serialized(self) -> None:
        summary = self._minimal_summary()
        parsed = json.loads(serialize_summary(summary))
        assert parsed["channel_contributions"][0]["channel"] == "search"

    def test_convergence_serialized(self) -> None:
        summary = self._minimal_summary()
        parsed = json.loads(serialize_summary(summary))
        assert parsed["convergence"]["n_divergences"] == 0
        assert parsed["convergence"]["n_chains"] == 4

    def test_none_predictive_serialized_as_null(self) -> None:
        summary = self._minimal_summary()
        assert summary.in_sample_predictive is None
        parsed = json.loads(serialize_summary(summary))
        assert parsed["in_sample_predictive"] is None

    def test_round_trip_preserves_interval_mass(self) -> None:
        summary = self._minimal_summary()
        parsed = json.loads(serialize_summary(summary))
        assert parsed["parameters"][0]["interval_mass"] == pytest.approx(0.95)

    def test_empty_summary_serialized(self) -> None:
        summary = PosteriorSummary()
        parsed = json.loads(serialize_summary(summary))
        assert parsed["parameters"] == []
        assert parsed["channel_contributions"] == []
        assert parsed["convergence"] is None


# ---------------------------------------------------------------------------
# hash_config excludes artifact_dir
# ---------------------------------------------------------------------------


class TestAnalyticalConfigHash:
    def test_same_config_same_hash(self, tmp_path: Path) -> None:
        cfg = _minimal_config(_dummy_csv(tmp_path))
        d1 = cfg.model_dump()
        d1.get("run", {}).pop("artifact_dir", None)
        d2 = cfg.model_dump()
        d2.get("run", {}).pop("artifact_dir", None)
        assert hash_config(d1) == hash_config(d2)

    def test_different_seed_different_hash(self, tmp_path: Path) -> None:
        csv = _dummy_csv(tmp_path)
        cfg1 = WanamakerConfig(
            data=DataConfig(csv_path=csv, date_column="week", target_column="revenue"),
            channels=[ChannelConfig(name="search", category="paid_search")],
            run=RunConfig(seed=0),
        )
        cfg2 = WanamakerConfig(
            data=DataConfig(csv_path=csv, date_column="week", target_column="revenue"),
            channels=[ChannelConfig(name="search", category="paid_search")],
            run=RunConfig(seed=99),
        )
        def _hash(cfg: WanamakerConfig) -> str:
            d = cfg.model_dump()
            d.get("run", {}).pop("artifact_dir", None)
            return hash_config(d)
        assert _hash(cfg1) != _hash(cfg2)

    def test_different_artifact_dir_same_hash(self, tmp_path: Path) -> None:
        """Changing only artifact_dir must not change the analytical hash."""
        csv = _dummy_csv(tmp_path)
        cfg1 = WanamakerConfig(
            data=DataConfig(csv_path=csv, date_column="week", target_column="revenue"),
            channels=[ChannelConfig(name="search", category="paid_search")],
            run=RunConfig(seed=0, artifact_dir=Path(".wanamaker")),
        )
        cfg2 = WanamakerConfig(
            data=DataConfig(csv_path=csv, date_column="week", target_column="revenue"),
            channels=[ChannelConfig(name="search", category="paid_search")],
            run=RunConfig(seed=0, artifact_dir=Path("/tmp/other_dir")),
        )
        def _hash(cfg: WanamakerConfig) -> str:
            d = cfg.model_dump()
            d.get("run", {}).pop("artifact_dir", None)
            return hash_config(d)
        assert _hash(cfg1) == _hash(cfg2)
