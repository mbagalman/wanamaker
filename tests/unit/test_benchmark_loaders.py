"""Unit tests for benchmark dataset loaders."""

from __future__ import annotations

import pytest

from wanamaker.benchmarks.loaders import load_synthetic_ground_truth


def test_load_synthetic_ground_truth() -> None:
    data, truth = load_synthetic_ground_truth()

    assert len(data) == truth["n_weeks"] == 150
    assert truth["date_column"] in data.columns
    assert truth["target_column"] in data.columns
    assert len(truth["spend_columns"]) == 12
    assert set(truth["spend_columns"]).issubset(data.columns)
    assert set(truth["control_columns"]).issubset(data.columns)
    assert len(truth["channels"]) == 12
    assert len(truth["weekly_contributions"]) == 12
    assert len(truth["top_3_channels"]) == 3
    assert truth["total_media_contribution"] > 0


def test_synthetic_ground_truth_channel_totals_match_weekly_contributions() -> None:
    _, truth = load_synthetic_ground_truth()

    weekly = truth["weekly_contributions"]
    totals = {channel["name"]: channel["total_contribution"] for channel in truth["channels"]}
    for channel, values in weekly.items():
        assert sum(values) == pytest.approx(totals[channel])
