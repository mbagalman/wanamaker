"""Unit tests for anchoring weight resolution (FR-4.4)."""

from __future__ import annotations

import pytest

from wanamaker.refresh.anchor import (
    ANCHOR_PRESETS,
    DEFAULT_ANCHOR_STRENGTH,
    resolve_anchor_weight,
)


@pytest.mark.parametrize("preset,expected", list(ANCHOR_PRESETS.items()))
def test_named_presets_resolve(preset: str, expected: float) -> None:
    assert resolve_anchor_weight(preset) == expected


def test_default_preset_is_medium() -> None:
    assert DEFAULT_ANCHOR_STRENGTH == "medium"
    assert resolve_anchor_weight(DEFAULT_ANCHOR_STRENGTH) == 0.3


def test_numeric_override_passes_through() -> None:
    assert resolve_anchor_weight(0.42) == 0.42
    assert resolve_anchor_weight(0.0) == 0.0
    assert resolve_anchor_weight(1.0) == 1.0


def test_unknown_preset_rejected() -> None:
    with pytest.raises(ValueError, match="unknown anchor strength preset"):
        resolve_anchor_weight("aggressive")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
def test_out_of_range_numeric_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_anchor_weight(bad)
