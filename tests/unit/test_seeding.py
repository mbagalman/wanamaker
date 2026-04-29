"""Unit tests for the seeding discipline (NFR-2)."""

from __future__ import annotations

import pytest

from wanamaker.seeding import derive_seed, make_rng


def test_make_rng_is_deterministic() -> None:
    a = make_rng(42).standard_normal(5)
    b = make_rng(42).standard_normal(5)
    assert (a == b).all()


def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        make_rng(-1)


def test_derive_seed_is_deterministic_and_label_sensitive() -> None:
    a1 = derive_seed(42, "posterior_predictive")
    a2 = derive_seed(42, "posterior_predictive")
    b = derive_seed(42, "xgb_crosscheck")
    assert a1 == a2
    assert a1 != b
