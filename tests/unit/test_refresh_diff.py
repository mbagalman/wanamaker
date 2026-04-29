"""Unit tests for refresh diff computation and movement classification (issue #17).

Tests cover:
- classify_movement: each class produced under the right conditions
- unexplained_fraction: edge cases and non-trivial inputs
- compute_diff: parameter matching, channel contribution naming, skipping
  parameters absent from one summary, movement_class assignment
- Serialization round-trip preserves movement_class
"""

from __future__ import annotations

import json

import pytest

from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
)
from wanamaker.refresh.classify import (
    WEAKLY_IDENTIFIED_THRESHOLD,
    MovementClass,
    classify_movement,
    unexplained_fraction,
)
from wanamaker.refresh.diff import ParameterMovement, RefreshDiff, compute_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param(name: str, mean: float, low: float, high: float) -> ParameterSummary:
    return ParameterSummary(name=name, mean=mean, sd=0.1, hdi_low=low, hdi_high=high)


def _contrib(channel: str, mean: float, low: float, high: float) -> ChannelContributionSummary:
    return ChannelContributionSummary(
        channel=channel, mean_contribution=mean, hdi_low=low, hdi_high=high
    )


def _summary(
    params: list[ParameterSummary] | None = None,
    contribs: list[ChannelContributionSummary] | None = None,
) -> PosteriorSummary:
    return PosteriorSummary(
        parameters=params or [],
        channel_contributions=contribs or [],
        convergence=ConvergenceSummary(
            max_r_hat=1.005, min_ess_bulk=500.0, n_divergences=0, n_chains=4, n_draws=1000
        ),
    )


# ---------------------------------------------------------------------------
# classify_movement
# ---------------------------------------------------------------------------


class TestClassifyMovement:
    def test_within_prior_ci_when_mean_inside_previous_hdi(self) -> None:
        # curr_mean = 2.0, previous HDI = (1.5, 2.5) → inside
        assert (
            classify_movement((1.5, 2.5), 2.0, (1.8, 2.2))
            == MovementClass.WITHIN_PRIOR_CI
        )

    def test_within_prior_ci_at_lower_bound(self) -> None:
        assert (
            classify_movement((1.5, 2.5), 1.5, (1.4, 1.6))
            == MovementClass.WITHIN_PRIOR_CI
        )

    def test_within_prior_ci_at_upper_bound(self) -> None:
        assert (
            classify_movement((1.5, 2.5), 2.5, (2.4, 2.6))
            == MovementClass.WITHIN_PRIOR_CI
        )

    def test_unexplained_when_mean_outside_previous_hdi(self) -> None:
        # curr_mean = 5.0, far above previous HDI (1.5, 2.5), CI is tight
        assert (
            classify_movement((1.5, 2.5), 5.0, (4.8, 5.2))
            == MovementClass.UNEXPLAINED
        )

    def test_unexplained_below_previous_hdi(self) -> None:
        assert (
            classify_movement((3.0, 5.0), 1.0, (0.9, 1.1))
            == MovementClass.UNEXPLAINED
        )

    def test_weakly_identified_takes_priority_over_unexplained(self) -> None:
        # CI width = 8.0, |mean| = 2.0 → ratio = 4.0 > WEAKLY_IDENTIFIED_THRESHOLD
        # Even though mean is outside previous HDI, WEAKLY_IDENTIFIED wins
        assert (
            classify_movement((1.0, 2.0), 5.0, (1.0, 9.0))
            == MovementClass.WEAKLY_IDENTIFIED
        )

    def test_weakly_identified_takes_priority_over_within_ci(self) -> None:
        # Mean IS within previous CI but the CI is huge → WEAKLY_IDENTIFIED
        assert (
            classify_movement((0.0, 10.0), 1.0, (0.0, 8.0))
            == MovementClass.WEAKLY_IDENTIFIED
        )

    def test_not_weakly_identified_when_mean_is_zero(self) -> None:
        # Avoid division by zero; mean=0, wide CI → falls through to WITHIN_PRIOR_CI
        result = classify_movement((0.0, 2.0), 0.0, (0.0, 100.0))
        assert result != MovementClass.WEAKLY_IDENTIFIED

    def test_threshold_boundary_exactly_one(self) -> None:
        # CI width / |mean| = exactly 1.0 → NOT weakly identified (threshold is strict >)
        # mean = 2.0, CI = (1.0, 3.0), width = 2.0, ratio = 1.0
        result = classify_movement((1.5, 2.5), 2.0, (1.0, 3.0))
        assert result != MovementClass.WEAKLY_IDENTIFIED

    def test_just_above_threshold_is_weakly_identified(self) -> None:
        # mean = 2.0, CI = (0.9, 3.1), width = 2.2, ratio = 1.1 > 1.0
        assert (
            classify_movement((1.5, 2.5), 2.0, (0.9, 3.1))
            == MovementClass.WEAKLY_IDENTIFIED
        )


# ---------------------------------------------------------------------------
# unexplained_fraction
# ---------------------------------------------------------------------------


class TestUnexplainedFraction:
    def _movement(self, cls: MovementClass) -> ParameterMovement:
        return ParameterMovement(
            name="x", previous_mean=1.0, current_mean=1.0,
            previous_ci=(0.5, 1.5), current_ci=(0.5, 1.5),
            movement_class=cls,
        )

    def test_empty_list_returns_zero(self) -> None:
        assert unexplained_fraction([]) == 0.0

    def test_all_unexplained(self) -> None:
        ms = [self._movement(MovementClass.UNEXPLAINED)] * 3
        assert unexplained_fraction(ms) == pytest.approx(1.0)

    def test_none_unexplained(self) -> None:
        ms = [self._movement(MovementClass.WITHIN_PRIOR_CI)] * 4
        assert unexplained_fraction(ms) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        ms = [
            self._movement(MovementClass.UNEXPLAINED),
            self._movement(MovementClass.WITHIN_PRIOR_CI),
            self._movement(MovementClass.UNEXPLAINED),
            self._movement(MovementClass.WEAKLY_IDENTIFIED),
        ]
        assert unexplained_fraction(ms) == pytest.approx(0.5)

    def test_none_movement_class_not_counted(self) -> None:
        m = ParameterMovement(
            name="x", previous_mean=1.0, current_mean=1.0,
            previous_ci=(0.5, 1.5), current_ci=(0.5, 1.5),
            movement_class=None,
        )
        assert unexplained_fraction([m]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_diff — parameter movements
# ---------------------------------------------------------------------------


class TestComputeDiffParameters:
    def test_within_ci_parameter(self) -> None:
        prev = _summary([_param("roi.search", mean=2.0, low=1.5, high=2.5)])
        curr = _summary([_param("roi.search", mean=2.1, low=1.8, high=2.4)])
        diff = compute_diff(prev, curr, "run_a", "run_b")
        assert len(diff.movements) == 1
        m = diff.movements[0]
        assert m.name == "roi.search"
        assert m.previous_mean == pytest.approx(2.0)
        assert m.current_mean == pytest.approx(2.1)
        assert m.movement_class == MovementClass.WITHIN_PRIOR_CI

    def test_unexplained_parameter(self) -> None:
        prev = _summary([_param("roi.search", mean=2.0, low=1.5, high=2.5)])
        # Current mean well outside previous HDI, tight CI
        curr = _summary([_param("roi.search", mean=5.0, low=4.8, high=5.2)])
        diff = compute_diff(prev, curr, "run_a", "run_b")
        assert diff.movements[0].movement_class == MovementClass.UNEXPLAINED

    def test_weakly_identified_parameter(self) -> None:
        prev = _summary([_param("roi.search", mean=2.0, low=1.5, high=2.5)])
        # Very wide CI relative to mean
        curr = _summary([_param("roi.search", mean=2.0, low=0.1, high=10.0)])
        diff = compute_diff(prev, curr, "run_a", "run_b")
        assert diff.movements[0].movement_class == MovementClass.WEAKLY_IDENTIFIED

    def test_parameter_absent_from_previous_skipped(self) -> None:
        prev = _summary([_param("roi.search", mean=2.0, low=1.5, high=2.5)])
        curr = _summary([
            _param("roi.search", mean=2.1, low=1.8, high=2.4),
            _param("roi.tv", mean=1.5, low=1.0, high=2.0),  # new channel
        ])
        diff = compute_diff(prev, curr, "run_a", "run_b")
        names = [m.name for m in diff.movements]
        assert "roi.search" in names
        assert "roi.tv" not in names

    def test_parameter_absent_from_current_skipped(self) -> None:
        prev = _summary([
            _param("roi.search", mean=2.0, low=1.5, high=2.5),
            _param("roi.tv", mean=1.5, low=1.0, high=2.0),
        ])
        curr = _summary([_param("roi.search", mean=2.1, low=1.8, high=2.4)])
        diff = compute_diff(prev, curr, "run_a", "run_b")
        assert len(diff.movements) == 1
        assert diff.movements[0].name == "roi.search"

    def test_multiple_parameters(self) -> None:
        params = [
            _param(f"p{i}", mean=float(i), low=float(i) - 0.5, high=float(i) + 0.5)
            for i in range(5)
        ]
        prev = _summary(params)
        curr = _summary(params)
        diff = compute_diff(prev, curr, "run_a", "run_b")
        assert len(diff.movements) == 5

    def test_run_ids_preserved(self) -> None:
        prev = _summary([_param("x", 1.0, 0.5, 1.5)])
        curr = _summary([_param("x", 1.0, 0.5, 1.5)])
        diff = compute_diff(prev, curr, "prev_run", "curr_run")
        assert diff.previous_run_id == "prev_run"
        assert diff.current_run_id == "curr_run"

    def test_empty_summaries(self) -> None:
        diff = compute_diff(_summary(), _summary(), "a", "b")
        assert diff.movements == []

    def test_ci_values_stored(self) -> None:
        prev = _summary([_param("x", 2.0, 1.5, 2.5)])
        curr = _summary([_param("x", 2.1, 1.9, 2.3)])
        diff = compute_diff(prev, curr, "a", "b")
        m = diff.movements[0]
        assert m.previous_ci == pytest.approx((1.5, 2.5))
        assert m.current_ci == pytest.approx((1.9, 2.3))


# ---------------------------------------------------------------------------
# compute_diff — channel contribution movements
# ---------------------------------------------------------------------------


class TestComputeDiffContributions:
    def test_contribution_named_with_prefix(self) -> None:
        prev = _summary(contribs=[_contrib("search", 5000.0, 3000.0, 7000.0)])
        curr = _summary(contribs=[_contrib("search", 5100.0, 3200.0, 7100.0)])
        diff = compute_diff(prev, curr, "a", "b")
        assert len(diff.movements) == 1
        assert diff.movements[0].name == "channel.search.contribution"

    def test_contribution_within_ci(self) -> None:
        prev = _summary(contribs=[_contrib("tv", 10000.0, 8000.0, 12000.0)])
        curr = _summary(contribs=[_contrib("tv", 10200.0, 9000.0, 11000.0)])
        diff = compute_diff(prev, curr, "a", "b")
        assert diff.movements[0].movement_class == MovementClass.WITHIN_PRIOR_CI

    def test_contribution_unexplained(self) -> None:
        prev = _summary(contribs=[_contrib("tv", 5000.0, 4500.0, 5500.0)])
        curr = _summary(contribs=[_contrib("tv", 9000.0, 8800.0, 9200.0)])
        diff = compute_diff(prev, curr, "a", "b")
        assert diff.movements[0].movement_class == MovementClass.UNEXPLAINED

    def test_new_channel_skipped(self) -> None:
        prev = _summary(contribs=[_contrib("search", 5000.0, 3000.0, 7000.0)])
        curr = _summary(contribs=[
            _contrib("search", 5100.0, 3200.0, 7100.0),
            _contrib("new_channel", 1000.0, 500.0, 1500.0),
        ])
        diff = compute_diff(prev, curr, "a", "b")
        names = [m.name for m in diff.movements]
        assert "channel.new_channel.contribution" not in names


# ---------------------------------------------------------------------------
# compute_diff — parameters and contributions together
# ---------------------------------------------------------------------------


class TestComputeDiffMixed:
    def test_params_and_contribs_both_included(self) -> None:
        prev = _summary(
            params=[_param("roi.search", 2.0, 1.5, 2.5)],
            contribs=[_contrib("search", 5000.0, 3000.0, 7000.0)],
        )
        curr = _summary(
            params=[_param("roi.search", 2.1, 1.9, 2.3)],
            contribs=[_contrib("search", 5100.0, 3200.0, 7100.0)],
        )
        diff = compute_diff(prev, curr, "a", "b")
        names = [m.name for m in diff.movements]
        assert "roi.search" in names
        assert "channel.search.contribution" in names

    def test_result_is_frozen(self) -> None:
        prev = _summary([_param("x", 1.0, 0.5, 1.5)])
        curr = _summary([_param("x", 1.0, 0.5, 1.5)])
        diff = compute_diff(prev, curr, "a", "b")
        with pytest.raises((AttributeError, TypeError)):
            diff.previous_run_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Serialization round-trip preserves movement_class
# ---------------------------------------------------------------------------


class TestMovementClassRoundTrip:
    def test_round_trip_preserves_movement_class(self) -> None:
        from wanamaker.artifacts import deserialize_refresh_diff, serialize_refresh_diff

        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[
                ParameterMovement(
                    name="roi.search",
                    previous_mean=2.0,
                    current_mean=2.1,
                    previous_ci=(1.5, 2.5),
                    current_ci=(1.9, 2.3),
                    movement_class=MovementClass.WITHIN_PRIOR_CI,
                ),
                ParameterMovement(
                    name="roi.tv",
                    previous_mean=1.5,
                    current_mean=5.0,
                    previous_ci=(1.0, 2.0),
                    current_ci=(4.8, 5.2),
                    movement_class=MovementClass.UNEXPLAINED,
                ),
                ParameterMovement(
                    name="roi.ooh",
                    previous_mean=1.0,
                    current_mean=1.0,
                    previous_ci=(0.5, 1.5),
                    current_ci=(0.0, 8.0),
                    movement_class=MovementClass.WEAKLY_IDENTIFIED,
                ),
            ],
        )
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        for orig, rest in zip(diff.movements, restored.movements):
            assert rest.movement_class == orig.movement_class

    def test_none_movement_class_round_trips(self) -> None:
        from wanamaker.artifacts import deserialize_refresh_diff, serialize_refresh_diff

        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[
                ParameterMovement(
                    name="x",
                    previous_mean=1.0,
                    current_mean=1.0,
                    previous_ci=(0.5, 1.5),
                    current_ci=(0.5, 1.5),
                    movement_class=None,
                )
            ],
        )
        restored = deserialize_refresh_diff(serialize_refresh_diff(diff))
        assert restored.movements[0].movement_class is None

    def test_movement_class_in_json(self) -> None:
        from wanamaker.artifacts import serialize_refresh_diff

        diff = RefreshDiff(
            previous_run_id="a",
            current_run_id="b",
            movements=[
                ParameterMovement(
                    name="x",
                    previous_mean=1.0,
                    current_mean=3.0,
                    previous_ci=(0.9, 1.1),
                    current_ci=(2.9, 3.1),
                    movement_class=MovementClass.UNEXPLAINED,
                )
            ],
        )
        payload = json.loads(serialize_refresh_diff(diff))["payload"]
        assert payload["movements"][0]["movement_class"] == "unexplained"
