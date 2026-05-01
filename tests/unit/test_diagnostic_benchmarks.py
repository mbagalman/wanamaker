"""Each diagnostic benchmark dataset triggers its specific check (issue #25).

Per the issue's acceptance criterion: "Diagnostic checks pass their specific
acceptance criteria on their respective datasets (see FR-2.2)." These tests
run each check against its purpose-built dataset and assert that the
expected warning fires — i.e. the dataset really does exercise the
relevant check, not just a generic shape match.
"""

from __future__ import annotations

from wanamaker.benchmarks.loaders import (
    load_collinearity,
    load_lift_test_calibration,
    load_low_variation_channel,
    load_structural_break,
    load_target_leakage,
)
from wanamaker.diagnose.checks import (
    check_collinearity,
    check_spend_variation,
    check_structural_breaks,
    check_target_leakage,
)
from wanamaker.diagnose.readiness import CheckSeverity


class TestLowVariationDatasetTriggersSpendVariation:
    def test_warning_fires_for_invariant_channel(self) -> None:
        df, metadata = load_low_variation_channel()
        results = check_spend_variation(df, metadata["spend_columns"])

        invariant = metadata["low_variation_channel"]
        flagged = [
            r for r in results
            if r.severity == CheckSeverity.WARNING and invariant in r.message
        ]
        assert flagged, (
            f"Expected check_spend_variation to warn about {invariant!r}; "
            f"got results: {[r.message for r in results]}"
        )

    def test_normal_channels_not_flagged(self) -> None:
        df, metadata = load_low_variation_channel()
        results = check_spend_variation(df, metadata["spend_columns"])
        flagged_columns = {
            col for r in results for col in metadata["spend_columns"]
            if col in r.message and r.severity == CheckSeverity.WARNING
        }
        invariant = metadata["low_variation_channel"]
        # The invariant channel is the only one that should be flagged.
        assert flagged_columns == {invariant}


class TestCollinearityDatasetTriggersCollinearityCheck:
    def test_warning_fires_for_designed_pair(self) -> None:
        df, metadata = load_collinearity()
        results = check_collinearity(df, metadata["spend_columns"], control_columns=[])

        left, right = metadata["collinear_pair"]
        flagged = [
            r for r in results
            if r.severity == CheckSeverity.WARNING
            and left in r.message and right in r.message
        ]
        assert flagged, (
            f"Expected check_collinearity to warn about {left!r}/{right!r}; "
            f"got results: {[r.message for r in results]}"
        )


class TestTargetLeakageDatasetTriggersTargetLeakageCheck:
    def test_warning_fires_for_derived_control(self) -> None:
        df, metadata = load_target_leakage()
        results = check_target_leakage(
            df, metadata["target_column"], metadata["control_columns"],
        )

        leakage_col = metadata["leakage_control"]
        flagged = [
            r for r in results
            if r.severity == CheckSeverity.WARNING and leakage_col in r.message
        ]
        assert flagged, (
            f"Expected check_target_leakage to warn about {leakage_col!r}; "
            f"got results: {[r.message for r in results]}"
        )


class TestStructuralBreakDatasetTriggersStructuralBreakCheck:
    def test_at_least_one_break_detected(self) -> None:
        df, metadata = load_structural_break()
        results = check_structural_breaks(
            df, metadata["target_column"], metadata["date_column"],
        )
        assert results, (
            "Expected check_structural_breaks to detect the engineered "
            "step change; got an empty result list."
        )
        assert all(r.severity == CheckSeverity.WARNING for r in results)


class TestLiftTestCalibrationDatasetSchema:
    """Lift-test calibration is wired through ``LiftPrior`` rather than a
    diagnose check, so the unit-level acceptance is structural: the dataset
    advertises the channel under test, and the supplied lift estimate
    matches the true ROI used to simulate the data within the reported CI.
    """

    def test_lift_test_estimate_matches_true_roi(self) -> None:
        _, lift_tests, metadata = load_lift_test_calibration()
        row = lift_tests.iloc[0]
        true_roi = float(metadata["true_roi"])
        # The reported lift estimate is the true ROI by construction;
        # validating they round-trip through the metadata + CSV catches a
        # generator that drifts away from the documented contract.
        assert row["lift_estimate"] == metadata["lift_estimate"]
        assert row["ci_lower"] <= true_roi <= row["ci_upper"]
        assert row["channel"] == metadata["lift_test_channel"]
