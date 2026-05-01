"""Generate the five diagnostic benchmark datasets (issue #25).

Each dataset is engineered to trigger a specific check in
``wanamaker.diagnose.checks``. They are deliberately small — 80 weekly
rows is enough to surface every check while keeping the committed CSVs
under a few KB each.

Outputs:
    benchmark_data/low_variation_channel.csv (+ metadata)
    benchmark_data/collinearity.csv (+ metadata)
    benchmark_data/lift_test_calibration.csv
    benchmark_data/lift_test_calibration_lift_tests.csv
    benchmark_data/lift_test_calibration_metadata.json
    benchmark_data/target_leakage.csv (+ metadata)
    benchmark_data/structural_break.csv (+ metadata)

Run from the repository root:

    python benchmark_data/generate_diagnostic_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "week"
TARGET_COLUMN = "revenue"
N_WEEKS = 80
START_DATE = "2024-01-01"

_OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Generate all five diagnostic benchmark datasets."""
    _generate_low_variation()
    _generate_collinearity()
    _generate_lift_test_calibration()
    _generate_target_leakage()
    _generate_structural_break()


# ---------------------------------------------------------------------------
# 1. Low-variation channel — triggers FR-2.2 spend_variation + FR-3.2
# ---------------------------------------------------------------------------


def _generate_low_variation() -> None:
    rng = np.random.default_rng(seed=20260501)
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    paid_search = rng.uniform(800.0, 1500.0, size=N_WEEKS)
    paid_social = rng.uniform(600.0, 1200.0, size=N_WEEKS)
    linear_tv = rng.uniform(2500.0, 4500.0, size=N_WEEKS)
    # The invariant channel: tiny jitter so CV stays well below 0.10.
    static_display = 5000.0 + rng.normal(0.0, 5.0, size=N_WEEKS)

    revenue = (
        50000.0
        + 1.5 * paid_search
        + 0.8 * paid_social
        + 0.4 * linear_tv
        + 0.0 * static_display  # truly invariant: spend has no marginal effect
        + rng.normal(0.0, 1500.0, size=N_WEEKS)
    )

    df = pd.DataFrame(
        {
            DATE_COLUMN: weeks,
            TARGET_COLUMN: np.round(revenue, 2),
            "paid_search": np.round(paid_search, 2),
            "paid_social": np.round(paid_social, 2),
            "linear_tv": np.round(linear_tv, 2),
            "static_display": np.round(static_display, 2),
        }
    )
    df.to_csv(_OUTPUT_DIR / "low_variation_channel.csv", index=False)

    cv_static = float(df["static_display"].std(ddof=0) / abs(df["static_display"].mean()))
    _write_json(
        _OUTPUT_DIR / "low_variation_channel_metadata.json",
        {
            "dataset": "low_variation_channel",
            "purpose": (
                "Exercises FR-2.2 spend_variation and FR-3.2 spend-invariant "
                "saturation handling. The 'static_display' channel has CV well "
                "below the 0.10 default threshold."
            ),
            "n_weeks": N_WEEKS,
            "date_column": DATE_COLUMN,
            "target_column": TARGET_COLUMN,
            "spend_columns": [
                "paid_search", "paid_social", "linear_tv", "static_display",
            ],
            "control_columns": [],
            "low_variation_channel": "static_display",
            "low_variation_cv": round(cv_static, 6),
            "expected_check_warning": "spend_variation",
        },
    )


# ---------------------------------------------------------------------------
# 2. Collinearity — two paid channels with r > 0.95
# ---------------------------------------------------------------------------


def _generate_collinearity() -> None:
    rng = np.random.default_rng(seed=20260502)
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    paid_search = rng.uniform(800.0, 1500.0, size=N_WEEKS)
    linear_tv = rng.uniform(2500.0, 4500.0, size=N_WEEKS)
    # display_a is independent.
    display_a = rng.uniform(1500.0, 2800.0, size=N_WEEKS)
    # display_b tracks display_a almost perfectly (small noise → r > 0.95).
    display_b = display_a * 1.05 + rng.normal(0.0, 30.0, size=N_WEEKS)

    revenue = (
        50000.0
        + 1.5 * paid_search
        + 0.4 * linear_tv
        + 0.6 * display_a
        + 0.5 * display_b
        + rng.normal(0.0, 1500.0, size=N_WEEKS)
    )

    df = pd.DataFrame(
        {
            DATE_COLUMN: weeks,
            TARGET_COLUMN: np.round(revenue, 2),
            "paid_search": np.round(paid_search, 2),
            "linear_tv": np.round(linear_tv, 2),
            "display_a": np.round(display_a, 2),
            "display_b": np.round(display_b, 2),
        }
    )
    df.to_csv(_OUTPUT_DIR / "collinearity.csv", index=False)

    correlation = float(df[["display_a", "display_b"]].corr().iloc[0, 1])
    _write_json(
        _OUTPUT_DIR / "collinearity_metadata.json",
        {
            "dataset": "collinearity",
            "purpose": (
                "Exercises FR-2.2 collinearity. 'display_a' and 'display_b' are "
                "designed with absolute correlation > 0.95; the diagnostic must "
                "warn about the pair."
            ),
            "n_weeks": N_WEEKS,
            "date_column": DATE_COLUMN,
            "target_column": TARGET_COLUMN,
            "spend_columns": [
                "paid_search", "linear_tv", "display_a", "display_b",
            ],
            "control_columns": [],
            "collinear_pair": ["display_a", "display_b"],
            "pair_correlation": round(correlation, 6),
            "expected_check_warning": "collinearity",
        },
    )


# ---------------------------------------------------------------------------
# 3. Lift-test calibration — known-true ROI for one channel
# ---------------------------------------------------------------------------


def _generate_lift_test_calibration() -> None:
    rng = np.random.default_rng(seed=20260503)
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    # One channel ('paid_search') has a known true ROI. The lift test
    # measures it within a tight CI; the model should anchor near it.
    true_search_roi = 1.8
    paid_search = rng.uniform(800.0, 1500.0, size=N_WEEKS)
    linear_tv = rng.uniform(2500.0, 4500.0, size=N_WEEKS)
    paid_social = rng.uniform(600.0, 1200.0, size=N_WEEKS)

    revenue = (
        50000.0
        + true_search_roi * paid_search
        + 0.4 * linear_tv
        + 0.7 * paid_social
        + rng.normal(0.0, 1500.0, size=N_WEEKS)
    )

    df = pd.DataFrame(
        {
            DATE_COLUMN: weeks,
            TARGET_COLUMN: np.round(revenue, 2),
            "paid_search": np.round(paid_search, 2),
            "linear_tv": np.round(linear_tv, 2),
            "paid_social": np.round(paid_social, 2),
        }
    )
    df.to_csv(_OUTPUT_DIR / "lift_test_calibration.csv", index=False)

    # Simulated lift test: 95 % CI roughly centred on the true ROI.
    lift_low = round(true_search_roi - 0.20, 2)
    lift_high = round(true_search_roi + 0.20, 2)
    lift_tests = pd.DataFrame(
        {
            "channel": ["paid_search"],
            "test_start": ["2023-09-01"],
            "test_end": ["2023-12-15"],
            "roi_estimate": [round(true_search_roi, 6)],
            "roi_ci_lower": [lift_low],
            "roi_ci_upper": [lift_high],
        }
    )
    lift_tests.to_csv(
        _OUTPUT_DIR / "lift_test_calibration_lift_tests.csv", index=False
    )

    _write_json(
        _OUTPUT_DIR / "lift_test_calibration_metadata.json",
        {
            "dataset": "lift_test_calibration",
            "purpose": (
                "Exercises FR-1.3 (lift-test prior wiring) and FR-3.3 (recovered "
                "ROI close to known lift). The 'paid_search' channel was simulated "
                "with a known true ROI; the companion lift-test CSV reports a 95 % "
                "CI centred on that value."
            ),
            "n_weeks": N_WEEKS,
            "date_column": DATE_COLUMN,
            "target_column": TARGET_COLUMN,
            "spend_columns": ["paid_search", "linear_tv", "paid_social"],
            "control_columns": [],
            "lift_test_channel": "paid_search",
            "true_roi": true_search_roi,
            "roi_estimate": round(true_search_roi, 6),
            "roi_ci_lower": lift_low,
            "roi_ci_upper": lift_high,
            "lift_tests_csv": "lift_test_calibration_lift_tests.csv",
        },
    )


# ---------------------------------------------------------------------------
# 4. Target leakage — control derived from target
# ---------------------------------------------------------------------------


def _generate_target_leakage() -> None:
    rng = np.random.default_rng(seed=20260504)
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    paid_search = rng.uniform(800.0, 1500.0, size=N_WEEKS)
    linear_tv = rng.uniform(2500.0, 4500.0, size=N_WEEKS)

    revenue = (
        50000.0
        + 1.5 * paid_search
        + 0.4 * linear_tv
        + rng.normal(0.0, 1500.0, size=N_WEEKS)
    )
    # Leakage: 'derived_index' is a near-perfect transform of revenue with
    # only tiny noise, which the target_leakage check catches at r >= 0.95.
    derived_index = revenue / 1000.0 + rng.normal(0.0, 0.5, size=N_WEEKS)

    df = pd.DataFrame(
        {
            DATE_COLUMN: weeks,
            TARGET_COLUMN: np.round(revenue, 2),
            "paid_search": np.round(paid_search, 2),
            "linear_tv": np.round(linear_tv, 2),
            "derived_index": np.round(derived_index, 6),
            "promo_flag": (rng.random(N_WEEKS) > 0.7).astype(int),
        }
    )
    df.to_csv(_OUTPUT_DIR / "target_leakage.csv", index=False)

    correlation = float(df[[TARGET_COLUMN, "derived_index"]].corr().iloc[0, 1])
    _write_json(
        _OUTPUT_DIR / "target_leakage_metadata.json",
        {
            "dataset": "target_leakage",
            "purpose": (
                "Exercises FR-2.2 target_leakage. 'derived_index' is constructed "
                "as a noisy linear transform of revenue, producing |corr| >= 0.95 "
                "so the check fires."
            ),
            "n_weeks": N_WEEKS,
            "date_column": DATE_COLUMN,
            "target_column": TARGET_COLUMN,
            "spend_columns": ["paid_search", "linear_tv"],
            "control_columns": ["derived_index", "promo_flag"],
            "leakage_control": "derived_index",
            "leakage_correlation": round(correlation, 6),
            "expected_check_warning": "target_leakage",
        },
    )


# ---------------------------------------------------------------------------
# 5. Structural break — known step change in baseline
# ---------------------------------------------------------------------------


def _generate_structural_break() -> None:
    rng = np.random.default_rng(seed=20260505)
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    paid_search = rng.uniform(800.0, 1500.0, size=N_WEEKS)
    linear_tv = rng.uniform(2500.0, 4500.0, size=N_WEEKS)

    break_index = 40
    pre_baseline = 50000.0
    post_baseline = 80000.0
    baseline = np.where(np.arange(N_WEEKS) < break_index, pre_baseline, post_baseline)

    revenue = (
        baseline
        + 1.5 * paid_search
        + 0.4 * linear_tv
        + rng.normal(0.0, 1500.0, size=N_WEEKS)
    )

    df = pd.DataFrame(
        {
            DATE_COLUMN: weeks,
            TARGET_COLUMN: np.round(revenue, 2),
            "paid_search": np.round(paid_search, 2),
            "linear_tv": np.round(linear_tv, 2),
        }
    )
    df.to_csv(_OUTPUT_DIR / "structural_break.csv", index=False)

    _write_json(
        _OUTPUT_DIR / "structural_break_metadata.json",
        {
            "dataset": "structural_break",
            "purpose": (
                "Exercises FR-2.2 structural_breaks. A deliberate step change in "
                "baseline at week index 40 (60 % jump) is large enough for the "
                "binary-segmentation scan to detect."
            ),
            "n_weeks": N_WEEKS,
            "date_column": DATE_COLUMN,
            "target_column": TARGET_COLUMN,
            "spend_columns": ["paid_search", "linear_tv"],
            "control_columns": [],
            "break_index": break_index,
            "break_date": str(weeks[break_index].date()),
            "pre_baseline": pre_baseline,
            "post_baseline": post_baseline,
            "expected_check_warning": "structural_breaks",
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
