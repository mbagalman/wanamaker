"""CSV loading.

The MMM input is a single CSV with one row per time period (default weekly).
See FR-1.1 in the BRD/PRD for the contract.
"""

from __future__ import annotations

from pathlib import Path

import warnings

import pandas as pd

from wanamaker.config import DataConfig

LIFT_TEST_BASE_COLUMNS = {
    "channel",
    "test_start",
    "test_end",
}

LIFT_TEST_ROI_COLUMNS = {
    "roi_estimate",
    "roi_ci_lower",
    "roi_ci_upper",
}

LIFT_TEST_OUTCOME_COLUMNS = {
    "incremental_outcome",
    "incremental_spend",
    "ci_lower_outcome",
    "ci_upper_outcome",
}

LIFT_TEST_LEGACY_COLUMNS = {
    "lift_estimate",
    "ci_lower",
    "ci_upper",
}


def load_input_csv(config: DataConfig) -> pd.DataFrame:
    """Load the input CSV described by ``config``.

    The frame is returned with the date column parsed and sorted ascending.
    No imputation, no dropping — those decisions belong to the diagnostic
    step so that the user sees the raw shape first.

    Args:
        config: Validated data configuration.

    Returns:
        DataFrame indexed by row position, with parsed dates.

    Raises:
        FileNotFoundError: If ``config.csv_path`` does not exist.
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(config.csv_path)
    missing = {config.date_column, config.target_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"input CSV {config.csv_path} is missing required columns: "
            f"{sorted(missing)}"
        )
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    return df.sort_values(config.date_column).reset_index(drop=True)


def load_lift_test_csv(path: Path) -> pd.DataFrame:
    """Load a lift-test calibration CSV (FR-1.3).

    Supported schemas:
    - ROI (recommended): channel, test_start, test_end, roi_estimate,
      roi_ci_lower, roi_ci_upper
    - Outcome: channel, test_start, test_end, incremental_outcome,
      incremental_spend, ci_lower_outcome, ci_upper_outcome
    - Legacy (deprecated): channel, test_start, test_end, lift_estimate,
      ci_lower, ci_upper

    The returned frame converts any schema into the canonical ROI columns,
    parses test dates, converts numeric fields to floats, and rejects
    ambiguous duplicate channel rows or mixed schemas. Conversion from
    confidence intervals to ``LiftPrior`` objects is handled by
    ``wanamaker.model.builder``.

    Args:
        path: Path to the lift-test CSV.

    Returns:
        Validated lift-test results, one row per channel, with canonical
        ROI columns (roi_estimate, roi_ci_lower, roi_ci_upper).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing, mixed schemas are used,
            dates/numbers cannot be parsed, intervals are invalid, or a
            channel appears more than once.
    """
    df = pd.read_csv(path)
    columns = set(df.columns)

    missing_base = LIFT_TEST_BASE_COLUMNS - columns
    if missing_base:
        raise ValueError(
            f"lift-test CSV {path} is missing required base columns: {sorted(missing_base)}"
        )

    has_roi = LIFT_TEST_ROI_COLUMNS.issubset(columns)
    has_outcome = LIFT_TEST_OUTCOME_COLUMNS.issubset(columns)
    has_legacy = LIFT_TEST_LEGACY_COLUMNS.issubset(columns)

    if sum([has_roi, has_outcome, has_legacy]) > 1:
        raise ValueError(
            f"lift-test CSV {path} mixes multiple schemas (ROI, Outcome, Legacy). "
            "Please use exactly one schema."
        )

    if not any([has_roi, has_outcome, has_legacy]):
        raise ValueError(
            f"lift-test CSV {path} does not match any supported schema. "
            f"Required ROI columns: {sorted(LIFT_TEST_ROI_COLUMNS)}, "
            f"Outcome columns: {sorted(LIFT_TEST_OUTCOME_COLUMNS)}, "
            f"or Legacy columns: {sorted(LIFT_TEST_LEGACY_COLUMNS)}."
        )

    if has_roi:
        out = df[list(LIFT_TEST_BASE_COLUMNS | LIFT_TEST_ROI_COLUMNS)].copy()
    elif has_outcome:
        out = df[list(LIFT_TEST_BASE_COLUMNS | LIFT_TEST_OUTCOME_COLUMNS)].copy()
        # Ensure incremental_spend is positive before dividing
        out["incremental_spend"] = pd.to_numeric(out["incremental_spend"], errors="raise")
        if bool((out["incremental_spend"] <= 0).any()):
             raise ValueError(
                f"lift-test CSV {path} has non-positive incremental_spend"
            )
        out["roi_estimate"] = out["incremental_outcome"] / out["incremental_spend"]
        out["roi_ci_lower"] = out["ci_lower_outcome"] / out["incremental_spend"]
        out["roi_ci_upper"] = out["ci_upper_outcome"] / out["incremental_spend"]
        out = out.drop(columns=list(LIFT_TEST_OUTCOME_COLUMNS))
    else:  # has_legacy
        warnings.warn(
            f"lift-test CSV {path} uses legacy columns (lift_estimate, ci_lower, ci_upper). "
            "Please migrate to explicit ROI columns (roi_estimate, roi_ci_lower, roi_ci_upper) "
            "or Outcome columns.",
            FutureWarning,
            stacklevel=2,
        )
        out = df[list(LIFT_TEST_BASE_COLUMNS | LIFT_TEST_LEGACY_COLUMNS)].copy()
        out = out.rename(
            columns={
                "lift_estimate": "roi_estimate",
                "ci_lower": "roi_ci_lower",
                "ci_upper": "roi_ci_upper",
            }
        )

    out["channel"] = out["channel"].astype(str).str.strip()
    if bool((out["channel"] == "").any()):
        raise ValueError(f"lift-test CSV {path} contains blank channel names")

    duplicates = sorted(out.loc[out["channel"].duplicated(), "channel"].unique())
    if duplicates:
        raise ValueError(
            f"lift-test CSV {path} contains duplicate channel rows: {duplicates}"
        )

    for column in ("test_start", "test_end"):
        out[column] = pd.to_datetime(out[column], errors="raise")

    bad_date_order = out["test_end"] < out["test_start"]
    if bool(bad_date_order.any()):
        channels = sorted(out.loc[bad_date_order, "channel"].tolist())
        raise ValueError(
            f"lift-test CSV {path} has test_end before test_start for channels: {channels}"
        )

    for column in ("roi_estimate", "roi_ci_lower", "roi_ci_upper"):
        out[column] = pd.to_numeric(out[column], errors="raise")

    bad_intervals = out["roi_ci_upper"] <= out["roi_ci_lower"]
    if bool(bad_intervals.any()):
        channels = sorted(out.loc[bad_intervals, "channel"].tolist())
        raise ValueError(
            f"lift-test CSV {path} has roi_ci_upper <= roi_ci_lower for channels: {channels}"
        )

    return out
