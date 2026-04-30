"""CSV loading.

The MMM input is a single CSV with one row per time period (default weekly).
See FR-1.1 in the BRD/PRD for the contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wanamaker.config import DataConfig

LIFT_TEST_REQUIRED_COLUMNS = {
    "channel",
    "test_start",
    "test_end",
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

    Required columns: channel, test_start, test_end, lift_estimate,
    ci_lower, ci_upper.

    The returned frame preserves the required column names, parses test dates,
    converts numeric lift fields to floats, and rejects ambiguous duplicate
    channel rows. Conversion from confidence intervals to ``LiftPrior`` objects
    is handled by ``wanamaker.model.builder``.

    Args:
        path: Path to the lift-test CSV.

    Returns:
        Validated lift-test results, one row per channel.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing, dates/numbers cannot be
            parsed, intervals are invalid, or a channel appears more than once.
    """
    df = pd.read_csv(path)
    missing = LIFT_TEST_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"lift-test CSV {path} is missing required columns: {sorted(missing)}"
        )

    out = df[
        ["channel", "test_start", "test_end", "lift_estimate", "ci_lower", "ci_upper"]
    ].copy()
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

    for column in ("lift_estimate", "ci_lower", "ci_upper"):
        out[column] = pd.to_numeric(out[column], errors="raise")

    bad_intervals = out["ci_upper"] <= out["ci_lower"]
    if bool(bad_intervals.any()):
        channels = sorted(out.loc[bad_intervals, "channel"].tolist())
        raise ValueError(
            f"lift-test CSV {path} has ci_upper <= ci_lower for channels: {channels}"
        )

    return out
