"""CSV loading.

The MMM input is a single CSV with one row per time period (default weekly).
See FR-1.1 in the BRD/PRD for the contract.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from wanamaker.config import DataConfig

_LIFT_TEST_BASE_COLUMNS = (
    "channel",
    "test_start",
    "test_end",
)

_LIFT_TEST_ROI_COLUMNS = (
    "roi_estimate",
    "roi_ci_lower",
    "roi_ci_upper",
)

_LIFT_TEST_OUTCOME_COLUMNS = (
    "incremental_outcome",
    "incremental_spend",
    "ci_lower_outcome",
    "ci_upper_outcome",
)

_LIFT_TEST_LEGACY_COLUMNS = (
    "lift_estimate",
    "ci_lower",
    "ci_upper",
)


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
    parses test dates, and converts numeric fields to floats. Conversion
    from confidence intervals to ``LiftPrior`` objects (including
    precision-weighted pooling when a channel has multiple tests) is
    handled by ``wanamaker.model.builder``.

    Multiple rows per channel are allowed and pooled downstream as
    independent ROI estimates (#78). When two rows for the same channel
    have overlapping ``[test_start, test_end]`` windows, a
    ``UserWarning`` fires because the independence assumption the
    pooling formula relies on is violated — the user should either
    combine the tests externally or supply a wider interval.

    Args:
        path: Path to the lift-test CSV.

    Returns:
        Validated lift-test results with canonical ROI columns
        (``roi_estimate``, ``roi_ci_lower``, ``roi_ci_upper``).
        Multiple rows per channel are preserved.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing, mixed schemas are used,
            dates/numbers cannot be parsed, or intervals are invalid.
    """
    df = pd.read_csv(path)
    columns = set(df.columns)

    missing_base = set(_LIFT_TEST_BASE_COLUMNS) - columns
    if missing_base:
        raise ValueError(
            f"lift-test CSV {path} is missing required base columns: {sorted(missing_base)}"
        )

    has_roi = set(_LIFT_TEST_ROI_COLUMNS).issubset(columns)
    has_outcome = set(_LIFT_TEST_OUTCOME_COLUMNS).issubset(columns)
    has_legacy = set(_LIFT_TEST_LEGACY_COLUMNS).issubset(columns)

    if sum([has_roi, has_outcome, has_legacy]) > 1:
        raise ValueError(
            f"lift-test CSV {path} mixes multiple schemas (ROI, Outcome, Legacy). "
            "Please use exactly one schema."
        )

    if not any([has_roi, has_outcome, has_legacy]):
        raise ValueError(
            f"lift-test CSV {path} does not match any supported schema. "
            f"Required ROI columns: {sorted(_LIFT_TEST_ROI_COLUMNS)}, "
            f"Outcome columns: {sorted(_LIFT_TEST_OUTCOME_COLUMNS)}, "
            f"or Legacy columns: {sorted(_LIFT_TEST_LEGACY_COLUMNS)}."
        )

    if has_roi:
        out = df[list(_LIFT_TEST_BASE_COLUMNS) + list(_LIFT_TEST_ROI_COLUMNS)].copy()
    elif has_outcome:
        out = df[list(_LIFT_TEST_BASE_COLUMNS) + list(_LIFT_TEST_OUTCOME_COLUMNS)].copy()
        for col in _LIFT_TEST_OUTCOME_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="raise")
        # Ensure incremental_spend is positive before dividing
        if bool((out["incremental_spend"] <= 0).any()):
             raise ValueError(
                f"lift-test CSV {path} has non-positive incremental_spend"
            )
        out["roi_estimate"] = out["incremental_outcome"] / out["incremental_spend"]
        out["roi_ci_lower"] = out["ci_lower_outcome"] / out["incremental_spend"]
        out["roi_ci_upper"] = out["ci_upper_outcome"] / out["incremental_spend"]
        out = out.drop(columns=list(_LIFT_TEST_OUTCOME_COLUMNS))
    else:  # has_legacy
        warnings.warn(
            f"lift-test CSV {path} uses legacy columns (lift_estimate, ci_lower, ci_upper). "
            "Please migrate to explicit ROI columns (roi_estimate, roi_ci_lower, roi_ci_upper) "
            "or Outcome columns.",
            FutureWarning,
            stacklevel=2,
        )
        out = df[list(_LIFT_TEST_BASE_COLUMNS) + list(_LIFT_TEST_LEGACY_COLUMNS)].copy()
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

    overlapping = _channels_with_overlapping_test_windows(out)
    if overlapping:
        warnings.warn(
            f"lift-test CSV {path} has overlapping test windows for channels: "
            f"{sorted(overlapping)}. The pooling formula assumes the rows are "
            "independent; overlapping market/time/audience violates that. "
            "Combine these tests externally or supply a wider interval.",
            UserWarning,
            stacklevel=2,
        )

    return out


def _channels_with_overlapping_test_windows(df: pd.DataFrame) -> set[str]:
    """Return channels whose multiple test rows have overlapping date windows.

    Independence-violating overlap is the case the pooling formula in
    ``wanamaker.model.builder`` cannot recover from cleanly. Two
    closed intervals ``[a1, a2]`` and ``[b1, b2]`` overlap iff
    ``a1 <= b2 and b1 <= a2``.
    """
    overlapping: set[str] = set()
    for channel, group in df.groupby("channel"):
        if len(group) < 2:
            continue
        starts = group["test_start"].tolist()
        ends = group["test_end"].tolist()
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if starts[i] <= ends[j] and starts[j] <= ends[i]:
                    overlapping.add(str(channel))
                    break
            if channel in overlapping:
                break
    return overlapping
