"""CSV loading.

The MMM input is a single CSV with one row per time period (default weekly).
See FR-1.1 in the BRD/PRD for the contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wanamaker.config import DataConfig


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
    """
    raise NotImplementedError("Phase 1: lift-test calibration loader")
