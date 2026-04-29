"""Individual diagnostic checks (FR-2.2).

Each check is a pure function from input data to a ``CheckResult``. Keeping
them independent makes them straightforward to unit-test against the named
benchmark datasets in NFR-7.

Checks to implement (FR-2.2):
- history_length      (target: 78+ weeks, warning below 52)
- missing_values
- spend_variation     (low coefficient of variation per channel)
- collinearity        (paid-vs-paid and paid-vs-control)
- variable_count      (1:10 rule vs. observation count)
- date_regularity     (gaps and duplicates)
- target_stability    (outliers, structural breaks)
- target_leakage      (control corr to target above 0.95)
- structural_breaks   (change-point detection on residuals)

Acceptance criteria for each check are pinned to the named benchmark
datasets — see FR-2.2 acceptance and NFR-7.
"""

from __future__ import annotations

import pandas as pd

from wanamaker.diagnose.readiness import CheckResult


def check_history_length(df: pd.DataFrame, date_column: str) -> CheckResult:
    raise NotImplementedError("Phase 1: history-length check")


def check_target_leakage(
    df: pd.DataFrame,
    target_column: str,
    control_columns: list[str],
    threshold: float = 0.95,
) -> list[CheckResult]:
    raise NotImplementedError("Phase 1: target-leakage check")


def check_structural_breaks(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
) -> list[CheckResult]:
    raise NotImplementedError("Phase 1: structural-break check")
