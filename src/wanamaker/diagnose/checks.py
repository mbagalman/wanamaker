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

import numpy as np
import pandas as pd

from wanamaker.diagnose.readiness import CheckResult, CheckSeverity

# ---------------------------------------------------------------------------
# Internal helpers for structural-break detection
# ---------------------------------------------------------------------------

_MIN_SEGMENT = 12   # minimum periods on each side of a candidate break
_F_THRESHOLD = 15.0  # F-statistic threshold to declare a break significant
_MAX_BREAKS = 3      # greedy binary segmentation ceiling


def _seg_ssr(
    cs: np.ndarray,
    cs2: np.ndarray,
    a: int,
    b: int,
) -> float:
    """Sum of squared residuals for the segment [a, b) around its mean.

    Uses pre-built prefix-sum arrays for O(1) evaluation per candidate.
    The formula avoids a full refit:
        SSR = sum(r^2) - (sum(r))^2 / n
    which equals the within-segment sum of squares around the segment mean.

    The ``max(0.0, ...)`` guard prevents floating-point cancellation from
    producing a tiny negative SSR on near-constant or near-linear segments.
    """
    ns = b - a
    if ns <= 1:
        return 0.0
    return max(0.0, float((cs2[b] - cs2[a]) - (cs[b] - cs[a]) ** 2 / ns))


def _find_best_break(
    resid: np.ndarray,
    cs: np.ndarray,
    cs2: np.ndarray,
    lo: int,
    hi: int,
) -> tuple[int, float]:
    """Return the index and F-statistic of the best single break in [lo, hi).

    Scans all candidate break points in [lo+MIN_SEGMENT, hi-MIN_SEGMENT] and
    picks the one that minimises the sum of squared residuals in both halves.
    Returns ``(-1, 0.0)`` if the segment is too short to contain a valid break.
    """
    n = hi - lo
    if n < 2 * _MIN_SEGMENT + 1:
        return -1, 0.0

    ssr_full = _seg_ssr(cs, cs2, lo, hi)
    best_ssr = ssr_full
    best_t = -1

    for t_star in range(lo + _MIN_SEGMENT, hi - _MIN_SEGMENT):
        ssr = _seg_ssr(cs, cs2, lo, t_star) + _seg_ssr(cs, cs2, t_star, hi)
        if ssr < best_ssr:
            best_ssr = ssr
            best_t = t_star

    if best_t == -1:
        return -1, 0.0

    # F-statistic: improvement in fit relative to residual variance
    df_resid = n - 2
    if best_ssr == 0.0 or df_resid <= 0:
        return -1, 0.0
    f_stat = (ssr_full - best_ssr) / (best_ssr / df_resid)
    return best_t, f_stat


# ---------------------------------------------------------------------------
# Public check functions
# ---------------------------------------------------------------------------


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
    """Detect structural breaks (level or trend shifts) in the target series.

    Uses a greedy binary segmentation algorithm with an O(n) prefix-sum scan
    so the entire check runs in a few milliseconds even for long series. No
    external dependencies beyond numpy and pandas are required.

    Algorithm:
    1. Sort by ``date_column`` and extract the target values.
    2. Linearly detrend the series so the scan detects *level/slope changes*
       rather than a trend itself.
    3. Build prefix-sum arrays for O(1) SSR evaluation.
    4. Iteratively find the best single break in the current set of segments
       (greedy binary segmentation), accepting breaks where the F-statistic
       exceeds ``_F_THRESHOLD`` (default 15.0).
    5. Stop after ``_MAX_BREAKS`` (default 3) breaks or when no remaining
       segment contains a significant break.

    Each accepted break is returned as a separate ``CheckResult`` with
    ``WARNING`` severity. An empty list means no significant breaks were found.

    The check is config-independent and deterministic (issue #14 decision).

    Args:
        df: Input data frame; must contain ``target_column`` and ``date_column``.
        target_column: Name of the numeric target (e.g. revenue) column.
        date_column: Name of the date column (parseable by ``pd.to_datetime``).

    Returns:
        List of ``CheckResult`` objects, one per detected structural break,
        ordered by break date. Empty if no significant breaks are found.

    Raises:
        KeyError: If ``target_column`` or ``date_column`` is absent from ``df``.
        ValueError: If the target column contains non-numeric or all-NaN values.
    """
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    dates = pd.to_datetime(df_sorted[date_column])
    y = df_sorted[target_column].to_numpy(dtype=float)
    n = len(y)

    if n < 2 * _MIN_SEGMENT + 1:
        return []

    # Linear detrend: remove global linear trend so the scan targets
    # *changes in level or slope*, not the trend itself.
    t_idx = np.arange(n, dtype=float)
    coef = np.polyfit(t_idx, y, 1)
    resid = y - np.polyval(coef, t_idx)

    # Guard against near-constant or near-perfectly-linear series where
    # floating-point residuals are essentially numerical noise. If the
    # residual std is negligible relative to the data scale, there is
    # nothing to detect and the prefix-sum F-statistic is undefined.
    y_std = float(np.std(y))
    resid_std = float(np.std(resid))
    # Skip the scan if:
    #   - the target has no variation (constant series): y_std == 0
    #   - the residuals are exactly zero
    #   - the residual variation is negligible relative to the data scale
    #     (e.g. a near-perfect linear trend where polyfit leaves only
    #     floating-point noise in the residuals)
    if y_std == 0.0 or resid_std == 0.0 or resid_std / y_std < 1e-10:
        return []

    # Prefix-sum arrays for O(1) segment SSR evaluation.
    cs = np.empty(n + 1, dtype=float)
    cs2 = np.empty(n + 1, dtype=float)
    cs[0] = 0.0
    cs2[0] = 0.0
    cs[1:] = np.cumsum(resid)
    cs2[1:] = np.cumsum(resid ** 2)

    # Greedy binary segmentation: maintain a list of (lo, hi) segments,
    # grow it by splitting the segment with the best significant break.
    segments: list[tuple[int, int]] = [(0, n)]
    break_indices: list[int] = []

    for _ in range(_MAX_BREAKS):
        candidate_t = -1
        candidate_f = 0.0
        candidate_seg_idx = -1

        for seg_idx, (lo, hi) in enumerate(segments):
            t_star, f_stat = _find_best_break(resid, cs, cs2, lo, hi)
            if t_star != -1 and f_stat > candidate_f:
                candidate_f = f_stat
                candidate_t = t_star
                candidate_seg_idx = seg_idx

        if candidate_t == -1 or candidate_f < _F_THRESHOLD:
            break

        lo, hi = segments[candidate_seg_idx]
        segments[candidate_seg_idx] = (lo, candidate_t)
        segments.insert(candidate_seg_idx + 1, (candidate_t, hi))
        break_indices.append(candidate_t)

    if not break_indices:
        return []

    results: list[CheckResult] = []
    for idx in sorted(break_indices):
        break_date = dates.iloc[idx]
        date_str = break_date.strftime("%Y-%m-%d")
        message = (
            f"Structural break detected near {date_str} in '{target_column}'. "
            "The target series shows a significant level or slope shift at this "
            "point. Consider whether an external event (e.g. COVID-19, product "
            "launch, channel mix change) explains it. If so, add a binary control "
            "variable covering the affected period before fitting."
        )
        results.append(
            CheckResult(
                name="structural_breaks",
                severity=CheckSeverity.WARNING,
                message=message,
            )
        )
    return results
