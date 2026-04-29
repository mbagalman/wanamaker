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


def _absolute_correlation(df: pd.DataFrame, left: str, right: str) -> float | None:
    """Return absolute Pearson correlation, or None for degenerate pairs."""
    pair = pd.DataFrame(
        {
            left: pd.to_numeric(df[left], errors="raise"),
            right: pd.to_numeric(df[right], errors="raise"),
        }
    ).dropna()
    if len(pair) < 2:
        return None

    left_std = float(pair[left].std(ddof=0))
    right_std = float(pair[right].std(ddof=0))
    if left_std == 0.0 or right_std == 0.0:
        return None

    return abs(float(pair[left].corr(pair[right])))


# ---------------------------------------------------------------------------
# Public check functions
# ---------------------------------------------------------------------------


def check_history_length(df: pd.DataFrame, date_column: str) -> CheckResult:
    """Assess whether the CSV has enough weekly history for MMM.

    Per FR-2.2, 78+ weeks is the target, fewer than 52 weeks is a warning, and
    fewer than 26 weeks is a blocker.

    Args:
        df: Input data frame.
        date_column: Name of the date column.

    Returns:
        A ``CheckResult`` with INFO, WARNING, or BLOCKER severity.

    Raises:
        KeyError: If ``date_column`` is absent from ``df``.
        ValueError: If the date column cannot be parsed as datetimes.
    """
    dates = pd.to_datetime(df[date_column])
    n_periods = int(dates.dropna().nunique())

    if n_periods < 26:
        return CheckResult(
            name="history_length",
            severity=CheckSeverity.BLOCKER,
            message=(
                f"Only {n_periods} unique periods were found in '{date_column}'. "
                "MMM is not recommended below 26 weekly observations."
            ),
        )

    if n_periods < 52:
        return CheckResult(
            name="history_length",
            severity=CheckSeverity.WARNING,
            message=(
                f"Only {n_periods} unique periods were found in '{date_column}'. "
                "MMM can fit below 52 weeks, but results should be treated with caution."
            ),
        )

    return CheckResult(
        name="history_length",
        severity=CheckSeverity.INFO,
        message=f"{n_periods} unique periods were found in '{date_column}'.",
    )


def check_date_regularity(df: pd.DataFrame, date_column: str) -> list[CheckResult]:
    """Detect duplicate dates and likely gaps in the date column.

    The expected cadence is inferred from the most common positive interval
    between sorted unique dates, using the smallest interval as a deterministic
    tie-break. Duplicate dates are blockers because they make the row grain
    ambiguous; gaps are warnings because they may be intentional but need review
    before fitting.

    Args:
        df: Input data frame.
        date_column: Name of the date column.

    Returns:
        List of detected date regularity issues. Empty when none are found.

    Raises:
        KeyError: If ``date_column`` is absent from ``df``.
        ValueError: If the date column cannot be parsed as datetimes.
    """
    dates = pd.to_datetime(df[date_column])
    results: list[CheckResult] = []

    duplicate_mask = dates.duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicate_count = int(duplicate_mask.sum())
        unique_duplicate_dates = int(dates[duplicate_mask].nunique())
        results.append(
            CheckResult(
                name="date_regularity",
                severity=CheckSeverity.BLOCKER,
                message=(
                    f"'{date_column}' has {duplicate_count} rows on "
                    f"{unique_duplicate_dates} duplicate date(s). Each period must have one row."
                ),
            )
        )

    unique_dates = pd.Series(dates.dropna().drop_duplicates()).sort_values().reset_index(drop=True)
    if len(unique_dates) < 3:
        return results

    intervals = unique_dates.diff().dropna()
    positive_intervals = intervals[intervals > pd.Timedelta(0)]
    if positive_intervals.empty:
        return results

    expected_interval = positive_intervals.mode().min()
    gap_threshold = expected_interval * 1.5
    gap_mask = positive_intervals > gap_threshold

    if bool(gap_mask.any()):
        gap_count = int(gap_mask.sum())
        max_gap_days = int(positive_intervals[gap_mask].max() / pd.Timedelta(days=1))
        expected_days = int(expected_interval / pd.Timedelta(days=1))
        results.append(
            CheckResult(
                name="date_regularity",
                severity=CheckSeverity.WARNING,
                message=(
                    f"'{date_column}' has {gap_count} gap(s). The expected interval is "
                    f"about {expected_days} day(s), but the largest gap is {max_gap_days} day(s)."
                ),
            )
        )

    return results


def check_missing_values(df: pd.DataFrame) -> list[CheckResult]:
    """Report missing values in any CSV column.

    Args:
        df: Input data frame.

    Returns:
        One warning ``CheckResult`` per column containing missing values. Empty
        when the data frame has no missing values.
    """
    results: list[CheckResult] = []
    missing_counts = df.isna().sum()

    for column, count in missing_counts[missing_counts > 0].items():
        results.append(
            CheckResult(
                name="missing_values",
                severity=CheckSeverity.WARNING,
                message=f"Column '{column}' has {int(count)} missing value(s).",
            )
        )

    return results


def check_target_stability(df: pd.DataFrame, target_column: str) -> list[CheckResult]:
    """Flag extreme outliers in the target series.

    Uses a robust modified z-score based on median absolute deviation (MAD).
    When MAD is zero, the check falls back to the interquartile range. This is
    deterministic and intentionally conservative for small CSV-only diagnostics.

    Args:
        df: Input data frame.
        target_column: Name of the numeric target column.

    Returns:
        A single warning when extreme target outliers are detected, otherwise
        an empty list.

    Raises:
        KeyError: If ``target_column`` is absent from ``df``.
        ValueError: If the target column cannot be converted to numeric values.
    """
    target = pd.to_numeric(df[target_column], errors="raise").dropna()
    if len(target) < 4:
        return []

    median = float(target.median())
    deviations = (target - median).abs()
    mad = float(deviations.median())

    if mad > 0.0:
        modified_z = 0.6745 * deviations / mad
        outlier_mask = modified_z > 6.0
    else:
        q1 = float(target.quantile(0.25))
        q3 = float(target.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0.0:
            return []
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        outlier_mask = (target < lower) | (target > upper)

    if not bool(outlier_mask.any()):
        return []

    outlier_count = int(outlier_mask.sum())
    return [
        CheckResult(
            name="target_stability",
            severity=CheckSeverity.WARNING,
            message=(
                f"'{target_column}' has {outlier_count} extreme outlier(s). "
                "Inspect these periods before fitting; one-off spikes can distort MMM estimates."
            ),
        )
    ]


def check_spend_variation(
    df: pd.DataFrame,
    spend_columns: list[str],
    threshold: float = 0.10,
) -> list[CheckResult]:
    """Flag spend columns with too little observed variation for MMM.

    Uses coefficient of variation, ``std / abs(mean)``, per FR-2.2. Constant
    and all-zero spend columns are flagged because saturation and adstock
    parameters cannot be learned from invariant spend.

    Args:
        df: Input data frame.
        spend_columns: Paid media spend columns to inspect.
        threshold: Minimum acceptable coefficient of variation.

    Returns:
        One warning ``CheckResult`` per low-variation spend column.

    Raises:
        KeyError: If any spend column is absent from ``df``.
        ValueError: If any spend column cannot be converted to numeric values,
            or if ``threshold`` is negative.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative; got {threshold!r}")

    results: list[CheckResult] = []
    for column in spend_columns:
        values = pd.to_numeric(df[column], errors="raise").dropna()
        if values.empty:
            cv = 0.0
        else:
            mean_abs = abs(float(values.mean()))
            std = float(values.std(ddof=0))
            cv = 0.0 if mean_abs == 0.0 else std / mean_abs

        if cv < threshold:
            results.append(
                CheckResult(
                    name="spend_variation",
                    severity=CheckSeverity.WARNING,
                    message=(
                        f"Spend column '{column}' has low variation "
                        f"(coefficient of variation {cv:.3f}, threshold {threshold:.3f}). "
                        "Saturation and ROI for this channel may be weakly identified."
                    ),
                )
            )

    return results


def check_collinearity(
    df: pd.DataFrame,
    spend_columns: list[str],
    control_columns: list[str],
    threshold: float = 0.90,
) -> list[CheckResult]:
    """Flag highly correlated paid/control predictor pairs.

    Args:
        df: Input data frame.
        spend_columns: Paid media spend columns.
        control_columns: Non-media control columns.
        threshold: Absolute Pearson correlation at or above which to warn.

    Returns:
        One warning ``CheckResult`` per highly correlated predictor pair.

    Raises:
        KeyError: If any predictor column is absent from ``df``.
        ValueError: If any predictor column cannot be converted to numeric
            values, or if ``threshold`` is outside ``[0, 1]``.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")

    results: list[CheckResult] = []
    pairs: list[tuple[str, str, str]] = []
    for left_idx, left in enumerate(spend_columns):
        for right in spend_columns[left_idx + 1 :]:
            pairs.append((left, right, "paid channel"))
    for spend_column in spend_columns:
        for control_column in control_columns:
            pairs.append((spend_column, control_column, "control"))

    for left, right, right_kind in pairs:
        corr = _absolute_correlation(df, left, right)
        if corr is None or corr < threshold:
            continue
        results.append(
            CheckResult(
                name="collinearity",
                severity=CheckSeverity.WARNING,
                message=(
                    f"'{left}' and '{right}' are highly correlated "
                    f"(absolute correlation {corr:.3f}). The model may have trouble "
                    f"separating the effect of this channel from the {right_kind}."
                ),
            )
        )

    return results


def check_variable_count(
    df: pd.DataFrame,
    spend_columns: list[str],
    control_columns: list[str],
    observations_per_variable: int = 10,
) -> CheckResult:
    """Apply the FR-2.2 1:10 observations-to-predictors rule.

    Args:
        df: Input data frame.
        spend_columns: Paid media spend columns.
        control_columns: Non-media control columns.
        observations_per_variable: Minimum observations expected per predictor.

    Returns:
        ``INFO`` when the rule is satisfied, otherwise ``WARNING``.

    Raises:
        ValueError: If ``observations_per_variable`` is less than 1.
    """
    if observations_per_variable < 1:
        raise ValueError(
            "observations_per_variable must be at least 1; "
            f"got {observations_per_variable!r}"
        )

    n_observations = len(df)
    n_predictors = len(spend_columns) + len(control_columns)
    required = n_predictors * observations_per_variable

    if n_predictors == 0 or n_observations >= required:
        return CheckResult(
            name="variable_count",
            severity=CheckSeverity.INFO,
            message=(
                f"{n_observations} observations for {n_predictors} predictor(s); "
                f"minimum recommended is {required}."
            ),
        )

    return CheckResult(
        name="variable_count",
        severity=CheckSeverity.WARNING,
        message=(
            f"{n_observations} observations for {n_predictors} predictor(s). "
            f"The 1:{observations_per_variable} rule recommends at least {required}; "
            "estimates may be unstable."
        ),
    )


def check_target_leakage(
    df: pd.DataFrame,
    target_column: str,
    control_columns: list[str],
    threshold: float = 0.95,
) -> list[CheckResult]:
    """Flag controls that are likely derived from the target.

    Per FR-2.2, any control variable with absolute correlation to the target at
    or above 0.95 is a leakage candidate.

    Args:
        df: Input data frame.
        target_column: Numeric target column.
        control_columns: Non-media control columns to compare against target.
        threshold: Absolute Pearson correlation at or above which to warn.

    Returns:
        One warning ``CheckResult`` per likely leakage control.

    Raises:
        KeyError: If ``target_column`` or any control column is absent from
            ``df``.
        ValueError: If any compared column cannot be converted to numeric
            values, or if ``threshold`` is outside ``[0, 1]``.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")

    results: list[CheckResult] = []
    for column in control_columns:
        corr = _absolute_correlation(df, target_column, column)
        if corr is None or corr < threshold:
            continue
        results.append(
            CheckResult(
                name="target_leakage",
                severity=CheckSeverity.WARNING,
                message=(
                    f"Control column '{column}' is highly correlated with "
                    f"'{target_column}' (absolute correlation {corr:.3f}). "
                    "It may be derived from the target and should be inspected before fitting."
                ),
            )
        )

    return results


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
