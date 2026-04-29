"""Movement classification (FR-4.3).

Each historical estimate change is sorted into one of:

- WITHIN_PRIOR_CI    — small movement, expected variation
- IMPROVED_HOLDOUT   — large movement but the new model fits historical
                       data better; trustworthy update
- UNEXPLAINED        — large movement with no diagnostic explanation;
                       trust risk, flagged
- USER_INDUCED       — driven by configuration/prior change by the user;
                       not a model failure
- WEAKLY_IDENTIFIED  — movement in a channel the Trust Card flags as
                       poorly identifiable; expected instability

The fraction classified as UNEXPLAINED is the headline trust metric on
the refresh stability dimension of the Trust Card (FR-4.5).
"""

from __future__ import annotations

from enum import Enum


class MovementClass(str, Enum):
    WITHIN_PRIOR_CI = "within_prior_ci"
    IMPROVED_HOLDOUT = "improved_holdout"
    UNEXPLAINED = "unexplained"
    USER_INDUCED = "user_induced"
    WEAKLY_IDENTIFIED = "weakly_identified"


# A parameter is considered weakly identified when its 95% CI width
# exceeds this multiple of the absolute posterior mean. At 1.0 the CI
# spans more than the full magnitude of the estimate — a sign that the
# data provide little information about that parameter.
WEAKLY_IDENTIFIED_THRESHOLD: float = 1.0


def classify_movement(
    prev_hdi: tuple[float, float],
    curr_mean: float,
    curr_hdi: tuple[float, float],
) -> MovementClass:
    """Classify a single parameter movement between two model fits.

    Priority order (highest to lowest):
    1. WEAKLY_IDENTIFIED — the current estimate is poorly constrained;
       any apparent movement is likely noise.
    2. WITHIN_PRIOR_CI — the new mean falls within the previous 95% HDI;
       the change is consistent with normal posterior variation.
    3. UNEXPLAINED — the new mean falls outside the previous HDI with no
       other explanation available at this level of the call stack.
       USER_INDUCED and IMPROVED_HOLDOUT require external context (config
       diff, holdout metric comparison) and are resolved by callers that
       have that context; ``compute_diff`` uses this as the fallback.

    Args:
        prev_hdi: ``(hdi_low, hdi_high)`` from the previous run's summary.
        curr_mean: Posterior mean from the current run.
        curr_hdi: ``(hdi_low, hdi_high)`` from the current run's summary.

    Returns:
        The most specific applicable ``MovementClass``.
    """
    ci_width = curr_hdi[1] - curr_hdi[0]
    abs_mean = abs(curr_mean)

    # Weakly identified: CI wider than the magnitude of the estimate.
    if abs_mean > 0 and ci_width / abs_mean > WEAKLY_IDENTIFIED_THRESHOLD:
        return MovementClass.WEAKLY_IDENTIFIED

    # Within prior CI: current mean falls inside previous 95% HDI.
    if prev_hdi[0] <= curr_mean <= prev_hdi[1]:
        return MovementClass.WITHIN_PRIOR_CI

    return MovementClass.UNEXPLAINED


def unexplained_fraction(movements: list) -> float:
    """Return the fraction of movements classified as UNEXPLAINED.

    This is the headline refresh-stability metric fed to the Trust Card
    (FR-4.5). A value of 0.0 means every parameter change has a benign
    explanation; a value approaching 1.0 is a strong trust warning.

    Args:
        movements: A list of ``ParameterMovement`` instances.

    Returns:
        Float in ``[0.0, 1.0]``. Returns 0.0 for an empty list.
    """
    if not movements:
        return 0.0
    n_unexplained = sum(
        1 for m in movements
        if getattr(m, "movement_class", None) == MovementClass.UNEXPLAINED
    )
    return n_unexplained / len(movements)
