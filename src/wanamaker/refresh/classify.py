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
