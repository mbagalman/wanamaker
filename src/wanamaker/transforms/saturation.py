"""Saturation transforms (FR-3.1).

Saturation models diminishing returns: each additional dollar of spend
produces less incremental impact than the last. v1 uses the **Hill function**,
parameterized by an EC50 (the spend at half-maximum response) and a slope.

Reference:
- Hill function in MMM: Jin et al. (Google), "Bayesian Methods for Media
  Mix Modeling with Carryover and Shape Effects" (2017), §4.

Per FR-5.1, when response curves are shown, the portion that extrapolates
beyond historical observed spend must be visually distinguished. That
visual treatment is owned by ``wanamaker.reports``; this module is just
the math.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hill_saturation(
    spend: NDArray[np.float64],
    ec50: float,
    slope: float,
) -> NDArray[np.float64]:
    """Apply the Hill saturation function.

    ``f(x) = x^slope / (x^slope + ec50^slope)``

    Args:
        spend: 1-D array of (typically adstocked) per-period spend.
        ec50: Spend level at which response is half of its maximum. Must be
            strictly positive.
        slope: Hill slope (also called Hill coefficient). Must be strictly
            positive. Higher values produce a sharper transition.

    Returns:
        Array of the same shape as ``spend`` with values in ``[0, 1)``.

    Raises:
        ValueError: If ``ec50`` or ``slope`` is non-positive.
    """
    raise NotImplementedError("Phase 0: Hill saturation — needs unit tests vs known output")
