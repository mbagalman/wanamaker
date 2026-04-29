"""Saturation transforms (FR-3.1).

Saturation models diminishing returns: each additional dollar of spend
produces less incremental impact than the last. v1 uses the **Hill function**,
parameterized by an EC50 (the spend at half-maximum response) and a slope.

These functions own the canonical mathematical formulas and serve as test
fixtures and documentation references. EC50 and slope are sampled parameters
inside the engine backend's probabilistic program, not fixed values passed in
before fitting. See ``wanamaker.engine``.

Reference:
- Hill function in MMM: Jin et al. (Google), "Bayesian Methods for Media Mix
  Modeling with Carryover and Shape Effects" (2017), sec. 4.
- Canonical parameter ranges: docs/references/adstock_and_saturation.md

Per FR-5.1, when response curves are shown, the portion that extrapolates
beyond historical observed spend must be visually distinguished. That visual
treatment is owned by ``wanamaker.reports``; this module is just the math.
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

    The implementation uses the equivalent logistic form in log-space for
    positive spend values:

    ``f(x) = 1 / (1 + exp(-slope * (log(x) - log(ec50))))``

    Args:
        spend: 1-D array of non-negative, typically adstocked spend.
        ec50: Spend level at which response is half of its maximum. Must be
            strictly positive.
        slope: Hill slope (also called Hill coefficient). Must be strictly
            positive. Higher values produce a sharper transition.

    Returns:
        Array of the same shape as ``spend`` with values in ``[0, 1)``.

    Raises:
        ValueError: If ``ec50`` or ``slope`` is non-positive, if ``spend`` is
            not 1-D, or if any spend value is negative.

    Examples:
        At EC50, response is exactly half:

        >>> hill_saturation(np.array([10.0]), ec50=10.0, slope=2.0)
        array([0.5])
    """
    if ec50 <= 0:
        raise ValueError(f"ec50 must be strictly positive; got {ec50!r}")
    if slope <= 0:
        raise ValueError(f"slope must be strictly positive; got {slope!r}")

    spend = np.asarray(spend, dtype=np.float64)
    if spend.ndim != 1:
        raise ValueError(f"spend must be a 1-D array; got shape {spend.shape}")
    if np.any(spend < 0):
        raise ValueError("spend must be non-negative")

    out = np.zeros_like(spend)
    positive = spend > 0
    if not np.any(positive):
        return out

    log_ratio = slope * (np.log(spend[positive]) - np.log(ec50))
    out[positive] = 1.0 / (1.0 + np.exp(-log_ratio))
    return out
