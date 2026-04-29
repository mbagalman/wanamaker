"""Adstock transforms (FR-3.1, FR-3.4).

Adstock models the carryover effect of marketing spend across periods:
a purchase today may be influenced by an ad seen last week. v1 ships
**geometric** adstock as the default, with **Weibull** available as a
per-channel override.

References:
- Geometric adstock: Hanssens, Parsons & Schultz, *Market Response Models*
  (2nd ed., 2001), §10.3. The classic single-parameter recurrence
  ``x_t' = x_t + λ · x_{t-1}'`` with λ ∈ [0, 1).
- Weibull adstock: Jin et al. (Google), "Bayesian Methods for Media Mix
  Modeling with Carryover and Shape Effects" (2017), §3.

Per FR-3.4, the adstock family does **not** auto-flip across refreshes
(that would compromise refresh accountability). Selection is user-driven
and persists.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def geometric_adstock(spend: NDArray[np.float64], decay: float) -> NDArray[np.float64]:
    """Apply geometric (single-exponential) adstock.

    Recurrence: ``y_t = x_t + decay * y_{t-1}``, with ``y_{-1} = 0``.

    Args:
        spend: 1-D array of per-period spend, oldest period first.
        decay: Carryover rate in ``[0, 1)``. Higher values mean longer carryover.

    Returns:
        Array of the same shape as ``spend`` containing the adstocked series.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1)``.
    """
    raise NotImplementedError("Phase 0: geometric adstock — needs unit tests vs known output")


def weibull_adstock(
    spend: NDArray[np.float64],
    shape: float,
    scale: float,
) -> NDArray[np.float64]:
    """Apply Weibull adstock (FR-3.4 override path).

    See Jin et al. (2017) §3 for the parameterization.
    """
    raise NotImplementedError("Phase 0: Weibull adstock")
