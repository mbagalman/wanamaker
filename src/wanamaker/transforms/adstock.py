"""Adstock transforms (FR-3.1, FR-3.4).

Adstock models the carryover effect of marketing spend across periods:
a purchase today may be influenced by an ad seen last week. v1 ships
**geometric** adstock as the default, with **Weibull** available as a
per-channel override.

These functions own the canonical mathematical formulas and serve as
test fixtures and documentation references. The actual application of
adstock inside a model fit happens within the engine backend's
probabilistic program (where decay/shape/scale are sampled parameters),
not as a preprocessing step on fixed values. See ``wanamaker.engine``.

References:
- Geometric adstock: Hanssens, Parsons & Schultz, *Market Response Models*
  (2nd ed., 2001), ch. 10. The classic single-parameter recurrence
  ``A_t = X_t + decay * A_{t-1}`` with decay in [0, 1).
- Weibull adstock: Jin et al. (Google), "Bayesian Methods for Media Mix
  Modeling with Carryover and Shape Effects" (2017), sec. 3.
- Canonical parameter ranges: docs/references/adstock_and_saturation.md

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
