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

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def half_life_to_decay(half_life: float) -> float:
    """Convert an adstock half-life (in periods) to a decay rate.

    Wanamaker priors are specified in half-life space (more interpretable
    for users) and converted to decay theta internally. The relationship is:

        theta = 0.5 ** (1 / half_life)

    So a half-life of 1 period gives theta = 0.5, 2 periods gives ~0.707,
    4 periods gives ~0.841, and 8 periods gives ~0.917.

    Reference: docs/references/adstock_and_saturation.md sec. 2.1.

    Args:
        half_life: Number of periods for the adstock effect to decay to half
            its initial value. Must be strictly positive.

    Returns:
        Decay rate theta in [0, 1).

    Raises:
        ValueError: If ``half_life`` is not strictly positive.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be strictly positive; got {half_life!r}")
    return 0.5 ** (1.0 / half_life)


def geometric_adstock(spend: NDArray[np.float64], decay: float) -> NDArray[np.float64]:
    """Apply geometric (single-exponential) adstock.

    Recurrence: ``A_t = X_t + decay * A_{t-1}``, with ``A_{-1} = 0``.

    Equivalent closed form: ``A_t = sum_{k=0}^{t} decay^k * X_{t-k}``.

    The recursive form is used here for numerical stability on long series.
    This function owns the canonical formula and is used as a reference by
    engine backends and for unit tests. Inside a model fit, the engine
    reimplements this in backend-native tensor ops with ``decay`` as a
    sampled parameter.

    Reference: Hanssens, Parsons & Schultz, *Market Response Models*
    (2nd ed., 2001), ch. 10. See also docs/references/adstock_and_saturation.md
    sec. 2.1.

    Args:
        spend: 1-D array of per-period spend values, oldest period first.
            Accepts any numeric dtype; output is always float64.
        decay: Carryover rate in ``[0, 1)``. ``decay = 0`` means no
            carryover (output equals input). Higher values mean slower decay
            and longer-lived ad effects. Use ``half_life_to_decay`` to
            convert from half-life space.

    Returns:
        Array of the same shape as ``spend`` containing the adstocked series.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1)``.

    Examples:
        Single impulse with half-life of 1 period (decay = 0.5):

        >>> geometric_adstock(np.array([1.0, 0.0, 0.0, 0.0]), decay=0.5)
        array([1.   , 0.5  , 0.25 , 0.125])

        Two consecutive periods of spend:

        >>> geometric_adstock(np.array([1.0, 1.0, 0.0]), decay=0.5)
        array([1.  , 1.5 , 0.75])
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(
            f"decay must be in [0, 1); got {decay!r}. "
            "Use half_life_to_decay() to convert from half-life space."
        )
    spend = np.asarray(spend, dtype=np.float64)
    if spend.ndim != 1:
        raise ValueError(f"spend must be a 1-D array; got shape {spend.shape}")
    out = np.empty_like(spend)
    prev = 0.0
    for t in range(len(spend)):
        prev = float(spend[t]) + decay * prev
        out[t] = prev
    return out


def weibull_adstock(
    spend: NDArray[np.float64],
    shape: float = 2.0,
    scale: float = 2.0,
    *,
    variant: Literal["cdf", "pdf"] = "cdf",
) -> NDArray[np.float64]:
    """Apply Weibull adstock (FR-3.4 override path).

    Weibull adstock is the optional two-parameter carryover family described
    by Jin et al. (2017), sec. 3. Wanamaker supports two variants:

    - ``"cdf"``: survival-weighted carryover, where lag weights are
      ``exp(-(lag / scale) ** shape)``. With ``shape = 1``, this is exactly
      geometric adstock with ``decay = exp(-1 / scale)``.
    - ``"pdf"``: discrete Weibull-density lag weights, normalized so the
      largest lag weight is 1. This can represent a delayed peak before decay.

    Args:
        spend: 1-D array of per-period spend, oldest period first. Accepts any
            numeric dtype; output is always float64.
        shape: Weibull shape parameter. Must be strictly positive.
        scale: Weibull scale parameter in periods. Must be strictly positive.
        variant: Either ``"cdf"`` or ``"pdf"``. Defaults to ``"cdf"`` per
            docs/references/adstock_and_saturation.md sec. 2.2.

    Returns:
        Array of the same shape as ``spend`` containing the adstocked series.

    Raises:
        ValueError: If ``shape`` or ``scale`` is non-positive, if ``spend`` is
            not a 1-D array, or if ``variant`` is unknown.
    """
    spend = np.asarray(spend, dtype=np.float64)
    if spend.ndim != 1:
        raise ValueError(f"spend must be a 1-D array; got shape {spend.shape}")
    if shape <= 0:
        raise ValueError(f"shape must be strictly positive; got {shape!r}")
    if scale <= 0:
        raise ValueError(f"scale must be strictly positive; got {scale!r}")

    n_periods = len(spend)
    if n_periods == 0:
        return spend.copy()

    if variant == "cdf":
        lag_weights = _weibull_cdf_weights(n_periods, shape, scale)
    elif variant == "pdf":
        lag_weights = _weibull_pdf_weights(n_periods, shape, scale)
    else:
        raise ValueError(f"variant must be 'cdf' or 'pdf'; got {variant!r}")

    out = np.empty_like(spend)
    for t in range(n_periods):
        current_and_prior = spend[: t + 1][::-1]
        out[t] = np.dot(current_and_prior, lag_weights[: t + 1])
    return out


def _weibull_cdf_weights(
    n_periods: int,
    shape: float,
    scale: float,
) -> NDArray[np.float64]:
    """Return Weibull survival weights for lags 0 through ``n_periods - 1``."""
    lags = np.arange(n_periods, dtype=np.float64)
    return np.exp(-np.power(lags / scale, shape))


def _weibull_pdf_weights(
    n_periods: int,
    shape: float,
    scale: float,
) -> NDArray[np.float64]:
    """Return normalized discrete Weibull PDF weights for lagged carryover."""
    lags = np.arange(1, n_periods + 1, dtype=np.float64)
    weights = (shape / scale) * np.power(lags / scale, shape - 1) * np.exp(
        -np.power(lags / scale, shape)
    )
    max_weight = float(np.max(weights))
    if max_weight == 0.0:
        return weights
    return weights / max_weight
