"""Default priors per channel category (FR-1.2, FR-3.1).

Each of the 10 default channel categories in ``data/taxonomy.py`` carries
weakly-informative prior shapes for adstock half-life and Hill saturation
alpha. These priors nudge the model toward plausible values without
dominating the data; users can override any field per-channel via YAML.

**Prior distributions used**

Half-life and Hill-alpha are both strictly positive, so both use
log-normal priors parameterised in **log-scale**:

    θ ~ LogNormal(mu, sigma)

where ``mu`` is the log of the *median* of the prior and ``sigma`` is
the dispersion on the log scale.  This is the natural parameterisation
for PyMC ``pm.LogNormal`` and NumPyro ``dist.LogNormal``.

**Gamma (Hill EC50) is data-driven** — it is scaled to the channel's
observed median spend per period when a model is configured.  It is
*not* a static category default and is not stored here.

**Half-life sigma convention**

- Channels with short, predictable carryover (paid search, affiliate,
  email, promotions): sigma = 0.35 (tighter, widely agreed range).
- Mid-range channels: sigma = 0.40 (standard weakly-informative).
- Long-carryover channels (linear TV, CTV, audio/podcast): sigma = 0.45
  (slightly wider to acknowledge greater real-world variability).

At sigma = 0.40 and median = 1 week the 90 % prior range is [0.51, 1.96]
weeks; at median = 6 weeks it is [3.07, 11.7] weeks.

**Hill alpha sigma convention**

All categories share sigma = 0.50 on the log scale, giving a 90 %
prior range of approximately [0.66, 3.4] around median 1.5, which
spans the canonical [0.5, 3.0] range from Jin et al. (2017).

**Sources**

- Jin et al. (Google, 2017), *Bayesian Methods for Media Mix Modeling with
  Carryover and Shape Effects*, sec. 3 — foundational paper; canonical
  parameter ranges for carryover and saturation.
- Robyn (Meta) Analyst's Guide to MMM — practical guidance on half-life
  ranges by channel type; ``https://facebookexperimental.github.io/Robyn/``.
- ``docs/references/adstock_and_saturation.md`` sec. 4 — Wanamaker's
  canonical parameter range table (the primary internal reference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wanamaker.data.taxonomy import DEFAULT_CHANNEL_CATEGORIES


@dataclass(frozen=True)
class ChannelPriors:
    """Weakly-informative prior parameters for a single channel category.

    All log-normal priors follow the parameterisation::

        θ ~ LogNormal(mu, sigma)

    where ``mu = log(median)`` and ``sigma`` is the log-scale dispersion.

    Attributes:
        half_life_mu: Log-scale mean of the adstock half-life prior
            (= log of the prior *median* in weeks).
        half_life_sigma: Log-scale std of the adstock half-life prior.
        hill_alpha_mu: Log-scale mean of the Hill saturation alpha prior
            (= log of the prior median for alpha; all categories share
            median ≈ 1.5 per Jin et al., 2017).
        hill_alpha_sigma: Log-scale std of the Hill alpha prior.
            All categories share 0.50, giving a 90 % range ≈ [0.66, 3.4].
    """

    half_life_mu: float
    """log(median half-life in weeks); passed to LogNormal as mu."""
    half_life_sigma: float
    """Log-scale dispersion of the half-life prior."""
    hill_alpha_mu: float
    """log(median Hill alpha); passed to LogNormal as mu."""
    hill_alpha_sigma: float
    """Log-scale dispersion of the Hill alpha prior."""


# ---------------------------------------------------------------------------
# Per-category prior table
# ---------------------------------------------------------------------------
# Values are derived from docs/references/adstock_and_saturation.md sec. 4
# and the sources cited in the module docstring.  Median values match the
# midpoint of the range given in that table; sigma reflects the uncertainty
# range (tighter for well-understood direct-response channels, slightly
# wider for brand-awareness channels with more variability in practice).
#
# Hill alpha: all categories share median 1.5 (Jin et al., 2017 sec. 3.2)
# and sigma 0.50 (weakly informative, covers [0.5, 3.0] at ~90 %).
#
# Half-life mu = log(median_weeks); reference ranges per channel:
#   paid_search           0.5–1 w  → median 1.0 w  → mu = 0.0000
#   paid_social           1–3 w    → median 2.0 w  → mu = 0.6931
#   video (online)        2–4 w    → median 3.0 w  → mu = 1.0986
#   linear_tv             4–8 w    → median 6.0 w  → mu = 1.7918
#   ctv                   3–5 w    → median 4.0 w  → mu = 1.3863
#   audio_podcast         3–6 w    → median 4.5 w  → mu = 1.5041
#   display_programmatic  1–2 w    → median 1.5 w  → mu = 0.4055
#   affiliate             0.5–1 w  → median 1.0 w  → mu = 0.0000
#   email_crm             0.5–1 w  → median 0.7 w  → mu = −0.3567
#   promotions_discounting 0–1 w   → median 0.5 w  → mu = −0.6931

_HILL_ALPHA_MU: float = math.log(1.5)   # ≈ 0.4055
_HILL_ALPHA_SIGMA: float = 0.50

_PRIORS: dict[str, ChannelPriors] = {
    # -----------------------------------------------------------------------
    # Paid search — direct response, intent-driven.
    # Short carryover: most effect within the week of exposure; retargeting
    # adds a small tail. Median 1 week; tighter dispersion (sigma 0.35)
    # because this range is well-established across many advertisers.
    # Ref: Robyn Analyst's Guide §4; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "paid_search": ChannelPriors(
        half_life_mu=math.log(1.0),    # median 1.0 week
        half_life_sigma=0.35,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Paid social — mix of performance (short) and brand (longer).
    # Retargeting pixels and lookalike audiences extend the tail beyond
    # pure direct-response. Median 2 weeks; standard dispersion.
    # Ref: Jin et al. (2017) Table 2; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "paid_social": ChannelPriors(
        half_life_mu=math.log(2.0),    # median 2.0 weeks
        half_life_sigma=0.40,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Online video (YouTube, OTT pre-roll, etc.) — awareness + some response.
    # Mix of upper-funnel brand exposure and lower-funnel retargeting.
    # Median 3 weeks; standard dispersion.
    # Ref: Robyn Analyst's Guide; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "video": ChannelPriors(
        half_life_mu=math.log(3.0),    # median 3.0 weeks
        half_life_sigma=0.40,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Linear TV — primarily brand-building; long ambient carryover.
    # Broadbent (1979) documented 4–8 weeks for UK TV; subsequent studies
    # in digital-era markets converge on similar ranges. Median 6 weeks;
    # slightly wider dispersion because actual carryover varies with
    # creative quality and category (FMCG vs. financial services).
    # Ref: Broadbent (1979); Jin et al. (2017); adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "linear_tv": ChannelPriors(
        half_life_mu=math.log(6.0),    # median 6.0 weeks
        half_life_sigma=0.45,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Connected TV (CTV) — streaming inventory; awareness-leaning.
    # Shorter than linear TV (more targetable, closer to mid-funnel)
    # but longer than digital performance. Median 4 weeks.
    # Ref: Robyn Analyst's Guide; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "ctv": ChannelPriors(
        half_life_mu=math.log(4.0),    # median 4.0 weeks
        half_life_sigma=0.45,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Audio / podcast — episodic listening; host-read ads create persistent
    # recall. 3–6 week range is supported by Spotify/iHeart attribution
    # studies and Robyn community benchmarks. Median 4.5 weeks.
    # Ref: Robyn community benchmarks; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "audio_podcast": ChannelPriors(
        half_life_mu=math.log(4.5),    # median 4.5 weeks
        half_life_sigma=0.45,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Display / programmatic — short banner impressions; low recall.
    # Effect is largely immediate; retargeting pixels add a small 1–2 week
    # tail. Median 1.5 weeks; standard dispersion.
    # Ref: Jin et al. (2017); adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "display_programmatic": ChannelPriors(
        half_life_mu=math.log(1.5),    # median 1.5 weeks
        half_life_sigma=0.40,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Affiliate — conversion-driven; last-click attribution model means
    # the sales signal is near-instantaneous. Median 1 week; tighter
    # dispersion (sigma 0.35) because carryover is structurally short.
    # Ref: Robyn Analyst's Guide; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "affiliate": ChannelPriors(
        half_life_mu=math.log(1.0),    # median 1.0 week
        half_life_sigma=0.35,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Email / CRM — opened and acted on within days; nearly immediate
    # engagement. Median 0.7 weeks (~5 days); tighter dispersion because
    # the mechanics (inbox → click → purchase) are well understood.
    # Ref: industry email attribution studies; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "email_crm": ChannelPriors(
        half_life_mu=math.log(0.7),    # median 0.7 weeks (~5 days)
        half_life_sigma=0.35,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),

    # -----------------------------------------------------------------------
    # Promotions / discounting — immediate redemption; the spike is the
    # event itself. Very short carryover; median 0.5 weeks with tight
    # dispersion. Can be effectively zero for flash sales.
    # Ref: Robyn Analyst's Guide; adstock_and_saturation.md §4.1.
    # -----------------------------------------------------------------------
    "promotions_discounting": ChannelPriors(
        half_life_mu=math.log(0.5),    # median 0.5 weeks (~3–4 days)
        half_life_sigma=0.35,
        hill_alpha_mu=_HILL_ALPHA_MU,
        hill_alpha_sigma=_HILL_ALPHA_SIGMA,
    ),
}

# Sanity-check at import time: every category in the taxonomy must have an entry.
_missing = [c for c in DEFAULT_CHANNEL_CATEGORIES if c not in _PRIORS]
if _missing:
    raise RuntimeError(  # pragma: no cover
        f"Missing default priors for categories: {_missing}. "
        "Update _PRIORS in model/priors.py."
    )


def default_priors_for_category(category: str) -> ChannelPriors:
    """Return the default ``ChannelPriors`` for a channel category.

    The returned object is a frozen dataclass; its fields are log-normal
    prior parameters for adstock half-life and Hill saturation alpha.
    Gamma (Hill EC50) is data-driven and is not included here.

    Args:
        category: One of the 10 strings in
            ``wanamaker.data.taxonomy.DEFAULT_CHANNEL_CATEGORIES``.

    Returns:
        ``ChannelPriors`` containing ``half_life_mu``, ``half_life_sigma``,
        ``hill_alpha_mu``, and ``hill_alpha_sigma``.

    Raises:
        ValueError: If ``category`` is not in the default taxonomy.

    Examples:
        >>> priors = default_priors_for_category("paid_search")
        >>> round(priors.half_life_mu, 4)
        0.0
        >>> priors.half_life_sigma
        0.35

        >>> priors = default_priors_for_category("linear_tv")
        >>> import math; round(math.exp(priors.half_life_mu), 1)
        6.0
    """
    if category not in _PRIORS:
        valid = ", ".join(sorted(_PRIORS))
        raise ValueError(
            f"Unknown channel category {category!r}. "
            f"Valid categories: {valid}"
        )
    return _PRIORS[category]
