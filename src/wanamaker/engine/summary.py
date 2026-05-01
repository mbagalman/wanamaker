"""Engine-neutral typed posterior summaries.

These types are the stable contract between the engine backend and all
downstream modules (refresh/diff, trust_card, reports, forecast, advisor).
Core modules must depend on these typed summaries rather than reaching
into ``Posterior.raw``, which is engine-specific and opaque.

The raw engine object is still accessible for expert users via
``Posterior.raw``, but it must not appear in any import that crosses the
engine boundary.

All summary types are frozen dataclasses so they can be safely cached,
compared, and serialised to ``summary.json`` in the run artifact.

**Credible interval mass:** All HDI bounds in this module use
``interval_mass = 0.95`` (95%) by default, matching the BRD/PRD acceptance
criteria (FR-3.1) and user-facing wording. Engine backends must compute
intervals at the mass specified in each summary object, not the ArviZ
default of 0.94. Pass ``interval_mass=0.95`` explicitly when calling
``az.hdi`` or the equivalent in the chosen backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterSummary:
    """Marginal posterior summary for a single scalar parameter.

    Covers channel-level parameters (ROI coefficient, adstock half-life,
    saturation EC50, saturation slope), baseline, and control coefficients.
    """

    name: str
    """Stable dot-separated identifier, e.g. ``"channel.paid_search.roi"``."""

    mean: float
    sd: float
    hdi_low: float
    """Lower bound of the highest-density interval."""
    hdi_high: float
    """Upper bound of the highest-density interval."""
    interval_mass: float = 0.95
    """Probability mass of the HDI. Must be 0.95 for all v1 outputs (FR-3.1)."""
    r_hat: float | None = None
    """Gelman-Rubin convergence statistic. None if only one chain was run."""
    ess_bulk: float | None = None
    """Bulk effective sample size."""


@dataclass(frozen=True)
class ChannelContributionSummary:
    """Posterior summary of a channel's absolute contribution to the target.

    Used in the executive summary waterfall, scenario comparison, and
    the Experiment Advisor's spend-vs-uncertainty analysis.
    """

    channel: str
    mean_contribution: float
    """Expected contribution in target units (e.g. revenue)."""
    hdi_low: float
    hdi_high: float
    interval_mass: float = 0.95
    """Probability mass of the HDI. Must be 0.95 for all v1 outputs (FR-3.1)."""
    share_of_effect: float = 0.0
    """Mean contribution as a fraction of total media contribution."""
    roi_mean: float = 0.0
    """Expected return per unit of spend."""
    roi_hdi_low: float = 0.0
    roi_hdi_high: float = 0.0
    observed_spend_min: float = 0.0
    """Minimum observed spend per period during the model training window."""
    observed_spend_max: float = 0.0
    """Maximum observed spend per period during the model training window.

    Used by forecast and scenario comparison to flag extrapolation: any
    plan that exceeds ``observed_spend_max`` for this channel is outside
    the range the model has observed and must be visually distinguished
    (FR-5.1) and warned about (FR-5.2).
    """
    spend_invariant: bool = False
    """True when saturation could not be estimated from data (FR-3.2)."""


@dataclass(frozen=True)
class PredictiveSummary:
    """Posterior predictive summary for a set of time periods.

    Returned by ``Engine.posterior_predictive`` and used by
    ``wanamaker.forecast`` for scenario comparison.
    """

    periods: list[str]
    """ISO-8601 date strings, one per row, matching the input data order."""
    mean: list[float]
    hdi_low: list[float]
    hdi_high: list[float]
    interval_mass: float = 0.95
    """Probability mass of the HDI. Must be 0.95 for all v1 outputs (FR-3.1)."""
    draws: list[list[float]] | None = None
    """Raw posterior predictive draws, shape ``(n_draws, n_periods)``.

    Included so downstream consumers (like risk-adjusted allocation) can compute
    tail risks (e.g. CVaR, P_loss) without dropping back to engine-native code.
    """


@dataclass(frozen=True)
class ConvergenceSummary:
    """Aggregate convergence statistics for a fit.

    Feeds the Trust Card ``convergence`` dimension (FR-5.4).
    """

    max_r_hat: float | None
    """Worst (highest) R-hat across all parameters. Target: < 1.01.
    ``None`` when R-hat is not computable (e.g. single-chain run)."""
    min_ess_bulk: float | None
    """Worst (lowest) bulk ESS across all parameters. Target: > 400.
    ``None`` when ESS is not computable (e.g. single-chain run)."""
    n_divergences: int
    """Number of divergent transitions (NUTS). Target: 0."""
    n_chains: int
    n_draws: int


@dataclass(frozen=True)
class PosteriorSummary:
    """Complete engine-neutral summary of a model fit.

    This is what gets written to ``summary.json`` in the run artifact and
    what all downstream modules consume. Backends produce this; feature
    code consumes it.
    """

    parameters: list[ParameterSummary] = field(default_factory=list)
    channel_contributions: list[ChannelContributionSummary] = field(default_factory=list)
    convergence: ConvergenceSummary | None = None
    in_sample_predictive: PredictiveSummary | None = None
    """Posterior predictive over the training period (for holdout accuracy)."""
