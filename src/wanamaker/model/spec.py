"""Engine-agnostic model specification (FR-3.1, FR-3.4, FR-4.4).

A ``ModelSpec`` is a declarative description of the model: which channels,
which transforms, which priors, and how to run the sampler. It is pure
data — no engine, no sampling. Engines consume it via
``Engine.fit(model_spec, data, seed, runtime_mode)``.

Keeping the spec data-only means the same description can be rendered into
PyMC, NumPyro, or Stan code without leakage between layers.

**Design rules:**

- All types are frozen dataclasses so they are safely hashable, cacheable,
  and comparable.  The "frozen" guarantee covers attribute reassignment;
  mutable containers (``list``, ``dict``) used as field types are frozen
  references, not deeply immutable.  Callers must not mutate the contents
  after construction.
- ``ModelSpec`` carries *resolved* state: priors are already looked up,
  channel categories are already validated, the runtime mode is already
  set.  Building a ``ModelSpec`` from raw config is the responsibility of
  the config → model wiring layer, not of this module.
- Optional fields default to ``None`` or empty containers so a
  ``ModelSpec`` can be constructed incrementally for tests without
  specifying every field.

**Field summary:**

    channels           Per-channel structural description
    target_column      Target metric column name (required)
    date_column        Date column name (required)
    control_columns    Non-media predictor column names
    frequency          Temporal resolution; ``"weekly"`` only in v1
    channel_priors     Per-channel prior overrides (empty = use defaults)
    lift_test_priors   Informative ROI priors from lift test CSVs
    holdout_config     Hold-out window for Trust Card accuracy
    seasonality        Fourier seasonality / trend treatment
    anchor_priors      Mixture priors for refresh posterior anchoring
    spend_invariant_channels  Channels with no spend variation
    runtime_mode       Sampler effort tier (quick / standard / full)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from wanamaker.model.priors import ChannelPriors

# ---------------------------------------------------------------------------
# Per-channel structural spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSpec:
    """Structural description of a single media channel.

    Attributes:
        name: Column name in the input data frame.
        category: One of the 10 strings in
            ``wanamaker.data.taxonomy.DEFAULT_CHANNEL_CATEGORIES``.
            Used to look up default priors when ``channel_priors`` does
            not contain an entry for this channel.
        adstock_family: Which carryover family to use. ``"geometric"``
            (default) uses a single decay parameter; ``"weibull"`` uses
            the two-parameter Weibull adstock defined in
            ``wanamaker.transforms.adstock``.
    """

    name: str
    category: str
    adstock_family: Literal["geometric", "weibull"] = "geometric"


# ---------------------------------------------------------------------------
# Lift-test calibration prior (FR-1.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiftPrior:
    """Informative ROI prior derived from a lift test result.

    When a channel has a lift test, the measured incremental lift replaces
    the default uninformative coefficient prior with a Normal prior centred
    at the observed ROI estimate.

    The prior is placed on the channel *coefficient* (revenue per unit of
    transformed spend), not directly on adstock or saturation parameters,
    which continue to use their default priors.

    Reference: FR-1.3 — lift-test calibration.

    Attributes:
        mean_roi: Point estimate of revenue per unit of spend from the
            lift test (e.g. roi_estimate). Used as the mean of the Normal
            coefficient prior.
        sd_roi: Standard deviation of the ROI estimate from the lift test.
            Wider intervals result in a less informative calibration prior.
            Must be strictly positive.
        confidence: Nominal confidence level of the lift test interval
            (e.g. 0.95).  Stored for auditing; not used in the prior
            directly (the sd already encodes the uncertainty).
        n_tests: Number of underlying lift-test rows that contributed to
            this prior. ``1`` for a single-test prior. When multiple
            tests for the same channel are pooled (#78), this records
            how many rows the precision-weighted pool consumed so the
            Trust Card calibration message can report the right total.
    """

    mean_roi: float
    sd_roi: float
    confidence: float = 0.95
    n_tests: int = 1

    def __post_init__(self) -> None:
        """Validate lift-prior uncertainty."""
        if not math.isfinite(self.sd_roi) or self.sd_roi <= 0.0:
            raise ValueError(f"sd_roi must be strictly positive; got {self.sd_roi!r}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1); got {self.confidence!r}")
        if self.n_tests < 1:
            raise ValueError(f"n_tests must be >= 1; got {self.n_tests!r}")


# ---------------------------------------------------------------------------
# Hold-out configuration (for Trust Card accuracy, FR-5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutConfig:
    """Hold-out period excluded from fitting and used for accuracy evaluation.

    The engine withholds observations in ``[start_date, end_date]`` from
    the likelihood during sampling.  After fitting, the posterior predictive
    over this window is compared to actual values to produce the out-of-sample
    MAPE / coverage statistics in the Trust Card.

    Both dates are ISO-8601 strings (``YYYY-MM-DD``) matching the values in
    the ``date_column`` of the input data.

    Attributes:
        start_date: First date of the hold-out window (inclusive).
        end_date: Last date of the hold-out window (inclusive).
    """

    start_date: str
    end_date: str


# ---------------------------------------------------------------------------
# Seasonality specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonalitySpec:
    """Trend and seasonality treatment applied to the model intercept.

    Wanamaker v1 supports two optional adjustments to the additive baseline:

    1. **Fourier seasonality** — a sum of ``fourier_order`` sine / cosine
       pairs at the specified annual period, capturing repeating
       within-year patterns (holiday lift, summer trough, etc.).
       Setting ``fourier_order = 0`` disables seasonality.

    2. **Linear trend** — a linear time index term.  Useful when the target
       series shows a sustained upward or downward trend over the training
       window.  Disabled by default because trends can be confounded with
       marketing effects when data are short.

    Reference: standard Bayesian time-series decomposition; see also
    Prophet (Taylor & Letham, 2018) for the Fourier seasonality formulation.

    Attributes:
        fourier_order: Number of Fourier (sine/cosine) pairs for annual
            seasonality.  0 = no seasonality term.  Values of 2–6 are
            typical for weekly data.
        period_weeks: Seasonality period in weeks.  Defaults to 365.25/7
            ≈ 52.18 for annual seasonality on weekly data.
        include_trend: If ``True``, add a linear trend term to the baseline.
    """

    fourier_order: int = 2
    period_weeks: float = 365.25 / 7.0   # ≈ 52.18 weeks
    include_trend: bool = False


# ---------------------------------------------------------------------------
# Refresh posterior anchoring (FR-4.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredPrior:
    """Single-parameter mixture prior for refresh posterior anchoring.

    Implements the posterior anchoring formula from FR-4.4::

        Prior_new(θ) = (1 - weight) · Prior_default(θ)
                     + weight · Normal(mean, sd)

    where ``mean`` and ``sd`` are taken from the previous run's
    ``ParameterSummary`` for the same parameter, and ``weight`` is the
    global anchoring strength configured by the user (default 0.3 per
    FR-4.4; see issue #18 for tuning).

    The ``anchor_priors`` dict on ``ModelSpec`` is keyed by the stable
    parameter name from ``ParameterSummary.name`` (e.g.
    ``"channel.paid_search.half_life"``).  The engine is responsible for
    translating this mixture into its native prior syntax.

    Attributes:
        mean: Posterior mean from the previous run.
        sd: Posterior standard deviation from the previous run.
        weight: Mixture weight for the previous posterior component.
            Must be in ``[0, 1]``. 0 = no anchoring; 1 = full anchoring.
    """

    mean: float
    sd: float
    weight: float


# ---------------------------------------------------------------------------
# Top-level model specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Complete engine-agnostic model specification.

    This is the single object passed to ``Engine.fit``.  It carries
    everything the engine needs to build the probabilistic program.

    **Required fields:** ``channels``, ``target_column``, ``date_column``.

    All other fields have sensible defaults so tests and interactive use
    can construct minimal specs without specifying every option.

    Attributes:
        channels: Ordered list of media channels to include in the model.
            At least one channel is required for a meaningful fit.
        target_column: Name of the column in the input data frame that
            contains the target metric (e.g. ``"revenue"``).
        date_column: Name of the date column in the input data frame.
            The column must be parseable by ``pd.to_datetime``.
        control_columns: Non-media predictor column names (e.g. promotions,
            holiday flags, macroeconomic controls).
        frequency: Temporal resolution.  ``"weekly"`` is the only supported
            value in v1.  Reserved for future daily / monthly variants.
        channel_priors: Per-channel prior overrides.  Keys are channel
            *names* (matching ``ChannelSpec.name``).  If a channel is
            absent from this dict the engine falls back to
            ``default_priors_for_category(channel.category)``.
        lift_test_priors: Informative ROI priors from lift tests, keyed by
            channel name.  Replaces the uninformative coefficient prior for
            channels that have a lift test result.
        holdout_config: Optional hold-out window for Trust Card out-of-
            sample accuracy evaluation.  ``None`` means no holdout.
        seasonality: Optional Fourier seasonality and/or linear trend.
            ``None`` means no seasonality adjustment.
        anchor_priors: Parameter-level mixture priors for refresh posterior
            anchoring (FR-4.4).  Keys are stable parameter names from
            ``ParameterSummary.name``.  ``None`` means a fresh fit with no
            anchoring (first run, or explicit override).
        spend_invariant_channels: Set of channel names where spend has
            insufficient variation to estimate saturation.  The engine fixes
            Hill parameters at their prior medians for these channels and
            flags them in the Trust Card (FR-3.2).
        runtime_mode: Sampler effort tier.  One of ``"quick"``,
            ``"standard"`` (default), or ``"full"``.  Passed through to
            the engine backend; see ``engine/pymc.py`` for the mapping to
            concrete sampler settings.
    """

    # -- Required ---------------------------------------------------------
    channels: list[ChannelSpec]
    target_column: str
    date_column: str

    # -- Optional with defaults -------------------------------------------
    control_columns: list[str] = field(default_factory=list)
    frequency: Literal["weekly"] = "weekly"
    channel_priors: dict[str, ChannelPriors] = field(default_factory=dict)
    lift_test_priors: dict[str, LiftPrior] = field(default_factory=dict)
    holdout_config: HoldoutConfig | None = None
    seasonality: SeasonalitySpec | None = None
    anchor_priors: dict[str, AnchoredPrior] | None = None
    spend_invariant_channels: set[str] = field(default_factory=set)
    runtime_mode: str = "standard"
