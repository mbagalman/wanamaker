"""PyMC backend for Wanamaker's Bayesian engine abstraction.

PyMC is the selected modeling engine (Decision 0001). This module is the only
production location where direct PyMC imports belong. Imports of PyMC, ArviZ,
and PyTensor are intentionally lazy so core modules remain importable before
the engine extra has been installed and so import-graph tests can still guard
the local-first rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wanamaker.engine.base import FitResult, Posterior
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    ConvergenceSummary,
    ParameterSummary,
    PosteriorSummary,
    PredictiveSummary,
)
from wanamaker.model.priors import default_priors_for_category

# Naming conventions for the ``pm.Data`` containers wired into the model graph
# at fit time. These are stable and callers (forecast, scenario comparison) use
# them indirectly through the captured names in ``PyMCRawPosterior``.
_CHANNEL_DATA_PREFIX = "data__channel__"
_CONTROL_DATA_PREFIX = "data__control__"
_TARGET_DATA_NAME = "data__target"
_TIME_COORD_NAME = "time"

RuntimeMode = Literal["quick", "standard", "full"]
INTERVAL_MASS = 0.95


@dataclass(frozen=True)
class RuntimeSettings:
    """Sampler settings for a runtime tier."""

    draws: int
    tune: int
    chains: int
    target_accept: float


@dataclass(frozen=True)
class PyMCRawPosterior:
    """PyMC-native payload stored inside ``Posterior.raw``."""

    idata: Any
    model: Any
    model_spec: Any
    date_column: str | None
    target_column: str
    period_labels: list[str] | None = None
    """ISO-8601 date strings for each fitted period, extracted from the input
    data at fit time.  Used by ``posterior_predictive`` to populate
    ``PredictiveSummary.periods`` with real dates rather than integer indices."""

    control_means: dict[str, float] = field(default_factory=dict)
    """Per-control training-data mean used to centre new-plan controls.

    Forecasting on a future plan must apply the *same* centring as fit time —
    re-centring on the future plan would let post-fit covariate drift leak
    into the predictive draws.
    """
    control_stds: dict[str, float] = field(default_factory=dict)
    """Per-control training-data standard deviation used to scale new-plan
    controls. ``0.0`` means the control was constant in training and is left
    uncentered for forecasting (matches the fit-time guard)."""


RUNTIME_SETTINGS: dict[str, RuntimeSettings] = {
    "quick": RuntimeSettings(draws=250, tune=250, chains=2, target_accept=0.9),
    "standard": RuntimeSettings(draws=1000, tune=1000, chains=4, target_accept=0.9),
    "full": RuntimeSettings(draws=2000, tune=2000, chains=4, target_accept=0.95),
}


class PyMCEngine:
    """Concrete PyMC implementation of the ``Engine`` Protocol."""

    name = "pymc"

    def fit(
        self,
        model_spec: Any,
        data: pd.DataFrame,
        seed: int,
        runtime_mode: RuntimeMode,
    ) -> FitResult:
        """Fit ``model_spec`` to ``data`` with PyMC.

        Args:
            model_spec: Engine-neutral model specification.
            data: Input data frame containing target, channel, and control columns.
            seed: Explicit sampler seed.
            runtime_mode: One of ``"quick"``, ``"standard"``, or ``"full"``.

        Returns:
            Fit result containing the PyMC posterior and engine-neutral summary.

        Raises:
            ValueError: If the model spec or data is missing required fields.
            RuntimeError: If PyMC or its backend dependencies are not installed.
        """
        settings = runtime_settings(runtime_mode)
        target_column = _target_column(model_spec, data)
        date_column = getattr(model_spec, "date_column", None) or None
        period_labels: list[str] | None = (
            data[date_column].astype(str).tolist()
            if date_column and date_column in data.columns
            else None
        )
        _validate_columns(model_spec, data, target_column)

        pm, _, _, _ = _require_pymc_stack()
        model, control_means, control_stds = self._build_model(
            pm, model_spec, data, target_column
        )
        with model:
            idata = pm.sample(
                draws=settings.draws,
                tune=settings.tune,
                chains=settings.chains,
                target_accept=settings.target_accept,
                random_seed=seed,
                return_inferencedata=True,
            )

        raw = PyMCRawPosterior(
            idata=idata,
            model=model,
            model_spec=model_spec,
            date_column=date_column,
            target_column=target_column,
            period_labels=period_labels,
            control_means=control_means,
            control_stds=control_stds,
        )
        summary = summarize_inference_data(idata, model_spec, data, target_column, date_column)
        diagnostics = {
            "engine": self.name,
            "runtime_mode": runtime_mode,
            "draws": settings.draws,
            "tune": settings.tune,
            "chains": settings.chains,
            "target_accept": settings.target_accept,
            "convergence": summary.convergence,
        }
        return FitResult(posterior=Posterior(raw=raw), summary=summary, diagnostics=diagnostics)

    def posterior_predictive(
        self,
        posterior: Posterior,
        new_data: pd.DataFrame | None,
        seed: int,
    ) -> PredictiveSummary:
        """Draw posterior predictive samples for the fitted PyMC model.

        ``new_data=None`` returns an in-sample predictive summary keyed by
        the original training periods.

        Passing a future-plan ``DataFrame`` swaps the model's mutable data
        containers via ``pm.set_data`` and resamples — this is the path used
        by ``wanamaker forecast`` and ``wanamaker compare-scenarios``.
        Validation runs before any PyMC code is touched so the engine import
        stays lazy when the plan is malformed.

        Args:
            posterior: A ``Posterior`` whose ``raw`` is a ``PyMCRawPosterior``
                produced by ``self.fit``.
            new_data: Future plan as a ``DataFrame`` with one row per future
                period. Must contain the channels and control columns the
                model was fit on. ``None`` requests in-sample predictive draws.
            seed: Posterior-predictive sampling seed.

        Returns:
            ``PredictiveSummary`` over either the training periods (in-sample)
            or the future plan periods (forecast).

        Raises:
            ValueError: If ``new_data`` is missing required columns, has
                missing or negative spend, or is empty.
        """
        raw = _as_pymc_raw(posterior)
        if new_data is None:
            return self._posterior_predictive_in_sample(raw, seed)
        _validate_new_data(raw, new_data)
        return self._posterior_predictive_forecast(raw, new_data, seed)

    def _posterior_predictive_in_sample(
        self, raw: PyMCRawPosterior, seed: int
    ) -> PredictiveSummary:
        pm, az, _, _ = _require_pymc_stack()
        with raw.model:
            ppc = pm.sample_posterior_predictive(
                raw.idata,
                var_names=["target"],
                random_seed=seed,
                return_inferencedata=True,
                progressbar=False,
            )
        target_values = ppc.posterior_predictive["target"]
        target = target_values.values.reshape(-1, target_values.shape[-1])
        mean = target.mean(axis=0)
        hdi = _hdi_2d(target, az)
        n_periods = target.shape[1]
        periods = raw.period_labels if raw.period_labels else [str(i) for i in range(n_periods)]
        return PredictiveSummary(
            periods=periods,
            mean=mean.tolist(),
            hdi_low=hdi[:, 0].tolist(),
            hdi_high=hdi[:, 1].tolist(),
            interval_mass=INTERVAL_MASS,
        )

    def _posterior_predictive_forecast(
        self, raw: PyMCRawPosterior, new_data: pd.DataFrame, seed: int
    ) -> PredictiveSummary:
        pm, az, _, _ = _require_pymc_stack()
        n_new = len(new_data)
        channels = list(getattr(raw.model_spec, "channels", []))
        control_columns = list(getattr(raw.model_spec, "control_columns", []))

        data_updates: dict[str, NDArray[np.float64]] = {}
        for channel in channels:
            data_updates[f"{_CHANNEL_DATA_PREFIX}{channel.name}"] = (
                new_data[channel.name].to_numpy(dtype=np.float64)
            )
        for control_column in control_columns:
            raw_values = new_data[control_column].to_numpy(dtype=np.float64)
            mean = raw.control_means.get(control_column, 0.0)
            std = raw.control_stds.get(control_column, 0.0)
            data_updates[f"{_CONTROL_DATA_PREFIX}{control_column}"] = (
                _apply_control_centering(raw_values, mean, std)
            )
        # Target's observed values are not used by sample_posterior_predictive,
        # but the dim has to match the new period count or PyMC raises.
        data_updates[_TARGET_DATA_NAME] = np.zeros(n_new, dtype=np.float64)

        coord_updates = {_TIME_COORD_NAME: np.arange(n_new, dtype=np.int64)}

        with raw.model:
            pm.set_data(data_updates, coords=coord_updates)
            ppc = pm.sample_posterior_predictive(
                raw.idata,
                var_names=["target"],
                random_seed=seed,
                return_inferencedata=True,
                progressbar=False,
            )

        target_values = ppc.posterior_predictive["target"]
        target = target_values.values.reshape(-1, target_values.shape[-1])
        mean = target.mean(axis=0)
        hdi = _hdi_2d(target, az)
        periods = _new_data_period_labels(raw, new_data)
        return PredictiveSummary(
            periods=periods,
            mean=mean.tolist(),
            hdi_low=hdi[:, 0].tolist(),
            hdi_high=hdi[:, 1].tolist(),
            interval_mass=INTERVAL_MASS,
        )

    def _build_model(
        self,
        pm: Any,
        model_spec: Any,
        data: pd.DataFrame,
        target_column: str,
    ) -> tuple[Any, dict[str, float], dict[str, float]]:
        """Build the PyMC model with mutable ``pm.Data`` containers.

        Wrapping channel spend, control variables, and the target observation
        in ``pm.Data`` lets ``posterior_predictive`` later swap in a future
        plan via ``pm.set_data`` and re-sample without rebuilding the model.

        Returns:
            ``(model, control_means, control_stds)`` — the model plus the
            training-time centring statistics needed to apply the same
            centring to a future plan's controls.
        """
        pm, _, pt, pytensor = _require_pymc_stack()

        n_periods = len(data)
        coords = {_TIME_COORD_NAME: np.arange(n_periods, dtype=np.int64)}
        channels = list(getattr(model_spec, "channels", []))
        control_columns = list(getattr(model_spec, "control_columns", []))
        lift_test_priors = dict(getattr(model_spec, "lift_test_priors", {}))

        # Data-adaptive scale for priors whose natural magnitude is proportional
        # to the target.  Using 2 × std for the intercept gives a weakly
        # informative prior that covers plausible baseline values without
        # constraining the model.  The same scale is used for the channel
        # coefficient so that a unit saturated-spend signal can produce a
        # full-range contribution.
        target_std = (
            float(data[target_column].std())
            or float(abs(data[target_column].mean()))
            or 1.0
        )

        anchor_priors = dict(getattr(model_spec, "anchor_priors", {}) or {})

        # Pre-compute control centring stats from the training data. The same
        # statistics are reused at predict time so post-fit covariate drift
        # cannot leak into the predictive distribution.
        control_means: dict[str, float] = {}
        control_stds: dict[str, float] = {}
        control_centered_arrays: dict[str, NDArray[np.float64]] = {}
        for control_column in control_columns:
            values = data[control_column].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std())
            control_means[control_column] = mean
            control_stds[control_column] = std
            control_centered_arrays[control_column] = _apply_control_centering(values, mean, std)

        with pm.Model(coords=coords) as model:
            intercept = pm.Normal(
                "intercept",
                mu=float(data[target_column].mean()),
                sigma=target_std * 2.0,
            )
            mu = intercept

            for channel in channels:
                spend_array = data[channel.name].to_numpy(dtype=np.float64)
                spend_data = pm.Data(
                    f"{_CHANNEL_DATA_PREFIX}{channel.name}",
                    spend_array,
                    dims=_TIME_COORD_NAME,
                )
                priors = default_priors_for_category(channel.category)

                # --- adstock half-life ---
                ap_hl = anchor_priors.get(f"channel.{channel.name}.half_life")
                if ap_hl is not None:
                    mu_hl, sigma_hl = _blend_lognormal(
                        priors.half_life_mu, priors.half_life_sigma,
                        ap_hl.mean, ap_hl.sd, ap_hl.weight,
                    )
                else:
                    mu_hl, sigma_hl = priors.half_life_mu, priors.half_life_sigma
                half_life = pm.LogNormal(
                    f"channel__{channel.name}__half_life", mu=mu_hl, sigma=sigma_hl,
                )

                decay = pm.Deterministic(
                    f"channel__{channel.name}__decay",
                    pt.power(0.5, 1.0 / half_life),
                )
                adstocked = _geometric_adstock_tensor(pytensor, pt, spend_data, decay)

                # --- Hill EC50 ---
                # The prior centre uses the training-data spend median so the
                # prior is on the same scale as the observed channel.
                # Forecasts use this same prior — saturation is identifiable
                # from data, not re-derived from the future plan.
                ap_ec50 = anchor_priors.get(f"channel.{channel.name}.ec50")
                if ap_ec50 is not None:
                    mu_ec50, sigma_ec50 = _blend_lognormal(
                        _log_positive_median(spend_array), 1.0,
                        ap_ec50.mean, ap_ec50.sd, ap_ec50.weight,
                    )
                else:
                    mu_ec50, sigma_ec50 = _log_positive_median(spend_array), 1.0
                ec50 = pm.LogNormal(
                    f"channel__{channel.name}__ec50", mu=mu_ec50, sigma=sigma_ec50,
                )

                # --- Hill slope ---
                ap_slope = anchor_priors.get(f"channel.{channel.name}.slope")
                if ap_slope is not None:
                    mu_slope, sigma_slope = _blend_lognormal(
                        priors.hill_alpha_mu, priors.hill_alpha_sigma,
                        ap_slope.mean, ap_slope.sd, ap_slope.weight,
                    )
                else:
                    mu_slope, sigma_slope = priors.hill_alpha_mu, priors.hill_alpha_sigma
                slope = pm.LogNormal(
                    f"channel__{channel.name}__slope", mu=mu_slope, sigma=sigma_slope,
                )

                saturated = _hill_saturation_tensor(pt, adstocked, ec50, slope)

                # --- channel coefficient ---
                coefficient = _coefficient_prior(
                    pm=pm,
                    name=f"channel__{channel.name}__coefficient",
                    lift_prior=lift_test_priors.get(channel.name),
                    anchor_prior=anchor_priors.get(f"channel.{channel.name}.coefficient"),
                    default_sigma=target_std,
                )
                contribution = pm.Deterministic(
                    f"contribution__{channel.name}",
                    coefficient * saturated,
                    dims=_TIME_COORD_NAME,
                )
                mu = mu + contribution

            for control_column in control_columns:
                control_data = pm.Data(
                    f"{_CONTROL_DATA_PREFIX}{control_column}",
                    control_centered_arrays[control_column],
                    dims=_TIME_COORD_NAME,
                )
                beta = pm.Normal(f"control__{control_column}__coefficient", mu=0.0, sigma=1.0)
                mu = mu + beta * control_data

            sigma = pm.HalfNormal("sigma", sigma=float(data[target_column].std() or 1.0))
            target_data = pm.Data(
                _TARGET_DATA_NAME,
                data[target_column].to_numpy(dtype=np.float64),
                dims=_TIME_COORD_NAME,
            )
            pm.Normal(
                "target",
                mu=mu,
                sigma=sigma,
                observed=target_data,
                dims=_TIME_COORD_NAME,
            )
        return model, control_means, control_stds


def runtime_settings(runtime_mode: str) -> RuntimeSettings:
    """Return sampler settings for a runtime mode.

    Raises:
        ValueError: If ``runtime_mode`` is not one of quick, standard, or full.
    """
    try:
        return RUNTIME_SETTINGS[runtime_mode]
    except KeyError as exc:
        valid = ", ".join(RUNTIME_SETTINGS)
        raise ValueError(
            f"unknown runtime_mode {runtime_mode!r}; expected one of: {valid}"
        ) from exc


def summarize_inference_data(
    idata: Any,
    model_spec: Any,
    data: pd.DataFrame,
    target_column: str,
    date_column: str | None,
) -> PosteriorSummary:
    """Create a ``PosteriorSummary`` from PyMC / ArviZ inference data."""
    _, az, _, _ = _require_pymc_stack()
    posterior = idata.posterior
    az_summary = az.summary(idata, hdi_prob=INTERVAL_MASS)

    parameters = []
    for variable in posterior.data_vars:
        if variable.startswith("contribution__") or variable == "target":
            continue
        values = np.asarray(posterior[variable].values)
        if values.ndim > 2:
            continue
        flat = values.reshape(-1)
        hdi_low, hdi_high = _hdi_1d(flat, az)
        row = az_summary.loc[variable] if variable in az_summary.index else None
        parameters.append(
            ParameterSummary(
                name=variable.replace("__", "."),
                mean=float(np.mean(flat)),
                sd=float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
                hdi_low=float(hdi_low),
                hdi_high=float(hdi_high),
                interval_mass=INTERVAL_MASS,
                r_hat=_optional_summary_value(row, "r_hat"),
                ess_bulk=_optional_summary_value(row, "ess_bulk"),
            )
        )

    channel_contributions = _channel_contribution_summaries(posterior, model_spec, data, az)
    convergence = _convergence_summary(az_summary, idata)
    predictive = _in_sample_predictive_summary(idata, data, date_column, az)
    return PosteriorSummary(
        parameters=parameters,
        channel_contributions=channel_contributions,
        convergence=convergence,
        in_sample_predictive=predictive,
    )


def _require_pymc_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import arviz as az
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyMC backend requires pymc, arviz, and pytensor. "
            "Install Wanamaker with its production dependencies."
        ) from exc
    return pm, az, pt, pytensor


def _target_column(model_spec: Any, data: pd.DataFrame) -> str:
    target_column = getattr(model_spec, "target_column", None)
    if not target_column:
        raise ValueError(
            "ModelSpec.target_column is required. "
            "Pass the name of the target metric column when constructing ModelSpec."
        )
    return str(target_column)


def _validate_columns(model_spec: Any, data: pd.DataFrame, target_column: str) -> None:
    required = [target_column]
    required.extend(channel.name for channel in getattr(model_spec, "channels", []))
    required.extend(getattr(model_spec, "control_columns", []))
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing columns required by ModelSpec: {missing}")


def _as_pymc_raw(posterior: Posterior) -> PyMCRawPosterior:
    raw = posterior.raw
    if not isinstance(raw, PyMCRawPosterior):
        raise TypeError("posterior.raw is not a PyMCRawPosterior")
    return raw


def _validate_new_data(raw: PyMCRawPosterior, new_data: pd.DataFrame) -> None:
    """Reject a future plan that does not match the fitted model's contract.

    Validation runs *before* any PyMC code is touched so the lazy-import
    contract is preserved when a caller hands in a malformed plan.
    """
    if len(new_data) == 0:
        raise ValueError("new_data is empty; expected at least one future period")

    channels = list(getattr(raw.model_spec, "channels", []))
    control_columns = list(getattr(raw.model_spec, "control_columns", []))
    required = [channel.name for channel in channels] + list(control_columns)
    missing = [column for column in required if column not in new_data.columns]
    if missing:
        raise ValueError(
            f"new_data is missing columns required by the fitted ModelSpec: {missing}"
        )
    for column in required:
        if new_data[column].isna().any():
            raise ValueError(f"new_data has missing values in column {column!r}")
    for channel in channels:
        if (new_data[channel.name] < 0).any():
            raise ValueError(
                f"new_data has negative spend for channel {channel.name!r}; "
                "spend must be non-negative."
            )


def _apply_control_centering(
    values: NDArray[np.float64], mean: float, std: float
) -> NDArray[np.float64]:
    """Centre and scale a control column using fit-time training statistics."""
    centered = values - mean
    if std > 0:
        centered = centered / std
    return centered


def _new_data_period_labels(raw: PyMCRawPosterior, new_data: pd.DataFrame) -> list[str]:
    """Period labels for forecast output.

    Prefers the model's date column from the future plan; falls back to
    integer indices when the plan does not carry one (e.g. an in-memory
    DataFrame the caller built without dates).
    """
    if raw.date_column and raw.date_column in new_data.columns:
        return new_data[raw.date_column].astype(str).tolist()
    return [str(i) for i in range(len(new_data))]


def _geometric_adstock_tensor(pytensor: Any, pt: Any, spend: Any, decay: Any) -> Any:
    """Geometric adstock as a ``pytensor.scan`` over the spend tensor.

    Using a scan instead of a Python-unrolled loop lets the model accept
    variable-length data via ``pm.Data`` containers — the scan iterates over
    whatever length the spend tensor actually has at evaluate time, which is
    what makes future-plan posterior predictive draws possible.

    For each step the recurrence is ``s_t = x_t + decay * s_{t-1}`` with
    ``s_{-1} = 0``, identical to the previous unrolled implementation.
    """
    spend_tensor = pt.as_tensor_variable(spend, dtype="float64")
    initial = pt.zeros((), dtype="float64")
    result, _ = pytensor.scan(
        fn=lambda x_t, prev: x_t + decay * prev,
        sequences=[spend_tensor],
        outputs_info=[initial],
        strict=True,
    )
    return result


def _hill_saturation_tensor(pt: Any, spend: Any, ec50: Any, slope: Any) -> Any:
    positive_spend = pt.maximum(spend, 0.0)
    numerator = pt.power(positive_spend, slope)
    denominator = numerator + pt.power(ec50, slope) + 1e-12
    return numerator / denominator


def _blend_lognormal(
    mu_default: float,
    sigma_default: float,
    prev_mean: float,
    prev_sd: float,
    weight: float,
) -> tuple[float, float]:
    """Blend default and previous-posterior LogNormal parameters for anchoring.

    Converts the previous posterior summary (mean, sd) — both in linear scale
    — to lognormal log-space parameters via the exact lognormal moment equations:

        mu_log  = log(mean) - 0.5 * log(1 + cv²)
        sig_log = sqrt(log(1 + cv²))   where cv = sd / mean

    Then blends them linearly with the default parameters:

        mu_blend    = (1 - w) * mu_default  + w * mu_log_prev
        sigma_blend = (1 - w) * sigma_default + w * sigma_log_prev

    Args:
        mu_default: Default lognormal mu (log-space mean).
        sigma_default: Default lognormal sigma (log-space std).
        prev_mean: Previous posterior mean in linear scale (must be > 0).
        prev_sd: Previous posterior standard deviation in linear scale.
        weight: Anchoring weight ``w`` in ``[0, 1]``.

    Returns:
        ``(mu_blend, sigma_blend)`` — log-space parameters for a LogNormal prior.
    """
    prev_mean_pos = max(prev_mean, 1e-9)
    cv2 = (prev_sd / prev_mean_pos) ** 2
    mu_log_prev = math.log(prev_mean_pos) - 0.5 * math.log1p(cv2)
    sigma_log_prev = math.sqrt(math.log1p(cv2))
    mu_blend = (1.0 - weight) * mu_default + weight * mu_log_prev
    sigma_blend = (1.0 - weight) * sigma_default + weight * max(sigma_log_prev, 1e-6)
    return mu_blend, sigma_blend


def _coefficient_prior(
    pm: Any,
    name: str,
    lift_prior: Any | None,
    anchor_prior: Any | None,
    default_sigma: float,
) -> Any:
    """Return the channel coefficient prior.

    Priority (highest to lowest):
    1. Lift-test prior — explicit informative ROI constraint from an experiment.
    2. Anchor prior — mixture of default and previous posterior (FR-4.4).
    3. Default HalfNormal — uninformative, first-run baseline.
    """
    if lift_prior is not None:
        return pm.TruncatedNormal(
            name,
            mu=float(lift_prior.mean_roi),
            sigma=float(lift_prior.sd_roi),
            lower=0.0,
        )
    if anchor_prior is not None:
        # Blend: default HalfNormal has effective mu=0; anchored component pulls
        # toward the previous posterior mean with weight w.
        w = float(anchor_prior.weight)
        blended_mu = w * float(anchor_prior.mean)
        blended_sigma = max(
            (1.0 - w) * default_sigma + w * float(anchor_prior.sd), 1e-6
        )
        return pm.TruncatedNormal(name, mu=blended_mu, sigma=blended_sigma, lower=0.0)
    return pm.HalfNormal(name, sigma=default_sigma)


def _log_positive_median(values: NDArray[np.float64]) -> float:
    positive = values[values > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    return float(np.log(max(median, 1e-6)))


def _channel_contribution_summaries(
    posterior: Any,
    model_spec: Any,
    data: pd.DataFrame,
    az: Any,
) -> list[ChannelContributionSummary]:
    channels = list(getattr(model_spec, "channels", []))
    totals_by_channel: dict[str, NDArray[np.float64]] = {}
    mean_totals = []
    for channel in channels:
        var_name = f"contribution__{channel.name}"
        if var_name not in posterior:
            continue
        values = np.asarray(posterior[var_name].values)
        flattened = values.reshape(-1, values.shape[-1])
        totals = flattened.sum(axis=1)
        totals_by_channel[channel.name] = totals
        mean_totals.append(float(np.mean(totals)))

    total_media = sum(mean_totals)
    out = []
    for channel in channels:
        totals = totals_by_channel.get(channel.name)
        if totals is None:
            continue
        hdi_low, hdi_high = _hdi_1d(totals, az)
        spend = data[channel.name].to_numpy(dtype=np.float64)
        total_spend = float(np.sum(spend))
        roi_samples = totals / total_spend if total_spend > 0 else np.zeros_like(totals)
        roi_hdi_low, roi_hdi_high = _hdi_1d(roi_samples, az)
        mean_total = float(np.mean(totals))
        out.append(
            ChannelContributionSummary(
                channel=channel.name,
                mean_contribution=mean_total,
                hdi_low=float(hdi_low),
                hdi_high=float(hdi_high),
                interval_mass=INTERVAL_MASS,
                share_of_effect=mean_total / total_media if total_media else 0.0,
                roi_mean=float(np.mean(roi_samples)),
                roi_hdi_low=float(roi_hdi_low),
                roi_hdi_high=float(roi_hdi_high),
                observed_spend_min=float(np.min(spend)) if spend.size else 0.0,
                observed_spend_max=float(np.max(spend)) if spend.size else 0.0,
                spend_invariant=bool(np.std(spend) == 0.0),
            )
        )
    return out


def _convergence_summary(az_summary: Any, idata: Any) -> ConvergenceSummary:
    r_hat = az_summary["r_hat"].dropna() if "r_hat" in az_summary else []
    ess_bulk = az_summary["ess_bulk"].dropna() if "ess_bulk" in az_summary else []
    sample_stats = getattr(idata, "sample_stats", None)
    divergences = 0
    if sample_stats is not None and "diverging" in sample_stats:
        divergences = int(np.asarray(sample_stats["diverging"].values).sum())
    n_chains = int(idata.posterior.sizes.get("chain", 0))
    n_draws = int(idata.posterior.sizes.get("draw", 0))
    return ConvergenceSummary(
        max_r_hat=float(np.max(r_hat)) if len(r_hat) else None,
        min_ess_bulk=float(np.min(ess_bulk)) if len(ess_bulk) else None,
        n_divergences=divergences,
        n_chains=n_chains,
        n_draws=n_draws,
    )


def _in_sample_predictive_summary(
    idata: Any,
    data: pd.DataFrame,
    date_column: str | None,
    az: Any,
) -> PredictiveSummary | None:
    posterior_predictive = getattr(idata, "posterior_predictive", None)
    if posterior_predictive is None or "target" not in posterior_predictive:
        return None
    n_periods = len(data)
    target = np.asarray(posterior_predictive["target"].values).reshape(-1, n_periods)
    mean = target.mean(axis=0)
    hdi = _hdi_2d(target, az)
    if date_column and date_column in data.columns:
        periods = data[date_column].astype(str).tolist()
    else:
        periods = [str(i) for i in range(n_periods)]
    return PredictiveSummary(
        periods=periods,
        mean=mean.tolist(),
        hdi_low=hdi[:, 0].tolist(),
        hdi_high=hdi[:, 1].tolist(),
        interval_mass=INTERVAL_MASS,
    )


def _hdi_1d(values: NDArray[np.float64], az: Any) -> tuple[float, float]:
    """Return the 95% highest-density interval for a 1-D sample array.

    Uses ArviZ's ``hdi`` function, which finds the shortest interval containing
    ``INTERVAL_MASS`` of the posterior mass.  For skewed posteriors (log-normal
    half-life, EC50, slope samples) this differs meaningfully from an equal-
    tailed quantile interval.
    """
    if values.size == 0:
        return 0.0, 0.0
    result = az.hdi(values, hdi_prob=INTERVAL_MASS)
    return float(result[0]), float(result[1])


def _hdi_2d(values: NDArray[np.float64], az: Any) -> NDArray[np.float64]:
    """Return 95% HDI bounds for each column of a 2-D sample matrix.

    Args:
        values: Shape ``(n_samples, n_periods)``.
        az: ArviZ module returned by ``_require_pymc_stack()``.

    Returns:
        Shape ``(n_periods, 2)`` array of ``[hdi_low, hdi_high]`` per period.
    """
    n_periods = values.shape[1]
    out = np.empty((n_periods, 2), dtype=float)
    for i in range(n_periods):
        lo, hi = _hdi_1d(values[:, i], az)
        out[i] = [lo, hi]
    return out


def _optional_summary_value(row: Any, column: str) -> float | None:
    if row is None or column not in row:
        return None
    value = row[column]
    return None if pd.isna(value) else float(value)
