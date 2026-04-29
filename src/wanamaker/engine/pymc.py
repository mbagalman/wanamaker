"""PyMC backend for Wanamaker's Bayesian engine abstraction.

PyMC is the selected modeling engine (Decision 0001). This module is the only
production location where direct PyMC imports belong. Imports of PyMC, ArviZ,
and PyTensor are intentionally lazy so core modules remain importable before
the engine extra has been installed and so import-graph tests can still guard
the local-first rules.
"""

from __future__ import annotations

from dataclasses import dataclass
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

        pm, _, _ = _require_pymc_stack()
        with self._build_model(pm, model_spec, data, target_column) as model:
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

        The current implementation supports the fitted model's existing data.
        Forward plans with new spend require the planned mutable data contract
        in the forecast workstream.
        """
        raw = _as_pymc_raw(posterior)
        if new_data is not None:
            raise NotImplementedError(
                "PyMC posterior prediction for new data requires the forecast "
                "FitContext / mutable-data contract. Existing-data posterior "
                "predictive draws are supported."
            )

        pm, az, _ = _require_pymc_stack()
        with raw.model:
            ppc = pm.sample_posterior_predictive(
                raw.idata,
                var_names=["target"],
                random_seed=seed,
                return_inferencedata=True,
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

    def _build_model(
        self,
        pm: Any,
        model_spec: Any,
        data: pd.DataFrame,
        target_column: str,
    ) -> Any:
        _, pytensor, pt = _require_pymc_stack()
        del pytensor

        coords = {"time": np.arange(len(data), dtype=np.int64)}
        channels = list(getattr(model_spec, "channels", []))
        control_columns = list(getattr(model_spec, "control_columns", []))

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

        with pm.Model(coords=coords) as model:
            intercept = pm.Normal(
                "intercept",
                mu=float(data[target_column].mean()),
                sigma=target_std * 2.0,
            )
            mu = intercept

            for channel in channels:
                spend = data[channel.name].to_numpy(dtype=np.float64)
                priors = default_priors_for_category(channel.category)
                half_life = pm.LogNormal(
                    f"channel__{channel.name}__half_life",
                    mu=priors.half_life_mu,
                    sigma=priors.half_life_sigma,
                )
                decay = pm.Deterministic(
                    f"channel__{channel.name}__decay",
                    pt.power(0.5, 1.0 / half_life),
                )
                adstocked = _geometric_adstock_tensor(pt, spend, decay)
                ec50 = pm.LogNormal(
                    f"channel__{channel.name}__ec50",
                    mu=_log_positive_median(spend),
                    sigma=1.0,
                )
                slope = pm.LogNormal(
                    f"channel__{channel.name}__slope",
                    mu=priors.hill_alpha_mu,
                    sigma=priors.hill_alpha_sigma,
                )
                saturated = _hill_saturation_tensor(pt, adstocked, ec50, slope)
                coefficient = pm.HalfNormal(
                    f"channel__{channel.name}__coefficient",
                    sigma=target_std,
                )
                contribution = pm.Deterministic(
                    f"contribution__{channel.name}",
                    coefficient * saturated,
                    dims="time",
                )
                mu = mu + contribution

            for control_column in control_columns:
                values = data[control_column].to_numpy(dtype=np.float64)
                centered = values - values.mean()
                scale = values.std()
                if scale > 0:
                    centered = centered / scale
                beta = pm.Normal(f"control__{control_column}__coefficient", mu=0.0, sigma=1.0)
                mu = mu + beta * centered

            sigma = pm.HalfNormal("sigma", sigma=float(data[target_column].std() or 1.0))
            pm.Normal(
                "target",
                mu=mu,
                sigma=sigma,
                observed=data[target_column].to_numpy(dtype=np.float64),
                dims="time",
            )
        return model


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
    _, az, _ = _require_pymc_stack()
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


def _require_pymc_stack() -> tuple[Any, Any, Any]:
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyMC backend requires pymc, arviz, and pytensor. "
            "Install Wanamaker with its production dependencies."
        ) from exc
    return pm, az, pt


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


def _geometric_adstock_tensor(pt: Any, spend: NDArray[np.float64], decay: Any) -> Any:
    values = pt.as_tensor_variable(spend.astype(np.float64))
    outputs = []
    previous = pt.as_tensor_variable(0.0)
    for index in range(len(spend)):
        previous = values[index] + decay * previous
        outputs.append(previous)
    return pt.stack(outputs)


def _hill_saturation_tensor(pt: Any, spend: Any, ec50: Any, slope: Any) -> Any:
    positive_spend = pt.maximum(spend, 0.0)
    numerator = pt.power(positive_spend, slope)
    denominator = numerator + pt.power(ec50, slope) + 1e-12
    return numerator / denominator


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
