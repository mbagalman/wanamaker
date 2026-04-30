"""Generate Phase 1 benchmark datasets for public examples and refresh stability.

Outputs:
    benchmark_data/public_example.csv
    benchmark_data/public_example_metadata.json
    benchmark_data/refresh_stability_base.csv
    benchmark_data/refresh_stability_4_weeks_added.csv
    benchmark_data/refresh_stability_metadata.json

Run from the repository root:

    python benchmark_data/generate_phase1_benchmarks.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wanamaker.transforms.adstock import geometric_adstock, half_life_to_decay
from wanamaker.transforms.saturation import hill_saturation

DATE_COLUMN = "week"
TARGET_COLUMN = "revenue"
PUBLIC_SEED = 20260430
REFRESH_SEED = 20260431


@dataclass(frozen=True)
class ChannelPlan:
    """Ground-truth recipe for one benchmark media channel."""

    name: str
    category: str
    base_spend: float
    spend_noise: float
    half_life: float
    ec50: float
    slope: float
    coefficient: float
    phase: float


PUBLIC_CHANNELS: tuple[ChannelPlan, ...] = (
    ChannelPlan("paid_search", "paid_search", 18000, 0.18, 0.8, 16000, 1.2, 13000, 0.1),
    ChannelPlan("paid_social", "paid_social", 15000, 0.28, 1.6, 14500, 1.5, 9800, 1.3),
    ChannelPlan("linear_tv", "linear_tv", 28000, 0.35, 6.0, 29000, 1.8, 21000, 2.4),
    ChannelPlan("ctv", "ctv", 15000, 0.32, 4.0, 15500, 1.7, 12500, 0.8),
    ChannelPlan("online_video", "video", 12000, 0.36, 3.0, 12500, 1.6, 10300, 1.9),
    ChannelPlan("display", "display_programmatic", 8000, 0.24, 1.2, 7600, 1.3, 5900, 2.8),
    ChannelPlan("affiliate", "affiliate", 5500, 0.20, 0.7, 5200, 1.1, 4700, 0.4),
    ChannelPlan("email", "email_crm", 2600, 0.18, 0.5, 2500, 1.1, 3900, 2.0),
)

REFRESH_CHANNELS: tuple[ChannelPlan, ...] = (
    ChannelPlan("paid_search", "paid_search", 16000, 0.08, 0.8, 15000, 1.2, 11000, 0.0),
    ChannelPlan("paid_social", "paid_social", 13000, 0.08, 1.5, 13000, 1.4, 9000, 1.1),
    ChannelPlan("linear_tv", "linear_tv", 24000, 0.07, 6.0, 24000, 1.8, 18500, 2.2),
    ChannelPlan("ctv", "ctv", 11000, 0.07, 4.0, 11000, 1.6, 10500, 0.7),
    ChannelPlan("display", "display_programmatic", 7000, 0.06, 1.2, 7000, 1.3, 5200, 2.6),
    ChannelPlan("affiliate", "affiliate", 5200, 0.06, 0.7, 5000, 1.0, 4200, 0.3),
)


def main() -> None:
    """Generate all Phase 1 benchmark datasets."""
    output_dir = Path(__file__).resolve().parent
    public_df, public_meta = _make_dataset(
        dataset_name="public_example",
        channels=PUBLIC_CHANNELS,
        n_weeks=156,
        start_date="2022-01-03",
        seed=PUBLIC_SEED,
        stable=False,
    )
    public_df.to_csv(output_dir / "public_example.csv", index=False)
    _write_json(output_dir / "public_example_metadata.json", public_meta)

    # Generate the 132-week extended dataset, then slice the first 128 weeks
    # for the base. This way the base is *literally* the prefix of the extended
    # dataset — so a refresh that fits on the extended sees exactly the same
    # historical observations the original run did, plus four genuinely new
    # weeks. Independent regeneration would mix Monte-Carlo noise into the
    # historical window and corrupt the NFR-5 movement-classification benchmark.
    refresh_extended, refresh_meta = _make_dataset(
        dataset_name="refresh_stability",
        channels=REFRESH_CHANNELS,
        n_weeks=132,
        start_date="2023-01-02",
        seed=REFRESH_SEED,
        stable=True,
    )
    refresh_base = refresh_extended.iloc[:128].reset_index(drop=True)

    refresh_extended.to_csv(output_dir / "refresh_stability_4_weeks_added.csv", index=False)
    refresh_base.to_csv(output_dir / "refresh_stability_base.csv", index=False)

    refresh_meta["extended_n_weeks"] = refresh_meta["n_weeks"]
    refresh_meta["extended_total_revenue"] = refresh_meta["total_revenue"]
    refresh_meta["n_weeks"] = len(refresh_base)
    refresh_meta["total_revenue"] = round(float(refresh_base[TARGET_COLUMN].sum()), 6)
    refresh_meta["acceptance_target"] = {
        "default_anchor_weight": 0.3,
        "minimum_non_unexplained_fraction": 0.9,
        "purpose": "Known-stable data for NFR-5 refresh movement classification.",
    }
    _write_json(output_dir / "refresh_stability_metadata.json", refresh_meta)


def _make_dataset(
    *,
    dataset_name: str,
    channels: tuple[ChannelPlan, ...],
    n_weeks: int,
    start_date: str,
    seed: int,
    stable: bool,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_weeks, freq="W-MON")
    week_index = np.arange(n_weeks, dtype=np.float64)

    data: dict[str, NDArray[np.float64] | pd.DatetimeIndex] = {DATE_COLUMN: dates}
    weekly_contributions: dict[str, list[float]] = {}
    channel_summaries: list[dict[str, float | str]] = []
    media_total = np.zeros(n_weeks, dtype=np.float64)

    for channel in channels:
        spend = _spend_series(channel, week_index, rng, stable=stable)
        adstocked = geometric_adstock(spend, half_life_to_decay(channel.half_life))
        saturated = hill_saturation(adstocked, ec50=channel.ec50, slope=channel.slope)
        contribution = channel.coefficient * saturated

        data[channel.name] = np.round(spend, 2)
        media_total += contribution
        weekly_contributions[channel.name] = np.round(contribution, 6).tolist()
        total_spend = float(np.sum(spend))
        total_contribution = float(np.sum(contribution))
        channel_summaries.append(
            {
                **asdict(channel),
                "decay": float(half_life_to_decay(channel.half_life)),
                "total_spend": round(total_spend, 6),
                "total_contribution": round(total_contribution, 6),
                "roi": round(total_contribution / total_spend, 6),
            }
        )

    controls = _controls(week_index, rng, stable=stable)
    data.update({name: np.round(values, 6) for name, values in controls.items()})
    baseline = _baseline(week_index, stable=stable)
    control_effect = (
        -8500.0 * (controls["price_index"] - 1.0)
        + 12000.0 * controls["promo_flag"]
        + 4200.0 * controls["holiday_flag"]
        + 5200.0 * controls["macro_index"]
    )
    noise_sd = 900.0 if stable else 2200.0
    revenue = baseline + control_effect + media_total + rng.normal(0.0, noise_sd, n_weeks)
    data[TARGET_COLUMN] = np.round(revenue, 2)

    spend_columns = [channel.name for channel in channels]
    control_columns = list(controls)
    df = pd.DataFrame(data)
    df = df[[DATE_COLUMN, TARGET_COLUMN, *spend_columns, *control_columns]]

    media_sum = float(np.sum(media_total))
    for summary in channel_summaries:
        summary["share_of_media_contribution"] = round(
            float(summary["total_contribution"]) / media_sum,
            6,
        )

    metadata = {
        "dataset": dataset_name,
        "version": 1,
        "source": "Generated by benchmark_data/generate_phase1_benchmarks.py",
        "seed": seed,
        "n_weeks": n_weeks,
        "date_column": DATE_COLUMN,
        "target_column": TARGET_COLUMN,
        "spend_columns": spend_columns,
        "control_columns": control_columns,
        "noise_sd": noise_sd,
        "stable": stable,
        "baseline_total": round(float(np.sum(baseline)), 6),
        "control_effect_total": round(float(np.sum(control_effect)), 6),
        "total_media_contribution": round(media_sum, 6),
        "total_revenue": round(float(np.sum(revenue)), 6),
        "top_3_channels": [
            item["name"]
            for item in sorted(
                channel_summaries,
                key=lambda item: float(item["total_contribution"]),
                reverse=True,
            )[:3]
        ],
        "channels": channel_summaries,
        "weekly_contributions": weekly_contributions,
    }
    return df, metadata


def _spend_series(
    channel: ChannelPlan,
    week_index: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    stable: bool,
) -> NDArray[np.float64]:
    seasonal_strength = 0.10 if stable else 0.18
    pulse_count = 3 if stable else 9
    pulse_low, pulse_high = (0.08, 0.24) if stable else (0.18, 0.65)
    seasonal = 1.0 + seasonal_strength * np.sin(
        2.0 * np.pi * week_index / 52.0 + channel.phase,
    )
    pulses = np.zeros_like(week_index)
    for start in rng.choice(np.arange(len(week_index)), size=pulse_count, replace=False):
        width = int(rng.integers(2, 5))
        end = min(len(week_index), int(start) + width)
        pulses[int(start):end] += rng.uniform(pulse_low, pulse_high)
    trend = 1.0 + (0.0015 if not stable else 0.0004) * week_index
    noise = rng.lognormal(mean=0.0, sigma=channel.spend_noise, size=len(week_index))
    return np.maximum(channel.base_spend * seasonal * trend * (1.0 + pulses) * noise, 0.0)


def _controls(
    week_index: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    stable: bool,
) -> dict[str, NDArray[np.float64]]:
    noise_sd = 0.006 if stable else 0.012
    price_index = 1.0 + 0.018 * np.sin(2.0 * np.pi * week_index / 26.0) + rng.normal(
        0.0,
        noise_sd,
        size=len(week_index),
    )
    promo_flag = np.zeros(len(week_index), dtype=np.float64)
    promo_count = max(4, len(week_index) // (18 if stable else 12))
    promo_flag[rng.choice(np.arange(len(week_index)), size=promo_count, replace=False)] = 1.0
    holiday_flag = np.isin((week_index % 52).astype(int), [46, 47, 48, 49, 50, 51]).astype(
        np.float64
    )
    macro_noise = 0.025 if stable else 0.05
    macro_index = np.linspace(-0.25, 0.25, len(week_index)) + rng.normal(
        0.0,
        macro_noise,
        size=len(week_index),
    )
    return {
        "price_index": price_index,
        "promo_flag": promo_flag,
        "holiday_flag": holiday_flag,
        "macro_index": macro_index,
    }


def _baseline(week_index: NDArray[np.float64], *, stable: bool) -> NDArray[np.float64]:
    trend_step = 60.0 if stable else 125.0
    seasonal_amplitude = 6200.0 if stable else 7800.0
    trend = 70000.0 + trend_step * week_index
    seasonality = seasonal_amplitude * np.sin(2.0 * np.pi * week_index / 52.0)
    return trend + seasonality


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
