"""Generate the synthetic ground-truth benchmark dataset.

This script is engine-independent. It uses the canonical NumPy transform
functions from ``wanamaker.transforms`` to create a weekly marketing dataset
with known media parameters and known channel contributions.

Outputs:
    benchmark_data/synthetic_ground_truth.csv
    benchmark_data/synthetic_ground_truth_ground_truth.json

Run from the repository root:

    python benchmark_data/generate_synthetic_ground_truth.py
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

SEED = 20260429
N_WEEKS = 150
START_DATE = "2023-01-02"
DATE_COLUMN = "week"
TARGET_COLUMN = "revenue"


@dataclass(frozen=True)
class ChannelTruth:
    """Ground-truth parameters for one synthetic media channel."""

    name: str
    category: str
    base_spend: float
    spend_noise: float
    half_life: float
    ec50: float
    slope: float
    coefficient: float


CHANNELS: tuple[ChannelTruth, ...] = (
    ChannelTruth("paid_search_brand", "paid_search", 12000, 0.18, 0.6, 11000, 1.1, 9000),
    ChannelTruth("paid_search_nonbrand", "paid_search", 16000, 0.22, 1.0, 15000, 1.2, 11500),
    ChannelTruth("paid_social_performance", "paid_social", 14000, 0.30, 1.5, 13500, 1.4, 9500),
    ChannelTruth("paid_social_brand", "paid_social", 9000, 0.35, 2.5, 9000, 1.7, 8200),
    ChannelTruth("online_video", "video", 11000, 0.40, 3.0, 11500, 1.6, 10500),
    ChannelTruth("linear_tv", "linear_tv", 22000, 0.45, 6.0, 23000, 1.8, 18000),
    ChannelTruth("ctv", "ctv", 13000, 0.42, 4.0, 14000, 1.7, 12000),
    ChannelTruth("audio_podcast", "audio_podcast", 7000, 0.38, 4.5, 7600, 1.5, 7200),
    ChannelTruth("display_programmatic", "display_programmatic", 8000, 0.32, 1.5, 8500, 1.3, 6100),
    ChannelTruth("affiliate", "affiliate", 6000, 0.25, 0.8, 5600, 1.0, 5300),
    ChannelTruth("email_crm", "email_crm", 2500, 0.28, 0.6, 2600, 1.1, 4200),
    ChannelTruth(
        "promotions_discounting",
        "promotions_discounting",
        5000,
        0.55,
        0.3,
        5200,
        0.9,
        7800,
    ),
)


def main() -> None:
    """Generate the benchmark CSV and ground-truth JSON."""
    output_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
    week_index = np.arange(N_WEEKS, dtype=np.float64)

    data: dict[str, NDArray[np.float64] | pd.DatetimeIndex] = {DATE_COLUMN: dates}
    weekly_contributions: dict[str, list[float]] = {}
    channel_summaries: list[dict[str, float | str]] = []
    media_total = np.zeros(N_WEEKS, dtype=np.float64)

    for channel in CHANNELS:
        spend = _make_spend_series(channel, week_index, rng)
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

    controls = _make_controls(week_index, rng)
    data.update({name: np.round(values, 6) for name, values in controls.items()})
    baseline = _baseline(week_index)
    control_effect = (
        -9000.0 * (controls["price_index"] - 1.0)
        + 14000.0 * controls["promo_flag"]
        + 3500.0 * controls["holiday_flag"]
        + 6000.0 * controls["macro_index"]
    )
    noise_sd = 1800.0
    noise = rng.normal(0.0, noise_sd, size=N_WEEKS)
    revenue = baseline + control_effect + media_total + noise
    data[TARGET_COLUMN] = np.round(revenue, 2)

    df = pd.DataFrame(data)
    # Put revenue next to the date, then spends and controls.
    spend_columns = [channel.name for channel in CHANNELS]
    control_columns = list(controls)
    df = df[[DATE_COLUMN, TARGET_COLUMN, *spend_columns, *control_columns]]
    df.to_csv(output_dir / "synthetic_ground_truth.csv", index=False)

    media_sum = float(np.sum(media_total))
    for summary in channel_summaries:
        summary["share_of_media_contribution"] = round(
            float(summary["total_contribution"]) / media_sum,
            6,
        )

    ground_truth = {
        "dataset": "synthetic_ground_truth",
        "version": 1,
        "seed": SEED,
        "n_weeks": N_WEEKS,
        "date_column": DATE_COLUMN,
        "target_column": TARGET_COLUMN,
        "spend_columns": spend_columns,
        "control_columns": control_columns,
        "noise_sd": noise_sd,
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
    (output_dir / "synthetic_ground_truth_ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_spend_series(
    channel: ChannelTruth,
    week_index: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    seasonal = 1.0 + 0.18 * np.sin(2.0 * np.pi * week_index / 52.0 + channel.half_life)
    pulse_weeks = rng.choice(np.arange(N_WEEKS), size=10, replace=False)
    pulses = np.zeros(N_WEEKS, dtype=np.float64)
    for pulse_week in pulse_weeks:
        width = rng.integers(2, 6)
        strength = rng.uniform(0.15, 0.65)
        end = min(N_WEEKS, pulse_week + width)
        pulses[pulse_week:end] += strength
    noise = rng.lognormal(mean=0.0, sigma=channel.spend_noise, size=N_WEEKS)
    spend = channel.base_spend * seasonal * (1.0 + pulses) * noise
    return np.maximum(spend, 0.0)


def _make_controls(
    week_index: NDArray[np.float64],
    rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    price_index = 1.0 + 0.025 * np.sin(2.0 * np.pi * week_index / 26.0) + rng.normal(
        0.0,
        0.01,
        size=N_WEEKS,
    )
    promo_flag = np.zeros(N_WEEKS, dtype=np.float64)
    promo_flag[rng.choice(np.arange(N_WEEKS), size=12, replace=False)] = 1.0
    holiday_flag = np.isin((week_index % 52).astype(int), [46, 47, 48, 49, 50, 51]).astype(
        np.float64
    )
    macro_index = np.linspace(-0.5, 0.5, N_WEEKS) + rng.normal(0.0, 0.05, size=N_WEEKS)
    return {
        "price_index": price_index,
        "promo_flag": promo_flag,
        "holiday_flag": holiday_flag,
        "macro_index": macro_index,
    }


def _baseline(week_index: NDArray[np.float64]) -> NDArray[np.float64]:
    trend = 62000.0 + 140.0 * week_index
    seasonality = 8500.0 * np.sin(2.0 * np.pi * week_index / 52.0)
    return trend + seasonality


if __name__ == "__main__":
    main()
