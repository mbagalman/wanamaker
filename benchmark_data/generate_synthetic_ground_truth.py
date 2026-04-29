"""Generate the synthetic ground-truth benchmark dataset (NFR-7).

This script generates a dataset with known media contributions to be used
for the primary acceptance test for model recovery (FR-3.1).

It uses the canonical transform functions from `wanamaker.transforms` and
outputs the generated CSV and ground-truth JSON alongside it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wanamaker.transforms.adstock import geometric_adstock
from wanamaker.transforms.saturation import hill_saturation


def generate_synthetic_data(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Generate 150 weeks of synthetic MMM data for ~12 channels."""
    np.random.seed(seed)

    n_weeks = 150
    dates = pd.date_range(start="2020-01-01", periods=n_weeks, freq="W")

    channels = [
        "tv",
        "radio",
        "ooh",
        "print",
        "search",
        "social",
        "display",
        "video",
        "audio",
        "affiliate",
        "email",
        "direct_mail",
    ]
    n_channels = len(channels)

    # Ground truth parameters
    # ROI ranges from 0.5 to 3.0
    true_rois = np.random.uniform(0.5, 3.0, size=n_channels)
    # Adstock decay ranges from 0.1 to 0.8
    true_decays = np.random.uniform(0.1, 0.8, size=n_channels)
    # Hill saturation parameters
    # EC50 ranges from 0.1 to 0.9 of max observed spend
    true_ec50_fractions = np.random.uniform(0.1, 0.9, size=n_channels)
    # Slope ranges from 1.0 to 3.0
    true_slopes = np.random.uniform(1.0, 3.0, size=n_channels)

    data = {"date": dates}
    ground_truth = {"channels": {}}

    total_media_contribution = np.zeros(n_weeks)

    for i, channel in enumerate(channels):
        # Generate base spend
        base_spend = np.random.lognormal(mean=10, sigma=1, size=n_weeks)
        # Add some seasonality/trend
        trend = np.linspace(0.8, 1.2, n_weeks)
        seasonality = 1 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n_weeks))
        spend = base_spend * trend * seasonality

        # Ensure spend is positive
        spend = np.maximum(spend, 1e-6)

        data[f"{channel}_spend"] = spend

        # Apply adstock
        adstocked_spend = geometric_adstock(spend, true_decays[i])

        # Apply saturation
        ec50_abs = true_ec50_fractions[i] * np.max(adstocked_spend)
        saturated_spend = hill_saturation(adstocked_spend, ec50_abs, true_slopes[i])

        # Calculate contribution to match ROI
        # Sum(contribution) = Sum(spend) * roi
        # scale = (Sum(spend) * roi) / Sum(saturated_spend)
        scale = (np.sum(spend) * true_rois[i]) / np.sum(saturated_spend)
        contribution = saturated_spend * scale

        total_media_contribution += contribution

        ground_truth["channels"][f"{channel}_spend"] = {
            "roi": float(true_rois[i]),
            "decay": float(true_decays[i]),
            "ec50": float(ec50_abs),
            "slope": float(true_slopes[i]),
            "contribution": [float(x) for x in contribution],
            "total_contribution": float(np.sum(contribution)),
        }

    # Add control variables (e.g., price, competitor spend)
    price = np.random.normal(100, 5, size=n_weeks)
    comp_spend = np.random.lognormal(mean=12, sigma=0.5, size=n_weeks)

    data["price"] = price
    data["comp_spend"] = comp_spend

    # Baseline and control effects
    baseline = 10000 + 100 * np.linspace(0, 1, n_weeks)  # Trending baseline
    price_effect = -50 * (price - 100)
    comp_effect = -0.1 * comp_spend

    # Total target (e.g., sales)
    noise = np.random.normal(0, 500, size=n_weeks)
    target = baseline + price_effect + comp_effect + total_media_contribution + noise

    # Ensure target is strictly positive
    target = np.maximum(target, 1e-6)
    data["target"] = target

    df = pd.DataFrame(data)

    ground_truth["total_media_contribution"] = [float(x) for x in total_media_contribution]
    ground_truth["baseline"] = [float(x) for x in baseline]
    ground_truth["noise"] = [float(x) for x in noise]

    return df, ground_truth


if __name__ == "__main__":
    df, gt = generate_synthetic_data()
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    df.to_csv(out_dir / "synthetic_ground_truth.csv", index=False)
    with open(out_dir / "synthetic_ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)

    print("Generated synthetic ground truth benchmark dataset.")
