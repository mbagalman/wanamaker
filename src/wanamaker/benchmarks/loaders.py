"""Loader functions for the eight named benchmark datasets (NFR-7).

The datasets themselves are generated/curated in Phase 0 and committed to
``benchmark_data/`` at the repo root. These loaders return them as
DataFrames (or, for synthetic ground truth, as DataFrame plus the known
underlying contributions).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).parent.parent.parent.parent / "benchmark_data"


def load_public_example() -> pd.DataFrame:
    raise NotImplementedError("Phase 1: public benchmark dataset")


def load_synthetic_ground_truth() -> tuple[pd.DataFrame, dict]:
    """Return (data, ground_truth_contributions)."""
    csv_path = BENCHMARK_DIR / "synthetic_ground_truth.csv"
    json_path = BENCHMARK_DIR / "synthetic_ground_truth.json"

    if not csv_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            "Synthetic ground truth files not found. "
            "Run benchmark_data/generate_synthetic_ground_truth.py first."
        )

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        ground_truth = json.load(f)

    return df, ground_truth
