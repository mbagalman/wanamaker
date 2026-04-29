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

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmark_data"


def load_public_example() -> pd.DataFrame:
    raise NotImplementedError("Phase 1: public benchmark dataset")


def load_synthetic_ground_truth() -> tuple[pd.DataFrame, dict]:
    """Return (data, ground_truth_contributions)."""
    data_path = BENCHMARK_DIR / "synthetic_ground_truth.csv"
    truth_path = BENCHMARK_DIR / "synthetic_ground_truth_ground_truth.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"synthetic ground-truth CSV not found at {data_path}. "
            "Run benchmark_data/generate_synthetic_ground_truth.py."
        )
    if not truth_path.exists():
        raise FileNotFoundError(
            f"synthetic ground-truth metadata not found at {truth_path}. "
            "Run benchmark_data/generate_synthetic_ground_truth.py."
        )
    data = pd.read_csv(data_path, parse_dates=["week"])
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    return data, truth
