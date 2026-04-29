"""Loader functions for the eight named benchmark datasets (NFR-7).

The datasets themselves are generated/curated in Phase 0 and committed to
``benchmark_data/`` at the repo root. These loaders return them as
DataFrames (or, for synthetic ground truth, as DataFrame plus the known
underlying contributions).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path("benchmark_data")


def load_public_example() -> pd.DataFrame:
    raise NotImplementedError("Phase 1: public benchmark dataset")


def load_synthetic_ground_truth() -> tuple[pd.DataFrame, dict]:
    """Return (data, ground_truth_contributions)."""
    raise NotImplementedError("Phase 0: synthetic ground-truth benchmark")
