"""Benchmark dataset loaders (NFR-7).

The eight named benchmark datasets are v1 deliverables, used for both
acceptance testing and documentation:

    synthetic_ground_truth     — known media contributions; FR-3.1 acceptance
    public_example             — quickstart/tutorial dataset
    refresh_stability          — known-stable + 4 weeks; FR-4.3, NFR-5
    low_variation_channel      — spend-invariant channel; FR-3.2
    collinearity               — correlated channels; FR-2.2
    lift_test_calibration      — synthetic data with known lift; FR-1.3, FR-3.3
    target_leakage             — control derived from target; FR-2.2
    structural_break           — known break point; FR-2.2

Datasets live under ``benchmark_data/`` at the repo root and are loaded
lazily via these functions.
"""
