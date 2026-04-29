"""Central seeding discipline.

NFR-2 (Reproducibility) and AGENTS.md Hard Rule 5: given the same input data,
configuration, and seed, results must be bit-for-bit identical.

The discipline:

1. The seed is configured once at the top level (``config.run.seed``) and
   passed down explicitly. Library functions accept a seed argument or a
   ``numpy.random.Generator`` produced from one — they never read
   ``np.random`` global state and never call ``np.random.seed`` /
   ``random.seed``.
2. Sampler operations receive the explicit seed via the engine abstraction.
3. ``derive_seed`` produces deterministic child seeds from a parent seed,
   so independent components (e.g. the cross-check xgboost preview) stay
   reproducible without colliding with the main sampler stream.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

SEED_BYTES: Final[int] = 8


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh ``numpy.random.Generator`` for the given seed.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A new ``Generator`` independent of the global numpy random state.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}")
    return np.random.default_rng(seed)


def derive_seed(parent_seed: int, label: str) -> int:
    """Deterministically derive a child seed from a parent seed and a label.

    Used to give independent components (e.g. ``"xgb_crosscheck"``,
    ``"posterior_predictive"``) their own reproducible random streams
    without manual bookkeeping.

    Args:
        parent_seed: The top-level seed configured by the user.
        label: A short stable identifier for the child stream.

    Returns:
        A non-negative integer seed derived from the parent and label.
    """
    digest = hashlib.blake2b(
        f"{parent_seed}:{label}".encode(),
        digest_size=SEED_BYTES,
    ).digest()
    return int.from_bytes(digest, "big")
