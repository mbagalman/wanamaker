"""Wanamaker -- open-source Bayesian marketing mix model.

Strategic and product context lives in ``docs/wanamaker_brd_prd.md``.
Architectural invariants and contributor guidance live in ``AGENTS.md``.

Public workflows are exposed through ``wanamaker.cli``. The package remains
pre-1.0 while the modeling, reporting, and release contracts stabilize.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("wanamaker")
except PackageNotFoundError:  # pragma: no cover -- only hit when run from a
    # source checkout that has not been ``pip install``-ed (e.g. tarball
    # extraction). Keep ``pyproject.toml`` as the single source of truth and
    # fall back to a sentinel so consumers can still inspect the attribute.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
