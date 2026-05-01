"""Sanity checks for the Colab quickstart notebook.

We don't *execute* the notebook in CI (PyMC fits take minutes; that belongs
in a smoke job, not unit tests). Instead we validate that the file is
well-formed JSON in nbformat 4 shape and that the load-bearing cells
(install, end-to-end command, showcase render) are present and point at
the right things. Catches accidental corruption from manual edits or
incompatible regenerations of the notebook source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "quickstart.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK_PATH.exists(), f"notebook missing at {NOTEBOOK_PATH}"
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def test_notebook_is_valid_nbformat_4(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert notebook.get("nbformat_minor", 0) >= 0
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"], "notebook has no cells"


def test_every_cell_has_known_type(notebook: dict) -> None:
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code", "raw"}
        assert "source" in cell
        if cell["cell_type"] == "code":
            assert cell.get("execution_count", None) is None, (
                "committed code cells must have execution_count=null so the "
                "notebook is not bloated with stale outputs"
            )
            assert cell.get("outputs", []) == [], (
                "committed code cells must have outputs cleared"
            )


def test_install_cell_uses_canonical_pip_target(notebook: dict) -> None:
    """The install cell must use ``%pip install`` (works in Colab and Jupyter)
    and point at the canonical git source — not a fork."""
    sources = [_cell_source(c) for c in notebook["cells"] if c["cell_type"] == "code"]
    install_cells = [s for s in sources if "pip install" in s]
    assert install_cells, "no pip install cell found"
    install = install_cells[0]
    assert install.startswith("%pip install"), (
        "use the %pip magic so the install lands in the current kernel"
    )
    assert "git+https://github.com/mbagalman/wanamaker.git" in install


def test_end_to_end_command_is_invoked(notebook: dict) -> None:
    sources = [_cell_source(c) for c in notebook["cells"]]
    assert any(
        "wanamaker run --example public_benchmark" in s for s in sources
    ), "the notebook must run the canonical end-to-end demo"


def test_renders_showcase_inline(notebook: dict) -> None:
    sources = [_cell_source(c) for c in notebook["cells"] if c["cell_type"] == "code"]
    text = "\n".join(sources)
    assert "showcase.html" in text
    assert "IPython.display" in text and "HTML" in text


def test_renders_report_markdown_inline(notebook: dict) -> None:
    sources = [_cell_source(c) for c in notebook["cells"] if c["cell_type"] == "code"]
    text = "\n".join(sources)
    assert "report.md" in text
    assert "Markdown" in text


def test_opens_with_colab_badge(notebook: dict) -> None:
    """Colab uses the first markdown cell as the title page; we also rely on
    it for the badge link."""
    first_cell = notebook["cells"][0]
    assert first_cell["cell_type"] == "markdown"
    src = _cell_source(first_cell)
    assert "Open In Colab" in src
    assert "colab.research.google.com/github/mbagalman/wanamaker" in src


def test_no_secrets_or_credentials_committed(notebook: dict) -> None:
    """Quick paranoia check — the notebook should never reference auth tokens."""
    text = json.dumps(notebook)
    for forbidden in ("api_key", "API_KEY", "Bearer ", "secret_key"):
        assert forbidden not in text, f"unexpected token-shaped string: {forbidden!r}"
