"""Smoke tests for the CLI surface (FR-6.3).

We're checking that the six core commands are wired up and discoverable,
not that they do anything yet — they all raise ``NotImplementedError``
in the scaffold pass.
"""

from __future__ import annotations

from typer.testing import CliRunner

from wanamaker.cli import app

runner = CliRunner()


def test_help_lists_all_six_core_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "diagnose",
        "fit",
        "report",
        "forecast",
        "compare-scenarios",
        "refresh",
    ):
        assert cmd in result.output
