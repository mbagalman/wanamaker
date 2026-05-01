"""Smoke tests for the public CLI surface (FR-6.3).

We're checking that the public commands are wired up and discoverable. Command
behavior is covered by narrower unit and integration tests.
"""

from __future__ import annotations

from typer.testing import CliRunner

from wanamaker.cli import app

runner = CliRunner()


def test_help_lists_public_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "diagnose",
        "fit",
        "report",
        "run",
        "forecast",
        "compare-scenarios",
        "recommend-ramp",
        "refresh",
    ):
        assert cmd in result.output
