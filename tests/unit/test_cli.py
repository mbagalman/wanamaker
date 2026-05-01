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
        "showcase",
        "run",
        "forecast",
        "compare-scenarios",
        "recommend-ramp",
        "refresh",
    ):
        assert cmd in result.output


def test_version_flag_prints_version_and_exits_cleanly() -> None:
    """``wanamaker --version`` short-circuits before any subcommand."""
    from wanamaker import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"wanamaker {__version__}"


def test_version_attribute_matches_package_metadata() -> None:
    """``wanamaker.__version__`` is sourced from the installed package metadata
    so ``pyproject.toml`` stays the single source of truth."""
    from importlib.metadata import PackageNotFoundError, version

    import wanamaker

    try:
        expected = version("wanamaker")
    except PackageNotFoundError:
        # Source checkout that hasn't been pip-installed; sentinel still
        # has to be a valid PEP 440-ish string so consumers can format it.
        assert wanamaker.__version__.startswith("0.0.0")
    else:
        assert wanamaker.__version__ == expected
