"""``wanamaker`` CLI entry point.

Six core commands per FR-6.3:

    wanamaker diagnose <data.csv>
    wanamaker fit --config <config.yaml>
    wanamaker report --run-id <run_id>
    wanamaker forecast --run-id <run_id> --plan <plan.csv>
    wanamaker compare-scenarios --run-id <run_id> --plans <p1.csv> <p2.csv> ...
    wanamaker refresh --config <config.yaml>

Per AGENTS.md Hard Rule 1, none of these commands may make outbound
network calls. The CI gate ``tests/test_no_network_in_core.py`` enforces
this architecturally.

The CLI itself is intentionally thin — each command resolves config, then
delegates to the relevant subpackage.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="wanamaker",
    help="Open-source Bayesian marketing mix model. Knowing which half.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def diagnose(
    data: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        help=(
            "Optional YAML config. When provided, spend and control columns are "
            "read from the config and all checks run. Without it, only checks that "
            "do not require column knowledge (history length, date regularity, "
            "structural breaks) are run, and a warning is emitted."
        ),
    ),
) -> None:
    """Run the pre-flight data readiness diagnostic (FR-2.1)."""
    raise NotImplementedError("Phase 1: wire up diagnose command")


@app.command()
def fit(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help=(
            "Expert flag: bypass the data readiness diagnostic. "
            "Use only in automated pipelines or test harnesses. "
            "When set, the skip is recorded in the run manifest and "
            "flagged on the Trust Card. Not recommended for production analyses."
        ),
        hidden=True,
    ),
) -> None:
    """Fit the Bayesian model and persist a versioned run artifact (FR-3, FR-4.1)."""
    raise NotImplementedError("Phase 0/1: wire up fit command")


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Render the executive summary and trust card for a completed run (FR-5)."""
    raise NotImplementedError("Phase 2: wire up report command")


@app.command()
def forecast(
    run_id: str = typer.Option(..., "--run-id"),
    plan: Path = typer.Option(..., "--plan", exists=True),
) -> None:
    """Forecast the target metric under a single budget plan (FR-5.1 mode 2)."""
    raise NotImplementedError("Phase 1: wire up forecast command")


@app.command(name="compare-scenarios")
def compare_scenarios(
    run_id: str = typer.Option(..., "--run-id"),
    plans: list[Path] = typer.Option(..., "--plans", exists=True),
) -> None:
    """Rank multiple budget plans with uncertainty (FR-5.2)."""
    raise NotImplementedError("Phase 1: wire up compare-scenarios command")


@app.command()
def refresh(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    anchor_strength: str = typer.Option(
        "medium",
        "--anchor-strength",
        help="One of: none, light, medium, heavy, or a float in [0, 1]. See FR-4.4.",
    ),
) -> None:
    """Re-run the model with light posterior anchoring and emit a diff (FR-4)."""
    raise NotImplementedError("Phase 1: wire up refresh command")


if __name__ == "__main__":  # pragma: no cover
    app()
