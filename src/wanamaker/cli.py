"""``wanamaker`` CLI entry point.

Core commands per FR-6.3 plus the HTML showcase:

    wanamaker diagnose <data.csv>
    wanamaker fit --config <config.yaml>
    wanamaker report --run-id <run_id>
    wanamaker showcase --run-id <run_id>
    wanamaker trust-card --run-id <run_id>
    wanamaker forecast --run-id <run_id> --plan <plan.csv>
    wanamaker compare-scenarios --run-id <run_id> --plans <p1.csv> <p2.csv> ...
    wanamaker recommend-ramp --run-id <run_id> --baseline <base.csv> --target <alt.csv>
    wanamaker refresh --config <config.yaml>

Per AGENTS.md Hard Rule 1, none of these commands may make outbound
network calls. The CI gate ``tests/test_no_network_in_core.py`` enforces
this architecturally.

The CLI itself is intentionally thin — each command resolves config, then
delegates to the relevant subpackage.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="wanamaker",
    help="Open-source Bayesian marketing mix model. Knowing which half.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print ``wanamaker <version>`` and exit when ``--version`` is set.

    Eager so it short-circuits before the rest of the CLI runs.
    """
    if value:
        from wanamaker import __version__

        typer.echo(f"wanamaker {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed wanamaker version and exit.",
    ),
) -> None:
    """Top-level options shared by every ``wanamaker`` subcommand."""


@app.command()
def diagnose(
    data: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        help=(
            "Optional YAML config. When provided, column names and spend/control "
            "columns are read from the config and all checks run."
        ),
    ),
    date_column: str | None = typer.Option(
        None,
        "--date-column",
        "-d",
        help=(
            "Name of the date column. Auto-detected when omitted and --config "
            "is not provided."
        ),
    ),
    target_column: str | None = typer.Option(
        None,
        "--target-column",
        "-t",
        help=(
            "Name of the numeric target column (e.g. revenue). "
            "Auto-detected when omitted and --config is not provided."
        ),
    ),
) -> None:
    """Run the pre-flight data readiness diagnostic (FR-2.1)."""
    import pandas as pd

    from wanamaker.diagnose.readiness import CheckSeverity, ReadinessLevel

    # ------------------------------------------------------------------
    # 1. Load CSV (needed before column inference)
    # ------------------------------------------------------------------
    typer.echo(f"Loading data: {data}")
    df = pd.read_csv(data)
    typer.echo(f"  {len(df)} rows × {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Resolve column names
    # ------------------------------------------------------------------
    resolved_spend_cols: list[str] = []
    resolved_control_cols: list[str] = []

    if config is not None:
        from wanamaker.config import load_config
        cfg = load_config(config)
        resolved_date_col = cfg.data.date_column
        resolved_target_col = cfg.data.target_column
        resolved_spend_cols = list(cfg.data.spend_columns or [])
        resolved_control_cols = list(cfg.data.control_columns or [])
    else:
        resolved_date_col = date_column
        resolved_target_col = target_column

        if resolved_date_col is None or resolved_target_col is None:
            inferred_date, inferred_target = _infer_columns(df)
            if resolved_date_col is None:
                resolved_date_col = inferred_date
            if resolved_target_col is None:
                resolved_target_col = inferred_target

        if resolved_date_col is not None or resolved_target_col is not None:
            # Show what was inferred so users can verify or override
            if date_column is None and resolved_date_col is not None:
                typer.echo(
                    typer.style(
                        f"  [INFO] Inferred date column: '{resolved_date_col}'. "
                        "Use --date-column to override.",
                        fg=typer.colors.CYAN,
                    )
                )
            if target_column is None and resolved_target_col is not None:
                typer.echo(
                    typer.style(
                        f"  [INFO] Inferred target column: '{resolved_target_col}'. "
                        "Use --target-column to override.",
                        fg=typer.colors.CYAN,
                    )
                )

        if resolved_date_col is None and resolved_target_col is None:
            typer.echo(
                typer.style(
                    f"Error: Could not infer date or target columns from "
                    f"{list(df.columns)}. "
                    "Use --date-column and --target-column to specify them.",
                    fg=typer.colors.RED,
                ),
                err=True,
            )
            raise typer.Exit(code=2)

    # ------------------------------------------------------------------
    # 3. Run checks, tolerating NotImplementedError for unfinished ones
    # ------------------------------------------------------------------
    results = _run_readiness_checks(
        df,
        target_column=resolved_target_col,
        date_column=resolved_date_col,
        spend_columns=resolved_spend_cols,
        control_columns=resolved_control_cols,
    )

    # ------------------------------------------------------------------
    # 4. Determine readiness level
    # ------------------------------------------------------------------
    level = _readiness_level_from_results(results)

    # ------------------------------------------------------------------
    # 5. Print report
    # ------------------------------------------------------------------
    for r in results:
        if r.severity == CheckSeverity.BLOCKER:
            colour = typer.colors.RED
            label = "BLOCKER"
        elif r.severity == CheckSeverity.WARNING:
            colour = typer.colors.YELLOW
            label = "WARNING"
        else:
            colour = typer.colors.GREEN
            label = "INFO"
        typer.echo(typer.style(f"  [{label}] {r.message}", fg=colour))

    if not results:
        typer.echo(typer.style("  No issues found.", fg=typer.colors.GREEN))

    level_colour = typer.colors.GREEN if level == ReadinessLevel.READY else (
        typer.colors.YELLOW if level == ReadinessLevel.USABLE_WITH_WARNINGS else typer.colors.RED
    )
    typer.echo(typer.style(f"\nReadiness: {level.value}", fg=level_colour, bold=True))

    # ------------------------------------------------------------------
    # 6. Exit code: non-zero when not recommended or diagnostic-only
    # ------------------------------------------------------------------
    if level in (ReadinessLevel.NOT_RECOMMENDED, ReadinessLevel.DIAGNOSTIC_ONLY):
        raise typer.Exit(code=1)


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
    from wanamaker import __version__
    from wanamaker.artifacts import (
        hash_config,
        hash_file,
        make_run_fingerprint,
        run_paths,
        serialize_summary,
        write_manifest,
    )
    from wanamaker.config import load_config
    from wanamaker.data.io import load_input_csv
    from wanamaker.engine.pymc import PyMCEngine
    from wanamaker.model.builder import build_model_spec

    # ------------------------------------------------------------------
    # 1. Load and validate config
    # ------------------------------------------------------------------
    typer.echo(f"Loading config: {config}")
    cfg = load_config(config)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    typer.echo(f"Loading data: {cfg.data.csv_path}")
    data = load_input_csv(cfg.data)
    typer.echo(f"  {len(data)} rows × {len(data.columns)} columns")

    # ------------------------------------------------------------------
    # 3. Data readiness diagnostic (or record skip)
    # ------------------------------------------------------------------
    readiness_level: str | None
    if skip_validation:
        typer.echo(
            typer.style(
                "WARNING: --skip-validation set. Diagnostic bypassed. "
                "This will be recorded in the run manifest.",
                fg=typer.colors.YELLOW,
            )
        )
        readiness_level = "skipped"
    else:
        readiness_level = _run_diagnostics(data, cfg)

    # ------------------------------------------------------------------
    # 4. Build ModelSpec
    # ------------------------------------------------------------------
    model_spec = build_model_spec(cfg)
    typer.echo(
        f"Model: {len(model_spec.channels)} channel(s), "
        f"{len(model_spec.control_columns)} control(s), "
        f"runtime_mode={model_spec.runtime_mode}"
    )

    # ------------------------------------------------------------------
    # 5. Compute hashes and run fingerprint
    # ------------------------------------------------------------------
    data_hash = hash_file(cfg.data.csv_path)
    config_hash = _analytical_config_hash(cfg, hash_config)
    engine = PyMCEngine()
    engine_version = _get_pymc_version()
    fingerprint = make_run_fingerprint(
        data_hash=data_hash,
        config_hash=config_hash,
        package_version=__version__,
        engine_name=engine.name,
        engine_version=engine_version,
        seed=cfg.run.seed,
    )

    # ------------------------------------------------------------------
    # 6. Create run_id and artifact directory
    # ------------------------------------------------------------------
    timestamp_utc = datetime.now(UTC)
    timestamp_str = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{fingerprint[:8]}_{timestamp_str}"
    paths = run_paths(cfg.run.artifact_dir, run_id)
    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Artifacts: {paths.root}")

    # ------------------------------------------------------------------
    # 7. Fit the model
    # ------------------------------------------------------------------
    typer.echo("Fitting model…")
    result = engine.fit(
        model_spec=model_spec,
        data=data,
        seed=cfg.run.seed,
        runtime_mode=cfg.run.runtime_mode,
    )
    typer.echo("Fit complete.")

    # ------------------------------------------------------------------
    # 8. Persist artifacts
    # ------------------------------------------------------------------
    # manifest.json
    write_manifest(
        paths,
        run_id=run_id,
        run_fingerprint=fingerprint,
        timestamp=timestamp_utc.isoformat(),
        seed=cfg.run.seed,
        engine_name=engine.name,
        engine_version=engine_version,
        wanamaker_version=__version__,
        skip_validation=skip_validation,
        readiness_level=readiness_level,
    )

    # config.yaml — snapshot of the original file
    shutil.copy2(config, paths.config)

    # data_hash.txt
    paths.data_hash.write_text(data_hash)

    # timestamp.txt
    paths.timestamp.write_text(timestamp_utc.isoformat())

    # engine.txt
    paths.engine.write_text(f"{engine.name}=={engine_version}")

    # summary.json
    paths.summary.write_text(serialize_summary(result.summary))

    # posterior.nc — written by the engine via the raw ArviZ InferenceData
    _write_posterior(result.posterior, paths.posterior)

    typer.echo(
        typer.style(f"\nDone. Run ID: {run_id}", fg=typer.colors.GREEN, bold=True)
    )
    typer.echo(f"  {paths.root}")


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Path to save the report. Default: <run_dir>/report.md",
    ),
) -> None:
    """Render the executive summary and Trust Card for a completed run (FR-5)."""
    from wanamaker.artifacts import deserialize_refresh_diff, deserialize_summary, run_paths
    from wanamaker.config import load_config
    from wanamaker.reports import (
        build_executive_summary_context,
        build_trust_card_context,
        render_executive_summary,
        render_trust_card,
    )
    from wanamaker.trust_card.compute import build_trust_card

    paths = run_paths(artifact_dir, run_id)
    if not paths.config.exists():
        typer.echo(
            typer.style(
                f"Error: run {run_id!r} not found in {artifact_dir / 'runs'}. "
                "Run 'wanamaker fit' first or check --artifact-dir.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if not paths.summary.exists():
        typer.echo(
            typer.style(
                f"Error: summary.json missing for run {run_id!r} at {paths.summary}.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    summary = deserialize_summary(paths.summary.read_text())

    refresh_diff = None
    if paths.refresh_diff.exists():
        refresh_diff = deserialize_refresh_diff(paths.refresh_diff.read_text())

    cfg = load_config(paths.config)
    lift_test_priors = _load_lift_priors_if_any(cfg)
    spend_by_channel = _spend_by_channel_from_training_data(cfg)

    trust_card = build_trust_card(
        summary,
        refresh_diff=refresh_diff,
        lift_test_priors=lift_test_priors,
    )

    # Anchor in the same Markdown file rather than a separate trust_card.md;
    # report.md is meant to be self-contained per the issue acceptance.
    exec_context = build_executive_summary_context(
        summary,
        trust_card,
        refresh_diff=refresh_diff,
        advisor_recommendations=_safe_advisor_recommendations(
            summary, spend_by_channel=spend_by_channel,
        ),
        trust_card_link="#model-trust-card",
    )
    trust_context = build_trust_card_context(summary, trust_card)

    body = (
        render_executive_summary(exec_context)
        + "\n---\n\n"
        + render_trust_card(trust_context)
    )

    output_path = output if output is not None else paths.root / "report.md"
    output_path.write_text(body, encoding="utf-8")
    typer.echo(typer.style(f"Report saved: {output_path}", fg=typer.colors.GREEN))


@app.command()
def showcase(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Path to save the HTML showcase. Default: <run_dir>/showcase.html",
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        help="Override the showcase title. Default: 'Wanamaker MMM — <run_id short>'.",
    ),
    scenario: Path | None = typer.Option(
        None,
        "--scenario",
        exists=True,
        help=(
            "Optional plan CSV. When provided, the showcase includes a "
            "forecast section for that plan."
        ),
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the rendered showcase in the default browser when done.",
    ),
) -> None:
    """Render a single self-contained HTML showcase of a completed run.

    The showcase bundles the executive summary, channel contributions
    (with credible intervals), ROI table, the Trust Card, and an optional
    scenario forecast into one file with all CSS and SVG charts inlined.
    """
    import webbrowser
    from datetime import datetime

    from wanamaker import __version__
    from wanamaker.artifacts import (
        deserialize_refresh_diff,
        deserialize_summary,
        load_manifest,
        run_paths,
    )
    from wanamaker.config import load_config
    from wanamaker.reports import build_showcase_context, render_showcase
    from wanamaker.trust_card.compute import build_trust_card

    paths = run_paths(artifact_dir, run_id)
    if not paths.config.exists():
        typer.echo(
            typer.style(
                f"Error: run {run_id!r} not found in {artifact_dir / 'runs'}. "
                "Run 'wanamaker fit' first or check --artifact-dir.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if not paths.summary.exists():
        typer.echo(
            typer.style(
                f"Error: summary.json missing for run {run_id!r} at {paths.summary}.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    summary = deserialize_summary(paths.summary.read_text())

    refresh_diff = None
    if paths.refresh_diff.exists():
        refresh_diff = deserialize_refresh_diff(paths.refresh_diff.read_text())

    cfg = load_config(paths.config)
    lift_test_priors = _load_lift_priors_if_any(cfg)
    spend_by_channel = _spend_by_channel_from_training_data(cfg)

    trust_card = build_trust_card(
        summary,
        refresh_diff=refresh_diff,
        lift_test_priors=lift_test_priors,
    )

    # Manifest gives us the engine label and fingerprint. The showcase
    # falls back gracefully when manifest is absent (e.g. older runs).
    runtime_mode = cfg.run.runtime_mode
    run_fingerprint = ""
    engine_label = ""
    if paths.manifest.exists():
        try:
            manifest = load_manifest(paths.manifest.read_text())
            run_fingerprint = str(manifest.get("run_fingerprint", ""))
            engine = manifest.get("engine", {}) or {}
            engine_label = (
                f"{engine.get('name', '?')} {engine.get('version', '?')}"
            )
        except ValueError:
            engine_label = ""
    data_hash = (
        paths.data_hash.read_text().strip() if paths.data_hash.exists() else ""
    )

    scenario_forecast = None
    scenario_plan_name = None
    if scenario is not None:
        from wanamaker.forecast.posterior_predictive import forecast as run_forecast

        plan_engine, _, _, _, run_seed = _load_run_for_forecast(
            artifact_dir, run_id
        )
        typer.echo(f"Forecasting {scenario} for showcase…")
        scenario_forecast = run_forecast(summary, scenario, run_seed, plan_engine)
        scenario_plan_name = Path(scenario).stem

    advisor_lines = _safe_advisor_recommendations(
        summary, spend_by_channel=spend_by_channel,
    )

    context = build_showcase_context(
        summary,
        trust_card,
        title=title or f"Wanamaker MMM — {run_id[:8]}",
        run_id=run_id,
        generated_at=datetime.now(UTC),
        runtime_mode=runtime_mode,
        package_version=__version__,
        data_hash=data_hash,
        run_fingerprint=run_fingerprint,
        engine_label=engine_label or "(unknown)",
        refresh_diff=refresh_diff,
        scenario_forecast=scenario_forecast,
        scenario_plan_name=scenario_plan_name,
        advisor_recommendations=advisor_lines,
    )

    html = render_showcase(context)

    output_path = output if output is not None else paths.root / "showcase.html"
    output_path.write_text(html, encoding="utf-8")
    typer.echo(typer.style(f"Showcase saved: {output_path}", fg=typer.colors.GREEN))

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


@app.command(name="trust-card")
def trust_card(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Path to save the one-pager HTML. Default: <run_dir>/trust_card.html",
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        help=(
            "Override the one-pager title. "
            "Default: 'Wanamaker MMM — <run_id short>'."
        ),
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the rendered one-pager in the default browser when done.",
    ),
) -> None:
    """Render the executive-facing Trust Card one-pager for a completed run.

    The one-pager is a single self-contained HTML file designed for
    forwarding to non-technical executives. It prints to a single page
    on US Letter or A4 and uses plain-English translations of each
    Trust Card dimension instead of the analyst-facing technical text.
    """
    import webbrowser
    from datetime import datetime, timezone

    from wanamaker import __version__
    from wanamaker.artifacts import (
        deserialize_refresh_diff,
        deserialize_summary,
        load_manifest,
        run_paths,
    )
    from wanamaker.config import load_config
    from wanamaker.reports import (
        build_trust_card_one_pager_context,
        render_trust_card_one_pager,
    )
    from wanamaker.trust_card.compute import build_trust_card

    paths = run_paths(artifact_dir, run_id)
    if not paths.config.exists():
        typer.echo(
            typer.style(
                f"Error: run {run_id!r} not found in {artifact_dir / 'runs'}. "
                "Run 'wanamaker fit' first or check --artifact-dir.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if not paths.summary.exists():
        typer.echo(
            typer.style(
                f"Error: summary.json missing for run {run_id!r} at {paths.summary}.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    summary = deserialize_summary(paths.summary.read_text())

    refresh_diff = None
    if paths.refresh_diff.exists():
        refresh_diff = deserialize_refresh_diff(paths.refresh_diff.read_text())

    cfg = load_config(paths.config)
    lift_test_priors = _load_lift_priors_if_any(cfg)

    card = build_trust_card(
        summary,
        refresh_diff=refresh_diff,
        lift_test_priors=lift_test_priors,
    )

    run_fingerprint = ""
    if paths.manifest.exists():
        try:
            manifest = load_manifest(paths.manifest.read_text())
            run_fingerprint = str(manifest.get("run_fingerprint", ""))
        except ValueError:
            run_fingerprint = ""

    context = build_trust_card_one_pager_context(
        summary,
        card,
        title=title or f"Wanamaker MMM — {run_id[:8]}",
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        package_version=__version__,
        run_fingerprint=run_fingerprint,
    )

    html = render_trust_card_one_pager(context)

    output_path = output if output is not None else paths.root / "trust_card.html"
    output_path.write_text(html, encoding="utf-8")
    typer.echo(
        typer.style(f"Trust Card one-pager saved: {output_path}", fg=typer.colors.GREEN)
    )

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


_RUN_EXAMPLES: tuple[str, ...] = ("public_benchmark",)


@app.command()
def run(
    example: str = typer.Option(
        ...,
        "--example",
        help=(
            "Bundled example to run end-to-end. "
            f"Choices: {', '.join(_RUN_EXAMPLES)}."
        ),
    ),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
) -> None:
    """One-command end-to-end demo on a bundled example dataset (FR-7.1).

    Chains ``diagnose`` → ``fit`` → ``report`` against an example shipped
    inside the package itself, so a fresh ``pip install`` is enough to
    produce an executive summary without any user-supplied data. Quick
    runtime tier so the whole flow finishes in a few minutes on a
    laptop.
    """
    import importlib.resources

    import yaml as _yaml

    from wanamaker.config import load_config

    if example not in _RUN_EXAMPLES:
        typer.echo(
            typer.style(
                f"Unknown example {example!r}. Available: "
                f"{', '.join(_RUN_EXAMPLES)}.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=2)

    # Materialise the bundled config to a stable path inside the
    # artifact dir, with csv_path and artifact_dir resolved against the
    # current run. The fit command will copy this file into the run
    # directory as part of its standard manifest.
    example_root = importlib.resources.files(
        f"wanamaker._examples.{example}"
    )
    bundled_yaml = example_root.joinpath("public_example.yaml").read_text(
        encoding="utf-8"
    )
    cfg_dict = _yaml.safe_load(bundled_yaml)
    cfg_dict["data"]["csv_path"] = str(
        example_root.joinpath(cfg_dict["data"]["csv_path"])
    )
    cfg_dict["run"]["artifact_dir"] = str(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    materialised_config = artifact_dir / f"_example_{example}_config.yaml"
    materialised_config.write_text(_yaml.safe_dump(cfg_dict), encoding="utf-8")

    typer.echo(
        typer.style(
            f"\n=== wanamaker run --example {example} ===\n",
            fg=typer.colors.CYAN, bold=True,
        )
    )

    typer.echo(
        typer.style("Step 1/3 · readiness diagnostic", fg=typer.colors.CYAN, bold=True)
    )
    cfg = load_config(materialised_config)
    from wanamaker.data.io import load_input_csv
    data_for_diagnostic = load_input_csv(cfg.data)
    _run_diagnostics(data_for_diagnostic, cfg)
    typer.echo("")

    typer.echo(
        typer.style("Step 2/3 · fitting model (quick mode)", fg=typer.colors.CYAN, bold=True)
    )
    runs_before = _existing_run_ids(artifact_dir)
    fit(config=materialised_config, skip_validation=False)
    runs_after = _existing_run_ids(artifact_dir)
    new_run_ids = sorted(runs_after - runs_before)
    if not new_run_ids:
        typer.echo(
            typer.style(
                "Error: fit did not produce a new run directory.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    run_id = new_run_ids[-1]
    typer.echo("")

    typer.echo(
        typer.style("Step 3/3 · executive summary + Trust Card", fg=typer.colors.CYAN, bold=True)
    )
    report(run_id=run_id, artifact_dir=artifact_dir, output=None)
    typer.echo("")

    # Bonus: produce the HTML showcase too. The example flow is the
    # canonical "show your CMO" demo moment; rendering it here means a
    # single ``wanamaker run --example`` produces both the analyst-facing
    # Markdown and the stakeholder-facing HTML in one go.
    showcase(
        run_id=run_id,
        artifact_dir=artifact_dir,
        output=None,
        title=None,
        scenario=None,
        open_browser=False,
    )
    trust_card(
        run_id=run_id,
        artifact_dir=artifact_dir,
        output=None,
        title=None,
        open_browser=False,
    )
    typer.echo("")

    report_path = artifact_dir / "runs" / run_id / "report.md"
    typer.echo(
        typer.style(f"--- {report_path} ---", fg=typer.colors.CYAN)
    )
    typer.echo(report_path.read_text(encoding="utf-8"))


def _existing_run_ids(artifact_dir: Path) -> set[str]:
    """Return the set of run IDs currently under ``artifact_dir/runs/``."""
    runs_root = artifact_dir / "runs"
    if not runs_root.exists():
        return set()
    return {p.name for p in runs_root.iterdir() if p.is_dir()}


@app.command()
def forecast(
    run_id: str = typer.Option(..., "--run-id"),
    plan: Path = typer.Option(..., "--plan", exists=True),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Override the run's stored seed for posterior-predictive sampling.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Path to save the Markdown report. Default: <run_dir>/forecast_<plan>.md",
    ),
) -> None:
    """Forecast the target metric under a single budget plan (FR-5.1 mode 2)."""
    from wanamaker.forecast.posterior_predictive import forecast as run_forecast

    plan_engine, posterior, summary, paths, run_seed = _load_run_for_forecast(
        artifact_dir, run_id
    )
    seed_used = seed if seed is not None else run_seed

    typer.echo(f"Forecasting {plan} with run {run_id} (seed={seed_used})…")
    result = run_forecast(summary, plan, seed_used, plan_engine)

    _print_forecast_table(result)

    output_path = output if output is not None else (
        paths.root / f"forecast_{Path(plan).stem}.md"
    )
    output_path.write_text(_format_forecast_markdown(result, run_id, plan))
    typer.echo(typer.style(f"\nReport saved: {output_path}", fg=typer.colors.GREEN))


@app.command(name="compare-scenarios")
def compare_scenarios(
    run_id: str = typer.Option(..., "--run-id"),
    plans: list[Path] = typer.Option(..., "--plans", exists=True),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Override the run's stored seed for posterior-predictive sampling.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help=(
            "Path to save the Markdown report. "
            "Default: <run_dir>/scenario_comparison.md"
        ),
    ),
) -> None:
    """Rank multiple budget plans with uncertainty (FR-5.2)."""
    from wanamaker.forecast.scenarios import compare_scenarios as run_compare

    plan_engine, _, summary, paths, run_seed = _load_run_for_forecast(
        artifact_dir, run_id
    )
    seed_used = seed if seed is not None else run_seed

    typer.echo(f"Comparing {len(plans)} plan(s) for run {run_id} (seed={seed_used})…")
    plan_paths: list[Any] = list(plans)
    results = run_compare(summary, plan_paths, seed_used, plan_engine)

    _print_scenario_comparison_table(results)

    output_path = output if output is not None else paths.root / "scenario_comparison.md"
    output_path.write_text(_format_scenario_comparison_markdown(results, run_id))
    typer.echo(typer.style(f"\nReport saved: {output_path}", fg=typer.colors.GREEN))


@app.command(name="recommend-ramp")
def recommend_ramp(
    run_id: str = typer.Option(..., "--run-id"),
    baseline: Path = typer.Option(..., "--baseline", exists=True),
    target: Path = typer.Option(..., "--target", exists=True),
    artifact_dir: Path = typer.Option(
        Path(".wanamaker"),
        "--artifact-dir",
        help="Root of the runs directory (default: .wanamaker).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Override the run's stored seed for posterior-predictive sampling.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help=(
            "Path to save the Markdown report. "
            "Default: <run_dir>/ramp_<baseline>_to_<target>.md"
        ),
    ),
) -> None:
    """Recommend a risk-adjusted ramp from a baseline plan to a target plan."""
    from wanamaker.artifacts import deserialize_refresh_diff
    from wanamaker.config import load_config
    from wanamaker.forecast.ramp import recommend_ramp as run_recommend_ramp
    from wanamaker.reports import (
        build_ramp_recommendation_context,
        render_ramp_recommendation,
    )
    from wanamaker.trust_card.compute import build_trust_card

    plan_engine, _, summary, paths, run_seed = _load_run_for_forecast(
        artifact_dir, run_id
    )
    seed_used = seed if seed is not None else run_seed

    refresh_diff = None
    if paths.refresh_diff.exists():
        refresh_diff = deserialize_refresh_diff(paths.refresh_diff.read_text())

    cfg = load_config(paths.config)
    trust_card = build_trust_card(
        summary,
        refresh_diff=refresh_diff,
        lift_test_priors=_load_lift_priors_if_any(cfg),
    )
    spend_by_channel = _spend_by_channel_from_training_data(cfg)

    typer.echo(
        f"Recommending ramp from {baseline} to {target} "
        f"with run {run_id} (seed={seed_used})..."
    )
    recommendation = run_recommend_ramp(
        summary,
        baseline,
        target,
        seed_used,
        plan_engine,
        trust_card=trust_card,
    )

    _print_ramp_recommendation(recommendation)
    advisor_handoff = _ramp_advisor_handoff(
        recommendation, summary, spend_by_channel=spend_by_channel,
    )

    output_path = output if output is not None else (
        paths.root / f"ramp_{baseline.stem}_to_{target.stem}.md"
    )
    context = build_ramp_recommendation_context(
        recommendation,
        run_id=run_id,
        baseline_path=baseline,
        target_path=target,
        advisor_handoff=advisor_handoff,
    )
    output_path.write_text(render_ramp_recommendation(context), encoding="utf-8")
    typer.echo(typer.style(f"\nReport saved: {output_path}", fg=typer.colors.GREEN))


@app.command()
def refresh(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    anchor_strength: str | None = typer.Option(
        None,
        "--anchor-strength",
        help=(
            "Override the config's anchor strength. One of: none, light, "
            "medium, heavy, or a float in [0, 1]. See FR-4.4."
        ),
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help=(
            "Expert flag: bypass the data readiness diagnostic. "
            "Recorded in the run manifest and flagged on the Trust Card."
        ),
        hidden=True,
    ),
) -> None:
    """Re-run the model with light posterior anchoring and emit a diff (FR-4)."""
    import dataclasses

    from wanamaker import __version__
    from wanamaker.artifacts import (
        hash_config,
        hash_file,
        make_run_fingerprint,
        run_paths,
        serialize_refresh_diff,
        serialize_summary,
        write_manifest,
    )
    from wanamaker.config import load_config
    from wanamaker.data.io import load_input_csv
    from wanamaker.engine.pymc import PyMCEngine
    from wanamaker.model.builder import build_model_spec
    from wanamaker.refresh.diff import compute_diff

    # ------------------------------------------------------------------
    # 1. Load config and resolve anchor strength (CLI overrides config)
    # ------------------------------------------------------------------
    typer.echo(f"Loading config: {config}")
    cfg = load_config(config)

    raw_anchor = anchor_strength if anchor_strength is not None else cfg.refresh.anchor_strength
    weight = _resolve_anchor_strength(raw_anchor)
    typer.echo(f"Anchor strength: {raw_anchor!r} (weight={weight:.3f})")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    typer.echo(f"Loading data: {cfg.data.csv_path}")
    data = load_input_csv(cfg.data)
    typer.echo(f"  {len(data)} rows × {len(data.columns)} columns")

    # ------------------------------------------------------------------
    # 3. Find the most recent prior run for this data file
    # ------------------------------------------------------------------
    prev_run_id, prev_summary = _find_previous_run(cfg.run.artifact_dir, cfg.data.csv_path)
    if prev_run_id is None:
        typer.echo(
            typer.style(
                f"Error: no previous run found in {cfg.run.artifact_dir / 'runs'} for "
                f"data file {cfg.data.csv_path}. Run 'wanamaker fit --config {config}' first.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"Previous run: {prev_run_id}")

    # ------------------------------------------------------------------
    # 4. Diagnostic (or record skip)
    # ------------------------------------------------------------------
    readiness_level: str | None
    if skip_validation:
        typer.echo(
            typer.style(
                "WARNING: --skip-validation set. Diagnostic bypassed. "
                "This will be recorded in the run manifest.",
                fg=typer.colors.YELLOW,
            )
        )
        readiness_level = "skipped"
    else:
        readiness_level = _run_diagnostics(data, cfg)

    # ------------------------------------------------------------------
    # 5. Build ModelSpec with anchor priors derived from the previous run
    # ------------------------------------------------------------------
    base_spec = build_model_spec(cfg)
    anchor_priors = _build_anchor_priors(prev_summary, weight)
    model_spec = dataclasses.replace(base_spec, anchor_priors=anchor_priors)
    typer.echo(
        f"Model: {len(model_spec.channels)} channel(s), "
        f"{len(model_spec.control_columns)} control(s), "
        f"runtime_mode={model_spec.runtime_mode}, "
        f"anchored_params={len(anchor_priors)}"
    )

    # ------------------------------------------------------------------
    # 6. Hashes, fingerprint, run_id
    # ------------------------------------------------------------------
    data_hash = hash_file(cfg.data.csv_path)
    config_hash = _analytical_config_hash(cfg, hash_config)
    engine = PyMCEngine()
    engine_version = _get_pymc_version()
    fingerprint = make_run_fingerprint(
        data_hash=data_hash,
        config_hash=config_hash,
        package_version=__version__,
        engine_name=engine.name,
        engine_version=engine_version,
        seed=cfg.run.seed,
    )
    timestamp_utc = datetime.now(UTC)
    timestamp_str = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{fingerprint[:8]}_{timestamp_str}"
    paths = run_paths(cfg.run.artifact_dir, run_id)
    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Artifacts: {paths.root}")

    # ------------------------------------------------------------------
    # 7. Re-fit the model with anchored priors
    # ------------------------------------------------------------------
    typer.echo("Re-fitting model with anchored priors…")
    result = engine.fit(
        model_spec=model_spec,
        data=data,
        seed=cfg.run.seed,
        runtime_mode=cfg.run.runtime_mode,
    )
    typer.echo("Fit complete.")

    # ------------------------------------------------------------------
    # 8. Persist standard run artifacts
    # ------------------------------------------------------------------
    write_manifest(
        paths,
        run_id=run_id,
        run_fingerprint=fingerprint,
        timestamp=timestamp_utc.isoformat(),
        seed=cfg.run.seed,
        engine_name=engine.name,
        engine_version=engine_version,
        wanamaker_version=__version__,
        skip_validation=skip_validation,
        readiness_level=readiness_level,
    )
    shutil.copy2(config, paths.config)
    paths.data_hash.write_text(data_hash)
    paths.timestamp.write_text(timestamp_utc.isoformat())
    paths.engine.write_text(f"{engine.name}=={engine_version}")
    paths.summary.write_text(serialize_summary(result.summary))
    _write_posterior(result.posterior, paths.posterior)

    # ------------------------------------------------------------------
    # 9. Compute and persist the refresh diff (FR-4.2 / FR-4.3)
    # ------------------------------------------------------------------
    diff = compute_diff(prev_summary, result.summary, prev_run_id, run_id)
    paths.refresh_diff.write_text(serialize_refresh_diff(diff))
    _print_diff_summary(diff)

    typer.echo(
        typer.style(f"\nDone. Run ID: {run_id}", fg=typer.colors.GREEN, bold=True)
    )
    typer.echo(f"  {paths.root}")


# ---------------------------------------------------------------------------
# Private helpers (not part of the public CLI surface)
# ---------------------------------------------------------------------------


def _infer_columns(df: object) -> tuple[str | None, str | None]:
    """Heuristically infer date and target columns from a DataFrame.

    Resolution order for **date column**:
    1. Any column whose lowercase name contains "date", "week", "day", "month",
       "period", "timestamp", or "time".
    2. The first non-numeric column whose first five non-null values all parse
       as datetimes via ``pd.to_datetime``.

    Resolution order for **target column**:
    1. Any numeric column whose lowercase name contains "revenue", "sales",
       "target", "kpi", or "metric".
    2. The first numeric column that is not the inferred date column.

    Returns a ``(date_col, target_col)`` tuple; either element may be ``None``
    when inference fails.
    """
    import pandas as pd

    assert isinstance(df, pd.DataFrame)

    _DATE_HINTS = ("date", "week", "day", "month", "period", "timestamp", "time")
    _TARGET_HINTS = ("revenue", "sales", "target", "kpi", "metric")

    # --- Date column ---
    date_col: str | None = None
    for col in df.columns:
        if any(hint in col.lower() for hint in _DATE_HINTS):
            date_col = col
            break
    if date_col is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                pd.to_datetime(df[col].dropna().head(5))
                date_col = col
                break
            except (ValueError, TypeError):
                pass

    # --- Target column ---
    # Only infer from name hints; do not fall back to "first numeric column"
    # so that columns with opaque names like "col_a" are not silently mis-labelled.
    # Users with non-standard column names should use --target-column explicitly.
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != date_col
    ]
    target_col: str | None = None
    for col in numeric_cols:
        if any(hint in col.lower() for hint in _TARGET_HINTS):
            target_col = col
            break

    return date_col, target_col


def _run_diagnostics(data: object, cfg: object) -> str:
    """Run the implemented readiness suite for ``fit`` / ``refresh``.

    Calls the same shared check-runner used by ``wanamaker diagnose`` so
    the gate that decides whether to write run artifacts sees every
    available signal — history length, date regularity, missing values,
    target stability, structural breaks, and (with config in hand)
    spend variation, collinearity, variable count, and target leakage.

    Returns the resulting ``ReadinessLevel`` as a string and prints any
    blockers / warnings to stdout. ``--skip-validation`` remains the
    explicit opt-out path (handled by the caller).
    """
    import pandas as pd

    from wanamaker.config import WanamakerConfig
    from wanamaker.diagnose.readiness import CheckSeverity

    assert isinstance(data, pd.DataFrame)
    assert isinstance(cfg, WanamakerConfig)

    results = _run_readiness_checks(
        data,
        target_column=cfg.data.target_column,
        date_column=cfg.data.date_column,
        spend_columns=list(cfg.data.spend_columns or []),
        control_columns=list(cfg.data.control_columns or []),
    )
    level = _readiness_level_from_results(results)

    for r in results:
        if r.severity in (CheckSeverity.BLOCKER, CheckSeverity.WARNING):
            colour = (
                typer.colors.RED if r.severity == CheckSeverity.BLOCKER
                else typer.colors.YELLOW
            )
            typer.echo(
                typer.style(
                    f"  [{r.severity.value.upper()}] {r.message}", fg=colour,
                )
            )

    typer.echo(f"Readiness: {level.value}")
    return level.value


def _run_readiness_checks(
    df: object,
    *,
    target_column: str | None,
    date_column: str | None,
    spend_columns: list[str] | None = None,
    control_columns: list[str] | None = None,
) -> list[Any]:
    """Run the full implemented readiness suite against ``df``.

    Spend-aware checks (variation, collinearity, variable count) only run
    when ``spend_columns`` is non-empty; target-leakage runs only when
    both ``control_columns`` and ``target_column`` are present. Checks
    that raise ``NotImplementedError`` are silently skipped (so this
    helper degrades gracefully against future check rollouts).
    ``KeyError`` / ``ValueError`` from a single check become an inline
    WARNING — the user gets a clear "this check was skipped" line rather
    than a crash that masks every other check's results.
    """
    import pandas as pd

    from wanamaker.diagnose.checks import (
        check_collinearity,
        check_date_regularity,
        check_history_length,
        check_missing_values,
        check_spend_variation,
        check_structural_breaks,
        check_target_leakage,
        check_target_stability,
        check_variable_count,
    )
    from wanamaker.diagnose.readiness import CheckResult, CheckSeverity

    assert isinstance(df, pd.DataFrame)

    results: list[CheckResult] = []

    def _try(fn: Any, *args: Any, **kwargs: Any) -> None:
        try:
            out = fn(*args, **kwargs)
            if isinstance(out, list):
                results.extend(out)
            elif out is not None:
                results.append(out)
        except NotImplementedError:
            pass
        except (KeyError, ValueError) as exc:
            results.append(
                CheckResult(
                    name=fn.__name__.removeprefix("check_"),
                    severity=CheckSeverity.WARNING,
                    message=f"Check skipped — {exc}",
                )
            )

    _try(check_history_length, df, date_column)
    _try(check_date_regularity, df, date_column)
    _try(check_missing_values, df)
    _try(check_target_stability, df, target_column)
    _try(check_structural_breaks, df, target_column, date_column)

    if spend_columns:
        _try(check_spend_variation, df, spend_columns)
        _try(check_collinearity, df, spend_columns, control_columns or [])
        _try(check_variable_count, df, spend_columns, control_columns or [])
    if control_columns and target_column:
        _try(check_target_leakage, df, target_column, control_columns)

    return results


def _readiness_level_from_results(results: list[Any]) -> Any:
    """Map a list of ``CheckResult`` to a ``ReadinessLevel``.

    BLOCKER → NOT_RECOMMENDED · WARNING → USABLE_WITH_WARNINGS · else READY.
    """
    from wanamaker.diagnose.readiness import CheckSeverity, ReadinessLevel

    blockers = [r for r in results if r.severity == CheckSeverity.BLOCKER]
    warnings = [r for r in results if r.severity == CheckSeverity.WARNING]
    if blockers:
        return ReadinessLevel.NOT_RECOMMENDED
    if warnings:
        return ReadinessLevel.USABLE_WITH_WARNINGS
    return ReadinessLevel.READY


def _analytical_config_hash(cfg: object, hash_config_fn: object) -> str:
    """Hash only the analytical fields of the config (exclude storage paths)."""
    from wanamaker.config import WanamakerConfig
    assert isinstance(cfg, WanamakerConfig)
    assert callable(hash_config_fn)

    # Include everything except artifact_dir (storage, not analytical).
    d = cfg.model_dump()
    d.get("run", {}).pop("artifact_dir", None)
    return hash_config_fn(d)  # type: ignore[operator]


def _get_pymc_version() -> str:
    """Return the installed PyMC version string, or 'unknown' if not available.

    Uses ``importlib.metadata`` so the engine version can be read without
    actually importing PyMC (which would pull in arviz / pytensor and
    violate the lazy-import contract that ``test_pymc_engine`` relies on).
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pymc")
    except PackageNotFoundError:
        return "unknown"


def _write_posterior(posterior: object, path: object) -> None:
    """Write the posterior draws to a NetCDF file via ArviZ InferenceData."""
    from pathlib import Path as _Path

    from wanamaker.engine.base import Posterior
    from wanamaker.engine.pymc import PyMCRawPosterior

    assert isinstance(posterior, Posterior)
    assert isinstance(path, _Path)

    raw = posterior.raw
    if isinstance(raw, PyMCRawPosterior):
        try:
            raw.idata.to_netcdf(str(path))
        except Exception as exc:  # pragma: no cover
            typer.echo(
                typer.style(
                    f"WARNING: could not write posterior.nc: {exc}",
                    fg=typer.colors.YELLOW,
                )
            )
    else:  # pragma: no cover
        typer.echo(
            typer.style(
                "WARNING: posterior type not recognised; posterior.nc not written.",
                fg=typer.colors.YELLOW,
            )
        )


def _resolve_anchor_strength(value: str | float) -> float:
    """Translate a raw CLI/YAML anchor-strength value to a numeric weight.

    Strings that parse as floats are treated as numeric weights;
    everything else is treated as a preset name and validated by
    ``resolve_anchor_weight``.
    """
    from wanamaker.refresh.anchor import resolve_anchor_weight

    if isinstance(value, str):
        try:
            parsed: str | float = float(value)
        except ValueError:
            parsed = value
    else:
        parsed = value
    return resolve_anchor_weight(parsed)


def _find_previous_run(
    artifact_dir: Path, csv_path: Path
) -> tuple[str | None, object | None]:
    """Find the most recent prior run whose snapshotted config used ``csv_path``.

    Run IDs encode an ISO timestamp, so the lexicographically newest entry
    in ``list_runs`` is the most recent. We iterate newest-first and return
    on the first match. Runs with missing or malformed artifacts are skipped
    silently so a corrupt run cannot block refresh.

    Args:
        artifact_dir: Directory containing the ``runs/`` subdirectory.
        csv_path: Data file path from the current config.

    Returns:
        A tuple ``(run_id, posterior_summary)`` for the matching run, or
        ``(None, None)`` if no run matches.
    """
    from wanamaker.artifacts import deserialize_summary, list_runs, run_paths
    from wanamaker.config import load_config

    target = _normalize_path(csv_path)
    for run_id in reversed(list_runs(artifact_dir)):
        paths = run_paths(artifact_dir, run_id)
        if not paths.config.exists() or not paths.summary.exists():
            continue
        try:
            prev_cfg = load_config(paths.config)
        except Exception:  # noqa: BLE001 — corrupt snapshot, skip
            continue
        if _normalize_path(prev_cfg.data.csv_path) != target:
            continue
        try:
            summary = deserialize_summary(paths.summary.read_text())
        except (ValueError, KeyError):
            continue
        return run_id, summary
    return None, None


def _normalize_path(path: Path) -> str:
    """Best-effort canonical form for path comparison.

    Uses ``Path.resolve`` (which works on non-existent paths) and falls
    back to ``str(path)`` if resolution fails on the platform.
    """
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _build_anchor_priors(prev_summary: object, weight: float) -> dict[str, object]:
    """Build per-parameter mixture priors from a previous PosteriorSummary.

    Every scalar parameter from the previous run is given a corresponding
    ``AnchoredPrior`` keyed by the parameter's stable name. The PyMC engine
    only consumes the keys it recognises (e.g. ``channel.<n>.half_life``)
    and silently ignores the rest, so we don't filter here.

    Args:
        prev_summary: ``PosteriorSummary`` from the prior run.
        weight: Mixture weight in ``[0, 1]`` for the previous-posterior
            component. ``0.0`` produces priors that fully fall back to
            defaults; ``1.0`` produces full anchoring.

    Returns:
        A dict suitable for ``ModelSpec.anchor_priors``.
    """
    from wanamaker.engine.summary import PosteriorSummary
    from wanamaker.model.spec import AnchoredPrior

    assert isinstance(prev_summary, PosteriorSummary)
    return {
        p.name: AnchoredPrior(mean=p.mean, sd=p.sd, weight=weight)
        for p in prev_summary.parameters
    }


def _print_diff_summary(diff: object) -> None:
    """Print a short user-facing summary of a refresh diff.

    Lists the count of movements per ``MovementClass`` and the unexplained
    fraction (the headline refresh-stability metric, FR-4.5).
    """
    from collections import Counter

    from wanamaker.refresh.classify import unexplained_fraction
    from wanamaker.refresh.diff import RefreshDiff

    assert isinstance(diff, RefreshDiff)
    typer.echo("\nRefresh diff:")
    typer.echo(f"  {len(diff.movements)} parameter(s) compared")

    counts = Counter(
        m.movement_class.value if m.movement_class is not None else "unclassified"
        for m in diff.movements
    )
    for cls, n in sorted(counts.items()):
        typer.echo(f"    {cls}: {n}")

    fraction = unexplained_fraction(list(diff.movements))
    colour = typer.colors.GREEN if fraction < 0.1 else (
        typer.colors.YELLOW if fraction < 0.25 else typer.colors.RED
    )
    typer.echo(
        typer.style(f"  Unexplained fraction: {fraction:.1%}", fg=colour)
    )


def _load_run_for_forecast(
    artifact_dir: Path, run_id: str
) -> tuple[Any, Any, Any, Any, int]:
    """Load everything ``forecast`` and ``compare-scenarios`` need for a run.

    Returns ``(plan_engine, posterior, summary, paths, run_seed)``:

    - ``plan_engine`` conforms to ``PosteriorPredictiveEngine`` and routes
      calls to the underlying ``PyMCEngine`` with the loaded ``Posterior``
      bound. The Protocol takes a ``PosteriorSummary`` while the engine
      needs a ``Posterior``; the bound adapter bridges them.
    - ``posterior`` is the reconstructed PyMC posterior (also returned so
      the caller can use it for unrelated work such as report rendering).
    - ``summary`` is the ``PosteriorSummary`` from ``summary.json``.
    - ``paths`` is the resolved ``RunPaths`` for the run directory.
    - ``run_seed`` comes from the snapshotted config; the CLI applies it
      when ``--seed`` is not specified.
    """
    from wanamaker.artifacts import deserialize_summary, run_paths
    from wanamaker.config import load_config
    from wanamaker.data.io import load_input_csv
    from wanamaker.engine.pymc import PyMCEngine
    from wanamaker.model.builder import build_model_spec

    paths = run_paths(artifact_dir, run_id)
    if not paths.config.exists():
        typer.echo(
            typer.style(
                f"Error: run {run_id!r} not found in {artifact_dir / 'runs'}. "
                "Run 'wanamaker fit' first or check --artifact-dir.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if not paths.posterior.exists():
        typer.echo(
            typer.style(
                f"Error: posterior.nc missing for run {run_id!r} at {paths.posterior}. "
                "The fit may have been interrupted; re-run 'wanamaker fit'.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if not paths.summary.exists():
        typer.echo(
            typer.style(
                f"Error: summary.json missing for run {run_id!r} at {paths.summary}.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    cfg = load_config(paths.config)
    typer.echo(f"Loading data: {cfg.data.csv_path}")
    data = load_input_csv(cfg.data)
    model_spec = build_model_spec(cfg)
    summary = deserialize_summary(paths.summary.read_text())

    typer.echo("Reconstructing PyMC model from run artifact…")
    engine = PyMCEngine()
    posterior = engine.load_posterior(paths.root, model_spec, data)
    plan_engine = _PosteriorBoundEngine(engine, posterior)
    return plan_engine, posterior, summary, paths, int(cfg.run.seed)


class _PosteriorBoundEngine:
    """Adapt ``PyMCEngine`` to the forecast layer's
    ``PosteriorPredictiveEngine`` protocol by binding a specific
    ``Posterior``.

    The protocol's signature takes a ``PosteriorSummary``; the engine call
    actually needs a ``Posterior``. Binding lets the CLI hand the engine
    object the forecast layer's ``forecast()`` and ``compare_scenarios()``
    expect without changing those public surfaces.
    """

    def __init__(self, engine: Any, posterior: Any) -> None:
        self._engine = engine
        self._posterior = posterior

    def posterior_predictive(
        self,
        posterior_summary: Any,  # noqa: ARG002 — Protocol contract
        new_data: Any,
        seed: int,
    ) -> Any:
        return self._engine.posterior_predictive(self._posterior, new_data, seed)


def _print_forecast_table(result: Any) -> None:
    """Print the forecast result as a human-readable table."""
    typer.echo("\nPosterior predictive forecast")
    if result.spend_invariant_channels:
        typer.echo(
            typer.style(
                f"  Spend-invariant channels (no reallocation recommendation): "
                f"{', '.join(result.spend_invariant_channels)}",
                fg=typer.colors.YELLOW,
            )
        )
    if result.extrapolation_flags:
        typer.echo(
            typer.style(
                f"  Extrapolation warnings: {len(result.extrapolation_flags)} cell(s) "
                "outside the observed historical range:",
                fg=typer.colors.YELLOW,
            )
        )
        for flag in result.extrapolation_flags:
            typer.echo(
                typer.style(
                    f"    {flag.period} · {flag.channel}: planned {flag.planned_spend:,.0f} "
                    f"({flag.direction.replace('_', ' ')}; observed range "
                    f"{flag.observed_spend_min:,.0f}–{flag.observed_spend_max:,.0f})",
                    fg=typer.colors.YELLOW,
                )
            )
    typer.echo("")
    typer.echo(f"  {'Period':<12} {'Mean':>14} {'95% HDI low':>14} {'95% HDI high':>14}")
    for period, mean, low, high in zip(
        result.periods, result.mean, result.hdi_low, result.hdi_high, strict=True,
    ):
        typer.echo(f"  {period:<12} {mean:>14,.2f} {low:>14,.2f} {high:>14,.2f}")
    total_mean = sum(result.mean)
    typer.echo(f"  {'Total':<12} {total_mean:>14,.2f}")


def _format_forecast_markdown(result: Any, run_id: str, plan_path: Path) -> str:
    """Produce the saved Markdown report for a forecast."""
    lines: list[str] = []
    lines.append(f"# Forecast: {plan_path.name}")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Plan: `{plan_path}`")
    lines.append(f"- Credible interval: {result.interval_mass:.0%} HDI")
    lines.append("")

    if result.spend_invariant_channels:
        lines.append("## Spend-invariant channels")
        lines.append("")
        lines.append(
            "Saturation could not be estimated from training data. "
            "Reallocation recommendations are not produced for:"
        )
        lines.append("")
        for channel in result.spend_invariant_channels:
            lines.append(f"- `{channel}`")
        lines.append("")

    if result.extrapolation_flags:
        lines.append("## Extrapolation warnings")
        lines.append("")
        lines.append(
            f"{len(result.extrapolation_flags)} plan cell(s) fall outside the "
            "observed historical range. Predictions in these cells extrapolate "
            "the model beyond what data supports."
        )
        lines.append("")
        lines.append("| Period | Channel | Planned | Direction | Observed range |")
        lines.append("|---|---|---|---|---|")
        for flag in result.extrapolation_flags:
            lines.append(
                f"| {flag.period} | `{flag.channel}` | {flag.planned_spend:,.0f} | "
                f"{flag.direction.replace('_', ' ')} | "
                f"{flag.observed_spend_min:,.0f}–{flag.observed_spend_max:,.0f} |"
            )
        lines.append("")

    lines.append("## Predictive draws")
    lines.append("")
    lines.append("| Period | Mean | 95% HDI low | 95% HDI high |")
    lines.append("|---|---|---|---|")
    for period, mean, low, high in zip(
        result.periods, result.mean, result.hdi_low, result.hdi_high, strict=True,
    ):
        lines.append(f"| {period} | {mean:,.2f} | {low:,.2f} | {high:,.2f} |")
    total_mean = sum(result.mean)
    lines.append(f"| **Total** | **{total_mean:,.2f}** | | |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _print_scenario_comparison_table(results: list[Any]) -> None:
    """Print the ranked scenario comparison as a human-readable table."""
    typer.echo("\nScenario comparison (ranked by expected outcome)")
    typer.echo("")
    typer.echo(
        f"  {'Rank':<5} {'Plan':<24} {'Expected outcome':>18} {'95% HDI':>30}"
    )
    for rank, result in enumerate(results, start=1):
        hdi_label = (
            f"{result.expected_outcome_hdi_low:,.0f}–"
            f"{result.expected_outcome_hdi_high:,.0f}"
        )
        typer.echo(
            f"  {rank:<5} {result.plan_name:<24} "
            f"{result.expected_outcome_mean:>18,.2f} {hdi_label:>30}"
        )
    for result in results:
        if result.spend_invariant_channels:
            typer.echo(
                typer.style(
                    f"  {result.plan_name}: spend-invariant channels "
                    f"({', '.join(result.spend_invariant_channels)}) "
                    "are excluded from reallocation guidance.",
                    fg=typer.colors.YELLOW,
                )
            )
        if result.extrapolation_flags:
            typer.echo(
                typer.style(
                    f"  {result.plan_name}: {len(result.extrapolation_flags)} "
                    "extrapolation warning(s) — see saved report.",
                    fg=typer.colors.YELLOW,
                )
            )


def _format_scenario_comparison_markdown(results: list[Any], run_id: str) -> str:
    """Produce the saved Markdown report for a scenario comparison."""
    lines: list[str] = []
    lines.append("# Scenario comparison")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Plans compared: {len(results)}")
    lines.append("")

    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Plan | Expected outcome | 95% HDI low | 95% HDI high |")
    lines.append("|---|---|---|---|---|")
    for rank, result in enumerate(results, start=1):
        lines.append(
            f"| {rank} | `{result.plan_name}` | "
            f"{result.expected_outcome_mean:,.2f} | "
            f"{result.expected_outcome_hdi_low:,.2f} | "
            f"{result.expected_outcome_hdi_high:,.2f} |"
        )
    lines.append("")

    lines.append("## Total spend by channel")
    lines.append("")
    if results:
        all_channels = sorted({c for r in results for c in r.total_spend_by_channel})
        header = "| Plan | " + " | ".join(f"`{c}`" for c in all_channels) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(all_channels) + 1))
        for result in results:
            row = [f"`{result.plan_name}`"]
            for channel in all_channels:
                row.append(f"{result.total_spend_by_channel.get(channel, 0.0):,.0f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    flagged = [r for r in results if r.extrapolation_flags or r.spend_invariant_channels]
    if flagged:
        lines.append("## Caveats")
        lines.append("")
        for result in flagged:
            if result.spend_invariant_channels:
                lines.append(
                    f"- **`{result.plan_name}`** — saturation not estimable "
                    f"for `{', '.join(result.spend_invariant_channels)}`. "
                    "These channels are excluded from reallocation guidance "
                    "(FR-3.2)."
                )
            if result.extrapolation_flags:
                lines.append(
                    f"- **`{result.plan_name}`** — "
                    f"{len(result.extrapolation_flags)} plan cell(s) fall "
                    "outside the observed historical range:"
                )
                for flag in result.extrapolation_flags:
                    lines.append(
                        f"    - {flag.period} · `{flag.channel}`: "
                        f"planned {flag.planned_spend:,.0f} "
                        f"({flag.direction.replace('_', ' ')}; "
                        f"observed {flag.observed_spend_min:,.0f}–"
                        f"{flag.observed_spend_max:,.0f})"
                    )
        lines.append("")
    return "\n".join(lines) + "\n"


def _print_ramp_recommendation(recommendation: Any) -> None:
    """Print the ramp recommendation as a compact human-readable table."""
    typer.echo("\nRisk-adjusted ramp recommendation")
    typer.echo(f"  Verdict: {recommendation.status}")
    typer.echo(f"  Recommended ramp: {recommendation.recommended_fraction:.0%}")
    typer.echo(f"  {recommendation.explanation}")
    if recommendation.blocking_reason:
        typer.echo(f"  Blocking reason: {recommendation.blocking_reason}")

    if not recommendation.candidates:
        return

    typer.echo("")
    typer.echo(
        f"  {'Ramp':<8} {'Pass':<6} {'Expected inc.':>14} "
        f"{'P(improve)':>12} {'P(loss)':>10} {'Failed gates':<28}"
    )
    for candidate in recommendation.candidates:
        failed = ", ".join(candidate.failed_gates) if candidate.failed_gates else "none"
        typer.echo(
            f"  {candidate.fraction:<8.0%} {str(candidate.passes):<6} "
            f"{candidate.expected_increment:>14,.0f} "
            f"{candidate.probability_positive:>12.0%} "
            f"{candidate.probability_material_loss:>10.0%} "
            f"{failed:<28}"
        )


def _ramp_advisor_handoff(
    recommendation: Any,
    summary: Any,
    *,
    spend_by_channel: dict[str, float] | None = None,
) -> str | None:
    """Return an Experiment Advisor handoff for evidence-bound ramp outcomes."""
    if recommendation.status != "test_first" or recommendation.blocking_reason:
        return None

    from wanamaker.advisor import flag_channels

    try:
        flags = flag_channels(summary, spend_by_channel=spend_by_channel)
    except NotImplementedError:
        flags = []

    if flags:
        channel = flags[0].channel
    else:
        contributions = list(getattr(summary, "channel_contributions", []) or [])
        if not contributions:
            return (
                "A controlled test would most reduce the binding gate for this "
                "recommendation."
            )
        channel = max(contributions, key=lambda c: c.mean_contribution).channel
    return f"A test on `{channel}` would most reduce the binding gate."


def _load_lift_priors_if_any(cfg: Any) -> Any:
    """Return lift-test priors keyed by channel if the config supplies them.

    Used by the report command so the Trust Card's lift-test-consistency
    dimension can be evaluated when calibration data is part of the run.
    Returns ``None`` when no lift-test CSV is configured, which omits the
    dimension cleanly.
    """
    from wanamaker.model.builder import build_model_spec

    if cfg.data.lift_test_csv is None:
        return None
    spec = build_model_spec(cfg)
    return spec.lift_test_priors or None


def _safe_advisor_recommendations(
    summary: Any,
    *,
    spend_by_channel: dict[str, float] | None = None,
) -> list[str]:
    """Experiment Advisor bullets for the executive summary.

    Each ``ChannelFlag.rationale`` is already a complete sentence
    (per FR-5.5), so the report bullets are the rationales verbatim —
    no extra prefix here.

    ``NotImplementedError`` is caught in case a future engine swap or
    advisor refactor reverts to a stub; in that case the executive
    summary falls back to its own "consider an experiment for weak
    channel X" template logic.
    """
    from wanamaker.advisor import flag_channels

    try:
        flags = flag_channels(summary, spend_by_channel=spend_by_channel)
    except NotImplementedError:
        return []
    return [flag.rationale for flag in flags]


def _spend_by_channel_from_training_data(cfg: Any) -> dict[str, float] | None:
    """Total spend per channel over the training window, or ``None``.

    Used by the report command so the Experiment Advisor's rationale
    can quote actual dollar spend rather than a modelled-contribution
    proxy. Returns ``None`` when the config does not list spend
    columns; the advisor handles that fallback cleanly.
    """
    from wanamaker.data.io import load_input_csv

    spend_columns = list(cfg.data.spend_columns or [])
    if not spend_columns:
        return None
    data = load_input_csv(cfg.data)
    return {column: float(data[column].sum()) for column in spend_columns}


if __name__ == "__main__":  # pragma: no cover
    app()
