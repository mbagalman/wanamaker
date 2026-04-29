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

import shutil
from datetime import datetime, timezone
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
    from wanamaker import __version__
    from wanamaker.artifacts import (
        hash_config,
        hash_file,
        list_runs,
        make_run_fingerprint,
        run_paths,
        serialize_summary,
        write_manifest,
    )
    from wanamaker.config import load_config
    from wanamaker.data.io import load_input_csv
    from wanamaker.diagnose.readiness import CheckSeverity, ReadinessLevel
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
    timestamp_utc = datetime.now(timezone.utc)
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


# ---------------------------------------------------------------------------
# Private helpers (not part of the public CLI surface)
# ---------------------------------------------------------------------------


def _run_diagnostics(data: object, cfg: object) -> str:
    """Run available diagnostic checks and return a ``ReadinessLevel`` string.

    Checks that are not yet implemented (raise ``NotImplementedError``) are
    silently skipped so the fit command works as more checks land in
    issues #12 and #13.
    """
    import pandas as pd

    from wanamaker.config import WanamakerConfig
    from wanamaker.diagnose.checks import check_structural_breaks
    from wanamaker.diagnose.readiness import CheckSeverity, CheckResult, ReadinessLevel

    assert isinstance(data, pd.DataFrame)
    assert isinstance(cfg, WanamakerConfig)

    results: list[CheckResult] = []

    # Structural-break check (implemented in issue #14)
    try:
        results.extend(
            check_structural_breaks(data, cfg.data.target_column, cfg.data.date_column)
        )
    except NotImplementedError:
        pass

    # Future checks (issues #12 and #13) will extend this list.

    blockers = [r for r in results if r.severity == CheckSeverity.BLOCKER]
    warnings = [r for r in results if r.severity == CheckSeverity.WARNING]

    if blockers:
        level = ReadinessLevel.NOT_RECOMMENDED
    elif warnings:
        level = ReadinessLevel.USABLE_WITH_WARNINGS
    else:
        level = ReadinessLevel.READY

    if warnings or blockers:
        for r in results:
            colour = typer.colors.RED if r.severity == CheckSeverity.BLOCKER else typer.colors.YELLOW
            typer.echo(typer.style(f"  [{r.severity.value.upper()}] {r.message}", fg=colour))

    typer.echo(f"Readiness: {level.value}")
    return level.value


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
    """Return the installed PyMC version string, or 'unknown' if not available."""
    try:
        import pymc
        return str(pymc.__version__)
    except ImportError:
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


if __name__ == "__main__":  # pragma: no cover
    app()
