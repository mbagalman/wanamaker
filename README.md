# Wanamaker

*Knowing which half.*

An open-source Bayesian marketing mix model for teams that need credible
measurement without a PhD program.

> "Half the money I spend on advertising is wasted; the trouble is I don't
> know which half." — attributed to John Wanamaker (probably apocryphally;
> the misattribution is part of the story).

---

**Status:** Active pre-1.0 development. The combined BRD/PRD is locked at v0.4
and lives in [`docs/wanamaker_brd_prd.md`](docs/wanamaker_brd_prd.md). The
core local workflow is now in place: `diagnose`, `fit`, `report`, `showcase`,
`trust-card`, `export`, `forecast`, `compare-scenarios`, `recommend-ramp`,
`refresh`, and `run --example public_benchmark`.

## What this is for

Mid-market consumer brands and the agencies that serve them — companies
with $5M–$200M in annual media spend, 1.5–3 years of weekly historical
data, and one analyst rather than a data science team.

If you have a deep data science bench and are running geo-hierarchical
models across hundreds of regions, you want [Meridian](https://github.com/google/meridian).
If you want a fully managed measurement service, you want
[Recast](https://getrecast.com) or [Measured](https://www.measured.com).
Wanamaker sits in the gap.

## What makes it different

1. **Refresh accountability** — when new data arrives and historical estimates
   change, Wanamaker shows what changed, why, and whether the change should
   alter your decisions. Light posterior anchoring (default on) plus a diff
   report that classifies every movement.
2. **Trustable usability** — pre-flight data readiness diagnostic, post-fit
   trust card with named credibility dimensions, plain-English executive
   summary generated from deterministic templates (no LLM).
3. **Local-first** — no network calls in any core operation. No telemetry,
   ever. Your spend data does not leave the machine.
4. **Decision-oriented** — forecasting and conservative scenario comparison,
   not just descriptive analysis.

## One-command end-to-end example

```bash
git clone https://github.com/mbagalman/wanamaker.git
cd wanamaker
pip install -e ".[dev]"
wanamaker run --example public_benchmark
```

That's the whole thing. The command runs the readiness diagnostic, fits a
quick-mode Bayesian model on the bundled `public_benchmark` dataset, and
prints the executive summary plus the Model Trust Card. Four artifacts land
in `.wanamaker/runs/<run_id>/`:

| File | Audience | Purpose |
|---|---|---|
| `report.md` | analyst | Full executive summary + Trust Card in Markdown — git-friendly, paste into Slack |
| `showcase.html` | stakeholders | Self-contained HTML with channel charts, response curves, waterfall — email this |
| `trust_card.html` | executives | One-page plain-English Trust Card — forward when "do I trust this MMM?" comes up |
| `summary.xlsx` | analysts | Structured tables (channels, ROI, parameters, scenarios) for pivots and slicing |

Expect a few minutes on a modern laptop.

## Try it without installing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbagalman/wanamaker/blob/master/notebooks/quickstart.ipynb)

The same flow runs end-to-end in [a hosted Colab notebook](notebooks/quickstart.ipynb)
— useful for evaluators who don't want to install PyMC locally first.

## How Wanamaker compares to other MMM tools

Wanamaker is built around the *workflow* gaps the other open-source MMM
tools leave to the user — readiness checks, credibility audit, refresh
accountability, decision-grade reports. The math layer is intentionally
the same proven Bayesian foundation everyone else uses.

|                                              | [Robyn](https://github.com/facebookexperimental/Robyn) | [Meridian](https://github.com/google/meridian) | [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | **Wanamaker** |
| -------------------------------------------- | :----------------------------------------------------: | :--------------------------------------------: | :-----------------------------------------------------------: | :-----------: |
| Full posterior distributions                 |                           –                            |                       ✓                        |                               ✓                               |     **✓**     |
| Pre-fit data readiness diagnostic            |                           –                            |                       –                        |                               –                               |     **✓**     |
| Post-fit Trust Card (credibility audit)      |                           –                            |                       –                        |                               –                               |     **✓**     |
| Refresh diff with movement classification    |                           –                            |                       –                        |                               –                               |     **✓**     |
| Risk-adjusted allocation ramp                |                           –                            |                       –                        |                               –                               |     **✓**     |
| Plain-English executive summary              |                        partial                         |                       –                        |                               –                               |     **✓**     |
| Self-contained HTML stakeholder report       |                           –                            |                       –                        |                               –                               |     **✓**     |
| Lift-test calibration                        |                           ✓                            |                       ✓                        |                               ✓                               |       ✓       |
| Geo-level hierarchical modeling              |                        partial                         |                       ✓                        |                            possible                           |    – (v1)     |
| Built-in budget optimizer                    |                       ✓ (mature)                       |                    partial                     |                               –                               |    – (v1)     |
| CPU-only at typical mid-market scale         |                           ✓                            |                  GPU advised                   |                               ✓                               |       ✓       |

**Where Wanamaker pulls ahead** is the operational layer above the math —
the work it takes to turn a fitted Bayesian model into a recurring decision
cadence. **Where Wanamaker lags** is intentional v1 scope: no geo-hierarchy,
no built-in budget optimizer.

If you have geo-level data, a JAX-comfortable team, or 50+ channels, Meridian
will fit better. If your team lives in R and budget allocation is the primary
output, Robyn will fit better. If you want maximum modeling flexibility and
have senior Bayesian practitioners on staff, PyMC-Marketing will fit better.
If you can budget $80–120K/year for a managed service,
[Recast](https://getrecast.com) is the obvious commercial comparison.

Full long-form comparison with honest tradeoffs:
[`docs/comparison.md`](docs/comparison.md).

## Reading order

1. [BRD/PRD](docs/wanamaker_brd_prd.md) — strategic and product context
2. [Privacy and Data Handling](docs/privacy.md) — confidentiality and data isolation guarantees
3. [`AGENTS.md`](AGENTS.md) — guidance for AI coding assistants and human
   contributors on how to work in this codebase
4. [Comparison to Other Tools](docs/comparison.md) — Wanamaker vs. Robyn,
   Meridian, Recast, PyMC-Marketing

## License

MIT — see [`LICENSE`](LICENSE).
