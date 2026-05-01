# Wanamaker

*Knowing which half.*

An open-source Bayesian marketing mix model for teams that need credible
measurement without a PhD program.

> "Half the money I spend on advertising is wasted; the trouble is I don't
> know which half." — attributed to John Wanamaker (probably apocryphally;
> the misattribution is part of the story).

---

**Status:** Pre-build. The combined BRD/PRD is locked at v0.4 and lives in
[`docs/wanamaker_brd_prd.md`](docs/wanamaker_brd_prd.md). The current phase
is **Phase -1**: engine decision spike and persona validation. Nothing in
this repository fits a model yet.

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
pip install wanamaker
wanamaker run --example public_benchmark
```

That's the whole thing. The command runs the readiness diagnostic, fits a
quick-mode Bayesian model on the bundled `public_benchmark` dataset, and
prints the executive summary plus the Model Trust Card. The full report
is also written to `.wanamaker/runs/<run_id>/report.md`. Expect a few
minutes on a modern laptop.

## Reading order

1. [BRD/PRD](docs/wanamaker_brd_prd.md) — strategic and product context
2. [Privacy and Data Handling](docs/privacy.md) — confidentiality and data isolation guarantees
3. [`AGENTS.md`](AGENTS.md) — guidance for AI coding assistants and human
   contributors on how to work in this codebase
4. The `Comparison to Other Tools` page (forthcoming) — Wanamaker vs.
   Robyn, Meridian, Recast, PyMC-Marketing

## License

MIT — see [`LICENSE`](LICENSE).
