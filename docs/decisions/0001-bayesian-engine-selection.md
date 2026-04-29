# Decision 0001: Bayesian Engine Selection

**Status:** Accepted
**Decision by:** Maintainer call based on prior experience
**Date:** 2026-04-29

---

## Context

Wanamaker's core value proposition depends on Bayesian inference: credible intervals,
posterior anchoring for refresh accountability, lift-test calibration, and convergence
diagnostics all require a proper MCMC sampler.

The original Phase -1 plan was to evaluate PyMC, NumPyro/JAX, and Stan via a formal
engine spike. The product decision criterion was deliberately practical:

> Can the target user install it on Windows in five minutes and get a credible result
> in under 30 minutes on a laptop CPU?

The maintainer has chosen to make the engine decision now based on prior experience,
rather than spend a full spike comparing all three engines. Benchmark and reproducibility
checks remain required for the PyMC backend before release, but they no longer block the
engine choice itself.

---

## Options Considered

### PyMC

- Mature Bayesian modeling ecosystem.
- Strong ArviZ integration for diagnostics, posterior summaries, and NetCDF artifacts.
- Large enough user community that senior data scientists can review and extend the model.
- Expected to be fast enough for the v1 design center: weekly, national-level MMM with
  roughly 150 periods and 12 channels.
- Main risks: PyTensor first-run compilation latency, Windows verification, and
  same-seed reproducibility still need testing in the Wanamaker backend.

### NumPyro / JAX

- Technically strong and likely faster on larger models.
- Installation history is less friendly for the target persona, especially given the
  JAX stack's platform-specific sharp edges.
- Too close to the LightweightMMM adoption failure mode that Wanamaker is explicitly
  trying to avoid.

### Stan via cmdstanpy

- Statistically excellent and extremely mature.
- Requires CmdStan and a working C++ toolchain, which adds too much first-run friction
  for Wanamaker's target user.

---

## Decision

Use **PyMC** as Wanamaker's Bayesian engine.

This decision optimizes for installability, debuggability, artifact support, and community
legibility over peak sampler speed. Those traits matter more for Wanamaker than raw
performance because the primary persona is a marketing analyst or agency analytics lead,
not a Bayesian infrastructure specialist.

---

## Consequences

- `pymc` is a production dependency.
- The concrete backend will live at `src/wanamaker/engine/pymc.py`.
- Feature code still imports through `wanamaker.engine`; direct PyMC imports are isolated
  to the engine backend and backend-specific tests.
- The backend must produce `PosteriorSummary` from `wanamaker.engine.summary` as its
  primary output.
- `Posterior.raw` may hold the PyMC / ArviZ-native object for expert access, but core
  modules should consume typed summaries.
- The README and installation docs should treat pip as the primary target unless
  cross-platform CI proves Docker needs to be the recommended path.

---

## Follow-Up Verification

These checks move from "engine decision gate" to "PyMC backend acceptance":

- Clean install on Windows, macOS Apple Silicon, and Linux.
- Runtime and peak memory on the synthetic ground-truth benchmark.
- Convergence quality: R-hat, ESS, and divergences.
- Same-machine reproducibility with identical data, config, and seed.
- Cross-platform numerical stability within documented tolerances.
