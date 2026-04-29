# Decision 0001: Bayesian Engine Selection

**Status:** Pending — output of Phase -1 workstream A
**Decision by:** Phase -1 spike results
**Date:** TBD

---

## Context

Wanamaker's core value proposition depends on Bayesian inference: credible intervals,
posterior anchoring for refresh accountability, lift-test calibration, and convergence
diagnostics all require a proper MCMC sampler. The engine choice is the single biggest
open technical risk (R-1 in the BRD/PRD): installation friction on Windows and Apple
Silicon killed LightweightMMM's adoption, and the wrong choice here could do the same.

The decision criterion is deliberately product-oriented, not statistical:
**"Can the target user install it on Windows in five minutes and get a credible
result in under 30 minutes on a laptop CPU?"**

---

## Options Evaluated

### Option A: PyMC

- **Install:** `pip install pymc` — clean on Linux, macOS (Intel + Apple Silicon), Windows
  since the move off the theano/aesara stack to pytensor + numba/C
- **Runtime:** Marginally slower than NumPyro for very large models; within tolerance
  for the v1 design center (150 weeks x 12 channels, national-level)
- **Ecosystem:** ArviZ integration for diagnostics and posterior summaries; strong
  community; `InferenceData` / NetCDF as a stable posterior artifact format
- **Risk:** pytensor compilation can add first-run latency; Windows support has
  historically been shaky but appears stable as of PyMC 5.x

### Option B: NumPyro / JAX

- **Install:** Requires `jax` + `jaxlib`, which historically had platform-specific
  wheels (especially on Windows). As of JAX 0.4.x, Windows CPU support improved.
- **Runtime:** Fastest of the three for large models; JIT compilation amortizes
  across repeated fits
- **Ecosystem:** Excellent but smaller community than PyMC; less MMM prior art
- **Risk:** JAX installation friction is the main reason LightweightMMM (Google's
  own tool) lost adoption. Must be re-evaluated empirically in Phase -1.

### Option C: Stan via cmdstanpy

- **Install:** Compiles a small C++ toolchain on first install (CmdStan); adds
  friction for the target persona who just wants to `pip install` and go
- **Runtime:** Comparable to PyMC; the Stan sampler (NUTS) is the reference
  implementation
- **Ecosystem:** Largest, most mature Bayesian ecosystem; unmatched library quality;
  strong stability guarantees
- **Risk:** The C++ compilation step is a real barrier for non-technical users
  and is hard to containerize away cleanly

---

## Decision Criteria (Phase -1 Spike)

The spike should test each candidate on these dimensions:

1. **Clean install on Windows 11** (no admin rights, no conda, pip only): pass/fail
2. **Clean install on macOS Apple Silicon** (pip only): pass/fail
3. **Clean install on Linux** (Ubuntu 22.04 LTS, pip only): pass/fail
4. **Time to first fit** on the synthetic ground-truth benchmark
   (150 weeks, 12 channels, standard runtime tier): minutes
5. **Peak memory** on the same benchmark: GB
6. **R-hat and ESS** on the same benchmark (convergence quality): values

---

## Decision

*To be filled in after Phase -1 spike completes.*

**Chosen engine:**
**Rationale:**
**Rejected options and why:**
**Remaining risks:**

---

## Consequences

Once this decision lands:
- Add the chosen engine to `pyproject.toml` production dependencies
- Implement `engine/<name>.py` satisfying the `Engine` Protocol in `engine/base.py`
- The backend must produce `PosteriorSummary` (from `engine/summary.py`) as its
  primary output; `Posterior.raw` holds the engine-native object for expert access
- Update `AGENTS.md` engine section to reflect the decision
- Update `docs/architecture.md` Section 3.7 open decisions table
- Decide whether Docker or pip is the primary recommended install path in the README
  (depends on this decision)
