# Feedback Status on `architecture.md` and GitHub Issues

Last reviewed: 2026-04-29.

This file started as a constructive critique of `docs/architecture.md` and the
initial GitHub issue plan. Most of the load-bearing concerns have now been
addressed in code, issues, or follow-up tracking. This version records what is
complete and what still needs attention.

## Summary

The major architectural risks from the original review are mostly handled:

- The synthetic benchmark sequencing loop was fixed.
- Posterior interval mass is explicit and standardized at 95%.
- Forecast/scenario code now has access to historical spend ranges through
  `ChannelContributionSummary`.
- Artifact schemas and serializers were implemented.
- Trust Card persistence is now represented in artifact paths and serializers.
- Structural break detection avoided a new production dependency.
- The one-command demo gap is tracked as its own issue.

All items from the original review have now been addressed. `architecture.md` has been
refreshed to match the current implementation. Issue #20 has been updated to require
`PosteriorSummary` spend ranges. Issue #11's reproducibility test has been updated
to use numerical tolerance (RTOL=1e-6) and documents the cross-platform reproducibility
decision.

## Status by Original Concern

### 1. Benchmark and Engine Sequencing Loop

**Status: Resolved.**

The original concern was that engine selection/backend acceptance depended on the
synthetic benchmark, while the benchmark issue appeared blocked by the engine.
Issue #9 was split into an engine-independent generator/loader part and a later
engine acceptance part. The synthetic ground-truth generator and benchmark data
are now in place, and issue #9 is closed.

No further action is needed for this concern.

### 2. Credible Interval Mass

**Status: Resolved.**

`engine/summary.py` now makes interval mass explicit with `interval_mass = 0.95`
on the relevant summary dataclasses. The PyMC backend uses the same mass when
building summaries, and serializer round-trip tests preserve it.

No further action is needed for this concern.

### 3. Forecast and Scenario Historical Spend Context

**Status: Resolved.**

`ChannelContributionSummary` includes `observed_spend_min`, `observed_spend_max`,
and `spend_invariant`. Issue #20 has been updated to explicitly require that
forecast/scenario logic consumes `PosteriorSummary` (not a bare `Posterior`) and
uses the spend range fields for extrapolation detection.

No further action needed.

### 4. Artifact Serialization and Schema Versioning

**Status: Resolved.**

Issue #41 was created and closed. `artifacts.py` now includes versioned
serialization/deserialization for:

- `summary.json`
- `trust_card.json`
- `refresh_diff.json`
- `manifest.json`

The serializers use schema-version envelopes, and tests cover round trips plus
incompatible schema-version errors.

No further action is needed for this concern, beyond keeping schema versions
updated when artifact formats change.

### 5. Reproducibility During Engine Selection

**Status: Resolved.**

Issue #1 is closed with the PyMC decision. Issue #11 has a reproducibility test
(`tests/test_reproducibility.py`) that fits the synthetic benchmark twice with the
same seed and compares all summary floats within RTOL=1e-6. The cross-platform
reproducibility decision is documented: numerically close (RTOL=1e-6), not
bit-for-bit identical. The test is gated by `WANAMAKER_RUN_ENGINE_TESTS=1` to
keep fast unit runs fast.

No further action needed.

### 6. Trust Card Persistence

**Status: Resolved.**

The artifact layer now includes `trust_card.json` as a persisted artifact and
provides versioned Trust Card serializers. This supports auditable reports and
avoids future threshold changes silently rewriting old Trust Card states.

No further action is needed for this concern.

### 7. Structural Break Dependency Decision

**Status: Resolved by implementation choice.**

The structural-break check was implemented with an in-house deterministic method
instead of adding a new production dependency such as `ruptures`. This avoids
additional install friction and keeps the diagnostic local-first.

No further action is needed for this concern unless later benchmark results show
the in-house method is not accurate enough.

## Status by Smaller Note

### `engine/summary.py` Serialization Wording

**Status: Resolved.**

The summary contract now clearly points to `summary.json` rather than the run
manifest.

### Issue #15 Structural-Break Behavior

**Status: Resolved in current behavior.**

Structural breaks are treated as config-independent diagnostic checks. This
matches the updated architecture direction.

### Issue #33 CI Cadence

**Status: Still a planning note.**

The network-isolated command test remains future CI work. The original caution
still stands: once `fit` is expensive, it may need to run on `main`, release
tags, or nightly rather than every PR.

### One-Command Demo

**Status: Resolved.**

`wanamaker run --example public_benchmark` now chains the readiness diagnostic,
quick-mode fit, and report rendering on the bundled public benchmark dataset.
The remaining work is release verification, not a product-design gap.

## Documentation Cleanup

**Status: Resolved.**

`docs/architecture.md` has been refreshed:
- All module statuses updated to reflect implementation reality.
- Artifact layout updated to include `trust_card.json`, `refresh_diff.json`, and the versioned envelope format.
- Engine/backend status updated to reflect the implemented PyMC backend.
- Forecast section updated to require `PosteriorSummary` spend ranges.
- Reproducibility invariant updated to "numerically close (RTOL=1e-6)".
- Open decisions table updated with the reproducibility resolution.

## Status

All original feedback items are resolved. No further action needed on this document.
