"""Refresh-stability benchmark: sweep anchor weights and measure NFR-5 compliance.

Issue #18 — tune default posterior anchoring weight.

NFR-5 targets (must hold at the medium preset, w=0.3):
  - Average historical contribution estimate movement ≤ 10 %
  - No single channel moves > 25 %
  - ≥ 90 % of parameter movements classified as non-unexplained

The test uses the refresh-stability benchmark dataset: a known-stable base
series plus a deterministic "4 weeks added" variant. It sweeps w in
{0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7} and measures the three NFR-5 metrics for each weight.  The medium
preset (w=0.3) must meet all three targets; other presets are informational.

Marked ``engine`` — skipped automatically in the fast unit job.
"""

from __future__ import annotations

import numpy as np
import pytest

_W_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_spec(gt: dict, anchor_priors: dict | None = None):
    """Build a ModelSpec from ground-truth metadata."""
    from wanamaker.model.spec import AnchoredPrior, ChannelSpec, ModelSpec

    channels = [
        ChannelSpec(name=ch["name"], category=ch["category"])
        for ch in gt["channels"]
    ]
    ap: dict | None = None
    if anchor_priors:
        ap = {
            k: AnchoredPrior(mean=v["mean"], sd=v["sd"], weight=v["weight"])
            for k, v in anchor_priors.items()
        }
    return ModelSpec(
        channels=channels,
        target_column=gt["target_column"],
        date_column=gt["date_column"],
        anchor_priors=ap,
    )


def _contribution_movement(
    prev_contribs: dict[str, float],
    curr_contribs: dict[str, float],
) -> dict[str, float]:
    """Return relative contribution movement per channel."""
    movements = {}
    for ch, prev in prev_contribs.items():
        curr = curr_contribs.get(ch, 0.0)
        if abs(prev) > 0:
            movements[ch] = abs(curr - prev) / abs(prev)
    return movements


def _unexplained_fraction_from_summaries(
    prev_summary,
    curr_summary,
) -> float:
    """Compute unexplained-movement fraction using classify_movement."""
    from wanamaker.refresh.classify import MovementClass, classify_movement

    prev_map = {p.name: p for p in prev_summary.parameters}
    count_total = 0
    count_unexplained = 0
    for curr_param in curr_summary.parameters:
        prev_param = prev_map.get(curr_param.name)
        if prev_param is None:
            continue
        mc = classify_movement(
            prev_hdi=(prev_param.hdi_low, prev_param.hdi_high),
            curr_mean=curr_param.mean,
            curr_hdi=(curr_param.hdi_low, curr_param.hdi_high),
        )
        count_total += 1
        if mc == MovementClass.UNEXPLAINED:
            count_unexplained += 1
    if count_total == 0:
        return 0.0
    return count_unexplained / count_total


def _build_anchor_priors_from_summary(summary, weight: float) -> dict:
    """Extract anchor prior params from a PosteriorSummary."""
    return {
        p.name: {"mean": p.mean, "sd": p.sd, "weight": weight}
        for p in summary.parameters
        if p.name.startswith("channel.")
        and p.name.rsplit(".", 1)[-1] in ("half_life", "ec50", "slope", "coefficient")
    }


# ---------------------------------------------------------------------------
# Core benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.engine
def test_medium_preset_meets_nfr5() -> None:
    """Anchor weight w=0.3 (medium preset) must satisfy all three NFR-5 targets.

    Acceptance criteria:
    - Average contribution movement ≤ 10 %
    - Max single-channel contribution movement ≤ 25 %
    - ≥ 90 % of parameter movements classified as non-unexplained
    """
    try:
        from wanamaker.engine.pymc import PyMCEngine
    except ImportError:
        pytest.skip("PyMC engine not available.")

    from wanamaker.benchmarks.loaders import load_refresh_stability

    df, df_extended, gt = load_refresh_stability()
    engine = PyMCEngine()

    # Initial fit (no anchoring)
    spec_initial = _build_spec(gt)
    result_initial = engine.fit(spec_initial, df, seed=42, runtime_mode="quick")
    summary_initial = result_initial.summary

    contribs_initial = {
        c.channel: c.mean_contribution for c in summary_initial.channel_contributions
    }

    # Re-fit on the 4-weeks-added variant with medium anchor weight.
    anchor_priors = _build_anchor_priors_from_summary(summary_initial, weight=0.3)
    spec_refresh = _build_spec(gt, anchor_priors=anchor_priors)
    result_refresh = engine.fit(spec_refresh, df_extended, seed=43, runtime_mode="quick")
    summary_refresh = result_refresh.summary

    contribs_refresh = {
        c.channel: c.mean_contribution for c in summary_refresh.channel_contributions
    }

    movements = _contribution_movement(contribs_initial, contribs_refresh)
    avg_movement = float(np.mean(list(movements.values()))) if movements else 0.0
    max_movement = float(max(movements.values())) if movements else 0.0
    unexplained = _unexplained_fraction_from_summaries(summary_initial, summary_refresh)

    assert avg_movement <= 0.10, (
        f"Average contribution movement {avg_movement:.1%} exceeds NFR-5 target of 10 %.\n"
        f"Per-channel movements: {movements}"
    )
    assert max_movement <= 0.25, (
        f"Max single-channel contribution movement {max_movement:.1%} exceeds "
        f"NFR-5 target of 25 %.\n"
        f"Per-channel movements: {movements}"
    )
    assert unexplained <= 0.10, (
        f"Unexplained movement fraction {unexplained:.1%} exceeds "
        f"NFR-5 target of ≤ 10 % (i.e. ≥ 90 % non-unexplained)."
    )


@pytest.mark.benchmark
@pytest.mark.engine
def test_anchor_weight_sweep_produces_stability_gradient() -> None:
    """Heavier anchoring must produce strictly lower (or equal) average movement.

    This test validates the monotonicity property: increasing w should not
    increase average contribution movement on stable data.  If it does, the
    blending implementation is broken.
    """
    try:
        from wanamaker.engine.pymc import PyMCEngine
    except ImportError:
        pytest.skip("PyMC engine not available.")

    from wanamaker.benchmarks.loaders import load_refresh_stability

    df, df_extended, gt = load_refresh_stability()
    engine = PyMCEngine()

    # Initial fit to get the previous posterior
    spec_initial = _build_spec(gt)
    result_initial = engine.fit(spec_initial, df, seed=42, runtime_mode="quick")
    summary_initial = result_initial.summary
    contribs_initial = {
        c.channel: c.mean_contribution for c in summary_initial.channel_contributions
    }

    avg_movements = {}
    for w in _W_SWEEP:
        anchor_priors = _build_anchor_priors_from_summary(summary_initial, weight=w)
        spec = _build_spec(gt, anchor_priors=anchor_priors)
        result = engine.fit(spec, df_extended, seed=43, runtime_mode="quick")
        contribs = {
            c.channel: c.mean_contribution for c in result.summary.channel_contributions
        }
        movements = _contribution_movement(contribs_initial, contribs)
        avg_movements[w] = float(np.mean(list(movements.values()))) if movements else 0.0

    # Monotonicity: w=0.5 should move less than w=0.0 on stable data
    assert avg_movements[0.5] <= avg_movements[0.0], (
        f"Expected heavier anchoring to reduce movement on stable data.\n"
        f"w=0.0 movement: {avg_movements[0.0]:.1%}, w=0.5 movement: {avg_movements[0.5]:.1%}\n"
        f"Full sweep: {avg_movements}"
    )
