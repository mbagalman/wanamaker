"""Plain-English translations of Trust Card dimensions for non-technical readers.

The full executive summary and the showcase use the *technical* explanation
that ``trust_card.compute`` produces — analyst-facing strings that reference
credible intervals, R-hat, MAPE, etc. The Trust Card one-pager is for
forwarding to non-technical executives, so it uses these translated strings
instead.

Each dimension maps to (status -> consequence sentence) plus an "if weak,
what should the reader do" decision sentence. The consequence sentence
answers "what does this mean about the analysis?"; the decision sentence
answers "and so what should I change?".

Voice: dry, direct, no jargon. None of the following words should appear in
the translations: credible interval, HDI, R-hat, MCMC, Bayesian, posterior,
ESS, Gelman-Rubin. The unit test ``test_trust_card_one_pager`` enforces this
on the rendered HTML.
"""

from __future__ import annotations

from typing import Final

from wanamaker.trust_card.card import TrustStatus

# The friendly label users see on the one-pager. The internal key remains
# ``saturation_identifiability`` etc. so the report and showcase keep their
# precise vocabulary; only the executive-facing artifact is translated.
DIMENSION_LABELS: Final[dict[str, str]] = {
    "convergence": "Math stability",
    "holdout_accuracy": "Recent-week accuracy",
    "refresh_stability": "Refresh stability",
    "prior_sensitivity": "Data-driven vs. assumption-driven",
    "saturation_identifiability": "Diminishing-returns curves",
    "lift_test_consistency": "Agreement with experiments",
}


# (dimension, status) -> plain-English consequence sentence.
# Missing entries fall back to the status label only.
_CONSEQUENCES: Final[dict[tuple[str, TrustStatus], str]] = {
    ("convergence", TrustStatus.PASS): (
        "The math behind the model settled cleanly. Estimates are stable "
        "across re-runs."
    ),
    ("convergence", TrustStatus.MODERATE): (
        "The math behind the model settled, with minor irregularities. "
        "Re-running may shift estimates slightly."
    ),
    ("convergence", TrustStatus.WEAK): (
        "The math did not settle cleanly. Some estimates may shift meaningfully "
        "if you re-run; treat them as approximate."
    ),
    ("holdout_accuracy", TrustStatus.PASS): (
        "The model's recent forecasts have been tracking actuals well."
    ),
    ("holdout_accuracy", TrustStatus.MODERATE): (
        "The model's recent forecasts are roughly tracking actuals, with some "
        "misses worth noting."
    ),
    ("holdout_accuracy", TrustStatus.WEAK): (
        "The model is not predicting recent weeks well. Be cautious about "
        "forward forecasts."
    ),
    ("refresh_stability", TrustStatus.PASS): (
        "When new data arrives, historical estimates stay stable. Decisions "
        "made on the prior run still hold."
    ),
    ("refresh_stability", TrustStatus.MODERATE): (
        "Some historical estimates moved with the latest data, but most are "
        "explained by routine change. Watch the diff for surprises."
    ),
    ("refresh_stability", TrustStatus.WEAK): (
        "Recent estimates moved enough that prior-run decisions may no longer "
        "hold. Read the refresh diff before acting."
    ),
    ("prior_sensitivity", TrustStatus.PASS): (
        "The data drove the answer. Built-in modeling assumptions had little "
        "influence on the result."
    ),
    ("prior_sensitivity", TrustStatus.MODERATE): (
        "The data is doing most of the work, but built-in assumptions matter "
        "on a few estimates."
    ),
    ("prior_sensitivity", TrustStatus.WEAK): (
        "The data is not pinning down the answer. Built-in assumptions are "
        "doing more work than the evidence — treat conclusions as soft."
    ),
    ("saturation_identifiability", TrustStatus.PASS): (
        "There was enough variation in spend over time to learn each channel's "
        "diminishing-returns curve."
    ),
    ("saturation_identifiability", TrustStatus.MODERATE): (
        "Most channels' diminishing-returns curves are reasonably learned; a "
        "few are uncertain. See the per-channel notes in the full report."
    ),
    ("saturation_identifiability", TrustStatus.WEAK): (
        "One or more channels had near-constant spend, so their "
        "diminishing-returns curves are not learned from data. They should be "
        "left out of reallocation decisions."
    ),
    ("lift_test_consistency", TrustStatus.PASS): (
        "The model agrees with the field experiments you provided."
    ),
    ("lift_test_consistency", TrustStatus.MODERATE): (
        "The model and field experiments are mostly aligned, with minor "
        "disagreements worth noting."
    ),
    ("lift_test_consistency", TrustStatus.WEAK): (
        "The model disagrees with one or more field experiments. Reconcile "
        "the two before acting on either."
    ),
}


# Decision sentence shown in the "what this means for decisions" block when a
# dimension is weak. Missing entry falls back to a generic caveat.
_DECISIONS: Final[dict[str, str]] = {
    "convergence": (
        "Treat channel estimates as approximate; small reallocation calls are "
        "OK, big bets are not."
    ),
    "holdout_accuracy": (
        "Use the model's directional read, not its point predictions, for "
        "forward-looking budget calls."
    ),
    "refresh_stability": (
        "Do not act on conclusions that contradict the previous run without "
        "first reviewing the refresh diff."
    ),
    "prior_sensitivity": (
        "Treat the analysis as suggestive. Aim for plans that work across a "
        "range of reasonable assumptions."
    ),
    "saturation_identifiability": (
        "Avoid large reallocation calls based on response-curve shape. "
        "Channels with constant historical spend are off-limits for "
        "reallocation."
    ),
    "lift_test_consistency": (
        "Reconcile the model and the experiment before acting on either. "
        "Default to trusting the experiment when in doubt."
    ),
}


# Words that must NEVER appear in the rendered one-pager. The plain-English
# guard test asserts this on the rendered HTML.
JARGON_BLACKLIST: Final[tuple[str, ...]] = (
    "credible interval",
    "HDI",
    "R-hat",
    "MCMC",
    "Bayesian",
    "posterior",
    "ESS",
    "Gelman-Rubin",
)


def consequence_for(dimension: str, status: TrustStatus) -> str:
    """Return the plain-English consequence sentence for a dimension/status.

    Falls back to a status-only sentence when an unknown dimension is
    encountered, so future dimensions added to the Trust Card don't break
    the one-pager render.
    """
    explicit = _CONSEQUENCES.get((dimension, status))
    if explicit is not None:
        return explicit
    if status == TrustStatus.PASS:
        return "This dimension passes."
    if status == TrustStatus.MODERATE:
        return "This dimension is moderate — read the full report for details."
    return "This dimension is weak — read the full report before acting."


def decision_for(dimension: str) -> str:
    """Return the executive-facing decision sentence for a weak dimension."""
    return _DECISIONS.get(
        dimension,
        "Read the full report before acting on conclusions that touch this dimension.",
    )


def label_for(dimension: str) -> str:
    """Return the friendly display label for a dimension name."""
    return DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").capitalize())
