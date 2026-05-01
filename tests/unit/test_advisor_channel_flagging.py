"""Unit tests for the Experiment Advisor's channel flagging (#29 / FR-5.5)."""

from __future__ import annotations

from wanamaker.advisor import ChannelFlag, flag_channels
from wanamaker.engine.summary import (
    ChannelContributionSummary,
    PosteriorSummary,
)


def _channel(
    name: str,
    *,
    contribution: float = 5000.0,
    contribution_hdi: tuple[float, float] = (4500.0, 5500.0),
    roi_mean: float = 2.0,
    roi_hdi: tuple[float, float] = (1.8, 2.2),
    spend_invariant: bool = False,
    observed_min: float = 10.0,
    observed_max: float = 50.0,
) -> ChannelContributionSummary:
    return ChannelContributionSummary(
        channel=name,
        mean_contribution=contribution,
        hdi_low=contribution_hdi[0],
        hdi_high=contribution_hdi[1],
        roi_mean=roi_mean,
        roi_hdi_low=roi_hdi[0],
        roi_hdi_high=roi_hdi[1],
        observed_spend_min=observed_min,
        observed_spend_max=observed_max,
        spend_invariant=spend_invariant,
    )


def _summary(*channels: ChannelContributionSummary) -> PosteriorSummary:
    return PosteriorSummary(channel_contributions=list(channels))


# ---------------------------------------------------------------------------
# Spend-invariant channels are always flagged at top priority
# ---------------------------------------------------------------------------


class TestSpendInvariantFlagging:
    def test_invariant_channel_flagged_even_when_tiny(self) -> None:
        # Tiny share, narrow CI — would normally be skipped — but spend
        # invariant overrides both conditions.
        flags = flag_channels(
            _summary(
                _channel("search", contribution=10000.0, roi_hdi=(1.95, 2.05)),
                _channel(
                    "tv", contribution=50.0, roi_hdi=(0.95, 1.05),
                    spend_invariant=True,
                ),
            )
        )
        names = [f.channel for f in flags]
        assert "tv" in names

    def test_invariant_rationale_explains_why(self) -> None:
        flags = flag_channels(
            _summary(_channel("tv", spend_invariant=True))
        )
        assert flags[0].rationale.startswith("Channel `tv` has invariant spend")
        assert "saturation cannot be estimated" in flags[0].rationale
        assert "controlled experiment" in flags[0].rationale

    def test_invariants_appear_before_non_invariants(self) -> None:
        # A high-spend, wide-HDI non-invariant channel ranks below an
        # invariant channel even though both get flagged.
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=20000.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
                _channel(
                    "tv", contribution=50.0,
                    roi_hdi=(0.95, 1.05), spend_invariant=True,
                ),
            )
        )
        # Invariant channel comes first regardless of magnitude.
        assert flags[0].channel == "tv"
        assert flags[1].channel == "search"


# ---------------------------------------------------------------------------
# Wide-HDI + influential-channel rule
# ---------------------------------------------------------------------------


class TestWidePosteriorRule:
    def test_wide_hdi_high_share_channel_is_flagged(self) -> None:
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=10000.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),  # width 3 / |2| = 1.5 > 0.5
                ),
            )
        )
        assert len(flags) == 1
        assert flags[0].channel == "search"
        assert "high posterior uncertainty" in flags[0].rationale
        assert "controlled experiment" in flags[0].rationale

    def test_narrow_hdi_channel_not_flagged(self) -> None:
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=10000.0,
                    roi_mean=2.0, roi_hdi=(1.95, 2.05),  # width 0.10 / 2 = 0.05
                ),
            )
        )
        assert flags == []

    def test_low_share_wide_hdi_channel_not_flagged_by_default(self) -> None:
        # tv has wide HDI but only ~5 % of total media impact, so it
        # falls below high_share_threshold=0.10.
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=10000.0,
                    roi_mean=2.0, roi_hdi=(1.95, 2.05),
                ),
                _channel(
                    "tv", contribution=500.0,
                    roi_mean=1.0, roi_hdi=(0.0, 2.0),
                ),
            )
        )
        names = [f.channel for f in flags]
        assert "tv" not in names

    def test_threshold_overrides_let_caller_widen_the_net(self) -> None:
        # Same setup as above but with thresholds dropped to 0.0 — both
        # the share gate and the uncertainty gate pass.
        flags = flag_channels(
            _summary(
                _channel(
                    "tv", contribution=500.0,
                    roi_mean=1.0, roi_hdi=(0.0, 2.0),
                ),
            ),
            high_share_threshold=0.0,
            uncertainty_threshold=0.0,
        )
        assert len(flags) == 1
        assert flags[0].channel == "tv"


# ---------------------------------------------------------------------------
# Sorting / prioritization
# ---------------------------------------------------------------------------


class TestPrioritization:
    def test_higher_spend_channels_rank_above_lower_spend_within_tier(self) -> None:
        # Both wide-HDI, both above the share gate, both flagged; the
        # remaining tie-break is spend (or contribution proxy).
        flags = flag_channels(
            _summary(
                _channel(
                    "small", contribution=8000.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
                _channel(
                    "big", contribution=20000.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
            )
        )
        order = [f.channel for f in flags]
        assert order == ["big", "small"]

    def test_spend_by_channel_overrides_contribution_proxy_in_ranking(self) -> None:
        # Both channels need to pass the share gate first; the share gate
        # uses contribution, so we set both contributions equal. Once
        # both are flagged, the spend mapping decides the order.
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=10000.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
                _channel(
                    "tv", contribution=10000.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
            ),
            spend_by_channel={"search": 100000.0, "tv": 5000.0},
        )
        order = [f.channel for f in flags]
        # Higher real spend wins even though contributions are equal.
        assert order == ["search", "tv"]


# ---------------------------------------------------------------------------
# Rationale text
# ---------------------------------------------------------------------------


class TestRationaleText:
    def test_rationale_includes_real_spend_when_supplied(self) -> None:
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=2000.0,
                    contribution_hdi=(1500.0, 2500.0),
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
            ),
            spend_by_channel={"search": 75432.0},
        )
        rationale = flags[0].rationale
        # Real spend appears in the rationale (with thousands separator).
        assert "75,432" in rationale
        assert "significant spend (75,432)" in rationale
        # 95 % HDI bounds appear too.
        assert "1,500 to 2,500" in rationale

    def test_rationale_falls_back_to_contribution_when_spend_absent(self) -> None:
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=2500.0,
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
            ),
        )
        rationale = flags[0].rationale
        assert "modelled contribution" in rationale
        assert "significant spend" not in rationale

    def test_invariant_rationale_does_not_quote_spend(self) -> None:
        # A spend-invariant channel gets the saturation-can't-be-learned
        # rationale, not the high-uncertainty one — even when spend is
        # large.
        flags = flag_channels(
            _summary(_channel("tv", contribution=10000.0, spend_invariant=True)),
            spend_by_channel={"tv": 50000.0},
        )
        rationale = flags[0].rationale
        assert "invariant spend" in rationale
        assert "high posterior uncertainty" not in rationale


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_returns_channelflag_records_with_required_fields(self) -> None:
        flags = flag_channels(
            _summary(
                _channel(
                    "search", contribution=5000.0,
                    contribution_hdi=(4000.0, 6000.0),
                    roi_mean=2.0, roi_hdi=(0.5, 3.5),
                ),
            ),
            spend_by_channel={"search": 12345.0},
        )
        assert len(flags) == 1
        flag = flags[0]
        assert isinstance(flag, ChannelFlag)
        assert flag.channel == "search"
        assert flag.posterior_ci == (4000.0, 6000.0)
        assert flag.spend == 12345.0
        assert isinstance(flag.rationale, str) and flag.rationale

    def test_no_channels_yields_empty_list(self) -> None:
        assert flag_channels(_summary()) == []

    def test_clean_summary_yields_no_flags(self) -> None:
        # All channels have tight HDIs and no invariant flag — nothing to
        # recommend.
        flags = flag_channels(
            _summary(
                _channel("search", roi_hdi=(1.95, 2.05)),
                _channel("tv", roi_hdi=(0.45, 0.55), roi_mean=0.5),
            )
        )
        assert flags == []
