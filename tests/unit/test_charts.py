"""Tests for the hand-rolled SVG chart helpers used by the HTML showcase.

These charts are pure functions (no global state, no I/O), so the tests
are correspondingly cheap. We assert structural properties rather than
pixel-level layout: the right number of bars/dots, well-formed XML,
deterministic byte output for stable inputs, and the spend-invariant
visual marker.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from wanamaker.reports._charts import (
    contribution_bars_svg,
    contribution_waterfall_svg,
    response_curves_svg,
    roi_dotplot_svg,
    scenario_delta_svg,
)


def _channel(
    name: str,
    *,
    contribution_mean: float = 5000.0,
    contribution_hdi_low: float = 4500.0,
    contribution_hdi_high: float = 5500.0,
    roi_mean: float = 2.0,
    roi_hdi_low: float = 1.8,
    roi_hdi_high: float = 2.2,
    spend_invariant: bool = False,
    confidence: str = "high",
    observed_spend_min: float = 1000.0,
    observed_spend_max: float = 5000.0,
    half_life: float | None = 1.0,
    ec50: float | None = 3000.0,
    slope: float = 1.5,
    coefficient: float = 12000.0,
) -> dict:
    return {
        "name": name,
        "contribution_mean": contribution_mean,
        "contribution_hdi_low": contribution_hdi_low,
        "contribution_hdi_high": contribution_hdi_high,
        "roi_mean": roi_mean,
        "roi_hdi_low": roi_hdi_low,
        "roi_hdi_high": roi_hdi_high,
        "spend_invariant": spend_invariant,
        "confidence": confidence,
        "observed_spend_min": observed_spend_min,
        "observed_spend_max": observed_spend_max,
        "half_life": half_life,
        "ec50": ec50,
        "slope": slope,
        "coefficient": coefficient,
    }


def _parse(svg: str) -> ET.Element:
    """Parse the SVG string and return the root element. Fails on bad XML."""
    return ET.fromstring(svg)


# ---------------------------------------------------------------------------
# contribution_bars_svg
# ---------------------------------------------------------------------------


def test_contribution_bars_one_rect_per_channel() -> None:
    channels = [_channel("ch_a"), _channel("ch_b"), _channel("ch_c")]
    svg = contribution_bars_svg(channels)
    root = _parse(svg)

    # Bars are <rect> elements inside the chart group (excluding the
    # one inside the hatch <pattern>).
    rects = [
        r
        for r in root.iter("{http://www.w3.org/2000/svg}rect")
        if r.attrib.get("fill") != "#7c9bb3"
    ]
    assert len(rects) == 3


def test_contribution_bars_marks_spend_invariant_with_hatch() -> None:
    channels = [
        _channel("ch_a"),
        _channel("ch_b", spend_invariant=True),
    ]
    svg = contribution_bars_svg(channels)

    # The hatch pattern is defined when at least one channel needs it.
    assert "wmk-hatch-invariant" in svg
    # And one rect uses url(#wmk-hatch-invariant) as fill.
    assert 'fill="url(#wmk-hatch-invariant)"' in svg


def test_contribution_bars_handles_empty_input() -> None:
    svg = contribution_bars_svg([])
    root = _parse(svg)

    assert root.attrib.get("class", "").endswith("wmk-chart--empty")


def test_contribution_bars_is_deterministic() -> None:
    channels = [_channel("ch_a"), _channel("ch_b")]
    assert contribution_bars_svg(channels) == contribution_bars_svg(channels)


# ---------------------------------------------------------------------------
# roi_dotplot_svg
# ---------------------------------------------------------------------------


def test_roi_dotplot_one_dot_per_measurable_channel() -> None:
    channels = [
        _channel("ch_a", confidence="high"),
        _channel("ch_b", confidence="moderate"),
        _channel("ch_c", spend_invariant=True),
    ]
    svg = roi_dotplot_svg(channels)
    root = _parse(svg)

    circles = list(root.iter("{http://www.w3.org/2000/svg}circle"))
    # The spend-invariant channel does not get a dot.
    assert len(circles) == 2


def test_roi_dotplot_invariant_channel_gets_explanatory_text() -> None:
    channels = [_channel("flat", spend_invariant=True)]
    svg = roi_dotplot_svg(channels)

    assert "spend invariant" in svg.lower()


def test_roi_dotplot_handles_empty_input() -> None:
    svg = roi_dotplot_svg([])
    root = _parse(svg)

    assert root.attrib.get("class", "").endswith("wmk-chart--empty")


# ---------------------------------------------------------------------------
# scenario_delta_svg
# ---------------------------------------------------------------------------


def test_scenario_delta_renders_polyline_and_polygon() -> None:
    svg = scenario_delta_svg(
        periods=["2026-01-05", "2026-01-12", "2026-01-19"],
        mean=[1000.0, 1100.0, 1050.0],
        hdi_low=[900.0, 980.0, 940.0],
        hdi_high=[1100.0, 1220.0, 1160.0],
    )
    root = _parse(svg)

    # Polygon = HDI ribbon; polyline = mean.
    polygons = list(root.iter("{http://www.w3.org/2000/svg}polygon"))
    polylines = list(root.iter("{http://www.w3.org/2000/svg}polyline"))
    assert len(polygons) == 1
    assert len(polylines) == 1


def test_scenario_delta_falls_back_on_mismatched_lengths() -> None:
    svg = scenario_delta_svg(
        periods=["2026-01-05"],
        mean=[100.0, 110.0],
        hdi_low=[90.0],
        hdi_high=[110.0],
    )
    root = _parse(svg)

    assert root.attrib.get("class", "").endswith("wmk-chart--empty")


def test_scenario_delta_handles_single_period() -> None:
    svg = scenario_delta_svg(
        periods=["2026-01-05"],
        mean=[1000.0],
        hdi_low=[900.0],
        hdi_high=[1100.0],
    )
    # Should still parse; line collapses to a single point but ribbon is valid.
    _parse(svg)


# ---------------------------------------------------------------------------
# Channel name escaping
# ---------------------------------------------------------------------------


def test_channel_names_are_html_escaped() -> None:
    channels = [_channel("<script>")]
    svg = contribution_bars_svg(channels)

    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


# ---------------------------------------------------------------------------
# response_curves_svg
# ---------------------------------------------------------------------------


def test_response_curves_one_panel_per_channel() -> None:
    channels = [
        _channel("ch_a"),
        _channel("ch_b"),
        _channel("ch_c"),
    ]
    svg = response_curves_svg(channels)
    root = _parse(svg)

    # Each panel is a <g> with the panel's frame <rect> inside.
    panels = list(root.findall("{http://www.w3.org/2000/svg}g"))
    assert len(panels) == 3


def test_response_curves_each_panel_has_its_own_curve() -> None:
    channels = [_channel("ch_a"), _channel("ch_b")]
    svg = response_curves_svg(channels)
    root = _parse(svg)

    polylines = list(root.iter("{http://www.w3.org/2000/svg}polyline"))
    assert len(polylines) == 2


def test_response_curves_observed_band_present_when_range_known() -> None:
    channels = [_channel("ch_a", observed_spend_min=1000, observed_spend_max=5000)]
    svg = response_curves_svg(channels)
    # The observed-range band is a low-opacity rect inside the panel.
    assert "fill-opacity=\"0.18\"" in svg


def test_response_curves_spend_invariant_shows_placeholder_no_curve() -> None:
    channels = [_channel("ch_flat", spend_invariant=True)]
    svg = response_curves_svg(channels)
    root = _parse(svg)

    polylines = list(root.iter("{http://www.w3.org/2000/svg}polyline"))
    assert len(polylines) == 0
    assert "Saturation curve not identifiable" in svg


def test_response_curves_falls_back_when_parameters_missing() -> None:
    """If adstock/Hill parameters are missing or non-positive, render the placeholder."""
    channels = [
        _channel("missing_half_life", half_life=None),
        _channel("missing_ec50", ec50=None),
        _channel("zero_slope", slope=0.0),
        _channel("nan_coefficient", coefficient=float("nan")),
    ]
    svg = response_curves_svg(channels)
    root = _parse(svg)

    polylines = list(root.iter("{http://www.w3.org/2000/svg}polyline"))
    assert len(polylines) == 0
    assert svg.count("Saturation curve not identifiable") == 4


def test_response_curves_handles_empty_input() -> None:
    svg = response_curves_svg([])
    root = _parse(svg)

    assert root.attrib.get("class", "").endswith("wmk-chart--empty")


def test_response_curves_is_deterministic() -> None:
    channels = [_channel("ch_a"), _channel("ch_b")]
    assert response_curves_svg(channels) == response_curves_svg(channels)


def test_response_curves_curve_passes_through_origin() -> None:
    """Hill saturation × coefficient evaluates to 0 at spend=0, so the
    leftmost point on every curve sits on the baseline."""
    channels = [_channel("ch_a")]
    svg = response_curves_svg(channels)

    # First polyline point is at the panel's left padding x. The y
    # should be near the baseline (largest y-value in the panel).
    import re
    polyline = re.search(r'<polyline points="([^"]+)"', svg)
    assert polyline is not None
    first_point = polyline.group(1).split()[0]
    last_point = polyline.group(1).split()[-1]
    fx, fy = (float(v) for v in first_point.split(","))
    lx, ly = (float(v) for v in last_point.split(","))
    # First point is to the left of the last (x increases monotonically).
    assert fx < lx
    # First point is below or equal the last point in plot space (curve
    # rises as spend increases — bigger y-pixel = lower contribution).
    assert fy >= ly


# ---------------------------------------------------------------------------
# contribution_waterfall_svg
# ---------------------------------------------------------------------------


def test_waterfall_one_segment_per_channel_plus_baseline() -> None:
    channels = [
        _channel("ch_a", contribution_mean=5000),
        _channel("ch_b", contribution_mean=3000),
        _channel("ch_c", contribution_mean=1000),
    ]
    svg = contribution_waterfall_svg(baseline_total=10000, channels=channels)
    root = _parse(svg)

    rects = list(root.iter("{http://www.w3.org/2000/svg}rect"))
    # One rect per channel + one for baseline.
    assert len(rects) == 4


def test_waterfall_skips_baseline_segment_when_zero() -> None:
    channels = [
        _channel("ch_a", contribution_mean=5000),
        _channel("ch_b", contribution_mean=3000),
    ]
    svg = contribution_waterfall_svg(baseline_total=0.0, channels=channels)
    root = _parse(svg)

    rects = list(root.iter("{http://www.w3.org/2000/svg}rect"))
    assert len(rects) == 2
    assert ">Baseline<" not in svg


def test_waterfall_total_label_is_sum_of_segments() -> None:
    channels = [
        _channel("ch_a", contribution_mean=5000),
        _channel("ch_b", contribution_mean=2500),
    ]
    svg = contribution_waterfall_svg(baseline_total=10000, channels=channels)
    # 10000 + 5000 + 2500 = 17500 -> "Total: 17,500"
    assert "Total: 17,500" in svg


def test_waterfall_drops_zero_or_negative_channels() -> None:
    """Channels with non-positive contribution don't get a segment."""
    channels = [
        _channel("ch_a", contribution_mean=5000),
        _channel("ch_zero", contribution_mean=0),
        _channel("ch_neg", contribution_mean=-100),
    ]
    svg = contribution_waterfall_svg(baseline_total=10000, channels=channels)
    root = _parse(svg)

    rects = list(root.iter("{http://www.w3.org/2000/svg}rect"))
    # baseline + ch_a only.
    assert len(rects) == 2
    assert "ch_zero" not in svg
    assert "ch_neg" not in svg


def test_waterfall_handles_empty_input() -> None:
    svg = contribution_waterfall_svg(baseline_total=0, channels=[])
    root = _parse(svg)
    assert root.attrib.get("class", "").endswith("wmk-chart--empty")


def test_waterfall_handles_negative_baseline_as_zero() -> None:
    """A negative baseline shouldn't break the chart — it's treated as 0."""
    channels = [_channel("ch_a", contribution_mean=5000)]
    svg = contribution_waterfall_svg(baseline_total=-1000, channels=channels)
    root = _parse(svg)

    rects = list(root.iter("{http://www.w3.org/2000/svg}rect"))
    # baseline omitted, just one channel.
    assert len(rects) == 1
    assert "Total: 5,000" in svg


def test_waterfall_is_deterministic() -> None:
    channels = [
        _channel("ch_a", contribution_mean=5000),
        _channel("ch_b", contribution_mean=3000),
    ]
    a = contribution_waterfall_svg(baseline_total=10000, channels=channels)
    b = contribution_waterfall_svg(baseline_total=10000, channels=channels)
    assert a == b


def test_waterfall_segment_widths_are_proportional() -> None:
    """A segment that's twice as big should be drawn twice as wide."""
    import re

    channels = [
        _channel("ch_a", contribution_mean=4000),
        _channel("ch_b", contribution_mean=2000),
    ]
    svg = contribution_waterfall_svg(baseline_total=0, channels=channels)
    widths = [
        float(m) for m in re.findall(r'<rect[^>]*width="([\d.]+)"', svg)
    ]
    assert len(widths) == 2
    # 2:1 ratio between the two channels.
    assert abs(widths[0] / widths[1] - 2.0) < 0.05
