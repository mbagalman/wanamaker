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
