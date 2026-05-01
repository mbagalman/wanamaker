"""Hand-rolled SVG chart helpers for the HTML showcase.

Hand-written rather than ``matplotlib``-driven so the byte output is fully
deterministic across platforms (no font fallbacks, no rasterisation, no
locale-dependent floats). That keeps the showcase HTML golden-file testable
and aligned with the project's deterministic-template ethos
(see ``reports/__init__.py``).

Each public helper takes plain dicts (already shaped by ``render.py``) and
returns a complete ``<svg>...</svg>`` string. The HTML template inlines the
result with ``| safe``.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

# Palette — single source of truth so the CSS and the SVG agree.
_INK = "#0f172a"          # near-black for text
_MUTED = "#64748b"        # muted grey for axes/ticks
_PRIMARY = "#2b5876"      # deep blue for bars
_PRIMARY_LIGHT = "#7c9bb3"
_HDI = "#94a3b8"          # whisker grey
_PASS = "#059669"
_MODERATE = "#d97706"
_WEAK = "#dc2626"
_GRID = "#e2e8f0"

# Fonts: stack of system fonts. No external font files.
_FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def _fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def _fmt_2(x: float) -> str:
    return f"{x:.2f}"


def _confidence_color(confidence: str) -> str:
    if confidence == "high":
        return _PASS
    if confidence == "moderate":
        return _MODERATE
    return _WEAK


def contribution_bars_svg(channels: Sequence[dict[str, Any]]) -> str:
    """Horizontal bar chart of channel contributions with 95% HDI whiskers.

    ``channels`` matches the dicts produced by ``_channel_view`` in
    ``render.py`` (already pre-ranked by mean contribution). Spend-invariant
    channels render with a hatched fill to flag that the saturation curve
    came from priors only.
    """
    if not channels:
        return _empty_chart_svg("No channel contributions to display.")

    width = 720
    row_height = 32
    top_pad = 24
    bottom_pad = 48
    left_pad = 160
    right_pad = 80
    bar_height = 18

    n = len(channels)
    height = top_pad + row_height * n + bottom_pad

    # Domain is [0, max(hdi_high, mean)] across channels, clamped to >=1.
    max_value = max(
        max(c["contribution_hdi_high"], c["contribution_mean"]) for c in channels
    )
    max_value = max(max_value, 1.0)
    plot_width = width - left_pad - right_pad

    def x(value: float) -> float:
        return left_pad + (value / max_value) * plot_width

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'aria-label="Channel contributions with 95% credible intervals" '
        f'class="wmk-chart wmk-chart--contributions">'
    )
    parts.append(_hatch_pattern_def("wmk-hatch-invariant", _PRIMARY_LIGHT))

    # X-axis baseline + ticks at 0, 25%, 50%, 75%, 100% of max.
    baseline_y = top_pad + row_height * n + 6
    parts.append(
        f'<line x1="{left_pad}" y1="{top_pad - 4}" x2="{left_pad}" '
        f'y2="{baseline_y}" stroke="{_MUTED}" stroke-width="1"/>'
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tick_x = left_pad + frac * plot_width
        tick_value = frac * max_value
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{baseline_y}" '
            f'x2="{tick_x:.1f}" y2="{baseline_y + 4}" '
            f'stroke="{_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tick_x:.1f}" y="{baseline_y + 18}" '
            f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'text-anchor="middle">{_fmt_int(tick_value)}</text>'
        )
        if frac > 0:
            parts.append(
                f'<line x1="{tick_x:.1f}" y1="{top_pad - 4}" '
                f'x2="{tick_x:.1f}" y2="{baseline_y}" '
                f'stroke="{_GRID}" stroke-width="1" stroke-dasharray="2,3"/>'
            )

    # One row per channel.
    for i, channel in enumerate(channels):
        row_y = top_pad + i * row_height
        bar_y = row_y + (row_height - bar_height) / 2
        mean_x = x(channel["contribution_mean"])
        hdi_low_x = x(max(channel["contribution_hdi_low"], 0.0))
        hdi_high_x = x(channel["contribution_hdi_high"])
        invariant = channel.get("spend_invariant", False)
        fill = (
            "url(#wmk-hatch-invariant)" if invariant else _PRIMARY
        )
        # Channel label (left).
        parts.append(
            f'<text x="{left_pad - 10}" y="{row_y + row_height / 2 + 4:.1f}" '
            f'fill="{_INK}" font-family="{_FONT_FAMILY}" font-size="12" '
            f'text-anchor="end" '
            f'font-weight="500">{escape(str(channel["name"]))}</text>'
        )
        # Bar from 0 to mean.
        bar_width = max(mean_x - left_pad, 0.0)
        parts.append(
            f'<rect x="{left_pad}" y="{bar_y:.1f}" '
            f'width="{bar_width:.1f}" height="{bar_height}" '
            f'fill="{fill}" rx="2" ry="2"/>'
        )
        # HDI whisker (low → high) at the bar's vertical centre.
        whisker_y = bar_y + bar_height / 2
        parts.append(
            f'<line x1="{hdi_low_x:.1f}" y1="{whisker_y:.1f}" '
            f'x2="{hdi_high_x:.1f}" y2="{whisker_y:.1f}" '
            f'stroke="{_HDI}" stroke-width="2"/>'
        )
        for cap_x in (hdi_low_x, hdi_high_x):
            parts.append(
                f'<line x1="{cap_x:.1f}" y1="{whisker_y - 5:.1f}" '
                f'x2="{cap_x:.1f}" y2="{whisker_y + 5:.1f}" '
                f'stroke="{_HDI}" stroke-width="2"/>'
            )
        # Value label (right).
        label_x = max(hdi_high_x, mean_x) + 8
        parts.append(
            f'<text x="{label_x:.1f}" y="{row_y + row_height / 2 + 4:.1f}" '
            f'fill="{_INK}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'text-anchor="start">'
            f'{_fmt_int(channel["contribution_mean"])}</text>'
        )

    # X-axis label.
    parts.append(
        f'<text x="{left_pad + plot_width / 2:.1f}" y="{height - 8}" '
        f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
        f'text-anchor="middle">Estimated contribution (target units)</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def roi_dotplot_svg(channels: Sequence[dict[str, Any]]) -> str:
    """Per-channel ROI mean with a 95% HDI line.

    Spend-invariant channels are rendered as a flat label only — there is
    no defensible ROI for a curve that wasn't learned from data.
    """
    if not channels:
        return _empty_chart_svg("No ROI estimates to display.")

    width = 720
    row_height = 28
    top_pad = 24
    bottom_pad = 44
    left_pad = 160
    right_pad = 80
    n = len(channels)
    height = top_pad + row_height * n + bottom_pad

    measurable = [c for c in channels if not c.get("spend_invariant", False)]
    if measurable:
        domain_low = min(c["roi_hdi_low"] for c in measurable)
        domain_high = max(c["roi_hdi_high"] for c in measurable)
    else:
        domain_low, domain_high = 0.0, 1.0
    domain_low = min(domain_low, 0.0)
    domain_high = max(domain_high, domain_low + 0.5)
    plot_width = width - left_pad - right_pad
    span = domain_high - domain_low

    def x(value: float) -> float:
        return left_pad + ((value - domain_low) / span) * plot_width

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Channel ROI with 95% credible intervals" '
        f'class="wmk-chart wmk-chart--roi">'
    )

    baseline_y = top_pad + row_height * n + 6
    # Ticks at 0, 25%, 50%, 75%, 100% of span.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tick_x = left_pad + frac * plot_width
        tick_value = domain_low + frac * span
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{baseline_y}" '
            f'x2="{tick_x:.1f}" y2="{baseline_y + 4}" '
            f'stroke="{_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tick_x:.1f}" y="{baseline_y + 18}" '
            f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'text-anchor="middle">{_fmt_2(tick_value)}</text>'
        )
        if frac > 0:
            parts.append(
                f'<line x1="{tick_x:.1f}" y1="{top_pad - 4}" '
                f'x2="{tick_x:.1f}" y2="{baseline_y}" '
                f'stroke="{_GRID}" stroke-width="1" stroke-dasharray="2,3"/>'
            )

    # Zero line if 0 is inside the domain.
    if domain_low < 0 < domain_high:
        zero_x = x(0.0)
        parts.append(
            f'<line x1="{zero_x:.1f}" y1="{top_pad - 4}" '
            f'x2="{zero_x:.1f}" y2="{baseline_y}" '
            f'stroke="{_MUTED}" stroke-width="1" stroke-dasharray="3,3"/>'
        )

    for i, channel in enumerate(channels):
        row_y = top_pad + i * row_height + row_height / 2
        confidence = channel.get("confidence", "weak")
        invariant = channel.get("spend_invariant", False)
        # Channel label.
        parts.append(
            f'<text x="{left_pad - 10}" y="{row_y + 4:.1f}" '
            f'fill="{_INK}" font-family="{_FONT_FAMILY}" font-size="12" '
            f'text-anchor="end" '
            f'font-weight="500">{escape(str(channel["name"]))}</text>'
        )
        if invariant:
            parts.append(
                f'<text x="{left_pad}" y="{row_y + 4:.1f}" '
                f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
                f'font-style="italic" text-anchor="start">'
                f'spend invariant — ROI not identifiable</text>'
            )
            continue
        color = _confidence_color(confidence)
        hdi_low_x = x(channel["roi_hdi_low"])
        hdi_high_x = x(channel["roi_hdi_high"])
        mean_x = x(channel["roi_mean"])
        # HDI line.
        parts.append(
            f'<line x1="{hdi_low_x:.1f}" y1="{row_y:.1f}" '
            f'x2="{hdi_high_x:.1f}" y2="{row_y:.1f}" '
            f'stroke="{_HDI}" stroke-width="2"/>'
        )
        for cap_x in (hdi_low_x, hdi_high_x):
            parts.append(
                f'<line x1="{cap_x:.1f}" y1="{row_y - 5:.1f}" '
                f'x2="{cap_x:.1f}" y2="{row_y + 5:.1f}" '
                f'stroke="{_HDI}" stroke-width="2"/>'
            )
        # Mean dot.
        parts.append(
            f'<circle cx="{mean_x:.1f}" cy="{row_y:.1f}" r="5" '
            f'fill="{color}" stroke="white" stroke-width="1"/>'
        )
        # Value label.
        label_x = max(hdi_high_x, mean_x) + 8
        parts.append(
            f'<text x="{label_x:.1f}" y="{row_y + 4:.1f}" '
            f'fill="{_INK}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'text-anchor="start">{_fmt_2(channel["roi_mean"])}</text>'
        )

    parts.append(
        f'<text x="{left_pad + plot_width / 2:.1f}" y="{height - 8}" '
        f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
        f'text-anchor="middle">Return per unit spend</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def scenario_delta_svg(
    periods: Sequence[str],
    mean: Sequence[float],
    hdi_low: Sequence[float],
    hdi_high: Sequence[float],
) -> str:
    """Posterior predictive line + 95% HDI ribbon over the forecast periods.

    All three series must be the same length as ``periods``.
    """
    n = len(periods)
    if n == 0 or len(mean) != n or len(hdi_low) != n or len(hdi_high) != n:
        return _empty_chart_svg("No forecast available.")

    width = 720
    height = 280
    top_pad = 24
    bottom_pad = 48
    left_pad = 80
    right_pad = 24

    plot_width = width - left_pad - right_pad
    plot_height = height - top_pad - bottom_pad

    y_min = min(min(hdi_low), min(mean))
    y_max = max(max(hdi_high), max(mean))
    if y_max == y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    def x(i: int) -> float:
        if n == 1:
            return left_pad + plot_width / 2
        return left_pad + (i / (n - 1)) * plot_width

    def y(value: float) -> float:
        return top_pad + (1.0 - (value - y_min) / (y_max - y_min)) * plot_height

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Predicted outcome over forecast periods" '
        f'class="wmk-chart wmk-chart--scenario">'
    )

    # Y-axis grid + ticks (5 lines).
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = top_pad + (1 - frac) * plot_height
        gv = y_min + frac * (y_max - y_min)
        parts.append(
            f'<line x1="{left_pad}" y1="{gy:.1f}" '
            f'x2="{left_pad + plot_width}" y2="{gy:.1f}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left_pad - 8}" y="{gy + 4:.1f}" '
            f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'text-anchor="end">{_fmt_int(gv)}</text>'
        )

    # HDI ribbon as a closed polygon.
    upper = " ".join(f"{x(i):.1f},{y(hdi_high[i]):.1f}" for i in range(n))
    lower = " ".join(
        f"{x(i):.1f},{y(hdi_low[i]):.1f}" for i in reversed(range(n))
    )
    parts.append(
        f'<polygon points="{upper} {lower}" '
        f'fill="{_PRIMARY_LIGHT}" fill-opacity="0.35" '
        f'stroke="none"/>'
    )

    # Mean line.
    line_points = " ".join(f"{x(i):.1f},{y(mean[i]):.1f}" for i in range(n))
    parts.append(
        f'<polyline points="{line_points}" '
        f'fill="none" stroke="{_PRIMARY}" stroke-width="2" '
        f'stroke-linejoin="round"/>'
    )

    # X-axis: first/last period labels and a midpoint when n > 2.
    indices = [0, n - 1]
    if n > 2:
        indices.insert(1, n // 2)
    seen = set()
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        label_x = x(i)
        parts.append(
            f'<text x="{label_x:.1f}" y="{height - bottom_pad + 18}" '
            f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'text-anchor="middle">{escape(str(periods[i]))}</text>'
        )

    # Axis frame.
    parts.append(
        f'<line x1="{left_pad}" y1="{top_pad + plot_height}" '
        f'x2="{left_pad + plot_width}" y2="{top_pad + plot_height}" '
        f'stroke="{_MUTED}" stroke-width="1"/>'
    )

    # Y-axis label.
    parts.append(
        f'<text x="{left_pad}" y="{height - 12}" '
        f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
        f'text-anchor="start">Period</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def contribution_waterfall_svg(
    baseline_total: float,
    channels: Sequence[dict[str, Any]],
) -> str:
    """Horizontal cumulative waterfall: baseline + channels = total.

    The chart answers "where does the revenue come from?" at a glance.
    Each segment is a contiguous horizontal rectangle; the baseline is
    rendered in a muted fill and channels in primary blue, with thin
    white separators so adjacent segments are visually distinct. The
    rightmost label states the total.

    ``channels`` matches the dicts produced by ``_channel_view`` in
    ``render.py`` (already pre-ranked by mean contribution). Only
    ``name`` and ``contribution_mean`` are required; spend-invariant and
    other fields are ignored — this chart shows headline magnitudes
    only, not credibility nuance (that lives in the bar chart with HDI
    whiskers).

    ``baseline_total`` may be 0 or negative; in either case the chart
    still composes (baseline is omitted or shown with a small placeholder).
    """
    channels = [c for c in channels if c.get("contribution_mean", 0.0) > 0]
    if not channels and baseline_total <= 0:
        return _empty_chart_svg("No contributions to display.")

    width = 720
    height = 160
    top_pad = 40
    left_pad = 12
    right_pad = 12
    bar_h = 36
    bar_y = top_pad + 18
    bar_top = bar_y
    bar_bottom = bar_y + bar_h

    # Total = baseline + channels. Width is scaled so total fills the plot.
    contribs = [
        (str(c["name"]), float(c["contribution_mean"])) for c in channels
    ]
    total = max(baseline_total, 0.0) + sum(value for _, value in contribs)
    plot_w = width - left_pad - right_pad
    if total <= 0:
        return _empty_chart_svg("No contributions to display.")

    def x_of(running_total: float) -> float:
        return left_pad + (running_total / total) * plot_w

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'aria-label="Contribution waterfall: baseline plus media equals total" '
        f'class="wmk-chart wmk-chart--waterfall">'
    )

    segments: list[tuple[str, float, str]] = []
    if baseline_total > 0:
        segments.append(("Baseline", float(baseline_total), _MUTED))
    for name, value in contribs:
        segments.append((name, value, _PRIMARY))

    running = 0.0
    for label, value, fill in segments:
        seg_x = x_of(running)
        seg_w = (value / total) * plot_w
        parts.append(
            f'<rect x="{seg_x:.1f}" y="{bar_top}" '
            f'width="{max(seg_w, 0.5):.1f}" height="{bar_h}" '
            f'fill="{fill}"/>'
        )
        # Top label: channel name (skip when segment is too narrow).
        if seg_w >= 36:
            parts.append(
                f'<text x="{seg_x + seg_w / 2:.1f}" y="{bar_top - 6:.1f}" '
                f'fill="{_INK}" font-family="{_FONT_FAMILY}" font-size="11" '
                f'font-weight="500" text-anchor="middle">{escape(label)}</text>'
            )
        # Bottom label: numeric value (skip when too narrow).
        if seg_w >= 48:
            parts.append(
                f'<text x="{seg_x + seg_w / 2:.1f}" y="{bar_bottom + 14:.1f}" '
                f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="10" '
                f'text-anchor="middle">{_fmt_int(value)}</text>'
            )
        # Thin white separator between segments.
        parts.append(
            f'<line x1="{seg_x + seg_w:.1f}" y1="{bar_top}" '
            f'x2="{seg_x + seg_w:.1f}" y2="{bar_bottom}" '
            f'stroke="white" stroke-width="1.5"/>'
        )
        running += value

    # Total label at the right end.
    parts.append(
        f'<text x="{left_pad + plot_w:.1f}" y="{bar_top - 22:.1f}" '
        f'fill="{_INK}" font-family="{_FONT_FAMILY}" font-size="13" '
        f'font-weight="600" text-anchor="end">'
        f'Total: {_fmt_int(total)}</text>'
    )

    # Caption below.
    parts.append(
        f'<text x="{left_pad + plot_w / 2:.1f}" y="{height - 8}" '
        f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
        f'text-anchor="middle">'
        f'Modelled target = baseline + each channel’s contribution</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def response_curves_svg(channels: Sequence[dict[str, Any]]) -> str:
    """Small-multiples grid of saturation curves, one panel per channel.

    Each panel plots the channel's mean Hill saturation curve over a spend
    domain that comfortably exceeds the historical range. A vertical band
    marks the observed spend range so the reader can see where the curve
    is supported by data and where it is extrapolation. Spend-invariant
    channels render as a single annotated panel without a curve — there
    is no defensible response shape when the data didn't move.

    Each ``channels`` dict must include:
      - ``name``, ``spend_invariant``, ``observed_spend_min``,
        ``observed_spend_max``, ``mean_contribution``
      - ``half_life`` (geometric adstock half-life)
      - ``ec50`` (Hill EC50, on the adstocked-spend scale)
      - ``slope`` (Hill slope)
      - ``coefficient`` (saturated-spend → contribution scale factor)

    When ``half_life``, ``ec50``, ``slope``, or ``coefficient`` is missing
    or non-finite, the panel falls back to the spend-invariant rendering so
    the chart still composes for any engine that doesn't expose the full
    adstock + Hill parameter set.
    """
    if not channels:
        return _empty_chart_svg("No response curves to display.")

    cols = 2
    rows = (len(channels) + cols - 1) // cols
    panel_w = 360
    panel_h = 200
    gap_x = 16
    gap_y = 24
    width = cols * panel_w + (cols - 1) * gap_x
    height = rows * panel_h + (rows - 1) * gap_y

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'aria-label="Per-channel response (saturation) curves" '
        f'class="wmk-chart wmk-chart--response-curves">'
    )

    for i, channel in enumerate(channels):
        col = i % cols
        row = i // cols
        ox = col * (panel_w + gap_x)
        oy = row * (panel_h + gap_y)
        parts.append(
            f'<g transform="translate({ox},{oy})">'
            + _response_curve_panel(channel, panel_w, panel_h)
            + "</g>"
        )

    parts.append("</svg>")
    return "".join(parts)


def _response_curve_panel(channel: dict[str, Any], w: int, h: int) -> str:
    """Render a single saturation-curve panel inside its own coordinate box."""
    name = str(channel.get("name", ""))
    invariant = bool(channel.get("spend_invariant", False))
    spend_min = float(channel.get("observed_spend_min", 0.0))
    spend_max = float(channel.get("observed_spend_max", 0.0))
    half_life = channel.get("half_life")
    ec50 = channel.get("ec50")
    slope = channel.get("slope")
    coefficient = channel.get("coefficient")

    parameters_ok = all(
        isinstance(v, int | float) and v == v and v > 0  # not NaN, positive
        for v in (half_life, ec50, slope, coefficient)
    )

    pad_top = 28
    pad_bottom = 36
    pad_left = 48
    pad_right = 16
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bottom

    parts: list[str] = []

    # Frame + title.
    parts.append(
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" '
        f'fill="white" stroke="{_GRID}" stroke-width="1" rx="4" ry="4"/>'
    )
    parts.append(
        f'<text x="{pad_left}" y="18" fill="{_INK}" '
        f'font-family="{_FONT_FAMILY}" font-size="12" font-weight="600" '
        f'text-anchor="start">{escape(name)}</text>'
    )

    if invariant or not parameters_ok:
        # No defensible curve. Render a muted placeholder.
        parts.append(
            f'<text x="{w / 2:.1f}" y="{h / 2:.1f}" '
            f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="11" '
            f'font-style="italic" text-anchor="middle">'
            f'Saturation curve not identifiable from data</text>'
        )
        return "".join(parts)

    # Domain is raw weekly spend, but the model applies Hill saturation to
    # adstocked spend. Use the steady-state weekly spend implied by EC50 so
    # the displayed curve is on the same scale as the x-axis label.
    decay = _half_life_to_decay(half_life)
    weekly_ec50 = ec50 * max(1.0 - decay, 1e-12)
    spend_domain_max = max(spend_max * 1.5, weekly_ec50 * 2.0, 1.0)
    n_samples = 64
    xs = [spend_domain_max * (k / (n_samples - 1)) for k in range(n_samples)]
    ys = [
        _hill_contribution(
            _steady_state_adstocked_spend(s, decay),
            ec50,
            slope,
            coefficient,
        )
        for s in xs
    ]
    y_max = max(ys) if ys else 1.0
    y_max = max(y_max, 1.0)

    def to_x(spend: float) -> float:
        return pad_left + (spend / spend_domain_max) * plot_w

    def to_y(value: float) -> float:
        return pad_top + (1.0 - value / y_max) * plot_h

    # Axes: zero baseline + left axis.
    baseline_y = pad_top + plot_h
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{baseline_y}" stroke="{_MUTED}" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{baseline_y}" '
        f'x2="{pad_left + plot_w}" y2="{baseline_y}" '
        f'stroke="{_MUTED}" stroke-width="1"/>'
    )

    # Observed-spend range as a soft vertical band.
    if spend_max > 0 and spend_min >= 0 and spend_max > spend_min:
        band_x1 = to_x(spend_min)
        band_x2 = to_x(spend_max)
        parts.append(
            f'<rect x="{band_x1:.1f}" y="{pad_top}" '
            f'width="{band_x2 - band_x1:.1f}" height="{plot_h}" '
            f'fill="{_PRIMARY_LIGHT}" fill-opacity="0.18"/>'
        )

    # Curve.
    point_strs = [f"{to_x(xs[k]):.1f},{to_y(ys[k]):.1f}" for k in range(n_samples)]
    parts.append(
        f'<polyline points="{" ".join(point_strs)}" '
        f'fill="none" stroke="{_PRIMARY}" stroke-width="2" '
        f'stroke-linejoin="round"/>'
    )

    # X-axis labels — start, observed-max, domain end.
    label_pts = [(0.0, "0"), (spend_max, _fmt_int(spend_max)), (spend_domain_max, "")]
    seen_x: set[int] = set()
    for spend, label in label_pts:
        if not label:
            continue
        gx = to_x(spend)
        gx_rounded = int(round(gx))
        if gx_rounded in seen_x:
            continue
        seen_x.add(gx_rounded)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{baseline_y}" '
            f'x2="{gx:.1f}" y2="{baseline_y + 4}" '
            f'stroke="{_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{baseline_y + 18}" '
            f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="10" '
            f'text-anchor="middle">{escape(label)}</text>'
        )

    # Footnote: spend axis label.
    parts.append(
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{h - 6}" '
        f'fill="{_MUTED}" font-family="{_FONT_FAMILY}" font-size="10" '
        f'text-anchor="middle">Steady weekly spend (shaded = observed range)</text>'
    )

    return "".join(parts)


def _hill_contribution(
    spend: float, ec50: float, slope: float, coefficient: float
) -> float:
    """Hill saturation × coefficient evaluated at ``spend``.

    Mirrors ``_hill_saturation_tensor`` in the PyMC engine — same shape,
    same semantics, just plain Python so the chart can render without an
    engine import.
    """
    if spend <= 0.0:
        return 0.0
    s_pow = spend ** slope
    e_pow = ec50 ** slope
    return float(coefficient) * (s_pow / (s_pow + e_pow + 1e-12))


def _half_life_to_decay(half_life: float) -> float:
    """Convert geometric adstock half-life to decay.

    Mirrors ``wanamaker.transforms.adstock.half_life_to_decay`` without
    importing transform code from report rendering.
    """
    return 0.5 ** (1.0 / half_life)


def _steady_state_adstocked_spend(weekly_spend: float, decay: float) -> float:
    """Map constant weekly spend to its geometric-adstock steady state.

    The PyMC engine uses ``A_t = X_t + decay * A_{t-1}``. For a constant
    spend level, the steady-state value is ``X / (1 - decay)``.
    """
    return weekly_spend / max(1.0 - decay, 1e-12)


def _empty_chart_svg(message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 720 80" role="img" '
        f'aria-label="{escape(message)}" class="wmk-chart wmk-chart--empty">'
        f'<text x="360" y="44" fill="{_MUTED}" '
        f'font-family="{_FONT_FAMILY}" font-size="12" '
        f'text-anchor="middle" font-style="italic">{escape(message)}</text>'
        '</svg>'
    )


def _hatch_pattern_def(pattern_id: str, color: str) -> str:
    """Diagonal hatch pattern used to mark spend-invariant channels."""
    return (
        f'<defs><pattern id="{pattern_id}" patternUnits="userSpaceOnUse" '
        f'width="6" height="6" patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{color}" fill-opacity="0.35"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{color}" '
        f'stroke-width="2"/></pattern></defs>'
    )
