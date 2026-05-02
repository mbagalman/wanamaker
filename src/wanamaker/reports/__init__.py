"""Report rendering.

All user-facing report text — executive summary, trust card, refresh diff
narrative — is generated from deterministic Jinja2 templates driven by
posterior statistics.

Per AGENTS.md Hard Rule 2: **no LLM calls for output generation**, ever.
If a template feels limiting, expand the template logic. Do not reach for
an LLM.

Templates ship inside this subpackage so the package is self-contained.
"""

from wanamaker.reports.calibration_comparison import (
    CalibrationComparison,
    CalibrationComparisonError,
    CalibrationModeError,
    ChannelComparison,
    ChannelSetMismatchError,
    DataHashMismatchError,
    build_calibration_comparison_context,
    compare_calibration,
    render_calibration_comparison,
)
from wanamaker.reports.excel import (
    WorkbookMetadata,
    build_excel_workbook,
    write_excel_workbook,
)
from wanamaker.reports.render import (
    build_executive_summary_context,
    build_ramp_recommendation_context,
    build_trust_card_context,
    render_executive_summary,
    render_ramp_recommendation,
    render_trust_card,
)
from wanamaker.reports.showcase import build_showcase_context, render_showcase
from wanamaker.reports.trust_card_one_pager import (
    build_trust_card_one_pager_context,
    render_trust_card_one_pager,
)

__all__ = [
    "CalibrationComparison",
    "CalibrationComparisonError",
    "CalibrationModeError",
    "ChannelComparison",
    "ChannelSetMismatchError",
    "DataHashMismatchError",
    "WorkbookMetadata",
    "build_calibration_comparison_context",
    "build_excel_workbook",
    "build_executive_summary_context",
    "build_ramp_recommendation_context",
    "build_showcase_context",
    "build_trust_card_context",
    "build_trust_card_one_pager_context",
    "compare_calibration",
    "render_calibration_comparison",
    "render_executive_summary",
    "render_ramp_recommendation",
    "render_showcase",
    "render_trust_card",
    "render_trust_card_one_pager",
    "write_excel_workbook",
]
