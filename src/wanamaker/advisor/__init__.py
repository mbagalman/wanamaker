"""Experiment Advisor — minimal v1 (FR-5.5).

v1 identifies channels that would most benefit from experimental validation:
wide posterior CI + significant spend = a controlled experiment would
substantially improve confidence.

v1.1 will additionally recommend experiment designs (geo holdout vs.
budget split, sample size, duration, geo selection) — deferred per
BRD/PRD §4.2.
"""

from wanamaker.advisor.channel_flagging import ChannelFlag, flag_channels

__all__ = ["ChannelFlag", "flag_channels"]
