"""Compatibility facade for :mod:`askme.robot_interaction.intent_router`."""

from __future__ import annotations

from askme.robot_interaction.intent_router import (
    _ESTOP_KEYWORDS,
    _QUICK_REPLIES,
    Intent,
    IntentRouter,
    IntentType,
)

__all__ = [
    "Intent",
    "IntentRouter",
    "IntentType",
    "_ESTOP_KEYWORDS",
    "_QUICK_REPLIES",
]
