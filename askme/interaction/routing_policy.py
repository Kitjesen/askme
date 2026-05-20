"""Compatibility facade for :mod:`askme.robot_interaction.routing_policy`."""

from __future__ import annotations

from askme.robot_interaction.routing_policy import (
    DEFAULT_BUILTIN_COMMANDS,
    DEFAULT_ESTOP_KEYWORDS,
    DEFAULT_NEGATION_PREFIXES,
    DEFAULT_QUESTION_SAFE_SKILLS,
    DEFAULT_QUESTION_SUFFIXES,
    DEFAULT_QUICK_REPLIES,
    DEFAULT_ROUTING_POLICY,
    RoutingPolicy,
)

__all__ = [
    "DEFAULT_BUILTIN_COMMANDS",
    "DEFAULT_ESTOP_KEYWORDS",
    "DEFAULT_NEGATION_PREFIXES",
    "DEFAULT_QUESTION_SAFE_SKILLS",
    "DEFAULT_QUESTION_SUFFIXES",
    "DEFAULT_QUICK_REPLIES",
    "DEFAULT_ROUTING_POLICY",
    "RoutingPolicy",
]
