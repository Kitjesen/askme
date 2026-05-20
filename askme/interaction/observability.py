"""Compatibility facade for :mod:`askme.robot_interaction.observability`."""

from __future__ import annotations

from askme.robot_interaction.observability import (
    attach_intent_route_trace,
    intent_route_payload,
)

__all__ = ["attach_intent_route_trace", "intent_route_payload"]
