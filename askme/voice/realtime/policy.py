"""Fail-closed routing policy for speculative realtime speech responses."""

from __future__ import annotations

from dataclasses import dataclass

from askme.voice.realtime.config import SUPPORTED_REALTIME_PROVIDERS


@dataclass(frozen=True)
class RealtimeRouteDecision:
    route: str
    allow_provider_audio: bool
    interrupt_provider: bool
    reason: str


def decide_realtime_route(
    *,
    mode: str,
    interaction_admitted: bool,
    intent_type: str,
    provider_ready: bool,
    provider: str = "",
    emergency: bool = False,
    pending_approval: bool = False,
    robot_task: bool = False,
    tool_route: bool = False,
) -> RealtimeRouteDecision:
    """Allow S2S audio only for admitted, ordinary general conversation."""

    normalized_mode = str(mode or "split").strip().lower()
    normalized_intent = str(intent_type or "").strip().lower()
    normalized_provider = str(provider or "").strip().lower()
    if normalized_mode == "split":
        return RealtimeRouteDecision("cascade", False, False, "realtime_disabled")
    if not provider_ready:
        return RealtimeRouteDecision("cascade", False, True, "provider_unavailable")
    if emergency:
        return RealtimeRouteDecision("cascade", False, True, "emergency")
    if not interaction_admitted:
        return RealtimeRouteDecision(
            "cascade", False, True, "interaction_not_admitted"
        )
    if pending_approval:
        return RealtimeRouteDecision("cascade", False, True, "pending_approval")
    if normalized_provider not in SUPPORTED_REALTIME_PROVIDERS:
        return RealtimeRouteDecision("cascade", False, True, "unsupported_provider")
    if normalized_mode == "shadow":
        return RealtimeRouteDecision("shadow", False, False, "shadow_observation")
    if robot_task or tool_route or normalized_intent != "general":
        return RealtimeRouteDecision("cascade", False, True, "robot_or_tool_route")
    if normalized_mode != "general_chat":
        return RealtimeRouteDecision("cascade", False, True, "unsupported_mode")
    return RealtimeRouteDecision(
        normalized_provider,
        True,
        False,
        "admitted_general_chat",
    )
