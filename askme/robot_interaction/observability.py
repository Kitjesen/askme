"""Observability helpers for interaction routing decisions."""

from __future__ import annotations

from typing import Any

from askme.robot_interaction.intent_router import Intent

_TEXT_PREVIEW_LIMIT = 160


def intent_route_payload(intent: Intent, *, source: str = "") -> dict[str, Any]:
    """Return a stable, audit-ready summary of one routing decision."""
    payload: dict[str, Any] = {
        "type": intent.type.value,
        "reason": intent.reason or "",
        "source": source,
        "raw_text_preview": _preview(intent.raw_text),
    }
    if intent.skill_name:
        payload["skill_name"] = intent.skill_name
    if intent.command:
        payload["command"] = intent.command
    if intent.trigger_phrase:
        payload["trigger_phrase"] = intent.trigger_phrase
    if intent.scenario_id:
        payload["scenario_id"] = intent.scenario_id
    if intent.confidence is not None:
        payload["confidence"] = intent.confidence
    if intent.route_evidence:
        payload["route_evidence"] = intent.route_evidence
    if intent.reply_text:
        payload["reply_text_preview"] = _preview(intent.reply_text)
    return payload


def attach_intent_route_trace(
    trace: Any,
    intent: Intent,
    *,
    source: str,
    stage: str = "intent_route",
) -> dict[str, Any]:
    """Attach route metadata to a PipelineTrace-like object and return it.

    The helper accepts ``Any`` so loops can call it defensively with test fakes
    or a missing trace without coupling interaction code to the trace module.
    """
    payload = intent_route_payload(intent, source=source)
    if trace is not None and hasattr(trace, "metadata"):
        trace.metadata[stage] = payload
    return payload


def _preview(text: str | None) -> str:
    value = str(text or "")
    return value[:_TEXT_PREVIEW_LIMIT]
