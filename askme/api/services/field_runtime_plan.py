"""Build runtime handoff plans from accepted Field events."""

from __future__ import annotations

import re
from typing import Any


def build_field_runtime_plan_from_event(
    event: dict[str, Any],
    *,
    operator_id: str,
) -> dict[str, Any]:
    """Build a runtime-handoff plan from an accepted field incident."""

    event_id = str(event.get("event_id") or "")
    scenario_id = str(event.get("scenario_id") or "field_event")
    playbook = event.get("playbook") if isinstance(event.get("playbook"), dict) else {}
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    policy = str(playbook.get("robot_motion_policy") or "observe_then_continue")
    location = str(
        event.get("location")
        or payload.get("location")
        or payload.get("target_location")
        or "-"
    )
    area_id = field_runtime_area_id(event, payload)
    task_type = field_runtime_task_type(scenario_id, policy)
    risk_tier = field_runtime_risk_tier(event)
    goal = (
        f"Handle field event {scenario_id} at {location}. "
        f"Apply robot policy {policy} and keep operator in control."
    )
    return {
        "plan_id": f"field-{event_id or scenario_id}",
        "planning_session_id": f"field-session-{event_id or scenario_id}",
        "intent": task_type,
        "goal": goal,
        "handoff_ready": True,
        "operator_id": operator_id,
        "operator_roles": ["operator"],
        "safety_constraints": [
            "Do not bypass field safety policy.",
            "Do not execute low-level motor commands from LLM output.",
            "Keep hardware dispatch disabled unless the runtime profile explicitly enables it.",
        ],
        "missing_inputs": [],
        "reference": {
            "resolved": {
                "area_id": area_id,
                "label": location,
                "field_event_id": event_id,
                "scenario_id": scenario_id,
            }
        },
        "mission": {
            "mission": {
                "mission_type": task_type,
                "goal": goal,
                "risk_tier": risk_tier,
                "operator_id": operator_id,
                "operator_roles": ["operator"],
                "steps": [{"target": area_id, "policy": policy}],
                "safety_notes": [
                    f"field_event_id={event_id}",
                    f"robot_motion_policy={policy}",
                    f"priority={event.get('priority') or ''}",
                    "field event runtime handoff is high-level only",
                ],
                "field_event": {
                    "event_id": event_id,
                    "scenario_id": scenario_id,
                    "priority": event.get("priority"),
                    "severity": event.get("severity"),
                    "location": location,
                    "notification_group": event.get("notification_group"),
                    "robot_motion_policy": policy,
                },
            }
        },
    }


def field_runtime_area_id(event: dict[str, Any], payload: dict[str, Any]) -> str:
    """Resolve a stable high-level target area id for runtime handoff."""

    for value in (
        payload.get("zone_id"),
        payload.get("map_zone_id"),
        payload.get("help_point_id"),
        event.get("location"),
        payload.get("location"),
        payload.get("target_location"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith(("area-", "zone-", "checkpoint-", "route-")):
            return lowered
        slug = re.sub(r"[^a-z0-9_-]+", "-", lowered).strip("-")
        if slug:
            return f"zone-{slug[:48]}"
    return "zone-field-event"


def field_runtime_task_type(scenario_id: str, policy: str) -> str:
    """Map a field scenario/policy pair to a high-level runtime task type."""

    normalized = f"{scenario_id} {policy}".strip()
    if normalized:
        return "field_incident_response"
    return "status_report"


def field_runtime_risk_tier(event: dict[str, Any]) -> str:
    """Map field event severity into runtime risk tier."""

    priority = str(event.get("priority") or "").upper()
    severity = str(event.get("severity") or "").lower()
    if priority == "P0" or severity == "error":
        return "high"
    if priority in {"P1", "P2"}:
        return "medium"
    return "low"


def field_runtime_delivery_status(
    runtime_result: dict[str, Any],
    run: dict[str, Any],
) -> str:
    """Resolve the Field event delivery status from a runtime submission result."""

    if runtime_result.get("accepted") is False:
        return "rejected"
    state = str(run.get("current_state") or runtime_result.get("state") or "").strip()
    if state:
        return state
    return "submitted"


__all__ = [
    "build_field_runtime_plan_from_event",
    "field_runtime_area_id",
    "field_runtime_delivery_status",
    "field_runtime_risk_tier",
    "field_runtime_task_type",
]
