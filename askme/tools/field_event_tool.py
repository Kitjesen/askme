"""Tool bridge from skills to the field-operation event workflow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from askme.tools.tool_registry import BaseTool

FieldServiceFactory = Callable[[dict[str, Any]], Any]


class FieldEventTriggerTool(BaseTool):
    """Create a product field event through FieldOperationsService.

    The tool is intentionally marked dangerous because it may notify external
    responders when DingTalk or webhook credentials are configured.
    """

    name = "field_event_trigger"
    description = (
        "Create an auditable robot field-operation event. Use it for robot faults, "
        "night stranger photo, illegal parking, fire or smoke, trash-bin full, "
        "crowd gathering, wayfinding help-point, and urgent patrol dispatch. "
        "It records the event and may notify responders if notification credentials are configured."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "scenario_id": {
                "type": "string",
                "description": "Field scenario id, e.g. robot_abnormal_incident, illegal_parking, fire_or_smoke.",
            },
            "location": {
                "type": "string",
                "description": "Human-readable site location.",
            },
            "fault_type": {
                "type": "string",
                "description": "Robot fault type for robot_abnormal_incident.",
            },
            "image_path": {
                "type": "string",
                "description": "Local or bridged evidence image path when available.",
            },
            "operator_id": {
                "type": "string",
                "description": "Operator id responsible for the manual trigger.",
            },
            "description": {
                "type": "string",
                "description": "Short incident description or operator note.",
            },
            "payload": {
                "type": "object",
                "description": "Additional scenario-specific payload fields.",
            },
        },
        "required": ["scenario_id"],
    }
    safety_level = "dangerous"
    agent_allowed = True
    voice_label = "记录现场事件"

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        service_factory: FieldServiceFactory | None = None,
    ) -> None:
        self._config = config or {}
        self._service_factory = service_factory

    def execute(
        self,
        *,
        scenario_id: str = "",
        location: str = "",
        fault_type: str = "",
        image_path: str = "",
        operator_id: str = "",
        description: str = "",
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        body: dict[str, Any] = dict(payload or {})
        body.update({key: value for key, value in kwargs.items() if value not in ("", None)})
        if scenario_id:
            body["scenario_id"] = scenario_id
        if location:
            body["location"] = location
        if fault_type:
            body["fault_type"] = fault_type
        if image_path:
            body["image_path"] = image_path
        if operator_id:
            body["operator_id"] = operator_id
        if description:
            body["description"] = description

        if not body.get("scenario_id"):
            return _json({"accepted": False, "status": "rejected", "reason": "scenario_id_required"})

        result = _run_sync(lambda: self._service().trigger_payload(body))
        return _json(_summarize_field_event_result(result))

    def _service(self) -> Any:
        if self._service_factory is not None:
            return self._service_factory(self._config)
        from askme.pipeline.field_operations import FieldOperationsService

        return FieldOperationsService(config=self._config)


def _run_sync(coro_factory: Callable[[], Any]) -> dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="askme-field-event-tool") as executor:
        return executor.submit(lambda: asyncio.run(coro_factory())).result(timeout=20.0)


def _summarize_field_event_result(result: dict[str, Any]) -> dict[str, Any]:
    event = result.get("event") if isinstance(result.get("event"), dict) else {}
    return {
        "accepted": bool(result.get("accepted")),
        "status": str(result.get("status") or event.get("status") or ""),
        "reason": str(result.get("reason") or ""),
        "missing_evidence": result.get("missing_evidence") or event.get("missing_evidence") or [],
        "event_id": str(event.get("event_id") or ""),
        "scenario_id": str(result.get("scenario_id") or event.get("scenario_id") or ""),
        "scenario_name": str(event.get("scenario_name") or ""),
        "priority": str(event.get("priority") or ""),
        "location": str(event.get("location") or ""),
        "notification_group": str(event.get("notification_group") or ""),
        "sent_channels": event.get("sent_channels") or [],
        "voice": str(event.get("voice") or ""),
        "operator_action": str(event.get("operator_action") or ""),
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
