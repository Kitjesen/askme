"""Tool bridge from skills to park-space cognition services."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from askme.config import get_config
from askme.tools.tool_registry import BaseTool

SpaceServiceFactory = Callable[[dict[str, Any]], Any]


class SpaceLookupPlaceTool(BaseTool):
    """Resolve visitor language into a configured park point."""

    name = "space_lookup_place"
    description = (
        "Resolve a visitor's destination query against the park semantic map. "
        "Use it for questions like restroom, coffee shop, gate, parking area, "
        "service point, or named tenant lookup. It returns a confirmation prompt "
        "and refuses unknown destinations instead of inventing routes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Visitor destination query, e.g. 咖啡店在哪 or 最近的厕所。",
            },
            "current_point_id": {
                "type": "string",
                "description": "Current service point or map point id, if known.",
            },
        },
        "required": ["query"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "查询园区点位"

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        service_factory: SpaceServiceFactory | None = None,
    ) -> None:
        self._config = config
        self._service_factory = service_factory

    def execute(
        self,
        *,
        query: str = "",
        current_point_id: str = "",
        **kwargs: Any,
    ) -> str:
        payload = {
            "query": query,
            "current_point_id": current_point_id,
            **{key: value for key, value in kwargs.items() if value not in ("", None)},
        }
        result = self._service().resolve_destination_payload(payload)
        return _json(_summarize_lookup(result))

    def _service(self) -> Any:
        config = self._config if self._config is not None else get_config()
        if self._service_factory is not None:
            return self._service_factory(config)
        from askme.space import ParkSpaceService

        return ParkSpaceService.from_config(config)


class SpaceRecommendRouteTool(BaseTool):
    """Generate voice guidance or an escort-ready route recommendation."""

    name = "space_recommend_route"
    description = (
        "Recommend a route from the current park point to a destination using the "
        "park semantic map. It returns speech text and, when the route is robot "
        "passable and escort mode is requested, an escort handoff payload. It does "
        "not directly move the robot."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Destination query or place name.",
            },
            "current_point_id": {
                "type": "string",
                "description": "Current service point or map point id.",
            },
            "service_point_id": {
                "type": "string",
                "description": "Wayfinding service point id, if the interaction started at one.",
            },
            "guide_mode": {
                "type": "string",
                "description": "voice or escort. Escort only returns a handoff payload when the route is passable.",
                "enum": ["voice", "escort"],
            },
        },
        "required": ["query"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "推荐园区路线"

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        service_factory: SpaceServiceFactory | None = None,
    ) -> None:
        self._config = config
        self._service_factory = service_factory

    def execute(
        self,
        *,
        query: str = "",
        current_point_id: str = "",
        service_point_id: str = "",
        guide_mode: str = "",
        **kwargs: Any,
    ) -> str:
        payload = {
            "query": query,
            "current_point_id": current_point_id,
            "service_point_id": service_point_id,
            "guide_mode": guide_mode,
            **{key: value for key, value in kwargs.items() if value not in ("", None)},
        }
        result = self._service().guide_payload(payload)
        return _json(_summarize_route(result))

    def _service(self) -> Any:
        config = self._config if self._config is not None else get_config()
        if self._service_factory is not None:
            return self._service_factory(config)
        from askme.space import ParkSpaceService

        return ParkSpaceService.from_config(config)


def _summarize_lookup(result: dict[str, Any]) -> dict[str, Any]:
    point = result.get("point") if isinstance(result.get("point"), dict) else {}
    return {
        "resolved": bool(result.get("resolved")),
        "reason": str(result.get("reason") or ""),
        "query": str(result.get("query") or ""),
        "confidence": result.get("confidence"),
        "match_reason": str(result.get("match_reason") or ""),
        "point_id": str(point.get("point_id") or ""),
        "point_name": str(point.get("point_name") or ""),
        "point_type": str(point.get("point_type") or ""),
        "building": str(point.get("building") or ""),
        "floor": str(point.get("floor") or ""),
        "guide_mode": str(point.get("guide_mode") or ""),
        "confirmation_prompt": str(result.get("confirmation_prompt") or ""),
        "requires_confirmation": bool(result.get("requires_confirmation")),
        "requires_operator_update": bool(result.get("requires_operator_update")),
        "reply": str(result.get("reply") or ""),
    }


def _summarize_route(result: dict[str, Any]) -> dict[str, Any]:
    point = result.get("point") if isinstance(result.get("point"), dict) else {}
    route = result.get("route") if isinstance(result.get("route"), dict) else {}
    handoff = (
        result.get("field_event_payload")
        if isinstance(result.get("field_event_payload"), dict)
        else {}
    )
    return {
        "guide_ready": bool(result.get("guide_ready")),
        "resolved": bool(result.get("resolved", result.get("guide_ready"))),
        "reason": str(result.get("reason") or ""),
        "mode": str(result.get("mode") or ""),
        "point_id": str(point.get("point_id") or ""),
        "point_name": str(point.get("point_name") or ""),
        "route_id": str(route.get("route_id") or ""),
        "distance_m": route.get("distance_m"),
        "robot_passable": bool(route.get("robot_passable")),
        "speech_text": str(result.get("speech_text") or result.get("reply") or ""),
        "confirmation_prompt": str(result.get("confirmation_prompt") or ""),
        "requires_confirmation": bool(result.get("requires_confirmation")),
        "escort_handoff": handoff,
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
