"""Park semantic-map and visitor wayfinding services.

This module is intentionally above the robot control layer. It resolves
visitor language into park points and high-level guide decisions; it never
emits low-level motion commands.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "restroom": ("restroom", "toilet", "wc", "bathroom", "卫生间", "厕所", "洗手间"),
    "parking": ("parking", "park", "停车", "停车场", "停车区", "车位"),
    "restaurant": ("restaurant", "food", "dining", "吃饭", "餐厅", "饭", "小吃", "咖啡"),
    "exit": ("exit", "gate", "入口", "出口", "大门", "门"),
    "service": ("service", "help", "问询", "服务台", "服务点"),
}


def _slug(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[\s,，。.!！?？:：;；、\-_/\\]+", "", normalized)
    return normalized


def _list_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,，;；|、]", value) if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass(frozen=True)
class ParkPoint:
    point_id: str
    park_id: str = "default"
    map_id: str = "default"
    point_name: str = ""
    point_type: str = "place"
    aliases: tuple[str, ...] = ()
    building: str = ""
    floor: str = ""
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None
    guide_mode: str = "voice"
    accessibility: str = "unknown"
    opening_hours: str = ""
    remark: str = ""
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ParkPoint":
        return cls(
            point_id=str(raw.get("point_id") or raw.get("id") or "").strip(),
            park_id=str(raw.get("park_id") or "default").strip(),
            map_id=str(raw.get("map_id") or "default").strip(),
            point_name=str(raw.get("point_name") or raw.get("name") or "").strip(),
            point_type=str(raw.get("point_type") or raw.get("type") or "place").strip(),
            aliases=tuple(_list_texts(raw.get("aliases"))),
            building=str(raw.get("building") or "").strip(),
            floor=str(raw.get("floor") or "").strip(),
            x=_optional_float(raw.get("x")),
            y=_optional_float(raw.get("y")),
            z=_optional_float(raw.get("z")),
            yaw=_optional_float(raw.get("yaw")),
            guide_mode=str(raw.get("guide_mode") or "voice").strip().lower(),
            accessibility=str(raw.get("accessibility") or "unknown").strip(),
            opening_hours=str(raw.get("opening_hours") or "").strip(),
            remark=str(raw.get("remark") or "").strip(),
            enabled=bool(raw.get("enabled", True)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "park_id": self.park_id,
            "map_id": self.map_id,
            "point_name": self.point_name,
            "point_type": self.point_type,
            "aliases": list(self.aliases),
            "building": self.building,
            "floor": self.floor,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "guide_mode": self.guide_mode,
            "accessibility": self.accessibility,
            "opening_hours": self.opening_hours,
            "remark": self.remark,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ServicePoint:
    service_point_id: str
    park_id: str = "default"
    map_id: str = "default"
    point_id: str = ""
    service_point_name: str = ""
    trigger_region_polygon: list[list[float]] = field(default_factory=list)
    dwell_seconds: float = 3.0
    idle_prompt: str = ""
    greeting_prompt: str = ""
    supported_intents: tuple[str, ...] = ()
    enabled: bool = True
    remark: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ServicePoint":
        polygon = raw.get("trigger_region_polygon")
        if not isinstance(polygon, list):
            polygon = []
        return cls(
            service_point_id=str(raw.get("service_point_id") or raw.get("id") or "").strip(),
            park_id=str(raw.get("park_id") or "default").strip(),
            map_id=str(raw.get("map_id") or "default").strip(),
            point_id=str(raw.get("point_id") or raw.get("help_point_id") or "").strip(),
            service_point_name=str(raw.get("service_point_name") or raw.get("name") or "").strip(),
            trigger_region_polygon=polygon,
            dwell_seconds=float(raw.get("dwell_seconds") or raw.get("dwell_s") or 3.0),
            idle_prompt=str(raw.get("idle_prompt") or "").strip(),
            greeting_prompt=str(raw.get("greeting_prompt") or "").strip(),
            supported_intents=tuple(_list_texts(raw.get("supported_intents"))),
            enabled=bool(raw.get("enabled", True)),
            remark=str(raw.get("remark") or "").strip(),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "service_point_id": self.service_point_id,
            "park_id": self.park_id,
            "map_id": self.map_id,
            "point_id": self.point_id,
            "service_point_name": self.service_point_name,
            "trigger_region_polygon": self.trigger_region_polygon,
            "dwell_seconds": self.dwell_seconds,
            "idle_prompt": self.idle_prompt,
            "greeting_prompt": self.greeting_prompt,
            "supported_intents": list(self.supported_intents),
            "enabled": self.enabled,
            "remark": self.remark,
        }


@dataclass(frozen=True)
class GuideRoute:
    route_id: str
    from_point_id: str
    to_point_id: str
    instructions: str
    guide_mode: str = "voice"
    distance_m: float | None = None
    robot_passable: bool = False
    risk_notes: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GuideRoute":
        return cls(
            route_id=str(raw.get("route_id") or raw.get("id") or "").strip(),
            from_point_id=str(raw.get("from_point_id") or "").strip(),
            to_point_id=str(raw.get("to_point_id") or "").strip(),
            instructions=str(raw.get("instructions") or raw.get("voice_instructions") or "").strip(),
            guide_mode=str(raw.get("guide_mode") or "voice").strip().lower(),
            distance_m=_optional_float(raw.get("distance_m")),
            robot_passable=bool(raw.get("robot_passable", False)),
            risk_notes=tuple(_list_texts(raw.get("risk_notes"))),
            enabled=bool(raw.get("enabled", True)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "from_point_id": self.from_point_id,
            "to_point_id": self.to_point_id,
            "instructions": self.instructions,
            "guide_mode": self.guide_mode,
            "distance_m": self.distance_m,
            "robot_passable": self.robot_passable,
            "risk_notes": list(self.risk_notes),
            "enabled": self.enabled,
        }


class ParkSpaceService:
    """Product-facing service for visitor destination resolution and guidance."""

    def __init__(
        self,
        *,
        park_id: str = "default",
        points: list[ParkPoint] | None = None,
        service_points: list[ServicePoint] | None = None,
        routes: list[GuideRoute] | None = None,
    ) -> None:
        self.park_id = park_id
        self._points = [point for point in (points or []) if point.point_id]
        self._service_points = [point for point in (service_points or []) if point.service_point_id]
        self._routes = [route for route in (routes or []) if route.route_id]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ParkSpaceService":
        raw = config.get("space_cognition") if isinstance(config.get("space_cognition"), dict) else {}
        park_id = str(raw.get("park_id") or "default")
        points = [ParkPoint.from_dict(item) for item in _dict_list(raw.get("points"))]
        service_points = [ServicePoint.from_dict(item) for item in _dict_list(raw.get("service_points"))]
        routes = [GuideRoute.from_dict(item) for item in _dict_list(raw.get("routes"))]

        if not points:
            points.extend(_points_from_site_map(config, park_id=park_id))
        if not service_points:
            service_points.extend(_service_points_from_site_map(config, park_id=park_id))
        return cls(park_id=park_id, points=points, service_points=service_points, routes=routes)

    def health_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = body
        return {
            "enabled": bool(self._points),
            "park_id": self.park_id,
            "points": len(self._points),
            "service_points": len(self._service_points),
            "routes": len(self._routes),
            "capabilities": [
                "destination_resolve",
                "alias_match",
                "nearest_type_match",
                "voice_guidance",
                "escort_handoff_payload",
            ],
        }

    def points_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        point_type = str(body.get("point_type") or "").strip()
        points = [point for point in self._points if point.enabled]
        if point_type:
            points = [point for point in points if point.point_type == point_type]
        return {"ok": True, "park_id": self.park_id, "points": [point.to_payload() for point in points]}

    def service_points_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = body
        return {
            "ok": True,
            "park_id": self.park_id,
            "service_points": [
                service_point.to_payload()
                for service_point in self._service_points
                if service_point.enabled
            ],
        }

    def routes_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        from_point_id = str(body.get("from_point_id") or "").strip()
        to_point_id = str(body.get("to_point_id") or "").strip()
        routes = [route for route in self._routes if route.enabled]
        if from_point_id:
            routes = [route for route in routes if route.from_point_id == from_point_id]
        if to_point_id:
            routes = [route for route in routes if route.to_point_id == to_point_id]
        return {"ok": True, "park_id": self.park_id, "routes": [route.to_payload() for route in routes]}

    def resolve_destination_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        query = str(body.get("query") or body.get("destination") or body.get("text") or "").strip()
        if not query:
            return {
                "resolved": False,
                "reason": "empty_query",
                "reply": "请说出想去的地点，例如西门、卫生间或某个商户名称。",
            }
        current_point_id = str(body.get("current_point_id") or body.get("from_point_id") or "").strip()
        match = self._resolve(query, current_point_id=current_point_id)
        if match is None:
            return {
                "resolved": False,
                "reason": "destination_not_found",
                "query": query,
                "reply": "我还没有在园区点位库里找到这个地点，请换一个说法或联系工作人员确认。",
                "requires_operator_update": True,
            }
        point, confidence, reason = match
        return {
            "resolved": True,
            "query": query,
            "confidence": confidence,
            "match_reason": reason,
            "point": point.to_payload(),
            "confirmation_prompt": f"你是要去{point.point_name}吗？",
            "requires_confirmation": True,
        }

    def guide_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        resolved = self.resolve_destination_payload(body)
        if not resolved.get("resolved"):
            return {**resolved, "guide_ready": False}
        point = ParkPoint.from_dict(resolved["point"])
        from_point_id = str(body.get("current_point_id") or body.get("from_point_id") or "").strip()
        requested_mode = str(body.get("guide_mode") or body.get("mode") or "").strip().lower()
        route = self._select_route(from_point_id=from_point_id, to_point_id=point.point_id)
        mode = self._guide_mode(point, route, requested_mode)
        route_payload = route.to_payload() if route else None
        instructions = self._instructions(point, route)
        payload = {
            "ok": True,
            "guide_ready": True,
            "mode": mode,
            "point": point.to_payload(),
            "route": route_payload,
            "speech_text": instructions,
            "requires_confirmation": True,
            "confirmation_prompt": resolved.get("confirmation_prompt"),
        }
        if mode == "escort":
            payload["field_event_payload"] = self._escort_event_payload(
                point=point,
                route=route,
                body=body,
                speech_text=instructions,
            )
        return payload

    def _resolve(
        self,
        query: str,
        *,
        current_point_id: str = "",
    ) -> tuple[ParkPoint, float, str] | None:
        candidates = [point for point in self._points if point.enabled]
        normalized = _slug(query)
        if not normalized:
            return None
        if self._asks_nearest(query):
            point_type = self._query_type(query)
            if point_type:
                nearest = self._nearest_of_type(point_type, current_point_id=current_point_id)
                if nearest is not None:
                    return nearest, 0.82, f"nearest_{point_type}"
        best: tuple[ParkPoint, float, str] | None = None
        for point in candidates:
            score, reason = self._score_point(point, normalized)
            if score <= 0:
                continue
            if best is None or score > best[1]:
                best = (point, score, reason)
        return best

    def _score_point(self, point: ParkPoint, normalized_query: str) -> tuple[float, str]:
        names = [point.point_name, *point.aliases]
        for name in names:
            normalized_name = _slug(name)
            if not normalized_name:
                continue
            if normalized_query == normalized_name:
                return (0.99, "exact_name_or_alias")
            if normalized_query in normalized_name or normalized_name in normalized_query:
                return (0.86, "partial_name_or_alias")
        point_type = self._query_type(normalized_query)
        if point_type and point.point_type == point_type:
            return (0.68, "type_keyword")
        return (0.0, "")

    def _query_type(self, query: str) -> str:
        normalized = _slug(query)
        for point_type, keywords in _TYPE_KEYWORDS.items():
            if any(_slug(keyword) in normalized for keyword in keywords):
                return point_type
        return ""

    def _asks_nearest(self, query: str) -> bool:
        normalized = _slug(query)
        return any(keyword in normalized for keyword in ("最近", "附近", "nearest", "closest"))

    def _nearest_of_type(self, point_type: str, *, current_point_id: str) -> ParkPoint | None:
        typed = [point for point in self._points if point.enabled and point.point_type == point_type]
        if not typed:
            return None
        origin = self._point_by_id(current_point_id)
        if origin is None or origin.x is None or origin.y is None:
            return typed[0]
        return min(typed, key=lambda point: _distance(origin, point))

    def _select_route(self, *, from_point_id: str, to_point_id: str) -> GuideRoute | None:
        enabled = [route for route in self._routes if route.enabled and route.to_point_id == to_point_id]
        if from_point_id:
            for route in enabled:
                if route.from_point_id == from_point_id:
                    return route
        return enabled[0] if enabled else None

    def _guide_mode(
        self,
        point: ParkPoint,
        route: GuideRoute | None,
        requested_mode: str,
    ) -> str:
        if requested_mode == "voice":
            return "voice"
        if requested_mode == "escort" and route is not None and route.robot_passable:
            return "escort"
        if route is not None and route.guide_mode == "escort" and route.robot_passable:
            return "escort"
        if point.guide_mode == "escort" and route is not None and route.robot_passable:
            return "escort"
        return "voice"

    def _instructions(self, point: ParkPoint, route: GuideRoute | None) -> str:
        if route is not None and route.instructions:
            return route.instructions
        location = point.point_name
        floor = f"{point.floor}" if point.floor else ""
        building = f"{point.building}" if point.building else ""
        where = "".join(part for part in (building, floor, location) if part)
        return f"{location}在{where or location}。请沿园区主通道前往，途中以现场标识为准。"

    def _escort_event_payload(
        self,
        *,
        point: ParkPoint,
        route: GuideRoute | None,
        body: dict[str, Any],
        speech_text: str,
    ) -> dict[str, Any]:
        service_point_id = str(body.get("service_point_id") or body.get("help_point_id") or "").strip()
        return {
            "scenario_id": "visitor_escort",
            "destination": point.point_name,
            "destination_point_id": point.point_id,
            "route_id": route.route_id if route else "",
            "location": str(body.get("location") or service_point_id or body.get("current_point_id") or "-"),
            "map_id": point.map_id,
            "service_point_id": service_point_id,
            "guide_mode": "escort",
            "speech_text": speech_text,
            "timestamp": time.time(),
        }

    def _point_by_id(self, point_id: str) -> ParkPoint | None:
        for point in self._points:
            if point.point_id == point_id:
                return point
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [
            {"id": key, **item}
            if isinstance(item, dict) and "id" not in item and "point_id" not in item
            else item
            for key, item in value.items()
            if isinstance(item, dict)
        ]
    return []


def _distance(a: ParkPoint, b: ParkPoint) -> float:
    if a.x is None or a.y is None or b.x is None or b.y is None:
        return math.inf
    return math.hypot(a.x - b.x, a.y - b.y)


def _points_from_site_map(config: dict[str, Any], *, park_id: str) -> list[ParkPoint]:
    field_cfg = config.get("field_operations") if isinstance(config.get("field_operations"), dict) else {}
    site_map = field_cfg.get("site_map") if isinstance(field_cfg.get("site_map"), dict) else {}
    points: list[ParkPoint] = []
    for zone in _dict_list(site_map.get("zones")):
        zone_id = str(zone.get("id") or zone.get("zone_id") or "").strip()
        if not zone_id:
            continue
        zone_type = str(zone.get("type") or "place").strip()
        point_type = "service" if zone_type in {"help_point", "service_point"} else zone_type
        points.append(
            ParkPoint(
                point_id=zone_id,
                park_id=park_id,
                map_id=str(site_map.get("map_id") or zone.get("map_id") or "default"),
                point_name=str(zone.get("name") or zone_id),
                point_type=point_type,
                aliases=tuple(_list_texts(zone.get("aliases"))),
                guide_mode=str(zone.get("guide_mode") or "voice"),
                accessibility=str(zone.get("accessibility") or "unknown"),
                enabled=bool(zone.get("enabled", True)),
            )
        )
    return points


def _service_points_from_site_map(config: dict[str, Any], *, park_id: str) -> list[ServicePoint]:
    field_cfg = config.get("field_operations") if isinstance(config.get("field_operations"), dict) else {}
    site_map = field_cfg.get("site_map") if isinstance(field_cfg.get("site_map"), dict) else {}
    service_points: list[ServicePoint] = []
    for zone in _dict_list(site_map.get("zones")):
        if str(zone.get("type") or "") not in {"help_point", "service_point"}:
            continue
        zone_id = str(zone.get("id") or zone.get("zone_id") or "").strip()
        help_point_id = str(zone.get("help_point_id") or zone_id).strip()
        if not help_point_id:
            continue
        service_points.append(
            ServicePoint(
                service_point_id=help_point_id,
                park_id=park_id,
                map_id=str(site_map.get("map_id") or zone.get("map_id") or "default"),
                point_id=zone_id,
                service_point_name=str(zone.get("name") or help_point_id),
                dwell_seconds=float(zone.get("dwell_seconds") or zone.get("dwell_s") or 3.0),
                greeting_prompt=str(zone.get("greeting_prompt") or "你好，请问需要指路吗？"),
                supported_intents=("wayfinding", "escort"),
                enabled=bool(zone.get("enabled", True)),
            )
        )
    return service_points
