"""Park semantic-map and visitor wayfinding services.

This module stays above the robot control layer. It resolves visitor language
into park points and high-level guide decisions; it never emits low-level
motion commands.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from askme.runtime.task.handoff import SkillRegistry, TaskHandoff

_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "restroom": (
        "restroom",
        "toilet",
        "wc",
        "bathroom",
        "\u536b\u751f\u95f4",
        "\u5395\u6240",
        "\u6d17\u624b\u95f4",
    ),
    "parking": (
        "parking",
        "park",
        "\u505c\u8f66",
        "\u505c\u8f66\u573a",
        "\u505c\u8f66\u533a",
        "\u8f66\u4f4d",
    ),
    "restaurant": (
        "restaurant",
        "food",
        "dining",
        "\u5403\u996d",
        "\u9910\u5385",
        "\u996d",
        "\u5c0f\u5403",
        "\u5496\u5561",
    ),
    "exit": (
        "exit",
        "gate",
        "\u5165\u53e3",
        "\u51fa\u53e3",
        "\u5927\u95e8",
        "\u95e8",
    ),
    "service": (
        "service",
        "help",
        "\u95ee\u8be2",
        "\u670d\u52a1\u53f0",
        "\u670d\u52a1\u70b9",
    ),
}

_EMPTY_DESTINATION_REPLY = (
    "\u8bf7\u8bf4\u51fa\u60f3\u53bb\u7684\u5730\u70b9\uff0c"
    "\u4f8b\u5982\u897f\u95e8\u3001\u536b\u751f\u95f4\u6216\u5546\u6237\u540d\u79f0\u3002"
)
_DESTINATION_NOT_FOUND_REPLY = (
    "\u6211\u8fd8\u6ca1\u6709\u5728\u56ed\u533a\u70b9\u4f4d\u5e93\u91cc\u627e\u5230\u8fd9\u4e2a\u5730\u70b9\uff0c"
    "\u8bf7\u6362\u4e00\u79cd\u8bf4\u6cd5\u6216\u8054\u7cfb\u5de5\u4f5c\u4eba\u5458\u786e\u8ba4\u3002"
)
_DEFAULT_GREETING = "\u4f60\u597d\uff0c\u8bf7\u95ee\u9700\u8981\u6307\u8def\u5417\uff1f"
_CATEGORY_SEARCH_KEYWORDS = (
    "\u6709\u54ea\u4e9b",
    "\u54ea\u4e9b",
    "\u6709\u4ec0\u4e48",
    "\u54ea\u91cc\u6709",
    "\u9644\u8fd1\u6709",
    "list",
    "available",
)
_POINT_TYPE_LABELS = {
    "restroom": "\u536b\u751f\u95f4",
    "parking": "\u505c\u8f66\u533a",
    "restaurant": "\u9910\u996e\u5730\u70b9",
    "exit": "\u51fa\u5165\u53e3",
    "service": "\u670d\u52a1\u70b9",
}


def _slug(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    for separator in (",", "\uff0c", ";", "\uff1b", "\u3001", "?", "\uff1f", "!", "\uff01"):
        normalized = normalized.replace(separator, "")
    return re.sub(r"[\s\-_\\/]+", "", normalized)


def _list_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value
        for separator in ("\uff0c", "\uff1b", "\u3001"):
            text = text.replace(separator, ",")
        return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "confirmed", "ok", "\u662f", "\u786e\u8ba4"}


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
    def from_dict(cls, raw: dict[str, Any]) -> ParkPoint:
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
    def from_dict(cls, raw: dict[str, Any]) -> ServicePoint:
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
    def from_dict(cls, raw: dict[str, Any]) -> GuideRoute:
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
        store_path: str | Path | None = None,
        revision: int = 0,
        change_log: list[dict[str, Any]] | None = None,
        snapshots: list[dict[str, Any]] | None = None,
        pending_changes: list[dict[str, Any]] | None = None,
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.park_id = park_id
        self._points = [point for point in (points or []) if point.point_id]
        self._service_points = [point for point in (service_points or []) if point.service_point_id]
        self._routes = [route for route in (routes or []) if route.route_id]
        self._store_path = Path(str(store_path)) if store_path else None
        self._store_lock = RLock()
        self._revision = max(0, int(revision or 0))
        self._change_log = _dict_list(change_log)
        self._snapshots = _dict_list(snapshots)
        self._pending_changes = _dict_list(pending_changes)
        self._interactions = _dict_list(interactions)
        if not self._snapshot_for_revision(self._revision):
            self._snapshots.append(self._snapshot_payload(revision=self._revision))
        self._snapshots = self._snapshots[-100:]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParkSpaceService:
        raw = config.get("space_cognition") if isinstance(config.get("space_cognition"), dict) else {}
        park_id = str(raw.get("park_id") or "default")
        points = [ParkPoint.from_dict(item) for item in _dict_list(raw.get("points"))]
        service_points = [ServicePoint.from_dict(item) for item in _dict_list(raw.get("service_points"))]
        routes = [GuideRoute.from_dict(item) for item in _dict_list(raw.get("routes"))]

        if not points:
            points.extend(_points_from_site_map(config, park_id=park_id))
        if not service_points:
            service_points.extend(_service_points_from_site_map(config, park_id=park_id))
        store_path = _optional_store_path(
            raw.get("store_path") or raw.get("json_path"),
            project_root=config.get("_project_root"),
        )
        store = _load_store(store_path)
        points = _merge_points(points, [ParkPoint.from_dict(item) for item in _dict_list(store.get("points"))])
        service_points = _merge_service_points(
            service_points,
            [ServicePoint.from_dict(item) for item in _dict_list(store.get("service_points"))],
        )
        routes = _merge_routes(routes, [GuideRoute.from_dict(item) for item in _dict_list(store.get("routes"))])
        return cls(
            park_id=park_id,
            points=points,
            service_points=service_points,
            routes=routes,
            store_path=store_path,
            revision=int(store.get("revision") or 0),
            change_log=_dict_list(store.get("change_log")),
            snapshots=_dict_list(store.get("snapshots")),
            pending_changes=_dict_list(store.get("pending_changes")),
            interactions=_dict_list(store.get("interactions")),
        )

    def health_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = body
        return {
            "enabled": bool(self._points),
            "park_id": self.park_id,
            "points": len(self._points),
            "service_points": len(self._service_points),
            "routes": len(self._routes),
            "revision": self._revision,
            "changes": len(self._change_log),
            "snapshots": len(self._snapshots),
            "pending_changes": len([item for item in self._pending_changes if item.get("status") == "pending"]),
            "interactions": len(self._interactions),
            "store": {
                "configured": self._store_path is not None,
                "path": str(self._store_path or ""),
                "exists": bool(self._store_path and self._store_path.is_file()),
            },
            "capabilities": [
                "destination_resolve",
                "alias_match",
                "nearest_type_match",
                "service_point_trigger",
                "voice_guidance",
                "escort_handoff_payload",
                "space_catalog_management",
                "space_catalog_change_history",
                "space_catalog_rollback",
                "space_catalog_approval",
                "interaction_records",
            ],
        }

    def history_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        limit = int(body.get("limit") or 50)
        limit = min(max(limit, 1), 200)
        return {
            "ok": True,
            "park_id": self.park_id,
            "revision": self._revision,
            "changes": list(reversed(self._change_log[-limit:])),
            "available_revisions": [
                revision
                for revision in (
                    _optional_int(snapshot.get("revision"))
                    for snapshot in self._snapshots
                )
                if revision is not None
            ],
            "store": {
                "configured": self._store_path is not None,
                "path": str(self._store_path or ""),
            },
        }

    def proposals_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        status = str(body.get("status") or "").strip().lower()
        proposals = self._pending_changes
        if status:
            proposals = [item for item in proposals if str(item.get("status") or "") == status]
        return {
            "ok": True,
            "park_id": self.park_id,
            "revision": self._revision,
            "proposals": [dict(item) for item in reversed(proposals)],
            "pending_count": len([item for item in self._pending_changes if item.get("status") == "pending"]),
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

    def interactions_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        limit = max(1, min(int(body.get("limit") or 50), 200))
        service_point_id = str(body.get("service_point_id") or "").strip()
        destination_point_id = str(body.get("destination_point_id") or "").strip()
        status = str(body.get("status") or "").strip()
        records = list(reversed(self._interactions))
        if service_point_id:
            records = [
                item
                for item in records
                if str(item.get("service_point_id") or "") == service_point_id
            ]
        if destination_point_id:
            records = [
                item
                for item in records
                if str(item.get("destination_point_id") or "") == destination_point_id
            ]
        if status:
            records = [item for item in records if str(item.get("status") or "") == status]
        visible = records[:limit]
        return {
            "ok": True,
            "park_id": self.park_id,
            "count": len(visible),
            "total": len(records),
            "limit": limit,
            "interactions": [dict(item) for item in visible],
        }

    def service_point_trigger_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        service_point = self._service_point_from_body(body)
        if service_point is None:
            return {
                "ok": True,
                "should_prompt": False,
                "reason": "service_point_not_found",
                "speech_text": "",
                "admission": "ignore",
            }
        if not service_point.enabled:
            payload = {
                "ok": True,
                "should_prompt": False,
                "reason": "service_point_disabled",
                "speech_text": "",
                "admission": "ignore",
                "service_point": service_point.to_payload(),
            }
            self._record_interaction(
                event_type="service_point_trigger",
                status="ignored",
                body=body,
                result=payload,
                service_point=service_point,
            )
            return payload
        if not bool(body.get("person_present", True)):
            payload = {
                "ok": True,
                "should_prompt": False,
                "reason": "no_person",
                "speech_text": "",
                "admission": "observe",
                "service_point": service_point.to_payload(),
            }
            self._record_interaction(
                event_type="service_point_trigger",
                status="observed",
                body=body,
                result=payload,
                service_point=service_point,
            )
            return payload
        dwell_seconds = float(body.get("dwell_seconds") or body.get("dwell_s") or 0.0)
        required = max(0.0, float(service_point.dwell_seconds or 0.0))
        if dwell_seconds < required:
            payload = {
                "ok": True,
                "should_prompt": False,
                "reason": "dwell_time_too_short",
                "speech_text": "",
                "admission": "observe",
                "dwell_seconds": dwell_seconds,
                "required_dwell_seconds": required,
                "service_point": service_point.to_payload(),
            }
            self._record_interaction(
                event_type="service_point_trigger",
                status="observed",
                body=body,
                result=payload,
                service_point=service_point,
            )
            return payload
        speech_text = service_point.greeting_prompt or service_point.idle_prompt or _DEFAULT_GREETING
        payload = {
            "ok": True,
            "should_prompt": True,
            "reason": "visitor_dwelling_at_service_point",
            "admission": "prompt",
            "speech_text": speech_text,
            "dwell_seconds": dwell_seconds,
            "required_dwell_seconds": required,
            "service_point": service_point.to_payload(),
            "supported_intents": list(service_point.supported_intents or ("wayfinding", "escort")),
            "next_expected_input": "visitor_destination",
        }
        payload["interaction_id"] = self._record_interaction(
            event_type="service_point_trigger",
            status="prompted",
            body=body,
            result=payload,
            service_point=service_point,
        )
        return payload

    def manage_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        entity = str(body.get("entity") or body.get("type") or "").strip().lower()
        action = str(body.get("action") or "upsert").strip().lower()
        if entity not in {"point", "service_point", "route"}:
            return {
                "ok": False,
                "reason": "invalid_space_entity",
                "allowed_entities": ["point", "service_point", "route"],
            }
        if action not in {"upsert", "enable", "disable", "delete"}:
            return {
                "ok": False,
                "reason": "invalid_space_action",
                "allowed_actions": ["upsert", "enable", "disable", "delete"],
            }
        item = body.get("item") if isinstance(body.get("item"), dict) else body
        with self._store_lock:
            validation = self._validate_manage_request(entity=entity, action=action, item=item)
            if validation is not None:
                return validation
            result = self._apply_manage_action(entity=entity, action=action, item=item)
            if result.get("ok"):
                self._revision += 1
                change = self._change_entry(
                    body=body,
                    entity=entity,
                    action=action,
                    item=item,
                    result=result,
                )
                self._change_log.append(change)
                self._change_log = self._change_log[-500:]
                result["revision"] = self._revision
                result["change"] = change
                self._append_snapshot()
            if result.get("ok") and self._store_path is not None:
                result["persisted"] = self._persist()
            elif result.get("ok"):
                result["persisted"] = {
                    "written": False,
                    "reason": "space_store_path_not_configured",
                }
        return result

    def propose_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        entity = str(body.get("entity") or body.get("type") or "").strip().lower()
        action = str(body.get("action") or "upsert").strip().lower()
        if entity not in {"point", "service_point", "route"}:
            return {
                "ok": False,
                "reason": "invalid_space_entity",
                "allowed_entities": ["point", "service_point", "route"],
            }
        if action not in {"upsert", "enable", "disable", "delete"}:
            return {
                "ok": False,
                "reason": "invalid_space_action",
                "allowed_actions": ["upsert", "enable", "disable", "delete"],
            }
        item = body.get("item") if isinstance(body.get("item"), dict) else body
        validation = self._validate_manage_request(entity=entity, action=action, item=item)
        if validation is not None:
            return {**validation, "proposal_created": False}
        proposal = {
            "proposal_id": _proposal_id(self._pending_changes),
            "status": "pending",
            "created_at": time.time(),
            "operator_id": str(body.get("operator_id") or body.get("actor_id") or "unknown").strip(),
            "entity": entity,
            "action": action,
            "item": item,
            "reason": str(body.get("reason") or body.get("change_reason") or "").strip(),
            "base_revision": self._revision,
        }
        with self._store_lock:
            self._pending_changes.append(proposal)
            self._pending_changes = self._pending_changes[-500:]
            persisted = self._persist() if self._store_path is not None else {
                "written": False,
                "reason": "space_store_path_not_configured",
            }
        return {
            "ok": True,
            "proposal_created": True,
            "proposal": dict(proposal),
            "persisted": persisted,
        }

    def review_proposal_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        proposal_id = str(body.get("proposal_id") or body.get("id") or "").strip()
        decision = str(body.get("decision") or body.get("action") or "").strip().lower()
        if not proposal_id:
            return {"ok": False, "reason": "proposal_id_required"}
        if decision not in {"approve", "reject"}:
            return {"ok": False, "reason": "invalid_review_decision"}
        with self._store_lock:
            proposal = self._proposal_by_id(proposal_id)
            if proposal is None:
                return {"ok": False, "reason": "proposal_not_found", "proposal_id": proposal_id}
            if proposal.get("status") != "pending":
                return {
                    "ok": False,
                    "reason": "proposal_already_reviewed",
                    "proposal_id": proposal_id,
                    "status": proposal.get("status"),
                }
            reviewer_id = str(body.get("operator_id") or body.get("actor_id") or "unknown").strip()
            if decision == "reject":
                proposal["status"] = "rejected"
                proposal["reviewed_at"] = time.time()
                proposal["reviewer_id"] = reviewer_id
                proposal["review_reason"] = str(body.get("reason") or body.get("review_reason") or "").strip()
                persisted = self._persist() if self._store_path is not None else {
                    "written": False,
                    "reason": "space_store_path_not_configured",
                }
                return {"ok": True, "reviewed": True, "proposal": proposal, "persisted": persisted}

            entity = str(proposal.get("entity") or "")
            action = str(proposal.get("action") or "")
            item = proposal.get("item") if isinstance(proposal.get("item"), dict) else {}
            validation = self._validate_manage_request(entity=entity, action=action, item=item)
            if validation is not None:
                return {
                    **validation,
                    "proposal_id": proposal_id,
                    "reviewed": False,
                    "reason": f"proposal_apply_failed:{validation.get('reason')}",
                }
            result = self._apply_manage_action(entity=entity, action=action, item=item)
            if not result.get("ok"):
                return {**result, "proposal_id": proposal_id, "reviewed": False}
            self._revision += 1
            change = self._change_entry(
                body={
                    **proposal,
                    "operator_id": reviewer_id,
                    "reason": body.get("reason") or proposal.get("reason") or "",
                },
                entity=entity,
                action=action,
                item=item,
                result=result,
            )
            change["proposal_id"] = proposal_id
            self._change_log.append(change)
            self._change_log = self._change_log[-500:]
            self._append_snapshot()
            proposal["status"] = "approved"
            proposal["reviewed_at"] = time.time()
            proposal["reviewer_id"] = reviewer_id
            proposal["applied_revision"] = self._revision
            result["revision"] = self._revision
            result["change"] = change
            result["proposal"] = proposal
            result["reviewed"] = True
            result["persisted"] = self._persist() if self._store_path is not None else {
                "written": False,
                "reason": "space_store_path_not_configured",
            }
            return result

    def rollback_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        target_revision = _optional_int(body.get("revision") or body.get("target_revision"))
        if target_revision is None or target_revision < 0:
            return {"ok": False, "reason": "target_revision_required"}
        with self._store_lock:
            snapshot = self._snapshot_for_revision(target_revision)
            if snapshot is None:
                return {
                    "ok": False,
                    "reason": "space_revision_not_found",
                    "target_revision": target_revision,
                    "available_revisions": [
                        revision
                        for revision in (
                            _optional_int(item.get("revision"))
                            for item in self._snapshots
                        )
                        if revision is not None
                    ],
                }
            if target_revision == self._revision:
                return {
                    "ok": False,
                    "reason": "space_revision_already_current",
                    "target_revision": target_revision,
                    "revision": self._revision,
                }
            self._apply_snapshot(snapshot)
            self._revision += 1
            change = {
                "revision": self._revision,
                "timestamp": time.time(),
                "operator_id": str(body.get("operator_id") or body.get("actor_id") or "unknown").strip(),
                "entity": "catalog",
                "action": "rollback",
                "item_id": f"revision-{target_revision}",
                "reason": str(body.get("reason") or body.get("change_reason") or "").strip(),
                "status": "applied",
                "restored_revision": target_revision,
            }
            self._change_log.append(change)
            self._change_log = self._change_log[-500:]
            self._append_snapshot()
            result = {
                "ok": True,
                "revision": self._revision,
                "restored_revision": target_revision,
                "change": change,
            }
            if self._store_path is not None:
                result["persisted"] = self._persist()
            else:
                result["persisted"] = {
                    "written": False,
                    "reason": "space_store_path_not_configured",
                }
            return result

    def _change_entry(
        self,
        *,
        body: dict[str, Any],
        entity: str,
        action: str,
        item: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "revision": self._revision,
            "timestamp": time.time(),
            "operator_id": str(body.get("operator_id") or body.get("actor_id") or "unknown").strip(),
            "entity": entity,
            "action": action,
            "item_id": _managed_item_id(entity=entity, item=item, result=result),
            "reason": str(body.get("reason") or body.get("change_reason") or "").strip(),
            "status": "applied",
        }

    def _validate_manage_request(
        self,
        *,
        entity: str,
        action: str,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        if entity == "point":
            point_id = str(item.get("point_id") or item.get("id") or "").strip()
            if not point_id:
                return {"ok": False, "reason": "point_id_required"}
            existing = self._point_by_id(point_id)
            if action == "delete":
                if not bool(item.get("force")):
                    references = self._point_references(point_id)
                    if references:
                        return {
                            "ok": False,
                            "reason": "point_in_use",
                            "point_id": point_id,
                            "references": references,
                            "hint": "disable or delete dependent service points and routes first, or pass force=true",
                        }
                return None
            if action in {"enable", "disable"}:
                if existing is None:
                    return {"ok": False, "reason": "point_not_found", "point_id": point_id}
                return None
            if not ParkPoint.from_dict(item).point_name:
                return {"ok": False, "reason": "point_name_required"}
            return None
        if entity == "service_point":
            service_point_id = str(item.get("service_point_id") or item.get("id") or "").strip()
            if not service_point_id:
                return {"ok": False, "reason": "service_point_id_required"}
            existing = self._service_point_by_id(service_point_id)
            if action == "delete":
                return None
            if action in {"enable", "disable"}:
                if existing is None:
                    return {"ok": False, "reason": "service_point_not_found", "service_point_id": service_point_id}
                return None
            service_point = ServicePoint.from_dict(item)
            if not service_point.point_id:
                return {"ok": False, "reason": "service_point_requires_point_id"}
            if self._point_by_id(service_point.point_id) is None:
                return {
                    "ok": False,
                    "reason": "service_point_point_not_found",
                    "point_id": service_point.point_id,
                }
            return None
        route_id = str(item.get("route_id") or item.get("id") or "").strip()
        if not route_id:
            return {"ok": False, "reason": "route_id_required"}
        existing = self._route_by_id(route_id)
        if action == "delete":
            return None
        if action in {"enable", "disable"}:
            if existing is None:
                return {"ok": False, "reason": "route_not_found", "route_id": route_id}
            return None
        route = GuideRoute.from_dict(item)
        if not route.from_point_id or not route.to_point_id:
            return {"ok": False, "reason": "route_requires_from_and_to"}
        missing_point_ids = [
            point_id
            for point_id in (route.from_point_id, route.to_point_id)
            if self._point_by_id(point_id) is None
        ]
        if missing_point_ids:
            return {
                "ok": False,
                "reason": "route_point_not_found",
                "missing_point_ids": missing_point_ids,
            }
        return None

    def _snapshot_payload(self, *, revision: int | None = None) -> dict[str, Any]:
        return {
            "revision": self._revision if revision is None else revision,
            "timestamp": time.time(),
            "points": [point.to_payload() for point in self._points],
            "service_points": [point.to_payload() for point in self._service_points],
            "routes": [route.to_payload() for route in self._routes],
        }

    def _append_snapshot(self) -> None:
        self._snapshots = [
            snapshot
            for snapshot in self._snapshots
            if _optional_int(snapshot.get("revision")) != self._revision
        ]
        self._snapshots.append(self._snapshot_payload())
        self._snapshots = self._snapshots[-100:]

    def _snapshot_for_revision(self, revision: int) -> dict[str, Any] | None:
        for snapshot in reversed(self._snapshots):
            if _optional_int(snapshot.get("revision")) == revision:
                return snapshot
        return None

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._points = [ParkPoint.from_dict(item) for item in _dict_list(snapshot.get("points"))]
        self._service_points = [
            ServicePoint.from_dict(item)
            for item in _dict_list(snapshot.get("service_points"))
        ]
        self._routes = [GuideRoute.from_dict(item) for item in _dict_list(snapshot.get("routes"))]

    def resolve_destination_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        query = str(body.get("query") or body.get("destination") or body.get("text") or "").strip()
        if not query:
            return {
                "resolved": False,
                "reason": "empty_query",
                "reply": _EMPTY_DESTINATION_REPLY,
            }
        current_point_id = str(body.get("current_point_id") or body.get("from_point_id") or "").strip()
        category_result = self._category_resolution_payload(
            query,
            current_point_id=current_point_id,
        )
        if category_result is not None:
            return category_result
        match = self._resolve(query, current_point_id=current_point_id)
        if match is None:
            return {
                "resolved": False,
                "reason": "destination_not_found",
                "query": query,
                "reply": _DESTINATION_NOT_FOUND_REPLY,
                "requires_operator_update": True,
            }
        point, confidence, reason = match
        return {
            "resolved": True,
            "query": query,
            "confidence": confidence,
            "match_reason": reason,
            "point": point.to_payload(),
            "confirmation_prompt": f"\u4f60\u662f\u8981\u53bb{point.point_name}\u5417\uff1f",
            "requires_confirmation": True,
        }

    def _category_resolution_payload(
        self,
        query: str,
        *,
        current_point_id: str = "",
    ) -> dict[str, Any] | None:
        point_type = self._query_type(query)
        if not point_type:
            return None
        if self._asks_nearest(query):
            return None
        candidates = self._category_candidates(
            query,
            point_type=point_type,
            current_point_id=current_point_id,
        )
        if not candidates:
            return None
        label = self._query_category_label(query, point_type)
        candidate_payloads = [self._candidate_payload(point) for point in candidates]
        asks_list = self._asks_category_list(query)
        if len(candidates) > 1 or asks_list:
            names = "\u3001".join(point.point_name for point in candidates[:5])
            more = "" if len(candidates) <= 5 else f"\u7b49{len(candidates)}\u4e2a"
            needs_choice = len(candidates) > 1
            suffix = "\u8bf7\u544a\u8bc9\u6211\u4f60\u60f3\u53bb\u54ea\u4e00\u4e2a\u3002" if needs_choice else ""
            return {
                "resolved": False,
                "reason": "multiple_destinations" if len(candidates) > 1 else "category_candidates_found",
                "query": query,
                "point_type": point_type,
                "point_type_label": label,
                "candidate_count": len(candidates),
                "candidates": candidate_payloads,
                "reply": (
                    f"\u6211\u627e\u5230{len(candidates)}\u4e2a{label}\uff1a{names}{more}\u3002"
                    f"{suffix}"
                ),
                "requires_clarification": needs_choice,
                "listing_only": asks_list,
            }
        point = candidates[0]
        return {
            "resolved": True,
            "query": query,
            "confidence": 0.84,
            "match_reason": f"single_{point_type}_candidate",
            "point_type": point_type,
            "point_type_label": label,
            "candidate_count": 1,
            "candidates": candidate_payloads,
            "selection_policy": "single_category_candidate",
            "point": point.to_payload(),
            "reply": (
                f"\u76ee\u524d\u70b9\u4f4d\u5e93\u91cc\u627e\u5230\u4e00\u4e2a{label}\uff1a"
                f"{point.point_name}\u3002"
            ),
            "confirmation_prompt": f"\u4f60\u662f\u8981\u53bb{point.point_name}\u5417\uff1f",
            "requires_confirmation": True,
        }

    def guide_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        resolved = self.resolve_destination_payload(body)
        if not resolved.get("resolved"):
            payload = {**resolved, "guide_ready": False}
            payload["interaction_id"] = self._record_interaction(
                event_type="guide_request",
                status="refused",
                body=body,
                result=payload,
            )
            return payload
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
            payload.update(
                self._escort_runtime_handoff_payload(
                    point=point,
                    route=route,
                    body=body,
                    speech_text=instructions,
                )
            )
        payload["interaction_id"] = self._record_interaction(
            event_type="guide_request",
            status="escort_ready" if mode == "escort" else "voice_ready",
            body=body,
            result=payload,
            point=point,
            route=route,
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

    def _category_candidates(
        self,
        query: str,
        *,
        point_type: str,
        current_point_id: str = "",
    ) -> list[ParkPoint]:
        normalized = _slug(query)
        matched_keywords = [
            _slug(keyword)
            for keyword in _TYPE_KEYWORDS.get(point_type, ())
            if _slug(keyword) and _slug(keyword) in normalized
        ]
        typed = [point for point in self._points if point.enabled and point.point_type == point_type]
        if matched_keywords:
            focused = [
                point
                for point in typed
                if any(
                    keyword in _slug(name)
                    for keyword in matched_keywords
                    for name in (point.point_name, *point.aliases)
                )
            ]
            if focused:
                typed = focused
        origin = self._point_by_id(current_point_id)
        if origin is not None and origin.x is not None and origin.y is not None:
            typed = sorted(typed, key=lambda point: _distance(origin, point))
        return typed

    def _candidate_payload(self, point: ParkPoint) -> dict[str, Any]:
        payload = point.to_payload()
        return {
            "point_id": payload.get("point_id"),
            "point_name": payload.get("point_name"),
            "point_type": payload.get("point_type"),
            "aliases": payload.get("aliases") or [],
            "building": payload.get("building") or "",
            "floor": payload.get("floor") or "",
            "guide_mode": payload.get("guide_mode") or "voice",
            "accessibility": payload.get("accessibility") or "",
        }

    def _query_category_label(self, query: str, point_type: str) -> str:
        normalized = _slug(query)
        if point_type == "restaurant" and any(
            keyword in normalized for keyword in ("\u5496\u5561", "coffee", "cafe")
        ):
            return "\u5496\u5561\u5e97"
        return _POINT_TYPE_LABELS.get(point_type, point_type)

    def _query_type(self, query: str) -> str:
        normalized = _slug(query)
        for point_type, keywords in _TYPE_KEYWORDS.items():
            if any(_slug(keyword) in normalized for keyword in keywords):
                return point_type
        return ""

    def _asks_nearest(self, query: str) -> bool:
        normalized = _slug(query)
        return any(keyword in normalized for keyword in ("\u6700\u8fd1", "\u9644\u8fd1", "nearest", "closest"))

    def _asks_category_list(self, query: str) -> bool:
        normalized = _slug(query)
        return any(_slug(keyword) in normalized for keyword in _CATEGORY_SEARCH_KEYWORDS)

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
        return (
            f"{where or location}\u3002"
            "\u8bf7\u6cbf\u56ed\u533a\u4e3b\u901a\u9053\u524d\u5f80\uff0c"
            "\u9014\u4e2d\u4ee5\u73b0\u573a\u6807\u8bc6\u4e3a\u51c6\u3002"
        )

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

    def _escort_runtime_handoff_payload(
        self,
        *,
        point: ParkPoint,
        route: GuideRoute | None,
        body: dict[str, Any],
        speech_text: str,
    ) -> dict[str, Any]:
        confirmed = _truthy(
            body.get("visitor_confirmed")
            or body.get("destination_confirmed")
            or body.get("confirmed")
        )
        plan = self._escort_handoff_plan(
            point=point,
            route=route,
            body=body,
            speech_text=speech_text,
            confirmed=confirmed,
        )
        registry = SkillRegistry()
        handoff = TaskHandoff.from_plan(
            plan,
            world_state_snapshot=self._escort_world_snapshot(point=point, route=route, body=body),
            skill_registry=registry,
            default_operator_id=str(body.get("operator_id") or "space.operator"),
            planner_version="askme-space-cognition-v1",
            ttl_s=float(body.get("handoff_ttl_s") or 300.0),
        )
        validation = handoff.validate(registry)
        key = "runtime_handoff" if confirmed else "runtime_handoff_preview"
        return {
            key: handoff.to_dict(),
            "runtime_handoff_plan": plan,
            "runtime_handoff_validation": validation,
            "runtime_handoff_ready": confirmed and not validation,
            "runtime_handoff_reason": (
                "visitor_destination_confirmed"
                if confirmed
                else "visitor_destination_confirmation_required"
            ),
        }

    def _escort_handoff_plan(
        self,
        *,
        point: ParkPoint,
        route: GuideRoute | None,
        body: dict[str, Any],
        speech_text: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        service_point_id = str(body.get("service_point_id") or body.get("help_point_id") or "").strip()
        route_id = route.route_id if route else ""
        area_id = route_id or point.point_id
        operator_id = str(body.get("operator_id") or "space.operator")
        return {
            "plan_id": str(body.get("plan_id") or f"space-escort-{point.point_id}"),
            "planning_session_id": str(
                body.get("planning_session_id")
                or body.get("session_id")
                or f"space-guide-{service_point_id or point.point_id}"
            ),
            "intent": "visitor_escort",
            "handoff_ready": confirmed,
            "operator_id": operator_id,
            "operator_roles": _list_texts(body.get("operator_roles") or ["operator"]),
            "missing_inputs": [] if confirmed else ["visitor_destination_confirmation"],
            "safety_constraints": [
                "low_speed_only",
                "visitor_must_remain_tracked",
                "pause_on_obstacle_or_lost_visitor",
                "return_to_patrol_after_completion_or_cancel",
            ],
            "reference": {
                "resolved": {
                    "area_id": area_id,
                    "point_id": point.point_id,
                    "route_id": route_id,
                    "map_id": point.map_id,
                },
            },
            "mission": {
                "mission": {
                    "mission_type": "visitor_escort",
                    "operator_id": operator_id,
                    "risk_tier": "high",
                    "goal": f"escort visitor to {point.point_name}",
                    "field_event": {
                        "scenario_id": "visitor_escort",
                        "robot_motion_policy": "low_speed_escort",
                        "destination": point.point_name,
                        "destination_point_id": point.point_id,
                        "route_id": route_id,
                        "map_id": point.map_id,
                        "service_point_id": service_point_id,
                        "speech_text": speech_text,
                        "speed_limit": "low",
                    },
                    "steps": [
                        {
                            "target": area_id,
                            "summary": "low_speed_visitor_escort",
                        }
                    ],
                    "safety_notes": [
                        "Visitor escort requires fresh localization and interaction target lock.",
                        "If the visitor is lost or path becomes crowded, pause and ask for operator help.",
                    ],
                }
            },
        }

    def _escort_world_snapshot(
        self,
        *,
        point: ParkPoint,
        route: GuideRoute | None,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = body.get("world_state_snapshot")
        if isinstance(snapshot, dict):
            return dict(snapshot)
        return {
            "updated_at": float(body.get("world_updated_at") or time.time()),
            "fact_count": 4,
            "space_revision": self._revision,
            "park_id": self.park_id,
            "destination_point_id": point.point_id,
            "route_id": route.route_id if route else "",
            "map_id": point.map_id,
        }

    def _service_point_from_body(self, body: dict[str, Any]) -> ServicePoint | None:
        service_point_id = str(body.get("service_point_id") or body.get("help_point_id") or "").strip()
        point_id = str(body.get("point_id") or body.get("current_point_id") or "").strip()
        for service_point in self._service_points:
            if service_point_id and service_point.service_point_id == service_point_id:
                return service_point
            if point_id and service_point.point_id == point_id:
                return service_point
        return None

    def _apply_manage_action(self, *, entity: str, action: str, item: dict[str, Any]) -> dict[str, Any]:
        if entity == "point":
            return self._manage_point(action=action, item=item)
        if entity == "service_point":
            return self._manage_service_point(action=action, item=item)
        return self._manage_route(action=action, item=item)

    def _manage_point(self, *, action: str, item: dict[str, Any]) -> dict[str, Any]:
        point_id = str(item.get("point_id") or item.get("id") or "").strip()
        if not point_id:
            return {"ok": False, "reason": "point_id_required"}
        existing = self._point_by_id(point_id)
        if action == "delete":
            if not bool(item.get("force")):
                references = self._point_references(point_id)
                if references:
                    return {
                        "ok": False,
                        "reason": "point_in_use",
                        "point_id": point_id,
                        "references": references,
                        "hint": "disable or delete dependent service points and routes first, or pass force=true",
                    }
            before = len(self._points)
            self._points = [point for point in self._points if point.point_id != point_id]
            return {"ok": True, "action": action, "entity": "point", "deleted": len(self._points) < before}
        if action in {"enable", "disable"}:
            if existing is None:
                return {"ok": False, "reason": "point_not_found", "point_id": point_id}
            item = {**existing.to_payload(), "enabled": action == "enable"}
        point = ParkPoint.from_dict(item)
        if not point.point_name:
            return {"ok": False, "reason": "point_name_required"}
        self._points = [candidate for candidate in self._points if candidate.point_id != point.point_id]
        self._points.append(point)
        return {"ok": True, "action": action, "entity": "point", "point": point.to_payload()}

    def _manage_service_point(self, *, action: str, item: dict[str, Any]) -> dict[str, Any]:
        service_point_id = str(item.get("service_point_id") or item.get("id") or "").strip()
        if not service_point_id:
            return {"ok": False, "reason": "service_point_id_required"}
        existing = self._service_point_by_id(service_point_id)
        if action == "delete":
            before = len(self._service_points)
            self._service_points = [
                point for point in self._service_points if point.service_point_id != service_point_id
            ]
            return {"ok": True, "action": action, "entity": "service_point", "deleted": len(self._service_points) < before}
        if action in {"enable", "disable"}:
            if existing is None:
                return {"ok": False, "reason": "service_point_not_found", "service_point_id": service_point_id}
            item = {**existing.to_payload(), "enabled": action == "enable"}
        service_point = ServicePoint.from_dict(item)
        if not service_point.point_id:
            return {"ok": False, "reason": "service_point_requires_point_id"}
        if self._point_by_id(service_point.point_id) is None:
            return {
                "ok": False,
                "reason": "service_point_point_not_found",
                "point_id": service_point.point_id,
            }
        self._service_points = [
            candidate for candidate in self._service_points if candidate.service_point_id != service_point.service_point_id
        ]
        self._service_points.append(service_point)
        return {
            "ok": True,
            "action": action,
            "entity": "service_point",
            "service_point": service_point.to_payload(),
        }

    def _manage_route(self, *, action: str, item: dict[str, Any]) -> dict[str, Any]:
        route_id = str(item.get("route_id") or item.get("id") or "").strip()
        if not route_id:
            return {"ok": False, "reason": "route_id_required"}
        existing = self._route_by_id(route_id)
        if action == "delete":
            before = len(self._routes)
            self._routes = [route for route in self._routes if route.route_id != route_id]
            return {"ok": True, "action": action, "entity": "route", "deleted": len(self._routes) < before}
        if action in {"enable", "disable"}:
            if existing is None:
                return {"ok": False, "reason": "route_not_found", "route_id": route_id}
            item = {**existing.to_payload(), "enabled": action == "enable"}
        route = GuideRoute.from_dict(item)
        if not route.from_point_id or not route.to_point_id:
            return {"ok": False, "reason": "route_requires_from_and_to"}
        missing_point_ids = [
            point_id
            for point_id in (route.from_point_id, route.to_point_id)
            if self._point_by_id(point_id) is None
        ]
        if missing_point_ids:
            return {
                "ok": False,
                "reason": "route_point_not_found",
                "missing_point_ids": missing_point_ids,
            }
        self._routes = [candidate for candidate in self._routes if candidate.route_id != route.route_id]
        self._routes.append(route)
        return {"ok": True, "action": action, "entity": "route", "route": route.to_payload()}

    def _point_references(self, point_id: str) -> list[dict[str, str]]:
        references: list[dict[str, str]] = []
        for service_point in self._service_points:
            if service_point.point_id == point_id:
                references.append(
                    {
                        "entity": "service_point",
                        "id": service_point.service_point_id,
                    }
                )
        for route in self._routes:
            if route.from_point_id == point_id or route.to_point_id == point_id:
                references.append(
                    {
                        "entity": "route",
                        "id": route.route_id,
                    }
                )
        return references

    def _service_point_by_id(self, service_point_id: str) -> ServicePoint | None:
        for service_point in self._service_points:
            if service_point.service_point_id == service_point_id:
                return service_point
        return None

    def _route_by_id(self, route_id: str) -> GuideRoute | None:
        for route in self._routes:
            if route.route_id == route_id:
                return route
        return None

    def _point_by_id(self, point_id: str) -> ParkPoint | None:
        for point in self._points:
            if point.point_id == point_id:
                return point
        return None

    def _proposal_by_id(self, proposal_id: str) -> dict[str, Any] | None:
        for proposal in self._pending_changes:
            if str(proposal.get("proposal_id") or "") == proposal_id:
                return proposal
        return None

    def _record_interaction(
        self,
        *,
        event_type: str,
        status: str,
        body: dict[str, Any],
        result: dict[str, Any],
        service_point: ServicePoint | None = None,
        point: ParkPoint | None = None,
        route: GuideRoute | None = None,
    ) -> str:
        service_point_id = str(
            body.get("service_point_id")
            or body.get("help_point_id")
            or (service_point.service_point_id if service_point else "")
        ).strip()
        result_point = result.get("point") if isinstance(result.get("point"), dict) else {}
        destination_point_id = str(
            (point.point_id if point else "")
            or result.get("destination_point_id")
            or result_point.get("point_id")
            or ""
        ).strip()
        interaction = {
            "interaction_id": f"space-interaction-{int(time.time() * 1000)}-{len(self._interactions) + 1}",
            "timestamp": time.time(),
            "event_type": str(event_type or ""),
            "status": str(status or ""),
            "park_id": self.park_id,
            "service_point_id": service_point_id,
            "destination_point_id": destination_point_id,
            "route_id": route.route_id if route else str(result.get("route_id") or ""),
            "query": str(body.get("query") or body.get("destination") or body.get("text") or "").strip(),
            "reason": str(result.get("reason") or result.get("match_reason") or ""),
            "guide_mode": str(result.get("mode") or body.get("guide_mode") or body.get("mode") or ""),
            "speech_text": str(result.get("speech_text") or result.get("reply") or ""),
            "operator_id": str(body.get("operator_id") or body.get("actor_id") or "").strip(),
            "requires_confirmation": bool(result.get("requires_confirmation", False)),
            "runtime_handoff_ready": bool(result.get("runtime_handoff_ready", False)),
        }
        with self._store_lock:
            self._interactions.append(interaction)
            self._interactions = self._interactions[-1000:]
            if self._store_path is not None:
                self._persist()
        return interaction["interaction_id"]

    def _persist(self) -> dict[str, Any]:
        if self._store_path is None:
            return {"written": False, "reason": "space_store_path_not_configured"}
        payload = {
            "park_id": self.park_id,
            "updated_at": time.time(),
            "revision": self._revision,
            "points": [point.to_payload() for point in self._points],
            "service_points": [point.to_payload() for point in self._service_points],
            "routes": [route.to_payload() for route in self._routes],
            "change_log": self._change_log,
            "snapshots": self._snapshots,
            "pending_changes": self._pending_changes,
            "interactions": self._interactions,
        }
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
                newline="\n",
            )
            tmp_path.replace(self._store_path)
        except OSError as exc:
            return {"written": False, "path": str(self._store_path), "error": str(exc)}
        return {"written": True, "path": str(self._store_path)}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_store_path(value: Any, *, project_root: Any = None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    root = Path(str(project_root)) if project_root else Path.cwd()
    return root / path


def _load_store(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_points(base: list[ParkPoint], overlay: list[ParkPoint]) -> list[ParkPoint]:
    by_id = {point.point_id: point for point in base if point.point_id}
    for point in overlay:
        if point.point_id:
            by_id[point.point_id] = point
    return list(by_id.values())


def _merge_service_points(base: list[ServicePoint], overlay: list[ServicePoint]) -> list[ServicePoint]:
    by_id = {point.service_point_id: point for point in base if point.service_point_id}
    for point in overlay:
        if point.service_point_id:
            by_id[point.service_point_id] = point
    return list(by_id.values())


def _merge_routes(base: list[GuideRoute], overlay: list[GuideRoute]) -> list[GuideRoute]:
    by_id = {route.route_id: route for route in base if route.route_id}
    for route in overlay:
        if route.route_id:
            by_id[route.route_id] = route
    return list(by_id.values())


def _managed_item_id(*, entity: str, item: dict[str, Any], result: dict[str, Any]) -> str:
    if entity == "point":
        point = result.get("point") if isinstance(result.get("point"), dict) else {}
        return str(point.get("point_id") or item.get("point_id") or item.get("id") or "").strip()
    if entity == "service_point":
        service_point = result.get("service_point") if isinstance(result.get("service_point"), dict) else {}
        return str(
            service_point.get("service_point_id")
            or item.get("service_point_id")
            or item.get("id")
            or ""
        ).strip()
    route = result.get("route") if isinstance(result.get("route"), dict) else {}
    return str(route.get("route_id") or item.get("route_id") or item.get("id") or "").strip()


def _proposal_id(existing: list[dict[str, Any]]) -> str:
    suffix = len(existing) + 1
    return f"space-proposal-{int(time.time() * 1000)}-{suffix}"


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
                greeting_prompt=str(zone.get("greeting_prompt") or _DEFAULT_GREETING),
                supported_intents=("wayfinding", "escort"),
                enabled=bool(zone.get("enabled", True)),
            )
        )
    return service_points
