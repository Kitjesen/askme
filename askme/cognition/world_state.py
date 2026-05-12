"""Timestamped world-state cache for robot-aware planning.

This module is intentionally local and deterministic. It stores what askme
currently believes about the robot, scene, task, and safety context, but it
does not own perception processing or physical control.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorldFact:
    """One observed fact with freshness and confidence metadata."""

    key: str
    value: Any
    source: str = "unknown"
    confidence: float = 1.0
    observed_at: float = field(default_factory=time.time)
    stale_after_s: float | None = 10.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_stale(self, now: float | None = None) -> bool:
        if self.stale_after_s is None:
            return False
        return ((now if now is not None else time.time()) - self.observed_at) > self.stale_after_s

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.observed_at)

    def to_dict(self, *, now: float | None = None) -> dict[str, Any]:
        current = now if now is not None else time.time()
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "age_s": round(self.age_s(current), 3),
            "stale": self.is_stale(current),
            "stale_after_s": self.stale_after_s,
            "metadata": dict(self.metadata),
        }


class WorldStateService:
    """Small in-memory world model used by the voice/cognition adapter."""

    def __init__(
        self,
        *,
        max_facts: int = 200,
        max_events: int = 100,
        default_stale_after_s: float = 10.0,
    ) -> None:
        self.max_facts = max(1, int(max_facts))
        self.default_stale_after_s = max(0.1, float(default_stale_after_s))
        self._facts: dict[str, WorldFact] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_events)))

    def update_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str = "unknown",
        confidence: float = 1.0,
        observed_at: float | None = None,
        stale_after_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorldFact:
        cleaned_key = str(key).strip()
        if not cleaned_key:
            raise ValueError("world fact key is required")
        fact = WorldFact(
            key=cleaned_key,
            value=value,
            source=str(source or "unknown"),
            confidence=min(max(float(confidence), 0.0), 1.0),
            observed_at=observed_at if observed_at is not None else time.time(),
            stale_after_s=self.default_stale_after_s if stale_after_s is None else stale_after_s,
            metadata=dict(metadata or {}),
        )
        self._facts[cleaned_key] = fact
        self._trim_facts()
        return fact

    def update_scene(
        self,
        *,
        summary: str = "",
        objects: list[dict[str, Any]] | None = None,
        source: str = "vision",
        observed_at: float | None = None,
        stale_after_s: float | None = 2.0,
    ) -> None:
        timestamp = observed_at if observed_at is not None else time.time()
        normalized_objects = [_normalize_object(item) for item in objects or []]
        resolved_summary = str(summary or "").strip() or _summarize_objects(normalized_objects)
        self.update_fact(
            "scene.summary",
            resolved_summary,
            source=source,
            observed_at=timestamp,
            stale_after_s=stale_after_s,
        )
        self.update_fact(
            "scene.objects",
            normalized_objects,
            source=source,
            confidence=_max_object_confidence(normalized_objects),
            observed_at=timestamp,
            stale_after_s=stale_after_s,
        )

    def upsert_scene_object(
        self,
        item: dict[str, Any],
        *,
        source: str = "perception",
        observed_at: float | None = None,
        stale_after_s: float | None = 3.0,
    ) -> None:
        """Insert or update one scene object while preserving other fresh objects."""
        timestamp = observed_at if observed_at is not None else time.time()
        obj = _normalize_object(item)
        current = self.fresh_objects()
        identity = _object_identity(obj)
        replaced = False
        next_objects: list[dict[str, Any]] = []
        for existing in current:
            if _object_identity(existing) == identity:
                next_objects.append(obj)
                replaced = True
            else:
                next_objects.append(existing)
        if not replaced:
            next_objects.append(obj)
        self.update_scene(
            objects=next_objects,
            source=source,
            observed_at=timestamp,
            stale_after_s=stale_after_s,
        )

    def remove_scene_object(
        self,
        *,
        label: str = "",
        track_id: str = "",
        source: str = "perception",
        observed_at: float | None = None,
        stale_after_s: float | None = 3.0,
    ) -> None:
        """Remove one scene object by track id, or the best matching label."""
        timestamp = observed_at if observed_at is not None else time.time()
        current = self.fresh_objects()
        target_track = str(track_id or "").strip()
        target_label = str(label or "").strip()
        removed = False
        next_objects: list[dict[str, Any]] = []
        for existing in current:
            existing_track = str(existing.get("track_id", "")).strip()
            existing_label = str(
                existing.get("label")
                or existing.get("class_id")
                or existing.get("class")
                or ""
            ).strip()
            if target_track and existing_track == target_track:
                removed = True
                continue
            if not target_track and target_label and existing_label == target_label and not removed:
                removed = True
                continue
            next_objects.append(existing)
        self.update_scene(
            objects=next_objects,
            source=source,
            observed_at=timestamp,
            stale_after_s=stale_after_s,
        )

    def apply_change_event(
        self,
        event: Any,
        *,
        source: str | None = None,
        stale_after_s: float | None = 3.0,
    ) -> None:
        """Apply a perception ChangeEvent without taking ownership of perception."""
        event_type = _event_type_value(getattr(event, "event_type", ""))
        observed_at = float(getattr(event, "timestamp", 0.0) or time.time())
        subject = str(getattr(event, "subject_class", "") or "")
        event_source = source or str(getattr(event, "source", "change_detector") or "change_detector")
        payload = event.to_dict() if hasattr(event, "to_dict") else {
            "event_type": event_type,
            "timestamp": observed_at,
            "subject_class": subject,
        }
        self.record_event(
            event_type,
            payload if isinstance(payload, dict) else {},
            source=event_source,
            observed_at=observed_at,
        )

        if event_type in {"person_appeared", "object_appeared"}:
            self.upsert_scene_object(
                {
                    "label": subject,
                    "class_id": subject,
                    "confidence": float(getattr(event, "confidence", 0.0) or 0.0),
                    "bbox": getattr(event, "bbox", None),
                    "track_id": getattr(event, "track_id", ""),
                    "distance_m": getattr(event, "distance_m", None),
                },
                source=event_source,
                observed_at=observed_at,
                stale_after_s=stale_after_s,
            )
        elif event_type in {"person_left", "object_disappeared"}:
            self.remove_scene_object(
                label=subject,
                track_id=str(getattr(event, "track_id", "") or ""),
                source=event_source,
                observed_at=observed_at,
                stale_after_s=stale_after_s,
            )

    def update_robot_state(
        self,
        state: dict[str, Any],
        *,
        source: str = "runtime",
        stale_after_s: float | None = 5.0,
    ) -> None:
        for key, value in dict(state).items():
            self.update_fact(
                f"robot.{key}",
                value,
                source=source,
                stale_after_s=stale_after_s,
            )

    def update_area_catalog(
        self,
        areas: list[dict[str, Any]],
        *,
        source: str = "catalog",
        map_id: str = "",
        map_version: str = "",
        stale_after_s: float | None = 300.0,
    ) -> None:
        """Publish known operational areas for planning and runtime preflight."""
        normalized = [
            _normalize_area(item, default_map_id=map_id, default_map_version=map_version)
            for item in areas
            if isinstance(item, dict)
        ]
        self.update_fact(
            "environment.areas",
            normalized,
            source=source,
            stale_after_s=stale_after_s,
        )

    def update_device_catalog(
        self,
        devices: list[dict[str, Any]],
        *,
        source: str = "catalog",
        stale_after_s: float | None = 300.0,
    ) -> None:
        """Publish known devices/assets for inspection planning."""
        normalized = [
            _normalize_device(item)
            for item in devices
            if isinstance(item, dict)
        ]
        self.update_fact(
            "environment.devices",
            normalized,
            source=source,
            stale_after_s=stale_after_s,
        )

    def update_map_state(
        self,
        *,
        map_id: str,
        map_version: str = "",
        localized: bool | None = None,
        localization_quality: float | None = None,
        source: str = "nav",
        stale_after_s: float | None = 30.0,
    ) -> None:
        """Publish current localization/map facts from nav or simulator."""
        self.update_fact(
            "map.current_id",
            str(map_id or "").strip(),
            source=source,
            stale_after_s=stale_after_s,
        )
        if map_version:
            self.update_fact(
                "map.current_version",
                str(map_version or "").strip(),
                source=source,
                stale_after_s=stale_after_s,
            )
        if localized is not None:
            self.update_fact(
                "map.localized",
                bool(localized),
                source=source,
                stale_after_s=stale_after_s,
            )
        if localization_quality is not None:
            self.update_fact(
                "map.localization_quality",
                float(localization_quality),
                source=source,
                stale_after_s=stale_after_s,
            )

    def get_area(self, area_id: str, *, include_stale: bool = False) -> dict[str, Any] | None:
        target = str(area_id or "").strip().lower()
        if not target:
            return None
        fact = self.get_fact("environment.areas", include_stale=include_stale)
        if fact is None or not isinstance(fact.value, list):
            return None
        for item in fact.value:
            if not isinstance(item, dict):
                continue
            if str(item.get("area_id", "")).strip().lower() == target:
                return dict(item)
        return None

    def get_device(self, device_id: str, *, include_stale: bool = False) -> dict[str, Any] | None:
        target = str(device_id or "").strip().lower()
        if not target:
            return None
        fact = self.get_fact("environment.devices", include_stale=include_stale)
        if fact is None or not isinstance(fact.value, list):
            return None
        for item in fact.value:
            if not isinstance(item, dict):
                continue
            if str(item.get("device_id", "")).strip().lower() == target:
                return dict(item)
        return None

    def record_event(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        source: str = "runtime",
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        event = {
            "kind": str(kind or "event"),
            "payload": dict(payload or {}),
            "source": str(source or "runtime"),
            "observed_at": observed_at if observed_at is not None else time.time(),
        }
        self._events.append(event)
        return event

    def get_fact(self, key: str, *, include_stale: bool = False) -> WorldFact | None:
        fact = self._facts.get(key)
        if fact is None:
            return None
        if not include_stale and fact.is_stale():
            return None
        return fact

    def fresh_objects(self, *, max_age_s: float | None = None) -> list[dict[str, Any]]:
        fact = self.get_fact("scene.objects")
        if fact is None:
            return []
        if max_age_s is not None and fact.age_s() > max_age_s:
            return []
        if not isinstance(fact.value, list):
            return []
        return [dict(item) for item in fact.value if isinstance(item, dict)]

    def resolve_reference(self, text: str) -> dict[str, Any]:
        """Resolve deictic terms like "this" and "there" against fresh objects."""
        text_l = str(text or "").lower()
        has_reference = any(
            marker in text_l
            for marker in (
                "这个", "那个", "这里", "那里", "这边", "那边", "前面",
                "left", "right", "there", "this", "that",
            )
        )
        if not has_reference:
            return {"requires_reference": False, "resolved": None, "needs_clarification": False}

        objects = self.fresh_objects(max_age_s=3.0)
        if not objects:
            return {
                "requires_reference": True,
                "resolved": None,
                "needs_clarification": True,
                "reason": "no_fresh_scene_object",
            }

        best = max(
            objects,
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                -float(item.get("distance_m", 99.0) or 99.0),
            ),
        )
        return {
            "requires_reference": True,
            "resolved": best,
            "needs_clarification": False,
            "reason": "fresh_scene_object",
        }

    def snapshot(self, *, include_stale: bool = True) -> dict[str, Any]:
        now = time.time()
        scene_fact = self.get_fact("scene.objects", include_stale=include_stale)
        facts = [
            fact.to_dict(now=now)
            for fact in self._facts.values()
            if include_stale or not fact.is_stale(now)
        ]
        facts.sort(key=lambda item: item["key"])
        return {
            "updated_at": now,
            "fact_count": len(facts),
            "stale_keys": [item["key"] for item in facts if item["stale"]],
            "facts": facts,
            "scene": {
                "summary": _fact_value(self.get_fact("scene.summary")),
                "observed_at": scene_fact.observed_at if scene_fact is not None else None,
                "stale": scene_fact.is_stale(now) if scene_fact is not None else True,
                "objects": self.fresh_objects(),
            },
            "robot": self._group_prefix("robot.", include_stale=include_stale, now=now),
            "task": self._group_prefix("task.", include_stale=include_stale, now=now),
            "environment": self._group_prefix("environment.", include_stale=include_stale, now=now),
            "map": self._group_prefix("map.", include_stale=include_stale, now=now),
            "events": list(self._events),
        }

    def context_summary(self, *, max_objects: int = 5) -> str:
        scene_summary = _fact_value(self.get_fact("scene.summary")) or "unknown scene"
        objects = self.fresh_objects()[:max_objects]
        object_labels = [
            str(item.get("label") or item.get("class_id") or item.get("class") or "object")
            for item in objects
        ]
        robot = self._group_prefix("robot.", include_stale=False, now=time.time())
        parts = [f"scene={scene_summary}"]
        if object_labels:
            parts.append("objects=" + ", ".join(object_labels))
        if robot:
            parts.append("robot=" + ", ".join(f"{k}:{v}" for k, v in sorted(robot.items())))
        return "; ".join(parts)

    def _group_prefix(
        self,
        prefix: str,
        *,
        include_stale: bool,
        now: float,
    ) -> dict[str, Any]:
        grouped: dict[str, Any] = {}
        for key, fact in self._facts.items():
            if not key.startswith(prefix):
                continue
            if not include_stale and fact.is_stale(now):
                continue
            grouped[key.removeprefix(prefix)] = fact.value
        return grouped

    def _trim_facts(self) -> None:
        if len(self._facts) <= self.max_facts:
            return
        oldest = sorted(self._facts.values(), key=lambda fact: fact.observed_at)
        for fact in oldest[: len(self._facts) - self.max_facts]:
            self._facts.pop(fact.key, None)


def _normalize_object(item: dict[str, Any]) -> dict[str, Any]:
    label = item.get("label") or item.get("class_id") or item.get("class") or item.get("name")
    normalized = dict(item)
    normalized["label"] = str(label or "object")
    normalized["confidence"] = float(item.get("confidence", 0.0) or 0.0)
    if "distance_m" in item and item.get("distance_m") is not None:
        normalized["distance_m"] = float(item["distance_m"])
    if "bbox" in item and item.get("bbox") is not None:
        normalized["bbox"] = list(item["bbox"])
    if "track_id" in item and item.get("track_id"):
        normalized["track_id"] = str(item["track_id"])
    return normalized


def _normalize_area(
    item: dict[str, Any],
    *,
    default_map_id: str = "",
    default_map_version: str = "",
) -> dict[str, Any]:
    area_id = item.get("area_id") or item.get("id") or item.get("name")
    normalized = dict(item)
    normalized["area_id"] = str(area_id or "").strip().lower()
    normalized["name"] = str(item.get("name") or item.get("label") or normalized["area_id"])
    normalized["allowed"] = bool(item.get("allowed", True))
    normalized["status"] = str(item.get("status") or "active").strip().lower()
    normalized["risk_level"] = str(item.get("risk_level") or "medium").strip().lower()
    normalized["route_id"] = str(item.get("route_id") or "").strip()
    normalized["map_id"] = str(item.get("map_id") or default_map_id or "").strip()
    normalized["map_version"] = str(item.get("map_version") or default_map_version or "").strip()
    return normalized


def _normalize_device(item: dict[str, Any]) -> dict[str, Any]:
    device_id = item.get("device_id") or item.get("id") or item.get("name")
    normalized = dict(item)
    normalized["device_id"] = str(device_id or "").strip().lower()
    normalized["name"] = str(item.get("name") or item.get("label") or normalized["device_id"])
    normalized["area_id"] = str(item.get("area_id") or "").strip().lower()
    normalized["device_type"] = str(item.get("device_type") or item.get("type") or "asset").strip()
    normalized["inspection_skill"] = str(
        item.get("inspection_skill") or "inspect_equipment"
    ).strip()
    normalized["status"] = str(item.get("status") or "unknown").strip().lower()
    return normalized


def _max_object_confidence(objects: list[dict[str, Any]]) -> float:
    if not objects:
        return 0.0
    return max(float(item.get("confidence", 0.0) or 0.0) for item in objects)


def _fact_value(fact: WorldFact | None) -> Any:
    return None if fact is None else fact.value


def _object_identity(item: dict[str, Any]) -> str:
    track_id = str(item.get("track_id", "")).strip()
    if track_id:
        return f"track:{track_id}"
    label = str(item.get("label") or item.get("class_id") or item.get("class") or "object")
    bbox = item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        return f"label:{label}:{int(center_x // 80)}:{int(center_y // 80)}"
    return f"label:{label}"


def _event_type_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "")


def _summarize_objects(objects: list[dict[str, Any]]) -> str:
    if not objects:
        return "no fresh scene objects"
    counts: dict[str, int] = {}
    for item in objects:
        label = str(item.get("label") or item.get("class_id") or item.get("class") or "object")
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{label}:{count}" for label, count in sorted(counts.items()))
