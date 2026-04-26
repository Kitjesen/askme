"""WorldState — live snapshot of the observed scene.

Maintained by applying ChangeEvents from ChangeDetector.
Provides thread-safe access to tracked objects and a Chinese-language
summary for LLM context injection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from askme.schemas.events import ChangeEvent, ChangeEventType

logger = logging.getLogger(__name__)

# Chinese class name translations for get_summary()
_CLASS_ZH: dict[str, str] = {
    "person": "人",
    "chair": "椅子",
    "car": "汽车",
    "bicycle": "自行车",
    "motorcycle": "摩托车",
    "dog": "狗",
    "cat": "猫",
    "bottle": "瓶子",
    "cup": "杯子",
    "laptop": "笔记本电脑",
    "cell phone": "手机",
    "backpack": "背包",
    "suitcase": "行李箱",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "book": "书",
    "clock": "时钟",
    "tv": "电视",
    "keyboard": "键盘",
    "mouse": "鼠标",
}


@dataclass
class TrackedObject:
    """A currently-visible object maintained by WorldState."""

    track_id: str
    class_id: str
    confidence: float
    bbox: tuple[float, float, float, float] | None
    first_seen: float           # epoch seconds
    last_seen: float            # epoch seconds
    distance_m: float | None = None

    @property
    def duration_s(self) -> float:
        return self.last_seen - self.first_seen

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_s": round(self.duration_s, 1),
        }
        if self.bbox:
            d["bbox"] = [round(v, 1) for v in self.bbox]
        if self.distance_m is not None:
            d["distance_m"] = round(self.distance_m, 2)
        return d


class WorldState:
    """Thread-safe live snapshot of tracked objects in the scene.

    Updated by calling ``apply_event()`` for each ChangeEvent.
    All public methods are safe to call from any asyncio task or thread.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        # track_id → TrackedObject (primary store)
        self._objects: dict[str, TrackedObject] = {}
        # Auto-generated track_id for events that lack one
        self._id_counter: int = 0
        # Recent event history (capped at 100)
        self._event_history: list[ChangeEvent] = []
        self._max_history: int = 100

    # ------------------------------------------------------------------
    # Event application
    # ------------------------------------------------------------------

    async def apply_event(self, event: ChangeEvent) -> None:
        """Update scene state based on a ChangeEvent (async, lock-safe)."""
        async with self._lock:
            self._record_event(event)
            if event.event_type in (
                ChangeEventType.PERSON_APPEARED,
                ChangeEventType.OBJECT_APPEARED,
            ):
                self._add_object(event)
            elif event.event_type in (
                ChangeEventType.PERSON_LEFT,
                ChangeEventType.OBJECT_DISAPPEARED,
            ):
                self._remove_object(event)
            elif event.event_type == ChangeEventType.COUNT_CHANGED:
                self._handle_count_changed(event)

    def apply_event_sync(self, event: ChangeEvent) -> None:
        """Synchronous variant for use in non-async contexts (no locking)."""
        self._record_event(event)
        if event.event_type in (
            ChangeEventType.PERSON_APPEARED,
            ChangeEventType.OBJECT_APPEARED,
        ):
            self._add_object(event)
        elif event.event_type in (
            ChangeEventType.PERSON_LEFT,
            ChangeEventType.OBJECT_DISAPPEARED,
        ):
            self._remove_object(event)
        elif event.event_type == ChangeEventType.COUNT_CHANGED:
            self._handle_count_changed(event)

    def _add_object(self, event: ChangeEvent) -> None:
        track_id = event.track_id or self._next_id()
        now = event.timestamp
        if track_id in self._objects:
            # Already tracked — update last_seen
            obj = self._objects[track_id]
            obj.last_seen = now
            obj.confidence = event.confidence
            if event.bbox:
                obj.bbox = event.bbox
            if event.distance_m is not None:
                obj.distance_m = event.distance_m
        else:
            self._objects[track_id] = TrackedObject(
                track_id=track_id,
                class_id=event.subject_class,
                confidence=event.confidence,
                bbox=event.bbox,
                first_seen=now,
                last_seen=now,
                distance_m=event.distance_m,
            )

    def _remove_object(self, event: ChangeEvent) -> None:
        track_id = event.track_id
        if track_id and track_id in self._objects:
            del self._objects[track_id]
        else:
            # Remove by class_id (oldest match)
            targets = [
                tid for tid, obj in self._objects.items()
                if obj.class_id == event.subject_class
            ]
            if targets:
                # Remove the earliest-seen one
                oldest = min(targets, key=lambda t: self._objects[t].first_seen)
                del self._objects[oldest]

    def _handle_count_changed(self, event: ChangeEvent) -> None:
        """For count changes, just record the event. No object tracking needed."""
        pass  # history already recorded in _record_event

    def _record_event(self, event: ChangeEvent) -> None:
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"auto_{self._id_counter:04d}"

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    async def get_objects(self) -> dict[str, TrackedObject]:
        """Return a shallow copy of the current tracked objects dict."""
        async with self._lock:
            return dict(self._objects)

    def get_objects_sync(self) -> dict[str, TrackedObject]:
        """Synchronous accessor — no lock, use only in single-threaded tests."""
        return dict(self._objects)

    async def get_persons(self) -> list[TrackedObject]:
        """Return all currently tracked persons."""
        async with self._lock:
            return [obj for obj in self._objects.values() if obj.class_id == "person"]

    def get_persons_sync(self) -> list[TrackedObject]:
        """Synchronous variant of get_persons()."""
        return [obj for obj in self._objects.values() if obj.class_id == "person"]

    async def get_summary(self) -> str:
        """Return a Chinese-language summary of the current scene for LLM context."""
        async with self._lock:
            return self._build_summary()

    def get_summary_sync(self) -> str:
        """Synchronous variant of get_summary()."""
        return self._build_summary()

    def _build_summary(self) -> str:
        if not self._objects:
            return "当前场景无检测目标。"

        # Group by class
        by_class: dict[str, list[TrackedObject]] = {}
        for obj in self._objects.values():
            by_class.setdefault(obj.class_id, []).append(obj)

        now = time.time()
        parts: list[str] = []
        for class_id, objs in sorted(by_class.items()):
            name_zh = _CLASS_ZH.get(class_id, class_id)
            count = len(objs)

            # Include distance for persons if available
            if class_id == "person" and count == 1:
                obj = objs[0]
                dist_str = f"（距离{obj.distance_m:.1f}米）" if obj.distance_m else ""
                duration = now - obj.first_seen
                dur_str = f"，已持续{duration:.0f}秒" if duration > 5 else ""
                parts.append(f"1名{name_zh}{dist_str}{dur_str}")
            elif class_id == "person":
                parts.append(f"{count}名{name_zh}")
            else:
                parts.append(f"{count}个{name_zh}")

        scene_str = "、".join(parts)
        return f"当前场景：{scene_str}。"

    async def event_history(self) -> list[ChangeEvent]:
        """Return a copy of recent event history."""
        async with self._lock:
            return list(self._event_history)

    def event_history_sync(self) -> list[ChangeEvent]:
        """Synchronous variant."""
        return list(self._event_history)

    async def snapshot(self) -> dict[str, Any]:
        """Return full state snapshot as a dict (for logging/debugging)."""
        async with self._lock:
            return {
                "object_count": len(self._objects),
                "objects": [obj.to_dict() for obj in self._objects.values()],
                "summary": self._build_summary(),
            }
