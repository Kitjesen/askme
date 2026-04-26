"""ChangeEvent — discrete event emitted when the scene changes.

Events are produced by ChangeDetector and consumed by ProactiveAgent.
Designed for JSONL serialization and Phase 2 WorldState compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChangeEventType(str, Enum):  # noqa: UP042 — keep (str, Enum) for str() back-compat
    PERSON_APPEARED = "person_appeared"
    PERSON_LEFT = "person_left"
    OBJECT_APPEARED = "object_appeared"
    OBJECT_DISAPPEARED = "object_disappeared"
    COUNT_CHANGED = "count_changed"


# Importance weights by event type — person events are higher priority
_IMPORTANCE_WEIGHTS: dict[ChangeEventType, float] = {
    ChangeEventType.PERSON_APPEARED: 0.8,
    ChangeEventType.PERSON_LEFT: 0.7,
    ChangeEventType.OBJECT_APPEARED: 0.4,
    ChangeEventType.OBJECT_DISAPPEARED: 0.5,
    ChangeEventType.COUNT_CHANGED: 0.3,
}


@dataclass
class ChangeEvent:
    """A discrete scene change event."""

    event_type: ChangeEventType
    timestamp: float
    subject_class: str                                        # "person", "chair", etc.
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None     # where it happened
    prev_count: int = 0
    curr_count: int = 0
    importance: float = 0.0
    source: str = "change_detector"
    track_id: str = ""                                        # Phase 2: stable ID
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if self.importance == 0.0:
            self.importance = _IMPORTANCE_WEIGHTS.get(self.event_type, 0.3)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "subject_class": self.subject_class,
            "confidence": round(self.confidence, 3),
            "importance": round(self.importance, 2),
            "source": self.source,
        }
        if self.bbox:
            d["bbox"] = [round(v, 1) for v in self.bbox]
        if self.prev_count or self.curr_count:
            d["prev_count"] = self.prev_count
            d["curr_count"] = self.curr_count
        if self.track_id:
            d["track_id"] = self.track_id
        if self.distance_m is not None:
            d["distance_m"] = round(self.distance_m, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChangeEvent:
        bbox_raw = d.get("bbox")
        bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 else None
        return cls(
            event_type=ChangeEventType(d["event_type"]),
            timestamp=d.get("timestamp", time.time()),
            subject_class=d.get("subject_class", ""),
            confidence=d.get("confidence", 0.0),
            bbox=bbox,
            prev_count=d.get("prev_count", 0),
            curr_count=d.get("curr_count", 0),
            importance=d.get("importance", 0.0),
            source=d.get("source", "change_detector"),
            track_id=d.get("track_id", ""),
            distance_m=d.get("distance_m"),
        )

    @property
    def is_person_event(self) -> bool:
        return self.event_type in (
            ChangeEventType.PERSON_APPEARED,
            ChangeEventType.PERSON_LEFT,
        )

    def description_zh(self) -> str:
        """Human-readable Chinese description for TTS."""
        cls = self.subject_class
        dist = f"，距离{self.distance_m:.1f}米" if self.distance_m else ""
        if self.event_type == ChangeEventType.PERSON_APPEARED:
            return f"检测到有人出现{dist}"
        if self.event_type == ChangeEventType.PERSON_LEFT:
            return "人已离开"
        if self.event_type == ChangeEventType.OBJECT_APPEARED:
            return f"检测到新物体：{cls}{dist}"
        if self.event_type == ChangeEventType.OBJECT_DISAPPEARED:
            return f"{cls}消失了"
        if self.event_type == ChangeEventType.COUNT_CHANGED:
            return f"{cls}数量变化：{self.prev_count}→{self.curr_count}"
        return f"场景变化：{cls}"
