"""Observation — structured representation of a single perception frame.

Mirrors the frame_daemon JSON output for zero-cost parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    """Single detected object in a frame."""

    class_id: str                                    # COCO class name ("person", "chair")
    confidence: float                                # 0.0–1.0
    bbox: tuple[float, float, float, float]          # (x1, y1, x2, y2) in original frame coords
    distance_m: float | None = None                  # depth enrichment (meters), None if unavailable

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area(self) -> float:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


@dataclass
class Observation:
    """One perception frame — all detections at a single point in time."""

    timestamp: float                                 # epoch seconds from daemon
    detections: list[Detection] = field(default_factory=list)
    infer_ms: float = 0.0                            # BPU inference time
    source: str = "frame_daemon"

    @classmethod
    def from_daemon_json(cls, data: dict[str, Any]) -> Observation:
        """Parse frame_daemon's /tmp/askme_frame_detections.json output."""
        dets = []
        for d in data.get("detections", []):
            bbox_raw = d.get("bbox", [0, 0, 0, 0])
            dets.append(Detection(
                class_id=d.get("class_id", ""),
                confidence=d.get("confidence", 0.0),
                bbox=tuple(bbox_raw) if len(bbox_raw) == 4 else (0, 0, 0, 0),
                distance_m=d.get("distance_m"),
            ))
        return cls(
            timestamp=data.get("timestamp", 0.0),
            detections=dets,
            infer_ms=data.get("infer_ms", 0.0),
        )

    def by_class(self) -> dict[str, list[Detection]]:
        """Group detections by class_id."""
        groups: dict[str, list[Detection]] = {}
        for d in self.detections:
            groups.setdefault(d.class_id, []).append(d)
        return groups

    def count(self, class_id: str) -> int:
        return sum(1 for d in self.detections if d.class_id == class_id)
