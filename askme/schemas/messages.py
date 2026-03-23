"""Typed message dataclasses for Pulse bus topics.

Each message has ``from_dict`` (parse raw dict) and ``to_dict`` (serialize back).
These replace the raw dicts that Pulse returns for well-known topics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self", bound="EstopState")  # type: ignore[misc,assignment]

from askme.schemas.observation import Detection


@dataclass(frozen=True)
class EstopState:
    """Emergency stop state from ``/thunder/estop``."""

    active: bool
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EstopState:
        return cls(
            active=bool(d.get("active", False)),
            timestamp=float(d.get("_ts", d.get("timestamp", 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"active": self.active, "timestamp": self.timestamp}


@dataclass(frozen=True)
class DetectionFrame:
    """A frame of YOLO detections from ``/thunder/detections``."""

    timestamp: float
    frame_id: int
    detections: list[Detection]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionFrame:
        raw_dets = d.get("detections", [])
        dets: list[Detection] = []
        for rd in raw_dets:
            bbox_raw = rd.get("bbox", [0, 0, 0, 0])
            bbox = tuple(bbox_raw) if len(bbox_raw) == 4 else (0, 0, 0, 0)
            dets.append(Detection(
                class_id=rd.get("class_id", rd.get("label", "")),
                confidence=float(rd.get("confidence", 0.0)),
                bbox=bbox,
                distance_m=rd.get("distance_m"),
            ))
        return cls(
            timestamp=float(d.get("timestamp", d.get("_ts", 0.0))),
            frame_id=int(d.get("frame_id", 0)),
            detections=dets,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detections": [
                {
                    "class_id": det.class_id,
                    "confidence": det.confidence,
                    "bbox": list(det.bbox),
                    "distance_m": det.distance_m,
                }
                for det in self.detections
            ],
        }


@dataclass(frozen=True)
class JointStateSnapshot:
    """Joint state snapshot from ``/thunder/joint_states``."""

    name: list[str]
    position: list[float]
    velocity: list[float]
    effort: list[float]
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JointStateSnapshot:
        return cls(
            name=list(d.get("name", [])),
            position=[float(v) for v in d.get("position", [])],
            velocity=[float(v) for v in d.get("velocity", [])],
            effort=[float(v) for v in d.get("effort", [])],
            timestamp=float(d.get("_ts", d.get("timestamp", 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "velocity": self.velocity,
            "effort": self.effort,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ImuSnapshot:
    """IMU reading from ``/thunder/imu``."""

    angular_velocity: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # x, y, z, w
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImuSnapshot:
        av = d.get("angular_velocity", {})
        ori = d.get("orientation", {})
        return cls(
            angular_velocity=(
                float(av.get("x", 0.0)),
                float(av.get("y", 0.0)),
                float(av.get("z", 0.0)),
            ),
            orientation=(
                float(ori.get("x", 0.0)),
                float(ori.get("y", 0.0)),
                float(ori.get("z", 0.0)),
                float(ori.get("w", 0.0)),
            ),
            timestamp=float(d.get("_ts", d.get("timestamp", 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "angular_velocity": {
                "x": self.angular_velocity[0],
                "y": self.angular_velocity[1],
                "z": self.angular_velocity[2],
            },
            "orientation": {
                "x": self.orientation[0],
                "y": self.orientation[1],
                "z": self.orientation[2],
                "w": self.orientation[3],
            },
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CmsState:
    """CMS (brainstem) connection/FSM state from ``/thunder/cms_state``."""

    state: str  # "connected", "disconnected", "Grounded", "Standing", etc.
    addr: str = ""
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CmsState:
        return cls(
            state=str(d.get("state", "unknown")),
            addr=str(d.get("addr", "")),
            timestamp=float(d.get("_ts", d.get("timestamp", 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"state": self.state, "timestamp": self.timestamp}
        if self.addr:
            d["addr"] = self.addr
        return d
