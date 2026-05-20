"""Adapters that feed perception/runtime signals into cognition world state."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

from askme.cognition.world import WorldStateService
from askme.constants import CHANGE_EVENTS_PATH
from askme.schemas.events import ChangeEvent
from askme.schemas.messages import CmsState, DetectionFrame, EstopState

_TOPIC_DETECTIONS = "/thunder/detections"
_TOPIC_ESTOP = "/thunder/estop"
_TOPIC_CMS_STATE = "/thunder/cms_state"
_TIMESTAMP_FIELDS = ("observed_at", "timestamp", "last_seen", "first_seen", "_ts")


class CognitionPerceptionSync:
    """Synchronize existing perception/runtime signals into WorldStateService."""

    def __init__(
        self,
        world_state: WorldStateService,
        *,
        event_file: str | Path | None = None,
        scene_stale_after_s: float = 3.0,
        robot_stale_after_s: float = 5.0,
        max_event_lines_per_sync: int = 100,
    ) -> None:
        self.world_state = world_state
        self.event_file = Path(event_file or CHANGE_EVENTS_PATH)
        self.scene_stale_after_s = max(0.1, float(scene_stale_after_s))
        self.robot_stale_after_s = max(0.1, float(robot_stale_after_s))
        self.max_event_lines_per_sync = max(1, int(max_event_lines_per_sync))
        self._event_offset = 0
        self.last_sync_at = 0.0
        self.synced_event_count = 0
        self.last_errors: list[str] = []

    async def sync_once(
        self,
        *,
        pulse_bus: Any | None = None,
        perception_world_state: Any | None = None,
    ) -> dict[str, Any]:
        """Run one best-effort sync pass and return diagnostics."""
        self.last_errors = []
        topics = self.sync_from_pulse(pulse_bus)
        event_count = self.sync_from_event_file()
        perception_payload = await self.sync_from_perception_world_state(perception_world_state)
        self.last_sync_at = time.time()
        snapshot = self.world_state.snapshot()
        return {
            "synced": True,
            "last_sync_at": self.last_sync_at,
            "pulse_topics": topics,
            "event_count": event_count,
            "synced_event_count": self.synced_event_count,
            "perception_world_state": perception_payload,
            "fresh_object_count": len(snapshot["scene"]["objects"]),
            "stale_keys": snapshot["stale_keys"],
            "errors": list(self.last_errors),
        }

    def sync_from_pulse(self, pulse_bus: Any | None) -> list[str]:
        if pulse_bus is None or not callable(getattr(pulse_bus, "get_latest", None)):
            return []

        synced: list[str] = []
        detections = _safe_latest(pulse_bus, _TOPIC_DETECTIONS, self.last_errors)
        if detections is not None:
            self._sync_detection_frame(detections)
            synced.append(_TOPIC_DETECTIONS)

        estop = _safe_latest(pulse_bus, _TOPIC_ESTOP, self.last_errors)
        if estop is not None:
            self._sync_estop(estop)
            synced.append(_TOPIC_ESTOP)

        cms_state = _safe_latest(pulse_bus, _TOPIC_CMS_STATE, self.last_errors)
        if cms_state is not None:
            self._sync_cms_state(cms_state)
            synced.append(_TOPIC_CMS_STATE)

        return synced

    def sync_from_event_file(self) -> int:
        path = self.event_file
        if not path.is_file():
            return 0
        try:
            size = path.stat().st_size
            if size < self._event_offset:
                self._event_offset = 0
            count = 0
            with path.open(encoding="utf-8") as fh:
                fh.seek(self._event_offset)
                while count < self.max_event_lines_per_sync:
                    line = fh.readline()
                    if not line:
                        break
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        event = ChangeEvent.from_dict(json.loads(raw))
                    except Exception as exc:
                        self.last_errors.append(f"change_event_parse:{exc}")
                        continue
                    event_ts = float(getattr(event, "timestamp", 0.0) or 0.0)
                    if event_ts and (time.time() - event_ts) > self.scene_stale_after_s:
                        continue
                    self.world_state.apply_change_event(
                        event,
                        stale_after_s=self.scene_stale_after_s,
                    )
                    count += 1
                self._event_offset = fh.tell()
            self.synced_event_count += count
            return count
        except Exception as exc:
            self.last_errors.append(f"change_event_file:{exc}")
            return 0

    async def sync_from_perception_world_state(self, perception_world_state: Any | None) -> dict[str, Any]:
        if perception_world_state is None:
            return {"synced": False, "reason": "not_configured"}

        try:
            snapshot_func = getattr(perception_world_state, "snapshot", None)
            if callable(snapshot_func):
                result = snapshot_func()
                snapshot = await result if inspect.isawaitable(result) else result
            else:
                snapshot_sync = getattr(perception_world_state, "snapshot_sync", None)
                if not callable(snapshot_sync):
                    return {"synced": False, "reason": "snapshot_unavailable"}
                snapshot = snapshot_sync()
            scene_payload = normalize_scene_snapshot(snapshot)
            if not scene_payload["valid"]:
                reason = str(scene_payload["reason"])
                self.last_errors.append(f"perception_world_state:{reason}")
                return {"synced": False, "reason": reason}
            objects = scene_payload["objects"]
            observed_at = scene_payload["observed_at"]
            self.world_state.update_scene(
                summary=str(snapshot.get("summary", "")),
                objects=objects,
                source="perception_world_state",
                observed_at=observed_at,
                stale_after_s=self.scene_stale_after_s,
            )
            return {
                "synced": True,
                "object_count": len(objects),
                "observed_at": observed_at,
            }
        except Exception as exc:
            self.last_errors.append(f"perception_world_state:{exc}")
            return {"synced": False, "reason": str(exc)}

    def _sync_detection_frame(self, payload: dict[str, Any]) -> None:
        frame = DetectionFrame.from_dict(payload)
        observed_at = frame.timestamp or float(payload.get("_ts", 0.0) or time.time())
        objects = [
            {
                "label": det.class_id,
                "class_id": det.class_id,
                "confidence": det.confidence,
                "bbox": list(det.bbox),
                "distance_m": det.distance_m,
            }
            for det in frame.detections
        ]
        self.world_state.update_scene(
            objects=objects,
            source="pulse:detections",
            observed_at=observed_at,
            stale_after_s=self.scene_stale_after_s,
        )

    def _sync_estop(self, payload: dict[str, Any]) -> None:
        state = EstopState.from_dict(payload)
        observed_at = state.timestamp or float(payload.get("_ts", 0.0) or time.time())
        self.world_state.update_robot_state(
            {
                "estop_active": state.active,
                "estop_timestamp": observed_at,
            },
            source="pulse:estop",
            observed_at=observed_at,
            stale_after_s=self.robot_stale_after_s,
        )

    def _sync_cms_state(self, payload: dict[str, Any]) -> None:
        state = CmsState.from_dict(payload)
        observed_at = state.timestamp or float(payload.get("_ts", 0.0) or time.time())
        self.world_state.update_robot_state(
            {
                "cms_state": state.state,
                "cms_addr": state.addr,
                "cms_timestamp": observed_at,
            },
            source="pulse:cms_state",
            observed_at=observed_at,
            stale_after_s=self.robot_stale_after_s,
        )


def normalize_scene_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate and normalize a perception scene snapshot without mutating state."""
    if not isinstance(snapshot, dict):
        return {"valid": False, "reason": "snapshot_non_object"}
    if "objects" not in snapshot:
        return {"valid": False, "reason": "snapshot_objects_missing"}
    raw_objects = snapshot.get("objects")
    if not isinstance(raw_objects, list):
        return {"valid": False, "reason": "snapshot_objects_not_list"}
    if _timestamp_invalid(snapshot):
        return {"valid": False, "reason": "snapshot_timestamp_invalid"}
    objects: list[dict[str, Any]] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        if _timestamp_invalid(item):
            return {"valid": False, "reason": "snapshot_object_timestamp_invalid"}
        objects.append(item)
    return {
        "valid": True,
        "objects": objects,
        "observed_at": _snapshot_observed_at(snapshot, objects),
    }


def _safe_latest(pulse_bus: Any, topic: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = pulse_bus.get_latest(topic)
    except Exception as exc:
        errors.append(f"pulse:{topic}:{exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _snapshot_observed_at(snapshot: dict[str, Any], objects: list[dict[str, Any]]) -> float | None:
    """Return the freshest observation timestamp from a perception snapshot."""
    direct = _float_timestamp(
        snapshot.get("observed_at")
        or snapshot.get("timestamp")
        or snapshot.get("last_seen")
    )
    if direct is not None:
        return direct

    latest: float | None = None
    for item in objects:
        ts = _float_timestamp(
            item.get("observed_at")
            or item.get("last_seen")
            or item.get("timestamp")
        )
        if ts is None:
            continue
        latest = ts if latest is None else max(latest, ts)
    return latest


def _timestamp_invalid(payload: dict[str, Any]) -> bool:
    for field in _TIMESTAMP_FIELDS:
        if field not in payload:
            continue
        value = payload.get(field)
        if value in (None, ""):
            continue
        if _float_timestamp(value) is None:
            return True
    return False


def _float_timestamp(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
