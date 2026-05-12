"""Provider adapter for real interaction-perception algorithm outputs.

External algorithms can be separate processes. This adapter gives them a
simple product contract: write fresh JSON facts for pose/gaze, gesture,
microphone-array DOA, audio-visual association, approach/dwell tracking, and
multi-person arbitration; askme will merge only fresh facts into the
InteractionGate snapshot.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SENSOR_KEYS = (
    "pose_gaze",
    "gesture",
    "sound_source",
    "audio_visual_association",
    "approach_dwell",
    "multi_person_arbitration",
)


@dataclass(frozen=True)
class InteractionProviderConfig:
    enabled: bool = False
    max_age_s: float = 2.0
    paths: dict[str, str] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> InteractionProviderConfig:
        if not isinstance(payload, dict):
            return cls()
        paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            max_age_s=max(0.1, float(payload.get("max_age_s", 2.0))),
            paths={str(key): str(value) for key, value in paths.items() if str(value).strip()},
        )


class FileInteractionPerceptionProvider:
    """Merge real algorithm JSON files into one gate-ready snapshot."""

    def __init__(self, config: InteractionProviderConfig | dict[str, Any] | None = None) -> None:
        self.config = (
            config
            if isinstance(config, InteractionProviderConfig)
            else InteractionProviderConfig.from_mapping(config)
        )
        self._paths = {key: Path(value) for key, value in (self.config.paths or {}).items()}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self._paths)

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        if not self.enabled:
            return {
                "source": "interaction_provider",
                "reason": "provider_disabled",
                "observed_at": None,
                "objects": [],
                "metadata": {"configured_sensors": sorted(self._paths)},
            }

        payloads: dict[str, dict[str, Any]] = {}
        freshness: dict[str, dict[str, Any]] = {}
        for key in _SENSOR_KEYS:
            raw = self._read_sensor_payload(key)
            if raw is None:
                freshness[key] = {"status": "missing", "age_s": None}
                continue
            observed_at = _timestamp(raw)
            age_s = None if observed_at is None else max(0.0, current - observed_at)
            fresh = bool(age_s is not None and age_s <= self.config.max_age_s)
            freshness[key] = {
                "status": "fresh" if fresh else "stale",
                "age_s": age_s,
                "observed_at": observed_at,
            }
            if fresh:
                payloads[key] = raw

        merged = self._merge(payloads, freshness=freshness, now=current)
        return merged

    def _read_sensor_payload(self, sensor_key: str) -> dict[str, Any] | None:
        path = self._paths.get(sensor_key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _merge(
        self,
        payloads: dict[str, dict[str, Any]],
        *,
        freshness: dict[str, dict[str, Any]],
        now: float,
    ) -> dict[str, Any]:
        pose = payloads.get("pose_gaze", {})
        gesture = payloads.get("gesture", {})
        sound = payloads.get("sound_source", {})
        association = payloads.get("audio_visual_association", {})
        approach = payloads.get("approach_dwell", {})
        arbitration = payloads.get("multi_person_arbitration", {})

        objects = _objects_from(pose, approach, arbitration)
        observed_at_values = [
            item.get("observed_at")
            for item in freshness.values()
            if item.get("status") == "fresh" and item.get("observed_at") is not None
        ]
        observed_at = max(observed_at_values) if observed_at_values else None
        active_track_id = _first_text(
            arbitration.get("active_person_track_id"),
            association.get("matched_track_id"),
            approach.get("track_id"),
            pose.get("track_id"),
        )
        speaker_track_id = _first_text(
            arbitration.get("speaker_track_id"),
            association.get("speaker_track_id"),
            sound.get("speaker_track_id"),
        )

        snapshot = {
            "source": "interaction_provider",
            "snapshot_id": _first_text(
                arbitration.get("snapshot_id"),
                pose.get("snapshot_id"),
                gesture.get("snapshot_id"),
                sound.get("snapshot_id"),
            ),
            "observed_at": observed_at,
            "reason": "fresh" if payloads else "no_fresh_provider_input",
            "objects": objects,
            "person_count": _first_int(
                arbitration.get("person_count"),
                approach.get("person_count"),
                pose.get("person_count"),
                len(objects) if objects else None,
            ),
            "nearest_person_distance_m": _first_float(
                approach.get("distance_m"),
                pose.get("distance_m"),
                arbitration.get("nearest_person_distance_m"),
            ),
            "person_angle_deg": _first_float(
                pose.get("person_angle_deg"),
                approach.get("person_angle_deg"),
                association.get("person_angle_deg"),
            ),
            "person_facing_robot": _bool_or_none(
                pose.get("person_facing_robot")
                if "person_facing_robot" in pose
                else pose.get("facing_robot")
            ),
            "gaze": _first_text(pose.get("gaze"), pose.get("head_pose")),
            "posture": _first_text(pose.get("posture"), approach.get("approach_state")),
            "gesture": _first_text(gesture.get("gesture"), pose.get("gesture")),
            "sound_source_angle_deg": _first_float(
                sound.get("sound_source_angle_deg"),
                sound.get("doa_angle_deg"),
                sound.get("source_angle_deg"),
            ),
            "sound_source_matches_person": _bool_or_none(
                association.get("sound_source_matches_person")
            ),
            "metadata": {
                "freshness_by_sensor": freshness,
                "active_person_track_id": active_track_id,
                "speaker_track_id": speaker_track_id,
                "association_confidence": _first_float(
                    association.get("association_confidence"),
                    association.get("confidence"),
                ),
                "ambiguity_reason": _first_text(arbitration.get("ambiguity_reason")),
                "approach_state": _first_text(approach.get("approach_state")),
                "dwell_s": _first_float(approach.get("dwell_s")),
                "generated_at": now,
            },
        }
        return {key: value for key, value in snapshot.items() if value is not None}


def _objects_from(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    for payload in payloads:
        for key in ("objects", "persons", "detections"):
            raw = payload.get(key)
            if isinstance(raw, list):
                return [dict(item) for item in raw if isinstance(item, dict)]
    return []


def _timestamp(payload: dict[str, Any]) -> float | None:
    return _first_float(
        payload.get("observed_at"),
        payload.get("timestamp"),
        payload.get("updated_at"),
        payload.get("created_at"),
    )


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _first_float(value)
        if parsed is not None:
            return int(parsed)
    return 0


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None
