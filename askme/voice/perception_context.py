"""Normalized perception context for real-world voice interaction gating.

This module keeps sensor-specific details out of the voice loop. Camera,
microphone-array, pose, and world-state providers can all publish loose dicts;
the interaction gate receives one compact, timestamped snapshot.

TODO(real perception algorithms):
- Add a pose/gaze provider that estimates whether a person is facing the robot.
- Add a gesture provider for wave, raised hand, pointing, and stop gestures.
- Add microphone-array DOA input for sound_source_angle_deg.
- Add sound/vision association across frames, not only per-turn angle matching.
- Add dwell-time and approach/retreat tracking for visitor intent scoring.
- Add freshness contracts per sensor, because camera, depth, pose, and audio
  may have different acceptable ages.
- Add multi-person arbitration so the robot knows which person is speaking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

_PERSON_LABELS = {
    "person",
    "human",
    "visitor",
    "tourist",
    "\u4eba",
    "\u6e38\u5ba2",
    "\u884c\u4eba",
}

_ATTENTION_GESTURES = {
    "wave",
    "waving",
    "raise_hand",
    "raised_hand",
    "pointing",
    "\u6325\u624b",
    "\u4e3e\u624b",
    "\u6307\u5411",
}

_ENGAGED_POSTURES = {
    "standing",
    "leaning_forward",
    "approaching",
    "squatting_near",
    "\u7ad9\u7acb",
    "\u9760\u8fd1",
    "\u524d\u503e",
}

_DISENGAGED_POSTURES = {
    "walking_away",
    "back_turned",
    "sitting_away",
    "lying_down",
    "\u8d70\u5f00",
    "\u80cc\u5bf9",
    "\u8eba\u4e0b",
}

_FACING_VALUES = {"front", "facing", "towards_robot", "toward_robot", "\u9762\u5411"}


@dataclass(frozen=True)
class InteractionPerceptionSnapshot:
    """Sensor evidence used to decide whether speech is addressed to the robot."""

    source: str = "unknown"
    snapshot_id: str = ""
    observed_at: float | None = None
    freshness_s: float | None = None
    fresh: bool = False
    reason: str = ""
    person_detected: bool | None = None
    person_count: int = 0
    nearest_person_distance_m: float | None = None
    person_angle_deg: float | None = None
    visual_attention: bool | None = None
    person_facing_robot: bool | None = None
    posture: str = ""
    gesture: str = ""
    sound_source_matches_person: bool | None = None
    sound_source_angle_deg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unknown(cls, reason: str = "not_available") -> InteractionPerceptionSnapshot:
        return cls(reason=reason)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        max_age_s: float = 2.0,
        max_interaction_distance_m: float = 4.0,
        sound_angle_tolerance_deg: float = 35.0,
        now: float | None = None,
    ) -> InteractionPerceptionSnapshot:
        if payload is None:
            return cls.unknown()
        if isinstance(payload, InteractionPerceptionSnapshot):
            return payload
        if not isinstance(payload, dict):
            return cls.unknown("invalid_payload")

        current = time.time() if now is None else now
        observed_at = _float_or_none(
            payload.get("observed_at")
            or payload.get("timestamp")
            or payload.get("updated_at")
            or payload.get("created_at")
        )
        freshness_s = None if observed_at is None else max(0.0, current - observed_at)
        fresh = bool(observed_at is not None and freshness_s <= max_age_s)

        objects = _extract_objects(payload)
        persons = [item for item in objects if _is_person(item)]
        person_count = _int_or_zero(payload.get("person_count"), default=len(persons))
        if persons and person_count < len(persons):
            person_count = len(persons)
        person_detected = _bool_or_none(payload.get("person_detected"))
        if person_detected is None and person_count:
            person_detected = True
        if person_detected is None and objects:
            person_detected = False

        nearest_person = _nearest_person(persons)
        nearest_distance = _first_float(
            payload,
            ("nearest_person_distance_m", "distance_m", "person_distance_m"),
        )
        if nearest_distance is None and nearest_person is not None:
            nearest_distance = _float_or_none(nearest_person.get("distance_m"))

        person_angle = _first_float(payload, ("person_angle_deg", "target_angle_deg"))
        if person_angle is None and nearest_person is not None:
            person_angle = _first_float(nearest_person, ("angle_deg", "azimuth_deg"))

        sound_angle = _first_float(
            payload,
            ("sound_source_angle_deg", "audio_source_angle_deg", "source_angle_deg"),
        )
        sound_match = _bool_or_none(payload.get("sound_source_matches_person"))
        if sound_match is None and sound_angle is not None and person_angle is not None:
            sound_match = _angle_delta(sound_angle, person_angle) <= sound_angle_tolerance_deg

        facing = _bool_or_none(payload.get("person_facing_robot"))
        if facing is None:
            facing = _facing_from_payload(payload)
        if facing is None and nearest_person is not None:
            facing = _facing_from_payload(nearest_person)

        gesture = str(payload.get("gesture") or "").strip().lower()
        posture = str(payload.get("posture") or "").strip().lower()
        if nearest_person is not None:
            gesture = gesture or str(nearest_person.get("gesture") or "").strip().lower()
            posture = posture or str(nearest_person.get("posture") or "").strip().lower()

        visual_attention = _bool_or_none(payload.get("visual_attention"))
        if visual_attention is None:
            visual_attention = _infer_visual_attention(
                person_detected=person_detected,
                nearest_distance=nearest_distance,
                max_interaction_distance_m=max_interaction_distance_m,
                facing=facing,
                gesture=gesture,
                posture=posture,
                nearest_person=nearest_person,
            )

        reason = str(payload.get("reason") or "").strip()
        if not reason:
            if observed_at is None:
                reason = "no_timestamp"
            elif not fresh:
                reason = "stale"
            elif person_detected is False:
                reason = "no_person"
            else:
                reason = "fresh"

        return cls(
            source=str(payload.get("source") or "perception"),
            snapshot_id=str(payload.get("snapshot_id") or payload.get("id") or ""),
            observed_at=observed_at,
            freshness_s=freshness_s,
            fresh=fresh,
            reason=reason,
            person_detected=person_detected,
            person_count=person_count,
            nearest_person_distance_m=nearest_distance,
            person_angle_deg=person_angle,
            visual_attention=visual_attention,
            person_facing_robot=facing,
            posture=posture,
            gesture=gesture,
            sound_source_matches_person=sound_match,
            sound_source_angle_deg=sound_angle,
            metadata={
                "object_count": len(objects),
                "max_age_s": max_age_s,
                "max_interaction_distance_m": max_interaction_distance_m,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "snapshot_id": self.snapshot_id,
            "observed_at": self.observed_at,
            "freshness_s": self.freshness_s,
            "fresh": self.fresh,
            "reason": self.reason,
            "person_detected": self.person_detected,
            "person_count": self.person_count,
            "nearest_person_distance_m": self.nearest_person_distance_m,
            "person_angle_deg": self.person_angle_deg,
            "visual_attention": self.visual_attention,
            "person_facing_robot": self.person_facing_robot,
            "posture": self.posture,
            "gesture": self.gesture,
            "sound_source_matches_person": self.sound_source_matches_person,
            "sound_source_angle_deg": self.sound_source_angle_deg,
            "metadata": dict(self.metadata),
        }


def _extract_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("objects")
    if raw is None and isinstance(payload.get("scene"), dict):
        raw = payload["scene"].get("objects")
    if raw is None and isinstance(payload.get("detections"), list):
        raw = payload["detections"]
    if raw is None and isinstance(payload.get("persons"), list):
        raw = payload["persons"]
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _is_person(item: dict[str, Any]) -> bool:
    label = str(
        item.get("label")
        or item.get("class_id")
        or item.get("class")
        or item.get("name")
        or item.get("type")
        or ""
    ).strip().lower()
    return label in _PERSON_LABELS or "person" in label


def _nearest_person(persons: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not persons:
        return None
    return min(
        persons,
        key=lambda item: (
            _float_or_none(item.get("distance_m")) is None,
            _float_or_none(item.get("distance_m")) or 99.0,
            -(_float_or_none(item.get("confidence")) or 0.0),
        ),
    )


def _infer_visual_attention(
    *,
    person_detected: bool | None,
    nearest_distance: float | None,
    max_interaction_distance_m: float,
    facing: bool | None,
    gesture: str,
    posture: str,
    nearest_person: dict[str, Any] | None,
) -> bool | None:
    if person_detected is False:
        return False
    if gesture in _ATTENTION_GESTURES:
        return True
    if posture in _DISENGAGED_POSTURES:
        return False
    if facing is True and (
        nearest_distance is None or nearest_distance <= max_interaction_distance_m
    ):
        return True
    if facing is False:
        return False
    if posture in _ENGAGED_POSTURES and (
        nearest_distance is None or nearest_distance <= max_interaction_distance_m
    ):
        return True
    if nearest_person is None:
        return None
    bbox_centered = _bbox_centered(nearest_person.get("bbox"), nearest_person)
    if bbox_centered is not None and nearest_distance is not None:
        return bool(bbox_centered and nearest_distance <= max_interaction_distance_m)
    return None


def _bbox_centered(bbox: Any, payload: dict[str, Any]) -> bool | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    frame_width = _first_float(payload, ("frame_width", "image_width", "width")) or 1280.0
    try:
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    except (TypeError, ValueError):
        return None
    ratio = center_x / max(frame_width, 1.0)
    return 0.30 <= ratio <= 0.70


def _facing_from_payload(payload: dict[str, Any]) -> bool | None:
    gaze = str(
        payload.get("gaze")
        or payload.get("head_pose")
        or payload.get("orientation")
        or payload.get("body_orientation")
        or ""
    ).strip().lower()
    if not gaze:
        return None
    if gaze in _FACING_VALUES or "toward" in gaze or "front" in gaze:
        return True
    if "back" in gaze or "away" in gaze:
        return False
    return None


def _first_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _float_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _int_or_zero(value: Any, *, default: int = 0) -> int:
    if value is None or value == "":
        return max(0, int(default))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    clean = str(value).strip().lower()
    if clean in {"1", "true", "yes", "y", "on", "\u662f"}:
        return True
    if clean in {"0", "false", "no", "n", "off", "\u5426"}:
        return False
    return None


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)
