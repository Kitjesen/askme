"""Stable product I/O contracts for robot capability packages.

The goal is not to replace lower-level runtime structures such as TaskHandoff.
These models define the product boundary every package should speak:

- PerceptionInput: what the robot currently sees, hears, senses, and where it is.
- IntentInput: what the user/system appears to want, with actor and confidence.
- ActionDecision: the controlled action the brain may take next.
- UserFacingOutput: what the customer/operator/visitor can see or hear.

The contracts intentionally use standard-library dataclasses so they remain
lightweight inside MCP, CLI, FastAPI, tests, and embedded robot processes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActorType(StrEnum):
    VISITOR = "visitor"
    OPERATOR = "operator"
    SECURITY = "security"
    CLEANER = "cleaner"
    SYSTEM = "system"
    ROBOT = "robot"
    UNKNOWN = "unknown"


class IntentType(StrEnum):
    ASK_DIRECTION = "ask_direction"
    REQUEST_LEAD = "request_lead"
    START_INSPECTION = "start_inspection"
    REPORT_ISSUE = "report_issue"
    EMERGENCY_EVENT = "emergency_event"
    KNOWLEDGE_QUESTION = "knowledge_question"
    ROBOT_CONTROL = "robot_control"
    SMALLTALK = "smalltalk"
    UNKNOWN = "unknown"


class RobotActionType(StrEnum):
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"
    GUIDE_BY_VOICE = "guide_by_voice"
    CREATE_TASK_HANDOFF = "create_task_handoff"
    NOTIFY_HUMAN = "notify_human"
    RECORD_EVENT = "record_event"
    REJECT = "reject"
    IGNORE = "ignore"
    DEFER = "defer"
    ESCALATE = "escalate"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LocationRef:
    site_id: str = ""
    map_id: str = ""
    point_id: str = ""
    name: str = ""
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> LocationRef:
        if isinstance(payload, LocationRef):
            return payload
        if not isinstance(payload, dict):
            return cls(name=str(payload or ""))
        return cls(
            site_id=_clean_text(payload.get("site_id") or payload.get("park_id")),
            map_id=_clean_text(payload.get("map_id")),
            point_id=_clean_text(payload.get("point_id") or payload.get("location_id")),
            name=_clean_text(payload.get("name") or payload.get("location_name")),
            x=_float_or_none(payload.get("x")),
            y=_float_or_none(payload.get("y")),
            z=_float_or_none(payload.get("z")),
            yaw=_float_or_none(payload.get("yaw")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "map_id": self.map_id,
            "point_id": self.point_id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
        }


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str = ""
    evidence_type: str = ""
    source: str = ""
    uri: str = ""
    summary: str = ""
    confidence: float | None = None
    observed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> EvidenceRef:
        if isinstance(payload, EvidenceRef):
            return payload
        if not isinstance(payload, dict):
            return cls(summary=str(payload or ""))
        return cls(
            evidence_id=_clean_text(payload.get("evidence_id") or payload.get("id")),
            evidence_type=_clean_text(payload.get("evidence_type") or payload.get("type")),
            source=_clean_text(payload.get("source")),
            uri=_clean_text(payload.get("uri") or payload.get("url") or payload.get("path")),
            summary=_clean_text(payload.get("summary") or payload.get("text")),
            confidence=_float_or_none(payload.get("confidence")),
            observed_at=_float_or_none(payload.get("observed_at") or payload.get("timestamp")),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "uri": self.uri,
            "summary": self.summary,
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Freshness:
    vision_ms: int | None = None
    audio_ms: int | None = None
    sensor_ms: int | None = None
    world_state_ms: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> Freshness:
        if isinstance(payload, Freshness):
            return payload
        if not isinstance(payload, dict):
            return cls()
        return cls(
            vision_ms=_int_or_none(payload.get("vision_ms")),
            audio_ms=_int_or_none(payload.get("audio_ms")),
            sensor_ms=_int_or_none(payload.get("sensor_ms")),
            world_state_ms=_int_or_none(payload.get("world_state_ms")),
        )

    def stale_channels(
        self,
        max_age_ms: dict[str, int] | None = None,
        *,
        missing_is_stale: bool = False,
    ) -> list[str]:
        limits = {
            "vision": 2000,
            "audio": 1000,
            "sensor": 5000,
            "world_state": 5000,
            **(max_age_ms or {}),
        }
        values = {
            "vision": self.vision_ms,
            "audio": self.audio_ms,
            "sensor": self.sensor_ms,
            "world_state": self.world_state_ms,
        }
        stale: list[str] = []
        for channel, age_ms in values.items():
            if age_ms is None:
                if missing_is_stale:
                    stale.append(channel)
                continue
            if age_ms > limits[channel]:
                stale.append(channel)
        return stale

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision_ms": self.vision_ms,
            "audio_ms": self.audio_ms,
            "sensor_ms": self.sensor_ms,
            "world_state_ms": self.world_state_ms,
        }


@dataclass(frozen=True)
class VisionInput:
    persons: list[dict[str, Any]] = field(default_factory=list)
    vehicles: list[dict[str, Any]] = field(default_factory=list)
    smoke_fire: list[dict[str, Any]] = field(default_factory=list)
    trash_bins: list[dict[str, Any]] = field(default_factory=list)
    obstacles: list[dict[str, Any]] = field(default_factory=list)
    service_points: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> VisionInput:
        if isinstance(payload, VisionInput):
            return payload
        data = _dict(payload)
        return cls(
            persons=_list_of_dicts(data.get("persons")),
            vehicles=_list_of_dicts(data.get("vehicles")),
            smoke_fire=_list_of_dicts(data.get("smoke_fire")),
            trash_bins=_list_of_dicts(data.get("trash_bins")),
            obstacles=_list_of_dicts(data.get("obstacles")),
            service_points=_list_of_dicts(data.get("service_points")),
            raw=_dict(data.get("raw")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persons": list(self.persons),
            "vehicles": list(self.vehicles),
            "smoke_fire": list(self.smoke_fire),
            "trash_bins": list(self.trash_bins),
            "obstacles": list(self.obstacles),
            "service_points": list(self.service_points),
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class AudioInput:
    transcript: str = ""
    language: str = ""
    direction_deg: float | None = None
    confidence: float | None = None
    noise_level: float | None = None
    is_addressing_robot: bool | None = None
    source: str = ""

    @classmethod
    def from_payload(cls, payload: Any) -> AudioInput:
        if isinstance(payload, AudioInput):
            return payload
        data = _dict(payload)
        addressing_value = (
            data.get("is_addressing_robot")
            if "is_addressing_robot" in data
            else data.get("addressed")
        )
        return cls(
            transcript=_clean_text(data.get("transcript") or data.get("text")),
            language=_clean_text(data.get("language") or data.get("lang")),
            direction_deg=_float_or_none(
                data.get("direction_deg")
                or data.get("sound_source_angle_deg")
                or data.get("source_angle_deg")
            ),
            confidence=_float_or_none(data.get("confidence")),
            noise_level=_float_or_none(data.get("noise_level")),
            is_addressing_robot=_bool_or_none(addressing_value),
            source=_clean_text(data.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "language": self.language,
            "direction_deg": self.direction_deg,
            "confidence": self.confidence,
            "noise_level": self.noise_level,
            "is_addressing_robot": self.is_addressing_robot,
            "source": self.source,
        }


@dataclass(frozen=True)
class SensorInput:
    temperature_c: float | None = None
    smoke_detected: bool | None = None
    battery_percent: float | None = None
    motor_status: str = ""
    fall_status: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> SensorInput:
        if isinstance(payload, SensorInput):
            return payload
        data = _dict(payload)
        return cls(
            temperature_c=_float_or_none(data.get("temperature_c")),
            smoke_detected=_bool_or_none(data.get("smoke_detected")),
            battery_percent=_float_or_none(data.get("battery_percent")),
            motor_status=_clean_text(data.get("motor_status")),
            fall_status=_clean_text(data.get("fall_status")),
            raw=_dict(data.get("raw")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "smoke_detected": self.smoke_detected,
            "battery_percent": self.battery_percent,
            "motor_status": self.motor_status,
            "fall_status": self.fall_status,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class PerceptionInput:
    timestamp: float = field(default_factory=time.time)
    robot_id: str = ""
    location: LocationRef = field(default_factory=LocationRef)
    vision: VisionInput = field(default_factory=VisionInput)
    audio: AudioInput = field(default_factory=AudioInput)
    sensors: SensorInput = field(default_factory=SensorInput)
    freshness: Freshness = field(default_factory=Freshness)
    evidence_refs: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> PerceptionInput:
        if isinstance(payload, PerceptionInput):
            return payload
        data = _dict(payload)
        return cls(
            timestamp=_float_or_none(data.get("timestamp") or data.get("observed_at"))
            or time.time(),
            robot_id=_clean_text(data.get("robot_id")),
            location=LocationRef.from_payload(data.get("location")),
            vision=VisionInput.from_payload(data.get("vision")),
            audio=AudioInput.from_payload(data.get("audio")),
            sensors=SensorInput.from_payload(data.get("sensors")),
            freshness=Freshness.from_payload(data.get("freshness")),
            evidence_refs=tuple(
                EvidenceRef.from_payload(item)
                for item in _list(data.get("evidence_refs") or data.get("evidence"))
            ),
            metadata=_dict(data.get("metadata")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.timestamp <= 0:
            errors.append("timestamp must be positive")
        stale = self.freshness.stale_channels()
        if stale:
            errors.append("stale_channels:" + ",".join(stale))
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "robot_id": self.robot_id,
            "location": self.location.to_dict(),
            "vision": self.vision.to_dict(),
            "audio": self.audio.to_dict(),
            "sensors": self.sensors.to_dict(),
            "freshness": self.freshness.to_dict(),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IntentInput:
    intent_type: IntentType = IntentType.UNKNOWN
    actor_type: ActorType = ActorType.UNKNOWN
    text: str = ""
    confidence: float = 0.0
    target: str = ""
    location: LocationRef = field(default_factory=LocationRef)
    evidence_refs: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> IntentInput:
        if isinstance(payload, IntentInput):
            return payload
        data = _dict(payload)
        return cls(
            intent_type=_enum_value(IntentType, data.get("intent_type"), IntentType.UNKNOWN),
            actor_type=_enum_value(ActorType, data.get("actor_type"), ActorType.UNKNOWN),
            text=_clean_text(data.get("text")),
            confidence=_float_or_none(data.get("confidence")) or 0.0,
            target=_clean_text(data.get("target")),
            location=LocationRef.from_payload(data.get("location")),
            evidence_refs=tuple(
                EvidenceRef.from_payload(item)
                for item in _list(data.get("evidence_refs") or data.get("evidence"))
            ),
            metadata=_dict(data.get("metadata")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0 and 1")
        if not self.text and self.intent_type not in {
            IntentType.EMERGENCY_EVENT,
            IntentType.UNKNOWN,
        }:
            errors.append("text is required for non-system intents")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent_type.value,
            "actor_type": self.actor_type.value,
            "text": self.text,
            "confidence": self.confidence,
            "target": self.target,
            "location": self.location.to_dict(),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActionDecision:
    action_type: RobotActionType = RobotActionType.IGNORE
    reason: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    skill_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> ActionDecision:
        if isinstance(payload, ActionDecision):
            return payload
        data = _dict(payload)
        return cls(
            action_type=_enum_value(
                RobotActionType,
                data.get("action_type") or data.get("action"),
                RobotActionType.IGNORE,
            ),
            reason=_clean_text(data.get("reason")),
            risk_level=_enum_value(RiskLevel, data.get("risk_level"), RiskLevel.LOW),
            requires_confirmation=bool(data.get("requires_confirmation", False)),
            skill_name=_clean_text(data.get("skill_name")),
            parameters=_dict(data.get("parameters")),
            evidence_refs=tuple(
                EvidenceRef.from_payload(item)
                for item in _list(data.get("evidence_refs") or data.get("evidence"))
            ),
            confidence=_float_or_none(data.get("confidence")) or 0.0,
            metadata=_dict(data.get("metadata")),
        )

    @property
    def should_speak(self) -> bool:
        return self.action_type in {
            RobotActionType.ANSWER,
            RobotActionType.ASK_CLARIFICATION,
            RobotActionType.GUIDE_BY_VOICE,
            RobotActionType.REJECT,
            RobotActionType.DEFER,
            RobotActionType.ESCALATE,
        }

    @property
    def is_blocking(self) -> bool:
        return self.action_type in {
            RobotActionType.REJECT,
            RobotActionType.DEFER,
            RobotActionType.ESCALATE,
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.reason:
            errors.append("reason is required")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0 and 1")
        if self.action_type == RobotActionType.CREATE_TASK_HANDOFF and not self.skill_name:
            errors.append("skill_name is required for create_task_handoff")
        return errors

    def to_user_output(
        self,
        *,
        spoken_text: str = "",
        display_text: str = "",
        audit_id: str = "",
    ) -> UserFacingOutput:
        return UserFacingOutput(
            spoken_text=spoken_text,
            display_text=display_text or spoken_text,
            status=self.action_type.value,
            next_action=self.reason,
            evidence=self.evidence_refs,
            confidence=self.confidence,
            fallback=_clean_text(self.metadata.get("fallback")),
            audit_id=audit_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "reason": self.reason,
            "risk_level": self.risk_level.value,
            "requires_confirmation": self.requires_confirmation,
            "skill_name": self.skill_name,
            "parameters": dict(self.parameters),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "should_speak": self.should_speak,
            "is_blocking": self.is_blocking,
        }


@dataclass(frozen=True)
class UserFacingOutput:
    spoken_text: str = ""
    display_text: str = ""
    status: str = ""
    next_action: str = ""
    evidence: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    fallback: str = ""
    audit_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> UserFacingOutput:
        if isinstance(payload, UserFacingOutput):
            return payload
        data = _dict(payload)
        return cls(
            spoken_text=_clean_text(data.get("spoken_text")),
            display_text=_clean_text(data.get("display_text")),
            status=_clean_text(data.get("status")),
            next_action=_clean_text(data.get("next_action")),
            evidence=tuple(
                EvidenceRef.from_payload(item)
                for item in _list(data.get("evidence") or data.get("evidence_refs"))
            ),
            confidence=_float_or_none(data.get("confidence")) or 0.0,
            fallback=_clean_text(data.get("fallback")),
            audit_id=_clean_text(data.get("audit_id")),
            metadata=_dict(data.get("metadata")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.status:
            errors.append("status is required")
        if not (self.spoken_text or self.display_text):
            errors.append("spoken_text or display_text is required")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0 and 1")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "spoken_text": self.spoken_text,
            "display_text": self.display_text,
            "status": self.status,
            "next_action": self.next_action,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "fallback": self.fallback,
            "audit_id": self.audit_id,
            "metadata": dict(self.metadata),
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _list(value) if isinstance(item, dict)]


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _enum_value(enum_cls: type[StrEnum], value: Any, default: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is not None:
        try:
            return enum_cls(str(value))
        except ValueError:
            pass
    return default
