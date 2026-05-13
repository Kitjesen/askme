"""Adapters from existing robot internals into product I/O contracts."""

from __future__ import annotations

import time
from typing import Any

from askme.contracts.io import (
    ActionDecision,
    AudioInput,
    EvidenceRef,
    Freshness,
    PerceptionInput,
    RiskLevel,
    RobotActionType,
    VisionInput,
)
from askme.voice.interaction_gate import InteractionAction, InteractionDecision
from askme.voice.perception_context import InteractionPerceptionSnapshot

_ACTION_MAP: dict[InteractionAction, RobotActionType] = {
    InteractionAction.IGNORE: RobotActionType.IGNORE,
    InteractionAction.RECORD_ONLY: RobotActionType.RECORD_EVENT,
    InteractionAction.CLARIFY: RobotActionType.ASK_CLARIFICATION,
    InteractionAction.RESPOND: RobotActionType.ANSWER,
    InteractionAction.DEFER: RobotActionType.DEFER,
    InteractionAction.REFUSE: RobotActionType.REJECT,
}


def perception_snapshot_to_input(
    snapshot: InteractionPerceptionSnapshot | dict[str, Any] | None,
    *,
    robot_id: str = "",
    transcript: str = "",
    addressed: bool | None = None,
    asr_confidence: float | None = None,
) -> PerceptionInput:
    """Normalize interaction-gate perception evidence into PerceptionInput."""
    if snapshot is None:
        return PerceptionInput(
            robot_id=robot_id,
            audio=AudioInput(
                transcript=transcript,
                confidence=asr_confidence,
                is_addressing_robot=addressed,
            ),
        )
    normalized = (
        snapshot
        if isinstance(snapshot, InteractionPerceptionSnapshot)
        else InteractionPerceptionSnapshot.from_payload(snapshot)
    )
    freshness_ms = (
        int(normalized.freshness_s * 1000)
        if normalized.freshness_s is not None
        else None
    )
    person: dict[str, Any] = {
        "detected": normalized.person_detected,
        "count": normalized.person_count,
        "distance_m": normalized.nearest_person_distance_m,
        "angle_deg": normalized.person_angle_deg,
        "facing_robot": normalized.person_facing_robot,
        "posture": normalized.posture,
        "gesture": normalized.gesture,
        "visual_attention": normalized.visual_attention,
    }
    evidence = EvidenceRef(
        evidence_id=normalized.snapshot_id,
        evidence_type="interaction_perception",
        source=normalized.source,
        summary=normalized.reason,
        observed_at=normalized.observed_at,
        metadata=normalized.to_dict(),
    )
    return PerceptionInput(
        timestamp=normalized.observed_at or time.time(),
        robot_id=robot_id,
        vision=VisionInput(persons=[person], raw=normalized.to_dict()),
        audio=AudioInput(
            transcript=transcript,
            direction_deg=normalized.sound_source_angle_deg,
            confidence=asr_confidence,
            is_addressing_robot=addressed,
            source=normalized.source,
        ),
        freshness=Freshness(
            vision_ms=freshness_ms,
            audio_ms=freshness_ms,
            world_state_ms=freshness_ms,
        ),
        evidence_refs=(evidence,),
        metadata={
            "sound_source_matches_person": normalized.sound_source_matches_person,
            "fresh": normalized.fresh,
        },
    )


def interaction_decision_to_action_decision(
    decision: InteractionDecision,
    *,
    user_text: str = "",
    addressed: bool | None = None,
    perception: InteractionPerceptionSnapshot | dict[str, Any] | None = None,
) -> ActionDecision:
    """Convert InteractionGate output into the standard ActionDecision."""
    action_type = _ACTION_MAP.get(decision.action, RobotActionType.IGNORE)
    risk_level = _risk_for_interaction(decision)
    evidence_refs: tuple[EvidenceRef, ...] = ()
    if perception is not None:
        perception_input = perception_snapshot_to_input(
            perception,
            transcript=user_text,
            addressed=addressed,
        )
        evidence_refs = perception_input.evidence_refs
    return ActionDecision(
        action_type=action_type,
        reason=decision.reason,
        risk_level=risk_level,
        requires_confirmation=action_type
        in {
            RobotActionType.CREATE_TASK_HANDOFF,
            RobotActionType.NOTIFY_HUMAN,
            RobotActionType.ESCALATE,
        },
        parameters={
            "user_text": user_text,
            "addressed": addressed,
            "interaction_action": decision.action.value,
            "reply": decision.reply,
            "should_record_environment": decision.should_record_environment,
        },
        evidence_refs=evidence_refs,
        confidence=decision.confidence,
        metadata={
            "source": "interaction_gate",
            "fallback": decision.reply,
        },
    )


def _risk_for_interaction(decision: InteractionDecision) -> RiskLevel:
    if decision.action == InteractionAction.REFUSE:
        return RiskLevel.HIGH
    if decision.reason in {
        "unsafe_or_privacy_intent",
        "safety_stop_gesture",
    }:
        return RiskLevel.HIGH
    if decision.action in {InteractionAction.CLARIFY, InteractionAction.DEFER}:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
