from __future__ import annotations

import time

from askme.contracts import (
    ActionDecision,
    Freshness,
    IntentInput,
    IntentType,
    PerceptionInput,
    RobotActionType,
    interaction_decision_to_action_decision,
    perception_snapshot_to_input,
)
from askme.voice.interaction_gate import InteractionAction, InteractionDecision
from askme.voice.perception_context import InteractionPerceptionSnapshot


def test_perception_input_round_trips_robot_sensor_context() -> None:
    payload = {
        "timestamp": 123.0,
        "robot_id": "thunder-01",
        "location": {"site_id": "fanmu", "point_id": "west_gate", "name": "West Gate"},
        "vision": {"persons": [{"distance_m": 1.8}], "vehicles": [{"plate": "demo"}]},
        "audio": {"transcript": "where is the cafe", "confidence": 0.88, "addressed": True},
        "sensors": {"battery_percent": 72, "motor_status": "ok"},
        "freshness": {"vision_ms": 500, "audio_ms": 120, "sensor_ms": 800},
        "evidence_refs": [{"id": "frame-1", "type": "image", "source": "camera"}],
    }

    contract = PerceptionInput.from_payload(payload)

    assert contract.robot_id == "thunder-01"
    assert contract.location.point_id == "west_gate"
    assert contract.vision.persons[0]["distance_m"] == 1.8
    assert contract.audio.is_addressing_robot is True
    assert contract.validate() == []
    assert contract.to_dict()["evidence_refs"][0]["evidence_id"] == "frame-1"


def test_freshness_reports_stale_channels() -> None:
    freshness = Freshness(vision_ms=2500, audio_ms=200, sensor_ms=None)

    assert freshness.stale_channels() == ["vision"]
    assert freshness.stale_channels(missing_is_stale=True) == [
        "vision",
        "sensor",
        "world_state",
    ]


def test_intent_and_action_contracts_validate_control_boundaries() -> None:
    intent = IntentInput.from_payload(
        {
            "intent_type": "ask_direction",
            "actor_type": "visitor",
            "text": "where is the cafe",
            "confidence": 0.9,
            "target": "cafe",
        }
    )
    decision = ActionDecision(
        action_type=RobotActionType.GUIDE_BY_VOICE,
        reason="park_wayfinding_answer",
        confidence=0.91,
    )

    assert intent.intent_type == IntentType.ASK_DIRECTION
    assert intent.validate() == []
    assert decision.should_speak is True
    assert decision.validate() == []
    assert decision.to_user_output(spoken_text="Go straight.").status == "guide_by_voice"


def test_interaction_gate_decision_has_product_action_contract() -> None:
    observed_at = time.time()
    perception = InteractionPerceptionSnapshot(
        source="vision_bridge",
        snapshot_id="snap-1",
        observed_at=observed_at,
        freshness_s=0.2,
        fresh=True,
        reason="fresh",
        person_detected=True,
        person_count=1,
        nearest_person_distance_m=1.2,
        sound_source_matches_person=True,
    )
    gate_decision = InteractionDecision(
        InteractionAction.CLARIFY,
        "weak_greeting_with_visual_attention",
        0.7,
        reply="How can I help?",
        should_record_environment=True,
    )

    perception_contract = perception_snapshot_to_input(
        perception,
        transcript="hello",
        addressed=True,
    )
    action_contract = interaction_decision_to_action_decision(
        gate_decision,
        user_text="hello",
        addressed=True,
        perception=perception,
    )

    assert perception_contract.audio.transcript == "hello"
    assert perception_contract.vision.persons[0]["distance_m"] == 1.2
    assert action_contract.action_type == RobotActionType.ASK_CLARIFICATION
    assert action_contract.risk_level.value == "medium"
    assert action_contract.evidence_refs[0].evidence_id == "snap-1"
    assert action_contract.metadata["fallback"] == "How can I help?"


def test_refuse_decision_maps_to_high_risk_reject() -> None:
    decision = interaction_decision_to_action_decision(
        InteractionDecision(
            InteractionAction.REFUSE,
            "unsafe_or_privacy_intent",
            0.9,
            reply="I cannot do that.",
            should_record_environment=True,
        )
    )

    assert decision.action_type == RobotActionType.REJECT
    assert decision.risk_level.value == "high"
    assert decision.is_blocking is True
