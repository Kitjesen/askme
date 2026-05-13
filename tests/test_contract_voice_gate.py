from __future__ import annotations

from askme.contracts import (
    RobotActionType,
    interaction_decision_to_action_decision,
)
from askme.voice.interaction_gate import InteractionAction, InteractionGate
from askme.voice.perception_context import InteractionPerceptionSnapshot


def test_bystander_speech_records_without_brain_continuation() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("我们去那边看看", addressed=False)
    action = interaction_decision_to_action_decision(
        decision,
        user_text="我们去那边看看",
        addressed=False,
    )

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.should_continue_to_brain is False
    assert action.action_type == RobotActionType.RECORD_EVENT
    assert action.parameters["addressed"] is False


def test_public_wayfinding_without_wake_word_enters_answer_contract() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("请问厕所在哪里", addressed=False)
    action = interaction_decision_to_action_decision(
        decision,
        user_text="请问厕所在哪里",
        addressed=False,
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "public_help_or_wayfinding"
    assert decision.should_continue_to_brain is True
    assert action.action_type == RobotActionType.ANSWER
    assert action.parameters["interaction_action"] == "respond"
    assert action.parameters["user_text"] == "请问厕所在哪里"


def test_multi_person_ambiguity_keeps_perception_evidence() -> None:
    gate = InteractionGate({"enabled": True})
    perception = InteractionPerceptionSnapshot(
        source="vision_bridge",
        snapshot_id="snap-multi",
        observed_at=123.0,
        freshness_s=0.1,
        fresh=True,
        reason="fresh",
        person_detected=True,
        person_count=2,
        sound_source_matches_person=False,
    )

    decision = gate.evaluate("你好", addressed=True, perception=perception)
    action = interaction_decision_to_action_decision(
        decision,
        user_text="你好",
        addressed=True,
        perception=perception,
    )

    assert decision.action == InteractionAction.CLARIFY
    assert action.action_type == RobotActionType.ASK_CLARIFICATION
    assert action.evidence_refs[0].evidence_type == "interaction_perception"
    assert action.evidence_refs[0].evidence_id == "snap-multi"
