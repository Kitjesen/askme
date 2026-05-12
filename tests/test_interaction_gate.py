import time

from askme.voice.interaction_gate import InteractionAction, InteractionGate
from askme.voice.perception_context import InteractionPerceptionSnapshot


def test_interaction_gate_records_casual_bystander_speech() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("我们去那边看看", addressed=False)

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.should_record_environment is True
    assert decision.should_continue_to_brain is False


def test_interaction_gate_records_casual_robot_mention_without_addressing() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("我们去那边看看，这个机器狗好可爱", addressed=False)

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.reason == "not_addressed_casual_robot_mention"
    assert decision.should_record_environment is True
    assert decision.should_continue_to_brain is False


def test_interaction_gate_responds_to_tourist_wayfinding() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("请问厕所在哪里", addressed=False)

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "public_help_or_wayfinding"
    assert decision.should_continue_to_brain is True


def test_interaction_gate_clarifies_weak_greeting() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("你好", addressed=False)

    assert decision.action == InteractionAction.CLARIFY
    assert "?" in decision.reply or "？" in decision.reply


def test_interaction_gate_refuses_privacy_or_unsafe_intent() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("拍一下那个人", addressed=True)

    assert decision.action == InteractionAction.REFUSE
    assert decision.should_record_environment is True


def test_interaction_gate_defers_when_task_cannot_be_interrupted() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("今天天气怎么样", addressed=True, task_interruptible=False)

    assert decision.action == InteractionAction.DEFER
    assert decision.should_continue_to_brain is False


def test_interaction_gate_keeps_explicit_robot_task_during_active_task() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate("停下", addressed=False, task_interruptible=False)

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "robot_task_intent"


def test_perception_snapshot_infers_attention_from_fresh_centered_person() -> None:
    snapshot = InteractionPerceptionSnapshot.from_payload(
        {
            "source": "vision",
            "observed_at": 100.0,
            "objects": [
                {
                    "class_id": "person",
                    "confidence": 0.91,
                    "distance_m": 1.8,
                    "bbox": [500, 120, 700, 460],
                    "frame_width": 1280,
                }
            ],
        },
        now=101.0,
    )

    assert snapshot.fresh is True
    assert snapshot.person_detected is True
    assert snapshot.nearest_person_distance_m == 1.8
    assert snapshot.visual_attention is True


def test_interaction_gate_requires_fresh_perception_for_weak_greeting() -> None:
    gate = InteractionGate({"enabled": True, "max_perception_age_s": 2.0})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "vision",
            "observed_at": 100.0,
            "objects": [{"class_id": "person", "distance_m": 1.5}],
        },
    )

    assert decision.action == InteractionAction.CLARIFY
    assert decision.reason == "stale_perception_needs_refresh"
    assert decision.should_record_environment is True


def test_interaction_gate_records_when_person_is_too_far() -> None:
    gate = InteractionGate({"enabled": True, "max_interaction_distance_m": 3.0})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "vision",
            "observed_at": time.time(),
            "objects": [{"class_id": "person", "distance_m": 5.5}],
        },
    )

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.reason == "person_too_far"
    assert decision.should_record_environment is True
    assert decision.should_continue_to_brain is False


def test_interaction_gate_allows_explicit_robot_task_when_person_is_too_far() -> None:
    gate = InteractionGate({"enabled": True, "max_interaction_distance_m": 3.0})

    decision = gate.evaluate(
        "\u505c\u4e0b",
        addressed=False,
        perception={
            "source": "vision",
            "observed_at": time.time(),
            "objects": [{"class_id": "person", "distance_m": 9.0}],
        },
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "robot_task_intent"
    assert decision.should_continue_to_brain is True


def test_interaction_gate_records_audio_visual_mismatch_from_angles() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "vision_audio",
            "observed_at": time.time(),
            "person_angle_deg": 0,
            "sound_source_angle_deg": 90,
            "objects": [{"class_id": "person", "distance_m": 1.2}],
        },
    )

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.reason == "audio_visual_mismatch"
    assert decision.should_record_environment is True


def test_perception_snapshot_marks_sound_source_mismatch() -> None:
    snapshot = InteractionPerceptionSnapshot.from_payload(
        {
            "source": "vision_audio",
            "observed_at": 100.0,
            "person_angle_deg": 0,
            "sound_source_angle_deg": 90,
            "objects": [{"class_id": "person", "distance_m": 1.2}],
        },
        now=101.0,
        sound_angle_tolerance_deg=35.0,
    )

    assert snapshot.sound_source_matches_person is False
    assert snapshot.sound_source_angle_deg == 90
    assert snapshot.person_angle_deg == 0


def test_perception_snapshot_infers_attention_from_raise_hand_gesture() -> None:
    snapshot = InteractionPerceptionSnapshot.from_payload(
        {
            "source": "pose",
            "observed_at": 100.0,
            "objects": [
                {
                    "class_id": "person",
                    "distance_m": 2.0,
                    "gesture": "raise_hand",
                    "bbox": [10, 120, 160, 460],
                    "frame_width": 1280,
                }
            ],
        },
        now=101.0,
    )

    assert snapshot.person_detected is True
    assert snapshot.gesture == "raise_hand"
    assert snapshot.visual_attention is True


def test_interaction_gate_clarifies_when_multiple_people_present_without_speaker_lock() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "vision",
            "observed_at": time.time(),
            "objects": [
                {"class_id": "person", "distance_m": 1.2, "angle_deg": -10},
                {"class_id": "person", "distance_m": 1.8, "angle_deg": 20},
            ],
        },
    )

    assert decision.action == InteractionAction.CLARIFY
    assert decision.reason == "multi_person_ambiguous_speaker"
    assert decision.should_record_environment is True


def test_interaction_gate_clarifies_multi_person_even_with_sound_mismatch() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "vision_audio",
            "observed_at": time.time(),
            "person_count": 2,
            "person_angle_deg": 0,
            "sound_source_angle_deg": 80,
            "objects": [
                {"class_id": "person", "distance_m": 1.2, "angle_deg": -10},
                {"class_id": "person", "distance_m": 1.8, "angle_deg": 20},
            ],
        },
    )

    assert decision.action == InteractionAction.CLARIFY
    assert decision.reason == "multi_person_ambiguous_speaker"
    assert decision.should_record_environment is True


def test_interaction_gate_allows_explicit_robot_task_despite_multi_person_ambiguity() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "\u505c\u4e0b",
        addressed=False,
        perception={
            "source": "vision",
            "observed_at": time.time(),
            "objects": [
                {"class_id": "person", "distance_m": 1.2, "angle_deg": -10},
                {"class_id": "person", "distance_m": 1.8, "angle_deg": 20},
            ],
        },
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "robot_task_intent"


def test_interaction_gate_responds_to_stop_gesture() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "pose",
            "observed_at": time.time(),
            "objects": [{"class_id": "person", "distance_m": 1.2, "gesture": "stop"}],
        },
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "safety_stop_gesture"


def test_interaction_gate_records_when_person_is_not_facing_robot() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "pose",
            "observed_at": time.time(),
            "objects": [
                {
                    "class_id": "person",
                    "distance_m": 1.2,
                    "orientation": "away",
                }
            ],
        },
    )

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.reason == "person_not_facing_robot"
    assert decision.should_record_environment is True


def test_interaction_gate_records_disengaged_posture() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "hello",
        addressed=True,
        perception={
            "source": "pose",
            "observed_at": time.time(),
            "objects": [
                {
                    "class_id": "person",
                    "distance_m": 1.2,
                    "posture": "walking_away",
                }
            ],
        },
    )

    assert decision.action == InteractionAction.RECORD_ONLY
    assert decision.reason == "disengaged_posture"


def test_interaction_gate_uses_engaged_posture_as_auditable_reason() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "can you explain this",
        addressed=True,
        perception={
            "source": "pose",
            "observed_at": time.time(),
            "objects": [
                {
                    "class_id": "person",
                    "distance_m": 1.2,
                    "posture": "leaning_forward",
                }
            ],
        },
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "engaged_posture"


def test_interaction_gate_uses_attention_gesture_as_auditable_reason() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "can you explain this",
        addressed=True,
        perception={
            "source": "pose",
            "observed_at": time.time(),
            "objects": [{"class_id": "person", "distance_m": 1.2, "gesture": "wave"}],
        },
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "attention_gesture"


def test_interaction_gate_refuses_unsafe_request_before_soft_perception_rules() -> None:
    gate = InteractionGate({"enabled": True, "max_interaction_distance_m": 3.0})

    decision = gate.evaluate(
        "\u62cd\u4e00\u4e0b\u90a3\u4e2a\u4eba",
        addressed=True,
        perception={
            "source": "vision",
            "observed_at": time.time(),
            "objects": [{"class_id": "person", "distance_m": 9.0}],
        },
    )

    assert decision.action == InteractionAction.REFUSE
    assert decision.reason == "unsafe_or_privacy_intent"


def test_interaction_gate_allows_explicit_stop_even_with_stale_perception() -> None:
    gate = InteractionGate({"enabled": True})

    decision = gate.evaluate(
        "\u505c\u4e0b",
        addressed=False,
        perception={
            "source": "vision",
            "observed_at": 100.0,
            "objects": [],
        },
    )

    assert decision.action == InteractionAction.RESPOND
    assert decision.reason == "robot_task_intent"
