from __future__ import annotations

from askme.voice.realtime.policy import decide_realtime_route


def test_split_mode_always_uses_existing_cascade() -> None:
    decision = decide_realtime_route(
        mode="split",
        interaction_admitted=True,
        intent_type="general",
        provider_ready=True,
    )

    assert decision.route == "cascade"
    assert decision.allow_provider_audio is False
    assert decision.reason == "realtime_disabled"


def test_shadow_mode_never_plays_provider_audio() -> None:
    decision = decide_realtime_route(
        mode="shadow",
        interaction_admitted=True,
        intent_type="general",
        provider_ready=True,
        provider="volcengine_s2s",
    )

    assert decision.route == "shadow"
    assert decision.allow_provider_audio is False
    assert decision.interrupt_provider is False


def test_general_chat_can_use_s2s_only_after_all_safety_gates() -> None:
    decision = decide_realtime_route(
        mode="general_chat",
        interaction_admitted=True,
        intent_type="general",
        provider_ready=True,
        provider="volcengine_s2s",
    )

    assert decision.route == "volcengine_s2s"
    assert decision.allow_provider_audio is True
    assert decision.interrupt_provider is False


def test_general_chat_route_preserves_selected_realtime_provider() -> None:
    decision = decide_realtime_route(
        mode="general_chat",
        interaction_admitted=True,
        intent_type="general",
        provider_ready=True,
        provider="qwen3_5_omni",
    )

    assert decision.route == "qwen3_5_omni"
    assert decision.allow_provider_audio is True


def test_unknown_realtime_provider_fails_closed_to_cascade() -> None:
    decision = decide_realtime_route(
        mode="general_chat",
        interaction_admitted=True,
        intent_type="general",
        provider_ready=True,
        provider="unknown_provider",
    )

    assert decision.route == "cascade"
    assert decision.allow_provider_audio is False
    assert decision.interrupt_provider is True
    assert decision.reason == "unsupported_provider"


def test_robot_task_never_uses_s2s_generated_response() -> None:
    decision = decide_realtime_route(
        mode="general_chat",
        interaction_admitted=True,
        intent_type="voice_trigger",
        provider_ready=True,
        provider="volcengine_s2s",
        robot_task=True,
    )

    assert decision.route == "cascade"
    assert decision.allow_provider_audio is False
    assert decision.interrupt_provider is True
    assert decision.reason == "robot_or_tool_route"


def test_estop_pending_approval_and_rejected_ambient_speech_fail_closed() -> None:
    cases = [
        {"emergency": True, "reason": "emergency"},
        {"pending_approval": True, "reason": "pending_approval"},
        {"interaction_admitted": False, "reason": "interaction_not_admitted"},
        {"provider_ready": False, "reason": "provider_unavailable"},
    ]

    for case in cases:
        reason = case.pop("reason")
        decision = decide_realtime_route(
            mode="general_chat",
            interaction_admitted=case.pop("interaction_admitted", True),
            intent_type="general",
            provider_ready=case.pop("provider_ready", True),
            **case,
        )
        assert decision.route == "cascade"
        assert decision.allow_provider_audio is False
        assert decision.interrupt_provider is True
        assert decision.reason == reason
