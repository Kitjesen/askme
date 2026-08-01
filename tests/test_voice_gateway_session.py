from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from askme.voice_gateway.service import VoiceGatewayService
from askme.voice_gateway.session import ConversationSessionManager


class StepClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 5, 20, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def test_get_or_create_creates_and_reuses_active_session() -> None:
    manager = ConversationSessionManager(clock=StepClock())

    first = manager.get_or_create(
        channel="voice",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
        active_planning_session_id="plan-1",
        metadata={"locale": "zh-CN"},
    )
    second = manager.get_or_create(
        channel="voice",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
    )

    assert second.session_id == first.session_id
    assert second.channel == "voice"
    assert second.operator_id == "operator-1"
    assert second.robot_id == "robot-1"
    assert second.site_id == "site-a"
    assert second.status == "active"
    assert second.active_planning_session_id == "plan-1"
    assert second.metadata == {"locale": "zh-CN"}
    assert second.created_at.tzinfo is UTC


def test_get_or_create_honors_new_explicit_session_id() -> None:
    manager = ConversationSessionManager(clock=StepClock())

    first = manager.get_or_create(channel="text", session_id="conv-a")
    second = manager.get_or_create(channel="text", session_id="conv-b")

    assert first.session_id == "conv-a"
    assert second.session_id == "conv-b"
    assert first.session_id != second.session_id


def test_anonymous_sessions_are_fail_new_instead_of_reusing_process_history() -> None:
    manager = ConversationSessionManager(clock=StepClock())

    first = manager.get_or_create(channel="voice")
    manager.append_turn(
        first.session_id,
        user_text="private first request",
        assistant_text="private first response",
    )
    second = manager.get_or_create(channel="voice")

    assert second.session_id != first.session_id
    assert manager.context_payload(second.session_id)["turn_count"] == 0


@pytest.mark.parametrize(
    ("overrides", "conflicting_field"),
    [
        ({"channel": "text"}, "channel"),
        ({"operator_id": "operator-2"}, "operator_id"),
        ({"robot_id": "robot-2"}, "robot_id"),
        ({"site_id": "site-b"}, "site_id"),
    ],
)
def test_explicit_session_id_rejects_identity_takeover(
    overrides: dict[str, str],
    conflicting_field: str,
) -> None:
    manager = ConversationSessionManager(clock=StepClock())
    original = manager.get_or_create(
        channel="voice",
        session_id="private-session",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
    )
    request = {
        "channel": "voice",
        "session_id": original.session_id,
        "operator_id": "operator-1",
        "robot_id": "robot-1",
        "site_id": "site-a",
        **overrides,
    }

    with pytest.raises(ValueError, match=conflicting_field):
        manager.get_or_create(**request)

    snapshot = manager.snapshot(original.session_id)
    assert snapshot is not None
    assert snapshot.operator_id == "operator-1"
    assert snapshot.robot_id == "robot-1"
    assert snapshot.site_id == "site-a"


def test_append_turn_updates_timestamp_and_summary() -> None:
    clock = StepClock()
    manager = ConversationSessionManager(clock=clock)
    session = manager.get_or_create(channel="voice")
    original_updated_at = session.updated_at

    turn = manager.append_turn(
        session.session_id,
        user_text="Where is the lobby?",
        assistant_text="The lobby is ahead.",
        intent="wayfinding",
        gate_decision="allow",
        skill_name="lookup_place",
        tool_calls=[{"name": "lookup_place", "args": {"place": "lobby"}}],
        handoff_id="handoff-1",
        metadata={"confidence": 0.91},
    )
    snapshot = manager.snapshot(session.session_id)

    assert snapshot is not None
    assert len(snapshot.turns) == 1
    assert snapshot.turns[0].turn_id == turn.turn_id
    assert snapshot.turns[0].intent == "wayfinding"
    assert snapshot.turns[0].gate_decision == "allow"
    assert snapshot.turns[0].skill_name == "lookup_place"
    assert snapshot.turns[0].handoff_id == "handoff-1"
    assert snapshot.turns[0].tool_calls == [{"name": "lookup_place", "args": {"place": "lobby"}}]
    assert snapshot.turns[0].metadata == {"confidence": 0.91}
    assert snapshot.updated_at > original_updated_at
    assert snapshot.last_activity_at == snapshot.updated_at
    assert snapshot.summary == "User: Where is the lobby? | Assistant: The lobby is ahead."
    assert snapshot.turns[0].created_at.tzinfo is UTC


def test_expire_idle_sessions_closes_only_idle_active_sessions() -> None:
    manager = ConversationSessionManager(clock=StepClock())
    idle = manager.get_or_create(channel="voice", operator_id="idle")
    fresh = manager.get_or_create(channel="voice", operator_id="fresh")
    closed = manager.get_or_create(channel="voice", operator_id="closed")
    manager.close_session(closed.session_id)

    now = datetime(2026, 5, 20, 1, 0, tzinfo=UTC)
    idle.last_activity_at = now - timedelta(minutes=31)
    fresh.last_activity_at = now - timedelta(minutes=5)
    closed.last_activity_at = now - timedelta(hours=2)

    expired = manager.expire_idle_sessions(timedelta(minutes=30), now=now)

    assert expired == [idle.session_id]
    assert manager.snapshot(idle.session_id).status == "expired"
    assert manager.snapshot(fresh.session_id).status == "active"
    assert manager.snapshot(closed.session_id).status == "closed"
    assert manager.snapshot(idle.session_id).closed_at == now
    assert manager.snapshot(idle.session_id).close_reason == "expired"


def test_snapshot_protects_internal_mutable_state() -> None:
    manager = ConversationSessionManager(clock=StepClock())
    session = manager.get_or_create(channel="voice", metadata={"nested": {"live": True}})
    manager.append_turn(
        session.session_id,
        user_text="start",
        tool_calls=[{"name": "tool", "args": {"value": 1}}],
        metadata={"turn": {"safe": True}},
    )

    snapshot = manager.snapshot(session.session_id)
    assert snapshot is not None

    snapshot.metadata["nested"]["live"] = False
    snapshot.turns[0].tool_calls[0]["args"]["value"] = 99
    snapshot.turns[0].metadata["turn"]["safe"] = False

    fresh = manager.snapshot(session.session_id)
    assert fresh is not None
    assert fresh.metadata == {"nested": {"live": True}}
    assert fresh.turns[0].tool_calls == [{"name": "tool", "args": {"value": 1}}]
    assert fresh.turns[0].metadata == {"turn": {"safe": True}}


def test_context_payload_returns_recent_turns_and_session_state() -> None:
    manager = ConversationSessionManager(clock=StepClock())
    session = manager.get_or_create(
        channel="text",
        session_id="conv-1",
        active_planning_session_id="plan-1",
        current_task_id="task-1",
    )
    manager.append_turn(
        session.session_id,
        user_text="first",
        assistant_text="first reply",
    )
    manager.append_turn(
        session.session_id,
        user_text="second",
        assistant_text="second reply",
        handoff_id="handoff-1",
    )

    payload = manager.context_payload(
        session.session_id,
        recent_turn_limit=1,
        max_chars=80,
    )

    assert payload["session_id"] == "conv-1"
    assert payload["active_planning_session_id"] == "plan-1"
    assert payload["current_task_id"] == "task-1"
    assert payload["handoff_id"] == "handoff-1"
    assert payload["turn_count"] == 2
    assert [turn["sequence"] for turn in payload["recent_turns"]] == [2]
    assert "second" in payload["text"]
    assert "first" not in payload["text"]


def test_closed_session_cannot_be_reused_or_appended() -> None:
    manager = ConversationSessionManager(clock=StepClock())
    session = manager.get_or_create(channel="voice")
    manager.close_session(session.session_id, reason="operator_left")

    with pytest.raises(ValueError, match="not active"):
        manager.get_or_create(channel="voice", session_id=session.session_id)
    with pytest.raises(ValueError, match="not active"):
        manager.append_turn(session.session_id, user_text="should not append")

    snapshot = manager.snapshot(session.session_id)
    assert snapshot is not None
    assert snapshot.status == "closed"
    assert snapshot.close_reason == "operator_left"
    assert snapshot.turns == ()


def test_gateway_service_passes_conversation_session_to_bridge() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def status_snapshot(self) -> dict[str, object]:
            return {"enabled": True}

        def handle_text_input(
            self,
            text: str,
            *,
            session_id: str | None = None,
            channel: str | None = None,
            operator_id: str | None = None,
            robot_id: str | None = None,
            site_id: str | None = None,
            metadata: dict[str, object] | None = None,
        ) -> dict[str, object]:
            self.calls.append(
                {
                    "text": text,
                    "session_id": session_id,
                    "channel": channel,
                    "operator_id": operator_id,
                    "robot_id": robot_id,
                    "site_id": site_id,
                    "metadata": dict(metadata or {}),
                }
            )
            return {
                "handled": True,
                "turn": {
                    "spoken_reply": "ready",
                    "action_type": "general",
                    "planning_session_id": "plan-1",
                },
            }

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)

    result = gateway.handle_text_input(
        "inspect area A",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
        metadata={"locale": "zh-CN"},
        include_session=True,
    )

    assert result is not None
    conversation_session_id = result["conversation_session_id"]
    assert bridge.calls == [
        {
            "text": "inspect area A",
            "session_id": conversation_session_id,
            "channel": "text",
            "operator_id": "operator-1",
            "robot_id": "robot-1",
            "site_id": "site-a",
            "metadata": {"locale": "zh-CN"},
        }
    ]
    snapshot = gateway.conversation_snapshot(str(conversation_session_id))
    context = gateway.conversation_context(str(conversation_session_id))

    assert snapshot is not None
    assert snapshot.active_planning_session_id == "plan-1"
    assert snapshot.turns[0].user_text == "inspect area A"
    assert snapshot.turns[0].assistant_text == "ready"
    assert context["recent_turns"][0]["sequence"] == 1
    assert "inspect area A" in context["text"]


def test_gateway_anonymous_requests_do_not_share_conversation_context() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.contexts: list[dict[str, object]] = []

        def status_snapshot(self) -> dict[str, object]:
            return {"enabled": True}

        def handle_text_input(self, text: str, **kwargs) -> dict[str, object]:
            self.contexts.append(dict(kwargs["conversation_context"]))
            return {
                "handled": True,
                "turn": {"spoken_reply": f"reply:{text}"},
            }

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)

    first = gateway.handle_text_input("first private request", include_session=True)
    second = gateway.handle_text_input("second private request", include_session=True)

    assert first is not None
    assert second is not None
    assert first["conversation_session_id"] != second["conversation_session_id"]
    assert [context["turn_count"] for context in bridge.contexts] == [0, 0]
    assert all(context["text"] == "" for context in bridge.contexts)


def test_gateway_service_records_local_fallback_turns() -> None:
    gateway = VoiceGatewayService()

    recorded = gateway.record_local_turn(
        "conv-local",
        user_text="inspect zone",
        assistant_text="local reply",
        metadata={"reason": "bridge_unhandled"},
    )
    context = gateway.conversation_context("conv-local")
    snapshot = gateway.conversation_snapshot("conv-local")

    assert recorded is True
    assert snapshot is not None
    assert snapshot.turns[0].user_text == "inspect zone"
    assert snapshot.turns[0].assistant_text == "local reply"
    assert snapshot.turns[0].metadata == {
        "bridge_handled": False,
        "local_fallback": True,
        "reason": "bridge_unhandled",
    }
    assert context["turn_count"] == 1
    assert "inspect zone" in context["text"]


def test_gateway_deferred_handled_turn_waits_for_explicit_delivery_record() -> None:
    class Bridge:
        def status_snapshot(self) -> dict[str, object]:
            return {"enabled": True}

        def handle_text_input(self, text: str, **kwargs) -> dict[str, object]:
            return {
                "handled": True,
                "turn": {"action_type": "general"},
            }

    gateway = VoiceGatewayService(Bridge())

    result = gateway.handle_text_input(
        "inspect zone",
        conversation_session_id="conv-deferred",
        defer_recording=True,
        include_session=True,
    )
    before_delivery = gateway.conversation_snapshot("conv-deferred")

    assert result is not None
    assert result["conversation_recording_deferred"] is True
    assert before_delivery is not None
    assert before_delivery.turns == ()
    assert result["conversation_session"]["turn_count"] == 0

    recorded = gateway.record_local_turn(
        "conv-deferred",
        user_text="inspect zone",
        assistant_text="local reply",
        metadata={"delivery_confirmed": True},
    )
    after_delivery = gateway.conversation_snapshot("conv-deferred")

    assert recorded is True
    assert after_delivery is not None
    assert len(after_delivery.turns) == 1
    assert after_delivery.turns[0].assistant_text == "local reply"


def test_gateway_deferred_spoken_reply_is_not_projected_before_delivery() -> None:
    class Bridge:
        def status_snapshot(self) -> dict[str, object]:
            return {"enabled": True}

        def handle_text_input(self, text: str, **kwargs) -> dict[str, object]:
            return {
                "handled": True,
                "turn": {"spoken_reply": "bridge reply"},
            }

    gateway = VoiceGatewayService(Bridge())

    result = gateway.handle_text_input(
        "inspect zone",
        conversation_session_id="conv-deferred-spoken",
        defer_recording=True,
        include_session=True,
    )
    snapshot = gateway.conversation_snapshot("conv-deferred-spoken")

    assert result is not None
    assert result["conversation_recording_deferred"] is True
    assert result["conversation_session"]["turn_count"] == 0
    assert snapshot is not None
    assert snapshot.turns == ()


def test_gateway_keeps_conversation_and_planning_session_ids_separate_across_turns() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def status_snapshot(self) -> dict[str, object]:
            return {"enabled": True}

        def handle_text_input(self, text: str, **kwargs) -> dict[str, object]:
            self.calls.append({"text": text, **kwargs})
            return {
                "handled": True,
                "turn": {
                    "spoken_reply": "ok",
                    "planning_session_id": "plan-1",
                },
            }

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)

    first = gateway.handle_text_input("inspect area A", include_session=True)
    assert first is not None
    second = gateway.handle_text_input(
        "confirm",
        conversation_session_id=str(first["conversation_session_id"]),
        include_session=True,
    )
    assert second is not None

    assert first["conversation_session_id"] == second["conversation_session_id"]
    assert first["conversation_session_id"] != "plan-1"
    assert {call["session_id"] for call in bridge.calls} == {first["conversation_session_id"]}
    assert {call["conversation_session_id"] for call in bridge.calls} == {
        first["conversation_session_id"]
    }
    assert "planning_session_id" not in bridge.calls[0]
    assert bridge.calls[0]["conversation_context"]["turn_count"] == 0
    assert bridge.calls[1]["conversation_context"]["turn_count"] == 1
    assert "inspect area A" in bridge.calls[1]["conversation_context"]["text"]
    snapshot = gateway.conversation_snapshot(str(first["conversation_session_id"]))
    assert snapshot is not None
    assert snapshot.active_planning_session_id == "plan-1"


@pytest.mark.parametrize(
    ("gateway_method", "bridge_method", "channel"),
    [
        ("handle_text_input", "handle_text_input", "text"),
        ("handle_voice_text", "handle_voice_text", "voice"),
    ],
)
def test_gateway_forwards_admitted_interaction_context(
    gateway_method: str,
    bridge_method: str,
    channel: str,
) -> None:
    class Bridge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def status_snapshot(self) -> dict[str, object]:
            return {"enabled": True}

        def handle_text_input(self, text: str, **kwargs) -> dict[str, object]:
            self.calls.append(("handle_text_input", text, kwargs))
            return {"handled": False}

        def handle_voice_text(self, text: str, **kwargs) -> dict[str, object]:
            self.calls.append(("handle_voice_text", text, kwargs))
            return {"handled": False}

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    cancel_token = object()

    getattr(gateway, gateway_method)(
        "inspect zone",
        conversation_session_id="conv-context",
        voice_turn_id="turn-context",
        turn_cancel_token=cancel_token,
        person_id="person-context",
    )

    called_method, text, context = bridge.calls[0]
    assert called_method == bridge_method
    assert text == "inspect zone"
    assert context["conversation_session_id"] == "conv-context"
    assert context["voice_turn_id"] == "turn-context"
    assert context["turn_cancel_token"] is cancel_token
    assert context["person_id"] == "person-context"
    assert context["channel"] == channel
