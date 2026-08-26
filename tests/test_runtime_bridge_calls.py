from unittest.mock import AsyncMock

import pytest

from askme.conversation import VoiceTurnLedger
from askme.pipeline.channels.runtime_bridge_calls import (
    call_bridge_turn,
    handle_runtime_bridge_result,
    runtime_bridge_result_outcome,
    try_runtime_bridge_turn,
)
from askme.voice_gateway import VoiceGatewayService


def test_call_bridge_turn_maps_conversation_session_to_session_id() -> None:
    captured = {}

    def bridge_method(text: str, *, session_id: str | None = None):
        captured["text"] = text
        captured["session_id"] = session_id
        return {"handled": True}

    result = call_bridge_turn(
        bridge_method,
        "status",
        conversation_session_id="conv-1",
    )

    assert result == {"handled": True}
    assert captured == {"text": "status", "session_id": "conv-1"}


def test_call_bridge_turn_passes_aliases_to_kwargs_bridge() -> None:
    captured = {}

    def bridge_method(text: str, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return {"handled": True}

    result = call_bridge_turn(
        bridge_method,
        "status",
        conversation_session_id="conv-1",
    )

    assert result == {"handled": True}
    assert captured["conversation_session_id"] == "conv-1"
    assert captured["session_id"] == "conv-1"


@pytest.mark.asyncio
async def test_handle_runtime_bridge_result_dispatches_general_skill() -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def dispatch(self, skill_name: str, user_text: str, *, source: str) -> None:
            self.calls.append((skill_name, user_text, source))

    dispatcher = Dispatcher()

    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": {"action_type": "general", "skill_name": "get_time"}},
        user_text="what time is it",
        conversation_session_id="thread-legacy-dispatcher",
        pipeline=object(),
        dispatcher=dispatcher,
        label="Text",
    )

    assert handled is True
    assert dispatcher.calls == [("get_time", "what time is it", "runtime")]


@pytest.mark.asyncio
async def test_runtime_skill_dispatch_receives_conversation_session_id() -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, str]] = []

        async def dispatch(
            self,
            skill_name: str,
            user_text: str,
            *,
            source: str,
            conversation_session_id: str,
        ) -> str:
            self.calls.append((skill_name, user_text, source, conversation_session_id))
            return "done"

    dispatcher = Dispatcher()

    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": {"action_type": "skill", "skill_name": "patrol"}},
        user_text="patrol zone a",
        conversation_session_id="thread-runtime-skill",
        pipeline=object(),
        dispatcher=dispatcher,
        label="Voice",
    )

    assert handled is True
    assert dispatcher.calls == [("patrol", "patrol zone a", "runtime", "thread-runtime-skill")]


@pytest.mark.asyncio
async def test_runtime_skill_dispatch_forwards_supported_caller_context() -> None:
    bridge_context: dict[str, object] = {}

    def bridge_method(
        text: str,
        *,
        conversation_session_id: str,
        voice_turn_id: str,
        turn_cancel_token: object,
        person_id: str,
        operator_id: str,
        robot_id: str,
        site_id: str,
        metadata: dict[str, object],
        defer_recording: bool,
    ) -> dict[str, object]:
        bridge_context.update(
            {
                "text": text,
                "conversation_session_id": conversation_session_id,
                "voice_turn_id": voice_turn_id,
                "turn_cancel_token": turn_cancel_token,
                "person_id": person_id,
                "operator_id": operator_id,
                "robot_id": robot_id,
                "site_id": site_id,
                "metadata": dict(metadata),
                "defer_recording": defer_recording,
            }
        )
        metadata["operator_id"] = "untrusted-bridge-mutation"
        return {
            "handled": True,
            "turn": {
                "action_type": "skill",
                "skill_name": "patrol",
                "operator_id": "untrusted-bridge-actor",
            },
        }

    class Dispatcher:
        def __init__(self) -> None:
            self.context: dict[str, object] = {}

        async def dispatch(
            self,
            skill_name: str,
            user_text: str,
            *,
            source: str,
            conversation_session_id: str,
            voice_turn_id: str,
            turn_cancel_token: object,
            person_id: str,
            operator_id: str,
            robot_id: str,
            site_id: str,
            metadata: dict[str, object],
        ) -> str:
            self.context = {
                "skill_name": skill_name,
                "user_text": user_text,
                "source": source,
                "conversation_session_id": conversation_session_id,
                "voice_turn_id": voice_turn_id,
                "turn_cancel_token": turn_cancel_token,
                "person_id": person_id,
                "operator_id": operator_id,
                "robot_id": robot_id,
                "site_id": site_id,
                "metadata": metadata,
            }
            return "done"

    dispatcher = Dispatcher()
    cancel_token = object()
    metadata = {"locale": "zh-CN"}

    outcome = await try_runtime_bridge_turn(
        bridge_method,
        "patrol zone a",
        conversation_session_id="thread-runtime-skill",
        voice_turn_id="turn-runtime-skill",
        turn_cancel_token=cancel_token,
        person_id="person-1",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
        metadata=metadata,
        pipeline=object(),
        dispatcher=dispatcher,
        label="Voice",
    )

    assert outcome.handled is True
    assert metadata == {"locale": "zh-CN"}
    assert bridge_context == {
        "text": "patrol zone a",
        "conversation_session_id": "thread-runtime-skill",
        "voice_turn_id": "turn-runtime-skill",
        "turn_cancel_token": cancel_token,
        "person_id": "person-1",
        "operator_id": "operator-1",
        "robot_id": "robot-1",
        "site_id": "site-a",
        "metadata": {"locale": "zh-CN"},
        "defer_recording": True,
    }
    assert dispatcher.context == {
        "skill_name": "patrol",
        "user_text": "patrol zone a",
        "source": "runtime",
        "conversation_session_id": "thread-runtime-skill",
        "voice_turn_id": "turn-runtime-skill",
        "turn_cancel_token": cancel_token,
        "person_id": "person-1",
        "operator_id": "operator-1",
        "robot_id": "robot-1",
        "site_id": "site-a",
        "metadata": {"locale": "zh-CN"},
    }


@pytest.mark.asyncio
async def test_runtime_skill_dispatch_keeps_strict_legacy_async_fake_compatible() -> None:
    calls: list[tuple[str, str, str]] = []

    async def legacy_dispatch(
        skill_name: str,
        user_text: str,
        *,
        source: str,
    ) -> str:
        calls.append((skill_name, user_text, source))
        return "done"

    class Dispatcher:
        def __init__(self) -> None:
            self.dispatch = AsyncMock(side_effect=legacy_dispatch)

    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": {"action_type": "skill", "skill_name": "patrol"}},
        user_text="patrol zone a",
        conversation_session_id="thread-runtime-skill",
        voice_turn_id="turn-runtime-skill",
        turn_cancel_token=object(),
        person_id="person-1",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
        metadata={"locale": "zh-CN"},
        pipeline=object(),
        dispatcher=Dispatcher(),
        label="Voice",
    )

    assert handled is True
    assert calls == [("patrol", "patrol zone a", "runtime")]


@pytest.mark.asyncio
async def test_runtime_skill_leaves_one_canonical_turn_to_pipeline(
    tmp_path,
) -> None:
    class Pipeline:
        def __init__(self) -> None:
            self.turn_ledger = VoiceTurnLedger(tmp_path / "runtime-skill.jsonl")

        async def execute_skill(
            self,
            skill_name: str,
            user_text: str,
            *,
            source: str,
            conversation_session_id: str,
            voice_turn_id: str,
            turn_cancel_token: object,
        ) -> str:
            del skill_name, turn_cancel_token
            thread = self.turn_ledger.resolve_thread(
                conversation_session_id=conversation_session_id,
                channel="voice",
            )
            turn = self.turn_ledger.start_turn(
                thread.thread_id,
                turn_id=voice_turn_id,
                source=source,
                user_text=user_text,
            )
            self.turn_ledger.commit_turn(
                turn.turn_id,
                assistant_text="patrol complete",
                heard_text="patrol complete",
            )
            return "patrol complete"

    pipeline = Pipeline()

    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": {"action_type": "skill", "skill_name": "patrol"}},
        user_text="patrol zone a",
        conversation_session_id="thread-runtime-skill",
        voice_turn_id="turn-runtime-skill",
        turn_cancel_token=object(),
        person_id="person-1",
        operator_id="operator-1",
        robot_id="robot-1",
        site_id="site-a",
        metadata={"locale": "zh-CN"},
        pipeline=pipeline,
        label="Voice",
    )

    turns = pipeline.turn_ledger.list_turns(thread_id="thread-runtime-skill")
    assert handled is True
    assert len(turns) == 1
    assert turns[0].turn_id == "turn-runtime-skill"
    assert turns[0].assistant_text == "patrol complete"


@pytest.mark.asyncio
async def test_handle_runtime_bridge_result_records_spoken_reply() -> None:
    class Conversation:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def add_user_message(self, text: str) -> None:
            self.messages.append(("user", text))

        def add_assistant_message(self, text: str) -> None:
            self.messages.append(("assistant", text))

    class Pipeline:
        def __init__(self) -> None:
            self._conversation = Conversation()

    pipeline = Pipeline()
    replies: list[str] = []

    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": {"spoken_reply": "runtime ok"}},
        user_text="status",
        pipeline=pipeline,
        on_spoken_reply=lambda reply: replies.append(reply),
        label="Voice",
    )

    assert handled is True
    assert replies == ["runtime ok"]
    assert pipeline._conversation.messages == [
        ("user", "status"),
        ("assistant", "runtime ok"),
    ]


@pytest.mark.asyncio
async def test_runtime_reply_is_not_committed_when_delivery_fails(tmp_path) -> None:
    class Conversation:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def add_user_message(self, text: str) -> None:
            self.messages.append(("user", text))

        def add_assistant_message(self, text: str) -> None:
            self.messages.append(("assistant", text))

    class Pipeline:
        def __init__(self) -> None:
            self._turn_ledger = VoiceTurnLedger(tmp_path / "runtime-delivery.jsonl")
            self._conversation = Conversation()
            self._episodic = None

    async def fail_delivery(_reply: str) -> None:
        raise RuntimeError("speaker failed")

    pipeline = Pipeline()
    with pytest.raises(RuntimeError, match="speaker failed"):
        await handle_runtime_bridge_result(
            {"handled": True, "turn": {"spoken_reply": "not delivered"}},
            user_text="status",
            conversation_session_id="thread-runtime",
            voice_turn_id="turn-runtime-delivery",
            pipeline=pipeline,
            on_spoken_reply=fail_delivery,
            label="Voice",
        )

    turn = pipeline._turn_ledger.list_turns(thread_id="thread-runtime")[0]
    assert turn.turn_id == "turn-runtime-delivery"
    assert turn.status.value == "cancelled"
    assert turn.assistant_text == ""
    assert pipeline._conversation.messages == [("user", "status")]


@pytest.mark.asyncio
async def test_gateway_history_waits_for_runtime_reply_delivery() -> None:
    class Bridge:
        def handle_voice_text(self, text: str, **kwargs):
            return {"handled": True, "turn": {"spoken_reply": f"reply:{text}"}}

        def status_snapshot(self):
            return {"enabled": True}

    gateway = VoiceGatewayService(Bridge())

    async def fail_delivery(_reply: str) -> None:
        raise RuntimeError("speaker failed")

    with pytest.raises(RuntimeError, match="speaker failed"):
        await try_runtime_bridge_turn(
            gateway.handle_voice_text,
            "first",
            conversation_session_id="thread-gateway-delivery",
            pipeline=object(),
            on_spoken_reply=fail_delivery,
            label="Voice",
        )

    failed_snapshot = gateway.conversation_snapshot("thread-gateway-delivery")
    assert failed_snapshot is not None
    assert failed_snapshot.turns == ()

    outcome = await try_runtime_bridge_turn(
        gateway.handle_voice_text,
        "second",
        conversation_session_id="thread-gateway-delivery",
        pipeline=object(),
        on_spoken_reply=lambda _reply: None,
        label="Voice",
    )

    assert outcome.handled is True
    delivered_snapshot = gateway.conversation_snapshot("thread-gateway-delivery")
    assert delivered_snapshot is not None
    assert len(delivered_snapshot.turns) == 1
    assert delivered_snapshot.turns[0].assistant_text == "reply:second"


@pytest.mark.asyncio
async def test_handle_runtime_bridge_result_invalid_payload_falls_back() -> None:
    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": None},
        user_text="status",
        pipeline=object(),
        label="Voice",
    )

    assert handled is False


@pytest.mark.asyncio
async def test_runtime_bridge_outcome_marks_malformed_handled_payload_ambiguous() -> None:
    outcome = await runtime_bridge_result_outcome(
        {"handled": True, "turn": None},
        user_text="status",
        pipeline=object(),
        label="Voice",
    )

    assert outcome.handled is False
    assert outcome.ambiguous is True
    assert outcome.explicitly_declined is False


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge_result", [None, "invalid", {}, {"handled": None}])
async def test_runtime_bridge_outcome_treats_missing_or_untyped_result_as_ambiguous(
    bridge_result,
) -> None:
    outcome = await runtime_bridge_result_outcome(
        bridge_result,
        user_text="status",
        pipeline=object(),
        label="Voice",
    )

    assert outcome.ambiguous is True
    assert outcome.explicitly_declined is False


@pytest.mark.asyncio
async def test_runtime_bridge_outcome_requires_explicit_handled_false_to_decline() -> None:
    outcome = await runtime_bridge_result_outcome(
        {"handled": False, "reason": "not_supported"},
        user_text="status",
        pipeline=object(),
        label="Voice",
    )

    assert outcome.explicitly_declined is True
    assert outcome.ambiguous is False


@pytest.mark.asyncio
async def test_runtime_bridge_outcome_fences_untracked_agent_task_dispatch() -> None:
    dispatcher = AsyncMock()

    outcome = await runtime_bridge_result_outcome(
        {"handled": True, "turn": {"action_type": "skill", "skill_name": "agent_task"}},
        user_text="生成状态报告",
        pipeline=object(),
        dispatcher=dispatcher,
        label="Voice",
        allow_agent_task_dispatch=False,
    )

    assert outcome.ambiguous is True
    dispatcher.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_runtime_bridge_turn_reraises_unexpected_bridge_bug() -> None:
    def bridge_method(text: str):
        raise RuntimeError(f"provider bug: {text}")

    with pytest.raises(RuntimeError, match="provider bug"):
        await try_runtime_bridge_turn(
            bridge_method,
            "status",
            pipeline=object(),
            label="Voice",
        )
