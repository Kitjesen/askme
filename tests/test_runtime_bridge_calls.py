import pytest

from askme.pipeline.channels.runtime_bridge_calls import (
    call_bridge_turn,
    handle_runtime_bridge_result,
    try_runtime_bridge_turn,
)


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
        pipeline=object(),
        dispatcher=dispatcher,
        label="Text",
    )

    assert handled is True
    assert dispatcher.calls == [("get_time", "what time is it", "runtime")]


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
async def test_handle_runtime_bridge_result_invalid_payload_falls_back() -> None:
    handled = await handle_runtime_bridge_result(
        {"handled": True, "turn": None},
        user_text="status",
        pipeline=object(),
        label="Voice",
    )

    assert handled is False


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
