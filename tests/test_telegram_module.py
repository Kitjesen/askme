from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from askme.runtime.modules.telegram_module import TelegramModule


class _RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def process(self, text: str, **kwargs: Any) -> str:
        self.calls.append((text, kwargs))
        return f"reply:{text}"


class _Message:
    def __init__(self, text: str) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def _text_update(*, chat_id: int, user_id: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
        message=_Message(text),
    )


@pytest.mark.asyncio
async def test_text_turns_use_a_stable_chat_scoped_conversation_session() -> None:
    pipeline = _RecordingPipeline()
    module = TelegramModule()
    module._pipeline = pipeline
    module._allowed_users = []

    first = _text_update(chat_id=987654321, user_id=1, text="first")
    second = _text_update(chat_id=987654321, user_id=2, text="second")
    other_chat = _text_update(chat_id=123456789, user_id=1, text="other")

    await module._handle_text(first, None)
    await module._handle_text(second, None)
    await module._handle_text(other_chat, None)

    first_context = pipeline.calls[0][1]
    second_context = pipeline.calls[1][1]
    other_context = pipeline.calls[2][1]

    assert first_context["source"] == "telegram"
    first_session = first_context["conversation_session_id"]
    other_session = other_context["conversation_session_id"]
    assert re.fullmatch(r"telegram:chat:[0-9a-f]{24}", first_session)
    assert "987654321" not in first_session
    assert second_context["conversation_session_id"] == first_session
    assert re.fullmatch(r"telegram:chat:[0-9a-f]{24}", other_session)
    assert "123456789" not in other_session
    assert other_session != first_session


class _IdentityAwarePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def process(
        self,
        text: str,
        *,
        source: str,
        conversation_session_id: str,
        person_id: str,
        operator_id: str,
        metadata: dict[str, str],
    ) -> str:
        self.calls.append(
            {
                "text": text,
                "source": source,
                "conversation_session_id": conversation_session_id,
                "person_id": person_id,
                "operator_id": operator_id,
                "metadata": metadata,
            }
        )
        return f"reply:{text}"


class _StrictPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def process(
        self,
        text: str,
        *,
        source: str,
        conversation_session_id: str,
    ) -> str:
        self.calls.append((text, source, conversation_session_id))
        return f"reply:{text}"


class _VoiceTelegramModule(TelegramModule):
    async def _transcribe_voice(self, update: Any, context: Any) -> str:
        return "spoken"


@pytest.mark.asyncio
async def test_supported_identity_context_is_stable_and_contains_no_raw_ids() -> None:
    pipeline = _IdentityAwarePipeline()
    module = TelegramModule()
    module._pipeline = pipeline
    module._allowed_users = []
    first = _text_update(
        chat_id=246813579,
        user_id=975318642,
        text="identity",
    )
    same_user = _text_update(
        chat_id=864297531,
        user_id=975318642,
        text="same user",
    )
    other_user = _text_update(
        chat_id=246813579,
        user_id=192837465,
        text="other user",
    )

    await module._handle_text(first, None)
    await module._handle_text(same_user, None)
    await module._handle_text(other_user, None)

    first_call, same_user_call, other_user_call = pipeline.calls
    person_id = first_call["person_id"]
    assert first_call["source"] == "telegram"
    assert re.fullmatch(r"telegram:user:[0-9a-f]{24}", person_id)
    assert "975318642" not in person_id
    assert first_call["operator_id"] == person_id
    assert same_user_call["person_id"] == person_id
    assert other_user_call["person_id"] != person_id
    assert first_call["metadata"] == {"channel": "telegram"}
    serialized_calls = repr(pipeline.calls)
    for raw_id in ("246813579", "864297531", "975318642", "192837465"):
        assert raw_id not in serialized_calls


@pytest.mark.asyncio
async def test_pipeline_without_identity_keywords_remains_supported() -> None:
    pipeline = _StrictPipeline()
    module = TelegramModule()
    module._pipeline = pipeline
    module._allowed_users = []
    update = _text_update(chat_id=1122334455, user_id=5566778899, text="strict")

    await module._handle_text(update, None)

    assert len(pipeline.calls) == 1
    text, source, session_id = pipeline.calls[0]
    assert (text, source) == ("strict", "telegram")
    assert re.fullmatch(r"telegram:chat:[0-9a-f]{24}", session_id)
    assert update.message.replies == ["reply:strict"]


@pytest.mark.asyncio
async def test_text_and_voice_turns_in_the_same_chat_share_a_session() -> None:
    pipeline = _RecordingPipeline()
    module = _VoiceTelegramModule()
    module._pipeline = pipeline
    module._allowed_users = []
    text_update = _text_update(chat_id=314159265, user_id=271828182, text="typed")
    voice_update = _text_update(chat_id=314159265, user_id=271828182, text="")

    await module._handle_text(text_update, None)
    await module._handle_voice(voice_update, None)

    assert [call[0] for call in pipeline.calls] == ["typed", "spoken"]
    assert (
        pipeline.calls[0][1]["conversation_session_id"]
        == pipeline.calls[1][1]["conversation_session_id"]
    )
