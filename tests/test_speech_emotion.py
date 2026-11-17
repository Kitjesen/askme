"""Device emotion extraction and the Xiaozhi llm payload."""

from __future__ import annotations

import json

import pytest

from askme.voice.speech.emotion import (
    DEFAULT_EMOJI,
    DEFAULT_EMOTION,
    EMOJI_MAP,
    build_llm_emotion_message,
    dumps_llm_emotion_message,
    extract_emotion,
    push_device_emotion,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("emoji", "emotion"), list(EMOJI_MAP.items()))
def test_every_official_emoji_maps_to_its_emotion(emoji: str, emotion: str) -> None:
    found = extract_emotion(f"前缀{emoji}后缀")

    assert found.emoji == emoji
    assert found.emotion == emotion


def test_laughing_face_is_funny_not_laughing() -> None:
    assert extract_emotion("😂") == extract_emotion("真的😂")
    assert extract_emotion("😂").emotion == "funny"
    assert extract_emotion("😆").emotion == "laughing"


def test_first_mapped_emoji_wins() -> None:
    found = extract_emotion("先😂再😭最后😠")

    assert found.emoji == "😂"
    assert found.emotion == "funny"


def test_no_emoji_defaults_to_happy() -> None:
    found = extract_emotion("今天巡检正常")

    assert found.emoji == DEFAULT_EMOJI
    assert found.emotion == DEFAULT_EMOTION
    assert extract_emotion("").emotion == "happy"


def test_unknown_emoji_is_ignored() -> None:
    assert extract_emotion("🚀🌟").emotion == "happy"
    assert extract_emotion("🚀😔").emotion == "sad"
    assert extract_emotion("🙂😭").emotion == "happy"


def test_payload_shape_and_session_id() -> None:
    message = build_llm_emotion_message("收到😭", session_id="sess-42")

    assert message == {
        "type": "llm",
        "text": "😭",
        "emotion": "crying",
        "session_id": "sess-42",
    }
    assert set(message) == {"type", "text", "emotion", "session_id"}
    raw = dumps_llm_emotion_message(message)
    assert json.loads(raw) == message


def test_missing_session_id_is_an_empty_string() -> None:
    message = build_llm_emotion_message("没有表情")

    assert message["session_id"] == ""
    assert message["text"] == "🙂"
    assert message["emotion"] == "happy"


def test_push_calls_a_callable_and_a_send_method() -> None:
    sent: list[str] = []

    class Socket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, raw: str) -> None:
            self.sent.append(raw)

    payload = push_device_emotion("😎", "dog-1", sent.append)
    socket = Socket()
    push_device_emotion("😎", "dog-1", socket)

    assert payload["emotion"] == "cool"
    assert payload["session_id"] == "dog-1"
    assert json.loads(sent[0]) == payload
    assert json.loads(socket.sent[0]) == payload


def test_push_without_a_sender_still_returns_the_message() -> None:
    message = push_device_emotion("没有设备", "sess")

    assert message["emotion"] == "happy"
    assert message["session_id"] == "sess"


def test_push_rejects_an_object_that_cannot_send(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        message = push_device_emotion("😂", "sess", object())

    assert message["emotion"] == "funny"
    assert "发送情绪表情失败" in caplog.text


def test_push_swallows_sender_errors(caplog: pytest.LogCaptureFixture) -> None:
    def fail(_raw: str) -> None:
        raise ConnectionError("socket closed")

    with caplog.at_level("WARNING"):
        message = push_device_emotion("😂", "sess", fail)

    assert message["emotion"] == "funny"
    assert "发送情绪表情失败" in caplog.text
