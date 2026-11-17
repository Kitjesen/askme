"""Voice gateway can build and send a Xiaozhi emotion message."""

from __future__ import annotations

import json

from askme.voice_gateway.service import VoiceGatewayService


def test_push_device_emotion_builds_the_official_payload() -> None:
    gateway = VoiceGatewayService()

    message = gateway.push_device_emotion("没有表情", session_id="sess-7")

    assert message == {
        "type": "llm",
        "text": "🙂",
        "emotion": "happy",
        "session_id": "sess-7",
    }


def test_recorded_turn_pushes_emotion_only_when_a_sender_is_registered() -> None:
    quiet = VoiceGatewayService()
    assert quiet.record_local_turn(
        "sess-quiet",
        user_text="巡检",
        assistant_text="完成了😂",
    )

    sent: list[str] = []
    gateway = VoiceGatewayService()
    gateway.set_device_emotion_sender(sent.append)
    assert gateway.record_local_turn(
        "sess-9",
        user_text="巡检",
        assistant_text="完成了😂",
    )

    assert len(sent) == 1
    assert json.loads(sent[0]) == {
        "type": "llm",
        "text": "😂",
        "emotion": "funny",
        "session_id": "sess-9",
    }
    snapshot = gateway.conversation_snapshot("sess-9")
    assert snapshot is not None
    assert snapshot.turns[0].assistant_text == "完成了😂"


def test_handled_reply_pushes_emotion_with_the_conversation_session() -> None:
    class Bridge:
        def handle_voice_text(self, text: str, **_kwargs: object) -> dict[str, object]:
            del text
            return {"handled": True, "turn": {"spoken_reply": "收到😭"}}

    sent: list[str] = []
    gateway = VoiceGatewayService(Bridge())  # type: ignore[arg-type]
    gateway.set_device_emotion_sender(sent.append)

    result = gateway.handle_voice_text("你好", conversation_session_id="sess-2")

    assert result is not None
    assert result["handled"] is True
    assert json.loads(sent[0])["emotion"] == "crying"
    assert json.loads(sent[0])["text"] == "😭"
    assert json.loads(sent[0])["session_id"] == "sess-2"
    assert json.loads(sent[0])["type"] == "llm"
