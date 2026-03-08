from askme.voice.runtime_bridge import VoiceRuntimeBridge


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self):
        return self._payload


def test_voice_runtime_bridge_posts_turn(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response({"handled": True, "turn": {"spoken_reply": "runtime handled"}})

    monkeypatch.setattr("askme.voice.runtime_bridge.requests.post", fake_post)

    bridge = VoiceRuntimeBridge(
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:5100",
            "api_key": "shared-secret",
            "operator_id": "askme.voice",
            "session_id": "voice-session",
            "channel": "voice",
            "robot_id": "dog-01",
            "site_id": "plant-a",
            "submit": True,
            "timeout": 1.5,
        }
    )

    result = bridge.handle_voice_text("start patrol")

    assert result["handled"] is True
    assert captured["url"] == "http://127.0.0.1:5100/api/voice/turns"
    assert captured["json"]["operator_id"] == "askme.voice"
    assert captured["headers"]["Authorization"] == "Bearer shared-secret"
    assert captured["json"]["channel"] == "voice"


def test_voice_runtime_bridge_posts_text_turn_with_text_channel(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        captured["headers"] = headers
        return _Response({"handled": True, "turn": {"spoken_reply": "status ok"}})

    monkeypatch.setattr("askme.voice.runtime_bridge.requests.post", fake_post)

    bridge = VoiceRuntimeBridge(
        {
            "enabled": True,
            "text_enabled": True,
            "base_url": "http://127.0.0.1:5100",
            "operator_id": "askme.console",
            "session_id": "voice-session",
            "channel": "voice",
            "text_session_id": "text-session",
            "text_channel": "text",
            "submit": True,
        }
    )

    result = bridge.handle_text_input("status")

    assert result["handled"] is True
    assert captured["json"]["operator_id"] == "askme.console"
    assert captured["json"]["session_id"] == "text-session"
    assert captured["json"]["channel"] == "text"


def test_voice_runtime_bridge_handle_turn_can_override_payload(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _Response({"handled": True, "turn": {"spoken_reply": "ok"}})

    monkeypatch.setattr("askme.voice.runtime_bridge.requests.post", fake_post)

    bridge = VoiceRuntimeBridge(
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:5100",
            "operator_id": "askme.voice",
            "session_id": "voice-session",
            "channel": "voice",
        }
    )

    bridge.handle_turn(
        "test",
        enabled=True,
        operator_id="override.operator",
        session_id="override-session",
        channel="text",
        robot_id="dog-02",
        site_id="plant-b",
        submit=False,
        metadata={"source": "unit-test"},
    )

    assert captured["json"]["operator_id"] == "override.operator"
    assert captured["json"]["session_id"] == "override-session"
    assert captured["json"]["channel"] == "text"
    assert captured["json"]["robot_id"] == "dog-02"
    assert captured["json"]["site_id"] == "plant-b"
    assert captured["json"]["submit"] is False
    assert captured["json"]["metadata"] == {"source": "unit-test"}


def test_voice_runtime_bridge_returns_none_on_invalid_json(monkeypatch) -> None:
    class _BadJsonResponse(_Response):
        def json(self):
            raise ValueError("invalid json")

    def fake_post(url, json, headers, timeout):
        return _BadJsonResponse({})

    monkeypatch.setattr("askme.voice.runtime_bridge.requests.post", fake_post)

    bridge = VoiceRuntimeBridge(
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:5100",
        }
    )

    assert bridge.handle_voice_text("status") is None
