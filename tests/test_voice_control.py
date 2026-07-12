"""Runtime voice-system control API contracts."""

from __future__ import annotations

from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from askme.runtime.voice_control import VoiceControlStateStore


class _VoiceControlHandler:
    def voice_profiles_payload(self) -> dict:
        return {"profiles": [], "active_profile": ""}

    def set_voice_profile_payload(self, payload: dict) -> dict:
        return {"updated": True, "active_profile": payload.get("profile_id", "")}

    def system_control_payload(self) -> dict:
        return {
            "status": "ready",
            "runtime": {"llm": {"provider": "deepseek", "model": "deepseek-v4-flash"}},
            "catalog": {},
            "prompt": {},
            "memory": {},
            "issues": [],
        }

    async def switch_system_component_payload(self, payload: dict) -> dict:
        return {
            "updated": True,
            "component": payload.get("component", ""),
            "state": "active",
        }

    def update_prompt_payload(self, payload: dict) -> dict:
        return {"updated": True, "prompt": payload}


def _health_snapshot() -> dict:
    return {
        "status": "ok",
        "voice_pipeline_status": {"pipeline_ok": True},
    }


def test_voice_system_control_routes_read_and_update_runtime():
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            voice_handler=_VoiceControlHandler(),
        )
    )

    snapshot = client.get("/api/voice/system")
    assert snapshot.status_code == 200
    assert snapshot.json()["runtime"]["llm"]["provider"] == "deepseek"

    switched = client.post(
        "/api/voice/system/switch",
        json={"component": "llm", "provider": "deepseek", "model": "deepseek-v4-flash", "validate": False},
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert switched.status_code == 200
    assert switched.json()["updated"] is True

    prompt = client.post(
        "/api/voice/system/prompt",
        json={"system_prompt": "你是小算。"},
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert prompt.status_code == 200
    assert prompt.json()["updated"] is True


def test_voice_system_control_update_requires_supervisor_permission():
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            voice_handler=_VoiceControlHandler(),
        )
    )

    response = client.post(
        "/api/voice/system/switch",
        json={"component": "tts", "backend": "edge"},
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )

    assert response.status_code == 403


def test_voice_control_state_is_disabled_without_explicit_path() -> None:
    store = VoiceControlStateStore({})

    assert store.enabled is False
    assert store.load() == {}
    assert store.save({"asr": {"provider": "local"}}) == {
        "asr": {"provider": "local"}
    }
