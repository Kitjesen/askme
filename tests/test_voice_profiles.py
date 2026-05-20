"""Tests for selectable TTS voice profiles."""

from __future__ import annotations

from fastapi.testclient import TestClient

from askme.api.schemas.voice import VoiceProfileCatalogResponse, VoiceProfileUpdateResponse
from askme.health_server import build_health_snapshot, create_health_app
from askme.voice.tts import TTSEngine
from askme.voice.voice_profiles import build_voice_profiles, resolve_voice_profile_id


def _health_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="test",
        model_name="test-model",
        metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
        active_skills=[],
        voice_status={"enabled": True, "pipeline_ok": True},
    )


def test_voice_profiles_include_customer_use_cases():
    profiles = build_voice_profiles({}, default_voice_id="male-qn-qingse")

    assert {
        "patrol_default",
        "visitor_friendly",
        "security_clear",
        "emergency_short",
        "cleaning_soft",
        "operations_calm",
        "crowd_clear",
        "guide_leading",
        "fault_urgent",
        "confirm_clear",
    } <= set(profiles)
    assert profiles["patrol_default"].label == "巡检播报"
    assert profiles["visitor_friendly"].label == "访客服务"
    assert profiles["emergency_short"].cue == "emergency_tone"
    assert profiles["fault_urgent"].category == "emergency"
    assert profiles["visitor_friendly"].emotion == "happy"
    assert profiles["emergency_short"].speed > profiles["night_quiet"].speed
    assert profiles["cleaning_soft"].volume < profiles["emergency_short"].volume
    assert profiles["fault_urgent"].emotion == "fearful"
    assert resolve_voice_profile_id("emergency_alert") == "emergency_short"
    assert resolve_voice_profile_id("service_notice") == "visitor_friendly"
    assert resolve_voice_profile_id("cleaning_notice") == "cleaning_soft"
    assert resolve_voice_profile_id("robot_fault") == "fault_urgent"
    assert resolve_voice_profile_id("confirm_prompt") == "confirm_clear"


def test_voice_profiles_ignore_corrupted_config_labels():
    profiles = build_voice_profiles(
        {
            "voice_profiles": {
                "visitor_friendly": {
                    "label": "鐠佸灝顓归張宥呭",
                    "use_case": "é—‚î†¿çŸ¾éŠ†",
                    "voice_id": "custom-visitor",
                },
                "custom_brand": {
                    "label": "客户定制音色",
                    "use_case": "客户演示专用。",
                    "voice_id": "brand-voice",
                    "cue": "brand_chime",
                },
            }
        },
        default_voice_id="male-qn-qingse",
    )

    assert profiles["visitor_friendly"].label == "访客服务"
    assert profiles["visitor_friendly"].use_case.startswith("游客问路")
    assert profiles["visitor_friendly"].voice_id == "custom-visitor"
    assert profiles["custom_brand"].label == "客户定制音色"
    assert profiles["custom_brand"].cue == "brand_chime"


def test_tts_engine_can_switch_voice_profile(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)

    tts = TTSEngine(
        {
            "backend": "edge",
            "voice_profile": "patrol_default",
            "minimax_voice_id": "male-qn-qingse",
        }
    )
    try:
        result = tts.set_voice_profile_payload({"profile_id": "service_notice"})
        snapshot = tts.status_snapshot()
    finally:
        tts.shutdown()

    assert result["updated"] is True
    assert result["requested_profile"] == "service_notice"
    assert result["resolved_profile"] == "visitor_friendly"
    assert result["applied_settings"]["profile_id"] == "visitor_friendly"
    assert result["applied_settings"]["label"] == "访客服务"
    assert result["applied_settings"]["speed"] == 0.96
    assert result["applied_settings"]["cue"] == "welcome_chime"
    assert result["persistence_status"] == "session_only"
    assert snapshot["minimax"]["active_profile"] == "visitor_friendly"
    assert snapshot["minimax"]["active_profile_settings"]["profile_id"] == "visitor_friendly"
    assert snapshot["minimax"]["active_profile_settings"]["voice_id"] == "male-qn-qingse"
    assert snapshot["minimax"]["voice_id"] == "male-qn-qingse"
    assert tts.speed == 0.96
    assert tts.volume == 0.95


def test_tts_engine_queues_local_sound_cue(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    tts = TTSEngine({"backend": "edge", "minimax_voice_id": "male-qn-qingse"})
    try:
        result = tts.queue_sound_cue("welcome_chime")
        unknown = tts.queue_sound_cue("missing-cue")
        disabled = TTSEngine(
            {
                "backend": "edge",
                "minimax_voice_id": "male-qn-qingse",
                "voice_profile_cues_enabled": False,
            }
        )
        try:
            disabled_result = disabled.queue_sound_cue("welcome_chime")
        finally:
            disabled.shutdown()
    finally:
        tts.shutdown()

    assert result["queued"] is True
    assert result["cue"] == "welcome_chime"
    assert result["samples"] > 0
    assert result["duration_s"] > 0
    assert unknown == {"queued": False, "cue": "missing-cue", "reason": "unknown_cue"}
    assert disabled_result == {"queued": False, "cue": "welcome_chime", "reason": "cues_disabled"}


def test_speak_sample_queues_profile_cue(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    monkeypatch.setattr(TTSEngine, "start_playback", lambda self: None)
    spoken: list[str] = []
    tts = TTSEngine({"backend": "edge", "minimax_voice_id": "male-qn-qingse"})
    tts.speak = lambda text: spoken.append(text)  # type: ignore[method-assign]
    try:
        result = tts.set_voice_profile_payload(
            {"profile_id": "security_clear", "speak_sample": True}
        )
        payload = tts.voice_profiles_payload()
    finally:
        tts.shutdown()

    assert result["updated"] is True
    assert result["sound_cue"]["queued"] is True
    assert result["sound_cue"]["cue"] == "notice_beep"
    assert spoken == [result["profile"]["sample_text"]]
    assert "notice_beep" in payload["available_sound_cues"]
    assert payload["sound_cues_enabled"] is True


def test_tts_engine_persists_voice_profile_when_state_path_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    state_path = tmp_path / "active_voice_profile.json"

    tts = TTSEngine(
        {
            "backend": "edge",
            "minimax_voice_id": "male-qn-qingse",
            "voice_profile_state_path": str(state_path),
        }
    )
    try:
        result = tts.set_voice_profile_payload({"profile_id": "night_security"})
    finally:
        tts.shutdown()

    assert result["updated"] is True
    assert result["active_profile"] == "night_quiet"
    assert result["persistence_status"] == "persistent"
    assert state_path.exists()

    restarted = TTSEngine(
        {
            "backend": "edge",
            "voice_profile": "patrol_default",
            "minimax_voice_id": "male-qn-qingse",
            "voice_profile_state_path": str(state_path),
        }
    )
    try:
        snapshot = restarted.status_snapshot()
        payload = restarted.voice_profiles_payload()
    finally:
        restarted.shutdown()

    assert snapshot["minimax"]["active_profile"] == "night_quiet"
    assert snapshot["minimax"]["profile_persistence_status"] == "persistent"
    assert payload["persistence_status"] == "persistent"


def test_voice_profile_http_endpoint(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    tts = TTSEngine({"backend": "edge", "minimax_voice_id": "male-qn-qingse"})
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            voice_handler=tts,
        )
    )
    try:
        profiles = client.get("/api/voice/profiles")
        assert profiles.status_code == 200
        VoiceProfileCatalogResponse.model_validate(profiles.json())
        assert profiles.json()["profiles"]

        update = client.post(
            "/api/voice/profile",
            json={"profile_id": "security_clear"},
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert update.status_code == 200
        VoiceProfileUpdateResponse.model_validate(update.json())
        assert update.json()["active_profile"] == "security_clear"
        assert update.json()["applied_settings"]["label"] == "安保提醒"
        assert update.json()["applied_settings"]["cue"] == "notice_beep"
        assert update.json()["persistence_status"] == "session_only"
    finally:
        tts.shutdown()


def test_voice_profile_routes_expose_response_schemas(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    tts = TTSEngine({"backend": "edge", "minimax_voice_id": "male-qn-qingse"})
    app = create_health_app(
        lambda: _health_snapshot(),
        voice_handler=tts,
    )
    try:
        paths = app.openapi()["paths"]
        assert (
            paths["/api/voice/profiles"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"]
            .endswith("/VoiceProfileCatalogResponse")
        )
        assert (
            paths["/api/voice/profile"]["post"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"]
            .endswith("/VoiceProfileUpdateResponse")
        )
    finally:
        tts.shutdown()


def test_voice_profile_http_endpoint_rejects_unknown_profile(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    tts = TTSEngine({"backend": "edge", "minimax_voice_id": "male-qn-qingse"})
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            voice_handler=tts,
        )
    )
    try:
        missing = client.post(
            "/api/voice/profile",
            json={},
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert missing.status_code == 422
        assert missing.json()["reason"] == "missing_profile_id"

        unknown = client.post(
            "/api/voice/profile",
            json={"profile_id": "not-a-profile"},
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert unknown.status_code == 422
        assert unknown.json()["reason"] == "unknown_profile"
        assert "visitor_friendly" in unknown.json()["available"]
    finally:
        tts.shutdown()


def test_voice_profile_http_endpoint_rejects_non_object_json_body(monkeypatch):
    monkeypatch.setattr(TTSEngine, "_log_output_devices", lambda self: None)
    tts = TTSEngine({"backend": "edge", "minimax_voice_id": "male-qn-qingse"})
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            voice_handler=tts,
        )
    )
    try:
        response = client.post(
            "/api/voice/profile",
            json=["security_clear"],
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "JSON object body required"
    finally:
        tts.shutdown()
