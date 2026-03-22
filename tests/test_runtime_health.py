from __future__ import annotations

from askme.robot.runtime_health import RuntimeHealthSnapshot


def _snapshot(*, voice_pipeline_ok: bool = True, ota_state: str = "connected") -> RuntimeHealthSnapshot:
    return RuntimeHealthSnapshot(
        app_name="askme",
        app_version="4.0.0",
        brain_config={
            "model": "claude-opus-4-6",
            "voice_model": "claude-haiku-4-5-20251001",
        },
        voice_mode=True,
        robot_mode=False,
        metrics_provider=lambda: {
            "uptime_seconds": 42.5,
            "conversation_count": 7,
            "llm": {
                "call_count": 3,
                "success_count": 3,
                "failure_count": 0,
                "last_latency_ms": 245.0,
                "average_latency_ms": 212.5,
                "last_mode": "stream",
                "last_model": "claude-haiku",
            },
            "skills": {
                "run_count": 2,
                "success_count": 2,
                "failure_count": 0,
                "success_rate": 1.0,
            },
            "voice_pipeline": {
                "mode": "voice",
                "enabled": True,
                "input_ready": True,
                "output_ready": True,
                "pipeline_ok": voice_pipeline_ok,
                "asr_available": True,
                "vad_available": True,
                "kws_available": True,
                "wake_word_enabled": True,
                "tts_backend": "edge",
                "tts_busy": False,
            },
        },
        active_skill_names_provider=lambda: ["daily_summary", "patrol_report"],
        voice_status_provider=lambda: {
            "mode": "voice",
            "enabled": True,
            "input_ready": True,
            "output_ready": True,
            "pipeline_ok": voice_pipeline_ok,
            "tts_backend": "edge",
            "tts_busy": False,
        },
        ota_status_provider=lambda: {
            "enabled": True,
            "registered": True,
            "state": ota_state,
            "channel": "stable",
            "device_id": "INVX-THUNDER-001",
        },
    )


def test_health_snapshot_exposes_required_runtime_fields() -> None:
    snapshot = _snapshot().health_snapshot()

    assert snapshot["status"] == "ok"
    assert snapshot["service_name"] == "askme"
    assert snapshot["service_version"] == "4.0.0"
    assert snapshot["uptime_seconds"] == 42.5
    assert snapshot["model_name"] == "claude-haiku"
    assert snapshot["last_llm_latency_ms"] == 245.0
    assert snapshot["total_conversations"] == 7
    assert snapshot["active_skills"] == ["daily_summary", "patrol_report"]
    assert snapshot["voice_pipeline_status"]["pipeline_ok"] is True
    assert snapshot["ota_bridge_status"]["device_id"] == "INVX-THUNDER-001"
    assert snapshot["degraded_reasons"] == []


def test_metrics_snapshot_reports_degraded_reasons() -> None:
    snapshot = _snapshot(voice_pipeline_ok=False, ota_state="degraded").metrics_snapshot()

    assert snapshot["status"] == "degraded"
    assert snapshot["degraded_reasons"] == [
        "voice_pipeline_unavailable",
        "ota_bridge_degraded",
    ]
    assert snapshot["model_name"] == "claude-haiku"
    assert snapshot["configured_model_name"] == "claude-opus-4-6"
    assert snapshot["active_skill_count"] == 2
