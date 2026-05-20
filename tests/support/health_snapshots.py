"""Reusable health snapshot fixtures for HTTP surface tests."""

from __future__ import annotations

from askme.health_server import build_health_snapshot


def minimal_runtime_snapshot() -> dict:
    return {
        "app": {"name": "askme", "version": "test"},
        "status": "ok",
        "uptime_seconds": 1.0,
    }


def runtime_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="4.0.0",
        model_name="claude-opus-4-6",
        metrics_snapshot={
            "uptime_seconds": 12.5,
            "conversation_count": 7,
            "llm": {
                "last_latency_ms": 245.0,
                "last_model": "claude-opus-4-6",
            },
            "voice_pipeline": {
                "last_input_at": "2026-03-09T04:00:00Z",
                "last_input_chars": 12,
            },
        },
        active_skills=["dock_charge", "inspect_zone"],
        voice_status={
            "mode": "voice",
            "enabled": True,
            "pipeline_ok": True,
            "input_ready": True,
            "output_ready": True,
            "asr_available": True,
            "vad_available": True,
            "kws_available": True,
            "wake_word_enabled": True,
            "woken_up": True,
            "tts_backend": "edge",
            "tts_busy": False,
        },
        ota_status={
            "enabled": True,
            "registered": True,
            "device_id": "INVX-THUNDER-001",
            "channel": "stable",
            "product": "inovxio-dog",
            "state": "connected",
        },
    )


def degraded_runtime_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="4.0.0",
        model_name="claude-opus-4-6",
        metrics_snapshot={"uptime_seconds": 12.5, "conversation_count": 7},
        active_skills=[],
        voice_status={
            "mode": "voice",
            "enabled": True,
            "pipeline_ok": False,
            "input_ready": False,
            "output_ready": True,
        },
        ota_status={
            "enabled": True,
            "registered": False,
            "device_id": "INVX-THUNDER-001",
            "channel": "stable",
            "product": "inovxio-dog",
            "state": "degraded",
        },
    )
