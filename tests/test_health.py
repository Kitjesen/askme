"""Tests for the MCP and HTTP health surfaces."""

import json

from fastapi.testclient import TestClient

from askme.health_server import build_health_snapshot, create_health_app


def _runtime_snapshot() -> dict:
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


class TestHealthResource:
    def test_health_returns_valid_json(self):
        from askme.mcp_resources.health_resources import health_check

        result = health_check()
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_health_has_version(self):
        from askme.mcp_resources.health_resources import health_check

        data = json.loads(health_check())
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_subsystems(self):
        from askme.mcp_resources.health_resources import health_check

        data = json.loads(health_check())
        assert "subsystems" in data
        assert "brain" in data["subsystems"]
        assert "robot" in data["subsystems"]
        assert "voice" in data["subsystems"]

    def test_health_has_uptime(self):
        from askme.mcp_resources.health_resources import health_check

        data = json.loads(health_check())
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestHealthServer:
    def test_http_health_endpoint_returns_runtime_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["uptime_seconds"] == 12.5
        assert data["model_name"] == "claude-opus-4-6"
        assert data["last_llm_latency_ms"] == 245.0
        assert data["total_conversations"] == 7
        assert data["active_skills"] == ["dock_charge", "inspect_zone"]
        assert data["voice_pipeline_status"]["pipeline_ok"] is True
        assert data["ota_bridge_status"]["registered"] is True

    def test_metrics_endpoint_returns_prometheus_text(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "askme_uptime_seconds 12.5" in response.text
        assert "askme_conversations_total 7" in response.text
        assert "askme_last_llm_latency_ms 245" in response.text
        assert 'askme_model_info{model_name="claude-opus-4-6"} 1' in response.text
        assert 'askme_active_skill_info{skill="dock_charge"} 1' in response.text
        assert "askme_voice_pipeline_ok 1" in response.text
        assert "askme_ota_bridge_registered 1" in response.text
