from __future__ import annotations

from fastapi.testclient import TestClient

from askme.health_server import create_health_app


def test_health_app_supports_separate_metrics_provider() -> None:
    health_payload = {
        "status": "ok",
        "service": "askme",
        "version": "4.0.0",
        "uptime_seconds": 12.0,
        "model_name": "claude-haiku",
        "last_llm_latency_ms": 123.0,
        "total_conversations": 4,
        "active_skills": ["daily_summary"],
        "active_skill_count": 1,
        "voice_pipeline_status": {
            "mode": "text",
            "pipeline_ok": True,
            "output_ready": True,
        },
    }
    metrics_payload = {
        **health_payload,
        "last_llm_latency_ms": 98.0,
        "total_conversations": 5,
        "ota_bridge_status": {
            "enabled": True,
            "registered": True,
            "device_id": "INVX-THUNDER-001",
            "channel": "stable",
            "product": "inovxio-dog",
            "state": "connected",
        },
    }
    client = TestClient(
        create_health_app(
            health_provider=lambda: dict(health_payload),
            metrics_provider=lambda: dict(metrics_payload),
        )
    )

    health_response = client.get("/health")
    metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert health_response.json() == health_payload
    assert metrics_response.status_code == 200
    assert "askme_last_llm_latency_ms 98" in metrics_response.text
    assert "askme_conversations_total 5" in metrics_response.text
    assert "askme_ota_bridge_registered 1" in metrics_response.text
