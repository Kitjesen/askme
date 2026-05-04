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
        from askme.mcp.resources.health_resources import health_check

        result = health_check()
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_health_has_version(self):
        from askme.mcp.resources.health_resources import health_check

        data = json.loads(health_check())
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_subsystems(self):
        from askme.mcp.resources.health_resources import health_check

        data = json.loads(health_check())
        assert "subsystems" in data
        assert "brain" in data["subsystems"]
        assert "robot" in data["subsystems"]
        assert "voice" in data["subsystems"]

    def test_health_has_uptime(self):
        from askme.mcp.resources.health_resources import health_check

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

    def test_capabilities_endpoint_returns_runtime_contracts(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {
                    "profile": {"name": "voice", "primary_loop": "voice"},
                    "components": {
                        "skills": {
                            "health": {"status": "ok"},
                            "capabilities": {"openapi_generated": True},
                        }
                    },
                    "skills": {"contract_count": 3, "code_contract_count": 2},
                },
            )
        )

        response = client.get("/api/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert data["profile"]["name"] == "voice"
        assert data["components"]["skills"]["capabilities"]["openapi_generated"] is True
        assert data["skills"]["contract_count"] == 3

    def test_mission_endpoints_delegate_to_handler(self):
        class DummyMissionHandler:
            def __init__(self):
                self.mission = {
                    "mission_id": "mission-1",
                    "goal": "inspect area-a",
                    "status": "draft",
                }

            def draft_from_payload(self, payload):
                self.mission["goal"] = payload["text"]
                return {"mission": self.mission, "drafted": True}

            def submit_from_payload(self, payload):
                self.mission["status"] = "dry_run" if payload.get("dry_run", True) else "submitted"
                return {
                    "mission": self.mission,
                    "submission": {"submitted": False, "dry_run": True},
                }

            def list_payload(self):
                return {"missions": [self.mission], "count": 1}

            def get_payload(self, mission_id):
                if mission_id != self.mission["mission_id"]:
                    return {"error": "mission not found", "mission_id": mission_id}
                return {"mission": self.mission}

            def report_payload(self, mission_id):
                if mission_id != self.mission["mission_id"]:
                    return {"error": "mission not found", "mission_id": mission_id}
                return {"report": {"mission_id": mission_id, "status": self.mission["status"]}}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                mission_handler=DummyMissionHandler(),
            )
        )

        draft = client.post("/api/missions/draft", json={"text": "inspect area-a"})
        assert draft.status_code == 200
        assert draft.json()["drafted"] is True

        submit = client.post("/api/missions", json={"text": "inspect area-a", "dry_run": True})
        assert submit.status_code == 200
        assert submit.json()["submission"]["dry_run"] is True

        mission_list = client.get("/api/missions")
        assert mission_list.status_code == 200
        assert mission_list.json()["count"] == 1

        mission_get = client.get("/api/missions/mission-1")
        assert mission_get.status_code == 200
        assert mission_get.json()["mission"]["mission_id"] == "mission-1"

        report = client.get("/api/missions/mission-1/report")
        assert report.status_code == 200
        assert report.json()["report"]["status"] == "dry_run"

        missing = client.get("/api/missions/missing")
        assert missing.status_code == 404

    def test_mission_endpoint_returns_unconfigured_status(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post("/api/missions/draft", json={"text": "inspect area-a"})

        assert response.status_code == 503
        assert response.json()["error"] == "mission handler not configured"


class TestVoiceBridgeSnapshot:
    """Cover the voice_bridge=None / voice_bridge=<dict> branches of build_health_snapshot."""

    _BASE_KWARGS = dict(
        app_name="askme",
        app_version="4.0.0",
        model_name="claude-opus-4-6",
        metrics_snapshot={"uptime_seconds": 1.0},
        active_skills=[],
        voice_status={"pipeline_ok": True},
    )

    def test_voice_bridge_none_key_absent(self):
        snapshot = build_health_snapshot(**self._BASE_KWARGS)
        assert "voice_bridge" not in snapshot

    def test_voice_bridge_present_key_included(self):
        bridge_payload = {"status": "connected"}
        snapshot = build_health_snapshot(**self._BASE_KWARGS, voice_bridge=bridge_payload)
        assert "voice_bridge" in snapshot
        assert snapshot["voice_bridge"] == bridge_payload
