"""Tests for the MCP and HTTP health surfaces."""

import json

import pytest
from fastapi.testclient import TestClient

from askme.health_server import AskmeHealthServer, build_health_snapshot, create_health_app


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


def _degraded_runtime_snapshot() -> dict:
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

    def test_field_notification_preflight_endpoint_reports_blocked(self):
        class Handler:
            def notification_preflight_payload(self):
                return {
                    "status": "blocked",
                    "ready": False,
                    "groups": {"security": {"ready": False}},
                    "blockers": ["security notification is not fully configured"],
                    "next_actions": ["Configure security"],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                field_operations_handler=Handler(),
            )
        )

        response = client.get("/api/field/notification-preflight")

        assert response.status_code == 409
        data = response.json()
        assert data["status"] == "blocked"
        assert data["ready"] is False

    def test_field_devices_endpoint_returns_status_payload(self):
        class Handler:
            def device_status_payload(self):
                return {
                    "status": "ok",
                    "summary": {"registered": 1, "online": 1},
                    "devices": [{"device_id": "smoke-01", "status": "online"}],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                field_operations_handler=Handler(),
            )
        )

        response = client.get("/api/field/devices")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["registered"] == 1
        assert data["devices"][0]["device_id"] == "smoke-01"

    def test_http_healthz_endpoint_matches_health_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "askme"
        assert data["voice_pipeline_status"]["pipeline_ok"] is True

    def test_http_health_endpoint_returns_degraded_snapshot_without_5xx(self):
        client = TestClient(create_health_app(lambda: _degraded_runtime_snapshot()))

        response = client.get("/health")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        assert data["status"] == "degraded"
        assert data["degraded_reasons"] == ["voice_pipeline", "ota_bridge"]

    def test_http_healthz_endpoint_returns_degraded_snapshot_without_5xx(self):
        client = TestClient(create_health_app(lambda: _degraded_runtime_snapshot()))

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        assert data["status"] == "degraded"
        assert data["voice_pipeline_status"]["pipeline_ok"] is False

    def test_http_health_endpoint_reports_provider_exception(self):
        def broken_provider():
            raise RuntimeError("provider failed")

        client = TestClient(create_health_app(broken_provider))

        response = client.get("/health")

        assert response.status_code == 500
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == {
            "status": "error",
            "error": "provider failed",
        }

    def test_metrics_endpoint_returns_prometheus_text(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "askme_uptime_seconds 12.5" in response.text
        assert "askme_conversations_total 7" in response.text
        assert "askme_last_llm_latency_ms 245" in response.text
        assert 'askme_model_info{model_name="claude-opus-4-6"} 1' in response.text
        assert 'askme_active_skill_info{skill="dock_charge"} 1' in response.text
        assert "askme_voice_pipeline_ok 1" in response.text
        assert "askme_ota_bridge_registered 1" in response.text

    def test_metrics_prometheus_endpoint_matches_metrics_contract(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/metrics/prometheus")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "text/plain" in response.headers["content-type"]
        assert "askme_up 1" in response.text
        assert "askme_health_status 1" in response.text

    def test_metrics_prometheus_endpoint_marks_degraded_snapshot_unhealthy(self):
        client = TestClient(create_health_app(lambda: _degraded_runtime_snapshot()))

        response = client.get("/metrics/prometheus")

        assert response.status_code == 200
        assert "askme_up 1" in response.text
        assert "askme_health_status 0" in response.text
        assert "askme_voice_pipeline_ok 0" in response.text
        assert "askme_ota_bridge_registered 0" in response.text

    def test_metrics_prometheus_endpoint_marks_provider_exception_unhealthy(self):
        def broken_provider():
            raise RuntimeError("provider failed")

        client = TestClient(create_health_app(broken_provider))

        response = client.get("/metrics/prometheus")

        assert response.status_code == 500
        assert response.headers["Cache-Control"] == "no-store"
        assert "askme_up 1" in response.text
        assert "askme_health_status 0" in response.text

    def test_chat_endpoint_forwards_speak_request_to_handler(self):
        seen: dict[str, object] = {}

        async def chat_handler(text: str, *, speak: bool = False):
            seen["text"] = text
            seen["speak"] = speak
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"text": "hello", "speak": True},
        )

        assert response.status_code == 200
        assert seen == {"text": "hello", "speak": True}
        assert response.json() == {
            "reply": "reply:hello",
            "spoken": True,
            "text": "hello",
            "evidence": [],
        }

    def test_chat_endpoint_accepts_message_alias(self):
        seen: dict[str, object] = {}

        async def chat_handler(text: str, *, speak: bool = False):
            seen["text"] = text
            seen["speak"] = speak
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"message": "hello", "speak": True},
        )

        assert response.status_code == 200
        assert seen == {"text": "hello", "speak": True}
        assert response.json() == {
            "reply": "reply:hello",
            "spoken": True,
            "text": "hello",
            "evidence": [],
        }

    def test_chat_endpoint_returns_voice_transcript_metadata_for_voice_turn(self):
        async def chat_handler(text: str, *, speak: bool = False):
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={
                "text": "纭",
                "voice": True,
                "transcript_id": "voice-confirm-1",
                "asr_confidence": 0.87,
            },
        )

        voice_turn = response.json()["voice_turn"]
        assert response.status_code == 200
        assert voice_turn["transcript_id"] == "voice-confirm-1"
        assert voice_turn["recognized_text"] == "纭"
        assert voice_turn["confidence"] == 0.87
        assert voice_turn["safety_bypass_allowed"] is False

    def test_chat_endpoint_keeps_text_only_handler_compatible(self):
        async def chat_handler(text: str):
            return f"reply:{text}"

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"text": "hello", "speak": True},
        )

        assert response.status_code == 200
        assert response.json() == {
            "reply": "reply:hello",
            "text": "hello",
            "spoken": False,
            "evidence": [],
        }

    def test_chat_endpoint_preserves_handler_evidence_payload(self):
        async def chat_handler(text: str, *, speak: bool = False):
            return {
                "reply": f"reply:{text}",
                "evidence": [{"text": "site fact", "source": "site.md"}],
                "rag": {"backend": "vector", "used_in_answer": True},
            }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post("/api/chat", json={"text": "hello"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["evidence"][0]["source"] == "site.md"
        assert payload["rag"]["backend"] == "vector"

    def test_chat_endpoint_attaches_memory_evidence_for_plain_text_handler(self):
        class MemoryHandler:
            def health(self):
                return {
                    "enabled": True,
                    "backend": "vector",
                    "available": True,
                    "last_backend": "vector",
                    "last_retrieve_ms": 12,
                    "last_retrieved_items": 1,
                    "last_evidence": [{
                        "text": "A 鍖哄叆鍙ｅ湪涓滈棬",
                        "source": "site.md",
                        "record_id": "rec-a",
                    }],
                    "last_dropped_evidence": [{
                        "text": "expired memory fact",
                        "drop_reason": "expired",
                        "record_id": "rec-old",
                    }],
                    "last_answer_policy": {
                        "state": "grounded",
                        "action": "answer_with_evidence",
                    },
                }

        async def chat_handler(text: str, *, speak: bool = False):
            return f"reply:{text}"

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
                memory_handler=MemoryHandler(),
            )
        )

        response = client.post("/api/chat", json={"text": "where is gate A?"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["evidence"][0]["record_id"] == "rec-a"
        assert payload["rag"]["answer_policy"]["state"] == "grounded"
        assert payload["rag"]["dropped_evidence"][0]["drop_reason"] == "expired"

    def test_chat_endpoint_does_not_overwrite_handler_evidence_with_memory_context(self):
        class MemoryHandler:
            def health(self):
                return {
                    "last_evidence": [{"text": "memory fact", "source": "memory.md"}],
                    "last_answer_policy": {"state": "grounded"},
                }

        async def chat_handler(text: str, *, speak: bool = False):
            return {
                "reply": f"reply:{text}",
                "evidence": [{"text": "handler fact", "source": "handler.md"}],
                "rag": {"backend": "handler"},
            }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
                memory_handler=MemoryHandler(),
            )
        )

        response = client.post("/api/chat", json={"text": "hello"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["evidence"][0]["source"] == "handler.md"
        assert payload["rag"]["backend"] == "handler"

    def test_chat_endpoint_forces_refusal_when_rag_policy_blocks_plain_text_reply(self):
        class MemoryHandler:
            def health(self):
                return {
                    "enabled": True,
                    "backend": "vector",
                    "available": True,
                    "last_backend": "vector",
                    "last_evidence": [],
                    "last_dropped_evidence": [{
                        "text": "old route",
                        "drop_reason": "expired",
                        "record_id": "route-old",
                    }],
                    "last_answer_policy": {
                        "state": "stale",
                        "action": "refuse_and_request_update",
                        "reason": "expired",
                        "required_operator_action": "refresh_knowledge",
                    },
                }

        async def chat_handler(text: str, *, speak: bool = False):
            return "go straight to the old gate"

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
                memory_handler=MemoryHandler(),
            )
        )

        response = client.post("/api/chat", json={"text": "how do I reach the gate?"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["reply"] != "go straight to the old gate"
        assert payload["rag_blocked"] is True
        assert payload["rag"]["answer_blocked"] is True
        assert payload["rag"]["forced_reply"] is True
        assert payload["rag"]["block_reason"] == "expired"
        assert payload["rag"]["dropped_evidence"][0]["drop_reason"] == "expired"

    def test_memory_search_endpoint_dispatches_handler(self):
        class Handler:
            async def search_payload(self, payload):
                return {
                    "query": payload["query"],
                    "results": [{"text": "site fact", "source": "site.md"}],
                    "rag": {"backend": "vector"},
                    "warnings": [],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post("/api/memory/search", json={"query": "site"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["query"] == "site"
        assert payload["results"][0]["source"] == "site.md"
        assert payload["rag"]["backend"] == "vector"

    def test_knowledge_preview_endpoint_dispatches_handler(self):
        class Handler:
            async def preview_payload(self, payload):
                return {
                    "source": payload["filename"],
                    "parsed": 1,
                    "records": [{"text": "fact", "category": "faq"}],
                    "errors": [],
                    "dry_run": True,
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/preview",
            json={"filename": "faq.md", "content": "- fact"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["source"] == "faq.md"
        assert payload["parsed"] == 1
        assert payload["records"][0]["text"] == "fact"

    def test_knowledge_import_endpoint_dispatches_handler(self):
        class Handler:
            async def import_payload(self, payload):
                return {
                    "source": payload["filename"],
                    "parsed": 1,
                    "imported": 1,
                    "skipped": 0,
                    "errors": [],
                    "rag": {"backend": "vector"},
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/import",
            json={"filename": "faq.md", "content": "- fact"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["imported"] == 1
        assert payload["rag"]["backend"] == "vector"

    def test_knowledge_list_endpoint_dispatches_handler(self):
        class Handler:
            async def list_knowledge_payload(self, payload):
                return {
                    "backend": "vector",
                    "total": 1,
                    "records": [{"record_id": "know_1", "text": "fact"}],
                    "rag": {"backend": "vector"},
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post("/api/knowledge/list", json={"limit": 50})

        assert response.status_code == 200
        payload = response.json()
        assert payload["records"][0]["record_id"] == "know_1"
        assert payload["total"] == 1

    def test_knowledge_update_endpoint_dispatches_handler(self):
        class Handler:
            async def update_knowledge_payload(self, payload):
                return {
                    "updated": True,
                    "record_id": payload["record_id"],
                    "patch": {"approval_status": "deleted"},
                    "rag": {"backend": "vector"},
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/update",
            json={"record_id": "know_1", "action": "delete"},
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["updated"] is True
        assert payload["record_id"] == "know_1"

    def test_knowledge_update_blocks_operator_without_approval_role(self):
        class Handler:
            async def update_knowledge_payload(self, payload):
                return {"updated": True}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/update",
            json={"record_id": "know_1", "action": "delete"},
            headers={"X-Askme-Operator-Id": "dashboard.operator"},
        )

        assert response.status_code == 403
        payload = response.json()
        assert payload["reason"] == "operator_missing_permission"
        assert payload["operator_auth"]["permission"] == "knowledge:delete"

    def test_governance_operator_directory_exposes_demo_boundary(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/governance/operator-directory")

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "demo_config"
        assert payload["identity_provider"] == "local_config"
        assert payload["production_binding_required"] is True
        assert payload["session_operator_header"] == "x-askme-operator-id"
        assert payload["permissions"]["operator"]
        assert payload["sso"]["configured"] is False
        assert any(
            operator["operator_id"] == "dashboard.operator"
            for operator in payload["operators"]
        )

    def test_dashboard_contains_cognition_planning_controls(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/dashboard")

        assert response.status_code == 200
        assert 'id="dashboard-nav"' in response.text
        assert 'id="app-page"' in response.text
        for page in (
            "/dashboard/conversation",
            "/dashboard/field",
            "/dashboard/knowledge",
            "/dashboard/voice",
            "/dashboard/delivery",
        ):
            page_response = client.get(page)
            assert page_response.status_code == 200
            assert 'id="app-page"' in page_response.text

        js_response = client.get("/dashboard/app.js")
        css_response = client.get("/dashboard/app.css")

        assert js_response.status_code == 200
        assert css_response.status_code == 200
        assert "/dashboard/conversation" in js_response.text
        assert "/dashboard/field" in js_response.text
        assert "/dashboard/knowledge" in js_response.text
        assert "/dashboard/voice" in js_response.text
        assert "/dashboard/delivery" in js_response.text
        assert "renderOverview" in js_response.text
        assert "renderConversation" in js_response.text
        assert "renderField" in js_response.text
        assert "renderKnowledge" in js_response.text
        assert "renderVoice" in js_response.text
        assert "renderDelivery" in js_response.text
        assert "/api/chat" in js_response.text
        assert "/api/knowledge/preview" in js_response.text
        assert "/api/knowledge/import" in js_response.text
        assert "/api/knowledge/list" in js_response.text
        assert "/api/memory/search" in js_response.text
        assert "知识管理" in js_response.text
        assert "导入并发布" in js_response.text
        assert "重建索引" in js_response.text
        assert "现场事件闭环看板" in js_response.text
        assert "/api/governance/operator-directory" in js_response.text
        assert "knowledge-operations" in js_response.text
        assert "/api/field/scenarios" in js_response.text
        assert "/api/field/events" in js_response.text
        assert "/api/field/readiness" in js_response.text
        assert "/api/field/devices" in js_response.text
        assert "暂无现场事件" in js_response.text
        assert "保安群" in js_response.text
        assert "/api/voice/profiles" in js_response.text
        assert "/api/voice/profile" in js_response.text
        assert "asr_final_ms" in js_response.text
        assert "llm_ttft_ms" in js_response.text
        assert "tts_first_audio_ms" in js_response.text
        assert "playback_start_ms" in js_response.text
        assert "speak: true" in js_response.text
        assert "play_audio: true" in js_response.text

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

    def test_control_api_key_protects_non_probe_routes(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {"profile": {"name": "voice"}},
                control_api_key="secret",
            )
        )

        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

        unauth = client.get("/api/capabilities")
        assert unauth.status_code == 401
        assert unauth.json()["error"] == "control API authentication required"

        wrong = client.get(
            "/api/capabilities",
            headers={"Authorization": "Bearer wrong"},
        )
        assert wrong.status_code == 401

        bearer = client.get(
            "/api/capabilities",
            headers={"Authorization": "Bearer secret"},
        )
        assert bearer.status_code == 200
        assert bearer.json()["profile"]["name"] == "voice"

        api_key = client.get(
            "/api/capabilities",
            headers={"X-Askme-Api-Key": "secret"},
        )
        assert api_key.status_code == 200

    def test_health_server_defaults_to_loopback_and_requires_remote_auth(self):
        server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
        assert server.host == "127.0.0.1"

        with pytest.raises(ValueError, match="binds outside loopback"):
            AskmeHealthServer(
                {"host": "0.0.0.0"},
                health_provider=lambda: _runtime_snapshot(),
            )

        remote = AskmeHealthServer(
            {"host": "0.0.0.0", "control_api_key": "secret"},
            health_provider=lambda: _runtime_snapshot(),
        )
        assert remote.host == "0.0.0.0"

    def test_cognition_endpoints_delegate_to_handler(self):
        class DummyCognitionHandler:
            def __init__(self):
                self.refresh_seen = None

            async def context_payload(self, *, refresh_perception: bool = False):
                self.refresh_seen = refresh_perception
                return {"world_state": {"fact_count": 1}, "working_memory": {"item_count": 0}}

            async def plan_from_payload(self, payload):
                return {"planned": True, "plan": {"goal": payload["text"]}}

        handler = DummyCognitionHandler()
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                cognition_handler=handler,
            )
        )

        context = client.get("/api/cognition/context?refresh_perception=true")
        assert context.status_code == 200
        assert context.json()["world_state"]["fact_count"] == 1
        assert handler.refresh_seen is True

        plan = client.post("/api/cognition/plan", json={"text": "inspect area-a"})
        assert plan.status_code == 200
        assert plan.json()["planned"] is True
        assert plan.json()["plan"]["goal"] == "inspect area-a"

    def test_runtime_endpoints_delegate_to_handler(self):
        class DummyRuntimeHandler:
            def context_payload(self):
                return {"profile": "sim", "active_run": {"run_id": "run-1", "current_state": "queued"}}

            def profiles_payload(self):
                return {"current_profile": "sim", "profiles": [{"name": "fake"}, {"name": "shadow"}, {"name": "sim"}]}

            def list_payload(self):
                return {"runs": [{"run_id": "run-1"}], "count": 1}

            def events_payload(self, *, after=None, limit=20):
                return {
                    "profile": "sim",
                    "hardware_dispatch": False,
                    "cursor": 123.0,
                    "events": [
                        {
                            "event_id": "evt-1",
                            "run_id": "run-1",
                            "event_type": "task_queued",
                            "state": "queued",
                            "message": "queued",
                            "created_at": 123.0,
                        }
                    ],
                    "event_count": 1,
                    "active_run": {"run_id": "run-1", "current_state": "queued"},
                }

            def get_payload(self, run_id):
                return {"run": {"run_id": run_id, "current_state": "queued"}}

            def report_payload(self, run_id):
                return {"report": {"run_id": run_id, "status": "queued"}}

            def pause_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "paused"}}

            def resume_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "executing"}}

            def cancel_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "cancelled"}}

            def advance_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "executing"}}

            def voice_turn_payload(self, text, **kwargs):
                return {
                    "handled": True,
                    "reply": "TaskRun paused.",
                    "runtime": {"run": {"run_id": "run-1", "current_state": "paused"}},
                    "voice_turn": {
                        "recognized_text": text,
                        "runtime_control_intent": "pause",
                        "safety_bypass_allowed": False,
                        "transcript_id": kwargs.get("transcript_id", ""),
                    },
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                runtime_handler=DummyRuntimeHandler(),
            )
        )

        context = client.get("/api/runtime/context")
        assert context.status_code == 200
        assert context.json()["active_run"]["run_id"] == "run-1"
        assert context.json()["profile"] == "sim"

        profiles = client.get("/api/runtime/profiles")
        assert profiles.status_code == 200
        assert profiles.json()["current_profile"] == "sim"

        runs = client.get("/api/runtime/runs")
        assert runs.status_code == 200
        assert runs.json()["count"] == 1

        events = client.get("/api/runtime/events?once=1")
        assert events.status_code == 200
        assert "text/event-stream" in events.headers["content-type"]
        assert "event: runtime.events" in events.text
        assert '"event_type":"task_queued"' in events.text

        run = client.get("/api/runtime/runs/run-1")
        assert run.status_code == 200
        assert run.json()["run"]["current_state"] == "queued"

        report = client.get("/api/runtime/runs/run-1/report")
        assert report.status_code == 200
        assert report.json()["report"]["status"] == "queued"

        paused = client.post("/api/runtime/runs/run-1/pause")
        assert paused.status_code == 200
        assert paused.json()["run"]["current_state"] == "paused"

        resumed = client.post("/api/runtime/runs/run-1/resume")
        assert resumed.status_code == 200
        assert resumed.json()["run"]["current_state"] == "executing"

        cancelled_forbidden = client.post("/api/runtime/runs/run-1/cancel")
        assert cancelled_forbidden.status_code == 403

        cancelled = client.post(
            "/api/runtime/runs/run-1/cancel",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["run"]["current_state"] == "cancelled"

        advanced = client.post(
            "/api/runtime/runs/run-1/advance",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert advanced.status_code == 200
        assert advanced.json()["run"]["current_state"] == "executing"

        voice = client.post(
            "/api/runtime/voice-turn",
            json={"text": "pause current task", "transcript_id": "voice-1", "confidence": 0.9},
        )
        assert voice.status_code == 200
        assert voice.json()["runtime"]["run"]["current_state"] == "paused"
        assert voice.json()["voice_turn"]["recognized_text"] == "pause current task"
        assert voice.json()["voice_turn"]["safety_bypass_allowed"] is False

    def test_runtime_control_endpoint_forwards_operator_context(self):
        class DummyRuntimeHandler:
            def __init__(self):
                self.seen = {}

            def pause_payload(
                self,
                run_id,
                *,
                operator_id="askme.operator",
                reason="",
                risk_acknowledgement=False,
            ):
                self.seen = {
                    "run_id": run_id,
                    "operator_id": operator_id,
                    "reason": reason,
                    "risk_acknowledgement": risk_acknowledgement,
                }
                return {
                    "handled": True,
                    "run": {"run_id": run_id, "current_state": "paused"},
                    "operator": self.seen,
                }

        runtime = DummyRuntimeHandler()
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                runtime_handler=runtime,
            )
        )

        response = client.post(
            "/api/runtime/runs/run-7/pause",
            json={
                "operator_id": "operator.alice",
                "reason": "visitor entered path",
                "risk_acknowledgement": True,
            },
        )

        assert response.status_code == 200
        assert runtime.seen == {
            "run_id": "run-7",
            "operator_id": "operator.alice",
            "reason": "visitor entered path",
            "risk_acknowledgement": True,
        }

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
