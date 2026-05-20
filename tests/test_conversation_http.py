"""HTTP tests for conversation, chat, and cognition routes."""

import json

from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.api.schemas.cognition import CognitionContextResponse, CognitionPlanResponse
from askme.api.schemas.conversation import (
    ChatResponse,
    ConversationDiagnosticsResponse,
    ConversationHistoryResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


def test_chat_endpoint_forwards_speak_request_to_handler():
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


def test_chat_endpoint_reports_timeout_from_config(monkeypatch):
    monkeypatch.setattr(
        health_server,
        "get_config",
        lambda: {"conversation": {"chat_timeout_s": 0.001}},
    )

    async def chat_handler(text: str, *, speak: bool = False):
        import asyncio

        await asyncio.sleep(0.05)
        return {"reply": text}

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            chat_handler=chat_handler,
        )
    )

    response = client.post("/api/chat", json={"text": "slow"})

    assert response.status_code == 504
    assert response.json()["error"] == "chat timed out"


def test_conversation_diagnostics_endpoint_reports_chat_state():
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
        json={"text": "hello"},
        headers={"X-Request-Id": "trace-route-1"},
    )
    diagnostics = client.get("/api/conversation/diagnostics")

    assert response.status_code == 200
    assert response.headers["X-Askme-Trace-Id"] == "trace-route-1"
    assert diagnostics.status_code == 200
    payload = diagnostics.json()["chat"]
    assert payload["configured"] is True
    assert payload["total_turns"] == 1
    assert payload["in_flight"] == 0
    assert payload["last_turn"]["status"] == "ok"
    assert payload["last_turn"]["trace_id"] == "trace-route-1"


def test_chat_endpoint_accepts_message_alias():
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


def test_chat_endpoint_returns_voice_transcript_metadata_for_voice_turn():
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
            "text": "confirm voice",
            "voice": True,
            "transcript_id": "voice-confirm-1",
            "asr_confidence": 0.87,
        },
    )

    voice_turn = response.json()["voice_turn"]
    assert response.status_code == 200
    assert voice_turn["transcript_id"] == "voice-confirm-1"
    assert voice_turn["recognized_text"] == "confirm voice"
    assert voice_turn["confidence"] == 0.87
    assert voice_turn["safety_bypass_allowed"] is False


def test_chat_endpoint_keeps_text_only_handler_compatible():
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


def test_chat_endpoint_rejects_non_object_json_body_before_dispatch():
    async def chat_handler(text: str):
        raise AssertionError("chat handler should not be called")

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            chat_handler=chat_handler,
        )
    )

    response = client.post("/api/chat", json=["hello"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"
    assert response.headers["X-Askme-Trace-Id"]


def test_chat_endpoint_returns_customer_safe_fallback_without_chat_handler():
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post("/api/chat", json={"text": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["degraded"] is True
    assert payload["reply_source"] == "dashboard_offline_fallback"
    assert payload["chat_backend"]["configured"] is False
    assert payload["answer_policy"]["reason"] == "chat_handler_not_configured"


def test_chat_endpoint_preserves_handler_evidence_payload():
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


def test_chat_endpoint_attaches_memory_evidence_for_plain_text_handler():
    class MemoryHandler:
        def health(self):
            return {
                "enabled": True,
                "backend": "vector",
                "available": True,
                "last_backend": "vector",
                "last_retrieve_ms": 12,
                "last_retrieved_items": 1,
                "last_evidence": [
                    {
                        "text": "Gate A entrance is by the east gate",
                        "source": "site.md",
                        "record_id": "rec-a",
                    }
                ],
                "last_dropped_evidence": [
                    {
                        "text": "expired memory fact",
                        "drop_reason": "expired",
                        "record_id": "rec-old",
                    }
                ],
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


def test_chat_endpoint_does_not_overwrite_handler_evidence_with_memory_context():
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


def test_chat_endpoint_forces_refusal_when_rag_policy_blocks_plain_text_reply():
    class MemoryHandler:
        def health(self):
            return {
                "enabled": True,
                "backend": "vector",
                "available": True,
                "last_backend": "vector",
                "last_evidence": [],
                "last_dropped_evidence": [
                    {
                        "text": "old route",
                        "drop_reason": "expired",
                        "record_id": "route-old",
                    }
                ],
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


def test_live_endpoint_uses_conversation_provider():
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            conversation_provider=lambda: [{"role": "user", "content": "hello"}],
        )
    )

    response = client.get("/api/live")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [{"role": "user", "content": "hello"}],
        "count": 1,
    }


def test_conversations_endpoint_reads_configured_history_file(tmp_path, monkeypatch):
    history = [{"role": "assistant", "content": "ready"}]
    history_path = tmp_path / "conversation-history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    monkeypatch.setattr(
        health_server,
        "get_config",
        lambda: {"conversation": {"history_file": str(history_path)}},
    )

    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.get("/api/conversations")

    assert response.status_code == 200
    assert response.json() == {"messages": history, "count": 1}


def test_cognition_endpoints_delegate_to_handler():
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


def test_cognition_plan_rejects_non_object_json_body():
    class DummyCognitionHandler:
        async def plan_from_payload(self, payload):
            raise AssertionError("cognition planner should not be called")

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            cognition_handler=DummyCognitionHandler(),
        )
    )

    response = client.post("/api/cognition/plan", json=["inspect area-a"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_chat_cognition_and_conversation_routes_expose_response_schemas():
    async def chat_handler(text: str, *, speak: bool = False):
        return {"reply": f"reply:{text}", "spoken": speak}

    class DummyCognitionHandler:
        async def context_payload(self, *, refresh_perception: bool = False):
            return {
                "world_state": {"fact_count": 1},
                "working_memory": {"item_count": 0},
                "perception": {"refreshed": refresh_perception},
            }

        async def plan_from_payload(self, payload):
            return {"planned": True, "plan": {"goal": payload["text"]}}

    app = create_health_app(
        lambda: _runtime_snapshot(),
        chat_handler=chat_handler,
        cognition_handler=DummyCognitionHandler(),
        conversation_provider=lambda: [{"role": "user", "content": "hello"}],
    )
    paths = app.openapi()["paths"]
    expected_refs = {
        ("/api/chat", "post"): "ChatResponse",
        ("/api/conversation/diagnostics", "get"): "ConversationDiagnosticsResponse",
        ("/api/live", "get"): "ConversationHistoryResponse",
        ("/api/conversations", "get"): "ConversationHistoryResponse",
        ("/api/cognition/context", "get"): "CognitionContextResponse",
        ("/api/cognition/plan", "post"): "CognitionPlanResponse",
    }
    for (path, method), schema_name in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        assert schema["$ref"].endswith(f"/{schema_name}")

    client = TestClient(app)
    ChatResponse.model_validate(client.post("/api/chat", json={"text": "hello"}).json())
    ConversationDiagnosticsResponse.model_validate(
        client.get("/api/conversation/diagnostics").json()
    )
    ConversationHistoryResponse.model_validate(client.get("/api/live").json())
    ConversationHistoryResponse.model_validate(client.get("/api/conversations").json())
    CognitionContextResponse.model_validate(
        client.get("/api/cognition/context?refresh_perception=true").json()
    )
    CognitionPlanResponse.model_validate(
        client.post("/api/cognition/plan", json={"text": "inspect area-a"}).json()
    )
