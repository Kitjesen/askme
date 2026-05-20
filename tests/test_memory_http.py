"""HTTP tests for memory and knowledge routes."""

from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


def test_memory_search_endpoint_dispatches_handler():
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


def test_memory_health_endpoint_dispatches_handler():
    class Handler:
        async def health_payload(self, payload):
            assert payload.get("operator_id") == "dashboard.operator"
            assert payload.get("operator_auth", {}).get("allowed") is True
            return {
                "status": "ready",
                "ready": True,
                "customer_status": "Customer knowledge is ready for evidence-backed answers.",
                "customer_next_step": "Keep published knowledge maintained and show evidence.",
                "current_backend": "vector",
                "configured_backend": "vector",
                "paths": {
                    "catalog": "data/memory/catalog/records.json",
                    "vector_store": "data/memory/vectors/store.json",
                },
                "counts": {"catalog_total": 2, "prompt_eligible": 1},
                "memory_strategy": {
                    "customer_knowledge": {"backend": "vector"},
                    "robot_behavior_memory": {
                        "backend": "robotmem",
                        "enabled": False,
                    },
                },
                "answer_contract": {
                    "contract_type": "askme.customer_knowledge_answer_contract.v1",
                    "evidence_required": True,
                    "approved_knowledge_only": True,
                    "current_knowledge_only": True,
                    "conflict_free_knowledge_only": True,
                    "show_evidence_in_answer": True,
                    "refuse_when_no_evidence": True,
                    "refuse_when_expired": True,
                    "refuse_when_conflicting": True,
                    "robot_behavior_memory_enters_customer_prompt": False,
                },
            }

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            memory_handler=Handler(),
        )
    )

    response = client.get("/api/memory/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["customer_status"] == (
        "Customer knowledge is ready for evidence-backed answers."
    )
    assert payload["customer_next_step"] == (
        "Keep published knowledge maintained and show evidence."
    )
    assert payload["current_backend"] == "vector"
    assert payload["memory_strategy"]["robot_behavior_memory"]["backend"] == "robotmem"
    assert payload["answer_contract"]["evidence_required"] is True
    assert payload["answer_contract"]["robot_behavior_memory_enters_customer_prompt"] is False


def test_memory_search_rejects_missing_query_before_dispatch():
    class Handler:
        def __init__(self):
            self.calls = 0

        async def search_payload(self, payload):
            self.calls += 1
            return {"query": payload.get("query"), "results": []}

    handler = Handler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            memory_handler=handler,
        )
    )

    response = client.post("/api/memory/search", json={"top_k": 3})

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "invalid_request"
    assert "query or text is required" in payload["message"]
    assert handler.calls == 0


def test_knowledge_preview_endpoint_dispatches_handler():
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


def test_knowledge_preview_rejects_non_object_json_body_before_dispatch():
    class Handler:
        def __init__(self):
            self.calls = 0

        async def preview_payload(self, payload):
            self.calls += 1
            return {"parsed": 0, "records": []}

    handler = Handler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            memory_handler=handler,
        )
    )

    response = client.post("/api/knowledge/preview", json=["not", "an", "object"])

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "invalid_request"
    assert payload["message"] == "JSON object body required"
    assert handler.calls == 0


def test_knowledge_preview_rejects_empty_content_before_dispatch():
    class Handler:
        def __init__(self):
            self.calls = 0

        async def preview_payload(self, payload):
            self.calls += 1
            return {"parsed": 0, "records": []}

    handler = Handler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            memory_handler=handler,
        )
    )

    response = client.post(
        "/api/knowledge/preview",
        json={"filename": "faq.md", "content": "   "},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "invalid_request"
    assert payload["field"] == "content"
    assert handler.calls == 0


def test_knowledge_import_endpoint_dispatches_handler():
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


def test_knowledge_list_endpoint_dispatches_handler():
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


def test_knowledge_update_endpoint_dispatches_handler():
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


def test_knowledge_update_rejects_non_object_json_body_before_dispatch():
    class Handler:
        async def update_knowledge_payload(self, payload):
            raise AssertionError("update handler should not be called")

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            memory_handler=Handler(),
        )
    )

    response = client.post(
        "/api/knowledge/update",
        json=["not", "an", "object"],
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "invalid_request"
    assert payload["message"] == "JSON object body required"


def test_knowledge_update_blocks_operator_without_approval_role():
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
