from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from askme.api.routes.memory import register_memory_routes
from askme.api.schemas.memory import (
    KnowledgeImportResponse,
    KnowledgeListResponse,
    KnowledgePreviewRequest,
    KnowledgePreviewResponse,
    KnowledgeUpdateResponse,
    MemoryHealthResponse,
    MemorySearchRequest,
    MemorySearchResponse,
)
from askme.api.services.knowledge_route_payloads import (
    invalid_request_payload,
    knowledge_update_permission,
    memory_route_failure,
    validate_memory_dispatch_payload,
    validate_payload,
    validation_error_message,
)


def test_invalid_request_payload_uses_stable_customer_envelope() -> None:
    assert invalid_request_payload("content is required.", field="content") == {
        "ok": False,
        "error": "invalid_request",
        "message": "content is required.",
        "field": "content",
    }


def test_validate_payload_normalizes_memory_search_text() -> None:
    payload = validate_payload({"text": "  west gate route  ", "top_k": 3}, MemorySearchRequest)

    assert payload["query"] == "west gate route"
    assert payload["text"] == "west gate route"
    assert payload["top_k"] == 3


def test_validate_payload_preserves_knowledge_preview_extension_fields() -> None:
    payload = validate_payload(
        {
            "filename": "site.md",
            "content": "- Restroom east",
            "category": "route",
            "custom_field": "kept",
        },
        KnowledgePreviewRequest,
    )

    assert payload["filename"] == "site.md"
    assert payload["content"] == "- Restroom east"
    assert payload["custom_field"] == "kept"


def test_validate_memory_dispatch_payload_only_validates_contract_routes() -> None:
    validated = validate_memory_dispatch_payload(
        "preview_payload",
        {"content": "Safety SOP", "owner": " delivery "},
    )
    passthrough = validate_memory_dispatch_payload("list_knowledge_payload", {"limit": 50})

    assert validated["content"] == "Safety SOP"
    assert validated["owner"] == "delivery"
    assert passthrough == {"limit": 50}


def test_memory_health_response_locks_customer_answer_contract() -> None:
    payload = MemoryHealthResponse.model_validate(
        {
            "status": "ready",
            "ready": True,
            "customer_status": "客户知识库可用于有证据回答。",
            "customer_next_step": "继续维护已发布知识，并在回答气泡展示引用证据。",
            "catalog_answer_ready": True,
            "retrieval_runtime_ready": True,
            "current_backend": "mempalace",
            "configured_backend": "mempalace",
            "selected_backend": "mempalace",
            "selected_backend_ready": True,
            "selected_backend_installed": True,
            "fallback_backend": "vector",
            "fallback_ready": True,
            "memory_strategy": {
                "customer_knowledge": {"enters_prompt": True},
                "robot_behavior_memory": {"enters_prompt": False},
            },
            "paths": {"catalog": "data/knowledge/catalog.json"},
            "counts": {"prompt_eligible": 3},
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
            "warnings": [],
        }
    )

    assert payload.ready is True
    assert payload.answer_contract.evidence_required is True
    assert payload.answer_contract.approved_knowledge_only is True
    assert payload.answer_contract.show_evidence_in_answer is True
    assert payload.answer_contract.robot_behavior_memory_enters_customer_prompt is False


def test_memory_health_response_exposes_expiry_policy_gap() -> None:
    payload = MemoryHealthResponse.model_validate(
        {
            "status": "ready",
            "ready": True,
            "customer_status": "知识过期拦截未启用，不能作为客户回答依据。",
            "customer_next_step": "先启用知识过期拦截，再允许知识进入回答。",
            "answer_contract": {
                "contract_type": "askme.customer_knowledge_answer_contract.v1",
                "evidence_required": True,
                "approved_knowledge_only": True,
                "current_knowledge_only": False,
                "conflict_free_knowledge_only": True,
                "show_evidence_in_answer": True,
                "refuse_when_no_evidence": True,
                "refuse_when_expired": False,
                "refuse_when_conflicting": True,
                "robot_behavior_memory_enters_customer_prompt": False,
            },
            "warnings": ["rag_expiry_not_enforced"],
        }
    )

    assert payload.answer_contract.current_knowledge_only is False
    assert payload.answer_contract.refuse_when_expired is False
    assert "rag_expiry_not_enforced" in payload.warnings


def test_memory_health_route_exposes_response_schema_in_openapi() -> None:
    app = FastAPI()

    async def dispatch_memory(method: str, payload: dict[str, object]) -> dict[str, object]:
        _ = method, payload
        return {}

    def mission_json(payload: dict[str, object], status_code: int = 200) -> JSONResponse:
        return JSONResponse(payload, status_code=status_code)

    def cors_options_response(methods: str) -> Response:
        return Response(headers={"Access-Control-Allow-Methods": methods})

    def authorize(request: Request, payload: dict[str, object], permission: str) -> JSONResponse | None:
        _ = request, payload, permission
        return None

    register_memory_routes(
        app,
        dispatch_memory=dispatch_memory,
        mission_json=mission_json,
        cors_options_response=cors_options_response,
        logger=logging.getLogger("test"),
        authorize=authorize,
    )

    response_schema = app.openapi()["paths"]["/api/memory/health"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert response_schema["$ref"].endswith("/MemoryHealthResponse")


def test_memory_and_knowledge_routes_expose_product_response_schemas() -> None:
    app = FastAPI()

    async def dispatch_memory(method: str, payload: dict[str, object]) -> dict[str, object]:
        _ = payload
        if method == "search_payload":
            return {"query": "west gate", "results": [{"text": "fact"}], "rag": {}}
        if method == "preview_payload":
            return {"source": "site.md", "parsed": 1, "records": [{"text": "fact"}]}
        if method == "import_payload":
            return {"source": "site.md", "parsed": 1, "imported": 1, "skipped": 0}
        if method == "list_knowledge_payload":
            return {"backend": "vector", "total": 1, "records": [{"record_id": "k1"}]}
        if method == "update_knowledge_payload":
            return {"updated": True, "record_id": "k1", "patch": {"approval_status": "approved"}}
        raise AssertionError(method)

    def mission_json(payload: dict[str, object], status_code: int = 200) -> JSONResponse:
        return JSONResponse(payload, status_code=status_code)

    def cors_options_response(methods: str) -> Response:
        return Response(headers={"Access-Control-Allow-Methods": methods})

    def authorize(request: Request, payload: dict[str, object], permission: str) -> JSONResponse | None:
        _ = request, payload, permission
        return None

    register_memory_routes(
        app,
        dispatch_memory=dispatch_memory,
        mission_json=mission_json,
        cors_options_response=cors_options_response,
        logger=logging.getLogger("test"),
        authorize=authorize,
    )

    paths = app.openapi()["paths"]
    expected_refs = {
        ("/api/memory/search", "post"): "MemorySearchResponse",
        ("/api/knowledge/preview", "post"): "KnowledgePreviewResponse",
        ("/api/knowledge/import", "post"): "KnowledgeImportResponse",
        ("/api/knowledge/list", "post"): "KnowledgeListResponse",
        ("/api/knowledge/update", "post"): "KnowledgeUpdateResponse",
    }
    for (path, method), schema_name in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        assert schema["$ref"].endswith(f"/{schema_name}")


def test_memory_and_knowledge_response_models_accept_customer_payloads() -> None:
    MemorySearchResponse.model_validate(
        {"query": "咖啡店在哪", "results": [{"text": "梵木咖啡在二号楼"}], "rag": {}}
    )
    KnowledgePreviewResponse.model_validate(
        {"source": "fanmu-routes.md", "parsed": 1, "records": [{"text": "路线"}]}
    )
    KnowledgeImportResponse.model_validate(
        {"source": "fanmu-routes.md", "parsed": 1, "imported": 1, "skipped": 0}
    )
    KnowledgeListResponse.model_validate(
        {"backend": "vector", "total": 1, "records": [{"record_id": "k1"}]}
    )
    KnowledgeUpdateResponse.model_validate(
        {"updated": True, "record_id": "k1", "patch": {"approval_status": "approved"}}
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "ready",
            "ready": True,
            "customer_status": "客户知识库可用于有证据回答。",
            "customer_next_step": "继续维护已发布知识。",
        },
        {
            "status": "ready",
            "ready": True,
            "customer_status": "",
            "customer_next_step": "继续维护已发布知识。",
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
        },
    ],
)
def test_memory_health_response_requires_customer_contract(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MemoryHealthResponse.model_validate(payload)


def test_memory_route_failure_marks_missing_backend_as_unavailable() -> None:
    status_code, payload = memory_route_failure(RuntimeError("mempalace not configured"))

    assert status_code == 503
    assert payload == {
        "ok": False,
        "error": "memory_route_failed",
        "message": "mempalace not configured",
    }


def test_memory_route_failure_defaults_to_internal_error() -> None:
    status_code, payload = memory_route_failure(RuntimeError("boom"))

    assert status_code == 500
    assert payload["error"] == "memory_route_failed"
    assert payload["message"] == "boom"


def test_validation_error_message_extracts_message_and_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        validate_payload({"content": ""}, KnowledgePreviewRequest)

    message, field = validation_error_message(exc_info.value)

    assert message
    assert field == "content"


@pytest.mark.parametrize(
    ("action", "permission"),
    [
        ("approve", "knowledge:approve"),
        ("resolve_conflict", "knowledge:approve"),
        ("delete", "knowledge:delete"),
        ("restore", "knowledge:delete"),
        ("rollback", "knowledge:rollback"),
        ("rebuild_index", "knowledge:rebuild"),
        ("unknown", "knowledge:approve"),
        ("  PUBLISH  ", "knowledge:approve"),
    ],
)
def test_knowledge_update_permission_maps_actions(action: str, permission: str) -> None:
    assert knowledge_update_permission(action) == permission
