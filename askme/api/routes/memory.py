"""Memory and knowledge FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

from askme.api.schemas.memory import (
    KnowledgeImportResponse,
    KnowledgeListResponse,
    KnowledgePreviewResponse,
    KnowledgeUpdateResponse,
    MemoryHealthResponse,
    MemorySearchResponse,
)
from askme.api.services.http_helpers import require_json_object
from askme.api.services.knowledge_route_payloads import (
    invalid_request_payload,
    knowledge_update_permission,
    memory_route_failure,
    validate_memory_dispatch_payload,
    validation_error_message,
)

MemoryDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
MissionJsonWithStatus = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]


def register_memory_routes(
    app: FastAPI,
    *,
    dispatch_memory: MemoryDispatch,
    mission_json: MissionJsonWithStatus,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize | None = None,
) -> None:
    """Register RAG and Knowledge Console routes."""

    app.include_router(
        create_memory_router(
            dispatch_memory=dispatch_memory,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
            authorize=authorize,
        )
    )


def create_memory_router(
    *,
    dispatch_memory: MemoryDispatch,
    mission_json: MissionJsonWithStatus,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize | None = None,
) -> APIRouter:
    """Create the memory and knowledge router without binding it to an app factory."""

    router = APIRouter(tags=["Memory"])

    async def _json_body(request: Request) -> dict[str, Any]:
        return require_json_object(await request.json())

    def _invalid_request_response(message: str, *, field: str | None = None) -> JSONResponse:
        return mission_json(invalid_request_payload(message, field=field), status_code=400)

    async def _dispatch_payload(
        route_name: str,
        method_name: str,
        request: Request,
        permission: str | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> JSONResponse:
        try:
            body = await _json_body(request)
            body = validate_memory_dispatch_payload(method_name, body)
            if authorize is not None and permission:
                failure = authorize(request, body, permission)
                if failure is not None:
                    return failure
            result = await dispatch_memory(method_name, body)
            if response_model is not None:
                response_model.model_validate(result)
            return mission_json(result)
        except ValidationError as exc:
            message, field = validation_error_message(exc)
            return _invalid_request_response(message, field=field)
        except ValueError as exc:
            return _invalid_request_response(str(exc))
        except Exception as exc:
            logger.error("%s endpoint failed: %s", route_name, exc)
            status_code, payload = memory_route_failure(exc)
            return mission_json(payload, status_code=status_code)

    @router.post(
        "/api/memory/search",
        response_model=MemorySearchResponse,
    )
    async def memory_search(request: Request) -> JSONResponse:
        """Search the configured memory/RAG backend and return auditable evidence."""

        return await _dispatch_payload(
            "Memory search",
            "search_payload",
            request,
            "knowledge:read",
            MemorySearchResponse,
        )

    @router.get(
        "/api/memory/health",
        response_model=MemoryHealthResponse,
        response_model_exclude_none=True,
    )
    async def memory_health(request: Request) -> JSONResponse:
        """Return product-facing memory backend readiness and data locations."""

        body: dict[str, Any] = {}
        if authorize is not None:
            failure = authorize(request, body, "knowledge:read")
            if failure is not None:
                return failure
        try:
            result = await dispatch_memory("health_payload", body)
            return mission_json(MemoryHealthResponse.model_validate(result).model_dump(mode="python"))
        except Exception as exc:
            logger.error("Memory health endpoint failed: %s", exc)
            status_code, payload = memory_route_failure(exc)
            return mission_json(payload, status_code=status_code)

    @router.options("/api/memory/search", include_in_schema=False)
    async def memory_search_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/memory/health", include_in_schema=False)
    async def memory_health_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.post(
        "/api/knowledge/preview",
        response_model=KnowledgePreviewResponse,
    )
    async def knowledge_preview(request: Request) -> JSONResponse:
        """Preview uploaded knowledge records without indexing them."""

        return await _dispatch_payload(
            "Knowledge preview",
            "preview_payload",
            request,
            "knowledge:preview",
            KnowledgePreviewResponse,
        )

    @router.post(
        "/api/knowledge/import",
        response_model=KnowledgeImportResponse,
    )
    async def knowledge_import(request: Request) -> JSONResponse:
        """Import uploaded knowledge records into the configured memory backend."""

        return await _dispatch_payload(
            "Knowledge import",
            "import_payload",
            request,
            "knowledge:import",
            KnowledgeImportResponse,
        )

    @router.post(
        "/api/knowledge/list",
        response_model=KnowledgeListResponse,
    )
    async def knowledge_list(request: Request) -> JSONResponse:
        """List locally indexed knowledge records for Knowledge Console."""

        return await _dispatch_payload(
            "Knowledge list",
            "list_knowledge_payload",
            request,
            "knowledge:read",
            KnowledgeListResponse,
        )

    @router.post(
        "/api/knowledge/update",
        response_model=KnowledgeUpdateResponse,
    )
    async def knowledge_update(request: Request) -> JSONResponse:
        """Update knowledge metadata such as approval status or soft deletion."""

        try:
            body = await _json_body(request)
        except ValueError as exc:
            return _invalid_request_response(str(exc))
        action = str(body.get("action") or "").strip().lower()
        permission = knowledge_update_permission(action)
        if authorize is not None:
            failure = authorize(request, body, permission)
            if failure is not None:
                return failure
        try:
            result = await dispatch_memory("update_knowledge_payload", body)
            KnowledgeUpdateResponse.model_validate(result)
            return mission_json(result)
        except Exception as exc:
            logger.error("Knowledge update endpoint failed: %s", exc)
            status_code = 503 if "not configured" in str(exc) else 500
            return mission_json({"error": str(exc)}, status_code=status_code)

    @router.options("/api/knowledge/preview", include_in_schema=False)
    async def knowledge_preview_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/knowledge/import", include_in_schema=False)
    async def knowledge_import_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/knowledge/list", include_in_schema=False)
    async def knowledge_list_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/knowledge/update", include_in_schema=False)
    async def knowledge_update_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router
