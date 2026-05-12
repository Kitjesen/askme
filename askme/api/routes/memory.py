"""Memory and knowledge FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MemoryDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
MissionJson = Callable[[dict[str, Any]], JSONResponse]
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

    async def _json_body(request: Request) -> dict[str, Any]:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("JSON object body required")
        return body

    async def _dispatch_payload(
        route_name: str,
        method_name: str,
        request: Request,
        permission: str | None = None,
    ) -> JSONResponse:
        try:
            body = await _json_body(request)
            if authorize is not None and permission:
                failure = authorize(request, body, permission)
                if failure is not None:
                    return failure
            result = await dispatch_memory(method_name, body)
            return mission_json(result)
        except Exception as exc:
            logger.error("%s endpoint failed: %s", route_name, exc)
            status_code = 503 if "not configured" in str(exc) else 500
            return mission_json({"error": str(exc)}, status_code=status_code)

    @app.post("/api/memory/search", tags=["Memory"])
    async def memory_search(request: Request) -> JSONResponse:
        """Search the configured memory/RAG backend and return auditable evidence."""

        return await _dispatch_payload("Memory search", "search_payload", request, "knowledge:read")

    @app.options("/api/memory/search", include_in_schema=False)
    async def memory_search_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.post("/api/knowledge/preview", tags=["Memory"])
    async def knowledge_preview(request: Request) -> JSONResponse:
        """Preview uploaded knowledge records without indexing them."""

        return await _dispatch_payload("Knowledge preview", "preview_payload", request, "knowledge:preview")

    @app.post("/api/knowledge/import", tags=["Memory"])
    async def knowledge_import(request: Request) -> JSONResponse:
        """Import uploaded knowledge records into the configured memory backend."""

        return await _dispatch_payload("Knowledge import", "import_payload", request, "knowledge:import")

    @app.post("/api/knowledge/list", tags=["Memory"])
    async def knowledge_list(request: Request) -> JSONResponse:
        """List locally indexed knowledge records for Knowledge Console."""

        return await _dispatch_payload("Knowledge list", "list_knowledge_payload", request, "knowledge:read")

    @app.post("/api/knowledge/update", tags=["Memory"])
    async def knowledge_update(request: Request) -> JSONResponse:
        """Update knowledge metadata such as approval status or soft deletion."""

        body = await _json_body(request)
        action = str(body.get("action") or "").strip().lower()
        permission = _knowledge_update_permission(action)
        if authorize is not None:
            failure = authorize(request, body, permission)
            if failure is not None:
                return failure
        try:
            result = await dispatch_memory("update_knowledge_payload", body)
            return mission_json(result)
        except Exception as exc:
            logger.error("Knowledge update endpoint failed: %s", exc)
            status_code = 503 if "not configured" in str(exc) else 500
            return mission_json({"error": str(exc)}, status_code=status_code)

    @app.options("/api/knowledge/preview", include_in_schema=False)
    async def knowledge_preview_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/knowledge/import", include_in_schema=False)
    async def knowledge_import_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/knowledge/list", include_in_schema=False)
    async def knowledge_list_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/knowledge/update", include_in_schema=False)
    async def knowledge_update_cors() -> Response:
        return cors_options_response("POST, OPTIONS")


def _knowledge_update_permission(action: str) -> str:
    if action in {"publish", "approve", "reject", "resolve", "resolve_conflict"}:
        return "knowledge:approve"
    if action in {"delete", "restore"}:
        return "knowledge:delete"
    if action in {"rollback"}:
        return "knowledge:rollback"
    if action in {"rebuild", "rebuild_index", "reindex"}:
        return "knowledge:rebuild"
    return "knowledge:approve"
