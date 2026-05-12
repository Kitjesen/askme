"""Park space cognition FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

SpaceDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
MissionJsonWithStatus = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]


def register_space_routes(
    app: FastAPI,
    *,
    dispatch_space: SpaceDispatch,
    mission_json: MissionJsonWithStatus,
    optional_json_body: Callable[[Request], Awaitable[dict[str, Any]]],
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize | None = None,
) -> None:
    """Register product-facing park point, service point, and guide routes."""

    async def _space_get(method_name: str) -> JSONResponse:
        try:
            return mission_json(await dispatch_space(method_name, {}))
        except Exception as exc:
            logger.error("Space endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    async def _space_post(
        request: Request,
        *,
        method_name: str,
        permission: str,
    ) -> JSONResponse:
        try:
            body = await optional_json_body(request)
            if authorize is not None:
                failure = authorize(request, body, permission)
                if failure is not None:
                    return failure
            return mission_json(await dispatch_space(method_name, body))
        except Exception as exc:
            logger.error("Space endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/space/health", tags=["Space"])
    async def space_health() -> JSONResponse:
        return await _space_get("health_payload")

    @app.get("/api/space/points", tags=["Space"])
    async def space_points() -> JSONResponse:
        return await _space_get("points_payload")

    @app.get("/api/space/service-points", tags=["Space"])
    async def space_service_points() -> JSONResponse:
        return await _space_get("service_points_payload")

    @app.get("/api/space/routes", tags=["Space"])
    async def space_routes() -> JSONResponse:
        return await _space_get("routes_payload")

    @app.post("/api/space/resolve-destination", tags=["Space"])
    async def space_resolve_destination(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="resolve_destination_payload",
            permission="knowledge:read",
        )

    @app.post("/api/space/guide", tags=["Space"])
    async def space_guide(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="guide_payload",
            permission="field:event:create",
        )

    @app.options("/api/space/resolve-destination", include_in_schema=False)
    async def space_resolve_destination_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/space/guide", include_in_schema=False)
    async def space_guide_cors() -> Response:
        return cors_options_response("POST, OPTIONS")
