"""Park space cognition FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from askme.api.schemas.space import (
    SpaceGuideResponse,
    SpaceHealthResponse,
    SpaceHistoryResponse,
    SpaceInteractionsResponse,
    SpaceManageResponse,
    SpacePointsResponse,
    SpaceProposalCreateResponse,
    SpaceProposalReviewResponse,
    SpaceProposalsResponse,
    SpaceResolveDestinationResponse,
    SpaceRollbackResponse,
    SpaceRoutesResponse,
    SpaceServicePointTriggerResponse,
    SpaceServicePointsResponse,
)

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

    app.include_router(
        create_space_router(
            dispatch_space=dispatch_space,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            logger=logger,
            authorize=authorize,
        )
    )


def create_space_router(
    *,
    dispatch_space: SpaceDispatch,
    mission_json: MissionJsonWithStatus,
    optional_json_body: Callable[[Request], Awaitable[dict[str, Any]]],
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize | None = None,
) -> APIRouter:
    """Create the park-space router without binding it to an app factory."""

    router = APIRouter(tags=["Space"])

    async def _space_get(
        method_name: str,
        response_model: type[BaseModel],
    ) -> JSONResponse:
        try:
            payload = await dispatch_space(method_name, {})
            response_model.model_validate(payload)
            return mission_json(payload)
        except Exception as exc:
            logger.error("Space endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    async def _space_post(
        request: Request,
        *,
        method_name: str,
        permission: str,
        response_model: type[BaseModel],
    ) -> JSONResponse:
        try:
            body = await optional_json_body(request)
            body.setdefault(
                "operator_id",
                request.headers.get("X-Askme-Operator-Id")
                or request.headers.get("X-Operator-Id")
                or "",
            )
            if authorize is not None:
                failure = authorize(request, body, permission)
                if failure is not None:
                    return failure
            payload = await dispatch_space(method_name, body)
            response_model.model_validate(payload)
            return mission_json(payload)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Space endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get("/api/space/health", response_model=SpaceHealthResponse)
    async def space_health() -> JSONResponse:
        return await _space_get("health_payload", SpaceHealthResponse)

    @router.get("/api/space/points", response_model=SpacePointsResponse)
    async def space_points() -> JSONResponse:
        return await _space_get("points_payload", SpacePointsResponse)

    @router.get(
        "/api/space/service-points",
        response_model=SpaceServicePointsResponse,
    )
    async def space_service_points() -> JSONResponse:
        return await _space_get("service_points_payload", SpaceServicePointsResponse)

    @router.get("/api/space/routes", response_model=SpaceRoutesResponse)
    async def space_routes() -> JSONResponse:
        return await _space_get("routes_payload", SpaceRoutesResponse)

    @router.get("/api/space/history", response_model=SpaceHistoryResponse)
    async def space_history() -> JSONResponse:
        return await _space_get("history_payload", SpaceHistoryResponse)

    @router.get("/api/space/proposals", response_model=SpaceProposalsResponse)
    async def space_proposals() -> JSONResponse:
        return await _space_get("proposals_payload", SpaceProposalsResponse)

    @router.get(
        "/api/space/interactions",
        response_model=SpaceInteractionsResponse,
    )
    async def space_interactions() -> JSONResponse:
        return await _space_get("interactions_payload", SpaceInteractionsResponse)

    @router.post(
        "/api/space/resolve-destination",
        response_model=SpaceResolveDestinationResponse,
    )
    async def space_resolve_destination(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="resolve_destination_payload",
            permission="knowledge:read",
            response_model=SpaceResolveDestinationResponse,
        )

    @router.post("/api/space/guide", response_model=SpaceGuideResponse)
    async def space_guide(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="guide_payload",
            permission="field:event:create",
            response_model=SpaceGuideResponse,
        )

    @router.post(
        "/api/space/service-point-trigger",
        response_model=SpaceServicePointTriggerResponse,
    )
    async def space_service_point_trigger(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="service_point_trigger_payload",
            permission="knowledge:read",
            response_model=SpaceServicePointTriggerResponse,
        )

    @router.post("/api/space/manage", response_model=SpaceManageResponse)
    async def space_manage(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="manage_payload",
            permission="knowledge:approve",
            response_model=SpaceManageResponse,
        )

    @router.post(
        "/api/space/proposals",
        response_model=SpaceProposalCreateResponse,
    )
    async def space_propose(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="propose_payload",
            permission="knowledge:import",
            response_model=SpaceProposalCreateResponse,
        )

    @router.post(
        "/api/space/proposals/review",
        response_model=SpaceProposalReviewResponse,
    )
    async def space_review_proposal(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="review_proposal_payload",
            permission="knowledge:approve",
            response_model=SpaceProposalReviewResponse,
        )

    @router.post("/api/space/rollback", response_model=SpaceRollbackResponse)
    async def space_rollback(request: Request) -> JSONResponse:
        return await _space_post(
            request,
            method_name="rollback_payload",
            permission="knowledge:rollback",
            response_model=SpaceRollbackResponse,
        )

    @router.options("/api/space/resolve-destination", include_in_schema=False)
    async def space_resolve_destination_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/space/guide", include_in_schema=False)
    async def space_guide_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/space/service-point-trigger", include_in_schema=False)
    async def space_service_point_trigger_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/space/manage", include_in_schema=False)
    async def space_manage_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/space/rollback", include_in_schema=False)
    async def space_rollback_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/space/history", include_in_schema=False)
    async def space_history_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/space/proposals", include_in_schema=False)
    async def space_proposals_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @router.options("/api/space/interactions", include_in_schema=False)
    async def space_interactions_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/space/proposals/review", include_in_schema=False)
    async def space_review_proposal_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router
