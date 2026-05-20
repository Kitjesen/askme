"""Governance and operator-directory FastAPI routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.governance import (
    AuthorizationDecisionResponse,
    CurrentOperatorResponse,
    IdentityGatewayReadinessResponse,
    OperatorDirectoryResponse,
)
from askme.api.services.http_helpers import require_json_object

GovernancePayload = Callable[[], dict[str, Any]]
IdentityReadinessPayload = Callable[[], dict[str, Any]]
CurrentOperatorPayload = Callable[..., dict[str, Any]]
AuthorizationPayload = Callable[..., dict[str, Any]]
MissionJsonWithStatus = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]


def register_governance_routes(
    app: FastAPI,
    *,
    governance_payload: GovernancePayload,
    identity_readiness_payload: IdentityReadinessPayload,
    current_operator_payload: CurrentOperatorPayload,
    authorization_payload: AuthorizationPayload,
    mission_json: MissionJsonWithStatus,
    cors_options_response: CorsOptions,
) -> None:
    """Register product-facing governance status routes."""

    app.include_router(
        create_governance_router(
            governance_payload=governance_payload,
            identity_readiness_payload=identity_readiness_payload,
            current_operator_payload=current_operator_payload,
            authorization_payload=authorization_payload,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
        )
    )


def create_governance_router(
    *,
    governance_payload: GovernancePayload,
    identity_readiness_payload: IdentityReadinessPayload,
    current_operator_payload: CurrentOperatorPayload,
    authorization_payload: AuthorizationPayload,
    mission_json: MissionJsonWithStatus,
    cors_options_response: CorsOptions,
) -> APIRouter:
    """Create the governance router without binding it to an app factory."""

    router = APIRouter(tags=["Governance"])

    @router.get(
        "/api/governance/operator-directory",
        response_model=OperatorDirectoryResponse,
    )
    async def operator_directory() -> JSONResponse:
        payload = governance_payload()
        OperatorDirectoryResponse.model_validate(payload)
        return mission_json(payload)

    @router.options("/api/governance/operator-directory", include_in_schema=False)
    async def operator_directory_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.get(
        "/api/governance/identity-readiness",
        response_model=IdentityGatewayReadinessResponse,
    )
    async def identity_readiness() -> JSONResponse:
        payload = identity_readiness_payload()
        IdentityGatewayReadinessResponse.model_validate(payload)
        return mission_json(payload)

    @router.options("/api/governance/identity-readiness", include_in_schema=False)
    async def identity_readiness_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.get(
        "/api/governance/current-operator",
        response_model=CurrentOperatorResponse,
    )
    async def current_operator(request: Request, operator_id: str | None = None) -> JSONResponse:
        payload = current_operator_payload(operator_id, request.headers)
        CurrentOperatorResponse.model_validate(payload)
        return mission_json(payload)

    @router.options("/api/governance/current-operator", include_in_schema=False)
    async def current_operator_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.post(
        "/api/governance/authorize",
        response_model=AuthorizationDecisionResponse,
    )
    async def authorize_operator(request: Request) -> JSONResponse:
        try:
            body = require_json_object(await request.json())
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        permission = str(body.get("permission") or "").strip()
        if not permission:
            return mission_json({"error": "permission is required"}, status_code=400)
        operator_id = (
            str(body.get("operator_id") or "").strip()
            or request.headers.get("x-askme-operator-id")
            or request.headers.get("x-operator-id")
        )
        decision = authorization_payload(operator_id, permission, request.headers, body)
        AuthorizationDecisionResponse.model_validate(decision)
        status_code = 200 if decision.get("allowed") else 403
        return mission_json(decision, status_code=status_code)

    @router.options("/api/governance/authorize", include_in_schema=False)
    async def authorize_operator_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router
