"""Governance and operator-directory FastAPI routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

GovernancePayload = Callable[[], dict[str, Any]]
CurrentOperatorPayload = Callable[..., dict[str, Any]]
AuthorizationPayload = Callable[..., dict[str, Any]]
MissionJsonWithStatus = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]


def register_governance_routes(
    app: FastAPI,
    *,
    governance_payload: GovernancePayload,
    current_operator_payload: CurrentOperatorPayload,
    authorization_payload: AuthorizationPayload,
    mission_json: MissionJsonWithStatus,
    cors_options_response: CorsOptions,
) -> None:
    """Register product-facing governance status routes."""

    @app.get("/api/governance/operator-directory", tags=["Governance"])
    async def operator_directory() -> JSONResponse:
        return mission_json(governance_payload())

    @app.options("/api/governance/operator-directory", include_in_schema=False)
    async def operator_directory_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.get("/api/governance/current-operator", tags=["Governance"])
    async def current_operator(request: Request, operator_id: str | None = None) -> JSONResponse:
        return mission_json(current_operator_payload(operator_id, request.headers))

    @app.options("/api/governance/current-operator", include_in_schema=False)
    async def current_operator_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.post("/api/governance/authorize", tags=["Governance"])
    async def authorize_operator(request: Request) -> JSONResponse:
        body = await request.json()
        permission = str(body.get("permission") or "").strip()
        if not permission:
            return mission_json({"error": "permission is required"}, status_code=400)
        operator_id = (
            str(body.get("operator_id") or "").strip()
            or request.headers.get("x-askme-operator-id")
            or request.headers.get("x-operator-id")
        )
        decision = authorization_payload(operator_id, permission, request.headers, body)
        status_code = 200 if decision.get("allowed") else 403
        return mission_json(decision, status_code=status_code)

    @app.options("/api/governance/authorize", include_in_schema=False)
    async def authorize_operator_cors() -> Response:
        return cors_options_response("POST, OPTIONS")
