"""Governance and operator-directory FastAPI routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

GovernancePayload = Callable[[], dict[str, Any]]
MissionJsonWithStatus = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]


def register_governance_routes(
    app: FastAPI,
    *,
    governance_payload: GovernancePayload,
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
