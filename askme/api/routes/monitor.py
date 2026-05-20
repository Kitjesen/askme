"""Dashboard monitor FastAPI routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from askme.api.schemas.conversation import ConversationHistoryResponse
from askme.api.schemas.monitor import SystemStatusResponse
from askme.api.services.monitor_service import MonitorService

_NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
_CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


def register_monitor_routes(
    app: FastAPI,
    *,
    monitor_service: MonitorService,
    logger: logging.Logger,
) -> None:
    """Register dashboard monitor and conversation history routes."""

    app.include_router(create_monitor_router(monitor_service=monitor_service, logger=logger))


def create_monitor_router(
    *,
    monitor_service: MonitorService,
    logger: logging.Logger,
) -> APIRouter:
    """Create the monitor router without binding it to an app factory."""

    router = APIRouter(tags=["Monitor"])

    @router.get(
        "/api/status",
        response_model=SystemStatusResponse,
        response_model_exclude_none=True,
    )
    async def system_status() -> JSONResponse:
        """Unified system status - all key metrics in one endpoint."""
        payload = monitor_service.system_status_payload()
        response = SystemStatusResponse.model_validate(payload)
        return JSONResponse(
            response.model_dump(mode="python", exclude_unset=True),
            headers=_NO_STORE_HEADERS,
        )

    @router.get(
        "/api/live",
        response_model=ConversationHistoryResponse,
        response_model_exclude_none=True,
    )
    async def live() -> JSONResponse:
        """Return in-memory conversation history (voice + web chat combined)."""
        try:
            payload = monitor_service.live_payload()
            ConversationHistoryResponse.model_validate(payload)
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            return JSONResponse(
                {"messages": [], "count": 0, "error": str(exc)},
                status_code=500,
                headers=_CORS_HEADERS,
            )

    @router.get(
        "/api/conversations",
        response_model=ConversationHistoryResponse,
        response_model_exclude_none=True,
    )
    async def conversations() -> JSONResponse:
        """Return conversation history for the monitor UI."""
        try:
            payload = monitor_service.conversation_history_payload()
            ConversationHistoryResponse.model_validate(payload)
            return JSONResponse(
                payload,
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.error("Conversations endpoint failed: %s", exc)
            return JSONResponse(
                {"messages": [], "count": 0, "error": str(exc)},
                status_code=500,
                headers=_CORS_HEADERS,
            )

    return router
