"""Dashboard monitor FastAPI routes."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

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

    @app.get("/api/status", tags=["Monitor"])
    async def system_status() -> JSONResponse:
        """Unified system status - all key metrics in one endpoint."""
        return JSONResponse(monitor_service.system_status_payload(), headers=_NO_STORE_HEADERS)

    @app.get("/api/live", tags=["Monitor"])
    async def live() -> JSONResponse:
        """Return in-memory conversation history (voice + web chat combined)."""
        try:
            return JSONResponse(monitor_service.live_payload(), headers=_NO_STORE_HEADERS)
        except Exception as exc:
            return JSONResponse(
                {"messages": [], "count": 0, "error": str(exc)},
                status_code=500,
                headers=_CORS_HEADERS,
            )

    @app.get("/api/conversations", tags=["Monitor"])
    async def conversations() -> JSONResponse:
        """Return conversation history for the monitor UI."""
        try:
            return JSONResponse(
                monitor_service.conversation_history_payload(),
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.error("Conversations endpoint failed: %s", exc)
            return JSONResponse(
                {"messages": [], "count": 0, "error": str(exc)},
                status_code=500,
                headers=_CORS_HEADERS,
            )
