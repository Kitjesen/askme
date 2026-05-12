"""Cognition planning FastAPI routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
JsonWithStatus = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]


def register_cognition_routes(
    app: FastAPI,
    *,
    dispatch_cognition: Dispatch,
    json_error: Callable[..., JSONResponse],
    cors_options_response: CorsOptions,
    cors_headers: dict[str, str],
) -> None:
    """Register cognition context and plan routes."""

    @app.get("/api/cognition/context", tags=["Cognition"])
    async def cognition_context(refresh_perception: bool = False) -> JSONResponse:
        try:
            payload = await dispatch_cognition(
                "context_payload",
                refresh_perception=refresh_perception,
            )
            return JSONResponse(payload, headers={"Cache-Control": "no-store", **cors_headers})
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return json_error(str(exc), status_code=500)

    @app.post("/api/cognition/plan", tags=["Cognition"])
    async def cognition_plan(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return json_error("JSON object body required", status_code=400)
            payload = await dispatch_cognition("plan_from_payload", body)
            return JSONResponse(payload, headers={"Cache-Control": "no-store", **cors_headers})
        except ValueError as exc:
            return json_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return json_error(str(exc), status_code=500)

    @app.options("/api/cognition/context", include_in_schema=False)
    async def cognition_context_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/cognition/plan", include_in_schema=False)
    async def cognition_plan_cors() -> Response:
        return cors_options_response("POST, OPTIONS")
