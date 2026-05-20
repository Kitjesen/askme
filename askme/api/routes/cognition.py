"""Cognition planning FastAPI routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.cognition import CognitionContextResponse, CognitionPlanResponse
from askme.api.services.http_helpers import require_json_object

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

    app.include_router(
        create_cognition_router(
            dispatch_cognition=dispatch_cognition,
            json_error=json_error,
            cors_options_response=cors_options_response,
            cors_headers=cors_headers,
        )
    )


def create_cognition_router(
    *,
    dispatch_cognition: Dispatch,
    json_error: Callable[..., JSONResponse],
    cors_options_response: CorsOptions,
    cors_headers: dict[str, str],
) -> APIRouter:
    """Create the cognition router without binding it to an app factory."""

    router = APIRouter(tags=["Cognition"])

    @router.get(
        "/api/cognition/context",
        response_model=CognitionContextResponse,
        response_model_exclude_none=True,
    )
    async def cognition_context(refresh_perception: bool = False) -> JSONResponse:
        try:
            payload = await dispatch_cognition(
                "context_payload",
                refresh_perception=refresh_perception,
            )
            CognitionContextResponse.model_validate(payload)
            return JSONResponse(payload, headers={"Cache-Control": "no-store", **cors_headers})
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return json_error(str(exc), status_code=500)

    @router.post(
        "/api/cognition/plan",
        response_model=CognitionPlanResponse,
        response_model_exclude_none=True,
    )
    async def cognition_plan(request: Request) -> JSONResponse:
        try:
            body = require_json_object(await request.json())
            payload = await dispatch_cognition("plan_from_payload", body)
            CognitionPlanResponse.model_validate(payload)
            return JSONResponse(payload, headers={"Cache-Control": "no-store", **cors_headers})
        except ValueError as exc:
            return json_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return json_error(str(exc), status_code=500)

    @router.options("/api/cognition/context", include_in_schema=False)
    async def cognition_context_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/cognition/plan", include_in_schema=False)
    async def cognition_plan_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router
