"""Voice profile FastAPI routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.voice import VoiceProfileCatalogResponse, VoiceProfileUpdateResponse

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]


def register_voice_routes(
    app: FastAPI,
    *,
    dispatch_voice: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    authorize: Authorize,
) -> None:
    """Register customer-selectable voice profile routes."""

    app.include_router(
        create_voice_router(
            dispatch_voice=dispatch_voice,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            authorize=authorize,
        )
    )


def create_voice_router(
    *,
    dispatch_voice: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    authorize: Authorize,
) -> APIRouter:
    """Create the voice profile router without binding it to an app factory."""

    router = APIRouter(tags=["Voice"])

    @router.get(
        "/api/voice/profiles",
        response_model=VoiceProfileCatalogResponse,
    )
    async def voice_profiles() -> JSONResponse:
        try:
            result = await dispatch_voice("voice_profiles_payload")
            VoiceProfileCatalogResponse.model_validate(result)
            return mission_json(result)
        except RuntimeError as exc:
            return mission_json({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/voice/profile",
        response_model=VoiceProfileUpdateResponse,
    )
    async def voice_profile_set(request: Request) -> JSONResponse:
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "voice:profile:update")
            if failure is not None:
                return failure
            result = await dispatch_voice("set_voice_profile_payload", body)
            VoiceProfileUpdateResponse.model_validate(result)
            status_code = 200 if result.get("updated") else 422
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return mission_json({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return mission_json({"error": str(exc)}, status_code=500)

    @router.options("/api/voice/profiles", include_in_schema=False)
    async def voice_profiles_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/voice/profile", include_in_schema=False)
    async def voice_profile_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router
