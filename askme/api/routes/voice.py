"""Voice profile FastAPI routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.voice import (
    VoiceProfileCatalogResponse,
    VoiceProfileUpdateResponse,
    VoiceSystemControlResponse,
    VoiceSystemUpdateResponse,
)

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

    @router.get(
        "/api/voice/system",
        response_model=VoiceSystemControlResponse,
    )
    async def voice_system_control(request: Request) -> JSONResponse:
        try:
            full_access = authorize(request, {}, "voice:system:update") is None
            if not full_access:
                failure = authorize(request, {}, "voice:profile:read")
                if failure is not None:
                    return failure
            result = await dispatch_voice("system_control_payload")
            if not full_access:
                result = _customer_voice_system_payload(result)
            VoiceSystemControlResponse.model_validate(result)
            return mission_json(result)
        except RuntimeError as exc:
            return mission_json({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/voice/system/switch",
        response_model=VoiceSystemUpdateResponse,
    )
    async def voice_system_switch(request: Request) -> JSONResponse:
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "voice:system:update")
            if failure is not None:
                return failure
            result = await dispatch_voice("switch_system_component_payload", body)
            VoiceSystemUpdateResponse.model_validate(result)
            status_code = 200 if result.get("updated") or result.get("state") == "pending" else 422
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return mission_json({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/voice/system/prompt",
        response_model=VoiceSystemUpdateResponse,
    )
    async def voice_system_prompt(request: Request) -> JSONResponse:
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "voice:system:update")
            if failure is not None:
                return failure
            result = await dispatch_voice("update_prompt_payload", body)
            VoiceSystemUpdateResponse.model_validate(result)
            return mission_json(result, status_code=200 if result.get("updated") else 422)
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

    @router.options("/api/voice/system", include_in_schema=False)
    async def voice_system_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/voice/system/switch", include_in_schema=False)
    async def voice_system_switch_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/voice/system/prompt", include_in_schema=False)
    async def voice_system_prompt_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router


def _customer_voice_system_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the customer-safe status view without prompt or memory internals."""

    result = _strip_customer_voice_internals(payload)
    prompt = payload.get("prompt")
    memory = payload.get("memory")
    memory_status = memory.get("status") if isinstance(memory, dict) else None
    result["prompt"] = {
        "redacted": True,
        "configured": bool(prompt),
    }
    result["memory"] = {
        "redacted": True,
        "available": bool(memory),
        "status": str(memory_status or "unknown"),
    }
    result.pop("persistence", None)
    return result


def _strip_customer_voice_internals(value: Any) -> Any:
    """Remove sensitive voice-console fields even if a provider nests them."""

    sensitive_keys = {
        "conversation_history",
        "digests",
        "episodes",
        "memory",
        "persistence",
        "persona",
        "prompt",
        "records",
        "system_prompt",
        "user_prefix",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _strip_customer_voice_internals(item)
            for key, item in value.items()
            if str(key).lower() not in sensitive_keys
        }
    if isinstance(value, list):
        return [_strip_customer_voice_internals(item) for item in value]
    return value
