"""Voice profile FastAPI routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from askme.api.schemas.voice import (
    VoicePlaybackCancelRequest,
    VoicePlaybackResponse,
    VoiceProfileCatalogResponse,
    VoiceProfileUpdateResponse,
    VoiceSystemControlResponse,
    VoiceSpeakRequest,
    VoiceSystemUpdateResponse,
)
from askme.ports import SpeechPlaybackError

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
    async def voice_system_control() -> JSONResponse:
        try:
            result = await dispatch_voice("system_control_payload")
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

    @router.post(
        "/api/voice/speak",
        response_model=VoicePlaybackResponse,
        status_code=202,
        summary="Play literal text on one explicitly targeted robot",
    )
    async def voice_speak(request: Request, body: VoiceSpeakRequest) -> JSONResponse:
        payload = body.model_dump()
        if str(payload.get("semantics") or "verbatim").strip().lower() != "verbatim":
            return mission_json(
                {
                    "error": "semantic_mode_mismatch",
                    "message": "This endpoint only speaks literal text and never invokes an LLM.",
                    "conversation_endpoint": "/api/chat",
                },
                status_code=422,
            )
        failure = authorize(request, payload, "voice:playback:create")
        if failure is not None:
            return failure
        if payload.get("priority") == "safety" or payload.get("queue_policy") == "replace_noncritical":
            failure = authorize(request, payload, "voice:playback:override")
            if failure is not None:
                return failure
        try:
            payload["idempotency_key"] = _required_idempotency_key(request)
            result = await dispatch_voice("speak_payload", payload)
            VoicePlaybackResponse.model_validate(result)
            response = mission_json(result, status_code=202)
            response.headers["Location"] = f"/api/voice/playbacks/{result['playback_id']}"
            return response
        except SpeechPlaybackError as exc:
            return mission_json(exc.to_payload(), status_code=exc.status_code)
        except ValueError as exc:
            return mission_json({"error": "invalid_playback_request", "message": str(exc)}, status_code=422)
        except RuntimeError as exc:
            return mission_json({"error": "voice_not_available", "message": str(exc)}, status_code=503)

    @router.post(
        "/api/voice/synthesize",
        response_model=VoicePlaybackResponse,
        status_code=202,
        summary="Synthesize literal text without playing it",
    )
    async def voice_synthesize(request: Request, body: VoiceSpeakRequest) -> JSONResponse:
        payload = body.model_dump()
        if str(payload.get("semantics") or "verbatim").strip().lower() != "verbatim":
            return mission_json(
                {
                    "error": "semantic_mode_mismatch",
                    "message": "Synthesis accepts literal text only.",
                    "conversation_endpoint": "/api/chat",
                },
                status_code=422,
            )
        failure = authorize(request, payload, "voice:synthesis:create")
        if failure is not None:
            return failure
        try:
            payload["idempotency_key"] = _required_idempotency_key(request)
            result = await dispatch_voice("synthesize_speech_payload", payload)
            VoicePlaybackResponse.model_validate(result)
            response = mission_json(result, status_code=202)
            response.headers["Location"] = f"/api/voice/playbacks/{result['playback_id']}"
            return response
        except SpeechPlaybackError as exc:
            return mission_json(exc.to_payload(), status_code=exc.status_code)
        except ValueError as exc:
            return mission_json({"error": "invalid_synthesis_request", "message": str(exc)}, status_code=422)
        except RuntimeError as exc:
            return mission_json({"error": "voice_not_available", "message": str(exc)}, status_code=503)

    @router.get(
        "/api/voice/playbacks/{playback_id}",
        response_model=VoicePlaybackResponse,
    )
    async def voice_playback_status(request: Request, playback_id: str) -> JSONResponse:
        actor: dict[str, Any] = {}
        failure = authorize(request, actor, "voice:playback:read")
        if failure is not None:
            return failure
        try:
            result = await dispatch_voice(
                "speech_playback_status_payload",
                playback_id,
                actor,
            )
            VoicePlaybackResponse.model_validate(result)
            return mission_json(result)
        except SpeechPlaybackError as exc:
            return mission_json(exc.to_payload(), status_code=exc.status_code)
        except RuntimeError as exc:
            return mission_json({"error": "voice_not_available", "message": str(exc)}, status_code=503)

    @router.post(
        "/api/voice/playbacks/{playback_id}/cancel",
        response_model=VoicePlaybackResponse,
    )
    async def voice_playback_cancel(
        request: Request,
        playback_id: str,
        body: VoicePlaybackCancelRequest,
    ) -> JSONResponse:
        payload = body.model_dump()
        failure = authorize(request, payload, "voice:playback:cancel")
        if failure is not None:
            return failure
        try:
            result = await dispatch_voice(
                "cancel_speech_playback_payload",
                playback_id,
                payload,
            )
            VoicePlaybackResponse.model_validate(result)
            return mission_json(result)
        except SpeechPlaybackError as exc:
            return mission_json(exc.to_payload(), status_code=exc.status_code)
        except RuntimeError as exc:
            return mission_json({"error": "voice_not_available", "message": str(exc)}, status_code=503)

    @router.get(
        "/api/voice/playbacks/{playback_id}/audio",
        response_class=FileResponse,
        responses={200: {"content": {"audio/wav": {}}}},
    )
    async def voice_playback_audio(request: Request, playback_id: str) -> Response:
        actor: dict[str, Any] = {}
        failure = authorize(request, actor, "voice:playback:read")
        if failure is not None:
            return failure
        try:
            artifact = await dispatch_voice(
                "speech_playback_audio_payload",
                playback_id,
                actor,
            )
            return FileResponse(
                path=str(artifact["path"]),
                media_type=str(artifact.get("media_type") or "audio/wav"),
                filename=str(artifact.get("filename") or f"{playback_id}.wav"),
                headers={"ETag": str(artifact.get("sha256") or "")},
            )
        except SpeechPlaybackError as exc:
            return mission_json(exc.to_payload(), status_code=exc.status_code)
        except (KeyError, RuntimeError) as exc:
            return mission_json(
                {"error": "audio_artifact_unavailable", "message": str(exc)},
                status_code=503,
            )

    @router.options("/api/voice/speak", include_in_schema=False)
    async def voice_speak_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/voice/synthesize", include_in_schema=False)
    async def voice_synthesize_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/voice/playbacks/{playback_id}", include_in_schema=False)
    async def voice_playback_status_cors(playback_id: str) -> Response:
        del playback_id
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/voice/playbacks/{playback_id}/cancel", include_in_schema=False)
    async def voice_playback_cancel_cors(playback_id: str) -> Response:
        del playback_id
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/voice/playbacks/{playback_id}/audio", include_in_schema=False)
    async def voice_playback_audio_cors(playback_id: str) -> Response:
        del playback_id
        return cors_options_response("GET, OPTIONS")

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


def _required_idempotency_key(request: Request) -> str:
    key = str(request.headers.get("Idempotency-Key") or "").strip()
    if not key:
        raise SpeechPlaybackError(
            "idempotency_key_required",
            "Idempotency-Key is required for speech creation.",
            status_code=428,
        )
    if len(key) > 128 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise SpeechPlaybackError(
            "invalid_idempotency_key",
            "Idempotency-Key must be 1-128 visible ASCII characters.",
            status_code=422,
        )
    return key
