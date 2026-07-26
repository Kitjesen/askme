"""FastAPI routes for the operator-facing target-hardware Voice Lab."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.voice import VoiceLabDevicesResponse, VoiceLabRunResponse
from askme.config import project_root
from askme.voice.diagnostics.hardware_audio_capture import HardwareAudioCaptureError
from askme.voice.lab.service import (
    VoiceLabConflict,
    VoiceLabError,
    VoiceLabEvidenceUnavailable,
    VoiceLabNotFound,
    VoiceLabService,
    VoiceLabStateError,
    VoiceLabValidationError,
)

MissionJson = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
_VOICE_LAB_PERMISSION = "voice:system:update"


def register_voice_lab_routes(
    app: FastAPI,
    *,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    authorize: Authorize,
    service: VoiceLabService | None = None,
    artifact_root: Path | None = None,
) -> None:
    """Register one app-scoped lab service and its product API."""

    resolved_service = service or VoiceLabService(
        artifact_root or (project_root() / "artifacts" / "voice-lab")
    )
    app.include_router(
        create_voice_lab_router(
            service=resolved_service,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            authorize=authorize,
        )
    )


def create_voice_lab_router(
    *,
    service: VoiceLabService,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    authorize: Authorize,
) -> APIRouter:
    router = APIRouter(tags=["Voice Lab"])

    async def read_body(request: Request) -> tuple[dict[str, Any], JSONResponse | None]:
        """Read one optional JSON object and translate decode/type failures at the HTTP edge."""

        try:
            body = await optional_json_body(request)
            if not isinstance(body, dict):
                raise ValueError("JSON object body required")
            return body, None
        except ValueError as exc:
            return {}, mission_json({"error": str(exc)}, status_code=400)

    @router.get(
        "/api/voice/lab/devices",
        response_model=VoiceLabDevicesResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_devices(request: Request) -> JSONResponse:
        failure = authorize(request, {}, _VOICE_LAB_PERMISSION)
        if failure is not None:
            return failure
        try:
            return mission_json(await asyncio.to_thread(service.list_devices))
        except Exception as exc:
            return mission_json({"error": str(exc)}, status_code=503)

    @router.post(
        "/api/voice/lab/runs",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
        status_code=201,
    )
    async def voice_lab_create_run(
        request: Request,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        body, invalid_body = await read_body(request)
        if invalid_body is not None:
            return invalid_body
        failure = authorize(request, body, _VOICE_LAB_PERMISSION)
        if failure is not None:
            return failure
        if not idempotency_key:
            return mission_json({"error": "Idempotency-Key header is required"}, status_code=428)
        try:
            payload = await asyncio.to_thread(
                service.create_run,
                body,
                idempotency_key=idempotency_key,
            )
            return mission_json(payload, status_code=201)
        except Exception as exc:
            return _error_response(exc, mission_json)

    @router.get(
        "/api/voice/lab/runs/{run_id}",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_get_run(run_id: str, request: Request) -> JSONResponse:
        failure = authorize(request, {}, _VOICE_LAB_PERMISSION)
        if failure is not None:
            return failure
        try:
            return mission_json(await asyncio.to_thread(service.get_run, run_id))
        except Exception as exc:
            return _error_response(exc, mission_json)

    async def mutate(
        request: Request,
        run_id: str,
        operation: Callable[..., dict[str, Any]],
        *,
        idempotency_key: str | None,
        if_match: str | None,
        include_body: bool,
        operation_args: tuple[Any, ...] = (),
    ) -> JSONResponse:
        if include_body:
            body, invalid_body = await read_body(request)
            if invalid_body is not None:
                return invalid_body
        else:
            body = {}
        failure = authorize(request, body, _VOICE_LAB_PERMISSION)
        if failure is not None:
            return failure
        if not idempotency_key:
            return mission_json({"error": "Idempotency-Key header is required"}, status_code=428)
        try:
            expected_version = _parse_if_match(if_match)
            kwargs: dict[str, Any] = {
                "expected_version": expected_version,
                "idempotency_key": idempotency_key,
            }
            args: tuple[Any, ...] = (run_id, *operation_args)
            if include_body:
                args = (*args, body)
            payload = await asyncio.to_thread(operation, *args, **kwargs)
            return mission_json(payload)
        except Exception as exc:
            return _error_response(exc, mission_json)

    @router.post(
        "/api/voice/lab/runs/{run_id}/device-check",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_device_check(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.check_devices,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/calibration",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_calibration(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.calibrate,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=True,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/trials",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_trial(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.submit_trial,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=True,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/trials/begin",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_begin_trial(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.begin_trial,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/trials/{attempt_id}/execute",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_execute_trial(
        request: Request,
        run_id: str,
        attempt_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.execute_trial,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
            operation_args=(attempt_id,),
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/pause",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_pause(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.pause,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/resume",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_resume(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.resume,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/abort",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_abort(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.abort,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
        )

    @router.post(
        "/api/voice/lab/runs/{run_id}/report",
        response_model=VoiceLabRunResponse,
        response_model_exclude_none=True,
    )
    async def voice_lab_report(
        request: Request,
        run_id: str,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        if_match: str | None = Header(None, alias="If-Match"),
    ) -> JSONResponse:
        return await mutate(
            request,
            run_id,
            service.generate_report,
            idempotency_key=idempotency_key,
            if_match=if_match,
            include_body=False,
        )

    for path in (
        "/api/voice/lab/devices",
        "/api/voice/lab/runs",
        "/api/voice/lab/runs/{run_id}",
        "/api/voice/lab/runs/{run_id}/device-check",
        "/api/voice/lab/runs/{run_id}/calibration",
        "/api/voice/lab/runs/{run_id}/trials",
        "/api/voice/lab/runs/{run_id}/trials/begin",
        "/api/voice/lab/runs/{run_id}/trials/{attempt_id}/execute",
        "/api/voice/lab/runs/{run_id}/pause",
        "/api/voice/lab/runs/{run_id}/resume",
        "/api/voice/lab/runs/{run_id}/abort",
        "/api/voice/lab/runs/{run_id}/report",
    ):

        async def options(endpoint: str = path) -> Response:
            del endpoint
            return cors_options_response("GET, POST, OPTIONS")

        router.add_api_route(path, options, methods=["OPTIONS"], include_in_schema=False)

    return router


def _parse_if_match(value: str | None) -> int:
    if value is None or not value.strip():
        raise VoiceLabValidationError("If-Match header is required")
    text = value.strip().strip('"')
    try:
        parsed = int(text)
    except ValueError as exc:
        raise VoiceLabValidationError("If-Match must contain the current integer version") from exc
    if parsed < 1:
        raise VoiceLabValidationError("If-Match must contain the current integer version")
    return parsed


def _error_response(exc: Exception, mission_json: MissionJson) -> JSONResponse:
    if isinstance(exc, VoiceLabNotFound):
        status_code = 404
    elif isinstance(exc, (VoiceLabConflict, VoiceLabStateError)):
        status_code = 409
    elif isinstance(exc, VoiceLabValidationError):
        status_code = 428 if "If-Match header" in str(exc) else 422
    elif isinstance(exc, (HardwareAudioCaptureError, VoiceLabEvidenceUnavailable)):
        status_code = 503
    elif isinstance(exc, VoiceLabError):
        status_code = 500
    else:
        status_code = 500
    return mission_json({"error": str(exc)}, status_code=status_code)


__all__ = ["create_voice_lab_router", "register_voice_lab_routes"]
