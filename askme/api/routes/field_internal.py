"""Internal field-ingest and robot callback routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.field_events import (
    FieldDeviceOnboardingResponse,
    FieldDeviceStatusResponse,
    FieldEventTriggerResponse,
    FieldIngestHelpResponse,
    FieldRuntimeDeliveryResponse,
)

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
FieldResultHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
FieldRuntimePolicy = Callable[..., Awaitable[dict[str, Any]]]
RuntimeCallbackTrust = Callable[..., dict[str, Any]]
RuntimeCallbackDeliveryBody = Callable[..., dict[str, Any]]


def register_field_internal_routes(
    app: FastAPI,
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    dispatch_field_voice_directive: FieldResultHook,
    dispatch_field_runtime_policy: FieldRuntimePolicy,
    runtime_callback_trust: RuntimeCallbackTrust,
    runtime_callback_delivery_body: RuntimeCallbackDeliveryBody,
    runtime_callback_secret: str | None,
    runtime_callback_max_age_s: float,
) -> None:
    """Register machine-facing field routes while preserving legacy URLs."""

    app.include_router(
        create_field_internal_router(
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            logger=logger,
            dispatch_field_voice_directive=dispatch_field_voice_directive,
            dispatch_field_runtime_policy=dispatch_field_runtime_policy,
            runtime_callback_trust=runtime_callback_trust,
            runtime_callback_delivery_body=runtime_callback_delivery_body,
            runtime_callback_secret=runtime_callback_secret,
            runtime_callback_max_age_s=runtime_callback_max_age_s,
        )
    )


def create_field_internal_router(
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    dispatch_field_voice_directive: FieldResultHook,
    dispatch_field_runtime_policy: FieldRuntimePolicy,
    runtime_callback_trust: RuntimeCallbackTrust,
    runtime_callback_delivery_body: RuntimeCallbackDeliveryBody,
    runtime_callback_secret: str | None,
    runtime_callback_max_age_s: float,
) -> APIRouter:
    """Create the internal field router without binding it to an app factory."""

    router = APIRouter(tags=["Field Operations"])

    @router.post(
        "/api/field/events/{event_id}/runtime-delivery",
        response_model=FieldRuntimeDeliveryResponse,
        response_model_exclude_none=True,
    )
    async def field_event_runtime_delivery(event_id: str, request: Request) -> JSONResponse:
        """Record a runtime-arbiter or robot callback for a field event."""
        try:
            body = await optional_json_body(request)
            trust = runtime_callback_trust(
                body,
                secret=runtime_callback_secret,
                max_age_s=runtime_callback_max_age_s,
            )
            if not trust.get("trusted"):
                return mission_json(
                    {
                        "recorded": False,
                        "reason": trust.get("reason") or "runtime_callback_not_trusted",
                        "runtime_callback_trust": trust,
                    },
                    status_code=403,
                )
            delivery = runtime_callback_delivery_body(body, trust=trust)
            result = await dispatch_field_operations(
                "record_runtime_delivery_payload",
                event_id,
                delivery,
            )
            status_code = 200 if result.get("recorded") else 422
            if result.get("reason") == "event_not_found":
                status_code = 404
            payload = FieldRuntimeDeliveryResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field runtime-delivery endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/devices",
        response_model=FieldDeviceStatusResponse,
        response_model_exclude_none=True,
    )
    async def field_devices() -> JSONResponse:
        """Return registered and observed field-device trust/online status."""
        try:
            result = await dispatch_field_operations("device_status_payload")
            payload = FieldDeviceStatusResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field devices endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/device-onboarding",
        response_model=FieldDeviceOnboardingResponse,
        response_model_exclude_none=True,
    )
    async def field_device_onboarding() -> JSONResponse:
        """Return delivery readiness evidence for real field-device onboarding."""
        try:
            result = await dispatch_field_operations("device_onboarding_payload")
            payload = FieldDeviceOnboardingResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field device onboarding endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/ingest",
        response_model=FieldIngestHelpResponse,
        response_model_exclude_none=True,
    )
    async def field_ingest_help() -> JSONResponse:
        """Return examples for raw camera/sensor/robot event ingestion."""
        try:
            result = await dispatch_field_operations("ingest_help_payload")
            payload = FieldIngestHelpResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field ingest help endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/field/ingest",
        response_model=FieldEventTriggerResponse,
        response_model_exclude_none=True,
    )
    async def field_ingest(request: Request) -> JSONResponse:
        """Normalize raw camera/sensor/robot/map payloads into field events."""
        try:
            body = await optional_json_body(request)
            result = await dispatch_field_operations("ingest_payload", body)
            result = await dispatch_field_voice_directive(result)
            result = await dispatch_field_runtime_policy(
                result,
                operator_id=str(body.get("operator_id") or "askme.operator"),
            )
            status_code = 200 if result.get("accepted", True) else 422
            payload = FieldEventTriggerResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field ingest endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.options("/api/field/events/{event_id}/runtime-delivery", include_in_schema=False)
    async def field_event_runtime_delivery_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/field/devices", include_in_schema=False)
    async def field_devices_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/device-onboarding", include_in_schema=False)
    async def field_device_onboarding_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/ingest", include_in_schema=False)
    async def field_ingest_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    return router
