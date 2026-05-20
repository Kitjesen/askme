"""Customer-facing field event routes."""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from askme.api.schemas.field_events import (
    FieldEventActionResponse,
    FieldEventDetailApiResponse,
    FieldEventListApiResponse,
    FieldEventReportResponse,
    FieldEventTriggerResponse,
    FieldScenarioAcceptanceResponse,
    FieldScenarioCatalogResponse,
)
from askme.api.services.field_evidence_access import (
    field_evidence_scope_allows,
    resolve_field_evidence_path,
)

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
ProjectReadAuth = Callable[[Request], tuple[JSONResponse | None, dict[str, Any]]]
OperatorProjectScope = Callable[[dict[str, Any]], dict[str, list[str]]]
ScopedQueryValue = Callable[[str, dict[str, list[str]], str], tuple[bool, str]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItem = Callable[[dict[str, Any]], dict[str, Any]]
HasExplicitProjectScope = Callable[[dict[str, Any]], bool]
ApplySingleScopeDefaults = Callable[[dict[str, Any], dict[str, list[str]]], None]
ProjectScopeForbidden = Callable[[], JSONResponse]
EventScopeFailure = Callable[[str, dict[str, list[str]]], Awaitable[JSONResponse | None]]
ManualTriggerBody = Callable[[Request, dict[str, Any]], dict[str, Any]]
LooksLikeDeviceIngest = Callable[[dict[str, Any]], bool]
FieldResultHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
FieldRuntimePolicy = Callable[..., Awaitable[dict[str, Any]]]


def register_field_event_routes(
    app: FastAPI,
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    project_read_auth: ProjectReadAuth,
    operator_project_scope: OperatorProjectScope,
    scoped_query_value: ScopedQueryValue,
    scope_allows: ScopeAllows,
    scope_item_from_event_detail: ScopeItem,
    scope_item_from_event_payload: ScopeItem,
    has_explicit_project_scope: HasExplicitProjectScope,
    apply_single_scope_defaults: ApplySingleScopeDefaults,
    project_scope_forbidden: ProjectScopeForbidden,
    field_event_scope_failure: EventScopeFailure,
    field_manual_trigger_body: ManualTriggerBody,
    looks_like_device_ingest_without_scenario: LooksLikeDeviceIngest,
    dispatch_field_voice_directive: FieldResultHook,
    dispatch_field_runtime_policy: FieldRuntimePolicy,
    cors_headers: dict[str, str],
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register customer-facing event, evidence, and incident workflow routes."""

    app.include_router(
        create_field_event_router(
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            authorize=authorize,
            project_read_auth=project_read_auth,
            operator_project_scope=operator_project_scope,
            scoped_query_value=scoped_query_value,
            scope_allows=scope_allows,
            scope_item_from_event_detail=scope_item_from_event_detail,
            scope_item_from_event_payload=scope_item_from_event_payload,
            has_explicit_project_scope=has_explicit_project_scope,
            apply_single_scope_defaults=apply_single_scope_defaults,
            project_scope_forbidden=project_scope_forbidden,
            field_event_scope_failure=field_event_scope_failure,
            field_manual_trigger_body=field_manual_trigger_body,
            looks_like_device_ingest_without_scenario=looks_like_device_ingest_without_scenario,
            dispatch_field_voice_directive=dispatch_field_voice_directive,
            dispatch_field_runtime_policy=dispatch_field_runtime_policy,
            cors_headers=cors_headers,
            cors_options_response=cors_options_response,
            logger=logger,
        )
    )


def create_field_event_router(
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    project_read_auth: ProjectReadAuth,
    operator_project_scope: OperatorProjectScope,
    scoped_query_value: ScopedQueryValue,
    scope_allows: ScopeAllows,
    scope_item_from_event_detail: ScopeItem,
    scope_item_from_event_payload: ScopeItem,
    has_explicit_project_scope: HasExplicitProjectScope,
    apply_single_scope_defaults: ApplySingleScopeDefaults,
    project_scope_forbidden: ProjectScopeForbidden,
    field_event_scope_failure: EventScopeFailure,
    field_manual_trigger_body: ManualTriggerBody,
    looks_like_device_ingest_without_scenario: LooksLikeDeviceIngest,
    dispatch_field_voice_directive: FieldResultHook,
    dispatch_field_runtime_policy: FieldRuntimePolicy,
    cors_headers: dict[str, str],
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> APIRouter:
    """Create the customer-facing field event router without binding it to an app factory."""

    router = APIRouter(tags=["Field Operations"])

    @router.get(
        "/api/field/scenarios",
        response_model=FieldScenarioCatalogResponse,
        response_model_exclude_none=True,
    )
    async def field_scenarios() -> JSONResponse:
        """Return customer-visible field operation scenarios."""
        try:
            result = await dispatch_field_operations("scenarios_payload")
            payload = FieldScenarioCatalogResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field scenarios endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/scenario-acceptance",
        response_model=FieldScenarioAcceptanceResponse,
        response_model_exclude_none=True,
    )
    async def field_scenario_acceptance() -> JSONResponse:
        """Return customer-readable scenario acceptance coverage and boundaries."""
        try:
            result = await dispatch_field_operations("scenario_acceptance_matrix_payload")
            payload = FieldScenarioAcceptanceResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field scenario acceptance endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/events",
        response_model=FieldEventListApiResponse,
        response_model_exclude_none=True,
    )
    async def field_events(
        request: Request,
        limit: int = 50,
        status: str = "",
        notification_group: str = "",
        needs_attention: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        managed_object_id: str = "",
    ) -> JSONResponse:
        """Return recent field operation events."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            tenant_allowed, tenant_id = scoped_query_value(tenant_id, scope, "tenant_ids")
            namespace_allowed, delivery_namespace = scoped_query_value(
                delivery_namespace,
                scope,
                "delivery_namespaces",
            )
            customer_allowed, customer_id = scoped_query_value(customer_id, scope, "customer_ids")
            project_allowed, project_id = scoped_query_value(project_id, scope, "project_ids")
            site_allowed, site_id = scoped_query_value(site_id, scope, "site_ids")
            if not (
                tenant_allowed
                and namespace_allowed
                and customer_allowed
                and project_allowed
                and site_allowed
            ):
                return project_scope_forbidden()
            result = await dispatch_field_operations(
                "list_payload",
                limit=limit,
                status=status or None,
                notification_group=notification_group or None,
                needs_attention=needs_attention,
                tenant_id=tenant_id or None,
                delivery_namespace=delivery_namespace or None,
                customer_id=customer_id or None,
                project_id=project_id or None,
                site_id=site_id or None,
                managed_object_id=managed_object_id or None,
                project_scope=scope,
            )
            payload = FieldEventListApiResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field events endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/events/{event_id}",
        response_model=FieldEventDetailApiResponse,
        response_model_exclude_none=True,
    )
    async def field_event_detail(event_id: str, request: Request) -> JSONResponse:
        """Return one field operation event with workflow and evidence detail."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            result = await dispatch_field_operations("detail_payload", event_id)
            if result.get("found") and not scope_allows(scope, scope_item_from_event_detail(result)):
                return project_scope_forbidden()
            status_code = 200 if result.get("found") else 404
            payload = FieldEventDetailApiResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except Exception as exc:
            logger.error("Field event detail endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get("/api/field/evidence", response_model=None)
    async def field_evidence(request: Request, path: str, event_id: str = "") -> Response:
        """Serve a local field evidence artifact from approved evidence roots."""
        failure, auth_body = project_read_auth(request)
        if failure is not None:
            return failure
        resolved = resolve_field_evidence_path(path)
        if resolved is None:
            return mission_json({"error": "field evidence not found"}, status_code=404)
        scope = operator_project_scope(auth_body)
        if not await field_evidence_scope_allows(
            path,
            resolved,
            scope,
            dispatch_field_operations=dispatch_field_operations,
            scope_allows=scope_allows,
            scope_item_from_event_detail=scope_item_from_event_detail,
            event_id=event_id,
        ):
            return project_scope_forbidden()
        media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        return FileResponse(
            resolved,
            media_type=media_type,
            filename=resolved.name,
            headers={
                "Cache-Control": "private, max-age=60",
                **cors_headers,
            },
        )

    @router.post(
        "/api/field/events",
        response_model=FieldEventTriggerResponse,
        response_model_exclude_none=True,
    )
    async def field_event_trigger(request: Request) -> JSONResponse:
        """Evaluate a field event and dispatch alerts when rules pass."""
        try:
            body = await optional_json_body(request)
            if looks_like_device_ingest_without_scenario(body):
                return mission_json(
                    {
                        "accepted": False,
                        "status": "rejected",
                        "reason": "device_payload_must_use_field_ingest",
                        "message": "Device camera, sensor, and robot payloads must be submitted to /api/field/ingest.",
                    },
                    status_code=422,
                )
            failure = authorize(request, body, "field:event:create")
            if failure is not None:
                return failure
            body = field_manual_trigger_body(request, body)
            scope = operator_project_scope(body)
            if has_explicit_project_scope(body) and not scope_allows(
                scope,
                scope_item_from_event_payload(body),
            ):
                return project_scope_forbidden()
            apply_single_scope_defaults(body, scope)
            result = await dispatch_field_operations("trigger_payload", body)
            result = await dispatch_field_voice_directive(result)
            result = await dispatch_field_runtime_policy(
                result,
                operator_id=str(body.get("operator_id") or "dashboard.operator"),
            )
            result.setdefault(
                "trigger_contract",
                {
                    "admission_path": "field_events_manual",
                    "trigger_source": body.get("trigger_source"),
                    "operator_id": body.get("operator_id"),
                    "device_payload_endpoint": "/api/field/ingest",
                },
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
            logger.error("Field event trigger endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/field/events/{event_id}/close",
        response_model=FieldEventActionResponse,
        response_model_exclude_none=True,
    )
    async def field_event_close(event_id: str, request: Request) -> JSONResponse:
        """Close a field operation event with an operator note."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:close")
            if failure is not None:
                return failure
            scope_failure = await field_event_scope_failure(
                event_id,
                operator_project_scope(body),
            )
            if scope_failure is not None:
                return scope_failure
            result = await dispatch_field_operations("close_payload", event_id, body)
            status_code = 200 if result.get("closed") else 404
            if result.get("reason") in {
                "close_requires_supervisor_approval",
                "event_already_closed",
                "event_not_closable",
            }:
                status_code = 409
            if result.get("reason") in {"operator_not_authorized", "supervisor_not_authorized"}:
                status_code = 403
            payload = FieldEventActionResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field event close endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/field/events/{event_id}/request-close",
        response_model=FieldEventActionResponse,
        response_model_exclude_none=True,
    )
    async def field_event_request_close(event_id: str, request: Request) -> JSONResponse:
        """Request supervisor approval before closing a high-risk field event."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:request_close")
            if failure is not None:
                return failure
            scope_failure = await field_event_scope_failure(
                event_id,
                operator_project_scope(body),
            )
            if scope_failure is not None:
                return scope_failure
            result = await dispatch_field_operations("request_close_payload", event_id, body)
            status_code = 200 if result.get("requested") else 404
            if result.get("reason") in {"event_already_closed", "event_not_closable"}:
                status_code = 409
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            payload = FieldEventActionResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field event close request endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/field/events/{event_id}/acknowledge",
        response_model=FieldEventActionResponse,
        response_model_exclude_none=True,
    )
    async def field_event_acknowledge(event_id: str, request: Request) -> JSONResponse:
        """Acknowledge a field operation event without closing it."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:acknowledge")
            if failure is not None:
                return failure
            scope_failure = await field_event_scope_failure(
                event_id,
                operator_project_scope(body),
            )
            if scope_failure is not None:
                return scope_failure
            result = await dispatch_field_operations("acknowledge_payload", event_id, body)
            status_code = 200 if result.get("acknowledged") else 409
            if result.get("reason") == "event_not_found":
                status_code = 404
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            payload = FieldEventActionResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field event acknowledge endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/field/events/{event_id}/resend-notification",
        response_model=FieldEventActionResponse,
        response_model_exclude_none=True,
    )
    async def field_event_resend_notification(event_id: str, request: Request) -> JSONResponse:
        """Retry notification delivery for an open field operation event."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:acknowledge")
            if failure is not None:
                return failure
            scope_failure = await field_event_scope_failure(
                event_id,
                operator_project_scope(body),
            )
            if scope_failure is not None:
                return scope_failure
            result = await dispatch_field_operations(
                "resend_notification_payload",
                event_id,
                body,
            )
            status_code = 200 if result.get("resent") else 409
            if result.get("reason") == "event_not_found":
                status_code = 404
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            payload = FieldEventActionResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field event notification resend endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/events/{event_id}/report",
        response_model=FieldEventReportResponse,
        response_model_exclude_none=True,
    )
    async def field_event_report(event_id: str, request: Request) -> JSONResponse:
        """Return an auditable customer-facing field event report."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            result = await dispatch_field_operations("event_report_payload", event_id)
            if result.get("found") and not scope_allows(scope, scope_item_from_event_detail(result.get("report", {}))):
                return project_scope_forbidden()
            status_code = 200 if result.get("found") else 404
            payload = FieldEventReportResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except Exception as exc:
            logger.error("Field event report endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.options("/api/field/scenarios", include_in_schema=False)
    async def field_scenarios_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/scenario-acceptance", include_in_schema=False)
    async def field_scenario_acceptance_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/events", include_in_schema=False)
    async def field_events_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @router.options("/api/field/events/{event_id}", include_in_schema=False)
    async def field_event_detail_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/evidence", include_in_schema=False)
    async def field_evidence_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/events/{event_id}/acknowledge", include_in_schema=False)
    async def field_event_acknowledge_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/field/events/{event_id}/close", include_in_schema=False)
    async def field_event_close_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/field/events/{event_id}/request-close", include_in_schema=False)
    async def field_event_request_close_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/field/events/{event_id}/resend-notification", include_in_schema=False)
    async def field_event_resend_notification_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/field/events/{event_id}/report", include_in_schema=False)
    async def field_event_report_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("GET, OPTIONS")

    return router

__all__ = ["create_field_event_router", "register_field_event_routes"]
