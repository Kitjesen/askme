"""Admin and governance routes for field operations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.field_events import (
    FieldActionAuditIntegrityResponse,
    FieldCustomerProjectFromTemplateResponse,
    FieldNotificationPreflightResponse,
    FieldNotificationTestResponse,
    FieldReadinessResponse,
)
from askme.pipeline.field.customer_project_templates import (
    create_customer_project_from_template,
)

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
PathProvider = Callable[[], Path]
OperatorProjectScope = Callable[[dict[str, Any]], dict[str, list[str]]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItemFromCreateBody = Callable[[dict[str, Any]], dict[str, Any]]
ProjectScopeForbidden = Callable[[], JSONResponse]


def register_field_admin_routes(
    app: FastAPI,
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize,
    site_profile_root: PathProvider,
    template_root: PathProvider,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_create_body: ScopeItemFromCreateBody,
    project_scope_forbidden: ProjectScopeForbidden,
) -> None:
    """Register operator-governance routes while preserving legacy URLs."""

    app.include_router(
        create_field_admin_router(
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            logger=logger,
            authorize=authorize,
            site_profile_root=site_profile_root,
            template_root=template_root,
            operator_project_scope=operator_project_scope,
            scope_allows=scope_allows,
            scope_item_from_create_body=scope_item_from_create_body,
            project_scope_forbidden=project_scope_forbidden,
        )
    )


def create_field_admin_router(
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize,
    site_profile_root: PathProvider,
    template_root: PathProvider,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_create_body: ScopeItemFromCreateBody,
    project_scope_forbidden: ProjectScopeForbidden,
) -> APIRouter:
    """Create the field admin router without binding it to an app factory."""

    router = APIRouter(tags=["Field Operations"])

    @router.post(
        "/api/field/notification-test",
        response_model=FieldNotificationTestResponse,
        response_model_exclude_none=True,
    )
    async def field_notification_test(request: Request) -> JSONResponse:
        """Send a low-risk notification smoke test to a responder group."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:notification:test")
            if failure is not None:
                return failure
            result = await dispatch_field_operations("test_notification_payload", body)
            status_code = 200 if result.get("status") != "invalid_group" else 422
            payload = FieldNotificationTestResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field notification test endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/notification-preflight",
        response_model=FieldNotificationPreflightResponse,
        response_model_exclude_none=True,
    )
    async def field_notification_preflight(status_as_200: bool = False) -> JSONResponse:
        """Check whether real DingTalk responder notification credentials are configured."""
        try:
            result = await dispatch_field_operations("notification_preflight_payload")
            status_code = 200 if status_as_200 or result.get("ready") else 409
            payload = FieldNotificationPreflightResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except Exception as exc:
            logger.error("Field notification preflight endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/readiness",
        response_model=FieldReadinessResponse,
        response_model_exclude_none=True,
    )
    async def field_readiness() -> JSONResponse:
        """Return deployment readiness gates for field operations."""
        try:
            result = await dispatch_field_operations("readiness_payload")
            payload = FieldReadinessResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload)
        except Exception as exc:
            logger.error("Field readiness endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/audit/integrity",
        response_model=FieldActionAuditIntegrityResponse,
        response_model_exclude_none=True,
    )
    async def field_action_audit_integrity() -> JSONResponse:
        """Verify the append-only field action audit hash chain."""
        try:
            result = await dispatch_field_operations("action_audit_integrity_payload")
            status_code = 200
            if result.get("enabled") is not False and not result.get("valid"):
                status_code = 409
            payload = FieldActionAuditIntegrityResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=status_code)
        except Exception as exc:
            logger.error("Field action audit integrity endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.post(
        "/api/field/customer-projects/from-template",
        response_model=FieldCustomerProjectFromTemplateResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_from_template(request: Request) -> JSONResponse:
        """Create a customer project profile from an industry template."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            scope = operator_project_scope(body)
            if not scope_allows(scope, scope_item_from_create_body(body)):
                return project_scope_forbidden()
            result = create_customer_project_from_template(
                template_root=template_root(),
                profile_root=site_profile_root(),
                template_id=str(body.get("template_id") or ""),
                customer=body.get("customer") if isinstance(body.get("customer"), dict) else {},
                site=body.get("site") if isinstance(body.get("site"), dict) else {},
                overwrite=bool(body.get("overwrite")),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            payload = FieldCustomerProjectFromTemplateResponse.model_validate(result).model_dump(
                mode="python",
                exclude_none=True,
            )
            return mission_json(payload, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project template create endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.options("/api/field/notification-test", include_in_schema=False)
    async def field_notification_test_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/field/notification-preflight", include_in_schema=False)
    async def field_notification_preflight_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/readiness", include_in_schema=False)
    async def field_readiness_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/audit/integrity", include_in_schema=False)
    async def field_action_audit_integrity_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/customer-projects/from-template", include_in_schema=False)
    async def field_customer_project_from_template_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router
