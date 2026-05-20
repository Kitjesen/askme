"""Customer-project template FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.customer_projects import CustomerProjectTemplateCatalogResponse
from askme.api.schemas.customer_projects import CustomerProjectTemplateHistoryResponse
from askme.api.schemas.customer_projects import (
    CustomerProjectTemplateReleaseNotesExportResponse,
)
from askme.api.schemas.customer_projects import (
    CustomerProjectTemplateReleaseNotesResponse,
)
from askme.api.schemas.customer_projects import (
    CustomerProjectTemplateReleaseRequestMutationResponse,
)
from askme.api.schemas.customer_projects import (
    CustomerProjectTemplateReleaseRequestsResponse,
)
from askme.api.schemas.customer_projects import CustomerProjectTemplateReleaseUpdateResponse
from askme.api.routes.field_template_scope import (
    scope_template_catalog as _scope_template_catalog,
)
from askme.api.routes.field_template_scope import (
    scope_template_release_requests_payload as _scope_template_release_requests_payload,
)
from askme.api.routes.field_template_scope import (
    template_visible_for_scope as _template_visible_for_scope,
)
from askme.api.routes.field_template_scope import (
    visible_template_ids as _visible_template_ids,
)
from askme.pipeline.field.customer_project_templates import (
    create_customer_project_template_release_request,
    customer_project_template_release_notes,
    export_customer_project_template_release_notes_bundle,
    list_customer_project_template_release_requests,
    list_customer_project_template_revisions,
    list_customer_project_templates,
    review_customer_project_template_release_request,
    update_customer_project_template_release,
)

MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
ProjectReadAuth = Callable[[Request], tuple[JSONResponse | None, dict[str, Any]]]
OperatorProjectScope = Callable[[dict[str, Any]], dict[str, list[str]]]
ProjectScopeForbidden = Callable[[], JSONResponse]
TemplateRoot = Callable[[], Path]


def register_customer_project_template_routes(
    app: FastAPI,
    *,
    template_root: TemplateRoot,
    project_read_auth: ProjectReadAuth,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_project_scope: OperatorProjectScope,
    project_scope_forbidden: ProjectScopeForbidden,
    mission_json: MissionJson,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register customer-project template catalog and release-governance routes."""

    def _template_scope_failure(
        template_id: str,
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        found, allowed = _template_visible_for_scope(template_root(), template_id, scope)
        if found and not allowed:
            return project_scope_forbidden()
        return None

    def _release_request_scope_failure(
        request_id: str,
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if not any(scope.values()):
            return None
        requests = list_customer_project_template_release_requests(
            template_root(),
            limit=10000,
        ).get("requests", [])
        matched = next(
            (
                item
                for item in requests
                if isinstance(item, dict) and str(item.get("request_id") or "") == str(request_id or "")
            ),
            None,
        )
        if matched is None:
            return None
        return _template_scope_failure(str(matched.get("template_id") or ""), scope)

    @app.get(
        "/api/field/customer-project-templates",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateCatalogResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_templates(
        request: Request,
        tenant_id: str = "",
        delivery_namespace: str = "",
        industry: str = "",
        publish_status: str = "",
        product_status: str = "",
        template_id: str = "",
        release_channel: str = "",
        owner: str = "",
    ) -> JSONResponse:
        """Return reusable industry templates for solution-provider delivery."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            result = list_customer_project_templates(
                template_root(),
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                industry=industry,
                publish_status=publish_status,
                product_status=product_status,
                template_id=template_id,
                release_channel=release_channel,
                owner=owner,
            )
            result = _scope_template_catalog(result, scope)
            response = CustomerProjectTemplateCatalogResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field customer project templates endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-project-templates/{template_id}/history",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateHistoryResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_history(
        template_id: str,
        request: Request,
        limit: int = 20,
    ) -> JSONResponse:
        """Return release-governance history for one industry template."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            result = list_customer_project_template_revisions(
                template_root(),
                template_id,
                limit=limit,
            )
            if result.get("found"):
                _template_found, template_allowed = _template_visible_for_scope(
                    template_root(),
                    template_id,
                    scope,
                )
                if not template_allowed:
                    return project_scope_forbidden()
            result = CustomerProjectTemplateHistoryResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project template history endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-project-template-release-requests",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateReleaseRequestsResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_release_requests(
        request: Request,
        template_id: str = "",
        status: str = "",
        limit: int = 50,
    ) -> JSONResponse:
        """Return reusable-template release requests for product-owner review."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            result = list_customer_project_template_release_requests(
                template_root(),
                template_id=template_id,
                status=status,
                limit=limit,
            )
            result = _scope_template_release_requests_payload(template_root(), result, scope)
            result = CustomerProjectTemplateReleaseRequestsResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project template release request list failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-project-template-release-notes",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateReleaseNotesResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_release_notes(
        request: Request,
        limit: int = 50,
    ) -> JSONResponse:
        """Return approved customer-facing reusable-template release notes."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            visible_template_ids = _visible_template_ids(template_root(), scope)
            result = customer_project_template_release_notes(
                template_root(),
                limit=limit,
                template_ids=visible_template_ids,
            )
            if visible_template_ids is not None:
                result.setdefault("summary", {})["scope_filtered"] = True
            result = CustomerProjectTemplateReleaseNotesResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project template release notes failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-project-template-release-notes/export",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateReleaseNotesExportResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_release_notes_export(request: Request) -> JSONResponse:
        """Return a portable proposal/handoff bundle for approved template release notes."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:read")
            if failure is not None:
                return failure
            scope = operator_project_scope(body)
            visible_template_ids = _visible_template_ids(template_root(), scope)
            result = export_customer_project_template_release_notes_bundle(
                template_root(),
                customer_context=body.get("customer_context") if isinstance(body.get("customer_context"), dict) else body,
                limit=int(body.get("limit") or 50),
                template_ids=visible_template_ids,
            )
            if visible_template_ids is not None:
                result.setdefault("summary", {})["scope_filtered"] = True
            result = CustomerProjectTemplateReleaseNotesExportResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project template release notes export failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-project-templates/{template_id}/release-requests",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateReleaseRequestMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_release_request_create(
        template_id: str,
        request: Request,
    ) -> JSONResponse:
        """Create a pending reusable-template release request."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "template:release:write")
            if failure is not None:
                return failure
            scope_failure = _template_scope_failure(template_id, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            release = body.get("release") if isinstance(body.get("release"), dict) else body
            result = create_customer_project_template_release_request(
                template_root(),
                template_id,
                release,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or release.get("reason") or ""),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "template_not_found":
                status_code = 404
            result = CustomerProjectTemplateReleaseRequestMutationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project template release request create failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-project-template-release-requests/{request_id}/review",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateReleaseRequestMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_release_request_review(
        request_id: str,
        request: Request,
    ) -> JSONResponse:
        """Approve or reject a pending reusable-template release request."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "template:release:approve")
            if failure is not None:
                return failure
            scope_failure = _release_request_scope_failure(request_id, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            result = review_customer_project_template_release_request(
                template_root(),
                request_id,
                decision=str(body.get("decision") or ""),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "release_request_not_found":
                status_code = 404
            if result.get("reason") in {
                "release_request_not_pending",
                "release_request_requires_second_approver",
                "template_changed_since_request",
            }:
                status_code = 409
            result = CustomerProjectTemplateReleaseRequestMutationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project template release request review failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-project-templates/{template_id}/release",
        tags=["Field Operations"],
        response_model=CustomerProjectTemplateReleaseUpdateResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_template_release(
        template_id: str,
        request: Request,
    ) -> JSONResponse:
        """Promote, demote, or block a reusable industry template package."""
        try:
            body = await optional_json_body(request)
            release = body.get("release") if isinstance(body.get("release"), dict) else body
            publish_status = str(release.get("publish_status") or "").strip()
            effective_publish_status = publish_status
            if not effective_publish_status:
                current_template = list_customer_project_template_revisions(
                    template_root(),
                    template_id,
                    limit=0,
                )
                current_package = current_template.get("template_package")
                if isinstance(current_package, dict):
                    effective_publish_status = str(
                        current_package.get("publish_status") or ""
                    ).strip()
            required_permission = (
                "template:release:approve"
                if effective_publish_status == "published"
                else "template:release:write"
            )
            failure = authorize(request, body, required_permission)
            if failure is not None:
                return failure
            scope_failure = _template_scope_failure(template_id, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            if effective_publish_status == "published":
                result = CustomerProjectTemplateReleaseUpdateResponse.model_validate(
                    {
                        "accepted": False,
                        "reason": "published_release_requires_approval_request",
                        "template_id": template_id,
                        "next_step": (
                            "Create /release-requests first, then approve it with a second product owner."
                        ),
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=409,
                )
            result = update_customer_project_template_release(
                template_root(),
                template_id,
                release,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or release.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "template_not_found":
                status_code = 404
            if result.get("reason") == "published_release_requires_approval_request":
                status_code = 409
            result = CustomerProjectTemplateReleaseUpdateResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project template release endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/customer-project-templates", include_in_schema=False)
    async def field_customer_project_templates_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-project-templates/{template_id}/history", include_in_schema=False)
    async def field_customer_project_template_history_cors(template_id: str) -> Response:
        _ = template_id
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-project-template-release-requests", include_in_schema=False)
    async def field_customer_project_template_release_requests_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-project-template-release-notes", include_in_schema=False)
    async def field_customer_project_template_release_notes_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-project-template-release-notes/export", include_in_schema=False)
    async def field_customer_project_template_release_notes_export_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-project-templates/{template_id}/release-requests", include_in_schema=False)
    async def field_customer_project_template_release_request_create_cors(template_id: str) -> Response:
        _ = template_id
        return cors_options_response("POST, OPTIONS")

    @app.options(
        "/api/field/customer-project-template-release-requests/{request_id}/review",
        include_in_schema=False,
    )
    async def field_customer_project_template_release_request_review_cors(request_id: str) -> Response:
        _ = request_id
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-project-templates/{template_id}/release", include_in_schema=False)
    async def field_customer_project_template_release_cors(template_id: str) -> Response:
        _ = template_id
        return cors_options_response("POST, OPTIONS")


__all__ = ["register_customer_project_template_routes"]
