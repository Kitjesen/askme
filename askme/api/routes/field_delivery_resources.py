"""Delivery-resource FastAPI routes for field operations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.delivery_resources import (
    DeliveryResourceGovernanceEscalationResponse,
    DeliveryResourceGovernanceMutationResponse,
    DeliveryResourceGovernanceRequestsResponse,
    DeliveryResourceHistoryResponse,
    DeliveryResourceMutationResponse,
    DeliveryResourceRegistryResponse,
    DeliveryResourceRollbackResponse,
)
from askme.pipeline.field.delivery_resources import (
    create_delivery_resource_governance_request,
    disable_delivery_resource,
    escalate_overdue_delivery_resource_governance_requests,
    list_delivery_resource_governance_requests,
    list_delivery_resource_registry,
    list_delivery_resource_revisions,
    review_delivery_resource_governance_request,
    rollback_delivery_resource_registry,
    upsert_delivery_resource,
)

MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
ProjectReadAuth = Callable[[Request], tuple[JSONResponse | None, dict[str, Any]]]
OperatorProjectScope = Callable[[dict[str, Any]], dict[str, list[str]]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItemFromResource = Callable[[dict[str, Any]], dict[str, Any]]
ResourceHasExplicitScope = Callable[[dict[str, Any]], bool]
ApplySingleScopeDefaults = Callable[[dict[str, Any], dict[str, list[str]]], None]
ProjectScopeForbidden = Callable[[], JSONResponse]
DeliveryResourceRoot = Callable[[], Path]
NotificationDelivery = Callable[[dict[str, Any]], dict[str, Any]]


def register_delivery_resource_routes(
    app: FastAPI,
    *,
    delivery_resource_root: DeliveryResourceRoot,
    project_read_auth: ProjectReadAuth,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_project_scope: OperatorProjectScope,
    apply_single_scope_defaults: ApplySingleScopeDefaults,
    scope_allows: ScopeAllows,
    scope_item_from_resource: ScopeItemFromResource,
    resource_has_explicit_scope: ResourceHasExplicitScope,
    project_scope_forbidden: ProjectScopeForbidden,
    notification_delivery: NotificationDelivery,
    mission_json: MissionJson,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register delivery-resource registry and governance endpoints."""

    def _scoped_registry(payload: dict[str, Any], scope: dict[str, list[str]]) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        resources = [
            resource
            for resource in payload.get("resources", [])
            if isinstance(resource, dict)
            and resource_has_explicit_scope(resource)
            and scope_allows(scope, scope_item_from_resource(resource))
        ]
        filtered = dict(payload)
        filtered["resources"] = resources
        summary = dict(payload.get("summary") or {})
        summary["resource_count"] = len(resources)
        summary["scope_filtered"] = True
        filtered["summary"] = summary
        return filtered

    @app.get(
        "/api/field/delivery-resource-registry",
        tags=["Field Operations"],
        response_model=DeliveryResourceRegistryResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_registry(request: Request) -> JSONResponse:
        """Return shared delivery resources that project objects can bind to."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            result = list_delivery_resource_registry(delivery_resource_root())
            result = _scoped_registry(
                result,
                operator_project_scope(auth_body),
            )
            result = DeliveryResourceRegistryResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource registry endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-registry",
        tags=["Field Operations"],
        response_model=DeliveryResourceMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_register(request: Request) -> JSONResponse:
        """Register one shared model, protocol, skill package, or acceptance resource."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:write")
            if failure is not None:
                return failure
            resource = body.get("resource") if isinstance(body.get("resource"), dict) else body
            metadata = dict(resource)
            resource_type = str(metadata.pop("resource_type", "") or body.get("resource_type") or "")
            resource_id = str(metadata.pop("resource_id", "") or body.get("resource_id") or "")
            scope = operator_project_scope(body)
            apply_single_scope_defaults(metadata, scope)
            if any(scope.values()) and not resource_has_explicit_scope(metadata):
                result = DeliveryResourceMutationResponse.model_validate(
                    {
                        "accepted": False,
                        "reason": "resource_scope_required",
                        "message": (
                            "Scoped operators must register resources with tenant, namespace, "
                            "customer, project, or site scope."
                        ),
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=403,
                )
            if not scope_allows(scope, scope_item_from_resource(metadata)):
                return project_scope_forbidden()
            result = upsert_delivery_resource(
                delivery_resource_root(),
                resource_type,
                resource_id,
                metadata,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                overwrite=bool(body.get("overwrite", True)),
            )
            result = DeliveryResourceMutationResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field delivery resource register endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/delivery-resource-registry/history",
        tags=["Field Operations"],
        response_model=DeliveryResourceHistoryResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_history(request: Request, limit: int = 20) -> JSONResponse:
        """Return shared delivery-resource registry revisions for audit review."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            if any(operator_project_scope(auth_body).values()):
                return mission_json(
                    {
                        "error": "resource registry history requires unrestricted operator scope",
                        "reason": "resource_registry_history_requires_unrestricted_operator",
                    },
                    status_code=403,
                )
            result = list_delivery_resource_revisions(
                delivery_resource_root(),
                limit=limit,
            )
            result = DeliveryResourceHistoryResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource history endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable",
        tags=["Field Operations"],
        response_model=DeliveryResourceMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_disable(
        resource_type: str,
        resource_id: str,
        request: Request,
    ) -> JSONResponse:
        """Disable one shared resource so customer-project bindings stop passing."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:approve")
            if failure is not None:
                return failure
            scope = operator_project_scope(body)
            registry = list_delivery_resource_registry(delivery_resource_root())
            resource = _find_delivery_resource(registry, resource_type, resource_id)
            if resource is not None:
                if any(scope.values()) and not resource_has_explicit_scope(resource):
                    result = DeliveryResourceMutationResponse.model_validate(
                        {
                            "accepted": False,
                            "reason": "resource_scope_required",
                            "message": "Scoped operators cannot mutate global shared resources.",
                        }
                    ).model_dump(mode="python")
                    return mission_json(
                        result,
                        status_code=403,
                    )
                if not scope_allows(scope, scope_item_from_resource(resource)):
                    return project_scope_forbidden()
            result = disable_delivery_resource(
                delivery_resource_root(),
                resource_type,
                resource_id,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "resource_not_found":
                status_code = 404
            result = DeliveryResourceMutationResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field delivery resource disable endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-registry/rollback",
        tags=["Field Operations"],
        response_model=DeliveryResourceRollbackResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_rollback(request: Request) -> JSONResponse:
        """Rollback the shared delivery-resource registry to a previous revision."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:approve")
            if failure is not None:
                return failure
            if any(operator_project_scope(body).values()):
                result = DeliveryResourceRollbackResponse.model_validate(
                    {
                        "accepted": False,
                        "reason": "resource_registry_rollback_requires_unrestricted_operator",
                        "message": "Registry rollback can affect multiple customers and requires unrestricted scope.",
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=403,
                )
            result = rollback_delivery_resource_registry(
                delivery_resource_root(),
                str(body.get("revision_id") or ""),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "revision_not_found":
                status_code = 404
            result = DeliveryResourceRollbackResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field delivery resource rollback endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/delivery-resource-governance-requests",
        tags=["Field Operations"],
        response_model=DeliveryResourceGovernanceRequestsResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_governance_requests(
        request: Request,
        status: str = "",
        action: str = "",
        limit: int = 50,
        overdue_only: bool = False,
    ) -> JSONResponse:
        """Return pending and reviewed shared-resource governance requests."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            if any(operator_project_scope(auth_body).values()):
                return mission_json(
                    {
                        "error": "resource governance requests require unrestricted operator scope",
                        "reason": "resource_governance_requests_require_unrestricted_operator",
                    },
                    status_code=403,
                )
            result = list_delivery_resource_governance_requests(
                delivery_resource_root(),
                status=status,
                action=action,
                limit=limit,
                overdue_only=overdue_only,
            )
            result = DeliveryResourceGovernanceRequestsResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource governance request list endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-governance-requests",
        tags=["Field Operations"],
        response_model=DeliveryResourceGovernanceMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_governance_request_create(
        request: Request,
    ) -> JSONResponse:
        """Create a high-risk shared-resource governance request for two-person review."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:write")
            if failure is not None:
                return failure
            operation = body.get("operation") if isinstance(body.get("operation"), dict) else dict(body)
            action = str(body.get("action") or operation.get("action") or "")
            scope = operator_project_scope(body)
            if action == "rollback_registry" and any(scope.values()):
                result = DeliveryResourceGovernanceMutationResponse.model_validate(
                    {
                        "accepted": False,
                        "reason": "resource_registry_rollback_requires_unrestricted_operator",
                        "message": "Registry rollback can affect multiple customers and requires unrestricted scope.",
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=403,
                )
            if action == "disable_resource" and any(scope.values()):
                registry = list_delivery_resource_registry(delivery_resource_root())
                resource_type = str(operation.get("resource_type") or "")
                resource_id = str(operation.get("resource_id") or "")
                resource = _find_delivery_resource(registry, resource_type, resource_id)
                if resource is not None:
                    if not resource_has_explicit_scope(resource):
                        result = DeliveryResourceGovernanceMutationResponse.model_validate(
                            {
                                "accepted": False,
                                "reason": "resource_scope_required",
                                "message": "Scoped operators cannot request global shared resource mutation.",
                            }
                        ).model_dump(mode="python")
                        return mission_json(
                            result,
                            status_code=403,
                        )
                    if not scope_allows(scope, scope_item_from_resource(resource)):
                        return project_scope_forbidden()
            result = create_delivery_resource_governance_request(
                delivery_resource_root(),
                action,
                operation,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                sla_target_s=body.get("sla_target_s") or operation.get("sla_target_s"),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "resource_not_found":
                status_code = 404
            result = DeliveryResourceGovernanceMutationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field delivery resource governance request create endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-governance-requests/{request_id}/review",
        tags=["Field Operations"],
        response_model=DeliveryResourceGovernanceMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_governance_request_review(
        request_id: str,
        request: Request,
    ) -> JSONResponse:
        """Approve or reject a pending shared-resource governance request."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:approve")
            if failure is not None:
                return failure
            if any(operator_project_scope(body).values()):
                result = DeliveryResourceGovernanceMutationResponse.model_validate(
                    {
                        "accepted": False,
                        "reason": "resource_governance_review_requires_unrestricted_operator",
                        "message": "Resource governance approval can affect multiple customers.",
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=403,
                )
            result = review_delivery_resource_governance_request(
                delivery_resource_root(),
                request_id,
                decision=str(body.get("decision") or ""),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "resource_governance_request_not_found":
                status_code = 404
            if result.get("reason") in {
                "resource_governance_request_not_pending",
                "resource_governance_request_requires_second_approver",
            }:
                status_code = 409
            result = DeliveryResourceGovernanceMutationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field delivery resource governance request review endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-governance-requests/escalate-overdue",
        tags=["Field Operations"],
        response_model=DeliveryResourceGovernanceEscalationResponse,
        response_model_exclude_none=True,
    )
    async def field_delivery_resource_governance_escalate_overdue(
        request: Request,
    ) -> JSONResponse:
        """Escalate overdue shared-resource governance requests to delivery owners."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:approve")
            if failure is not None:
                return failure
            if any(operator_project_scope(body).values()):
                result = DeliveryResourceGovernanceEscalationResponse.model_validate(
                    {
                        "accepted": False,
                        "reason": "resource_governance_escalation_requires_unrestricted_operator",
                        "message": "Resource governance escalation can affect multiple customers.",
                    }
                ).model_dump(mode="python")
                return mission_json(result, status_code=403)
            result = escalate_overdue_delivery_resource_governance_requests(
                delivery_resource_root(),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                limit=int(body.get("limit") or 50),
                dry_run=bool(body.get("dry_run")),
                notification_delivery=notification_delivery,
            )
            result = DeliveryResourceGovernanceEscalationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field delivery resource governance overdue escalation endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/delivery-resource-registry", include_in_schema=False)
    async def field_delivery_resource_registry_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/delivery-resource-registry/history", include_in_schema=False)
    async def field_delivery_resource_history_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options(
        "/api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable",
        include_in_schema=False,
    )
    async def field_delivery_resource_disable_cors(resource_type: str, resource_id: str) -> Response:
        _ = (resource_type, resource_id)
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/delivery-resource-registry/rollback", include_in_schema=False)
    async def field_delivery_resource_rollback_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/delivery-resource-governance-requests", include_in_schema=False)
    async def field_delivery_resource_governance_requests_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @app.options(
        "/api/field/delivery-resource-governance-requests/escalate-overdue",
        include_in_schema=False,
    )
    async def field_delivery_resource_governance_escalate_overdue_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options(
        "/api/field/delivery-resource-governance-requests/{request_id}/review",
        include_in_schema=False,
    )
    async def field_delivery_resource_governance_request_review_cors(request_id: str) -> Response:
        _ = request_id
        return cors_options_response("POST, OPTIONS")


def _find_delivery_resource(
    registry: dict[str, Any],
    resource_type: str,
    resource_id: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in registry.get("resources", [])
            if isinstance(item, dict)
            and str(item.get("resource_type") or "") == resource_type
            and str(item.get("resource_id") or "") == resource_id
        ),
        None,
    )


__all__ = ["register_delivery_resource_routes"]
