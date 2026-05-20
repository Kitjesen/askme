"""Customer-project profile and managed-object FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.customer_projects import (
    CustomerProjectArchiveResponse,
    CustomerProjectDetailResponse,
    CustomerProjectHistoryResponse,
    CustomerProjectManagedObjectMutationResponse,
    CustomerProjectMutationResponse,
    CustomerProjectRollbackResponse,
)
from askme.pipeline.field.customer_projects import (
    archive_customer_project_profile,
    delete_managed_object,
    get_customer_project_profile,
    list_customer_project_revisions,
    rollback_customer_project_profile,
    upsert_customer_project_profile,
    upsert_managed_object,
)

MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
ProjectReadAuth = Callable[[Request], tuple[JSONResponse | None, dict[str, Any]]]
OperatorProjectScope = Callable[[dict[str, Any]], dict[str, list[str]]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItem = Callable[[dict[str, Any]], dict[str, Any]]
ProjectScopeForbidden = Callable[[], JSONResponse]
PathProvider = Callable[[], Path]


def register_customer_project_profile_routes(
    app: FastAPI,
    *,
    site_profile_root: PathProvider,
    project_read_auth: ProjectReadAuth,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_detail: ScopeItem,
    scope_item_from_profile: ScopeItem,
    project_scope_forbidden: ProjectScopeForbidden,
    mission_json: MissionJson,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register customer-project profile, history, and managed-object routes."""

    def _detail_scope_failure(
        detail: dict[str, Any],
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if detail.get("found") and not scope_allows(scope, scope_item_from_detail(detail)):
            return project_scope_forbidden()
        return None

    @app.get(
        "/api/field/customer-projects/{identifier}",
        tags=["Field Operations"],
        response_model=CustomerProjectDetailResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_detail(
        identifier: str,
        request: Request,
        check_env: bool = False,
    ) -> JSONResponse:
        """Return one customer project profile with managed-object bindings."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = get_customer_project_profile(root, identifier, check_env=check_env)
            scope_failure = _detail_scope_failure(result, scope)
            if scope_failure is not None:
                return scope_failure
            if result.get("found"):
                result = CustomerProjectDetailResponse.model_validate(result).model_dump(
                    mode="python"
                )
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project detail endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects",
        tags=["Field Operations"],
        response_model=CustomerProjectMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_upsert(request: Request) -> JSONResponse:
        """Create or update a customer project profile from an explicit payload."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root()
            profile = body.get("profile") if isinstance(body.get("profile"), dict) else body
            scope = operator_project_scope(body)
            if not scope_allows(scope, scope_item_from_profile(profile)):
                return project_scope_forbidden()
            result = upsert_customer_project_profile(
                root,
                profile,
                overwrite=bool(body.get("overwrite", True)),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result = CustomerProjectMutationResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project upsert endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/managed-objects/{object_id}",
        tags=["Field Operations"],
        response_model=CustomerProjectManagedObjectMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_object_upsert(
        identifier: str,
        object_id: str,
        request: Request,
    ) -> JSONResponse:
        """Create or update one managed object in a customer project profile."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root()
            detail = get_customer_project_profile(root, identifier)
            scope_failure = _detail_scope_failure(detail, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            payload = body.get("managed_object") if isinstance(body.get("managed_object"), dict) else body
            result = upsert_managed_object(
                root,
                identifier,
                object_id,
                payload,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result = CustomerProjectManagedObjectMutationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field managed object upsert endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.delete(
        "/api/field/customer-projects/{identifier}/managed-objects/{object_id}",
        tags=["Field Operations"],
        response_model=CustomerProjectManagedObjectMutationResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_object_delete(
        identifier: str,
        object_id: str,
        request: Request,
    ) -> JSONResponse:
        """Remove one managed object from a customer project profile."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root()
            detail = get_customer_project_profile(root, identifier)
            scope_failure = _detail_scope_failure(detail, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            result = delete_managed_object(
                root,
                identifier,
                object_id,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result = CustomerProjectManagedObjectMutationResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field managed object delete endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-projects/{identifier}/history",
        tags=["Field Operations"],
        response_model=CustomerProjectHistoryResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_history(
        identifier: str,
        request: Request,
        limit: int = 20,
    ) -> JSONResponse:
        """Return saved customer project profile revisions for rollback review."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            detail = get_customer_project_profile(root, identifier)
            scope_failure = _detail_scope_failure(detail, scope)
            if scope_failure is not None:
                return scope_failure
            result = list_customer_project_revisions(root, identifier, limit=limit)
            result = CustomerProjectHistoryResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project history endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/rollback",
        tags=["Field Operations"],
        response_model=CustomerProjectRollbackResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_rollback(identifier: str, request: Request) -> JSONResponse:
        """Restore a customer project profile from a saved revision."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root()
            detail = get_customer_project_profile(root, identifier)
            scope_failure = _detail_scope_failure(detail, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            result = rollback_customer_project_profile(
                root,
                identifier,
                str(body.get("revision_id") or ""),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            result = CustomerProjectRollbackResponse.model_validate(result).model_dump(
                mode="python"
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") in {"profile_not_found", "revision_not_found"}:
                status_code = 404
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project rollback endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/archive",
        tags=["Field Operations"],
        response_model=CustomerProjectArchiveResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_archive(identifier: str, request: Request) -> JSONResponse:
        """Archive a customer project profile without permanent deletion."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root()
            detail = get_customer_project_profile(root, identifier)
            scope_failure = _detail_scope_failure(detail, operator_project_scope(body))
            if scope_failure is not None:
                return scope_failure
            result = archive_customer_project_profile(root, identifier)
            result = CustomerProjectArchiveResponse.model_validate(result).model_dump(
                mode="python"
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 404)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project archive endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/customer-projects/{identifier}", include_in_schema=False)
    async def field_customer_project_detail_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, POST, OPTIONS")

    @app.options(
        "/api/field/customer-projects/{identifier}/managed-objects/{object_id}",
        include_in_schema=False,
    )
    async def field_customer_project_object_cors(identifier: str, object_id: str) -> Response:
        _ = identifier, object_id
        return cors_options_response("POST, DELETE, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/history", include_in_schema=False)
    async def field_customer_project_history_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/rollback", include_in_schema=False)
    async def field_customer_project_rollback_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/archive", include_in_schema=False)
    async def field_customer_project_archive_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("POST, OPTIONS")


__all__ = ["register_customer_project_profile_routes"]
