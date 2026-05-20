"""Customer-project acceptance, evidence, and signoff FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.customer_projects import CustomerProjectAcceptanceClosureResponse
from askme.api.schemas.customer_projects import CustomerProjectAcceptanceReportResponse
from askme.api.schemas.customer_projects import (
    CustomerProjectAcceptanceReviewRegisterResponse,
)
from askme.api.schemas.customer_projects import (
    CustomerProjectCustomerSignoffRegisterResponse,
)
from askme.api.schemas.customer_projects import (
    CustomerProjectOnsiteEvidenceRegisterResponse,
)
from askme.api.schemas.customer_projects import CustomerProjectOnsiteEvidenceResponse
from askme.api.schemas.customer_projects import CustomerProjectSignoffResponse
from askme.pipeline.field.customer_projects import (
    customer_project_acceptance_closure,
    customer_project_acceptance_report,
    get_customer_project_profile,
    list_customer_project_customer_signoffs,
    list_customer_project_onsite_evidence,
    register_customer_project_acceptance_review,
    register_customer_project_customer_signoff,
    register_customer_project_onsite_evidence,
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


def register_customer_project_acceptance_routes(
    app: FastAPI,
    *,
    site_profile_root: PathProvider,
    project_read_auth: ProjectReadAuth,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_detail: ScopeItem,
    project_scope_forbidden: ProjectScopeForbidden,
    mission_json: MissionJson,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register customer-project acceptance, evidence, and signoff routes."""

    def _detail_scope_failure(
        detail: dict[str, Any],
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if detail.get("found") and not scope_allows(scope, scope_item_from_detail(detail)):
            return project_scope_forbidden()
        return None

    @app.get(
        "/api/field/customer-projects/{identifier}/acceptance-report",
        tags=["Field Operations"],
        response_model=CustomerProjectAcceptanceReportResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_acceptance_report(
        identifier: str,
        request: Request,
        check_env: bool = True,
    ) -> JSONResponse:
        """Return a customer-readable project acceptance report."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = customer_project_acceptance_report(root, identifier, check_env=check_env)
            scope_failure = _detail_scope_failure(result, scope)
            if scope_failure is not None:
                return scope_failure
            if result.get("found"):
                result = CustomerProjectAcceptanceReportResponse.model_validate(
                    result
                ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project acceptance report endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-projects/{identifier}/onsite-evidence",
        tags=["Field Operations"],
        response_model=CustomerProjectOnsiteEvidenceResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_onsite_evidence(
        identifier: str,
        request: Request,
        check_env: bool = True,
        include_readiness_auto: bool = True,
    ) -> JSONResponse:
        """Return onsite acceptance evidence receipts for one customer project."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = list_customer_project_onsite_evidence(
                root,
                identifier,
                check_env=check_env,
                include_readiness_auto=include_readiness_auto,
            )
            scope_failure = _detail_scope_failure(result, scope)
            if scope_failure is not None:
                return scope_failure
            if result.get("found"):
                result = CustomerProjectOnsiteEvidenceResponse.model_validate(
                    result
                ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project onsite evidence endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/onsite-evidence",
        tags=["Field Operations"],
        response_model=CustomerProjectOnsiteEvidenceRegisterResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_onsite_evidence_register(
        identifier: str,
        request: Request,
    ) -> JSONResponse:
        """Register one onsite acceptance evidence receipt for a customer project."""
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
            evidence = body.get("evidence") if isinstance(body.get("evidence"), dict) else body
            result = register_customer_project_onsite_evidence(
                root,
                identifier,
                evidence,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result = CustomerProjectOnsiteEvidenceRegisterResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project onsite evidence register endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-projects/{identifier}/acceptance-closure",
        tags=["Field Operations"],
        response_model=CustomerProjectAcceptanceClosureResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_acceptance_closure(
        identifier: str,
        request: Request,
        check_env: bool = True,
    ) -> JSONResponse:
        """Return a customer-readable acceptance closure summary."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = customer_project_acceptance_closure(root, identifier, check_env=check_env)
            scope_failure = _detail_scope_failure(result, scope)
            if scope_failure is not None:
                return scope_failure
            if result.get("found"):
                result = CustomerProjectAcceptanceClosureResponse.model_validate(
                    result
                ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project acceptance closure endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/acceptance-review",
        tags=["Field Operations"],
        response_model=CustomerProjectAcceptanceReviewRegisterResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_acceptance_review(
        identifier: str,
        request: Request,
    ) -> JSONResponse:
        """Record a manual delivery-owner acceptance review decision."""
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
            review = body.get("review") if isinstance(body.get("review"), dict) else body
            result = register_customer_project_acceptance_review(
                root,
                identifier,
                review,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result = CustomerProjectAcceptanceReviewRegisterResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project acceptance review endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get(
        "/api/field/customer-projects/{identifier}/customer-signoff",
        tags=["Field Operations"],
        response_model=CustomerProjectSignoffResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_customer_signoff(identifier: str, request: Request) -> JSONResponse:
        """Return customer signoff records for one customer project."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            root = site_profile_root()
            result = list_customer_project_customer_signoffs(root, identifier)
            scope_failure = _detail_scope_failure(result, operator_project_scope(auth_body))
            if scope_failure is not None:
                return scope_failure
            if result.get("found"):
                result = CustomerProjectSignoffResponse.model_validate(result).model_dump(
                    mode="python"
                )
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project signoff endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/customer-signoff",
        tags=["Field Operations"],
        response_model=CustomerProjectCustomerSignoffRegisterResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_customer_signoff_register(
        identifier: str,
        request: Request,
    ) -> JSONResponse:
        """Record a customer signoff decision after internal delivery review."""
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
            signoff = body.get("signoff") if isinstance(body.get("signoff"), dict) else body
            result = register_customer_project_customer_signoff(
                root,
                identifier,
                signoff,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result = CustomerProjectCustomerSignoffRegisterResponse.model_validate(
                result
            ).model_dump(mode="python")
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "profile_not_found":
                status_code = 404
            if result.get("reason") == "project_not_ready_for_customer_signoff":
                status_code = 409
            return mission_json(result, status_code=status_code)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project signoff register endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/customer-projects/{identifier}/acceptance-report", include_in_schema=False)
    async def field_customer_project_acceptance_report_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/onsite-evidence", include_in_schema=False)
    async def field_customer_project_onsite_evidence_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/acceptance-closure", include_in_schema=False)
    async def field_customer_project_acceptance_closure_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/acceptance-review", include_in_schema=False)
    async def field_customer_project_acceptance_review_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/customer-signoff", include_in_schema=False)
    async def field_customer_project_customer_signoff_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, POST, OPTIONS")

__all__ = ["register_customer_project_acceptance_routes"]
