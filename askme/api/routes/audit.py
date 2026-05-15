"""Audit and evidence FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.audit import AuditExportService, AuditQueryService, AuditReviewService
from askme.skills.audit import SkillAuditLog

ConfigProvider = Callable[[], dict[str, Any]]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
OperatorIdFromRequest = Callable[[Request, dict[str, Any]], str]

_NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
_CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}
_CORS_ALLOW_HEADERS = (
    "Content-Type, Authorization, X-Askme-Api-Key, "
    "X-Askme-Operator-Id, X-Operator-Id"
)


def register_audit_routes(
    app: FastAPI,
    *,
    config_provider: ConfigProvider,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_id_from_request: OperatorIdFromRequest,
    logger: logging.Logger,
) -> None:
    """Register skill and unified product audit routes."""

    def _operator_project_scope(auth_body: dict[str, Any]) -> dict[str, list[str]]:
        operator = (
            auth_body.get("operator_auth", {})
            if isinstance(auth_body.get("operator_auth"), dict)
            else {}
        ).get("operator", {})
        scope = operator.get("project_scope") if isinstance(operator, dict) else {}
        if not isinstance(scope, dict) or scope.get("unrestricted"):
            return {}
        return {
            "tenant_ids": _clean_scope_values(scope.get("tenant_ids")),
            "delivery_namespaces": _clean_scope_values(scope.get("delivery_namespaces")),
            "customer_ids": _clean_scope_values(scope.get("customer_ids")),
            "project_ids": _clean_scope_values(scope.get("project_ids")),
            "site_ids": _clean_scope_values(scope.get("site_ids")),
        }

    def _audit_scope_filters(
        requested: dict[str, str],
        scope: dict[str, list[str]],
    ) -> tuple[dict[str, str], JSONResponse | None]:
        values = {key: str(value or "").strip() for key, value in requested.items()}
        if not any(scope.values()):
            return values, None
        for value_key, scope_key in (
            ("tenant_id", "tenant_ids"),
            ("delivery_namespace", "delivery_namespaces"),
            ("customer_id", "customer_ids"),
            ("project_id", "project_ids"),
            ("site_id", "site_ids"),
        ):
            allowed = scope.get(scope_key) or []
            if not allowed or "*" in allowed:
                continue
            value = values.get(value_key, "")
            if value:
                if value not in allowed:
                    return values, _audit_project_scope_forbidden()
                continue
            if len(allowed) == 1:
                values[value_key] = allowed[0]
                continue
            return values, _audit_project_scope_required(scope_key, allowed)
        return values, None

    def _clean_scope_values(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    def _audit_project_scope_forbidden() -> JSONResponse:
        return JSONResponse(
            {
                "error": "operator not authorized for this customer project",
                "reason": "project_scope_not_allowed",
            },
            status_code=403,
            headers=_CORS_HEADERS,
        )

    def _audit_project_scope_required(scope_key: str, allowed: list[str]) -> JSONResponse:
        return JSONResponse(
            {
                "error": "audit scope filter required",
                "reason": "project_scope_filter_required",
                "scope_key": scope_key,
                "allowed": allowed,
            },
            status_code=422,
            headers=_CORS_HEADERS,
        )

    @app.options("/api/skill-audit", include_in_schema=False)
    async def skill_audit_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/audit/events", include_in_schema=False)
    async def audit_events_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/audit/reviews", include_in_schema=False)
    async def audit_reviews_cors() -> Response:
        return _cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/audit/export", include_in_schema=False)
    async def audit_export_cors() -> Response:
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/audit/exports", include_in_schema=False)
    async def audit_exports_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/audit/export/retry", include_in_schema=False)
    async def audit_export_retry_cors() -> Response:
        return _cors_options_response("GET, POST, OPTIONS")

    @app.get("/api/skill-audit", tags=["System"])
    async def skill_audit(limit: int = 50) -> JSONResponse:
        """Return recent skill execution audit records."""
        try:
            safe_limit = max(1, min(int(limit), 200))
            records = SkillAuditLog().recent(limit=safe_limit)
            return JSONResponse(
                {"records": records, "count": len(records), "limit": safe_limit},
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.error("Skill audit endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get("/api/audit/events", tags=["Governance"])
    async def audit_events(
        request: Request,
        limit: int = 100,
        source: str = "",
        actor_id: str = "",
        operator_id: str = "",
        action: str = "",
        outcome: str = "",
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        managed_object_id: str = "",
        q: str = "",
        since: str = "",
        until: str = "",
    ) -> JSONResponse:
        """Return a unified product audit timeline across field/runtime/skill records."""
        try:
            auth_body = {"operator_id": actor_id}
            denied = authorize(request, auth_body, "audit:read")
            if denied is not None:
                return denied
            scope_filters, scope_error = _audit_scope_filters(
                {
                    "tenant_id": tenant_id,
                    "delivery_namespace": delivery_namespace,
                    "customer_id": customer_id,
                    "project_id": project_id,
                    "site_id": site_id,
                    "managed_object_id": managed_object_id,
                },
                _operator_project_scope(auth_body),
            )
            if scope_error is not None:
                return scope_error
            payload = AuditQueryService(config_provider()).query(
                limit=limit,
                source=source,
                operator_id=operator_id,
                action=action,
                outcome=outcome,
                tenant_id=scope_filters.get("tenant_id", ""),
                delivery_namespace=scope_filters.get("delivery_namespace", ""),
                customer_id=scope_filters.get("customer_id", ""),
                project_id=scope_filters.get("project_id", ""),
                site_id=scope_filters.get("site_id", ""),
                managed_object_id=scope_filters.get("managed_object_id", ""),
                q=q,
                since=since,
                until=until,
            )
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Unified audit endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get("/api/audit/reviews", tags=["Governance"])
    async def audit_reviews(
        request: Request,
        actor_id: str = "",
        limit: int = 100,
    ) -> JSONResponse:
        """Return append-only unified audit review decisions."""
        try:
            denied = authorize(request, {"operator_id": actor_id}, "audit:read")
            if denied is not None:
                return denied
            payload = AuditReviewService(config_provider()).list(limit=limit)
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Unified audit review list failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.post("/api/audit/reviews", tags=["Governance"])
    async def audit_review_submit(request: Request) -> JSONResponse:
        """Append a supervisor review decision for one unified audit record."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "audit:review")
            if denied is not None:
                return denied
            reviewer_id = operator_id_from_request(request, body)
            record_id = str(body.get("record_id") or "")
            if not AuditQueryService(config_provider()).record_exists(record_id):
                return JSONResponse(
                    {
                        "ok": False,
                        "reason": "audit_record_not_found",
                        "record_id": record_id,
                    },
                    status_code=404,
                    headers=_NO_STORE_HEADERS,
                )
            payload = AuditReviewService(config_provider()).submit(
                record_id=record_id,
                reviewer_id=reviewer_id,
                decision=str(body.get("decision") or ""),
                note=str(body.get("note") or ""),
            )
            return JSONResponse(
                payload,
                status_code=200 if payload.get("ok") else 422,
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.error("Unified audit review submit failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get("/api/audit/export/retry", tags=["Governance"])
    async def audit_export_retry_status(
        request: Request,
        actor_id: str = "",
        limit: int = 50,
    ) -> JSONResponse:
        """Return pending unified audit export delivery retries."""
        try:
            denied = authorize(request, {"operator_id": actor_id}, "audit:export")
            if denied is not None:
                return denied
            payload = AuditExportService(config_provider()).retry_status(limit=limit)
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Unified audit export retry status failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get("/api/audit/exports", tags=["Governance"])
    async def audit_exports(
        request: Request,
        actor_id: str = "",
        limit: int = 20,
    ) -> JSONResponse:
        """Return recent unified audit export manifests."""
        try:
            denied = authorize(request, {"operator_id": actor_id}, "audit:export")
            if denied is not None:
                return denied
            payload = AuditExportService(config_provider()).list_exports(limit=limit)
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Unified audit export list failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.post("/api/audit/export/retry", tags=["Governance"])
    async def audit_export_retry_delivery(request: Request) -> JSONResponse:
        """Replay pending unified audit export deliveries."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "audit:export")
            if denied is not None:
                return denied
            payload = AuditExportService(config_provider()).retry_queued_deliveries(
                limit=int(body.get("limit") or 50),
            )
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Unified audit export retry delivery failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.post("/api/audit/export", tags=["Governance"])
    async def audit_export(request: Request) -> JSONResponse:
        """Create a signed unified audit export package and optionally deliver it."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "audit:export")
            if denied is not None:
                return denied
            actor_id = operator_id_from_request(request, body)
            scope_filters, scope_error = _audit_scope_filters(
                {
                    "tenant_id": str(body.get("tenant_id") or ""),
                    "delivery_namespace": str(body.get("delivery_namespace") or ""),
                    "customer_id": str(body.get("customer_id") or ""),
                    "project_id": str(body.get("project_id") or ""),
                    "site_id": str(body.get("site_id") or ""),
                    "managed_object_id": str(body.get("managed_object_id") or ""),
                },
                _operator_project_scope(body),
            )
            if scope_error is not None:
                return scope_error
            payload = AuditExportService(config_provider()).create_export(
                actor_id=actor_id,
                limit=int(body.get("limit") or 500),
                source=str(body.get("source") or ""),
                operator_id=str(body.get("filter_operator_id") or body.get("operator_filter") or ""),
                action=str(body.get("action") or ""),
                outcome=str(body.get("outcome") or ""),
                tenant_id=scope_filters.get("tenant_id", ""),
                delivery_namespace=scope_filters.get("delivery_namespace", ""),
                customer_id=scope_filters.get("customer_id", ""),
                project_id=scope_filters.get("project_id", ""),
                site_id=scope_filters.get("site_id", ""),
                managed_object_id=scope_filters.get("managed_object_id", ""),
                q=str(body.get("q") or ""),
                since=str(body.get("since") or body.get("from") or ""),
                until=str(body.get("until") or body.get("to") or ""),
                deliver=bool(body.get("deliver", False)),
                webhook_url=str(body.get("webhook_url") or ""),
            )
            return JSONResponse(
                payload,
                status_code=200 if payload.get("ok") else 400,
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.error("Unified audit export endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)


def _cors_options_response(methods: str) -> Response:
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": methods,
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
        },
    )
