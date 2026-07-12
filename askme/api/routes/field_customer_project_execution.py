"""Customer-project execution binding and rehearsal FastAPI routes."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.customer_projects import (
    CustomerProjectExecutionBindingsResponse,
    CustomerProjectExecutionRehearsalResponse,
)
from askme.pipeline.field.customer_projects import (
    build_customer_project_execution_bindings,
    register_customer_project_onsite_evidence,
)
from askme.pipeline.field.field_ingest_adapters import normalize_field_ingest_payload

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
Dispatch = Callable[..., Awaitable[dict[str, Any]]]

DEFAULT_DELIVERY_NAMESPACE = "default"


def register_customer_project_execution_routes(
    app: FastAPI,
    *,
    site_profile_root: PathProvider,
    delivery_resource_root: PathProvider,
    project_read_auth: ProjectReadAuth,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_detail: ScopeItem,
    project_scope_forbidden: ProjectScopeForbidden,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register customer project execution-binding and object rehearsal routes."""

    def _detail_scope_failure(
        detail: dict[str, Any],
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if detail.get("found") and not scope_allows(scope, scope_item_from_detail(detail)):
            return project_scope_forbidden()
        return None

    @app.get(
        "/api/field/customer-projects/{identifier}/execution-bindings",
        tags=["Field Operations"],
        response_model=CustomerProjectExecutionBindingsResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_execution_bindings(
        identifier: str,
        request: Request,
    ) -> JSONResponse:
        """Return executable ingest/runtime binding plans for one customer project."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = build_customer_project_execution_bindings(
                root,
                identifier,
                delivery_resource_root=delivery_resource_root(),
            )
            scope_failure = _detail_scope_failure(result, scope)
            if scope_failure is not None:
                return scope_failure
            if result.get("found"):
                result = CustomerProjectExecutionBindingsResponse.model_validate(
                    result
                ).model_dump(mode="python")
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project execution bindings endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal",
        tags=["Field Operations"],
        response_model=CustomerProjectExecutionRehearsalResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_object_rehearsal(
        identifier: str,
        object_id: str,
        request: Request,
    ) -> JSONResponse:
        """Run a lab-only object ingest rehearsal for one managed object."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            scope = operator_project_scope(body)
            root = site_profile_root()
            bindings = build_customer_project_execution_bindings(
                root,
                identifier,
                delivery_resource_root=delivery_resource_root(),
            )
            if not bindings.get("found"):
                result = CustomerProjectExecutionRehearsalResponse.model_validate(
                    bindings
                ).model_dump(mode="python")
                return mission_json(result, status_code=404)
            scope_item = scope_item_from_detail(bindings)
            if not scope_allows(scope, scope_item):
                return project_scope_forbidden()

            plans_by_object = (
                bindings.get("plans_by_object_id")
                if isinstance(bindings.get("plans_by_object_id"), dict)
                else {}
            )
            plan = plans_by_object.get(object_id)
            if not isinstance(plan, dict):
                result = CustomerProjectExecutionRehearsalResponse.model_validate(
                    {
                        "accepted": False,
                        "status": "not_found",
                        "reason": "managed_object_not_found",
                        "object_id": object_id,
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=404,
                )

            mode = str(body.get("mode") or "dry_run").strip() or "dry_run"
            if mode not in {"dry_run", "shadow_post"}:
                result = CustomerProjectExecutionRehearsalResponse.model_validate(
                    {
                        "accepted": False,
                        "status": "rejected",
                        "reason": "invalid_rehearsal_mode",
                        "allowed_modes": ["dry_run", "shadow_post"],
                    }
                ).model_dump(mode="python")
                return mission_json(
                    result,
                    status_code=422,
                )

            raw_payload = _object_rehearsal_payload(plan, body, scope_item)
            raw_payload["rehearsal"] = _object_rehearsal_boundary(mode)
            normalized = normalize_field_ingest_payload(raw_payload)
            normalized.setdefault("managed_object_id", plan.get("object_id") or object_id)
            normalized.setdefault("project_scope", raw_payload.get("project_scope") or {})
            boundary = _object_rehearsal_boundary(mode)
            wants_onsite_evidence = _wants_rehearsal_onsite_evidence(body)
            result: dict[str, Any] = {
                "accepted": True,
                "status": "lab_rehearsed",
                "rehearsal": boundary,
                "project_scope": scope_item,
                "object_id": object_id,
                "plan": _object_rehearsal_plan_summary(plan),
                "raw_payload": raw_payload,
                "normalized": normalized,
                "production_claim_allowed": False,
                "customer_status": boundary["customer_status"],
                "release_claim": boundary["release_claim"],
                "production_eligible": False,
                "evidence_tier": boundary["evidence_tier"],
            }
            if wants_onsite_evidence and mode != "shadow_post":
                result["onsite_evidence_registration"] = _rehearsal_onsite_evidence_rejection(mode)
            if mode == "shadow_post":
                if not bool(body.get("confirm_shadow_post")):
                    result.update(
                        {
                            "accepted": False,
                            "status": "manual_check",
                            "reason": "shadow_post_requires_explicit_confirmation",
                            "next_step": (
                                "仅在实验室或现场演练窗口确认 shadow_post；"
                                "如果外部通知可能已配置，请改用 dry_run。"
                            ),
                        }
                    )
                    if wants_onsite_evidence:
                        result["onsite_evidence_registration"] = {
                            "requested": True,
                            "accepted": False,
                            "registered": False,
                            "reason": "shadow_post_requires_explicit_confirmation",
                            "production_eligible": False,
                        }
                    result = CustomerProjectExecutionRehearsalResponse.model_validate(
                        result
                    ).model_dump(mode="python")
                    return mission_json(result, status_code=409)
                ingest_result = await dispatch_field_operations("ingest_payload", raw_payload)
                result["ingest_result"] = ingest_result
                result["accepted"] = bool(ingest_result.get("accepted"))
                result["status"] = "shadow_posted" if result["accepted"] else "manual_check"
                result["event_id"] = ingest_result.get("event_id") or ingest_result.get("id") or ""
                if isinstance(ingest_result.get("normalized"), dict):
                    result["normalized"] = ingest_result["normalized"]
                    normalized = ingest_result["normalized"]
                if wants_onsite_evidence:
                    registration = _rehearsal_onsite_evidence_candidate(
                        mode=mode,
                        plan=plan,
                        normalized=normalized,
                        ingest_result=ingest_result,
                        operator_id=str(body.get("operator_id") or ""),
                    )
                    evidence = registration.pop("evidence", None)
                    if registration.get("accepted") and isinstance(evidence, dict):
                        write_result = register_customer_project_onsite_evidence(
                            root,
                            identifier,
                            evidence,
                            operator_id=str(body.get("operator_id") or ""),
                            reason="将对象 shadow-post 演练登记为现场验收候选证据。",
                        )
                        registration["registered"] = bool(write_result.get("accepted"))
                        registration["receipt"] = write_result.get("receipt") or {}
                        registration["onsite_acceptance_evidence"] = (
                            write_result.get("onsite_acceptance_evidence") or {}
                        )
                        if not write_result.get("accepted"):
                            registration["accepted"] = False
                            registration["reason"] = str(
                                write_result.get("reason") or "onsite_evidence_write_failed"
                            )
                    result["onsite_evidence_registration"] = registration
            result = CustomerProjectExecutionRehearsalResponse.model_validate(
                result
            ).model_dump(mode="python")
            return mission_json(result)
        except ValueError as exc:
            return mission_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("Field customer project object rehearsal endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/customer-projects/{identifier}/execution-bindings", include_in_schema=False)
    async def field_customer_project_execution_bindings_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

    @app.options(
        "/api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal",
        include_in_schema=False,
    )
    async def field_customer_project_object_rehearsal_cors(
        identifier: str,
        object_id: str,
    ) -> Response:
        _ = (identifier, object_id)
        return cors_options_response("POST, OPTIONS")


def _object_rehearsal_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": plan.get("object_id") or "",
        "display_name": plan.get("display_name") or plan.get("object_id") or "",
        "overall_status": plan.get("overall_status") or "unknown",
        "customer_status": plan.get("customer_status") or "",
        "required_sources": plan.get("required_sources") or [],
        "input_adapters": plan.get("input_adapters") or [],
        "skill_routes": plan.get("skill_routes") or [],
        "ingest_contract": plan.get("ingest_contract") or {},
        "runtime_contract": plan.get("runtime_contract") or {},
        "bridge_contract": plan.get("bridge_contract") or {},
        "blockers": plan.get("blockers") or [],
        "manual_checks": plan.get("manual_checks") or [],
    }


def _first_object_device(plan: dict[str, Any]) -> dict[str, Any]:
    source_plans = plan.get("source_plans") if isinstance(plan.get("source_plans"), list) else []
    for source_plan in source_plans:
        if not isinstance(source_plan, dict):
            continue
        devices = source_plan.get("devices") if isinstance(source_plan.get("devices"), list) else []
        for device in devices:
            if isinstance(device, dict):
                return device
    return {}


def _object_rehearsal_payload(
    plan: dict[str, Any],
    body: dict[str, Any],
    scope_item: dict[str, Any],
) -> dict[str, Any]:
    ingest_contract = plan.get("ingest_contract") if isinstance(plan.get("ingest_contract"), dict) else {}
    sample = (
        ingest_contract.get("sample_payload")
        if isinstance(ingest_contract.get("sample_payload"), dict)
        else {}
    )
    payload = dict(sample)
    override = body.get("payload") if isinstance(body.get("payload"), dict) else {}
    payload.update(override)
    payload["managed_object_id"] = str(plan.get("object_id") or payload.get("managed_object_id") or "")
    payload.setdefault("trigger_source", "customer_project_object_rehearsal")
    payload.setdefault("source", next(iter(plan.get("required_sources") or []), "camera"))
    observed_at = str(payload.get("observed_at") or "").strip().lower()
    if not observed_at or observed_at.startswith("iso-8601"):
        payload["observed_at"] = time.time()
    project_scope = {
        "tenant_id": scope_item.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
        "delivery_namespace": scope_item.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE,
        "customer_id": scope_item.get("customer_id") or "",
        "project_id": scope_item.get("project_id") or "",
        "site_id": scope_item.get("site_id") or "",
    }
    payload["project_scope"] = project_scope
    for key, value in project_scope.items():
        if value:
            payload[key] = value
    device = _first_object_device(plan)
    if device:
        payload.setdefault("device_id", device.get("device_id") or "")
        if payload.get("source") == "camera":
            payload.setdefault("camera_id", device.get("camera_id") or device.get("device_id") or "")
        if payload.get("source") == "sensor":
            payload.setdefault("sensor_id", device.get("sensor_id") or device.get("device_id") or "")
        if payload.get("source") == "robot":
            payload.setdefault("robot_id", device.get("robot_id") or device.get("device_id") or "")
    return payload


def _object_rehearsal_boundary(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "lab_only": True,
        "evidence_tier": "lab_rehearsal",
        "production_eligible": False,
        "production_claim_allowed": False,
        "customer_status": "仅限实验室演练，不能作为生产上线验收依据。",
        "release_claim": (
            "本次演练只证明适配器解析、对象绑定和事件接入契约形态；"
            "客户验收仍需要现场真实证据、可信设备签名、事件归档记录和运行回调证据。"
        ),
    }


def _wants_rehearsal_onsite_evidence(body: dict[str, Any]) -> bool:
    return body.get("register_onsite_evidence") is True or body.get("promote_to_onsite_evidence") is True


def _rehearsal_onsite_evidence_rejection(mode: str) -> dict[str, Any]:
    reason = "dry_run_rehearsal_not_onsite_evidence"
    if mode == "shadow_post":
        reason = "shadow_post_not_accepted"
    return {
        "requested": True,
        "accepted": False,
        "registered": False,
        "reason": reason,
        "evidence_tier": "lab_rehearsal",
        "production_eligible": False,
        "customer_status": (
            "干跑演练只证明适配器形态；现场验收需要已确认的 shadow/live "
            "上报和真实交付证据。"
        ),
    }


def _rehearsal_onsite_evidence_candidate(
    *,
    mode: str,
    plan: dict[str, Any],
    normalized: dict[str, Any],
    ingest_result: dict[str, Any],
    operator_id: str,
) -> dict[str, Any]:
    if mode != "shadow_post":
        return _rehearsal_onsite_evidence_rejection(mode)
    if not bool(ingest_result.get("accepted")):
        result = _rehearsal_onsite_evidence_rejection(mode)
        result["upstream_reason"] = str(ingest_result.get("reason") or ingest_result.get("status") or "")
        return result

    event = ingest_result.get("event") if isinstance(ingest_result.get("event"), dict) else {}
    trust = normalized.get("device_trust") if isinstance(normalized.get("device_trust"), dict) else {}
    if not trust:
        result_normalized = (
            ingest_result.get("normalized")
            if isinstance(ingest_result.get("normalized"), dict)
            else {}
        )
        trust = (
            result_normalized.get("device_trust")
            if isinstance(result_normalized.get("device_trust"), dict)
            else {}
        )
    runtime_delivery = event.get("runtime_delivery") if isinstance(event.get("runtime_delivery"), dict) else {}
    signature_verified = trust.get("signature_verified") is True
    runtime_completed = str(runtime_delivery.get("status") or "") in {
        "completed",
        "delivered",
        "recorded",
    }
    missing: list[str] = []
    if not signature_verified:
        missing.append("trusted_device_signature")
    if not runtime_completed:
        missing.append("runtime_completion_callback")
    object_id_value = str(plan.get("object_id") or normalized.get("managed_object_id") or "")
    return {
        "requested": True,
        "accepted": True,
        "registered": True,
        "status": "manual_check",
        "evidence": {
            "evidence_type": "device_ingest",
            "status": "manual_check",
            "source": "object_execution_rehearsal",
            "label": f"对象演练 shadow 上报：{plan.get('display_name') or object_id_value}",
            "summary": (
                "已确认的 shadow-post 演练生成了现场事件。它只能作为现场验收候选证据；"
                "现场验收通过仍需要真实现场证据、可信签名、运行完成回调和交付复核。"
            ),
            "managed_object_id": object_id_value,
            "event_id": str(event.get("event_id") or ingest_result.get("event_id") or ""),
            "runtime_run_id": str(runtime_delivery.get("run_id") or runtime_delivery.get("runtime_run_id") or ""),
            "external_reference": f"object-rehearsal:shadow_post:{object_id_value}",
            "operator_id": operator_id,
            "reason": "将对象 shadow-post 演练登记为现场验收候选证据。",
            "evidence_tier": "acceptance_candidate",
            "production_eligible": False,
        },
        "eligibility": {
            "signature_verified": signature_verified,
            "runtime_completed": runtime_completed,
            "ready_for_delivery_review": signature_verified and runtime_completed,
            "missing_for_delivery_review": missing,
        },
        "production_eligible": False,
        "customer_status": (
            "已登记为验收候选证据；交付负责人复核并关联真实现场证据前，"
            "不能作为现场验收通过或生产上线证据。"
        ),
    }


__all__ = ["register_customer_project_execution_routes"]
