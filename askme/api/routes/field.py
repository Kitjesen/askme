"""Field operations FastAPI routes."""

from __future__ import annotations

import logging
import mimetypes
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from askme.pipeline.alert_dispatcher import AlertDispatcher
from askme.pipeline.field_ingest_adapters import normalize_field_ingest_payload
from askme.pipeline.field_site_profile import (
    archive_customer_project_profile,
    build_customer_project_acceptance_registry,
    build_customer_project_catalog,
    build_customer_project_execution_bindings,
    build_customer_project_resource_catalog,
    build_site_profile_catalog,
    build_solution_delivery_readiness,
    create_customer_project_from_template,
    create_customer_project_template_release_request,
    create_delivery_resource_governance_request,
    customer_project_acceptance_closure,
    customer_project_acceptance_report,
    customer_project_catalog_acceptance_gate,
    customer_project_catalog_summary_from_projects,
    customer_project_template_release_notes,
    customer_project_template_summary_from_items,
    delete_managed_object,
    diff_customer_project_package,
    disable_delivery_resource,
    escalate_overdue_delivery_resource_governance_requests,
    export_customer_project_acceptance_dossier,
    export_customer_project_package,
    export_customer_project_proposal_bundle,
    export_customer_project_template_release_notes_bundle,
    get_customer_project_profile,
    import_customer_project_package,
    list_customer_project_customer_signoffs,
    list_customer_project_onsite_evidence,
    list_customer_project_revisions,
    list_customer_project_template_release_requests,
    list_customer_project_template_revisions,
    list_customer_project_templates,
    list_delivery_resource_governance_requests,
    list_delivery_resource_registry,
    list_delivery_resource_revisions,
    register_customer_project_acceptance_review,
    register_customer_project_customer_signoff,
    register_customer_project_onsite_evidence,
    review_customer_project_template_release_request,
    review_delivery_resource_governance_request,
    rollback_customer_project_profile,
    rollback_delivery_resource_registry,
    update_customer_project_template_release,
    upsert_customer_project_profile,
    upsert_delivery_resource,
    upsert_managed_object,
    verify_customer_project_acceptance_dossier,
    verify_customer_project_package,
    verify_customer_project_proposal_bundle,
)
from askme.pipeline.product_launch_readiness import build_product_launch_readiness

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
ManualTriggerBody = Callable[[Request, dict[str, Any]], dict[str, Any]]
LooksLikeDeviceIngest = Callable[[dict[str, Any]], bool]
FieldResultHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
FieldRuntimePolicy = Callable[..., Awaitable[dict[str, Any]]]
RuntimeCallbackTrust = Callable[..., dict[str, Any]]
RuntimeCallbackDeliveryBody = Callable[..., dict[str, Any]]
ConfigProvider = Callable[[], dict[str, Any]]
IdentityReadinessPayload = Callable[[], dict[str, Any]]

_FIELD_EVIDENCE_ROOT_NAMES = ("artifacts", "output", "data")
_DEFAULT_DELIVERY_NAMESPACE = "default"

_CUSTOMER_PROJECT_TERMS = {
    "tenant_id": "客户空间",
    "delivery_namespace": "交付空间",
    "customer_project": "客户项目",
    "managed_object": "现场对象",
    "managed_object_directory": "对象目录",
    "bindings": "能力配置",
    "vision_models": "识别能力",
    "sensor_protocols": "设备接入方式",
    "skill_packages": "业务能力",
    "acceptance_tests": "验收项",
    "delivery_resources": "交付资源",
    "package_delivery_gate": "交付包准入检查",
    "dry_run": "预检",
    "runtime": "执行服务",
    "operator_id": "操作人",
}

_CUSTOMER_PROJECT_ACCEPTANCE_FLOW = [
    {
        "step_id": "scope_isolated",
        "label": "确认客户范围",
        "customer_value": "客户只能看到自己项目、对象、证据和交付包。",
        "acceptance_standard": "越权读取、导出、导入预检都会被拒绝或返回空结果。",
    },
    {
        "step_id": "template_selected",
        "label": "选择行业模板",
        "customer_value": "新项目从厂区、园区、仓储、景区模板复制，不从空白配置开始。",
        "acceptance_standard": "模板展示适用场景、默认对象、交付边界和客户准备项。",
    },
    {
        "step_id": "object_directory_ready",
        "label": "核对对象目录",
        "customer_value": "客户能看懂本项目覆盖哪些车辆、设备、游客、烟火、垃圾桶或通道对象。",
        "acceptance_standard": "每个对象展示识别能力、设备接入方式、业务能力、验收项和未完成原因。",
    },
    {
        "step_id": "package_preflight",
        "label": "交付包预检",
        "customer_value": "导入前先看到新增、覆盖、冲突和阻断项，避免误覆盖客户现场。",
        "acceptance_standard": "预检不写入项目；冲突和越权包不能导入。",
    },
    {
        "step_id": "customer_acceptance",
        "label": "输出验收材料",
        "customer_value": "客户按项目范围、对象覆盖、未完成项和证据结论验收。",
        "acceptance_standard": "报告不要求客户阅读 YAML、接口路径或测试节点即可判断交付状态。",
    },
]


def _customer_project_term_cards() -> list[dict[str, str]]:
    return [
        {"internal": key, "customer_label": value}
        for key, value in _CUSTOMER_PROJECT_TERMS.items()
    ]


def register_field_routes(
    app: FastAPI,
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize,
    field_manual_trigger_body: ManualTriggerBody,
    looks_like_device_ingest_without_scenario: LooksLikeDeviceIngest,
    dispatch_field_voice_directive: FieldResultHook,
    dispatch_field_runtime_policy: FieldRuntimePolicy,
    runtime_callback_trust: RuntimeCallbackTrust,
    runtime_callback_delivery_body: RuntimeCallbackDeliveryBody,
    runtime_callback_secret: str | None,
    runtime_callback_max_age_s: float,
    cors_headers: dict[str, str],
    identity_readiness_payload: IdentityReadinessPayload,
    site_profile_root: Path | None = None,
    config_provider: ConfigProvider | None = None,
) -> None:
    """Register customer-facing field event, notification, and ingest routes."""

    def _project_read_auth(request: Request) -> tuple[JSONResponse | None, dict[str, Any]]:
        body: dict[str, Any] = {}
        return authorize(request, body, "field:project:read"), body

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

    def _clean_scope_values(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    def _resource_governance_delivery_config() -> dict[str, Any]:
        config = config_provider() if config_provider is not None else {}
        if not isinstance(config, dict):
            return {}
        field_cfg = config.get("field_operations") if isinstance(config.get("field_operations"), dict) else {}
        governance_cfg = (
            field_cfg.get("delivery_resource_governance")
            if isinstance(field_cfg.get("delivery_resource_governance"), dict)
            else {}
        )
        notify_cfg = (
            governance_cfg.get("delivery_owner_notifications")
            if isinstance(governance_cfg.get("delivery_owner_notifications"), dict)
            else {}
        )
        return dict(notify_cfg)

    def _resource_governance_delivery_channels(notify_cfg: dict[str, Any]) -> list[str]:
        routes = notify_cfg.get("severity_routes")
        if isinstance(routes, dict) and isinstance(routes.get("warning"), list):
            return [
                str(item).strip()
                for item in routes.get("warning", [])
                if str(item).strip()
            ]
        channels: list[str] = []
        if notify_cfg.get("webhook_url"):
            channels.append("webhook")
        if notify_cfg.get("dingtalk_webhook"):
            channels.append("dingtalk")
        if notify_cfg.get("wecom_webhook"):
            channels.append("wecom")
        if notify_cfg.get("feishu_webhook"):
            channels.append("feishu")
        channels.append("log")
        return channels

    def _resource_governance_notification_delivery(
        escalation: dict[str, Any],
    ) -> dict[str, Any]:
        notify_cfg = _resource_governance_delivery_config()
        notification = (
            escalation.get("notification")
            if isinstance(escalation.get("notification"), dict)
            else {}
        )
        message = str(notification.get("message") or "")
        if not bool(notify_cfg.get("enabled")):
            return {
                "status": "queued",
                "delivery_mode": "local_queue",
                "reason": "delivery_owner_notification_not_enabled",
                "sent_channels": [],
                "delivery_report": [
                    {
                        "channel": "delivery_owner_queue",
                        "status": "queued",
                        "reason": "local_delivery_owner_queue",
                    }
                ],
            }
        channels = _resource_governance_delivery_channels(notify_cfg)
        dispatcher = AlertDispatcher(
            config={
                "webhook_url": notify_cfg.get("webhook_url") or "",
                "wecom_webhook": notify_cfg.get("wecom_webhook") or "",
                "dingtalk_webhook": notify_cfg.get("dingtalk_webhook") or "",
                "dingtalk_secret": notify_cfg.get("dingtalk_secret") or "",
                "feishu_webhook": notify_cfg.get("feishu_webhook") or "",
                "severity_routes": {
                    "warning": channels,
                    "info": ["log"],
                    "error": channels,
                },
                "incident_archive_path": notify_cfg.get("incident_archive_path") or "",
            },
            robot_id=str(notify_cfg.get("robot_id") or "askme-delivery"),
            robot_name=str(notify_cfg.get("robot_name") or "AskMe Delivery"),
        )
        sent_channels = dispatcher.dispatch(
            message,
            severity="warning",
            topic="delivery_resource_governance.overdue",
            payload={
                "escalation": escalation,
                "dingtalk_message": message,
            },
        )
        return {
            "status": "sent" if sent_channels else "not_sent",
            "delivery_mode": "configured_channels",
            "sent_channels": sent_channels,
            "delivery_report": dispatcher.last_delivery_report,
        }

    def _scope_allows(scope: dict[str, list[str]], item: dict[str, Any]) -> bool:
        if not any(scope.values()):
            return True
        for scope_key, item_key in (
            ("tenant_ids", "tenant_id"),
            ("delivery_namespaces", "delivery_namespace"),
            ("customer_ids", "customer_id"),
            ("project_ids", "project_id"),
            ("site_ids", "site_id"),
        ):
            allowed = scope.get(scope_key) or []
            if "*" in allowed:
                continue
            value = str(item.get(item_key) or "").strip()
            if allowed and value not in allowed:
                return False
        return True

    def _scoped_query_value(
        requested: str,
        scope: dict[str, list[str]],
        scope_key: str,
    ) -> tuple[bool, str]:
        value = str(requested or "").strip()
        allowed = scope.get(scope_key) or []
        if not allowed or "*" in allowed:
            return True, value
        if value:
            return value in allowed, value
        return True, ""

    def _project_scope_forbidden() -> JSONResponse:
        return mission_json(
            {
                "error": "operator not authorized for this customer project",
                "reason": "project_scope_not_allowed",
            },
            status_code=403,
        )

    def _has_explicit_project_scope(payload: dict[str, Any]) -> bool:
        explicit_scope = payload.get("project_scope")
        if isinstance(explicit_scope, dict) and any(
            explicit_scope.get(key)
            for key in ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id")
        ):
            return True
        nested_payload = payload.get("payload")
        if isinstance(nested_payload, dict) and _has_explicit_project_scope(nested_payload):
            return True
        return any(
            payload.get(key)
            for key in ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id")
        )

    def _scope_item_from_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
        nested_payload = payload.get("payload")
        source = nested_payload if isinstance(nested_payload, dict) else payload
        explicit_scope = source.get("project_scope") if isinstance(source.get("project_scope"), dict) else {}
        return {
            "tenant_id": source.get("tenant_id") or explicit_scope.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": (
                source.get("delivery_namespace")
                or explicit_scope.get("delivery_namespace")
                or _DEFAULT_DELIVERY_NAMESPACE
            ),
            "customer_id": source.get("customer_id") or explicit_scope.get("customer_id") or "",
            "project_id": source.get("project_id") or explicit_scope.get("project_id") or "",
            "site_id": source.get("site_id") or explicit_scope.get("site_id") or "",
        }

    def _apply_single_scope_defaults(payload: dict[str, Any], scope: dict[str, list[str]]) -> None:
        for payload_key, scope_key in (
            ("tenant_id", "tenant_ids"),
            ("delivery_namespace", "delivery_namespaces"),
            ("customer_id", "customer_ids"),
            ("project_id", "project_ids"),
            ("site_id", "site_ids"),
        ):
            allowed = scope.get(scope_key) or []
            if "*" in allowed or len(allowed) != 1:
                continue
            payload.setdefault(payload_key, allowed[0])

    def _scope_item_from_event_detail(payload: dict[str, Any]) -> dict[str, Any]:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
        return {
            "tenant_id": event.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": event.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": event.get("customer_id") or "",
            "project_id": event.get("project_id") or "",
            "site_id": event.get("site_id") or "",
        }

    async def _field_event_scope_failure(
        event_id: str,
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if not any(scope.values()):
            return None
        detail = await dispatch_field_operations("detail_payload", event_id)
        if not detail.get("found"):
            return None
        if not _scope_allows(scope, _scope_item_from_event_detail(detail)):
            return _project_scope_forbidden()
        return None

    def _scope_project_catalog(payload: dict[str, Any], scope: dict[str, list[str]]) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        projects = [
            project
            for project in payload.get("projects", [])
            if isinstance(project, dict) and _scope_allows(scope, project)
        ]
        filtered = dict(payload)
        filtered["projects"] = projects
        filtered["customers"] = _customer_rows_for_projects(projects)
        summary = customer_project_catalog_summary_from_projects(
            projects,
            base_summary=payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        )
        summary["scope_filtered"] = True
        filtered["summary"] = summary
        filtered["delivery_acceptance_gate"] = customer_project_catalog_acceptance_gate(projects)
        return filtered

    def _managed_object_delivery_status(item: dict[str, Any]) -> str:
        resource_status = item.get("resource_binding_status")
        acceptance_status = item.get("acceptance_status")
        resource = (
            str(resource_status.get("overall_status") or "manual_check")
            if isinstance(resource_status, dict)
            else "manual_check"
        )
        acceptance = (
            str(acceptance_status.get("status") or "manual_check")
            if isinstance(acceptance_status, dict)
            else "manual_check"
        )
        if resource in {"blocked", "failed"} or acceptance in {
            "blocked",
            "failed",
            "file_missing",
            "outside_project",
        }:
            return "blocked"
        if resource == "ready" and acceptance == "ready":
            return "ready"
        return "manual_check"

    def _managed_object_directory_rows(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for project in projects:
            objects = project.get("managed_objects") if isinstance(project.get("managed_objects"), list) else []
            for item in objects:
                if not isinstance(item, dict):
                    continue
                bindings = item.get("bindings") if isinstance(item.get("bindings"), dict) else {}
                resource_status = (
                    item.get("resource_binding_status")
                    if isinstance(item.get("resource_binding_status"), dict)
                    else {}
                )
                acceptance_status = (
                    item.get("acceptance_status")
                    if isinstance(item.get("acceptance_status"), dict)
                    else {}
                )
                row = {
                    "tenant_id": project.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
                    "delivery_namespace": (
                        project.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE
                    ),
                    "customer_id": project.get("customer_id") or "",
                    "customer_name": project.get("customer_name") or project.get("customer_id") or "",
                    "project_id": project.get("project_id") or "",
                    "project_name": project.get("project_name") or project.get("project_id") or "",
                    "site_id": project.get("site_id") or "",
                    "site_name": project.get("site_name") or project.get("site_id") or "",
                    "industry": project.get("industry") or "",
                    "deployment_stage": project.get("deployment_stage") or "",
                    "project_status": project.get("status") or "",
                    "object_id": item.get("object_id") or "",
                    "display_name": item.get("display_name") or item.get("object_id") or "",
                    "category": item.get("category") or "",
                    "object_labels": item.get("object_labels") if isinstance(item.get("object_labels"), list) else [],
                    "scenario_ids": item.get("scenario_ids") if isinstance(item.get("scenario_ids"), list) else [],
                    "zone_types": item.get("zone_types") if isinstance(item.get("zone_types"), list) else [],
                    "device_sources": (
                        item.get("device_sources") if isinstance(item.get("device_sources"), list) else []
                    ),
                    "responder_group": item.get("responder_group") or "",
                    "evidence_required": (
                        item.get("evidence_required")
                        if isinstance(item.get("evidence_required"), list)
                        else []
                    ),
                    "customer_visible": item.get("customer_visible") is not False,
                    "scope_constraints": {
                        "tenant_ids": item.get("tenant_ids") if isinstance(item.get("tenant_ids"), list) else [],
                        "delivery_namespaces": (
                            item.get("delivery_namespaces")
                            if isinstance(item.get("delivery_namespaces"), list)
                            else []
                        ),
                        "customer_ids": (
                            item.get("customer_ids") if isinstance(item.get("customer_ids"), list) else []
                        ),
                        "project_ids": (
                            item.get("project_ids") if isinstance(item.get("project_ids"), list) else []
                        ),
                        "site_ids": item.get("site_ids") if isinstance(item.get("site_ids"), list) else [],
                    },
                    "bindings": bindings,
                    "resource_binding_status": resource_status,
                    "acceptance_status": acceptance_status,
                    "delivery_status": _managed_object_delivery_status(item),
                    "resource_check_count": int(resource_status.get("check_count") or 0),
                    "acceptance_test_count": len(
                        bindings.get("acceptance_tests")
                        if isinstance(bindings.get("acceptance_tests"), list)
                        else []
                    ),
                    "acceptance_check_count": len(
                        acceptance_status.get("acceptance_checks")
                        if isinstance(acceptance_status.get("acceptance_checks"), list)
                        else []
                    ),
                }
                row["action_plan"] = _managed_object_directory_action_plan(row)
                row["action_count"] = len(row["action_plan"])
                row["blocked_action_count"] = sum(
                    1 for action in row["action_plan"] if action.get("severity") == "blocked"
                )
                row["manual_check_action_count"] = sum(
                    1 for action in row["action_plan"] if action.get("severity") == "manual_check"
                )
                row["next_step"] = (
                    str(row["action_plan"][0].get("next_step") or "")
                    if row["action_plan"]
                    else str(acceptance_status.get("next_step") or "Run object acceptance checks.")
                )
                rows.append(row)
        rows.sort(
            key=lambda item: (
                str(item.get("tenant_id") or ""),
                str(item.get("delivery_namespace") or ""),
                str(item.get("customer_id") or ""),
                str(item.get("project_id") or ""),
                str(item.get("object_id") or ""),
            )
        )
        return rows

    def _managed_object_directory_action_plan(row: dict[str, Any]) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        object_id = str(row.get("object_id") or "")

        def _resource_type_label(resource_type: str) -> str:
            return {
                "vision_models": "识别模型",
                "sensor_protocols": "传感器协议",
                "skill_packages": "能力包",
                "acceptance_tests": "验收用例",
            }.get(resource_type, "交付资源")

        def _owner_label(owner: str) -> str:
            return {
                "delivery_owner": "交付负责人",
                "qa_owner": "测试/验收负责人",
            }.get(owner, owner or "负责人")

        def _action_payload(
            *,
            action_id: str,
            action: str,
            action_label: str,
            reason_label: str,
            severity: str,
            owner: str,
            target: dict[str, Any],
            status: str,
            message: str,
            next_step: str,
        ) -> dict[str, Any]:
            return {
                "action_id": action_id,
                "action": action,
                "action_label": action_label,
                "reason_label": reason_label,
                "severity": severity,
                "owner": owner,
                "owner_label": _owner_label(owner),
                "target": target,
                "status": status,
                "message": message,
                "next_step": next_step,
                "customer_next_step": next_step,
            }

        resource_status = row.get("resource_binding_status") if isinstance(row.get("resource_binding_status"), dict) else {}
        resource_checks = resource_status.get("checks") if isinstance(resource_status.get("checks"), list) else []
        for check in resource_checks:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status") or "")
            if status == "linked":
                continue
            resource_type = str(check.get("resource_type") or "resource")
            resource_id = str(check.get("resource_id") or "")
            resource_label = _resource_type_label(resource_type)
            severity = "blocked" if status in {"missing", "blocked", "disabled", "failed"} else "manual_check"
            if status == "missing":
                action = "bind_required_resource"
                action_label = "补齐对象资源绑定"
                reason_label = f"缺少{resource_label}"
                next_step = f"在对象编辑区为 {object_id} 绑定至少一个{resource_label}。"
            elif status == "unregistered":
                action = "register_delivery_resource"
                action_label = "登记交付资源"
                reason_label = f"{resource_label}未登记"
                next_step = f"先在共享资源登记表登记 {resource_type}:{resource_id}，再回到对象目录复核。"
            elif status in {"manual_check", "draft", "pilot", "deprecated"}:
                action = "review_delivery_resource"
                action_label = "复核交付资源"
                reason_label = f"{resource_label}需复核"
                next_step = f"确认 {resource_type}:{resource_id} 是否可用于当前客户项目。"
            else:
                action = "replace_blocked_resource"
                action_label = "替换不可用资源"
                reason_label = f"{resource_label}不可用于交付"
                next_step = f"把 {resource_type}:{resource_id} 替换为已批准的交付资源。"
            actions.append(_action_payload(
                action_id=f"{object_id}:{resource_type}:{resource_id or 'missing'}:{status}",
                action=action,
                action_label=action_label,
                reason_label=reason_label,
                severity=severity,
                owner="delivery_owner",
                target={
                    "object_id": object_id,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                },
                status=status,
                message=str(check.get("message") or ""),
                next_step=next_step,
            ))
        acceptance_status = row.get("acceptance_status") if isinstance(row.get("acceptance_status"), dict) else {}
        missing_requirements = (
            acceptance_status.get("missing")
            if isinstance(acceptance_status.get("missing"), list)
            else []
        )
        for missing in missing_requirements:
            missing_key = str(missing or "")
            actions.append(_action_payload(
                action_id=f"{object_id}:acceptance:{missing_key}:missing",
                action="bind_acceptance_requirement",
                action_label="补齐验收要求",
                reason_label="缺少验收要求",
                severity="blocked",
                owner="delivery_owner",
                target={
                    "object_id": object_id,
                    "requirement": missing_key,
                },
                status="missing",
                message="Acceptance requirement is not configured.",
                next_step=f"在对象编辑区补齐 {missing_key}，否则客户验收前必须阻断。",
            ))
        for check in (
            acceptance_status.get("acceptance_checks", [])
            if isinstance(acceptance_status.get("acceptance_checks"), list)
            else []
        ):
            if not isinstance(check, dict):
                continue
            status = str(check.get("status") or "")
            if status == "linked":
                continue
            severity = "blocked" if status in {"file_missing", "invalid_reference", "outside_project"} else "manual_check"
            reason_label = {
                "file_missing": "验收用例文件缺失",
                "invalid_reference": "验收引用无效",
                "outside_project": "验收证据不在项目范围内",
                "node_unresolved": "验收用例节点待确认",
            }.get(status, "验收证据需复核")
            actions.append(_action_payload(
                action_id=f"{object_id}:acceptance_test:{check.get('reference') or check.get('path') or 'unknown'}:{status}",
                action="fix_acceptance_test_reference",
                action_label="修正验收证据引用",
                reason_label=reason_label,
                severity=severity,
                owner="qa_owner",
                target={
                    "object_id": object_id,
                    "reference": check.get("reference") or "",
                    "path": check.get("path") or "",
                    "node": check.get("node") or "",
                },
                status=status,
                message=str(check.get("message") or ""),
                next_step=str(
                    check.get("next_step")
                    or acceptance_status.get("next_step")
                    or "把验收用例引用修正到当前项目内可执行、可复核的证据。"
                ),
            ))
        return actions

    def _managed_object_directory_summary(
        rows: list[dict[str, Any]],
        *,
        projects: list[dict[str, Any]],
        base_summary: dict[str, Any],
        filtered: bool,
    ) -> dict[str, Any]:
        ready_count = sum(1 for row in rows if row.get("delivery_status") == "ready")
        manual_check_count = sum(1 for row in rows if row.get("delivery_status") == "manual_check")
        blocked_count = sum(1 for row in rows if row.get("delivery_status") == "blocked")
        scoped_object_count = sum(
            1
            for row in rows
            if any(row.get("scope_constraints", {}).get(key) for key in row.get("scope_constraints", {}))
        )
        overall_status = "manual_check"
        if rows and not blocked_count and not manual_check_count:
            overall_status = "ready"
        elif blocked_count:
            overall_status = "blocked"
        return {
            "object_count": len(rows),
            "project_count": len(projects),
            "customer_count": len({str(row.get("customer_id") or "") for row in rows if row.get("customer_id")}),
            "site_count": len({str(row.get("site_id") or "") for row in rows if row.get("site_id")}),
            "ready_count": ready_count,
            "manual_check_count": manual_check_count,
            "blocked_count": blocked_count,
            "customer_visible_count": sum(1 for row in rows if row.get("customer_visible") is not False),
            "acceptance_test_count": sum(int(row.get("acceptance_test_count") or 0) for row in rows),
            "scoped_object_count": scoped_object_count,
            "action_count": sum(int(row.get("action_count") or 0) for row in rows),
            "blocked_action_count": sum(int(row.get("blocked_action_count") or 0) for row in rows),
            "manual_check_action_count": sum(
                int(row.get("manual_check_action_count") or 0) for row in rows
            ),
            "categories": sorted({str(row.get("category") or "") for row in rows if row.get("category")}),
            "device_sources": sorted(
                {
                    str(source)
                    for row in rows
                    for source in row.get("device_sources", [])
                    if str(source).strip()
                }
            ),
            "scenario_ids": sorted(
                {
                    str(scenario)
                    for row in rows
                    for scenario in row.get("scenario_ids", [])
                    if str(scenario).strip()
                }
            ),
            "overall_status": overall_status,
            "scope_filtered": bool(base_summary.get("scope_filtered")),
            "filtered": filtered,
        }

    def _filter_managed_object_directory_rows(
        rows: list[dict[str, Any]],
        *,
        delivery_status: str,
        category: str,
        customer_visible: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        normalized_status = str(delivery_status or "").strip().lower()
        normalized_category = str(category or "").strip().lower()
        normalized_customer_visible = str(customer_visible or "").strip().lower()
        filtered = rows
        filters: dict[str, Any] = {}
        if normalized_status:
            filters["delivery_status"] = normalized_status
            filtered = [
                row
                for row in filtered
                if str(row.get("delivery_status") or "").lower() == normalized_status
            ]
        if normalized_category:
            filters["category"] = normalized_category
            filtered = [
                row
                for row in filtered
                if str(row.get("category") or "").lower() == normalized_category
            ]
        if normalized_customer_visible in {"true", "false"}:
            visible = normalized_customer_visible == "true"
            filters["customer_visible"] = visible
            filtered = [row for row in filtered if bool(row.get("customer_visible")) is visible]
        return filtered, filters

    def _scope_template_catalog(payload: dict[str, Any], scope: dict[str, list[str]]) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        templates = [
            item
            for item in payload.get("templates", [])
            if isinstance(item, dict) and _scope_allows_template(scope, item)
        ]
        filtered = dict(payload)
        filtered["templates"] = templates
        summary = customer_project_template_summary_from_items(templates)
        summary["scope_filtered"] = True
        if isinstance(payload.get("summary"), dict) and payload["summary"].get("filtered"):
            summary["filtered"] = True
            summary["filters"] = payload["summary"].get("filters") or payload.get("filters") or {}
        filtered["summary"] = summary
        return filtered

    def _scope_allows_template(scope: dict[str, list[str]], template: dict[str, Any]) -> bool:
        tenant_id = str(template.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE)
        namespace = str(template.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE)
        if tenant_id == _DEFAULT_DELIVERY_NAMESPACE and namespace == _DEFAULT_DELIVERY_NAMESPACE:
            return True
        return _scope_allows(
            scope,
            {
                "tenant_id": tenant_id,
                "delivery_namespace": namespace,
                "customer_id": str(template.get("customer_id") or ""),
                "project_id": str(template.get("project_id") or ""),
                "site_id": str(template.get("site_id") or ""),
            },
        )

    def _scope_site_catalog(payload: dict[str, Any], scope: dict[str, list[str]]) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        sites = [
            site
            for site in payload.get("sites", [])
            if isinstance(site, dict) and _scope_allows(scope, _scope_item_from_site(site))
        ]
        filtered = dict(payload)
        filtered["sites"] = sites
        summary = dict(payload.get("summary") if isinstance(payload.get("summary"), dict) else {})
        summary.update({
            "site_count": len(sites),
            "configured_count": len([item for item in sites if item.get("status") == "passed"]),
            "blocked_count": len([item for item in sites if item.get("status") != "passed"]),
            "production_ready_count": len([
                item for item in sites if item.get("deployment_stage") == "production_ready"
            ]),
            "scope_filtered": True,
        })
        filtered["summary"] = summary
        return filtered

    def _scope_acceptance_registry(payload: dict[str, Any], scope: dict[str, list[str]]) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        consumers = [
            consumer
            for consumer in payload.get("consumers", [])
            if isinstance(consumer, dict)
            and (
                consumer.get("scope_type") == "template"
                or _scope_allows(scope, consumer)
            )
        ]
        references = []
        for reference in payload.get("references", []):
            if not isinstance(reference, dict):
                continue
            scoped_consumers = [
                consumer
                for consumer in reference.get("consumers", [])
                if isinstance(consumer, dict)
                and (
                    consumer.get("scope_type") == "template"
                    or _scope_allows(scope, consumer)
                )
            ]
            if not scoped_consumers:
                continue
            row = dict(reference)
            row["consumers"] = scoped_consumers
            row["consumer_count"] = len(scoped_consumers)
            row["linked_count"] = len([
                item for item in scoped_consumers if _registry_status_bucket(item.get("status")) == "linked"
            ])
            row["manual_check_count"] = len([
                item for item in scoped_consumers if _registry_status_bucket(item.get("status")) == "manual_check"
            ])
            row["blocked_count"] = len([
                item for item in scoped_consumers if _registry_status_bucket(item.get("status")) == "blocked"
            ])
            row["status"] = _registry_overall_status(row)
            references.append(row)
        filtered = dict(payload)
        filtered["consumers"] = consumers
        filtered["references"] = references
        filtered["summary"] = _registry_summary(consumers, references) | {"scope_filtered": True}
        return filtered

    def _scope_resource_catalog(payload: dict[str, Any], scope: dict[str, list[str]]) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        consumers = [
            consumer
            for consumer in payload.get("consumers", [])
            if isinstance(consumer, dict)
            and (
                consumer.get("scope_type") == "template"
                or _scope_allows(scope, consumer)
            )
        ]
        resources = []
        for resource in payload.get("resources", []):
            if not isinstance(resource, dict):
                continue
            if _resource_has_explicit_scope(resource) and not _scope_allows(
                scope,
                _scope_item_from_resource(resource),
            ):
                continue
            scoped_consumers = [
                consumer
                for consumer in resource.get("consumers", [])
                if isinstance(consumer, dict)
                and (
                    consumer.get("scope_type") == "template"
                    or _scope_allows(scope, consumer)
                )
            ]
            if not scoped_consumers and int(resource.get("consumer_count") or 0) > 0:
                continue
            row = dict(resource)
            row["consumers"] = scoped_consumers
            row["consumer_count"] = len(scoped_consumers)
            row["project_count"] = len([
                item for item in scoped_consumers if item.get("scope_type") == "project"
            ])
            row["template_count"] = len([
                item for item in scoped_consumers if item.get("scope_type") == "template"
            ])
            row["unregistered_consumer_count"] = len([
                item for item in scoped_consumers if item.get("status") == "unregistered"
            ])
            resources.append(row)
        summary = _resource_summary(resources, consumers)
        filtered = dict(payload)
        filtered["consumers"] = consumers
        filtered["resources"] = resources
        filtered["summary"] = summary | {"scope_filtered": True}
        return filtered

    def _scope_delivery_resource_registry(
        payload: dict[str, Any],
        scope: dict[str, list[str]],
    ) -> dict[str, Any]:
        if not any(scope.values()):
            return payload
        resources = [
            resource
            for resource in payload.get("resources", [])
            if isinstance(resource, dict)
            and (
                not _resource_has_explicit_scope(resource)
                or _scope_allows(scope, _scope_item_from_resource(resource))
            )
        ]
        filtered = dict(payload)
        filtered["resources"] = resources
        filtered["summary"] = _resource_summary(resources, []) | {"scope_filtered": True}
        filtered["delivery_resources"] = _delivery_resource_tree_from_rows(resources)
        return filtered

    def _delivery_resource_tree_from_rows(resources: list[dict[str, Any]]) -> dict[str, Any]:
        tree: dict[str, dict[str, dict[str, Any]]] = {
            resource_type: {}
            for resource_type in (
                "vision_models",
                "sensor_protocols",
                "skill_packages",
                "acceptance_tests",
            )
        }
        for resource in resources:
            resource_type = str(resource.get("resource_type") or "")
            resource_id = str(resource.get("resource_id") or "")
            if not resource_type or not resource_id:
                continue
            row = dict(resource)
            row.pop("consumers", None)
            tree.setdefault(resource_type, {})[resource_id] = row
        return tree

    def _registry_status_bucket(status: Any) -> str:
        text = str(status or "").strip()
        if text in {"linked", "passed", "configured"}:
            return "linked"
        if text in {"node_unresolved", "read_error", "manual_check", "not_run"}:
            return "manual_check"
        return "blocked"

    def _registry_overall_status(row: dict[str, Any]) -> str:
        if int(row.get("blocked_count") or 0):
            return "blocked"
        if int(row.get("manual_check_count") or 0):
            return "manual_check"
        return "linked"

    def _registry_summary(
        consumers: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> dict[str, Any]:
        linked = len([item for item in consumers if _registry_status_bucket(item.get("status")) == "linked"])
        manual = len([item for item in consumers if _registry_status_bucket(item.get("status")) == "manual_check"])
        blocked = len([item for item in consumers if _registry_status_bucket(item.get("status")) == "blocked"])
        return {
            "overall_status": "blocked" if blocked or not consumers else "manual_check" if manual else "ready",
            "reference_count": len(references),
            "consumer_count": len(consumers),
            "linked_count": linked,
            "manual_check_count": manual,
            "blocked_count": blocked,
            "project_count": len({
                str(item.get("project_id") or "")
                for item in consumers
                if item.get("project_id")
            }),
            "template_count": len({
                str(item.get("template_id") or "")
                for item in consumers
                if item.get("template_id")
            }),
            "object_count": len({
                str(item.get("object_id") or "")
                for item in consumers
                if item.get("object_id")
            }),
        }

    def _resource_summary(
        resources: list[dict[str, Any]],
        consumers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unregistered = [item for item in resources if item.get("status") == "unregistered"]
        used = [item for item in resources if int(item.get("consumer_count") or 0) > 0]
        resource_types = sorted({
            str(item.get("resource_type") or "")
            for item in resources
            if item.get("resource_type")
        })
        return {
            "overall_status": "manual_check" if unregistered else "ready",
            "resource_count": len(resources),
            "used_resource_count": len(used),
            "consumer_count": len(consumers),
            "unregistered_resource_count": len(unregistered),
            "project_consumer_count": len([item for item in consumers if item.get("scope_type") == "project"]),
            "template_consumer_count": len([item for item in consumers if item.get("scope_type") == "template"]),
            "resource_types": resource_types,
        }

    def _scope_item_from_site(site: dict[str, Any]) -> dict[str, Any]:
        customer = site.get("customer") if isinstance(site.get("customer"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

    def _scope_item_from_detail(payload: dict[str, Any]) -> dict[str, Any]:
        customer = payload.get("customer") if isinstance(payload.get("customer"), dict) else {}
        site = payload.get("site") if isinstance(payload.get("site"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

    def _scope_item_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
        customer = profile.get("customer") if isinstance(profile.get("customer"), dict) else {}
        site = profile.get("site") if isinstance(profile.get("site"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

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
            "tenant_id": scope_item.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": scope_item.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": scope_item.get("customer_id") or "",
            "project_id": scope_item.get("project_id") or "",
            "site_id": scope_item.get("site_id") or "",
        }
        payload["project_scope"] = project_scope
        for key, value in project_scope.items():
            if value:
                payload.setdefault(key, value)
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
            "customer_status": "Lab rehearsal only; not production go-live evidence.",
            "release_claim": (
                "This rehearsal proves adapter parsing, object binding, and ingest contract shape. "
                "Customer signoff still needs onsite live evidence, trusted device signatures, "
                "event archive records, and runtime callback evidence."
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
                "Dry-run only proves adapter shape. Onsite acceptance needs a confirmed "
                "shadow/live post and real delivery evidence."
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
        trust = (
            normalized.get("device_trust")
            if isinstance(normalized.get("device_trust"), dict)
            else {}
        )
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
        runtime_delivery = (
            event.get("runtime_delivery")
            if isinstance(event.get("runtime_delivery"), dict)
            else {}
        )
        signature_verified = trust.get("signature_verified") is True
        runtime_completed = str(runtime_delivery.get("status") or "") in {
            "completed",
            "delivered",
            "recorded",
        }
        status = "passed" if signature_verified and runtime_completed else "manual_check"
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
            "status": status,
            "evidence": {
                "evidence_type": "device_ingest",
                "status": status,
                "source": "object_execution_rehearsal",
                "label": f"Object rehearsal shadow post: {plan.get('display_name') or object_id_value}",
                "summary": (
                    "Confirmed shadow-post rehearsal created a field event. "
                    "It is an onsite acceptance candidate; production release still requires "
                    "trusted signatures, runtime completion callbacks, and delivery review."
                ),
                "managed_object_id": object_id_value,
                "event_id": str(event.get("event_id") or ingest_result.get("event_id") or ""),
                "runtime_run_id": str(runtime_delivery.get("run_id") or runtime_delivery.get("runtime_run_id") or ""),
                "external_reference": f"object-rehearsal:shadow_post:{object_id_value}",
                "operator_id": operator_id,
                "reason": "Register object shadow-post rehearsal as an onsite evidence candidate.",
                "evidence_tier": "acceptance_candidate",
                "production_eligible": False,
            },
            "eligibility": {
                "signature_verified": signature_verified,
                "runtime_completed": runtime_completed,
                "missing_for_passed": missing,
            },
            "production_eligible": False,
            "customer_status": (
                "Registered as an acceptance candidate. It is not production go-live evidence "
                "until real device trust and runtime callback gates pass."
            ),
        }

    def _scope_item_from_create_body(body: dict[str, Any]) -> dict[str, Any]:
        customer = body.get("customer") if isinstance(body.get("customer"), dict) else {}
        site = body.get("site") if isinstance(body.get("site"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

    def _scope_item_from_package(payload: dict[str, Any]) -> dict[str, Any]:
        package = payload.get("package") if isinstance(payload.get("package"), dict) else {}
        customer = package.get("customer") if isinstance(package.get("customer"), dict) else {}
        site = package.get("site") if isinstance(package.get("site"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

    def _scope_item_from_dossier(payload: dict[str, Any]) -> dict[str, Any]:
        dossier = payload.get("dossier") if isinstance(payload.get("dossier"), dict) else {}
        customer = dossier.get("customer") if isinstance(dossier.get("customer"), dict) else {}
        site = dossier.get("site") if isinstance(dossier.get("site"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

    def _scope_item_from_proposal(payload: dict[str, Any]) -> dict[str, Any]:
        proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else payload
        customer = proposal.get("customer") if isinstance(proposal.get("customer"), dict) else {}
        site = proposal.get("site") if isinstance(proposal.get("site"), dict) else {}
        return {
            "tenant_id": customer.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "site_id": site.get("site_id") or "",
        }

    def _scope_item_from_resource(payload: dict[str, Any]) -> dict[str, Any]:
        explicit_scope = (
            payload.get("project_scope")
            if isinstance(payload.get("project_scope"), dict)
            else {}
        )
        return {
            "tenant_id": payload.get("tenant_id") or explicit_scope.get("tenant_id") or "",
            "delivery_namespace": (
                payload.get("delivery_namespace")
                or explicit_scope.get("delivery_namespace")
                or ""
            ),
            "customer_id": payload.get("customer_id") or explicit_scope.get("customer_id") or "",
            "project_id": payload.get("project_id") or explicit_scope.get("project_id") or "",
            "site_id": payload.get("site_id") or explicit_scope.get("site_id") or "",
        }

    def _resource_has_explicit_scope(payload: dict[str, Any]) -> bool:
        item = _scope_item_from_resource(payload)
        return any(str(item.get(key) or "").strip() for key in item)

    def _customer_rows_for_projects(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for project in projects:
            customer_id = str(project.get("customer_id") or "")
            if not customer_id:
                continue
            tenant_id = str(project.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE)
            delivery_namespace = str(project.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE)
            row_key = f"{tenant_id}/{delivery_namespace}/{customer_id}"
            row = rows.setdefault(
                row_key,
                {
                    "tenant_id": tenant_id,
                    "delivery_namespace": delivery_namespace,
                    "customer_id": customer_id,
                    "customer_name": str(project.get("customer_name") or customer_id),
                    "project_count": 0,
                    "projects": [],
                    "industries": [],
                },
            )
            row["project_count"] += 1
            row["projects"].append(str(project.get("project_id") or ""))
            industry = str(project.get("industry") or "")
            if industry and industry not in row["industries"]:
                row["industries"].append(industry)
        return list(rows.values())

    @app.get("/api/field/scenarios", tags=["Field Operations"])
    async def field_scenarios() -> JSONResponse:
        """Return customer-visible field operation scenarios."""
        try:
            result = await dispatch_field_operations("scenarios_payload")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field scenarios endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/events", tags=["Field Operations"])
    async def field_events(
        request: Request,
        limit: int = 50,
        status: str = "",
        notification_group: str = "",
        needs_attention: bool = False,
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        managed_object_id: str = "",
    ) -> JSONResponse:
        """Return recent field operation events."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            customer_allowed, customer_id = _scoped_query_value(customer_id, scope, "customer_ids")
            project_allowed, project_id = _scoped_query_value(project_id, scope, "project_ids")
            site_allowed, site_id = _scoped_query_value(site_id, scope, "site_ids")
            if not (customer_allowed and project_allowed and site_allowed):
                return _project_scope_forbidden()
            result = await dispatch_field_operations(
                "list_payload",
                limit=limit,
                status=status or None,
                notification_group=notification_group or None,
                needs_attention=needs_attention,
                customer_id=customer_id or None,
                project_id=project_id or None,
                site_id=site_id or None,
                managed_object_id=managed_object_id or None,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field events endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/events/{event_id}", tags=["Field Operations"])
    async def field_event_detail(event_id: str, request: Request) -> JSONResponse:
        """Return one field operation event with workflow and evidence detail."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            result = await dispatch_field_operations("detail_payload", event_id)
            if result.get("found") and not _scope_allows(scope, _scope_item_from_event_detail(result)):
                return _project_scope_forbidden()
            status_code = 200 if result.get("found") else 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event detail endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/evidence", tags=["Field Operations"], response_model=None)
    async def field_evidence(path: str) -> Response:
        """Serve a local field evidence artifact from approved evidence roots."""
        resolved = _resolve_field_evidence_path(path)
        if resolved is None:
            return mission_json({"error": "field evidence not found"}, status_code=404)
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

    @app.post("/api/field/events", tags=["Field Operations"])
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
            scope = _operator_project_scope(body)
            if _has_explicit_project_scope(body) and not _scope_allows(
                scope,
                _scope_item_from_event_payload(body),
            ):
                return _project_scope_forbidden()
            _apply_single_scope_defaults(body, scope)
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event trigger endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/close", tags=["Field Operations"])
    async def field_event_close(event_id: str, request: Request) -> JSONResponse:
        """Close a field operation event with an operator note."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:close")
            if failure is not None:
                return failure
            scope_failure = await _field_event_scope_failure(
                event_id,
                _operator_project_scope(body),
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event close endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/request-close", tags=["Field Operations"])
    async def field_event_request_close(event_id: str, request: Request) -> JSONResponse:
        """Request supervisor approval before closing a high-risk field event."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:request_close")
            if failure is not None:
                return failure
            scope_failure = await _field_event_scope_failure(
                event_id,
                _operator_project_scope(body),
            )
            if scope_failure is not None:
                return scope_failure
            result = await dispatch_field_operations("request_close_payload", event_id, body)
            status_code = 200 if result.get("requested") else 404
            if result.get("reason") in {"event_already_closed", "event_not_closable"}:
                status_code = 409
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event close request endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/acknowledge", tags=["Field Operations"])
    async def field_event_acknowledge(event_id: str, request: Request) -> JSONResponse:
        """Acknowledge a field operation event without closing it."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:acknowledge")
            if failure is not None:
                return failure
            scope_failure = await _field_event_scope_failure(
                event_id,
                _operator_project_scope(body),
            )
            if scope_failure is not None:
                return scope_failure
            result = await dispatch_field_operations("acknowledge_payload", event_id, body)
            status_code = 200 if result.get("acknowledged") else 409
            if result.get("reason") == "event_not_found":
                status_code = 404
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event acknowledge endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/resend-notification", tags=["Field Operations"])
    async def field_event_resend_notification(event_id: str, request: Request) -> JSONResponse:
        """Retry notification delivery for an open field operation event."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:event:acknowledge")
            if failure is not None:
                return failure
            scope_failure = await _field_event_scope_failure(
                event_id,
                _operator_project_scope(body),
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event notification resend endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/events/{event_id}/report", tags=["Field Operations"])
    async def field_event_report(event_id: str, request: Request) -> JSONResponse:
        """Return an auditable customer-facing field event report."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            result = await dispatch_field_operations("event_report_payload", event_id)
            if result.get("found") and not _scope_allows(scope, _scope_item_from_event_detail(result.get("report", {}))):
                return _project_scope_forbidden()
            status_code = 200 if result.get("found") else 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event report endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/runtime-delivery", tags=["Field Operations"])
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field runtime-delivery endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/notification-test", tags=["Field Operations"])
    async def field_notification_test(request: Request) -> JSONResponse:
        """Send a low-risk notification smoke test to a responder group."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:notification:test")
            if failure is not None:
                return failure
            result = await dispatch_field_operations("test_notification_payload", body)
            status_code = 200 if result.get("status") != "invalid_group" else 422
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field notification test endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/notification-preflight", tags=["Field Operations"])
    async def field_notification_preflight(status_as_200: bool = False) -> JSONResponse:
        """Check whether real DingTalk responder notification credentials are configured."""
        try:
            result = await dispatch_field_operations("notification_preflight_payload")
            status_code = 200 if status_as_200 or result.get("ready") else 409
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field notification preflight endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/devices", tags=["Field Operations"])
    async def field_devices() -> JSONResponse:
        """Return registered and observed field-device trust/online status."""
        try:
            result = await dispatch_field_operations("device_status_payload")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field devices endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/ingest", tags=["Field Operations"])
    async def field_ingest_help() -> JSONResponse:
        """Return examples for raw camera/sensor/robot event ingestion."""
        try:
            result = await dispatch_field_operations("ingest_help_payload")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field ingest help endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/readiness", tags=["Field Operations"])
    async def field_readiness() -> JSONResponse:
        """Return deployment readiness gates for field operations."""
        try:
            result = await dispatch_field_operations("readiness_payload")
            return mission_json(result)
        except Exception as exc:
            logger.error("Field readiness endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/site-profiles", tags=["Field Operations"])
    async def field_site_profiles(request: Request, check_env: bool = False) -> JSONResponse:
        """Return the multi-site field deployment catalog."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = build_site_profile_catalog(root, check_env=check_env)
            result = _scope_site_catalog(result, scope)
            return mission_json(result)
        except Exception as exc:
            logger.error("Field site profile catalog endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects", tags=["Field Operations"])
    async def field_customer_projects(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> JSONResponse:
        """Return customer, project, site, and managed-object rollout coverage."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = build_customer_project_catalog(
                root,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            )
            result = _scope_project_catalog(result, scope)
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project catalog endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/managed-object-directory", tags=["Field Operations"])
    async def field_customer_project_managed_object_directory(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
        delivery_status: str = "",
        category: str = "",
        customer_visible: str = "",
    ) -> JSONResponse:
        """Return scoped managed-object bindings for delivery and acceptance review."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            catalog = build_customer_project_catalog(
                root,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            )
            catalog = _scope_project_catalog(catalog, scope)
            projects = [
                project
                for project in catalog.get("projects", [])
                if isinstance(project, dict)
            ]
            rows = _managed_object_directory_rows(projects)
            rows, object_filters = _filter_managed_object_directory_rows(
                rows,
                delivery_status=delivery_status,
                category=category,
                customer_visible=customer_visible,
            )
            base_summary = catalog.get("summary") if isinstance(catalog.get("summary"), dict) else {}
            summary = _managed_object_directory_summary(
                rows,
                projects=projects,
                base_summary=base_summary,
                filtered=bool(object_filters) or bool(base_summary.get("filtered")),
            )
            filters = dict(catalog.get("filters") or {})
            filters.update(object_filters)
            return mission_json(
                {
                    "directory_type": "askme.customer_project_managed_object_directory",
                    "root": catalog.get("root") or str(root),
                    "check_env": check_env,
                    "filters": filters,
                    "summary": summary,
                    "objects": rows,
                    "customer_status": (
                        "Managed object directory is scoped for this operator and ready for "
                        "delivery review."
                        if rows
                        else "No managed objects are visible for this operator scope and filter."
                    ),
                    "next_step": (
                        "Review blocked/manual-check objects before exporting a customer handoff."
                        if rows
                        else "Adjust the customer project filters or operator project scope."
                    ),
                }
            )
        except Exception as exc:
            logger.error("Field managed object directory endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-project-templates", tags=["Field Operations"])
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
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            result = list_customer_project_templates(
                Path("deploy/customer-project-templates"),
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
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project templates endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-project-templates/{template_id}/history", tags=["Field Operations"])
    async def field_customer_project_template_history(
        template_id: str,
        request: Request,
        limit: int = 20,
    ) -> JSONResponse:
        """Return release-governance history for one industry template."""
        try:
            failure, _auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            result = list_customer_project_template_revisions(
                Path("deploy/customer-project-templates"),
                template_id,
                limit=limit,
            )
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project template history endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-project-template-release-requests", tags=["Field Operations"])
    async def field_customer_project_template_release_requests(
        request: Request,
        template_id: str = "",
        status: str = "",
        limit: int = 50,
    ) -> JSONResponse:
        """Return reusable-template release requests for product-owner review."""
        try:
            failure, _auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            result = list_customer_project_template_release_requests(
                Path("deploy/customer-project-templates"),
                template_id=template_id,
                status=status,
                limit=limit,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project template release request list failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-project-template-release-notes", tags=["Field Operations"])
    async def field_customer_project_template_release_notes(
        request: Request,
        limit: int = 50,
    ) -> JSONResponse:
        """Return approved customer-facing reusable-template release notes."""
        try:
            failure, _auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            result = customer_project_template_release_notes(
                Path("deploy/customer-project-templates"),
                limit=limit,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project template release notes failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-project-template-release-notes/export", tags=["Field Operations"])
    async def field_customer_project_template_release_notes_export(request: Request) -> JSONResponse:
        """Return a portable proposal/handoff bundle for approved template release notes."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:read")
            if failure is not None:
                return failure
            result = export_customer_project_template_release_notes_bundle(
                Path("deploy/customer-project-templates"),
                customer_context=body.get("customer_context") if isinstance(body.get("customer_context"), dict) else body,
                limit=int(body.get("limit") or 50),
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project template release notes export failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-project-templates/{template_id}/release-requests", tags=["Field Operations"])
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
            release = body.get("release") if isinstance(body.get("release"), dict) else body
            result = create_customer_project_template_release_request(
                Path("deploy/customer-project-templates"),
                template_id,
                release,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or release.get("reason") or ""),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "template_not_found":
                status_code = 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field customer project template release request create failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-project-template-release-requests/{request_id}/review", tags=["Field Operations"])
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
            result = review_customer_project_template_release_request(
                Path("deploy/customer-project-templates"),
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field customer project template release request review failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-project-templates/{template_id}/release", tags=["Field Operations"])
    async def field_customer_project_template_release(
        template_id: str,
        request: Request,
    ) -> JSONResponse:
        """Promote, demote, or block a reusable industry template package."""
        try:
            body = await optional_json_body(request)
            release = body.get("release") if isinstance(body.get("release"), dict) else body
            publish_status = str(release.get("publish_status") or "").strip()
            required_permission = (
                "template:release:approve"
                if publish_status == "published"
                else "template:release:write"
            )
            failure = authorize(request, body, required_permission)
            if failure is not None:
                return failure
            if publish_status == "published":
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "published_release_requires_approval_request",
                        "template_id": template_id,
                        "next_step": (
                            "Create /release-requests first, then approve it with a second product owner."
                        ),
                    },
                    status_code=409,
                )
            result = update_customer_project_template_release(
                Path("deploy/customer-project-templates"),
                template_id,
                release,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or release.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "template_not_found":
                status_code = 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field customer project template release endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-project-acceptance-registry", tags=["Field Operations"])
    async def field_customer_project_acceptance_registry(request: Request) -> JSONResponse:
        """Return managed-object acceptance references across projects and templates."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = build_customer_project_acceptance_registry(
                root,
                template_root=Path("deploy/customer-project-templates"),
            )
            result = _scope_acceptance_registry(result, scope)
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project acceptance registry endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-project-resource-catalog", tags=["Field Operations"])
    async def field_customer_project_resource_catalog(request: Request) -> JSONResponse:
        """Return model, protocol, skill, and acceptance bindings used by projects."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            result = build_customer_project_resource_catalog(
                site_profile_root or Path("deploy/site-profiles"),
                template_root=Path("deploy/customer-project-templates"),
            )
            result = _scope_resource_catalog(result, scope)
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project resource catalog endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/solution-delivery-readiness", tags=["Field Operations"])
    async def field_solution_delivery_readiness(
        request: Request,
        check_env: bool = False,
    ) -> JSONResponse:
        """Return one product-facing readiness gate for solution-provider delivery."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            project_root = site_profile_root or Path("deploy/site-profiles")
            template_root = Path("deploy/customer-project-templates")
            resource_root = Path("deploy/delivery-resources")
            project_catalog = _scope_project_catalog(
                build_customer_project_catalog(project_root, check_env=check_env),
                scope,
            )
            template_catalog = _scope_template_catalog(
                list_customer_project_templates(template_root),
                scope,
            )
            resource_catalog = _scope_resource_catalog(
                build_customer_project_resource_catalog(
                    project_root,
                    template_root=template_root,
                    delivery_resource_root=resource_root,
                ),
                scope,
            )
            if any(scope.values()):
                governance_requests = {
                    "skipped": True,
                    "reason": "resource_governance_requests_require_unrestricted_operator",
                }
            else:
                governance_requests = list_delivery_resource_governance_requests(
                    resource_root,
                    limit=20,
                )
            result = build_solution_delivery_readiness(
                project_catalog=project_catalog,
                template_catalog=template_catalog,
                resource_catalog=resource_catalog,
                governance_requests=governance_requests,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field solution delivery readiness endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    def _customer_project_workbench_payload(
        *,
        scope: dict[str, list[str]],
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> dict[str, Any]:
        project_root = site_profile_root or Path("deploy/site-profiles")
        template_root = Path("deploy/customer-project-templates")
        resource_root = Path("deploy/delivery-resources")
        project_catalog = _scope_project_catalog(
            build_customer_project_catalog(
                project_root,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            ),
            scope,
        )
        template_catalog = _scope_template_catalog(
            list_customer_project_templates(template_root),
            scope,
        )
        resource_catalog = _scope_resource_catalog(
            build_customer_project_resource_catalog(
                project_root,
                template_root=template_root,
                delivery_resource_root=resource_root,
            ),
            scope,
        )
        projects = [
            project
            for project in project_catalog.get("projects", [])
            if isinstance(project, dict)
        ]
        object_rows = _managed_object_directory_rows(projects)
        object_summary = _managed_object_directory_summary(
            object_rows,
            projects=projects,
            base_summary=(
                project_catalog.get("summary")
                if isinstance(project_catalog.get("summary"), dict)
                else {}
            ),
            filtered=bool(project_catalog.get("filters")),
        )
        governance_requests = (
            {
                "skipped": True,
                "reason": "resource_governance_requests_require_unrestricted_operator",
            }
            if any(scope.values())
            else list_delivery_resource_governance_requests(resource_root, limit=20)
        )
        readiness = build_solution_delivery_readiness(
            project_catalog=project_catalog,
            template_catalog=template_catalog,
            resource_catalog=resource_catalog,
            governance_requests=governance_requests,
        )
        surfaces = [
            {
                "surface_id": "customer_projects",
                "label": "客户项目目录",
                "customer_label": "客户项目目录",
                "customer_description": "按客户、项目、现场和交付阶段管理项目范围。",
                "customer_count_label": "项目",
                "customer_action": "选择客户项目并核对对象范围。",
                "status": project_catalog.get("summary", {}).get(
                    "delivery_acceptance_gate_status",
                    "unknown",
                ),
                "count": project_catalog.get("summary", {}).get("project_count", 0),
                "api": "/api/field/customer-projects",
            },
            {
                "surface_id": "template_market",
                "label": "行业模板市场",
                "customer_label": "行业模板市场",
                "customer_description": "提供厂区、园区、仓储、景区等可复用方案模板。",
                "customer_count_label": "模板",
                "customer_action": "从合适模板创建客户项目。",
                "status": template_catalog.get("summary", {}).get("overall_status", "unknown"),
                "count": template_catalog.get("summary", {}).get("template_count", 0),
                "api": "/api/field/customer-project-templates",
            },
            {
                "surface_id": "managed_objects",
                "label": "对象目录",
                "customer_label": "对象目录",
                "customer_description": "展示车辆、设备、游客、烟火、垃圾桶等现场对象及能力配置。",
                "customer_count_label": "对象",
                "customer_action": "补齐对象的识别、设备接入、业务能力和验收项。",
                "status": object_summary.get("overall_status", "unknown"),
                "count": object_summary.get("object_count", 0),
                "api": "/api/field/customer-projects/managed-object-directory",
            },
            {
                "surface_id": "delivery_resources",
                "label": "交付资源",
                "customer_label": "交付资源",
                "customer_description": "统一检查识别模型、设备接入方式、业务能力和验收项是否可交付。",
                "customer_count_label": "资源",
                "customer_action": "替换未注册或被阻断的交付资源。",
                "status": resource_catalog.get("summary", {}).get("overall_status", "unknown"),
                "count": resource_catalog.get("summary", {}).get("resource_count", 0),
                "api": "/api/field/customer-project-resource-catalog",
            },
            {
                "surface_id": "package_delivery_gate",
                "label": "交付包准入",
                "customer_label": "交付包准入",
                "customer_description": "导出、导入前统一检查范围、对象、资源、证据和验收风险。",
                "customer_count_label": "项目",
                "customer_action": "通过预检后再导出或导入客户项目包。",
                "status": readiness.get("overall_status", "unknown"),
                "count": readiness.get("summary", {}).get("project_count", 0),
                "api": "/api/field/customer-projects/{identifier}/export",
            },
        ]
        return {
            "workbench_type": "askme.solution_provider_customer_project_workbench.v1",
            "overall_status": readiness.get("overall_status", "unknown"),
            "customer_status": readiness.get("customer_status", ""),
            "release_claim": readiness.get("release_claim", ""),
            "next_step": readiness.get("next_step", ""),
            "scope_filtered": any(scope.values()),
            "filters": project_catalog.get("filters") or {},
            "delivery_surfaces": surfaces,
            "customer_vocabulary": _customer_project_term_cards(),
            "customer_acceptance_flow": _CUSTOMER_PROJECT_ACCEPTANCE_FLOW,
            "customer_readable_contract": {
                "contract_type": "askme.solution_provider_customer_delivery_contract.v1",
                "positioning": "面向多客户、多行业现场的可复用机器人方案交付平台。",
                "customer_can_verify": [
                    "客户项目是否按客户范围隔离",
                    "行业模板是否能复制成新项目",
                    "对象目录是否覆盖客户购买的现场对象",
                    "每个对象是否绑定识别能力、设备接入方式、业务能力和验收项",
                    "交付包导出、预检、导入、报告是否完整",
                ],
                "not_claimed": [
                    "不把演示或试点状态承诺为无人值守生产上线",
                    "不让客户阅读接口、YAML 或测试路径来理解验收结论",
                    "不绕过现场验收、权限隔离和交付包准入检查",
                ],
            },
            "solution_delivery_readiness": readiness,
            "customer_projects": {
                "summary": project_catalog.get("summary") or {},
                "project_count": len(projects),
                "projects": projects[:20],
            },
            "template_market": {
                "summary": template_catalog.get("summary") or {},
                "templates": (
                    template_catalog.get("templates", [])
                    if isinstance(template_catalog.get("templates"), list)
                    else []
                )[:20],
            },
            "managed_object_directory": {
                "summary": object_summary,
                "objects": object_rows[:50],
            },
            "delivery_resources": {
                "summary": resource_catalog.get("summary") or {},
                "resources": (
                    resource_catalog.get("resources", [])
                    if isinstance(resource_catalog.get("resources"), list)
                    else []
                )[:50],
            },
        }

    @app.get("/api/field/customer-project-workbench", tags=["Field Operations"])
    async def field_customer_project_workbench(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> JSONResponse:
        """Return one solution-provider workbench payload for customer delivery."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            return mission_json(
                _customer_project_workbench_payload(
                    scope=scope,
                    check_env=check_env,
                    tenant_id=tenant_id,
                    delivery_namespace=delivery_namespace,
                    customer_id=customer_id,
                    project_id=project_id,
                    site_id=site_id,
                    industry=industry,
                    gate_status=gate_status,
                    deployment_stage=deployment_stage,
                )
            )
        except Exception as exc:
            logger.error("Field customer project workbench endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/product-launch-readiness", tags=["Field Operations"])
    async def field_product_launch_readiness(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> JSONResponse:
        """Return one customer-facing launch decision across product gates."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            field_readiness = await dispatch_field_operations("readiness_payload")
            workbench = _customer_project_workbench_payload(
                scope=scope,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            )
            result = build_product_launch_readiness(
                identity_readiness=identity_readiness_payload(),
                field_readiness=field_readiness,
                solution_delivery_readiness=workbench.get("solution_delivery_readiness", {}),
                customer_project_workbench=workbench,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field product launch readiness endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/delivery-resource-registry", tags=["Field Operations"])
    async def field_delivery_resource_registry(request: Request) -> JSONResponse:
        """Return shared delivery resources that project objects can bind to."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            result = list_delivery_resource_registry(Path("deploy/delivery-resources"))
            result = _scope_delivery_resource_registry(
                result,
                _operator_project_scope(auth_body),
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource registry endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/delivery-resource-registry", tags=["Field Operations"])
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
            scope = _operator_project_scope(body)
            _apply_single_scope_defaults(metadata, scope)
            if any(scope.values()) and not _resource_has_explicit_scope(metadata):
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "resource_scope_required",
                        "message": (
                            "Scoped operators must register resources with tenant, namespace, "
                            "customer, project, or site scope."
                        ),
                    },
                    status_code=403,
                )
            if not _scope_allows(scope, _scope_item_from_resource(metadata)):
                return _project_scope_forbidden()
            result = upsert_delivery_resource(
                Path("deploy/delivery-resources"),
                resource_type,
                resource_id,
                metadata,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                overwrite=bool(body.get("overwrite", True)),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field delivery resource register endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/delivery-resource-registry/history", tags=["Field Operations"])
    async def field_delivery_resource_history(request: Request, limit: int = 20) -> JSONResponse:
        """Return shared delivery-resource registry revisions for audit review."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            if any(_operator_project_scope(auth_body).values()):
                return mission_json(
                    {
                        "error": "resource registry history requires unrestricted operator scope",
                        "reason": "resource_registry_history_requires_unrestricted_operator",
                    },
                    status_code=403,
                )
            result = list_delivery_resource_revisions(
                Path("deploy/delivery-resources"),
                limit=limit,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource history endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable",
        tags=["Field Operations"],
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
            scope = _operator_project_scope(body)
            registry = list_delivery_resource_registry(Path("deploy/delivery-resources"))
            resource = next(
                (
                    item
                    for item in registry.get("resources", [])
                    if isinstance(item, dict)
                    and str(item.get("resource_type") or "") == resource_type
                    and str(item.get("resource_id") or "") == resource_id
                ),
                None,
            )
            if resource is not None:
                if any(scope.values()) and not _resource_has_explicit_scope(resource):
                    return mission_json(
                        {
                            "accepted": False,
                            "reason": "resource_scope_required",
                            "message": "Scoped operators cannot mutate global shared resources.",
                        },
                        status_code=403,
                    )
                if not _scope_allows(scope, _scope_item_from_resource(resource)):
                    return _project_scope_forbidden()
            result = disable_delivery_resource(
                Path("deploy/delivery-resources"),
                resource_type,
                resource_id,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "resource_not_found":
                status_code = 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field delivery resource disable endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/delivery-resource-registry/rollback", tags=["Field Operations"])
    async def field_delivery_resource_rollback(request: Request) -> JSONResponse:
        """Rollback the shared delivery-resource registry to a previous revision."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "resource:governance:approve")
            if failure is not None:
                return failure
            if any(_operator_project_scope(body).values()):
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "resource_registry_rollback_requires_unrestricted_operator",
                        "message": "Registry rollback can affect multiple customers and requires unrestricted scope.",
                    },
                    status_code=403,
                )
            result = rollback_delivery_resource_registry(
                Path("deploy/delivery-resources"),
                str(body.get("revision_id") or ""),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "revision_not_found":
                status_code = 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field delivery resource rollback endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/delivery-resource-governance-requests", tags=["Field Operations"])
    async def field_delivery_resource_governance_requests(
        request: Request,
        status: str = "",
        action: str = "",
        limit: int = 50,
        overdue_only: bool = False,
    ) -> JSONResponse:
        """Return pending and reviewed shared-resource governance requests."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            if any(_operator_project_scope(auth_body).values()):
                return mission_json(
                    {
                        "error": "resource governance requests require unrestricted operator scope",
                        "reason": "resource_governance_requests_require_unrestricted_operator",
                    },
                    status_code=403,
                )
            result = list_delivery_resource_governance_requests(
                Path("deploy/delivery-resources"),
                status=status,
                action=action,
                limit=limit,
                overdue_only=overdue_only,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource governance request list endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/delivery-resource-governance-requests", tags=["Field Operations"])
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
            scope = _operator_project_scope(body)
            if action == "rollback_registry" and any(scope.values()):
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "resource_registry_rollback_requires_unrestricted_operator",
                        "message": "Registry rollback can affect multiple customers and requires unrestricted scope.",
                    },
                    status_code=403,
                )
            if action == "disable_resource" and any(scope.values()):
                registry = list_delivery_resource_registry(Path("deploy/delivery-resources"))
                resource_type = str(operation.get("resource_type") or "")
                resource_id = str(operation.get("resource_id") or "")
                resource = next(
                    (
                        item
                        for item in registry.get("resources", [])
                        if isinstance(item, dict)
                        and str(item.get("resource_type") or "") == resource_type
                        and str(item.get("resource_id") or "") == resource_id
                    ),
                    None,
                )
                if resource is not None:
                    if not _resource_has_explicit_scope(resource):
                        return mission_json(
                            {
                                "accepted": False,
                                "reason": "resource_scope_required",
                                "message": "Scoped operators cannot request global shared resource mutation.",
                            },
                            status_code=403,
                        )
                    if not _scope_allows(scope, _scope_item_from_resource(resource)):
                        return _project_scope_forbidden()
            result = create_delivery_resource_governance_request(
                Path("deploy/delivery-resources"),
                action,
                operation,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                sla_target_s=body.get("sla_target_s") or operation.get("sla_target_s"),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "resource_not_found":
                status_code = 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field delivery resource governance request create endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-governance-requests/{request_id}/review",
        tags=["Field Operations"],
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
            if any(_operator_project_scope(body).values()):
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "resource_governance_review_requires_unrestricted_operator",
                        "message": "Resource governance approval can affect multiple customers.",
                    },
                    status_code=403,
                )
            result = review_delivery_resource_governance_request(
                Path("deploy/delivery-resources"),
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field delivery resource governance request review endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/delivery-resource-governance-requests/escalate-overdue",
        tags=["Field Operations"],
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
            if any(_operator_project_scope(body).values()):
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "resource_governance_escalation_requires_unrestricted_operator",
                        "message": "Resource governance escalation can affect multiple customers.",
                    },
                    status_code=403,
                )
            result = escalate_overdue_delivery_resource_governance_requests(
                Path("deploy/delivery-resources"),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                limit=int(body.get("limit") or 50),
                dry_run=bool(body.get("dry_run")),
                notification_delivery=_resource_governance_notification_delivery,
            )
            return mission_json(result)
        except Exception as exc:
            logger.error("Field delivery resource governance overdue escalation endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}", tags=["Field Operations"])
    async def field_customer_project_detail(
        identifier: str,
        request: Request,
        check_env: bool = False,
    ) -> JSONResponse:
        """Return one customer project profile with managed-object bindings."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = get_customer_project_profile(root, identifier, check_env=check_env)
            if result.get("found") and not _scope_allows(scope, _scope_item_from_detail(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project detail endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/acceptance-report", tags=["Field Operations"])
    async def field_customer_project_acceptance_report(
        identifier: str,
        request: Request,
        check_env: bool = True,
    ) -> JSONResponse:
        """Return a customer-readable project acceptance report."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = customer_project_acceptance_report(root, identifier, check_env=check_env)
            if result.get("found") and not _scope_allows(scope, _scope_item_from_detail(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project acceptance report endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/execution-bindings", tags=["Field Operations"])
    async def field_customer_project_execution_bindings(
        identifier: str,
        request: Request,
    ) -> JSONResponse:
        """Return executable ingest/runtime binding plans for one customer project."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = build_customer_project_execution_bindings(
                root,
                identifier,
                delivery_resource_root=Path("deploy/delivery-resources"),
            )
            if result.get("found") and not _scope_allows(scope, _scope_item_from_detail(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project execution bindings endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal",
        tags=["Field Operations"],
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
            scope = _operator_project_scope(body)
            root = site_profile_root or Path("deploy/site-profiles")
            bindings = build_customer_project_execution_bindings(
                root,
                identifier,
                delivery_resource_root=Path("deploy/delivery-resources"),
            )
            if not bindings.get("found"):
                return mission_json(bindings, status_code=404)
            scope_item = _scope_item_from_detail(bindings)
            if not _scope_allows(scope, scope_item):
                return _project_scope_forbidden()

            plans_by_object = (
                bindings.get("plans_by_object_id")
                if isinstance(bindings.get("plans_by_object_id"), dict)
                else {}
            )
            plan = plans_by_object.get(object_id)
            if not isinstance(plan, dict):
                return mission_json(
                    {
                        "accepted": False,
                        "status": "not_found",
                        "reason": "managed_object_not_found",
                        "object_id": object_id,
                    },
                    status_code=404,
                )

            mode = str(body.get("mode") or "dry_run").strip() or "dry_run"
            if mode not in {"dry_run", "shadow_post"}:
                return mission_json(
                    {
                        "accepted": False,
                        "status": "rejected",
                        "reason": "invalid_rehearsal_mode",
                        "allowed_modes": ["dry_run", "shadow_post"],
                    },
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
                                "Confirm shadow_post only in lab or onsite rehearsal windows. "
                                "Use dry_run when external notifications may be configured."
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
                            reason="Register object shadow-post rehearsal as onsite evidence candidate.",
                        )
                        registration["registered"] = bool(write_result.get("accepted"))
                        registration["receipt"] = write_result.get("receipt") or {}
                        registration["onsite_acceptance_evidence"] = (
                            write_result.get("onsite_acceptance_evidence") or {}
                        )
                        if not write_result.get("accepted"):
                            registration["accepted"] = False
                            registration["reason"] = str(write_result.get("reason") or "onsite_evidence_write_failed")
                    result["onsite_evidence_registration"] = registration
            return mission_json(result)
        except Exception as exc:
            logger.error("Field customer project object rehearsal endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/onsite-evidence", tags=["Field Operations"])
    async def field_customer_project_onsite_evidence(
        identifier: str,
        request: Request,
        check_env: bool = True,
        include_readiness_auto: bool = True,
    ) -> JSONResponse:
        """Return onsite acceptance evidence receipts for one customer project."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = list_customer_project_onsite_evidence(
                root,
                identifier,
                check_env=check_env,
                include_readiness_auto=include_readiness_auto,
            )
            if result.get("found") and not _scope_allows(scope, _scope_item_from_detail(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project onsite evidence endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/{identifier}/onsite-evidence", tags=["Field Operations"])
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
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            evidence = body.get("evidence") if isinstance(body.get("evidence"), dict) else body
            result = register_customer_project_onsite_evidence(
                root,
                identifier,
                evidence,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field customer project onsite evidence register endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/acceptance-closure", tags=["Field Operations"])
    async def field_customer_project_acceptance_closure(
        identifier: str,
        request: Request,
        check_env: bool = True,
    ) -> JSONResponse:
        """Return a customer-readable acceptance closure summary."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = customer_project_acceptance_closure(root, identifier, check_env=check_env)
            if result.get("found") and not _scope_allows(scope, _scope_item_from_detail(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project acceptance closure endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/{identifier}/acceptance-review", tags=["Field Operations"])
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
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            review = body.get("review") if isinstance(body.get("review"), dict) else body
            result = register_customer_project_acceptance_review(
                root,
                identifier,
                review,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field customer project acceptance review endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/customer-signoff", tags=["Field Operations"])
    async def field_customer_project_customer_signoff(identifier: str, request: Request) -> JSONResponse:
        """Return customer signoff records for one customer project."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            root = site_profile_root or Path("deploy/site-profiles")
            result = list_customer_project_customer_signoffs(root, identifier)
            if result.get("found") and not _scope_allows(
                _operator_project_scope(auth_body),
                _scope_item_from_detail(result),
            ):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project signoff endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/{identifier}/customer-signoff", tags=["Field Operations"])
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
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            signoff = body.get("signoff") if isinstance(body.get("signoff"), dict) else body
            result = register_customer_project_customer_signoff(
                root,
                identifier,
                signoff,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") == "profile_not_found":
                status_code = 404
            if result.get("reason") == "project_not_ready_for_customer_signoff":
                status_code = 409
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field customer project signoff register endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/from-template", tags=["Field Operations"])
    async def field_customer_project_from_template(request: Request) -> JSONResponse:
        """Create a customer project profile from an industry template."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            scope = _operator_project_scope(body)
            if not _scope_allows(scope, _scope_item_from_create_body(body)):
                return _project_scope_forbidden()
            root = site_profile_root or Path("deploy/site-profiles")
            result = create_customer_project_from_template(
                template_root=Path("deploy/customer-project-templates"),
                profile_root=root,
                template_id=str(body.get("template_id") or ""),
                customer=body.get("customer") if isinstance(body.get("customer"), dict) else {},
                site=body.get("site") if isinstance(body.get("site"), dict) else {},
                overwrite=bool(body.get("overwrite")),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field customer project template create endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects", tags=["Field Operations"])
    async def field_customer_project_upsert(request: Request) -> JSONResponse:
        """Create or update a customer project profile from an explicit payload."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root or Path("deploy/site-profiles")
            profile = body.get("profile") if isinstance(body.get("profile"), dict) else body
            scope = _operator_project_scope(body)
            if not _scope_allows(scope, _scope_item_from_profile(profile)):
                return _project_scope_forbidden()
            result = upsert_customer_project_profile(
                root,
                profile,
                overwrite=bool(body.get("overwrite", True)),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field customer project upsert endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/import", tags=["Field Operations"])
    async def field_customer_project_import(request: Request) -> JSONResponse:
        """Import a customer project package generated by the export endpoint."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root or Path("deploy/site-profiles")
            package = body.get("package") if isinstance(body.get("package"), dict) else body
            scope = _operator_project_scope(body)
            package_scope = _scope_item_from_package({"package": package})
            if not _scope_allows(scope, package_scope):
                return _project_scope_forbidden()
            result = import_customer_project_package(
                root,
                package,
                overwrite=bool(body.get("overwrite")),
                dry_run=bool(body.get("dry_run")),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            result.setdefault("package_scope", package_scope)
            result.setdefault("operator_project_scope", scope)
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field customer project import endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/package/verify", tags=["Field Operations"])
    async def field_customer_project_package_verify(request: Request) -> JSONResponse:
        """Verify a customer project handoff package without writing anything."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:read")
            if failure is not None:
                return failure
            package = body.get("package") if isinstance(body.get("package"), dict) else body
            scope = _operator_project_scope(body)
            package_scope = _scope_item_from_package({"package": package})
            if isinstance(package, dict) and not _scope_allows(scope, package_scope):
                return _project_scope_forbidden()
            verification = verify_customer_project_package(package)
            return mission_json(
                {
                    "accepted": bool(verification.get("valid")),
                    "verification": verification,
                    "package_scope": package_scope,
                    "operator_project_scope": scope,
                },
                status_code=200 if verification.get("valid") else 422,
            )
        except Exception as exc:
            logger.error("Field customer project package verify endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/package/diff", tags=["Field Operations"])
    async def field_customer_project_package_diff(request: Request) -> JSONResponse:
        """Preview how a customer project handoff package would change local profiles."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:read")
            if failure is not None:
                return failure
            package = body.get("package") if isinstance(body.get("package"), dict) else body
            scope = _operator_project_scope(body)
            package_scope = _scope_item_from_package({"package": package})
            if isinstance(package, dict) and not _scope_allows(scope, package_scope):
                return _project_scope_forbidden()
            verification = verify_customer_project_package(package)
            if not verification.get("valid"):
                return mission_json(
                    {
                        "accepted": False,
                        "reason": "package_integrity_check_failed",
                        "verification": verification,
                        "package_scope": package_scope,
                        "operator_project_scope": scope,
                    },
                    status_code=422,
                )
            root = site_profile_root or Path("deploy/site-profiles")
            diff = diff_customer_project_package(root, package)
            incoming_delivery_gate = (
                diff.get("incoming_delivery_gate")
                if isinstance(diff.get("incoming_delivery_gate"), dict)
                else {}
            )
            return mission_json(
                {
                    "accepted": True,
                    "verification": verification,
                    "diff": diff,
                    "package_scope": package_scope,
                    "operator_project_scope": scope,
                    "would_write": bool(
                        diff.get("change_type") in {"create", "replace"}
                        and incoming_delivery_gate.get("import_allowed", True)
                    ),
                },
            )
        except Exception as exc:
            logger.error("Field customer project package diff endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/proposal-bundle/verify", tags=["Field Operations"])
    async def field_customer_project_proposal_bundle_verify(request: Request) -> JSONResponse:
        """Verify a customer project proposal bundle without writing anything."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:read")
            if failure is not None:
                return failure
            proposal = body.get("proposal") if isinstance(body.get("proposal"), dict) else body
            scope = _operator_project_scope(body)
            if isinstance(proposal, dict) and not _scope_allows(scope, _scope_item_from_proposal(proposal)):
                return _project_scope_forbidden()
            verification = verify_customer_project_proposal_bundle(proposal)
            return mission_json(
                {
                    "accepted": bool(verification.get("valid")),
                    "verification": verification,
                    "proposal_scope": _scope_item_from_proposal(proposal),
                },
                status_code=200 if verification.get("valid") else 422,
            )
        except Exception as exc:
            logger.error("Field customer project proposal bundle verify endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/acceptance-dossier/verify", tags=["Field Operations"])
    async def field_customer_project_acceptance_dossier_verify(request: Request) -> JSONResponse:
        """Verify a customer project acceptance dossier without writing anything."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:read")
            if failure is not None:
                return failure
            dossier = body.get("dossier") if isinstance(body.get("dossier"), dict) else body
            scope = _operator_project_scope(body)
            if isinstance(dossier, dict) and not _scope_allows(scope, _scope_item_from_dossier({"dossier": dossier})):
                return _project_scope_forbidden()
            verification = verify_customer_project_acceptance_dossier(dossier)
            return mission_json(
                {
                    "accepted": bool(verification.get("valid")),
                    "verification": verification,
                    "dossier_scope": _scope_item_from_dossier({"dossier": dossier}),
                },
                status_code=200 if verification.get("valid") else 422,
            )
        except Exception as exc:
            logger.error("Field customer project acceptance dossier verify endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/field/customer-projects/{identifier}/managed-objects/{object_id}",
        tags=["Field Operations"],
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
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            payload = body.get("managed_object") if isinstance(body.get("managed_object"), dict) else body
            result = upsert_managed_object(
                root,
                identifier,
                object_id,
                payload,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field managed object upsert endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.delete(
        "/api/field/customer-projects/{identifier}/managed-objects/{object_id}",
        tags=["Field Operations"],
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
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            result = delete_managed_object(
                root,
                identifier,
                object_id,
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
            )
            return mission_json(result, status_code=200 if result.get("accepted") else 422)
        except Exception as exc:
            logger.error("Field managed object delete endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/export", tags=["Field Operations"])
    async def field_customer_project_export(identifier: str, request: Request) -> JSONResponse:
        """Export a reusable customer project handoff package."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = export_customer_project_package(root, identifier)
            if result.get("accepted") and not _scope_allows(scope, _scope_item_from_package(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("accepted") else 404)
        except Exception as exc:
            logger.error("Field customer project export endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/acceptance-dossier", tags=["Field Operations"])
    async def field_customer_project_acceptance_dossier(
        identifier: str,
        request: Request,
        check_env: bool = True,
    ) -> JSONResponse:
        """Export a tamper-evident customer acceptance dossier."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = export_customer_project_acceptance_dossier(root, identifier, check_env=check_env)
            if result.get("accepted") and not _scope_allows(scope, _scope_item_from_dossier(result)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("accepted") else 404)
        except Exception as exc:
            logger.error("Field customer project acceptance dossier endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/proposal-bundle", tags=["Field Operations"])
    async def field_customer_project_proposal_bundle(
        identifier: str,
        request: Request,
        check_env: bool = True,
    ) -> JSONResponse:
        """Export a customer-facing proposal bundle bound to one project."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            result = export_customer_project_proposal_bundle(
                root,
                Path("deploy/customer-project-templates"),
                identifier,
                check_env=check_env,
            )
            proposal = result.get("proposal") if isinstance(result.get("proposal"), dict) else {}
            if result.get("accepted") and not _scope_allows(scope, _scope_item_from_profile(proposal)):
                return _project_scope_forbidden()
            return mission_json(result, status_code=200 if result.get("accepted") else 404)
        except Exception as exc:
            logger.error("Field customer project proposal bundle endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/customer-projects/{identifier}/history", tags=["Field Operations"])
    async def field_customer_project_history(
        identifier: str,
        request: Request,
        limit: int = 20,
    ) -> JSONResponse:
        """Return saved customer project profile revisions for rollback review."""
        try:
            failure, auth_body = _project_read_auth(request)
            if failure is not None:
                return failure
            scope = _operator_project_scope(auth_body)
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(scope, _scope_item_from_detail(detail)):
                return _project_scope_forbidden()
            result = list_customer_project_revisions(root, identifier, limit=limit)
            return mission_json(result, status_code=200 if result.get("found") else 404)
        except Exception as exc:
            logger.error("Field customer project history endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/{identifier}/rollback", tags=["Field Operations"])
    async def field_customer_project_rollback(identifier: str, request: Request) -> JSONResponse:
        """Restore a customer project profile from a saved revision."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            result = rollback_customer_project_profile(
                root,
                identifier,
                str(body.get("revision_id") or ""),
                operator_id=str(body.get("operator_id") or ""),
                reason=str(body.get("reason") or ""),
                dry_run=bool(body.get("dry_run")),
            )
            status_code = 200 if result.get("accepted") else 422
            if result.get("reason") in {"profile_not_found", "revision_not_found"}:
                status_code = 404
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field customer project rollback endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/customer-projects/{identifier}/archive", tags=["Field Operations"])
    async def field_customer_project_archive(identifier: str, request: Request) -> JSONResponse:
        """Archive a customer project profile without permanent deletion."""
        try:
            body = await optional_json_body(request)
            failure = authorize(request, body, "field:project:write")
            if failure is not None:
                return failure
            root = site_profile_root or Path("deploy/site-profiles")
            detail = get_customer_project_profile(root, identifier)
            if detail.get("found") and not _scope_allows(
                _operator_project_scope(body),
                _scope_item_from_detail(detail),
            ):
                return _project_scope_forbidden()
            result = archive_customer_project_profile(root, identifier)
            return mission_json(result, status_code=200 if result.get("accepted") else 404)
        except Exception as exc:
            logger.error("Field customer project archive endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/audit/integrity", tags=["Field Operations"])
    async def field_action_audit_integrity() -> JSONResponse:
        """Verify the append-only field action audit hash chain."""
        try:
            result = await dispatch_field_operations("action_audit_integrity_payload")
            status_code = 200
            if result.get("enabled") is not False and not result.get("valid"):
                status_code = 409
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field action audit integrity endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/ingest", tags=["Field Operations"])
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
            return mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field ingest endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/scenarios", include_in_schema=False)
    async def field_scenarios_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/events", include_in_schema=False)
    async def field_events_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/events/{event_id}", include_in_schema=False)
    async def field_event_detail_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/evidence", include_in_schema=False)
    async def field_evidence_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/events/{event_id}/acknowledge", include_in_schema=False)
    async def field_event_acknowledge_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/close", include_in_schema=False)
    async def field_event_close_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/request-close", include_in_schema=False)
    async def field_event_request_close_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/resend-notification", include_in_schema=False)
    async def field_event_resend_notification_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/report", include_in_schema=False)
    async def field_event_report_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/events/{event_id}/runtime-delivery", include_in_schema=False)
    async def field_event_runtime_delivery_cors(event_id: str) -> Response:
        _ = event_id
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/notification-test", include_in_schema=False)
    async def field_notification_test_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/notification-preflight", include_in_schema=False)
    async def field_notification_preflight_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/devices", include_in_schema=False)
    async def field_devices_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/site-profiles", include_in_schema=False)
    async def field_site_profiles_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects", include_in_schema=False)
    async def field_customer_projects_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-project-workbench", include_in_schema=False)
    async def field_customer_project_workbench_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/product-launch-readiness", include_in_schema=False)
    async def field_product_launch_readiness_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects/managed-object-directory", include_in_schema=False)
    async def field_customer_project_managed_object_directory_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

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

    @app.options("/api/field/customer-project-acceptance-registry", include_in_schema=False)
    async def field_customer_project_acceptance_registry_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-project-resource-catalog", include_in_schema=False)
    async def field_customer_project_resource_catalog_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/solution-delivery-readiness", include_in_schema=False)
    async def field_solution_delivery_readiness_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

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
        _ = resource_type, resource_id
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

    @app.options("/api/field/customer-projects/from-template", include_in_schema=False)
    async def field_customer_project_from_template_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/import", include_in_schema=False)
    async def field_customer_project_import_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/package/verify", include_in_schema=False)
    async def field_customer_project_package_verify_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/package/diff", include_in_schema=False)
    async def field_customer_project_package_diff_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/proposal-bundle/verify", include_in_schema=False)
    async def field_customer_project_proposal_bundle_verify_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/acceptance-dossier/verify", include_in_schema=False)
    async def field_customer_project_acceptance_dossier_verify_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}", include_in_schema=False)
    async def field_customer_project_detail_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/acceptance-report", include_in_schema=False)
    async def field_customer_project_acceptance_report_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

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

    @app.options(
        "/api/field/customer-projects/{identifier}/managed-objects/{object_id}",
        include_in_schema=False,
    )
    async def field_customer_project_object_cors(identifier: str, object_id: str) -> Response:
        _ = identifier, object_id
        return cors_options_response("POST, DELETE, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/export", include_in_schema=False)
    async def field_customer_project_export_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/acceptance-dossier", include_in_schema=False)
    async def field_customer_project_acceptance_dossier_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/field/customer-projects/{identifier}/proposal-bundle", include_in_schema=False)
    async def field_customer_project_proposal_bundle_cors(identifier: str) -> Response:
        _ = identifier
        return cors_options_response("GET, OPTIONS")

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

    @app.options("/api/field/ingest", include_in_schema=False)
    async def field_ingest_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/audit/integrity", include_in_schema=False)
    async def field_action_audit_integrity_cors() -> Response:
        return cors_options_response("GET, OPTIONS")


def _resolve_field_evidence_path(raw_path: str) -> Path | None:
    raw = str(raw_path or "").strip().replace("\\", "/")
    if not raw or "\x00" in raw or raw.startswith(("http://", "https://", "data:")):
        return None
    candidate = Path(raw)
    cwd = Path.cwd().resolve()
    allowed_roots = [(cwd / name).resolve() for name in _FIELD_EVIDENCE_ROOT_NAMES]
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if not candidate.parts or candidate.parts[0] not in _FIELD_EVIDENCE_ROOT_NAMES:
            return None
        resolved = (cwd / candidate).resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        return None
    if not resolved.is_file():
        return None
    return resolved
