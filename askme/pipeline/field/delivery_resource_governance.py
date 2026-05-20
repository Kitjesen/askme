"""Delivery-resource governance workflow.

This module owns the second-review workflow for high-risk delivery resource
changes. Registry mutation stays in ``delivery_resource_registry``; customer
impact analysis is loaded lazily from ``field_site_profile`` to avoid coupling
the public facade back to the full site-profile module at import time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from askme.pipeline.field.delivery_resource_registry import (
    DEFAULT_DELIVERY_RESOURCE_GOVERNANCE_SLA_S,
    DELIVERY_RESOURCE_GOVERNANCE_DUE_SOON_S,
    DELIVERY_RESOURCE_TYPES,
    _delivery_resource_descriptor,
    disable_delivery_resource,
    list_delivery_resource_registry,
    load_delivery_resource_registry,
    rollback_delivery_resource_registry,
)
from askme.pipeline.field.paths import (
    DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    DEFAULT_SITE_PROFILE_ROOT,
)


def create_delivery_resource_governance_request(
    resource_root: Path,
    action: str,
    operation: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
    profile_root: Path = DEFAULT_SITE_PROFILE_ROOT,
    template_root: Path | None = DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    sla_target_s: float | None = None,
) -> dict[str, Any]:
    """Create a pending resource-governance request without mutating resources."""
    normalized_action = str(action or operation.get("action") or "").strip()
    if normalized_action not in {"disable_resource", "rollback_registry"}:
        return {
            "accepted": False,
            "reason": "unsupported_resource_governance_action",
            "allowed_actions": ["disable_resource", "rollback_registry"],
        }
    clean_operation = _delivery_resource_governance_operation_payload(
        normalized_action,
        operation,
    )
    if not clean_operation.get("valid"):
        return {
            "accepted": False,
            "reason": clean_operation.get("reason") or "invalid_resource_governance_operation",
            "operation": clean_operation,
        }
    preview = _preview_delivery_resource_governance_operation(
        resource_root,
        normalized_action,
        clean_operation,
        profile_root=profile_root,
        template_root=template_root,
    )
    if not preview.get("accepted"):
        return {
            "accepted": False,
            "reason": preview.get("reason") or "resource_governance_request_invalid",
            "preview": preview,
        }
    created_at = time.time()
    review_sla_target_s = _delivery_resource_governance_sla_target_s(
        sla_target_s if sla_target_s is not None else operation.get("sla_target_s")
    )
    due_at = created_at + review_sla_target_s
    request_id = _slug(
        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created_at))}-"
        f"{int(created_at * 1000)}-{normalized_action}"
    )
    request_payload = {
        "request_type": "askme.delivery_resource_governance_request",
        "request_version": 1,
        "request_id": request_id,
        "status": "pending",
        "action": normalized_action,
        "operation": clean_operation,
        "requested_by": str(operator_id or "system"),
        "requested_at": created_at,
        "sla_target_s": review_sla_target_s,
        "due_at": due_at,
        "escalation_policy": "delivery_owner_review_overdue",
        "escalations": [],
        "reason": str(reason or operation.get("reason") or ""),
        "current_registry_sha256": _delivery_resource_current_registry_sha256(resource_root),
        "preview": preview,
    }
    target_dir = _delivery_resource_governance_request_dir(resource_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{request_id}.json"
    target.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    request_payload["request_path"] = str(target)
    return {
        "accepted": True,
        "request": _delivery_resource_governance_request_public_payload(request_payload),
        "preview": preview,
        "next_step": "A second delivery owner must approve this resource governance request.",
    }


def list_delivery_resource_governance_requests(
    resource_root: Path,
    *,
    status: str = "",
    action: str = "",
    limit: int = 50,
    overdue_only: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """List resource-governance requests for delivery-owner review."""
    requests = [
        _delivery_resource_governance_request_public_payload(payload, now=now)
        for payload in _iter_delivery_resource_governance_requests(resource_root)
    ]
    if status:
        requests = [item for item in requests if item.get("status") == str(status)]
    if action:
        requests = [item for item in requests if item.get("action") == str(action)]
    if overdue_only:
        requests = [
            item
            for item in requests
            if _mapping(item.get("review_sla")).get("state") == "overdue"
        ]
    requests.sort(key=lambda item: float(item.get("requested_at") or 0), reverse=True)
    capped = requests[: max(0, int(limit or 50))]
    return {
        "root": str(resource_root),
        "status": str(status or ""),
        "action": str(action or ""),
        "overdue_only": bool(overdue_only),
        "requests": capped,
        "request_count": len(requests),
        "summary": {
            "pending_count": len([item for item in requests if item.get("status") == "pending"]),
            "approved_count": len([item for item in requests if item.get("status") == "approved"]),
            "rejected_count": len([item for item in requests if item.get("status") == "rejected"]),
            "active_count": len([
                item for item in requests if _mapping(item.get("review_sla")).get("state") == "active"
            ]),
            "due_soon_count": len([
                item for item in requests if _mapping(item.get("review_sla")).get("state") == "due_soon"
            ]),
            "overdue_count": len([
                item for item in requests if _mapping(item.get("review_sla")).get("state") == "overdue"
            ]),
        },
    }


def review_delivery_resource_governance_request(
    resource_root: Path,
    request_id: str,
    *,
    decision: str,
    operator_id: str = "",
    reason: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Approve or reject a pending resource-governance request."""
    request_path, request_payload = _find_delivery_resource_governance_request(
        resource_root,
        request_id,
    )
    if not request_payload or request_path is None:
        return {
            "accepted": False,
            "reason": "resource_governance_request_not_found",
            "request_id": str(request_id or ""),
        }
    if str(request_payload.get("status") or "") != "pending":
        return {
            "accepted": False,
            "reason": "resource_governance_request_not_pending",
            "request": _delivery_resource_governance_request_public_payload(request_payload),
        }
    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in {"approve", "reject"}:
        return {
            "accepted": False,
            "reason": "invalid_resource_governance_review_decision",
            "allowed_decisions": ["approve", "reject"],
            "request": _delivery_resource_governance_request_public_payload(request_payload),
        }
    reviewer = str(operator_id or "system")
    if reviewer == str(request_payload.get("requested_by") or ""):
        return {
            "accepted": False,
            "reason": "resource_governance_request_requires_second_approver",
            "request": _delivery_resource_governance_request_public_payload(request_payload),
        }
    reviewed_at = time.time()
    next_payload = copy.deepcopy(request_payload)
    next_payload["reviewed_by"] = reviewer
    next_payload["reviewed_at"] = reviewed_at
    next_payload["review_reason"] = str(reason or "")
    if normalized_decision == "reject":
        next_payload["status"] = "rejected"
        if not dry_run:
            request_path.write_text(json.dumps(next_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "accepted": True,
            "dry_run": bool(dry_run),
            "request": _delivery_resource_governance_request_public_payload(next_payload),
            "next_step": "Resource governance request rejected. The registry was not changed.",
        }
    request_registry_sha256 = str(request_payload.get("current_registry_sha256") or "")
    current_registry_sha256 = _delivery_resource_current_registry_sha256(resource_root)
    if request_registry_sha256 and request_registry_sha256 != current_registry_sha256:
        return {
            "accepted": False,
            "reason": "resource_governance_registry_changed_since_request",
            "request": _delivery_resource_governance_request_public_payload(request_payload),
            "request_registry_sha256": request_registry_sha256,
            "current_registry_sha256": current_registry_sha256,
            "next_step": (
                "Recreate the resource governance request after reviewing the latest registry state."
            ),
        }
    apply_result = _apply_delivery_resource_governance_operation(
        resource_root,
        str(request_payload.get("action") or ""),
        _mapping(request_payload.get("operation")),
        operator_id=reviewer,
        reason=reason or str(request_payload.get("reason") or ""),
        dry_run=dry_run,
    )
    if not apply_result.get("accepted"):
        return {
            "accepted": False,
            "reason": apply_result.get("reason") or "resource_governance_apply_failed",
            "request": _delivery_resource_governance_request_public_payload(request_payload),
            "apply_result": apply_result,
        }
    next_payload["status"] = "approved"
    next_payload["apply_result"] = apply_result
    if not dry_run:
        request_path.write_text(json.dumps(next_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "accepted": True,
        "dry_run": bool(dry_run),
        "request": _delivery_resource_governance_request_public_payload(next_payload),
        "apply_result": apply_result,
        "next_step": "Resource governance request approved and applied.",
    }


def escalate_overdue_delivery_resource_governance_requests(
    resource_root: Path,
    *,
    operator_id: str = "",
    reason: str = "",
    limit: int = 50,
    now: float | None = None,
    dry_run: bool = False,
    notification_delivery: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record and optionally deliver escalation payloads for overdue governance requests."""
    current = time.time() if now is None else _float_value(now)
    overdue = list_delivery_resource_governance_requests(
        resource_root,
        status="pending",
        overdue_only=True,
        limit=limit,
        now=current,
    )
    escalations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in overdue.get("requests", []):
        request_id = str(item.get("request_id") or "")
        request_path, request_payload = _find_delivery_resource_governance_request(
            resource_root,
            request_id,
        )
        if not request_path or not request_payload:
            skipped.append({
                "request_id": request_id,
                "reason": "resource_governance_request_not_found",
            })
            continue
        if _delivery_resource_governance_has_open_escalation(request_payload, item):
            skipped.append({
                "request_id": request_id,
                "reason": "resource_governance_request_already_escalated",
                "last_escalation": _delivery_resource_governance_last_escalation(request_payload),
            })
            continue
        escalation = _delivery_resource_governance_escalation_record(
            item,
            operator_id=operator_id,
            reason=reason,
            now=current,
        )
        if notification_delivery is not None and not dry_run:
            escalation = _delivery_resource_governance_apply_notification_delivery(
                escalation,
                notification_delivery,
            )
        escalations.append(escalation)
        if not dry_run:
            next_payload = copy.deepcopy(request_payload)
            records = [
                _mapping(record)
                for record in next_payload.get("escalations", [])
                if isinstance(record, dict)
            ]
            records.append(escalation)
            next_payload["escalations"] = records
            next_payload["last_escalated_at"] = escalation["escalated_at"]
            next_payload["last_escalated_by"] = escalation["escalated_by"]
            request_path.write_text(
                json.dumps(next_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    refreshed = list_delivery_resource_governance_requests(
        resource_root,
        status="pending",
        overdue_only=True,
        limit=limit,
        now=current,
    )
    return {
        "accepted": True,
        "dry_run": bool(dry_run),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or ""),
        "checked_count": int(overdue.get("request_count") or 0),
        "escalated_count": len(escalations),
        "skipped_count": len(skipped),
        "escalations": escalations,
        "skipped": skipped,
        "requests": refreshed.get("requests", []),
        "summary": refreshed.get("summary", {}),
        "next_step": (
            "Escalated overdue resource governance requests to the delivery owner queue."
            if escalations
            else "No new overdue resource governance requests required escalation."
        ),
    }


def _delivery_resource_governance_operation_payload(
    action: str,
    operation: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a high-risk resource governance operation before review."""
    normalized_action = str(action or "").strip()
    if normalized_action == "disable_resource":
        resource_type = str(operation.get("resource_type") or "").strip()
        resource_id = str(operation.get("resource_id") or "").strip()
        if resource_type not in DELIVERY_RESOURCE_TYPES:
            return {
                "valid": False,
                "reason": "unsupported_resource_type",
                "resource_type": resource_type,
            }
        if not resource_id:
            return {"valid": False, "reason": "resource_id_required"}
        return {
            "valid": True,
            "action": normalized_action,
            "resource_type": resource_type,
            "resource_id": resource_id,
        }
    if normalized_action == "rollback_registry":
        revision_id = str(operation.get("revision_id") or "").strip()
        if not revision_id:
            return {"valid": False, "reason": "revision_id_required"}
        return {
            "valid": True,
            "action": normalized_action,
            "revision_id": revision_id,
        }
    return {
        "valid": False,
        "reason": "unsupported_resource_governance_action",
        "action": normalized_action,
    }


def _preview_delivery_resource_governance_operation(
    resource_root: Path,
    action: str,
    operation: dict[str, Any],
    *,
    profile_root: Path = DEFAULT_SITE_PROFILE_ROOT,
    template_root: Path | None = DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
) -> dict[str, Any]:
    """Return the expected impact of a resource-governance operation."""
    if action == "disable_resource":
        resource_type = str(operation.get("resource_type") or "")
        resource_id = str(operation.get("resource_id") or "")
        registry = list_delivery_resource_registry(resource_root)
        bucket = _mapping(_mapping(registry.get("delivery_resources")).get(resource_type))
        resource = _mapping(bucket.get(resource_id))
        if not resource:
            return {
                "accepted": False,
                "reason": "resource_not_found",
                "resource_type": resource_type,
                "resource_id": resource_id,
            }
        return {
            "accepted": True,
            "dry_run": True,
            "would_write": True,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "current_resource": _delivery_resource_descriptor(
                resource_id,
                resource_type,
                resource,
                source_default="shared_registry",
            ),
            "target_publish_status": "disabled",
            "impact": _delivery_resource_governance_impact(
                resource_root,
                action,
                operation,
                profile_root=profile_root,
                template_root=template_root,
            ),
        }
    if action == "rollback_registry":
        preview = rollback_delivery_resource_registry(
            resource_root,
            str(operation.get("revision_id") or ""),
            dry_run=True,
        )
        if preview.get("accepted"):
            preview["impact"] = _delivery_resource_governance_impact(
                resource_root,
                action,
                operation,
                preview=preview,
                profile_root=profile_root,
                template_root=template_root,
            )
        return preview
    return {
        "accepted": False,
        "reason": "unsupported_resource_governance_action",
        "action": action,
    }


def _delivery_resource_governance_impact(
    resource_root: Path,
    action: str,
    operation: dict[str, Any],
    *,
    preview: dict[str, Any] | None = None,
    profile_root: Path = DEFAULT_SITE_PROFILE_ROOT,
    template_root: Path | None = DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
) -> dict[str, Any]:
    """Summarize the customer-project blast radius for a resource governance request."""
    generated_at = time.time()
    if action == "rollback_registry":
        current = list_delivery_resource_registry(resource_root)
        target_summary = _mapping((preview or {}).get("target_summary"))
        return {
            "impact_type": "registry_rollback",
            "generated_at": generated_at,
            "analysis_status": "manual_review_required",
            "registry_wide": True,
            "current_resource_count": int(_mapping(current.get("summary")).get("resource_count") or 0),
            "target_resource_count": int(target_summary.get("resource_count") or 0),
            "affected_customer_project_count": int(
                _mapping(current.get("summary")).get("project_consumer_count") or 0
            ),
            "affected_template_count": int(_mapping(current.get("summary")).get("template_consumer_count") or 0),
            "message": (
                "Registry rollback can affect every customer project using shared delivery resources; "
                "review the dry-run summary and registry history before approval."
            ),
        }
    if action != "disable_resource":
        return {
            "impact_type": action or "unknown",
            "generated_at": generated_at,
            "analysis_status": "unsupported",
            "affected_consumer_count": 0,
            "affected_consumers": [],
        }
    resource_type = str(operation.get("resource_type") or "")
    resource_id = str(operation.get("resource_id") or "")
    try:
        catalog = _build_customer_project_resource_catalog(
            profile_root,
            template_root=template_root,
            delivery_resource_root=resource_root,
        )
    except Exception as exc:
        return {
            "impact_type": "resource_disable",
            "generated_at": generated_at,
            "analysis_status": "failed",
            "reason": str(exc),
            "affected_consumer_count": 0,
            "affected_consumers": [],
            "message": "Impact analysis failed; approval should require manual review.",
        }
    affected = [
        _delivery_resource_impact_consumer_payload(item)
        for item in catalog.get("consumers", [])
        if isinstance(item, dict)
        and str(item.get("resource_type") or "") == resource_type
        and str(item.get("resource_id") or "") == resource_id
    ]
    affected_projects = _delivery_resource_impact_projects(affected)
    affected_objects = _delivery_resource_impact_objects(affected)
    affected_templates = _delivery_resource_impact_templates(affected)
    return {
        "impact_type": "resource_disable",
        "generated_at": generated_at,
        "analysis_status": "complete",
        "resource_type": resource_type,
        "resource_id": resource_id,
        "affected_consumer_count": len(affected),
        "affected_customer_project_count": len(affected_projects),
        "affected_object_count": len(affected_objects),
        "affected_template_count": len(affected_templates),
        "affected_projects": affected_projects[:20],
        "affected_objects": affected_objects[:20],
        "affected_templates": affected_templates[:20],
        "affected_consumers": affected[:20],
        "truncated": (
            len(affected) > 20
            or len(affected_projects) > 20
            or len(affected_objects) > 20
            or len(affected_templates) > 20
        ),
        "message": (
            "No customer project, object, or template references were detected."
            if not affected
            else "Affected customer projects, objects, and templates should be reviewed before approval."
        ),
    }


def _delivery_resource_impact_projects(consumers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projects: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for item in consumers:
        if item.get("scope_type") == "template":
            continue
        key = (
            str(item.get("tenant_id") or ""),
            str(item.get("delivery_namespace") or ""),
            str(item.get("customer_id") or ""),
            str(item.get("project_id") or ""),
            str(item.get("site_id") or ""),
        )
        project = projects.setdefault(
            key,
            {
                "tenant_id": key[0],
                "delivery_namespace": key[1],
                "customer_id": key[2],
                "customer_name": str(item.get("customer_name") or ""),
                "project_id": key[3],
                "project_name": str(item.get("project_name") or ""),
                "site_id": key[4],
                "consumer_count": 0,
                "object_count": 0,
            },
        )
        project["consumer_count"] = int(project.get("consumer_count") or 0) + 1
        object_ids = set(project.get("_object_ids") or [])
        if item.get("object_id"):
            object_ids.add(str(item.get("object_id") or ""))
        project["_object_ids"] = sorted(object_ids)
        project["object_count"] = len(object_ids)
    for project in projects.values():
        project.pop("_object_ids", None)
    return sorted(projects.values(), key=lambda row: (row["customer_id"], row["project_id"], row["site_id"]))


def _delivery_resource_impact_objects(consumers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in consumers:
        object_id = str(item.get("object_id") or "")
        if not object_id:
            continue
        key = (
            str(item.get("scope_type") or ""),
            str(item.get("profile_path") or ""),
            object_id,
        )
        row = objects.setdefault(
            key,
            {
                "scope_type": key[0],
                "profile_path": key[1],
                "template_id": str(item.get("template_id") or ""),
                "customer_id": str(item.get("customer_id") or ""),
                "project_id": str(item.get("project_id") or ""),
                "site_id": str(item.get("site_id") or ""),
                "object_id": object_id,
                "display_name": str(item.get("display_name") or object_id),
                "category": str(item.get("category") or ""),
                "consumer_count": 0,
            },
        )
        row["consumer_count"] = int(row.get("consumer_count") or 0) + 1
    return sorted(objects.values(), key=lambda row: (row["scope_type"], row["profile_path"], row["object_id"]))


def _delivery_resource_impact_templates(consumers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for item in consumers:
        if item.get("scope_type") != "template":
            continue
        template_id = str(item.get("template_id") or "")
        if not template_id:
            continue
        row = templates.setdefault(
            template_id,
            {
                "template_id": template_id,
                "profile_path": str(item.get("profile_path") or ""),
                "consumer_count": 0,
                "object_count": 0,
            },
        )
        row["consumer_count"] = int(row.get("consumer_count") or 0) + 1
        object_ids = set(row.get("_object_ids") or [])
        if item.get("object_id"):
            object_ids.add(str(item.get("object_id") or ""))
        row["_object_ids"] = sorted(object_ids)
        row["object_count"] = len(object_ids)
    for row in templates.values():
        row.pop("_object_ids", None)
    return sorted(templates.values(), key=lambda row: row["template_id"])


def _delivery_resource_impact_consumer_payload(consumer: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope_type": str(consumer.get("scope_type") or ""),
        "profile_path": str(consumer.get("profile_path") or ""),
        "template_id": str(consumer.get("template_id") or ""),
        "tenant_id": str(consumer.get("tenant_id") or ""),
        "delivery_namespace": str(consumer.get("delivery_namespace") or ""),
        "customer_id": str(consumer.get("customer_id") or ""),
        "customer_name": str(consumer.get("customer_name") or ""),
        "project_id": str(consumer.get("project_id") or ""),
        "project_name": str(consumer.get("project_name") or ""),
        "site_id": str(consumer.get("site_id") or ""),
        "object_id": str(consumer.get("object_id") or ""),
        "display_name": str(consumer.get("display_name") or ""),
        "category": str(consumer.get("category") or ""),
        "resource_type": str(consumer.get("resource_type") or ""),
        "resource_id": str(consumer.get("resource_id") or ""),
        "status": str(consumer.get("status") or ""),
        "source": str(consumer.get("source") or ""),
        "message": str(consumer.get("message") or ""),
    }


def _apply_delivery_resource_governance_operation(
    resource_root: Path,
    action: str,
    operation: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply an approved resource-governance operation."""
    if action == "disable_resource":
        if dry_run:
            return _preview_delivery_resource_governance_operation(resource_root, action, operation)
        return disable_delivery_resource(
            resource_root,
            str(operation.get("resource_type") or ""),
            str(operation.get("resource_id") or ""),
            operator_id=operator_id,
            reason=reason,
        )
    if action == "rollback_registry":
        return rollback_delivery_resource_registry(
            resource_root,
            str(operation.get("revision_id") or ""),
            operator_id=operator_id,
            reason=reason,
            dry_run=dry_run,
        )
    return {
        "accepted": False,
        "reason": "unsupported_resource_governance_action",
        "action": action,
    }


def _delivery_resource_governance_request_dir(resource_root: Path) -> Path:
    return Path(resource_root) / "_resource_governance_requests"


def _iter_delivery_resource_governance_requests(resource_root: Path) -> list[dict[str, Any]]:
    root = _delivery_resource_governance_request_dir(resource_root)
    if not root.exists():
        return []
    requests: list[dict[str, Any]] = []
    for item in root.glob("*.json"):
        if not item.is_file():
            continue
        payload = _read_delivery_resource_governance_request_file(item)
        if payload:
            requests.append(payload)
    return requests


def _find_delivery_resource_governance_request(
    resource_root: Path,
    request_id: str,
) -> tuple[Path | None, dict[str, Any]]:
    target_id = str(request_id or "").strip()
    if not target_id:
        return None, {}
    root = _delivery_resource_governance_request_dir(resource_root)
    if not root.exists():
        return None, {}
    for item in root.glob("*.json"):
        payload = _read_delivery_resource_governance_request_file(item)
        if payload and str(payload.get("request_id") or "") == target_id:
            return item, payload
    return None, {}


def _read_delivery_resource_governance_request_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("request_type") != "askme.delivery_resource_governance_request"
    ):
        return {}
    payload["request_path"] = str(path)
    return payload


def _delivery_resource_governance_sla_target_s(value: Any = None) -> float:
    target_s = _float_value(value)
    if target_s <= 0:
        target_s = float(DEFAULT_DELIVERY_RESOURCE_GOVERNANCE_SLA_S)
    return max(60.0, target_s)


def _delivery_resource_governance_review_sla(
    request: dict[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    target_s = _delivery_resource_governance_sla_target_s(request.get("sla_target_s"))
    requested_at = _float_value(request.get("requested_at"))
    due_at = _float_value(request.get("due_at"))
    if due_at <= 0 and requested_at > 0:
        due_at = requested_at + target_s
    current = time.time() if now is None else _float_value(now)
    status = str(request.get("status") or "")
    if status != "pending":
        reviewed_at = _float_value(request.get("reviewed_at"))
        return {
            "state": "closed",
            "target_s": target_s,
            "requested_at": requested_at,
            "due_at": due_at,
            "closed_at": reviewed_at or current,
            "remaining_s": 0.0,
            "age_s": max(0.0, (reviewed_at or current) - requested_at) if requested_at else 0.0,
            "overdue_s": 0.0,
            "escalation_required": False,
            "message": "Review is closed.",
        }
    remaining_s = due_at - current if due_at else 0.0
    overdue_s = abs(remaining_s) if due_at and remaining_s < 0 else 0.0
    if due_at and remaining_s < 0:
        state = "overdue"
        message = "Review SLA is overdue; escalate to a delivery owner before applying customer-facing changes."
    elif due_at and remaining_s <= DELIVERY_RESOURCE_GOVERNANCE_DUE_SOON_S:
        state = "due_soon"
        message = "Review SLA is due soon; keep the delivery owner queue visible."
    else:
        state = "active"
        message = "Review SLA is active."
    return {
        "state": state,
        "target_s": target_s,
        "requested_at": requested_at,
        "due_at": due_at,
        "remaining_s": remaining_s,
        "age_s": max(0.0, current - requested_at) if requested_at else 0.0,
        "overdue_s": overdue_s,
        "due_soon_threshold_s": float(DELIVERY_RESOURCE_GOVERNANCE_DUE_SOON_S),
        "escalation_required": state == "overdue",
        "escalation_policy": str(request.get("escalation_policy") or "delivery_owner_review_overdue"),
        "message": message,
    }


def _delivery_resource_governance_last_escalation(request: dict[str, Any]) -> dict[str, Any]:
    records = [
        _mapping(item)
        for item in request.get("escalations", [])
        if isinstance(item, dict)
    ]
    return records[-1] if records else {}


def _delivery_resource_governance_has_open_escalation(
    request: dict[str, Any],
    public_request: dict[str, Any],
) -> bool:
    due_at = _float_value(public_request.get("due_at"))
    for record in request.get("escalations", []):
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "") not in {"queued", "sent", "acknowledged"}:
            continue
        if abs(_float_value(record.get("due_at")) - due_at) < 0.001:
            return True
    return False


def _delivery_resource_governance_escalation_record(
    public_request: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    current = time.time() if now is None else _float_value(now)
    operation = _mapping(public_request.get("operation"))
    target = (
        f"revision {operation.get('revision_id') or '-'}"
        if public_request.get("action") == "rollback_registry"
        else f"{operation.get('resource_type') or '-'}/{operation.get('resource_id') or '-'}"
    )
    sla = _mapping(public_request.get("review_sla"))
    request_id = str(public_request.get("request_id") or "")
    message = (
        "Delivery resource governance request is overdue: "
        f"{request_id} ({public_request.get('action') or '-'}) for {target}. "
        f"Requester: {public_request.get('requested_by') or 'system'}. "
        f"Overdue: {int(_float_value(sla.get('overdue_s')))} seconds. "
        "A second delivery owner must approve or reject it before customer-facing changes proceed."
    )
    return {
        "escalation_id": _slug(
            f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(current))}-"
            f"{int(current * 1000)}-{request_id}-overdue"
        ),
        "status": "queued",
        "escalated_at": current,
        "escalated_by": str(operator_id or "system"),
        "reason": str(reason or "review_sla_overdue"),
        "request_id": request_id,
        "action": str(public_request.get("action") or ""),
        "target": target,
        "requested_by": str(public_request.get("requested_by") or ""),
        "requested_at": public_request.get("requested_at"),
        "due_at": public_request.get("due_at"),
        "overdue_s": _float_value(sla.get("overdue_s")),
        "escalation_policy": str(
            public_request.get("escalation_policy")
            or sla.get("escalation_policy")
            or "delivery_owner_review_overdue"
        ),
        "delivery_group": "delivery_owners",
        "notification": {
            "channel": "delivery_owner_queue",
            "status": "queued",
            "message": message,
            "title": "Resource governance review overdue",
        },
        "delivery_report": [
            {
                "channel": "delivery_owner_queue",
                "status": "queued",
                "reason": "local_delivery_owner_queue",
            }
        ],
    }


def _delivery_resource_governance_apply_notification_delivery(
    escalation: dict[str, Any],
    notification_delivery: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    next_escalation = copy.deepcopy(escalation)
    notification = _mapping(next_escalation.get("notification"))
    try:
        delivery_result = _mapping(notification_delivery(copy.deepcopy(next_escalation)))
    except Exception as exc:  # pragma: no cover - defensive integration boundary
        notification["status"] = "failed"
        notification["delivery_mode"] = "configured_delivery"
        notification["reason"] = str(exc)
        next_escalation["status"] = "failed"
        next_escalation["notification"] = notification
        next_escalation["delivery_report"] = [
            {
                "channel": "delivery_owner_notification",
                "status": "failed",
                "reason": str(exc),
            }
        ]
        return next_escalation

    delivery_report = delivery_result.get("delivery_report")
    if isinstance(delivery_report, list):
        next_escalation["delivery_report"] = [
            _mapping(item) for item in delivery_report if isinstance(item, dict)
        ]
    sent_value = delivery_result.get("sent_channels")
    sent_channels = [
        str(item)
        for item in (sent_value if isinstance(sent_value, list) else [])
        if str(item).strip()
    ]
    status = str(
        delivery_result.get("status")
        or ("sent" if sent_channels else notification.get("status") or "queued")
    )
    notification["status"] = status
    notification["delivery_mode"] = str(
        delivery_result.get("delivery_mode") or "configured_delivery"
    )
    notification["sent_channels"] = sent_channels
    reason = str(delivery_result.get("reason") or "")
    if reason:
        notification["reason"] = reason
    next_escalation["status"] = status
    next_escalation["notification"] = notification
    return next_escalation


def _delivery_resource_governance_request_public_payload(
    request: dict[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    if not isinstance(request, dict) or not request:
        return {}
    review_sla = _delivery_resource_governance_review_sla(request, now=now)
    escalations = [
        _mapping(item)
        for item in request.get("escalations", [])
        if isinstance(item, dict)
    ]
    return {
        "request_id": str(request.get("request_id") or ""),
        "status": str(request.get("status") or ""),
        "action": str(request.get("action") or ""),
        "operation": _mapping(request.get("operation")),
        "request_path": str(request.get("request_path") or ""),
        "requested_by": str(request.get("requested_by") or ""),
        "requested_at": request.get("requested_at"),
        "sla_target_s": review_sla["target_s"],
        "due_at": review_sla["due_at"],
        "review_sla": review_sla,
        "escalation_policy": str(request.get("escalation_policy") or "delivery_owner_review_overdue"),
        "escalation_count": len(escalations),
        "last_escalation": escalations[-1] if escalations else {},
        "escalations": escalations[-5:],
        "reason": str(request.get("reason") or ""),
        "reviewed_by": str(request.get("reviewed_by") or ""),
        "reviewed_at": request.get("reviewed_at"),
        "review_reason": str(request.get("review_reason") or ""),
        "current_registry_sha256": str(request.get("current_registry_sha256") or ""),
        "preview": _mapping(request.get("preview")),
        "apply_result": _mapping(request.get("apply_result")),
    }


def _build_customer_project_resource_catalog(
    profile_root: Path,
    *,
    template_root: Path | None,
    delivery_resource_root: Path,
) -> dict[str, Any]:
    from askme.pipeline.field.customer_project_resource_catalog import (
        build_customer_project_resource_catalog,
    )

    return build_customer_project_resource_catalog(
        profile_root,
        template_root=template_root,
        delivery_resource_root=delivery_resource_root,
    )


def _delivery_resource_current_registry_sha256(resource_root: Path) -> str:
    registry = load_delivery_resource_registry(resource_root)
    return _sha256_json({
        "registry_type": "askme.delivery_resource_registry",
        "registry_version": int(registry.get("registry_version") or 1),
        "delivery_resources": _mapping(registry.get("delivery_resources")),
    })


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _slug(value: Any) -> str:
    text = str(value or "item").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-_")
    return text or "item"


__all__ = [
    "create_delivery_resource_governance_request",
    "escalate_overdue_delivery_resource_governance_requests",
    "list_delivery_resource_governance_requests",
    "review_delivery_resource_governance_request",
]
