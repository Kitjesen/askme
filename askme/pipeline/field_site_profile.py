"""Field site profile validation and conversion.

A site profile is the customer-facing deployment contract for field operations:
zones, help points, parking policy, responder groups, devices, and thresholds.
It lets a deployment team change the park map without editing Python code.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import html
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from askme.skills.field_capability_contracts import field_capability_routes

REQUIRED_RESPONDER_GROUPS = ("security", "cleaning", "operations")
REQUIRED_DEVICE_SOURCES = ("camera", "sensor", "robot")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DELIVERY_NAMESPACE = "default"
DEFAULT_DELIVERY_RESOURCE_ROOT = Path("deploy/delivery-resources")
TEMPLATE_PUBLISH_STATUSES = {"draft", "pilot", "published", "deprecated", "blocked"}
TEMPLATE_RELEASE_REQUEST_STATUSES = {"pending", "approved", "rejected", "cancelled"}
TEMPLATE_RELEASE_FIELDS = (
    "version",
    "publish_status",
    "release_channel",
    "owner",
    "upgrade_policy",
    "min_runtime_version",
    "release_note",
)

_ACCEPTANCE_TEST_ALIASES: dict[str, tuple[str, ...]] = {
    "crowd_gathering": ("crowd_gathering_records_security_event",),
    "fire_or_smoke": ("fire_sensor_notifies_security",),
    "illegal_parking": ("illegal_parking_camera_ingest",),
    "trash_bin_full": ("trash_bin_full_notifies_cleaning",),
    "visitor_escort": (
        "visitor_escort_is_archived_without_alert",
        "test_guide_returns_voice_text_or_escort_payload",
        "task_type\"] == \"visitor_escort",
    ),
    "visitor_wayfinding": (
        "visitor_wayfinding_grounded",
        "test_rag_trust_scenario_visitor_wayfinding_uses_grounded_evidence",
    ),
    "wayfinding_help_point": ("wayfinding_help_point_does_not_notify_security",),
}

_SCENARIO_REQUIRED_INPUTS: dict[str, tuple[str, ...]] = {
    "robot_abnormal_incident": ("location", "fault_type"),
    "night_stranger_photo": ("location", "zone_name", "image_path"),
    "illegal_parking": ("location", "zone_name", "image_path"),
    "fire_or_smoke": ("location", "image_path"),
    "trash_bin_full": ("location", "bin_id", "image_path"),
    "urgent_patrol_dispatch": ("target_location", "operator_id"),
    "crowd_gathering": ("location", "person_count", "duration_min", "image_path"),
    "wayfinding_help_point": ("help_point_id", "location"),
    "visitor_escort": ("destination", "location"),
}

_FIELD_READINESS_EVIDENCE_DEFAULTS: dict[str, str] = {
    "archive_path": "artifacts/field_operations/smoke/field-events.jsonl",
    "scenario_report_path": "artifacts/field_operations/scenario-evaluation.json",
    "smoke_report_path": "artifacts/field_operations/smoke/field-ingest-smoke.json",
    "voice_smoke_report_path": "artifacts/field_operations/smoke/field-voice-smoke.json",
    "notification_smoke_report_path": "artifacts/field_operations/smoke/field-notification-smoke.json",
    "runtime_roundtrip_report_path": "artifacts/runtime_handoff/field-runtime-roundtrip-live-smoke.json",
}

ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES = (
    "device_ingest",
    "voice_playback",
    "notification_delivery",
    "runtime_roundtrip",
)
ONSITE_ACCEPTANCE_EVIDENCE_TYPES = {
    *ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES,
    "customer_review",
}
ONSITE_ACCEPTANCE_STATUSES = {"passed", "failed", "manual_check"}
ACCEPTANCE_REVIEW_DECISIONS = {"accepted", "needs_fix", "rejected", "waived"}
CUSTOMER_SIGNOFF_DECISIONS = {"accepted", "needs_fix", "rejected"}

DELIVERY_RESOURCE_TYPES = (
    "vision_models",
    "sensor_protocols",
    "skill_packages",
    "acceptance_tests",
)
DELIVERY_RESOURCE_PUBLISH_STATUSES = {
    "draft",
    "pilot",
    "published",
    "deprecated",
    "disabled",
    "blocked",
}
DEFAULT_DELIVERY_RESOURCE_GOVERNANCE_SLA_S = 24 * 60 * 60
DELIVERY_RESOURCE_GOVERNANCE_DUE_SOON_S = 2 * 60 * 60

DEFAULT_DELIVERY_RESOURCE_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "vision_models": {
        "vehicle-detection": {
            "display_name": "Vehicle detection",
            "category": "traffic",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "smoke-fire-detection": {
            "display_name": "Smoke and fire detection",
            "category": "safety",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "trash-bin-fill-detection": {
            "display_name": "Trash bin fill detection",
            "category": "cleaning",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "person-dwell-detection": {
            "display_name": "Person dwell detection",
            "category": "visitor_service",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "person-counting": {
            "display_name": "Person counting",
            "category": "crowd_safety",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "equipment-panel-detection": {
            "display_name": "Equipment panel detection",
            "category": "factory",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "obstacle-pallet-detection": {
            "display_name": "Obstacle and pallet detection",
            "category": "warehouse",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "shelf-barcode-detection": {
            "display_name": "Shelf barcode detection",
            "category": "warehouse",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "gate-detection": {
            "display_name": "Gate detection",
            "category": "access",
            "version": "custom",
            "source": "project_extension",
        },
    },
    "sensor_protocols": {
        "camera-detection-json": {
            "display_name": "Camera detection JSON",
            "category": "camera",
            "version": "v1",
            "source": "askme_builtin",
        },
        "smoke-temperature-json": {
            "display_name": "Smoke and temperature JSON",
            "category": "sensor",
            "version": "v1",
            "source": "askme_builtin",
        },
        "voice-turn-json": {
            "display_name": "Voice turn JSON",
            "category": "voice",
            "version": "v1",
            "source": "askme_builtin",
        },
        "robot-route-status-json": {
            "display_name": "Robot route status JSON",
            "category": "robot",
            "version": "v1",
            "source": "askme_builtin",
        },
        "robot-status-json": {
            "display_name": "Robot status JSON",
            "category": "robot",
            "version": "v1",
            "source": "askme_builtin",
        },
    },
    "skill_packages": {
        "capability.detect_illegal_parking": {
            "display_name": "Illegal parking detection",
            "category": "traffic",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.detect_fire_smoke": {
            "display_name": "Fire and smoke detection",
            "category": "safety",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.inspect_trash_bin": {
            "display_name": "Trash bin inspection",
            "category": "cleaning",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.answer_wayfinding": {
            "display_name": "Visitor wayfinding answer",
            "category": "visitor_service",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.escort_visitor": {
            "display_name": "Visitor escort",
            "category": "visitor_service",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.detect_crowd_gathering": {
            "display_name": "Crowd gathering detection",
            "category": "crowd_safety",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.inspect_equipment": {
            "display_name": "Equipment inspection",
            "category": "factory",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.detect_aisle_blocking": {
            "display_name": "Aisle blocking detection",
            "category": "warehouse",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.inspect_shelf": {
            "display_name": "Shelf inspection",
            "category": "warehouse",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.inspect_gate": {
            "display_name": "Gate inspection",
            "category": "access",
            "version": "custom",
            "source": "project_extension",
        },
        "capability.navigate": {
            "display_name": "Navigation",
            "category": "robot",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.patrol_scan": {
            "display_name": "Patrol scan",
            "category": "patrol",
            "version": "builtin",
            "source": "askme_builtin",
        },
        "capability.agent_task": {
            "display_name": "Agent task handoff",
            "category": "robot_task",
            "version": "builtin",
            "source": "askme_builtin",
        },
    },
    "acceptance_tests": {},
}


def load_field_site_profile(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("site profile root must be a mapping")
    return payload


def load_delivery_resource_registry(resource_root: Path = DEFAULT_DELIVERY_RESOURCE_ROOT) -> dict[str, Any]:
    """Return shared delivery resources registered outside individual projects."""
    path = _delivery_resource_registry_path(resource_root)
    if not path.exists():
        return {
            "found": False,
            "registry_path": str(path),
            "delivery_resources": {resource_type: {} for resource_type in DELIVERY_RESOURCE_TYPES},
        }
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("delivery resource registry root must be a mapping")
    payload.setdefault("delivery_resources", {})
    payload["found"] = True
    payload["registry_path"] = str(path)
    return payload


def list_delivery_resource_registry(resource_root: Path = DEFAULT_DELIVERY_RESOURCE_ROOT) -> dict[str, Any]:
    """Return resources managed by the shared solution-provider registry."""
    registry = load_delivery_resource_registry(resource_root)
    catalog = _delivery_resource_catalog(
        registry,
        include_defaults=False,
        include_shared=False,
    )
    resources = _delivery_resource_rows(catalog, [])
    summary = _delivery_resource_registry_summary(resources, [])
    return {
        "found": bool(registry.get("found")),
        "registry_path": str(registry.get("registry_path") or _delivery_resource_registry_path(resource_root)),
        "summary": summary,
        "resources": resources,
        "delivery_resources": registry.get("delivery_resources") or {},
        "next_step": (
            "Register model, protocol, skill, and acceptance resources before binding customer objects."
            if not resources
            else "Review resource ownership, versions, and project bindings before customer export."
        ),
    }


def upsert_delivery_resource(
    resource_root: Path,
    resource_type: str,
    resource_id: str,
    metadata: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Create or update one shared delivery resource registry entry."""
    clean_type = str(resource_type or "").strip()
    clean_id = str(resource_id or "").strip()
    if clean_type not in DELIVERY_RESOURCE_TYPES:
        return {"accepted": False, "reason": "unsupported_resource_type", "resource_type": clean_type}
    if not clean_id:
        return {"accepted": False, "reason": "resource_id_required"}
    publish_status = str(_mapping(metadata).get("publish_status") or "published").strip()
    if publish_status not in DELIVERY_RESOURCE_PUBLISH_STATUSES:
        return {
            "accepted": False,
            "reason": "invalid_publish_status",
            "publish_status": publish_status,
            "allowed_publish_statuses": sorted(DELIVERY_RESOURCE_PUBLISH_STATUSES),
        }
    registry = load_delivery_resource_registry(resource_root)
    resources = registry.setdefault("delivery_resources", {})
    if not isinstance(resources, dict):
        resources = {}
        registry["delivery_resources"] = resources
    bucket = resources.setdefault(clean_type, {})
    if not isinstance(bucket, dict):
        bucket = {}
        resources[clean_type] = bucket
    exists = clean_id in bucket
    if exists and not overwrite:
        return {"accepted": False, "reason": "resource_already_exists", "resource_type": clean_type, "resource_id": clean_id}
    revision = {}
    path = _delivery_resource_registry_path(resource_root)
    if path.exists():
        revision = _snapshot_delivery_resource_registry_revision(
            resource_root,
            registry,
            action="resource_upsert",
            operator_id=operator_id,
            reason=reason,
        )
    now = time.time()
    data = _delivery_resource_descriptor(
        clean_id,
        clean_type,
        metadata,
        source_default="shared_registry",
    )
    for key in (
        "tenant_id",
        "delivery_namespace",
        "customer_id",
        "project_id",
        "site_id",
        "publish_status",
        "release_channel",
    ):
        value = str(_mapping(metadata).get(key) or "").strip()
        if value:
            data[key] = value
    previous = _mapping(bucket.get(clean_id))
    data["created_at"] = previous.get("created_at") or now
    data["updated_at"] = now
    data["updated_by"] = str(operator_id or "")
    data["update_reason"] = str(reason or "")
    data["publish_status"] = publish_status
    bucket[clean_id] = data
    registry["registry_type"] = "askme.delivery_resource_registry"
    registry["registry_version"] = 1
    registry["updated_at"] = now
    registry["updated_by"] = str(operator_id or "")
    _write_yaml(path, {
        "registry_type": registry["registry_type"],
        "registry_version": registry["registry_version"],
        "updated_at": registry["updated_at"],
        "updated_by": registry["updated_by"],
        "delivery_resources": resources,
    })
    return {
        "accepted": True,
        "created": not exists,
        "resource_type": clean_type,
        "resource_id": clean_id,
        "resource": data,
        "registry_path": str(path),
        "revision": _delivery_resource_revision_public_payload(revision),
        "next_step": "Bind this resource to managed objects, then export or verify the customer project package.",
    }


def list_delivery_resource_revisions(
    resource_root: Path = DEFAULT_DELIVERY_RESOURCE_ROOT,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return saved shared resource registry revisions for audit and rollback."""
    revision_dir = Path(resource_root) / "_resource_revisions"
    if not revision_dir.exists():
        return {
            "found": False,
            "registry_path": str(_delivery_resource_registry_path(resource_root)),
            "revision_count": 0,
            "revisions": [],
            "next_step": "No resource registry revisions exist yet.",
        }
    revisions = [
        _delivery_resource_revision_public_payload(payload)
        for payload in (
            _read_delivery_resource_revision_file(item)
            for item in revision_dir.glob("*.json")
            if item.is_file()
        )
        if payload
    ]
    revisions.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
    clean_limit = max(1, min(int(limit or 20), 100))
    return {
        "found": True,
        "registry_path": str(_delivery_resource_registry_path(resource_root)),
        "revision_count": len(revisions),
        "revisions": revisions[:clean_limit],
        "next_step": (
            "Use rollback dry-run before restoring a previous resource registry."
            if revisions
            else "No resource registry revisions exist yet."
        ),
    }


def disable_delivery_resource(
    resource_root: Path,
    resource_type: str,
    resource_id: str,
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Disable one shared resource so managed-object bindings stop passing readiness."""
    clean_type = str(resource_type or "").strip()
    clean_id = str(resource_id or "").strip()
    if clean_type not in DELIVERY_RESOURCE_TYPES:
        return {"accepted": False, "reason": "unsupported_resource_type", "resource_type": clean_type}
    if not clean_id:
        return {"accepted": False, "reason": "resource_id_required"}
    registry = load_delivery_resource_registry(resource_root)
    resources = _mapping(registry.get("delivery_resources"))
    bucket = _mapping(resources.get(clean_type))
    resource = _mapping(bucket.get(clean_id))
    if not resource:
        return {"accepted": False, "reason": "resource_not_found", "resource_type": clean_type, "resource_id": clean_id}
    revision = _snapshot_delivery_resource_registry_revision(
        resource_root,
        registry,
        action="resource_disable",
        operator_id=operator_id,
        reason=reason,
    )
    now = time.time()
    resource["publish_status"] = "disabled"
    resource["disabled_at"] = now
    resource["disabled_by"] = str(operator_id or "")
    resource["disable_reason"] = str(reason or "")
    resource["updated_at"] = now
    resource["updated_by"] = str(operator_id or "")
    resource["update_reason"] = str(reason or "")
    bucket[clean_id] = resource
    resources[clean_type] = bucket
    _write_delivery_resource_registry(resource_root, resources, operator_id=operator_id)
    return {
        "accepted": True,
        "resource_type": clean_type,
        "resource_id": clean_id,
        "resource": _delivery_resource_descriptor(
            clean_id,
            clean_type,
            resource,
            source_default="shared_registry",
        ),
        "revision": _delivery_resource_revision_public_payload(revision),
        "next_step": "Review managed-object bindings that still reference this disabled resource.",
    }


def rollback_delivery_resource_registry(
    resource_root: Path,
    revision_id: str,
    *,
    operator_id: str = "",
    reason: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore the shared resource registry from a saved revision."""
    target_revision_id = str(revision_id or "").strip()
    if not target_revision_id:
        return {"accepted": False, "reason": "revision_id_required"}
    revision = _find_delivery_resource_revision(resource_root, target_revision_id)
    if revision is None:
        return {"accepted": False, "reason": "revision_not_found"}
    target_registry = _mapping(revision.get("registry"))
    resources = _mapping(target_registry.get("delivery_resources"))
    if not resources:
        return {"accepted": False, "reason": "revision_missing_delivery_resources"}
    current = load_delivery_resource_registry(resource_root)
    payload = {
        "accepted": True,
        "dry_run": bool(dry_run),
        "registry_path": str(_delivery_resource_registry_path(resource_root)),
        "revision": _delivery_resource_revision_public_payload(revision),
        "current_registry_sha256": _sha256_json({
            "registry_type": "askme.delivery_resource_registry",
            "registry_version": int(current.get("registry_version") or 1),
            "delivery_resources": _mapping(current.get("delivery_resources")),
        }),
        "target_registry_sha256": str(revision.get("registry_sha256") or _sha256_json(target_registry)),
        "target_summary": _delivery_resource_registry_summary(
            _delivery_resource_rows(
                _delivery_resource_catalog(
                    target_registry,
                    include_defaults=False,
                    include_shared=False,
                ),
                [],
            ),
            [],
        ),
    }
    if dry_run:
        payload["would_write"] = True
        return payload
    snapshot = _snapshot_delivery_resource_registry_revision(
        resource_root,
        current,
        action="resource_registry_rollback_current",
        operator_id=operator_id,
        reason=reason or f"Rollback to resource registry revision {target_revision_id}.",
    )
    _write_delivery_resource_registry(resource_root, resources, operator_id=operator_id)
    payload["rollback_snapshot"] = _delivery_resource_revision_public_payload(snapshot)
    payload["next_step"] = "Reload the project resource catalog and review impacted object bindings."
    return payload


def create_delivery_resource_governance_request(
    resource_root: Path,
    action: str,
    operation: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
    profile_root: Path = Path("deploy/site-profiles"),
    template_root: Path | None = Path("deploy/customer-project-templates"),
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
    registry = load_delivery_resource_registry(resource_root)
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
        "current_registry_sha256": _sha256_json({
            "registry_type": "askme.delivery_resource_registry",
            "registry_version": int(registry.get("registry_version") or 1),
            "delivery_resources": _mapping(registry.get("delivery_resources")),
        }),
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


def build_site_profile_report(path: Path, *, check_env: bool = False) -> dict[str, Any]:
    profile = load_field_site_profile(path)
    report = validate_field_site_profile(profile, check_env=check_env)
    report["profile_path"] = str(path)
    if report["status"] == "passed":
        report["field_operations_config"] = field_operations_config_from_site_profile(profile)
    return report


def build_site_profile_catalog(
    root: Path,
    *,
    check_env: bool = False,
    pattern: str = "*.yaml",
) -> dict[str, Any]:
    """Return a customer-site catalog for multi-site field deployments."""
    root = Path(root)
    profile_paths = _site_profile_paths(root, pattern=pattern)
    sites = [_site_profile_catalog_item(path, check_env=check_env) for path in profile_paths]
    sites.sort(key=lambda item: (str(item.get("site_id") or ""), str(item.get("profile_path") or "")))
    passed = [item for item in sites if item.get("status") == "passed"]
    blocked = [item for item in sites if item.get("status") != "passed"]
    production_ready = [item for item in sites if item.get("deployment_stage") == "production_ready"]
    env_missing = sum(int(item.get("env_missing_count") or 0) for item in sites)
    customer_summary = _customer_project_summary(sites)
    return {
        "root": str(root),
        "check_env": bool(check_env),
        "sites": sites,
        "summary": {
            "site_count": len(sites),
            "configured_count": len(passed),
            "blocked_count": len(blocked),
            "production_ready_count": len(production_ready),
            "env_missing_count": env_missing,
            **customer_summary,
            "multi_site_ready": bool(sites) and not blocked,
        },
        "customer_claim": (
            "Customer projects can define their own sites, devices, responder groups, and managed objects."
            if sites and not blocked
            else "Some customer project profiles are missing required deployment configuration."
        ),
        "next_step": _site_catalog_next_step(sites, check_env=check_env),
    }


def build_customer_project_catalog(
    root: Path,
    *,
    check_env: bool = False,
    pattern: str = "*.yaml",
    tenant_id: str = "",
    delivery_namespace: str = "",
    customer_id: str = "",
    project_id: str = "",
    site_id: str = "",
    industry: str = "",
    gate_status: str = "",
    deployment_stage: str = "",
) -> dict[str, Any]:
    """Return product-facing customer/project/object coverage for solution rollout."""
    catalog = build_site_profile_catalog(root, check_env=check_env, pattern=pattern)
    sites = catalog.get("sites") if isinstance(catalog.get("sites"), list) else []
    projects: list[dict[str, Any]] = []
    for site in sites:
        customer = _mapping(site.get("customer"))
        project = {
            "tenant_id": customer.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "customer_name": customer.get("customer_name") or "Unassigned customer",
            "industry": customer.get("industry") or "unspecified",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "project_name": customer.get("project_name") or site.get("site_name") or "",
            "site_id": site.get("site_id") or "",
            "site_name": site.get("site_name") or "",
            "delivery_model": customer.get("delivery_model") or "solution_project",
            "deployment_stage": site.get("deployment_stage") or "blocked",
            "status": site.get("status") or "failed",
            "managed_objects_summary": site.get("managed_objects_summary") or {},
            "managed_objects": site.get("managed_objects") or [],
            "object_change_log": site.get("object_change_log") or [],
            "delivery_workflow": site.get("delivery_workflow") or {},
            "env_missing_count": int(site.get("env_missing_count") or 0),
            "next_step": site.get("next_step") or "",
        }
        project["product_acceptance_gate"] = _customer_project_product_acceptance_gate(project)
        projects.append(project)
    filters = _customer_project_catalog_filters(
        tenant_id=tenant_id,
        delivery_namespace=delivery_namespace,
        customer_id=customer_id,
        project_id=project_id,
        site_id=site_id,
        industry=industry,
        gate_status=gate_status,
        deployment_stage=deployment_stage,
    )
    if filters:
        projects = [
            project for project in projects if _customer_project_matches_filters(project, filters)
        ]
    delivery_acceptance_gate = customer_project_catalog_acceptance_gate(projects)
    summary = customer_project_catalog_summary_from_projects(
        projects,
        base_summary=catalog.get("summary") if isinstance(catalog.get("summary"), dict) else {},
    )
    if filters:
        summary["filtered"] = True
        summary["filters"] = filters
    return {
        "root": catalog.get("root"),
        "check_env": catalog.get("check_env"),
        "filters": filters,
        "summary": summary,
        "delivery_acceptance_gate": delivery_acceptance_gate,
        "projects": projects,
        "customers": _customer_rows(projects),
        "customer_claim": (
            "AskMe is configured as a repeatable solution product: each customer project declares its own managed objects."
        ),
        "next_step": catalog.get("next_step") or _site_catalog_next_step(sites, check_env=check_env),
    }


def customer_project_catalog_acceptance_gate(projects: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the aggregate signoff gate for a filtered customer-project set."""

    return _customer_project_catalog_delivery_acceptance_gate(projects)


def customer_project_catalog_summary_from_projects(
    projects: list[dict[str, Any]],
    *,
    base_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute customer-project summary after filters or permission scope."""

    summary = dict(base_summary or {})
    categories = sorted({
        str(category)
        for project in projects
        for category in (
            project.get("managed_objects_summary", {}).get("categories", [])
            if isinstance(project.get("managed_objects_summary"), dict)
            else []
        )
    })
    gate = customer_project_catalog_acceptance_gate(projects)
    summary.update({
        "site_count": len(projects),
        "configured_count": len([item for item in projects if item.get("status") == "passed"]),
        "blocked_count": len([item for item in projects if item.get("status") != "passed"]),
        "production_ready_count": len([
            item for item in projects if item.get("deployment_stage") == "production_ready"
        ]),
        "env_missing_count": sum(int(item.get("env_missing_count") or 0) for item in projects),
        "tenant_count": len({
            str(item.get("tenant_id") or "") for item in projects if item.get("tenant_id")
        }),
        "delivery_namespace_count": len({
            str(item.get("delivery_namespace") or "")
            for item in projects
            if item.get("delivery_namespace")
        }),
        "customer_count": len({
            str(item.get("customer_id") or "") for item in projects if item.get("customer_id")
        }),
        "project_count": len({
            str(item.get("project_id") or "") for item in projects if item.get("project_id")
        }),
        "industry_count": len({
            str(item.get("industry") or "") for item in projects if item.get("industry")
        }),
        "managed_object_type_count": sum(
            int(
                (
                    item.get("managed_objects_summary", {})
                    if isinstance(item.get("managed_objects_summary"), dict)
                    else {}
                ).get("object_type_count")
                or 0
            )
            for item in projects
        ),
        "managed_object_categories": categories,
        "multi_site_ready": bool(projects) and not gate["blocked_count"],
        "delivery_acceptance_gate_status": gate["overall_status"],
        "delivery_acceptance_blocked_count": gate["blocked_count"],
        "delivery_acceptance_manual_check_count": gate["manual_check_count"],
        "delivery_acceptance_ready_count": gate["ready_count"],
    })
    return summary


def _customer_project_catalog_filters(**values: str) -> dict[str, str]:
    return {
        key: str(value or "").strip()
        for key, value in values.items()
        if str(value or "").strip()
    }


def _customer_project_matches_filters(project: dict[str, Any], filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        if key == "gate_status":
            value = str(_mapping(project.get("product_acceptance_gate")).get("overall_status") or "")
        else:
            value = str(project.get(key) or "")
        if not _text_filter_matches(value, expected):
            return False
    return True


def _text_filter_matches(value: str, expected: str) -> bool:
    expected = str(expected or "").strip().lower()
    if not expected:
        return True
    options = [item.strip() for item in expected.split(",") if item.strip()]
    if not options:
        return True
    lower_value = str(value or "").strip().lower()
    return any(option in lower_value for option in options)


def list_customer_project_templates(
    root: Path,
    *,
    tenant_id: str = "",
    delivery_namespace: str = "",
    industry: str = "",
    publish_status: str = "",
    product_status: str = "",
    template_id: str = "",
    release_channel: str = "",
    owner: str = "",
) -> dict[str, Any]:
    """Return reusable industry templates for solution-provider rollout."""
    root = Path(root)
    templates = []
    filters = _customer_project_template_filters(
        tenant_id=tenant_id,
        delivery_namespace=delivery_namespace,
        industry=industry,
        publish_status=publish_status,
        product_status=product_status,
        template_id=template_id,
        release_channel=release_channel,
        owner=owner,
    )
    for path in _site_profile_paths(root, pattern="*.yaml"):
        try:
            profile = load_field_site_profile(path)
            report = validate_field_site_profile(profile)
            template = _mapping(profile.get("template"))
            customer = _mapping(profile.get("customer"))
            tenant = _delivery_tenant_id(customer)
            namespace = _delivery_namespace(customer)
            managed_objects = managed_object_catalog_from_site_profile(profile)
            delivery_summary = _template_delivery_summary(
                template=template,
                customer=customer,
                managed_objects=managed_objects,
                report=report,
            )
            customer_delivery = _customer_delivery_surface(
                profile=profile,
                template=template,
                customer=customer,
                managed_objects=managed_objects,
                report=report,
                surface="template",
            )
            delivery_summary = delivery_summary | customer_delivery
            template_package = _template_package_summary(
                profile=profile,
                template=template,
                path=path,
                report=report,
                delivery_summary=delivery_summary,
            )
            templates.append(
                {
                    "template_id": str(template.get("template_id") or path.stem),
                    "display_name": str(template.get("display_name") or path.stem),
                    "tenant_id": tenant,
                    "delivery_namespace": namespace,
                    "industry": str(customer.get("industry") or template.get("industry") or "unspecified"),
                    "template_version": template_package["version"],
                    "publish_status": template_package["publish_status"],
                    "release_channel": template_package["release_channel"],
                    "owner": template_package["owner"],
                    "product_status": template_package["product_status"],
                    "template_path": str(path),
                    "status": report.get("status"),
                    "errors": report.get("errors") or [],
                    "warnings": report.get("warnings") or [],
                    "managed_objects_summary": managed_objects | {
                        "objects": managed_objects["objects"][:8],
                        "objects_by_id": {},
                    },
                    "delivery_summary": delivery_summary,
                    "applicability_scope": customer_delivery["applicability_scope"],
                    "out_of_scope": customer_delivery["out_of_scope"],
                    "customer_prerequisites": customer_delivery["customer_prerequisites"],
                    "scenario_acceptance_criteria": customer_delivery["scenario_acceptance_criteria"],
                    "dependency_matrix": customer_delivery["dependency_matrix"],
                    "delivery_checklist": _template_delivery_checklist(delivery_summary),
                    "template_package": template_package,
                    "customer_claim": str(
                        template.get("customer_claim")
                        or "Reusable customer project starter for this industry."
                    ),
                    "next_step": str(
                        template.get("next_step")
                        or "Create a customer project from this template, then bind real devices and credentials."
                    ),
                }
            )
        except Exception as exc:
            templates.append(
                {
                    "template_id": path.stem,
                    "display_name": path.stem,
                    "tenant_id": DEFAULT_DELIVERY_NAMESPACE,
                    "delivery_namespace": DEFAULT_DELIVERY_NAMESPACE,
                    "industry": "unknown",
                    "template_version": "0.0.0",
                    "publish_status": "blocked",
                    "release_channel": "blocked",
                    "owner": "unassigned",
                    "product_status": "blocked",
                    "template_path": str(path),
                    "status": "failed",
                    "errors": [str(exc)],
                    "warnings": [],
                    "managed_objects_summary": {},
                    "delivery_summary": _template_delivery_summary(
                        template={},
                        customer={},
                        managed_objects={},
                        report={"status": "failed", "errors": [str(exc)], "warnings": []},
                    ),
                    "applicability_scope": _customer_delivery_applicability_scope(
                        template={},
                        customer={},
                        managed_objects={},
                        surface="template",
                    ),
                    "out_of_scope": _customer_delivery_out_of_scope("template"),
                    "customer_prerequisites": [],
                    "scenario_acceptance_criteria": [],
                    "dependency_matrix": [],
                    "delivery_checklist": _template_delivery_checklist(
                        _template_delivery_summary(
                            template={},
                            customer={},
                            managed_objects={},
                            report={"status": "failed", "errors": [str(exc)], "warnings": []},
                        )
                    ),
                    "template_package": _template_package_summary(
                        profile={},
                        template={"template_id": path.stem, "version": "0.0.0", "publish_status": "blocked"},
                        path=path,
                        report={"status": "failed", "errors": [str(exc)], "warnings": []},
                        delivery_summary={},
                    ),
                    "customer_claim": "Template cannot be used until validation errors are fixed.",
                    "next_step": "Fix template YAML and rerun validation.",
                }
            )
    if filters:
        templates = [
            item
            for item in templates
            if _customer_project_template_matches_filters(item, filters)
        ]
    templates.sort(key=lambda item: (str(item.get("industry") or ""), str(item.get("template_id") or "")))
    summary = customer_project_template_summary_from_items(templates)
    if filters:
        summary["filtered"] = True
        summary["filters"] = filters
    return {
        "root": str(root),
        "filters": filters,
        "templates": templates,
        "summary": summary,
        "customer_claim": (
            "Industry templates let delivery teams start a new customer project without custom code."
        ),
    }


def customer_project_template_summary_from_items(templates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return template-market summary for the current filtered result set."""

    return {
        "template_count": len(templates),
        "valid_count": len([item for item in templates if item.get("status") == "passed"]),
        "product_ready_count": len([
            item
            for item in templates
            if _mapping(item.get("template_package")).get("product_status") == "ready"
        ]),
        "manual_check_count": len([
            item
            for item in templates
            if _mapping(item.get("template_package")).get("product_status") == "manual_check"
        ]),
        "blocked_count": len([
            item
            for item in templates
            if _mapping(item.get("template_package")).get("product_status") == "blocked"
        ]),
        "tenant_count": len({
            str(item.get("tenant_id") or "") for item in templates if item.get("tenant_id")
        }),
        "delivery_namespace_count": len({
            str(item.get("delivery_namespace") or "")
            for item in templates
            if item.get("delivery_namespace")
        }),
        "industry_count": len({
            str(item.get("industry") or "") for item in templates if item.get("industry")
        }),
        "publish_statuses": sorted({
            str(item.get("publish_status") or "")
            for item in templates
            if item.get("publish_status")
        }),
        "product_statuses": sorted({
            str(_mapping(item.get("template_package")).get("product_status") or item.get("product_status") or "")
            for item in templates
            if _mapping(item.get("template_package")).get("product_status") or item.get("product_status")
        }),
        "managed_object_type_count": sum(
            int(_mapping(item.get("managed_objects_summary")).get("object_type_count") or 0)
            for item in templates
        ),
    }


def _customer_project_template_filters(**values: str) -> dict[str, str]:
    return {
        key: str(value or "").strip()
        for key, value in values.items()
        if str(value or "").strip()
    }


def _customer_project_template_matches_filters(
    template: dict[str, Any],
    filters: dict[str, str],
) -> bool:
    package = _mapping(template.get("template_package"))
    for key, expected in filters.items():
        value = str(package.get(key) or template.get(key) or "")
        if not _text_filter_matches(value, expected):
            return False
    return True


def build_solution_delivery_readiness(
    *,
    project_catalog: dict[str, Any],
    template_catalog: dict[str, Any],
    resource_catalog: dict[str, Any],
    governance_requests: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one product-facing readiness gate for solution-provider delivery."""

    governance_requests = governance_requests or {}
    gates = [
        _solution_delivery_customer_project_gate(project_catalog),
        _solution_delivery_template_market_gate(template_catalog),
        _solution_delivery_resource_binding_gate(resource_catalog),
        _solution_delivery_resource_governance_gate(governance_requests),
    ]
    overall_status = _delivery_gate_rollup_status(gates)
    blockers = [
        str(gate.get("next_step") or gate.get("evidence") or "")
        for gate in gates
        if gate.get("status") == "blocked"
    ]
    manual_checks = [
        str(gate.get("next_step") or gate.get("evidence") or "")
        for gate in gates
        if gate.get("status") == "manual_check"
    ]
    return {
        "readiness_type": "askme.solution_delivery_readiness",
        "overall_status": overall_status,
        "production_ready": overall_status == "ready",
        "customer_status": {
            "ready": "可进入客户试点验收；仍需按项目执行现场验收和客户签收。",
            "manual_check": "可用于客户方案演示或试点准备，但仍有交付负责人需要复核的项目。",
            "blocked": "不能对客户承诺验收通过；需要先处理阻塞项。",
        }[overall_status],
        "release_claim": {
            "ready": "可以声明具备受控试点交付能力，不能替代现场最终验收。",
            "manual_check": "只能声明演示或试点准备能力，不能声明客户验收完成。",
            "blocked": "不能声明客户可验收或可上线。",
        }[overall_status],
        "gates": gates,
        "summary": {
            "gate_count": len(gates),
            "ready_count": len([gate for gate in gates if gate.get("status") == "ready"]),
            "manual_check_count": len([
                gate for gate in gates if gate.get("status") == "manual_check"
            ]),
            "blocked_count": len([gate for gate in gates if gate.get("status") == "blocked"]),
            "project_count": int(_mapping(project_catalog.get("summary")).get("project_count") or 0),
            "template_count": int(_mapping(template_catalog.get("summary")).get("template_count") or 0),
            "resource_count": int(_mapping(resource_catalog.get("summary")).get("resource_count") or 0),
        },
        "blockers": blockers,
        "manual_checks": manual_checks,
        "next_step": (
            blockers[0]
            if blockers
            else manual_checks[0]
            if manual_checks
            else "Continue with onsite acceptance evidence, customer review, and signed handoff."
        ),
    }


def _solution_delivery_customer_project_gate(project_catalog: dict[str, Any]) -> dict[str, Any]:
    gate = _mapping(project_catalog.get("delivery_acceptance_gate"))
    project_count = int(gate.get("project_count") or _mapping(project_catalog.get("summary")).get("project_count") or 0)
    status = str(gate.get("overall_status") or "blocked")
    if project_count <= 0:
        status = "blocked"
    return {
        "gate_id": "customer_project_acceptance",
        "label": "客户项目验收门禁",
        "status": status,
        "evidence": (
            f"{project_count} project(s), ready={gate.get('ready_count', 0)}, "
            f"manual={gate.get('manual_check_count', 0)}, blocked={gate.get('blocked_count', 0)}"
        ),
        "next_step": str(
            gate.get("next_step")
            or "Create and verify at least one customer project before delivery."
        ),
    }


def _solution_delivery_template_market_gate(template_catalog: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(template_catalog.get("summary"))
    template_count = int(summary.get("template_count") or 0)
    ready = int(summary.get("product_ready_count") or 0)
    manual = int(summary.get("manual_check_count") or 0)
    blocked = int(summary.get("blocked_count") or 0)
    if template_count <= 0 or blocked >= template_count:
        status = "blocked"
    elif manual or blocked or ready <= 0:
        status = "manual_check"
    else:
        status = "ready"
    return {
        "gate_id": "template_market",
        "label": "行业模板市场",
        "status": status,
        "evidence": f"{template_count} template(s), ready={ready}, manual={manual}, blocked={blocked}",
        "next_step": (
            "Publish or approve at least one reusable template before customer rollout."
            if status == "blocked"
            else "Review pilot/manual template releases before creating customer projects."
            if status == "manual_check"
            else "Use published templates to seed customer projects."
        ),
    }


def _solution_delivery_resource_binding_gate(resource_catalog: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(resource_catalog.get("summary"))
    resource_count = int(summary.get("resource_count") or 0)
    unregistered = int(summary.get("unregistered_resource_count") or 0)
    overall = str(summary.get("overall_status") or "blocked")
    if resource_count <= 0:
        status = "blocked"
    elif unregistered or overall != "ready":
        status = "manual_check"
    else:
        status = "ready"
    return {
        "gate_id": "delivery_resource_bindings",
        "label": "模型/协议/技能/验收资源绑定",
        "status": status,
        "evidence": (
            f"{resource_count} resource(s), consumers={summary.get('consumer_count', 0)}, "
            f"unregistered={unregistered}"
        ),
        "next_step": str(
            resource_catalog.get("next_step")
            or "Register missing delivery resources before project signoff."
        ),
    }


def _solution_delivery_resource_governance_gate(
    governance_requests: dict[str, Any],
) -> dict[str, Any]:
    if governance_requests.get("skipped"):
        return {
            "gate_id": "delivery_resource_governance",
            "label": "共享资源治理队列",
            "status": "manual_check",
            "evidence": str(governance_requests.get("reason") or "governance queue not visible"),
            "next_step": "Use an unrestricted delivery owner to review shared resource governance queue.",
        }
    summary = _mapping(governance_requests.get("summary"))
    pending = int(summary.get("pending_count") or 0)
    due_soon = int(summary.get("due_soon_count") or 0)
    overdue = int(summary.get("overdue_count") or 0)
    if overdue:
        status = "blocked"
    elif pending or due_soon:
        status = "manual_check"
    else:
        status = "ready"
    return {
        "gate_id": "delivery_resource_governance",
        "label": "共享资源治理队列",
        "status": status,
        "evidence": f"pending={pending}, due_soon={due_soon}, overdue={overdue}",
        "next_step": (
            "Escalate overdue resource governance requests before customer signoff."
            if overdue
            else "Review pending shared resource governance requests."
            if pending or due_soon
            else "No open shared-resource governance blockers."
        ),
    }


def _delivery_gate_rollup_status(gates: list[dict[str, Any]]) -> str:
    statuses = {str(gate.get("status") or "blocked") for gate in gates}
    if "blocked" in statuses:
        return "blocked"
    if "manual_check" in statuses:
        return "manual_check"
    return "ready"


def list_customer_project_template_revisions(
    root: Path,
    template_id: str,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return release-governance history for one reusable industry template."""
    path = _find_template_path(root, template_id)
    if path is None:
        return {
            "found": False,
            "reason": "template_not_found",
            "template_id": str(template_id or ""),
            "revisions": [],
        }
    profile = load_field_site_profile(path)
    template = _mapping(profile.get("template"))
    report = validate_field_site_profile(profile)
    managed_objects = managed_object_catalog_from_site_profile(profile)
    delivery_summary = _template_delivery_summary(
        template=template,
        customer=_mapping(profile.get("customer")),
        managed_objects=managed_objects,
        report=report,
    )
    revisions = _load_customer_project_template_revisions(root, path, profile)
    return {
        "found": True,
        "template_id": str(template.get("template_id") or path.stem),
        "template_path": str(path),
        "template_package": _template_package_summary(
            profile=profile,
            template=template,
            path=path,
            report=report,
            delivery_summary=delivery_summary,
        ),
        "revisions": revisions[: max(0, int(limit or 20))],
        "revision_count": len(revisions),
    }


def update_customer_project_template_release(
    root: Path,
    template_id: str,
    release: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Update template release metadata with a reversible audit snapshot."""
    path = _find_template_path(root, template_id)
    if path is None:
        return {
            "accepted": False,
            "reason": "template_not_found",
            "template_id": str(template_id or ""),
        }
    if not isinstance(release, dict):
        return {
            "accepted": False,
            "reason": "release_payload_required",
            "template_id": str(template_id or ""),
            "template_path": str(path),
        }
    profile = load_field_site_profile(path)
    template = _mapping(profile.get("template"))
    current_template_id = str(template.get("template_id") or path.stem)
    next_status = str(release.get("publish_status") or template.get("publish_status") or "draft").strip()
    if next_status not in TEMPLATE_PUBLISH_STATUSES:
        return {
            "accepted": False,
            "reason": "invalid_publish_status",
            "template_id": current_template_id,
            "template_path": str(path),
            "allowed_publish_statuses": sorted(TEMPLATE_PUBLISH_STATUSES),
        }
    next_version = str(release.get("version") or template.get("version") or "0.0.0").strip()
    if not _is_semver(next_version):
        return {
            "accepted": False,
            "reason": "invalid_template_version",
            "template_id": current_template_id,
            "template_path": str(path),
            "version": next_version,
        }

    updated = copy.deepcopy(profile)
    updated_template = _mapping(updated.get("template"))
    updated_template.setdefault("template_id", current_template_id)
    for field in TEMPLATE_RELEASE_FIELDS:
        if field in release:
            updated_template[field] = str(release.get(field) or "").strip()
    updated_template["version"] = next_version
    updated_template["publish_status"] = next_status
    updated_template["release_channel"] = str(
        release.get("release_channel")
        or updated_template.get("release_channel")
        or next_status
    ).strip()
    updated_template["release_updated_by"] = str(operator_id or "system")
    updated_template["release_updated_at"] = time.time()
    if reason or release.get("reason"):
        updated_template["release_reason"] = str(reason or release.get("reason") or "")
    updated["template"] = _clean_nested_mapping(updated_template)

    report = validate_field_site_profile(updated)
    managed_objects = managed_object_catalog_from_site_profile(updated)
    delivery_summary = _template_delivery_summary(
        template=_mapping(updated.get("template")),
        customer=_mapping(updated.get("customer")),
        managed_objects=managed_objects,
        report=report,
    )
    template_package = _template_package_summary(
        profile=updated,
        template=_mapping(updated.get("template")),
        path=path,
        report=report,
        delivery_summary=delivery_summary,
    )
    if report.get("status") != "passed":
        return {
            "accepted": False,
            "reason": "updated_template_validation_failed",
            "template_id": current_template_id,
            "template_path": str(path),
            "errors": report.get("errors") or [],
            "warnings": report.get("warnings") or [],
            "template_package": template_package,
        }

    revision: dict[str, Any] = {}
    if not dry_run:
        revision = _snapshot_customer_project_template_revision(
            root,
            path,
            action=f"release_{next_status}",
            operator_id=operator_id,
            reason=reason or str(release.get("reason") or ""),
        )
        _write_yaml(path, updated)
    return {
        "accepted": True,
        "dry_run": bool(dry_run),
        "template_id": current_template_id,
        "template_path": str(path),
        "template": _mapping(updated.get("template")),
        "template_package": template_package,
        "delivery_summary": delivery_summary,
        "delivery_checklist": _template_delivery_checklist(delivery_summary),
        "revision": revision,
        "next_step": template_package.get("next_step") or "Review template release state.",
    }


def create_customer_project_template_release_request(
    root: Path,
    template_id: str,
    release: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Create a pending release request without changing the template."""
    preview = update_customer_project_template_release(
        root,
        template_id,
        release,
        operator_id=operator_id,
        reason=reason,
        dry_run=True,
    )
    if not preview.get("accepted"):
        return {
            "accepted": False,
            "reason": preview.get("reason") or "release_request_invalid",
            "template_id": preview.get("template_id") or str(template_id or ""),
            "template_path": preview.get("template_path") or "",
            "preview": preview,
        }
    profile = load_field_site_profile(Path(preview["template_path"]))
    created_at = time.time()
    request_id = _slug(
        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created_at))}-"
        f"{int(created_at * 1000)}-{preview['template_id']}-{preview['template_package']['publish_status']}"
    )
    request_payload = {
        "request_type": "askme.customer_project_template_release_request",
        "request_version": 1,
        "request_id": request_id,
        "status": "pending",
        "template_id": preview["template_id"],
        "template_path": preview["template_path"],
        "requested_by": str(operator_id or "system"),
        "requested_at": created_at,
        "reason": str(reason or release.get("reason") or ""),
        "release": _template_release_payload(release),
        "current_template_sha256": _sha256_json(profile),
        "current_template_package": list_customer_project_template_revisions(
            root,
            preview["template_id"],
            limit=0,
        ).get("template_package"),
        "proposed_template_package": preview.get("template_package"),
    }
    target_dir = _customer_project_template_release_request_dir(root, preview["template_id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{request_id}.json"
    target.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    request_payload["request_path"] = str(target)
    return {
        "accepted": True,
        "request": _template_release_request_public_payload(request_payload),
        "template_package": preview.get("template_package"),
        "next_step": "A second product owner must approve this template release before it is applied.",
    }


def list_customer_project_template_release_requests(
    root: Path,
    *,
    template_id: str = "",
    status: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """List pending and reviewed release requests for product-owner governance."""
    requests = [
        _template_release_request_public_payload(payload)
        for payload in _iter_customer_project_template_release_requests(root)
    ]
    if template_id:
        requests = [item for item in requests if item.get("template_id") == str(template_id)]
    if status:
        requests = [item for item in requests if item.get("status") == str(status)]
    requests.sort(key=lambda item: float(item.get("requested_at") or 0), reverse=True)
    capped = requests[: max(0, int(limit or 50))]
    return {
        "root": str(root),
        "template_id": str(template_id or ""),
        "status": str(status or ""),
        "requests": capped,
        "request_count": len(requests),
        "summary": {
            "pending_count": len([item for item in requests if item.get("status") == "pending"]),
            "approved_count": len([item for item in requests if item.get("status") == "approved"]),
            "rejected_count": len([item for item in requests if item.get("status") == "rejected"]),
        },
    }


def _template_release_note_delivery_details(root: Path, template_id: str) -> dict[str, Any]:
    path = _find_template_path(root, template_id)
    if path is None:
        surface = _customer_delivery_surface(
            profile={},
            template={"template_id": template_id},
            customer={},
            managed_objects={},
            report={"status": "failed", "errors": ["template_not_found"], "warnings": []},
            surface="template",
        )
        return {
            "delivery_summary": _template_delivery_summary(
                template={"template_id": template_id},
                customer={},
                managed_objects={},
                report={"status": "failed", "errors": ["template_not_found"], "warnings": []},
            ) | surface,
            **surface,
        }
    profile = load_field_site_profile(path)
    report = validate_field_site_profile(profile)
    template = _mapping(profile.get("template"))
    customer = _mapping(profile.get("customer"))
    managed_objects = managed_object_catalog_from_site_profile(profile)
    surface = _customer_delivery_surface(
        profile=profile,
        template=template,
        customer=customer,
        managed_objects=managed_objects,
        report=report,
        surface="template",
    )
    return {
        "delivery_summary": _template_delivery_summary(
            template=template,
            customer=customer,
            managed_objects=managed_objects,
            report=report,
        ) | surface,
        **surface,
    }


def customer_project_template_release_notes(
    root: Path,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    """Return customer-facing template release notes from approved requests."""
    requests = list_customer_project_template_release_requests(
        root,
        status="approved",
        limit=limit,
    ).get("requests", [])
    notes: list[dict[str, Any]] = []
    for request in requests:
        applied = _mapping(request.get("applied_template_package"))
        proposed = _mapping(request.get("proposed_template_package"))
        package = applied or proposed
        if str(package.get("publish_status") or "") != "published":
            continue
        template_id = str(request.get("template_id") or package.get("template_id") or "")
        delivery_details = _template_release_note_delivery_details(root, template_id)
        notes.append(
            {
                "release_note_id": str(request.get("request_id") or ""),
                "template_id": template_id,
                "version": str(package.get("version") or ""),
                "publish_status": str(package.get("publish_status") or ""),
                "release_channel": str(package.get("release_channel") or ""),
                "product_status": str(package.get("product_status") or ""),
                "customer_status": str(package.get("customer_status") or ""),
                "delivery_summary": delivery_details["delivery_summary"],
                "applicability_scope": delivery_details["applicability_scope"],
                "out_of_scope": delivery_details["out_of_scope"],
                "customer_prerequisites": delivery_details["customer_prerequisites"],
                "scenario_acceptance_criteria": delivery_details["scenario_acceptance_criteria"],
                "dependency_matrix": delivery_details["dependency_matrix"],
                "requested_by": str(request.get("requested_by") or ""),
                "requested_at": request.get("requested_at"),
                "approved_by": str(request.get("reviewed_by") or ""),
                "approved_at": request.get("reviewed_at"),
                "release_reason": str(request.get("review_reason") or request.get("reason") or ""),
                "customer_claim": (
                    "Approved reusable template package. Delivery must still bind customer scope, live devices, "
                    "credentials, and onsite acceptance evidence before production launch."
                ),
            }
        )
    notes.sort(key=lambda item: float(item.get("approved_at") or 0), reverse=True)
    return {
        "root": str(root),
        "notes": notes,
        "summary": {
            "approved_release_count": len(notes),
            "template_count": len({item["template_id"] for item in notes if item.get("template_id")}),
            "manual_check_count": len([item for item in notes if item.get("product_status") == "manual_check"]),
            "ready_count": len([item for item in notes if item.get("product_status") == "ready"]),
        },
        "customer_claim": "Only approved published template packages appear in these release notes.",
    }


def export_customer_project_template_release_notes_bundle(
    root: Path,
    *,
    customer_context: dict[str, Any] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Build a portable proposal/handoff bundle from approved template releases."""
    notes_payload = customer_project_template_release_notes(root, limit=limit)
    notes = notes_payload.get("notes") if isinstance(notes_payload.get("notes"), list) else []
    context = _release_notes_customer_context(customer_context or {})
    bundle_base: dict[str, Any] = {
        "bundle_schema": "askme.template_release_notes_bundle.v1",
        "generated_at": time.time(),
        "customer_context": context,
        "summary": notes_payload.get("summary") or {},
        "release_notes": notes,
        "proposal_insert": _template_release_notes_proposal_insert(
            context,
            notes,
            _mapping(notes_payload.get("summary")),
        ),
        "release_note_count": len(notes),
        "customer_claim": notes_payload.get("customer_claim")
        or "Only approved published template packages appear in these release notes.",
        "delivery_boundary": (
            "This bundle is suitable for proposal and pilot handoff only. "
            "Delivery must still bind customer scope, live devices, credentials, "
            "runtime evidence, and onsite acceptance before production launch."
        ),
        "files": {
            "json_filename": f"{_release_notes_bundle_slug(context)}-template-release-notes.json",
            "html_filename": f"{_release_notes_bundle_slug(context)}-template-release-notes.html",
        },
    }
    manifest = {
        "manifest_version": 1,
        "bundle_sha256": _sha256_json(bundle_base),
        "release_note_count": len(notes),
        "template_count": int(_mapping(bundle_base.get("summary")).get("template_count") or 0),
        "customer_name": str(context.get("customer_name") or ""),
        "project_name": str(context.get("project_name") or ""),
    }
    bundle = {
        **bundle_base,
        "manifest": manifest,
    }
    bundle["html"] = _template_release_notes_bundle_html(bundle)
    return {
        "accepted": True,
        "bundle": bundle,
        "summary": bundle["summary"],
        "next_step": (
            "Attach the JSON or HTML bundle to a customer proposal after confirming onsite delivery scope."
            if notes
            else "No approved published template release notes are available for export yet."
        ),
    }


def review_customer_project_template_release_request(
    root: Path,
    request_id: str,
    *,
    decision: str,
    operator_id: str = "",
    reason: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Approve or reject a pending template release request."""
    request_path, request_payload = _find_customer_project_template_release_request(root, request_id)
    if not request_payload or request_path is None:
        return {
            "accepted": False,
            "reason": "release_request_not_found",
            "request_id": str(request_id or ""),
        }
    if str(request_payload.get("status") or "") != "pending":
        return {
            "accepted": False,
            "reason": "release_request_not_pending",
            "request": _template_release_request_public_payload(request_payload),
        }
    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in {"approve", "reject"}:
        return {
            "accepted": False,
            "reason": "invalid_release_review_decision",
            "allowed_decisions": ["approve", "reject"],
            "request": _template_release_request_public_payload(request_payload),
        }
    reviewer = str(operator_id or "system")
    if reviewer == str(request_payload.get("requested_by") or ""):
        return {
            "accepted": False,
            "reason": "release_request_requires_second_approver",
            "request": _template_release_request_public_payload(request_payload),
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
            "request": _template_release_request_public_payload(next_payload),
            "next_step": "Release request rejected. The template YAML was not changed.",
        }

    template_path = _find_template_path(root, str(request_payload.get("template_id") or ""))
    if template_path is None:
        return {
            "accepted": False,
            "reason": "template_not_found",
            "request": _template_release_request_public_payload(request_payload),
        }
    current_profile = load_field_site_profile(template_path)
    if _sha256_json(current_profile) != str(request_payload.get("current_template_sha256") or ""):
        return {
            "accepted": False,
            "reason": "template_changed_since_request",
            "request": _template_release_request_public_payload(request_payload),
        }
    release_result = update_customer_project_template_release(
        root,
        str(request_payload.get("template_id") or ""),
        _mapping(request_payload.get("release")),
        operator_id=reviewer,
        reason=reason or str(request_payload.get("reason") or ""),
        dry_run=dry_run,
    )
    if not release_result.get("accepted"):
        return {
            "accepted": False,
            "reason": release_result.get("reason") or "release_apply_failed",
            "request": _template_release_request_public_payload(request_payload),
            "release_result": release_result,
        }
    next_payload["status"] = "approved"
    next_payload["applied_template_package"] = release_result.get("template_package")
    next_payload["applied_revision"] = _template_revision_public_payload(
        _mapping(release_result.get("revision"))
    )
    if not dry_run:
        request_path.write_text(json.dumps(next_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "accepted": True,
        "dry_run": bool(dry_run),
        "request": _template_release_request_public_payload(next_payload),
        "release_result": release_result,
        "next_step": "Template release approved and applied.",
    }


def _template_delivery_summary(
    *,
    template: dict[str, Any],
    customer: dict[str, Any],
    managed_objects: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize one industry template as a reusable delivery product."""
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    acceptance = _mapping(managed_objects.get("acceptance_summary"))
    return {
        "template_status": str(report.get("status") or "failed"),
        "template_version": str(template.get("version") or "0.0.0"),
        "publish_status": str(template.get("publish_status") or "draft"),
        "release_channel": str(template.get("release_channel") or template.get("publish_status") or "draft"),
        "customer_fit": str(
            template.get("customer_fit")
            or template.get("customer_claim")
            or "Use when the customer needs this industry scenario as a repeatable starter."
        ),
        "default_object_count": int(managed_objects.get("object_type_count") or len(objects)),
        "default_objects": [
            {
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or "uncategorized"),
            }
            for item in objects[:6]
        ],
        "object_categories": _string_list(managed_objects.get("categories")),
        "scenario_ids": _string_list(managed_objects.get("scenario_ids")),
        "device_sources": _unique_template_object_values(objects, "device_sources"),
        "responder_groups": sorted({
            str(item.get("responder_group") or "")
            for item in objects
            if str(item.get("responder_group") or "").strip()
        }),
        "vision_models": _unique_template_binding_values(objects, "vision_models"),
        "sensor_protocols": _unique_template_binding_values(objects, "sensor_protocols"),
        "skill_packages": _unique_template_binding_values(objects, "skill_packages"),
        "acceptance_tests": _unique_template_binding_values(objects, "acceptance_tests"),
        "acceptance_status": str(acceptance.get("overall_status") or "blocked"),
        "ready_object_count": int(acceptance.get("ready_object_count") or 0),
        "manual_check_object_count": int(acceptance.get("manual_check_object_count") or 0),
        "blocked_object_count": int(acceptance.get("blocked_object_count") or 0),
        "delivery_boundary": (
            "Template is a starter package. Delivery must replace customer scope, bind the real map/devices/"
            "credentials, and run onsite acceptance before production claims."
        ),
    }


def _customer_delivery_surface(
    *,
    profile: dict[str, Any],
    template: dict[str, Any],
    customer: dict[str, Any],
    managed_objects: dict[str, Any],
    report: dict[str, Any],
    env_references: list[dict[str, Any]] | None = None,
    surface: str = "project",
) -> dict[str, Any]:
    """Return customer-readable delivery scope, prerequisites, and acceptance criteria."""
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    return {
        "applicability_scope": _customer_delivery_applicability_scope(
            template=template,
            customer=customer,
            managed_objects=managed_objects,
            surface=surface,
        ),
        "out_of_scope": _customer_delivery_out_of_scope(surface),
        "customer_prerequisites": _customer_delivery_prerequisites(
            profile,
            env_references=env_references,
            report=report,
        ),
        "scenario_acceptance_criteria": _customer_delivery_scenario_acceptance_criteria(objects),
        "dependency_matrix": _customer_delivery_dependency_matrix(objects),
    }


def _customer_delivery_applicability_scope(
    *,
    template: dict[str, Any],
    customer: dict[str, Any],
    managed_objects: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    industry = str(customer.get("industry") or template.get("industry") or "unspecified")
    categories = _string_list(managed_objects.get("categories"))
    scenarios = _string_list(managed_objects.get("scenario_ids"))
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    managed_object_types = sorted({
        str(item.get("category") or "uncategorized")
        for item in objects
        if str(item.get("category") or "").strip()
    } or set(categories))
    site_types = {
        "manufacturing": ["factory", "production line", "utility corridor"],
        "creative_park": ["creative park", "visitor service area", "mixed-use campus"],
        "warehouse": ["warehouse", "loading zone", "logistics aisle"],
        "scenic_area": ["scenic area", "visitor route", "service point"],
    }.get(industry, [industry.replace("_", " ") or "customer site"])
    return {
        "scope_type": "askme.customer_delivery_applicability_scope.v1",
        "surface": surface,
        "industries": [industry] if industry else [],
        "site_types": site_types,
        "scenarios": scenarios,
        "managed_object_types": managed_object_types,
        "default_object_count": int(managed_objects.get("object_type_count") or len(objects)),
        "default_objects": [
            {
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or "uncategorized"),
            }
            for item in objects[:8]
        ],
        "customer_fit": str(
            template.get("customer_fit")
            or template.get("customer_claim")
            or f"Use this {industry} starter when the customer needs repeatable robot service delivery."
        ),
    }


def _customer_delivery_out_of_scope(surface: str) -> list[str]:
    noun = "template" if surface == "template" else "customer project package"
    return [
        f"This {noun} is not a production go-live certificate.",
        "It does not replace onsite map validation, live device tests, notification tests, or robot runtime acceptance.",
        "It does not prove enterprise IAM/SSO, customer network, or responder credentials are production-ready.",
        "Open-domain chat and unsupervised hardware control are outside this delivery boundary.",
    ]


def _customer_delivery_prerequisites(
    profile: dict[str, Any],
    *,
    env_references: list[dict[str, Any]] | None,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    site = _mapping(profile.get("site"))
    devices = _mapping(profile.get("devices"))
    responders = _mapping(profile.get("responder_groups"))
    env_refs = env_references if isinstance(env_references, list) else site_profile_env_references(profile)
    required_env_count = len([item for item in env_refs if isinstance(item, dict) and item.get("required")])
    missing_env_count = len([
        item
        for item in env_refs
        if isinstance(item, dict) and item.get("required") and not item.get("configured")
    ])
    validation_status = str(report.get("status") or "failed")
    return [
        {
            "prerequisite_id": "site_map_and_routes",
            "label": "Site map and service routes",
            "owner": "customer operations + delivery engineer",
            "required": True,
            "status": "manual_check" if site.get("map_version") else "blocked",
            "evidence_required": ["map_version", "route list", "no-go zones"],
            "next_step": "Confirm the customer map, route scope, service points, and restricted areas onsite.",
        },
        {
            "prerequisite_id": "field_devices",
            "label": "Cameras, sensors, and robot sources",
            "owner": "delivery engineer",
            "required": True,
            "status": "manual_check" if devices else "blocked",
            "evidence_required": ["device inventory", "source id", "zone binding"],
            "next_step": "Bind every managed object to real camera, sensor, voice, or robot event sources.",
        },
        {
            "prerequisite_id": "credentials_and_notifications",
            "label": "Credentials and responder notification groups",
            "owner": "customer IT + customer operations",
            "required": True,
            "status": (
                "ready"
                if required_env_count and missing_env_count == 0
                else "manual_check"
                if required_env_count
                else "blocked"
            ),
            "evidence_required": ["secret env vars", "DingTalk/WeCom/Feishu test", "responder roster"],
            "next_step": "Configure live credentials and run notification smoke tests before handoff.",
            "required_env_count": required_env_count,
            "missing_env_count": missing_env_count,
            "responder_group_count": len(responders),
        },
        {
            "prerequisite_id": "onsite_acceptance_window",
            "label": "Onsite acceptance owner and test window",
            "owner": "delivery manager + customer signatory",
            "required": True,
            "status": "manual_check" if validation_status == "passed" else "blocked",
            "evidence_required": ["test schedule", "customer reviewer", "signed acceptance dossier"],
            "next_step": "Schedule onsite scenario tests and collect signed customer acceptance evidence.",
        },
    ]


def _customer_delivery_scenario_acceptance_criteria(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        display_name = str(obj.get("display_name") or object_id)
        evidence = _string_list(obj.get("evidence_required"))
        bindings = _mapping(obj.get("bindings"))
        acceptance_tests = _string_list(bindings.get("acceptance_tests"))
        for scenario_id in _string_list(obj.get("scenario_ids")):
            row = rows.setdefault(
                scenario_id,
                {
                    "scenario_id": scenario_id,
                    "managed_object_ids": [],
                    "managed_object_labels": [],
                    "required_evidence": [],
                    "acceptance_tests": [],
                    "pass_condition": (
                        "Accepted when live event evidence, notification/archive evidence, "
                        "and linked scenario tests are all reviewed for this customer site."
                    ),
                    "blocking_if_missing": False,
                },
            )
            row["managed_object_ids"].append(object_id)
            row["managed_object_labels"].append(display_name)
            row["required_evidence"].extend(evidence)
            row["acceptance_tests"].extend(acceptance_tests)
    result = []
    for row in rows.values():
        row["managed_object_ids"] = sorted(set(row["managed_object_ids"]))
        row["managed_object_labels"] = sorted(set(row["managed_object_labels"]))
        row["required_evidence"] = sorted(set(row["required_evidence"]))
        row["acceptance_tests"] = sorted(set(row["acceptance_tests"]))
        row["blocking_if_missing"] = not row["required_evidence"] or not row["acceptance_tests"]
        result.append(row)
    return sorted(result, key=lambda item: str(item.get("scenario_id") or ""))


def _customer_delivery_dependency_matrix(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        bindings = _mapping(obj.get("bindings"))
        checks = _mapping(obj.get("resource_binding_status")).get("checks")
        check_by_key = {
            (str(_mapping(item).get("resource_type") or ""), str(_mapping(item).get("resource_id") or "")): _mapping(item)
            for item in checks
            if isinstance(item, dict)
        } if isinstance(checks, list) else {}
        for resource_type in DELIVERY_RESOURCE_TYPES:
            for resource_id in _string_list(bindings.get(resource_type)):
                key = (resource_type, resource_id)
                check = check_by_key.get(key, {})
                row = rows.setdefault(
                    key,
                    {
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "customer_label": str(check.get("display_name") or resource_id),
                        "status": str(check.get("status") or "manual_check"),
                        "source": str(check.get("source") or ""),
                        "managed_object_ids": [],
                        "blocking_if_missing": str(check.get("status") or "") in {"blocked", "unregistered"},
                    },
                )
                row["managed_object_ids"].append(object_id)
                if str(check.get("status") or "") in {"blocked", "unregistered"}:
                    row["blocking_if_missing"] = True
    result = []
    for row in rows.values():
        row["managed_object_ids"] = sorted(set(row["managed_object_ids"]))
        result.append(row)
    return sorted(result, key=lambda item: (str(item.get("resource_type") or ""), str(item.get("resource_id") or "")))


def _template_package_summary(
    *,
    profile: dict[str, Any],
    template: dict[str, Any],
    path: Path,
    report: dict[str, Any],
    delivery_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return product-release metadata for one reusable industry template."""
    template_id = str(template.get("template_id") or path.stem)
    version = str(template.get("version") or "0.0.0")
    publish_status = str(template.get("publish_status") or "draft")
    release_channel = str(template.get("release_channel") or publish_status)
    blockers: list[str] = []
    manual_checks: list[str] = []
    if not re.match(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?$", version):
        blockers.append("Template version must use semantic version format.")
    if report.get("status") != "passed":
        blockers.append("Template profile validation is failing.")
    if publish_status not in {"draft", "pilot", "published", "deprecated", "blocked"}:
        blockers.append("Template publish_status is not recognized.")
    if publish_status in {"draft", "pilot"}:
        manual_checks.append("Template is not marked as published; customer use requires delivery-owner approval.")
    if publish_status == "deprecated":
        manual_checks.append("Template is deprecated; use only for existing customer maintenance.")
    if int(delivery_summary.get("default_object_count") or 0) <= 0:
        blockers.append("Template has no default managed objects.")
    for field, label in (
        ("scenario_ids", "scenario coverage"),
        ("device_sources", "device sources"),
        ("skill_packages", "skill packages"),
        ("acceptance_tests", "acceptance evidence"),
    ):
        if not _string_list(delivery_summary.get(field)):
            blockers.append(f"Template is missing {label}.")
    acceptance_status = str(delivery_summary.get("acceptance_status") or "blocked")
    if acceptance_status == "blocked":
        blockers.append("Template managed-object acceptance is blocked.")
    elif acceptance_status == "manual_check":
        manual_checks.append("Template acceptance references require manual review before signoff.")
    if blockers:
        product_status = "blocked"
    elif manual_checks:
        product_status = "manual_check"
    else:
        product_status = "ready"
    dependencies = {
        "managed_object_count": int(delivery_summary.get("default_object_count") or 0),
        "scenario_count": len(_string_list(delivery_summary.get("scenario_ids"))),
        "device_source_count": len(_string_list(delivery_summary.get("device_sources"))),
        "vision_model_count": len(_string_list(delivery_summary.get("vision_models"))),
        "sensor_protocol_count": len(_string_list(delivery_summary.get("sensor_protocols"))),
        "skill_package_count": len(_string_list(delivery_summary.get("skill_packages"))),
        "acceptance_test_count": len(_string_list(delivery_summary.get("acceptance_tests"))),
    }
    return {
        "package_type": "askme.customer_project_template",
        "package_schema": "askme.customer_project_template.v1",
        "package_id": f"{_slug(template_id)}@{version}",
        "template_id": template_id,
        "version": version,
        "publish_status": publish_status,
        "release_channel": release_channel,
        "owner": str(template.get("owner") or "unassigned"),
        "upgrade_policy": str(template.get("upgrade_policy") or "manual_review"),
        "min_runtime_version": str(template.get("min_runtime_version") or ""),
        "product_status": product_status,
        "customer_status": {
            "ready": "Template is published and ready to seed a customer project after onsite binding.",
            "manual_check": "Template can be used for pilot delivery with product-owner review.",
            "blocked": "Template must be fixed before customer delivery.",
        }[product_status],
        "blocker_count": len(blockers),
        "manual_check_count": len(manual_checks),
        "blockers": blockers,
        "manual_checks": manual_checks,
        "dependencies": dependencies,
        "template_sha256": _sha256_json(profile),
        "source_path": str(path),
        "next_step": {
            "ready": "Create a scoped customer project and run onsite acceptance.",
            "manual_check": "Assign a delivery owner to approve pilot use before creating customer projects.",
            "blocked": "Fix blockers in the template YAML or managed-object bindings.",
        }[product_status],
    }


def _template_delivery_checklist(delivery_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return customer-project rollout steps that every template must pass."""
    object_count = int(delivery_summary.get("default_object_count") or 0)
    acceptance_status = str(delivery_summary.get("acceptance_status") or "blocked")
    binding_count = sum(
        len(_string_list(delivery_summary.get(key)))
        for key in ("vision_models", "sensor_protocols", "skill_packages")
    )
    return [
        {
            "step_id": "validate_template",
            "label": "Validate template package",
            "status": "ready" if delivery_summary.get("template_status") == "passed" else "blocked",
            "evidence": str(delivery_summary.get("template_status") or "failed"),
            "next_step": "Fix template YAML before using it for a customer project.",
        },
        {
            "step_id": "review_template_release",
            "label": "Review template release",
            "status": "ready" if delivery_summary.get("publish_status") == "published" else "manual_check",
            "evidence": (
                f"version {delivery_summary.get('template_version') or '0.0.0'} / "
                f"{delivery_summary.get('publish_status') or 'draft'}"
            ),
            "next_step": "Promote the template to published only after pilot acceptance evidence is attached.",
        },
        {
            "step_id": "replace_customer_scope",
            "label": "Replace customer scope",
            "status": "manual_check",
            "evidence": "Set tenant, delivery namespace, customer, project, and site identifiers.",
            "next_step": "Create a scoped customer project from this template.",
        },
        {
            "step_id": "review_managed_objects",
            "label": "Review managed objects",
            "status": "ready" if object_count else "blocked",
            "evidence": f"{object_count} default managed object(s).",
            "next_step": "Remove irrelevant objects and add customer-specific objects.",
        },
        {
            "step_id": "bind_runtime_capabilities",
            "label": "Bind runtime capabilities",
            "status": "manual_check" if binding_count else "blocked",
            "evidence": (
                f"{len(_string_list(delivery_summary.get('vision_models')))} vision model(s), "
                f"{len(_string_list(delivery_summary.get('sensor_protocols')))} sensor protocol(s), "
                f"{len(_string_list(delivery_summary.get('skill_packages')))} skill package(s)."
            ),
            "next_step": "Bind the project to real devices, model versions, protocols, and enabled skill packages.",
        },
        {
            "step_id": "run_acceptance",
            "label": "Run acceptance evidence",
            "status": acceptance_status if acceptance_status in {"ready", "manual_check", "blocked"} else "manual_check",
            "evidence": (
                f"{delivery_summary.get('ready_object_count', 0)} ready / "
                f"{delivery_summary.get('manual_check_object_count', 0)} manual / "
                f"{delivery_summary.get('blocked_object_count', 0)} blocked object(s)."
            ),
            "next_step": "Run repository and onsite acceptance tests before customer signoff.",
        },
        {
            "step_id": "export_handoff_package",
            "label": "Export handoff package",
            "status": "manual_check",
            "evidence": "Export package after scope, map, devices, responders, and acceptance evidence are reviewed.",
            "next_step": "Use the export package for deployment, review, and customer handoff.",
        },
    ]


def _unique_template_object_values(objects: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        value
        for item in objects
        for value in _string_list(item.get(key))
    })


def _unique_template_binding_values(objects: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        value
        for item in objects
        for value in _string_list(_mapping(item.get("bindings")).get(key))
    })


def _customer_project_implementation_handoff(
    profile: dict[str, Any],
    *,
    template_id: str,
    profile_path: Path | str,
) -> dict[str, Any]:
    """Return the customer-project implementation checklist after profile creation."""
    customer = _mapping(profile.get("customer"))
    site = _mapping(profile.get("site"))
    managed_objects = _mapping(profile.get("managed_objects"))
    binding_labels = {
        "vision_models": "识别模型",
        "sensor_protocols": "传感器协议",
        "skill_packages": "能力包",
        "acceptance_tests": "验收项",
    }
    object_todos: list[dict[str, Any]] = []
    for object_id, item in sorted(managed_objects.items()):
        if not isinstance(item, dict):
            continue
        bindings = _mapping(item.get("bindings"))
        missing = [
            resource_type
            for resource_type in DELIVERY_RESOURCE_TYPES
            if not _string_list(bindings.get(resource_type))
        ]
        object_todos.append(
            {
                "object_id": str(object_id),
                "display_name": str(item.get("display_name") or object_id),
                "category": str(item.get("category") or ""),
                "ready_for_site_acceptance": not missing,
                "missing_binding_types": missing,
                "missing_binding_labels": [binding_labels.get(key, key) for key in missing],
                "customer_next_step": (
                    "补齐识别模型、传感器协议、能力包和验收项后再进入现场验收。"
                    if missing
                    else "对象能力绑定已齐，可继续登记现场证据。"
                ),
            }
        )
    object_needs_binding_count = len([item for item in object_todos if item["missing_binding_types"]])
    status = "needs_object_binding" if object_needs_binding_count else "ready_for_acceptance_evidence"
    return {
        "handoff_schema": "askme.customer_project_implementation_handoff.v1",
        "template_id": str(template_id or ""),
        "project_id": str(customer.get("project_id") or ""),
        "project_name": str(customer.get("project_name") or ""),
        "customer_id": str(customer.get("customer_id") or ""),
        "customer_name": str(customer.get("customer_name") or ""),
        "site_id": str(site.get("site_id") or ""),
        "site_name": str(site.get("name") or ""),
        "profile_path": str(profile_path),
        "status": status,
        "customer_status": (
            "项目已创建，待补齐现场对象能力绑定。"
            if object_needs_binding_count
            else "项目已创建，对象能力绑定已齐，待登记现场验收证据。"
        ),
        "summary": {
            "object_count": len(object_todos),
            "object_ready_count": len(object_todos) - object_needs_binding_count,
            "object_needs_binding_count": object_needs_binding_count,
        },
        "next_steps": [
            {
                "step_id": "review_project_profile",
                "label": "核对项目基础信息",
                "status": "ready",
                "customer_next_step": "确认客户、项目、现场、行业模板和交付边界。",
            },
            {
                "step_id": "complete_object_bindings",
                "label": "补齐对象能力绑定",
                "status": "pending" if object_needs_binding_count else "ready",
                "customer_next_step": "为现场对象绑定识别模型、传感器协议、能力包和验收项。",
            },
            {
                "step_id": "register_onsite_evidence",
                "label": "登记现场验收证据",
                "status": "pending",
                "customer_next_step": "登记设备接入、语音播报、通知送达、客户复核和现场照片证据。",
            },
            {
                "step_id": "export_delivery_package",
                "label": "生成客户交付包",
                "status": "pending",
                "customer_next_step": "先验包和预览差异，再导出可复用客户项目包。",
            },
        ],
        "object_binding_todo": object_todos[:50],
    }


def create_customer_project_from_template(
    *,
    template_root: Path,
    profile_root: Path,
    template_id: str,
    customer: dict[str, Any],
    site: dict[str, Any],
    overwrite: bool = False,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Create or update a customer site profile from a reusable industry template."""
    template_path = _find_template_path(template_root, template_id)
    if template_path is None:
        return {
            "accepted": False,
            "reason": "template_not_found",
            "message": f"Template {template_id} was not found.",
        }
    profile = load_field_site_profile(template_path)
    profile = copy.deepcopy(profile)
    profile["customer"] = _mapping(profile.get("customer")) | _clean_mapping(customer)
    profile["site"] = _mapping(profile.get("site")) | _clean_mapping(site)
    profile.setdefault("customer", {})["delivery_model"] = str(
        profile.get("customer", {}).get("delivery_model") or "solution_project"
    )
    profile = _normalize_customer_project_profile(profile)
    report = validate_field_site_profile(profile)
    if report["status"] != "passed":
        return {
            "accepted": False,
            "reason": "generated_profile_invalid",
            "report": report,
        }
    target = _customer_profile_path(profile_root, profile)
    if target.exists() and not overwrite:
        return {
            "accepted": False,
            "reason": "profile_already_exists",
            "profile_path": str(target),
            "message": "Set overwrite=true to replace the existing customer project profile.",
        }
    if target.exists():
        _snapshot_customer_project_revision(
            profile_root,
            target,
            action="template_overwrite",
            operator_id=operator_id,
            reason=reason or f"Overwrite from template {template_id}.",
        )
    _write_yaml(target, profile)
    implementation_handoff = _customer_project_implementation_handoff(
        profile,
        template_id=template_id,
        profile_path=target,
    )
    return {
        "accepted": True,
        "profile_path": str(target),
        "profile": profile,
        "report": build_site_profile_report(target),
        "implementation_handoff": implementation_handoff,
        "next_step": implementation_handoff["customer_status"],
    }


def upsert_customer_project_profile(
    profile_root: Path,
    profile: dict[str, Any],
    *,
    overwrite: bool = True,
    operator_id: str = "",
    reason: str = "",
    revision_action: str = "profile_upsert",
) -> dict[str, Any]:
    """Create or update a customer project site profile from an explicit profile payload."""
    if not isinstance(profile, dict):
        return {"accepted": False, "reason": "profile_must_be_mapping"}
    profile = _normalize_customer_project_profile(profile)
    report = validate_field_site_profile(profile)
    if report["status"] != "passed":
        return {"accepted": False, "reason": "profile_validation_failed", "report": report}
    target = _customer_profile_target(profile_root, profile)
    if target.exists() and not overwrite:
        return {"accepted": False, "reason": "profile_already_exists", "profile_path": str(target)}
    if target.exists():
        _snapshot_customer_project_revision(
            profile_root,
            target,
            action=revision_action,
            operator_id=operator_id,
            reason=reason,
        )
    _write_yaml(target, profile)
    implementation_handoff = _customer_project_implementation_handoff(
        profile,
        template_id=str(_mapping(profile.get("template")).get("template_id") or ""),
        profile_path=target,
    )
    return {
        "accepted": True,
        "profile_path": str(target),
        "report": build_site_profile_report(target),
        "implementation_handoff": implementation_handoff,
        "next_step": implementation_handoff["customer_status"],
    }


def get_customer_project_profile(profile_root: Path, identifier: str, *, check_env: bool = False) -> dict[str, Any]:
    """Return one customer project profile with validation and object binding detail."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found"}
    profile = load_field_site_profile(path)
    report = build_site_profile_report(path, check_env=check_env)
    managed_objects = managed_object_catalog_from_site_profile(profile)
    implementation_handoff = _customer_project_implementation_handoff(
        profile,
        template_id=str(_mapping(profile.get("template")).get("template_id") or ""),
        profile_path=path,
    )
    env_missing = [
        item
        for item in site_profile_env_references(profile)
        if item.get("required") and not item.get("configured")
    ]
    return {
        "found": True,
        "profile_path": str(path),
        "profile": profile,
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "managed_objects": managed_objects,
        "object_change_log": _object_change_log_payload(profile),
        "delivery_workflow": _customer_project_delivery_workflow(
            profile=profile,
            report=report,
            managed_objects=managed_objects,
            env_missing=env_missing,
        ),
        "implementation_handoff": implementation_handoff,
        "next_step": implementation_handoff.get("customer_status", ""),
        "report": report,
    }


def list_customer_project_onsite_evidence(
    profile_root: Path,
    identifier: str,
    *,
    check_env: bool = True,
    field_evidence_config: dict[str, Any] | None = None,
    include_readiness_auto: bool = True,
) -> dict[str, Any]:
    """Return project-level onsite acceptance evidence receipts."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found"}
    profile = load_field_site_profile(path)
    field_readiness: dict[str, Any] = {}
    if include_readiness_auto:
        site_report = build_site_profile_report(path, check_env=check_env)
        field_readiness = _customer_project_field_readiness(
            path,
            profile,
            site_report=site_report,
            evidence_config=field_evidence_config,
        )
    payload = _customer_project_onsite_evidence_payload(
        profile,
        field_readiness=field_readiness if include_readiness_auto else None,
    )
    return {
        "found": True,
        "profile_path": str(path),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "readiness_auto_included": bool(include_readiness_auto),
        "field_readiness": field_readiness if include_readiness_auto else {},
        **payload,
    }


def register_customer_project_onsite_evidence(
    profile_root: Path,
    identifier: str,
    evidence: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Append one audited onsite acceptance evidence receipt to a customer project."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    if not isinstance(evidence, dict):
        return {"accepted": False, "reason": "evidence_must_be_object"}
    evidence_type = _normalize_onsite_evidence_type(
        evidence.get("evidence_type") or evidence.get("type")
    )
    if evidence_type not in ONSITE_ACCEPTANCE_EVIDENCE_TYPES:
        return {
            "accepted": False,
            "reason": "unsupported_onsite_evidence_type",
            "allowed_types": sorted(ONSITE_ACCEPTANCE_EVIDENCE_TYPES),
        }
    status = _normalize_onsite_evidence_status(evidence.get("status"))
    if status not in ONSITE_ACCEPTANCE_STATUSES:
        return {
            "accepted": False,
            "reason": "unsupported_onsite_evidence_status",
            "allowed_statuses": sorted(ONSITE_ACCEPTANCE_STATUSES),
        }
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    _snapshot_customer_project_revision(
        profile_root,
        path,
        action="onsite_evidence_register",
        operator_id=operator_id or str(evidence.get("operator_id") or "system"),
        reason=reason or str(evidence.get("reason") or "Register onsite acceptance evidence."),
    )
    recorded_at = time.time()
    evidence_path = str(evidence.get("path") or evidence.get("evidence_path") or "").strip()
    inventory = _evidence_file_inventory(evidence_path, evidence_url=_evidence_url(evidence_path)) if evidence_path else {}
    receipt = {
        "receipt_type": "askme.customer_project_onsite_evidence",
        "receipt_version": 1,
        "receipt_id": _slug(
            f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(recorded_at))}-"
            f"{int(recorded_at * 1000)}-{evidence_type}-{operator_id or evidence.get('operator_id') or 'system'}"
        ),
        "recorded_at": recorded_at,
        "operator_id": str(operator_id or evidence.get("operator_id") or "system"),
        "reason": str(reason or evidence.get("reason") or ""),
        "evidence_type": evidence_type,
        "status": status,
        "source": str(evidence.get("source") or ""),
        "label": str(evidence.get("label") or evidence_type.replace("_", " ").title()),
        "summary": str(evidence.get("summary") or evidence.get("note") or ""),
        "path": evidence_path,
        "evidence_url": str(inventory.get("evidence_url") or _evidence_url(evidence_path)),
        "exists": bool(inventory.get("exists")) if evidence_path else False,
        "size_bytes": int(inventory.get("size_bytes") or 0),
        "sha256": str(inventory.get("sha256") or evidence.get("sha256") or ""),
        "project_scope": _delivery_scope_payload(profile),
        "managed_object_id": str(evidence.get("managed_object_id") or ""),
        "event_id": str(evidence.get("event_id") or ""),
        "runtime_run_id": str(evidence.get("runtime_run_id") or ""),
        "external_reference": str(evidence.get("external_reference") or ""),
        "evidence_tier": str(evidence.get("evidence_tier") or "acceptance_candidate"),
        "production_eligible": evidence.get("production_eligible") is True,
    }
    if inventory.get("error"):
        receipt["error"] = str(inventory.get("error") or "")
    receipts = _customer_project_raw_onsite_evidence(profile)
    receipts.append(receipt)
    profile["onsite_acceptance_evidence"] = receipts
    _write_yaml(path, profile)
    payload = _customer_project_onsite_evidence_payload(profile)
    return {
        "accepted": True,
        "profile_path": str(path),
        "receipt": receipt,
        **payload,
    }


def customer_project_acceptance_closure(
    profile_root: Path,
    identifier: str,
    *,
    check_env: bool = True,
    field_evidence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a customer-facing project acceptance closure summary."""
    report = customer_project_acceptance_report(
        profile_root,
        identifier,
        check_env=check_env,
        field_evidence_config=field_evidence_config,
    )
    if not report.get("found"):
        return {"found": False, "reason": report.get("reason") or "profile_not_found"}
    path = find_site_profile_path(profile_root, identifier)
    profile = load_field_site_profile(path) if path else {}
    reviews = _customer_project_acceptance_reviews(profile)
    signoffs = _customer_project_customer_signoffs(profile)
    latest_review = reviews[0] if reviews else {}
    latest_signoff = signoffs[0] if signoffs else {}
    onsite = _mapping(report.get("onsite_acceptance_evidence"))
    onsite_summary = _mapping(onsite.get("summary"))
    site_checklist = _mapping(report.get("site_acceptance_checklist"))
    review_gate = _customer_project_acceptance_review_gate(latest_review)
    dossier_verification = _customer_project_acceptance_dossier_verification(report)
    proposal_verification = _customer_project_latest_proposal_verification(profile)
    audit_export = _customer_project_latest_audit_export(profile)
    pre_signoff_gates = [
        {
            "gate_id": "acceptance_report",
            "label": "Acceptance report",
            "status": _readiness_status(report.get("overall_status")),
            "evidence": f"report_status={report.get('overall_status') or 'unknown'}",
            "next_step": str(report.get("customer_status") or ""),
        },
        {
            "gate_id": "onsite_evidence",
            "label": "Onsite evidence",
            "status": str(onsite_summary.get("overall_status") or "manual_check"),
            "evidence": (
                f"{onsite_summary.get('passed_required_count') or 0}/"
                f"{onsite_summary.get('required_count') or len(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES)} "
                "required receipts passed"
            ),
            "next_step": str(onsite_summary.get("next_step") or ""),
        },
        {
            "gate_id": "site_acceptance_checklist",
            "label": "Site acceptance checklist",
            "status": str(site_checklist.get("overall_status") or "manual_check"),
            "evidence": (
                f"ready={site_checklist.get('ready_count') or 0}; "
                f"manual={site_checklist.get('manual_check_count') or 0}; "
                f"blocked={site_checklist.get('blocked_count') or 0}"
            ),
            "next_step": str(site_checklist.get("customer_message") or "Review site acceptance checklist."),
        },
        review_gate,
        {
            "gate_id": "dossier_verification",
            "label": "Dossier verification",
            "status": "ready" if dossier_verification.get("valid") else "blocked",
            "evidence": str(dossier_verification.get("reason") or "unknown"),
            "next_step": "Regenerate the acceptance dossier." if not dossier_verification.get("valid") else "Dossier manifest is self-consistent.",
        },
        {
            "gate_id": "proposal_verification",
            "label": "Proposal verification",
            "status": str(proposal_verification.get("status") or "manual_check"),
            "evidence": str(proposal_verification.get("evidence") or "No matching proposal bundle found."),
            "next_step": str(proposal_verification.get("next_step") or "Export and verify the customer proposal bundle."),
        },
        {
            "gate_id": "audit_export",
            "label": "Audit export",
            "status": str(audit_export.get("status") or "manual_check"),
            "evidence": str(audit_export.get("evidence") or "No matching audit export found."),
            "next_step": str(audit_export.get("next_step") or "Create a scoped audit export for customer handoff."),
        },
    ]
    pre_signoff_statuses = {str(gate.get("status") or "") for gate in pre_signoff_gates}
    base_ready_for_signoff = not (
        "blocked" in pre_signoff_statuses or "manual_check" in pre_signoff_statuses
    )
    signoff_gate = _customer_project_customer_signoff_gate(
        latest_signoff,
        base_ready_for_signoff=base_ready_for_signoff,
    )
    closure_gates = [*pre_signoff_gates, signoff_gate]
    statuses = {str(gate.get("status") or "") for gate in closure_gates}
    if "blocked" in statuses:
        overall = "blocked"
        customer_claim = "不能提交客户验收。需要先处理阻断项。"
    elif base_ready_for_signoff and signoff_gate.get("status") == "ready":
        overall = "accepted_by_customer"
        customer_claim = "客户签收已归档，可作为本项目试点验收结论。"
    elif base_ready_for_signoff:
        overall = "ready_for_customer_signoff"
        customer_claim = "证据和内部复核结论齐备，等待客户签收。"
    elif "manual_check" in statuses:
        overall = "manual_check"
        customer_claim = "可以进入交付复核，但还不能作为客户最终验收结论。"
    else:
        overall = "manual_check"
        customer_claim = "可以进入交付复核，但还不能作为客户最终验收结论。"
    return {
        "found": True,
        "profile_path": str(path or ""),
        "project_scope": _delivery_scope_payload(profile),
        "customer": report.get("customer"),
        "site": report.get("site"),
        "overall_status": overall,
        "customer_claim": customer_claim,
        "next_step": _customer_project_acceptance_closure_next_step(overall, closure_gates),
        "gates": closure_gates,
        "acceptance_report": {
            "overall_status": report.get("overall_status"),
            "customer_status": report.get("customer_status"),
            "release_claim": report.get("release_claim"),
        },
        "onsite_acceptance_evidence": onsite,
        "site_acceptance_checklist": site_checklist,
        "manual_review": {
            "latest": latest_review,
            "reviews": reviews,
            "review_count": len(reviews),
        },
        "customer_signoff": {
            "latest": latest_signoff,
            "signoffs": signoffs,
            "signoff_count": len(signoffs),
            "base_ready_for_signoff": base_ready_for_signoff,
        },
        "artifact_verification": {
            "acceptance_dossier": dossier_verification,
            "proposal_bundle": proposal_verification,
            "audit_export": audit_export,
        },
        "evidence_timeline": _customer_project_acceptance_evidence_timeline(onsite, reviews, signoffs),
        "blocked_uses": [
            "无人值守生产上线",
            "无人工复核的最终验收承诺",
        ]
        if overall != "accepted_by_customer"
        else [],
    }


def register_customer_project_acceptance_review(
    profile_root: Path,
    identifier: str,
    review: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Append a delivery-owner acceptance review decision to the project profile."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    if not isinstance(review, dict):
        return {"accepted": False, "reason": "review_must_be_object"}
    decision = _normalize_acceptance_review_decision(review.get("decision"))
    if decision not in ACCEPTANCE_REVIEW_DECISIONS:
        return {
            "accepted": False,
            "reason": "unsupported_acceptance_review_decision",
            "allowed_decisions": sorted(ACCEPTANCE_REVIEW_DECISIONS),
        }
    if decision == "accepted" and review.get("risk_acknowledgement") is not True:
        return {
            "accepted": False,
            "reason": "risk_acknowledgement_required",
        }
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    _snapshot_customer_project_revision(
        profile_root,
        path,
        action="acceptance_review_register",
        operator_id=operator_id or str(review.get("operator_id") or "system"),
        reason=reason or str(review.get("reason") or "Register acceptance review."),
    )
    reviewed_at = time.time()
    review_record = {
        "review_type": "askme.customer_project_acceptance_review",
        "review_version": 1,
        "review_id": _slug(
            f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(reviewed_at))}-"
            f"{int(reviewed_at * 1000)}-{decision}-{operator_id or review.get('operator_id') or 'system'}"
        ),
        "reviewed_at": reviewed_at,
        "operator_id": str(operator_id or review.get("operator_id") or "system"),
        "decision": decision,
        "reason": str(reason or review.get("reason") or ""),
        "risk_acknowledgement": review.get("risk_acknowledgement") is True,
        "evidence_refs": [
            str(item)
            for item in (review.get("evidence_refs") if isinstance(review.get("evidence_refs"), list) else [])
            if str(item).strip()
        ],
        "project_scope": _delivery_scope_payload(profile),
    }
    reviews = _customer_project_raw_acceptance_reviews(profile)
    reviews.append(review_record)
    profile["acceptance_reviews"] = reviews
    _write_yaml(path, profile)
    closure = customer_project_acceptance_closure(profile_root, identifier, check_env=False)
    return {
        "accepted": True,
        "profile_path": str(path),
        "review": review_record,
        "closure": closure,
    }


def list_customer_project_customer_signoffs(profile_root: Path, identifier: str) -> dict[str, Any]:
    """Return customer signoff records for one project."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found"}
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    signoffs = _customer_project_customer_signoffs(profile)
    return {
        "found": True,
        "profile_path": str(path),
        "project_scope": _delivery_scope_payload(profile),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "latest": signoffs[0] if signoffs else {},
        "signoffs": signoffs,
        "signoff_count": len(signoffs),
    }


def register_customer_project_customer_signoff(
    profile_root: Path,
    identifier: str,
    signoff: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Append a customer signoff decision after delivery acceptance review."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    if not isinstance(signoff, dict):
        return {"accepted": False, "reason": "signoff_must_be_object"}
    decision = _normalize_customer_signoff_decision(signoff.get("decision"))
    if decision not in CUSTOMER_SIGNOFF_DECISIONS:
        return {
            "accepted": False,
            "reason": "unsupported_customer_signoff_decision",
            "allowed_decisions": sorted(CUSTOMER_SIGNOFF_DECISIONS),
        }
    signatory_name = str(signoff.get("signatory_name") or "").strip()
    if not signatory_name:
        return {
            "accepted": False,
            "reason": "signatory_name_required",
        }
    risk_acknowledgement = signoff.get("risk_acknowledgement") is True
    if decision == "accepted" and not risk_acknowledgement:
        return {
            "accepted": False,
            "reason": "customer_risk_acknowledgement_required",
        }
    closure_before = customer_project_acceptance_closure(profile_root, identifier, check_env=False)
    if decision == "accepted" and closure_before.get("overall_status") not in {
        "ready_for_customer_signoff",
        "accepted_by_customer",
    }:
        return {
            "accepted": False,
            "reason": "project_not_ready_for_customer_signoff",
            "closure": closure_before,
        }
    credential_ref = str(
        signoff.get("credential_ref")
        or signoff.get("signature_ref")
        or signoff.get("signed_artifact_ref")
        or ""
    ).strip()
    credential_sha256 = _normalize_sha256_hex(
        signoff.get("credential_sha256")
        or signoff.get("signature_sha256")
        or signoff.get("signed_artifact_sha256")
    )
    if decision == "accepted" and (not credential_ref or not credential_sha256):
        return {
            "accepted": False,
            "reason": "customer_signoff_credential_required",
            "message": "Accepted customer signoff requires credential_ref and credential_sha256.",
        }
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    _snapshot_customer_project_revision(
        profile_root,
        path,
        action="customer_signoff_register",
        operator_id=operator_id or str(signoff.get("operator_id") or "system"),
        reason=reason or str(signoff.get("reason") or "Register customer signoff."),
    )
    signed_at = time.time()
    evidence_refs = [
        str(item)
        for item in (signoff.get("evidence_refs") if isinstance(signoff.get("evidence_refs"), list) else [])
        if str(item).strip()
    ]
    record = {
        "signoff_type": "askme.customer_project_customer_signoff",
        "signoff_version": 2,
        "signoff_id": _slug(
            f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(signed_at))}-"
            f"{int(signed_at * 1000)}-{decision}-{signatory_name}"
        ),
        "signed_at": signed_at,
        "operator_id": str(operator_id or signoff.get("operator_id") or "system"),
        "decision": decision,
        "signatory_name": signatory_name,
        "signatory_role": str(signoff.get("signatory_role") or ""),
        "organization": str(signoff.get("organization") or ""),
        "reason": str(reason or signoff.get("reason") or ""),
        "risk_acknowledgement": risk_acknowledgement,
        "credential_ref": credential_ref,
        "credential_sha256": credential_sha256,
        "evidence_refs": evidence_refs,
        "gate_snapshot": _customer_project_customer_signoff_gate_snapshot(closure_before),
        "handoff_materials": _customer_project_customer_signoff_handoff_materials(
            closure_before,
            evidence_refs=evidence_refs,
        ),
        "project_scope": _delivery_scope_payload(profile),
    }
    record["signoff_payload_sha256"] = _customer_project_customer_signoff_payload_sha256(record)
    signoffs = _customer_project_raw_customer_signoffs(profile)
    signoffs.append(record)
    profile["customer_signoffs"] = signoffs
    _write_yaml(path, profile)
    closure = customer_project_acceptance_closure(profile_root, identifier, check_env=False)
    return {
        "accepted": True,
        "profile_path": str(path),
        "signoff": record,
        "closure": closure,
    }


def customer_project_acceptance_report(
    profile_root: Path,
    identifier: str,
    *,
    check_env: bool = True,
    field_evidence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a customer-readable acceptance report for one project."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found"}
    profile = load_field_site_profile(path)
    report = build_site_profile_report(path, check_env=check_env)
    catalog = managed_object_catalog_from_site_profile(profile)
    execution_bindings = build_customer_project_execution_bindings(profile_root, identifier)
    execution_summary = _mapping(execution_bindings.get("summary"))
    acceptance = _customer_project_package_acceptance_summary(catalog)
    field_readiness = _customer_project_field_readiness(
        path,
        profile,
        site_report=report,
        evidence_config=field_evidence_config,
    )
    field_readiness_gates = _customer_project_field_readiness_gates(field_readiness)
    onsite_evidence = _customer_project_onsite_evidence_payload(
        profile,
        field_readiness=field_readiness,
    )["onsite_acceptance_evidence"]
    onsite_summary = _mapping(onsite_evidence.get("summary"))
    env_refs = site_profile_env_references(profile)
    missing_env = [
        item for item in env_refs if item.get("required") and not item.get("configured")
    ]
    delivery_workflow = _customer_project_delivery_workflow(
        profile=profile,
        report=report,
        managed_objects=catalog,
        env_missing=missing_env,
    )
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    gates = [
        {
            "gate_id": "site_profile",
            "label": "Site profile",
            "status": "ready" if report.get("status") == "passed" else "blocked",
            "evidence": str(path),
            "next_step": "Fix site profile validation errors." if errors else "Site profile schema is valid.",
        },
        {
            "gate_id": "managed_object_acceptance",
            "label": "Managed object acceptance",
            "status": acceptance["overall_status"],
            "evidence": f"{acceptance['ready_object_count']}/{acceptance['object_count']} objects ready",
            "next_step": acceptance["customer_status"],
        },
        {
            "gate_id": "managed_object_execution_bindings",
            "label": "Managed object execution bindings",
            "status": str(execution_summary.get("overall_status") or "blocked"),
            "evidence": (
                f"{execution_summary.get('ready_object_count') or 0}/"
                f"{execution_summary.get('object_count') or 0} objects have executable ingest plans"
            ),
            "next_step": str(
                execution_bindings.get("next_step")
                or "Generate execution bindings before customer signoff."
            ),
        },
        {
            "gate_id": "deployment_credentials",
            "label": "Deployment credentials",
            "status": "manual_check" if missing_env else "ready",
            "evidence": f"{len(missing_env)} missing environment values",
            "next_step": (
                "Fill DingTalk responder and signed device secrets before onsite acceptance."
                if missing_env
                else "Deployment environment references are configured."
            ),
        },
        {
            "gate_id": "onsite_acceptance_boundary",
            "label": "Onsite acceptance boundary",
            "status": str(onsite_summary.get("overall_status") or "manual_check"),
            "evidence": (
                f"{onsite_summary.get('passed_required_count') or 0}/"
                f"{onsite_summary.get('required_count') or len(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES)} "
                f"required onsite evidence receipts passed; receipts={onsite_summary.get('receipt_count') or 0}"
            ),
            "next_step": str(
                onsite_summary.get("next_step")
                or "Run live onsite smoke tests before any production launch claim."
            ),
        },
        *field_readiness_gates,
    ]
    site_acceptance_checklist = _customer_project_site_acceptance_checklist(
        profile=profile,
        report=report,
        acceptance=acceptance,
        field_readiness=field_readiness,
        onsite_evidence=onsite_evidence,
        missing_env=missing_env,
    )
    gate_statuses = {str(gate.get("status") or "") for gate in gates}
    if errors or acceptance["blocked_object_count"] or "blocked" in gate_statuses:
        overall = "blocked"
    elif (
        acceptance["manual_check_object_count"]
        or missing_env
        or warnings
        or "manual_check" in gate_statuses
    ):
        overall = "manual_check"
    else:
        overall = "ready_for_onsite_acceptance"
    launch_readiness = _customer_project_launch_readiness(
        project_status=overall,
        gates=gates,
        site_acceptance_checklist=site_acceptance_checklist,
        field_readiness=field_readiness,
        onsite_acceptance_evidence=onsite_evidence,
        missing_env=missing_env,
    )
    return {
        "found": True,
        "profile_path": str(path),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "overall_status": overall,
        "customer_status": {
            "ready_for_onsite_acceptance": "Project evidence is ready for onsite acceptance.",
            "manual_check": "Project needs delivery review before customer signoff.",
            "blocked": "Project is blocked and must not be presented as acceptance-ready.",
        }.get(overall, "Project acceptance status is unknown."),
        "release_claim": (
            "This report supports demo/trial acceptance review only. Production launch requires "
            "separate onsite evidence for devices, notifications, voice, and robot runtime."
        ),
        "launch_readiness": launch_readiness,
        "gates": gates,
        "site_acceptance_checklist": site_acceptance_checklist,
        "delivery_workflow": delivery_workflow,
        "acceptance_summary": acceptance,
        "execution_bindings": {
            "summary": execution_summary,
            "customer_claim": str(execution_bindings.get("customer_claim") or ""),
            "next_step": str(execution_bindings.get("next_step") or ""),
            "object_contracts": _execution_binding_report_contracts(execution_bindings),
        },
        "field_readiness": field_readiness,
        "onsite_acceptance_evidence": onsite_evidence,
        "acceptance_reviews": _customer_project_acceptance_reviews(profile),
        "customer_signoffs": _customer_project_customer_signoffs(profile),
        "env_missing": [
            {
                "env_name": str(item.get("env_name") or ""),
                "category": str(item.get("category") or ""),
                "owner": str(item.get("owner") or ""),
                "purpose": str(item.get("purpose") or ""),
            }
            for item in missing_env
        ],
        "warnings": warnings,
        "errors": errors,
    }


def _customer_project_launch_readiness(
    *,
    project_status: str,
    gates: list[dict[str, Any]],
    site_acceptance_checklist: dict[str, Any],
    field_readiness: dict[str, Any],
    onsite_acceptance_evidence: dict[str, Any],
    missing_env: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a customer-readable launch gate for one project export."""
    gate_by_id = {str(gate.get("gate_id") or ""): gate for gate in gates}
    onsite_summary = _mapping(onsite_acceptance_evidence.get("summary"))
    field_gates = _mapping(field_readiness.get("gates"))
    launch_gates = [
        _launch_readiness_gate(
            gate_id="project_acceptance_report",
            label="客户项目验收报告",
            status=_customer_project_launch_gate_status(project_status),
            evidence=f"overall_status={project_status}",
            next_step=_customer_project_launch_gate_next_step(project_status),
        ),
        _launch_readiness_gate(
            gate_id="managed_object_execution_bindings",
            label="对象执行绑定",
            status=_customer_project_launch_gate_status(
                _mapping(gate_by_id.get("managed_object_execution_bindings")).get("status")
            ),
            evidence=str(_mapping(gate_by_id.get("managed_object_execution_bindings")).get("evidence") or ""),
            next_step=str(_mapping(gate_by_id.get("managed_object_execution_bindings")).get("next_step") or ""),
        ),
        _launch_readiness_gate(
            gate_id="deployment_credentials",
            label="部署凭证和通知配置",
            status="manual_check" if missing_env else "ready",
            evidence=f"{len(missing_env)} missing required deployment value(s)",
            next_step=(
                "补齐钉钉通知、设备签名、运行回调等生产环境凭证。"
                if missing_env
                else "部署凭证已满足当前客户项目验收报告。"
            ),
        ),
        _launch_readiness_gate(
            gate_id="onsite_required_evidence",
            label="现场必需证据",
            status=_customer_project_launch_gate_status(onsite_summary.get("overall_status")),
            evidence=(
                f"{onsite_summary.get('passed_required_count') or 0}/"
                f"{onsite_summary.get('required_count') or len(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES)} "
                "required onsite evidence types passed"
            ),
            next_step=str(
                onsite_summary.get("next_step")
                or "完成设备接入、语音播报、钉钉通知、运行回调四类现场证据。"
            ),
        ),
        _launch_readiness_gate(
            gate_id="field_real_link",
            label="真实现场链路",
            status=_readiness_status(field_readiness.get("status")),
            evidence=(
                f"status={field_readiness.get('status') or 'unknown'}; "
                f"real_hardware={field_gates.get('uses_real_hardware') is True}; "
                f"external_services={field_gates.get('uses_external_services') is True}"
            ),
            next_step=str(
                (
                    field_readiness.get("next_actions")
                    if isinstance(field_readiness.get("next_actions"), list)
                    else []
                )[0]
                if isinstance(field_readiness.get("next_actions"), list)
                and field_readiness.get("next_actions")
                else "用真实传感器、MiniMax 语音、钉钉通知和 runtime 回调完成现场联调。"
            ),
        ),
        _launch_readiness_gate(
            gate_id="site_acceptance_checklist",
            label="客户现场验收清单",
            status=_customer_project_launch_gate_status(site_acceptance_checklist.get("overall_status")),
            evidence=(
                f"ready={site_acceptance_checklist.get('ready_count') or 0}, "
                f"manual={site_acceptance_checklist.get('manual_check_count') or 0}, "
                f"blocked={site_acceptance_checklist.get('blocked_count') or 0}"
            ),
            next_step=str(site_acceptance_checklist.get("next_step") or "完成客户现场验收清单。"),
        ),
    ]
    overall_status = _customer_project_launch_rollup_status(launch_gates)
    launch_stage = {
        "ready": "production_acceptance_ready",
        "manual_check": "pilot_or_site_trial",
        "blocked": "demo_or_integration_only",
    }[overall_status]
    blockers = [
        str(gate.get("next_step") or gate.get("evidence") or "")
        for gate in launch_gates
        if gate.get("status") == "blocked"
    ]
    manual_checks = [
        str(gate.get("next_step") or gate.get("evidence") or "")
        for gate in launch_gates
        if gate.get("status") == "manual_check"
    ]
    return {
        "readiness_type": "askme.customer_project_launch_readiness.v1",
        "overall_status": overall_status,
        "launch_stage": launch_stage,
        "production_ready": overall_status == "ready",
        "customer_status": {
            "ready": "可进入客户现场上线验收；上线前仍需客户签收、值守安排和回滚预案确认。",
            "manual_check": "可进入客户试点、现场联调或方案演示；不能承诺无人值守生产上线。",
            "blocked": "暂不能进入客户上线验收；只能说明当前处于演示、研发或现场联调阶段。",
        }[overall_status],
        "release_claim": {
            "ready": "可以声明具备受控现场上线验收条件，但不替代客户最终签收。",
            "manual_check": "只能声明试点或现场联调能力，不能声明生产上线或无人值守运行。",
            "blocked": "不能声明客户可上线、可验收通过或可无人值守运行。",
        }[overall_status],
        "next_step": (
            blockers[0]
            if blockers
            else manual_checks[0]
            if manual_checks
            else "安排客户现场验收、签署交付单，并确认人工接管和回滚预案。"
        ),
        "gates": launch_gates,
        "summary": {
            "gate_count": len(launch_gates),
            "ready_count": len([gate for gate in launch_gates if gate.get("status") == "ready"]),
            "manual_check_count": len(
                [gate for gate in launch_gates if gate.get("status") == "manual_check"]
            ),
            "blocked_count": len([gate for gate in launch_gates if gate.get("status") == "blocked"]),
            "missing_env_count": len(missing_env),
            "onsite_receipt_count": int(onsite_summary.get("receipt_count") or 0),
        },
        "blockers": blockers,
        "manual_checks": manual_checks,
        "source_snapshots": {
            "project_status": project_status,
            "field_readiness_status": field_readiness.get("status"),
            "onsite_evidence_status": onsite_summary.get("overall_status"),
            "site_acceptance_checklist_status": site_acceptance_checklist.get("overall_status"),
        },
    }


def _launch_readiness_gate(
    *,
    gate_id: str,
    label: str,
    status: str,
    evidence: str,
    next_step: str,
) -> dict[str, str]:
    return {
        "gate_id": gate_id,
        "label": label,
        "status": _customer_project_launch_gate_status(status),
        "evidence": evidence,
        "next_step": next_step,
    }


def _customer_project_launch_gate_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"ready", "ok", "healthy", "passed", "production_ready", "ready_for_onsite_acceptance"}:
        return "ready"
    if text in {"manual_check", "ready_for_lab", "ready_for_acceptance", "ready_for_customer_signoff"}:
        return "manual_check"
    if text in {"blocked", "failed", "missing", "invalid", "error"}:
        return "blocked"
    return "manual_check"


def _customer_project_launch_gate_next_step(project_status: str) -> str:
    status = _customer_project_launch_gate_status(project_status)
    if status == "ready":
        return "准备客户现场上线验收、交付签收和值守接管预案。"
    if status == "blocked":
        return "先修复阻塞 gate，再重新生成客户项目验收报告。"
    return "补齐人工复核项和现场证据后，再申请客户上线验收。"


def _customer_project_launch_rollup_status(gates: list[dict[str, Any]]) -> str:
    statuses = {str(gate.get("status") or "blocked") for gate in gates}
    if "blocked" in statuses:
        return "blocked"
    if "manual_check" in statuses:
        return "manual_check"
    return "ready"


def _execution_binding_report_contracts(execution_bindings: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact object-level contracts for customer acceptance reports."""
    plans = execution_bindings.get("plans")
    if not isinstance(plans, list):
        return []
    contracts: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        adapters = []
        for adapter in plan.get("input_adapters") if isinstance(plan.get("input_adapters"), list) else []:
            if not isinstance(adapter, dict):
                continue
            contract = _mapping(adapter.get("adapter_contract"))
            adapters.append({
                "protocol_id": str(adapter.get("protocol_id") or ""),
                "adapter": str(adapter.get("adapter") or ""),
                "status": str(adapter.get("status") or ""),
                "bridge": str(contract.get("bridge") or ""),
                "ingest_endpoint": str(contract.get("ingest_endpoint") or ""),
                "device_signature_required": bool(contract.get("device_signature_required")),
                "device_secret_envs": _string_list(contract.get("device_secret_envs")),
                "dry_run_command": str(contract.get("dry_run_command") or ""),
                "live_command": str(contract.get("live_command") or ""),
                "sample_fixture": str(contract.get("sample_fixture") or ""),
            })
        skill_routes = []
        for route in plan.get("skill_routes") if isinstance(plan.get("skill_routes"), list) else []:
            if not isinstance(route, dict):
                continue
            skill_routes.append({
                "capability": str(route.get("capability") or route.get("resource_id") or ""),
                "tool": str(route.get("tool") or ""),
                "output_contract": str(route.get("output_contract") or ""),
                "approval_policy": str(route.get("approval_policy") or ""),
                "hardware_boundary": str(route.get("hardware_boundary") or route.get("safety_boundary") or ""),
            })
        contracts.append({
            "object_id": str(plan.get("object_id") or ""),
            "display_name": str(plan.get("display_name") or plan.get("object_id") or ""),
            "overall_status": str(plan.get("overall_status") or ""),
            "input_adapters": adapters,
            "bridge_contract": _mapping(plan.get("bridge_contract")),
            "skill_routes": skill_routes,
        })
    return contracts


def _customer_project_field_readiness(
    profile_path: Path,
    profile: dict[str, Any],
    *,
    site_report: dict[str, Any],
    evidence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact onsite evidence snapshot for a customer project."""
    try:
        from askme.pipeline.field_operations import FieldOperationsService
    except Exception as exc:  # pragma: no cover - defensive import guard
        return {
            "available": False,
            "status": "manual_check",
            "reason": "field_operations_service_unavailable",
            "error": str(exc),
            "blockers": [],
            "warnings": [str(exc)],
            "next_actions": ["Check field operations service imports before customer acceptance."],
            "gates": {},
            "evidence_reports": [],
        }

    config = field_operations_config_from_site_profile(profile)
    config.update(_FIELD_READINESS_EVIDENCE_DEFAULTS)
    if isinstance(evidence_config, dict):
        config.update({str(key): value for key, value in evidence_config.items()})
    config["site_profile_path"] = str(profile_path)
    config["site_profile"] = {
        "status": site_report.get("status"),
        "summary": site_report.get("summary", {}),
        "readiness": site_report.get("readiness", {}),
        "warnings": site_report.get("warnings", []),
    }
    try:
        payload = FieldOperationsService(config=config).readiness_payload()
    except Exception as exc:  # pragma: no cover - product-safe fallback
        return {
            "available": False,
            "status": "manual_check",
            "reason": "field_readiness_failed",
            "error": str(exc),
            "blockers": [],
            "warnings": [str(exc)],
            "next_actions": ["Run field-readiness directly and fix the reported error."],
            "gates": {},
            "evidence_reports": [],
        }
    return _compact_field_readiness(payload)


def _compact_field_readiness(payload: dict[str, Any]) -> dict[str, Any]:
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    next_actions = payload.get("next_actions") if isinstance(payload.get("next_actions"), list) else []
    delivery_brief = payload.get("delivery_brief") if isinstance(payload.get("delivery_brief"), dict) else {}
    return {
        "available": True,
        "status": str(payload.get("status") or "unknown"),
        "deployment_mode": str(payload.get("deployment_mode") or ""),
        "blockers": [str(item) for item in blockers[:20]],
        "warnings": [str(item) for item in warnings[:30]],
        "next_actions": [str(item) for item in next_actions[:10]],
        "delivery_brief": {
            "stage_code": str(delivery_brief.get("stage_code") or ""),
            "customer_status": str(delivery_brief.get("customer_status") or ""),
            "release_scope": str(delivery_brief.get("release_scope") or ""),
            "release_claim": str(delivery_brief.get("release_claim") or ""),
            "top_issue": str(delivery_brief.get("top_issue") or ""),
        },
        "gates": {
            key: bool(gates.get(key))
            for key in (
                "scenario_eval_passed",
                "http_smoke_passed",
                "archive_has_events",
                "voice_smoke_passed",
                "voice_smoke_uses_live_tts",
                "notification_smoke_passed",
                "notification_smoke_uses_external_services",
                "runtime_roundtrip_smoke_passed",
                "runtime_roundtrip_final_status_verified",
                "action_audit_integrity_verified",
                "action_audit_signed",
                "unified_audit_review_clear",
                "all_registered_devices_signature_ready",
                "trusted_device_events_observed",
                "uses_real_hardware",
                "uses_external_services",
            )
        },
        "reports": {
            "scenario": _compact_evidence_report(payload.get("scenario_report")),
            "ingest_smoke": _compact_evidence_report(payload.get("smoke_report")),
            "voice_smoke": _compact_evidence_report(payload.get("voice_smoke_report")),
            "notification_smoke": _compact_evidence_report(payload.get("notification_smoke_report")),
            "runtime_roundtrip": _compact_evidence_report(payload.get("runtime_roundtrip_report")),
        },
        "archive": {
            "path": str(_mapping(payload.get("archive")).get("path") or ""),
            "event_count": _mapping(payload.get("archive")).get("event_count") or 0,
            "scenario_ids": _mapping(payload.get("archive")).get("scenario_ids") or [],
            "sources": _mapping(payload.get("archive")).get("sources") or [],
            "trusted_device_event_count": _mapping(payload.get("archive")).get("trusted_device_event_count") or 0,
        },
        "device_trust": {
            "registered_device_count": _mapping(payload.get("device_trust")).get("registered_device_count") or 0,
            "signed_device_count": _mapping(payload.get("device_trust")).get("signed_device_count") or 0,
            "unsigned_device_count": _mapping(payload.get("device_trust")).get("unsigned_device_count") or 0,
            "all_registered_devices_signature_ready": _mapping(payload.get("device_trust")).get(
                "all_registered_devices_signature_ready"
            )
            is True,
        },
        "evidence_reports": [
            report
            for report in (
                _compact_evidence_report(payload.get("scenario_report")),
                _compact_evidence_report(payload.get("smoke_report")),
                _compact_evidence_report(payload.get("voice_smoke_report")),
                _compact_evidence_report(payload.get("notification_smoke_report")),
                _compact_evidence_report(payload.get("runtime_roundtrip_report")),
            )
            if report.get("path")
        ],
    }


def _compact_evidence_report(value: Any) -> dict[str, Any]:
    report = _mapping(value)
    path = str(report.get("path") or "")
    return {
        "status": str(report.get("status") or "unknown"),
        "passed": report.get("passed") is True,
        "path": path,
        "evidence_url": _evidence_url(path) if path else "",
        "event_count": report.get("event_count"),
        "scenario_count": report.get("scenario_count"),
        "local_server": report.get("local_server"),
        "live_tts": report.get("live_tts"),
        "external_services": report.get("external_services"),
        "hardware_dispatch": report.get("hardware_dispatch"),
        "mode": report.get("mode"),
        "trusted_callbacks": report.get("trusted_callbacks"),
        "final_status_verified": report.get("final_status_verified"),
        "voice_delivery_status": report.get("voice_delivery_status"),
        "voice_profile": report.get("voice_profile"),
        "collector_request_count": report.get("collector_request_count"),
        "final_status": report.get("final_status"),
    }


def _evidence_url(path: str) -> str:
    if not path:
        return ""
    try:
        raw = Path(path)
        resolved = raw.resolve()
        rel = resolved.relative_to(PROJECT_ROOT.resolve())
        return f"/api/field/evidence?path={quote(rel.as_posix())}"
    except (OSError, ValueError):
        return ""


def _customer_project_raw_onsite_evidence(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("onsite_acceptance_evidence")
    if isinstance(raw, dict):
        raw = raw.get("receipts") or raw.get("items") or []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _customer_project_onsite_evidence_payload(
    profile: dict[str, Any],
    *,
    field_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipts = _customer_project_onsite_evidence_receipts(profile)
    if isinstance(field_readiness, dict):
        receipts = [
            *receipts,
            *_customer_project_auto_onsite_evidence_receipts(field_readiness, profile),
        ]
    return _customer_project_onsite_evidence_payload_from_receipts(receipts)


def _customer_project_onsite_evidence_payload_from_receipts(
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    receipts = [
        dict(item)
        for item in receipts
        if str(item.get("evidence_type") or "") in ONSITE_ACCEPTANCE_EVIDENCE_TYPES
    ]
    receipts.sort(key=lambda receipt: float(receipt.get("recorded_at") or 0), reverse=True)
    summary = _customer_project_onsite_evidence_summary(receipts)
    return {
        "onsite_acceptance_evidence": {
            "summary": summary,
            "required_evidence_types": list(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES),
            "allowed_evidence_types": sorted(ONSITE_ACCEPTANCE_EVIDENCE_TYPES),
            "receipts": receipts,
        }
    }


def _customer_project_auto_onsite_evidence_receipts(
    field_readiness: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project real-link readiness reports into read-only onsite evidence receipts."""
    gates = _mapping(field_readiness.get("gates"))
    reports = _mapping(field_readiness.get("reports"))
    archive = _mapping(field_readiness.get("archive"))
    scope = _delivery_scope_payload(profile)
    receipts: list[dict[str, Any]] = []

    ingest = _mapping(reports.get("ingest_smoke"))
    if (
        gates.get("scenario_eval_passed") is True
        and gates.get("http_smoke_passed") is True
        and gates.get("trusted_device_events_observed") is True
        and gates.get("uses_real_hardware") is True
        and ingest.get("local_server") is False
    ):
        receipt = _customer_project_auto_onsite_evidence_receipt(
            evidence_type="device_ingest",
            label="Trusted device ingest",
            summary=(
                "Trusted field-device event archive and ingest smoke prove the "
                "customer project can receive real robot/camera/sensor events."
            ),
            path=str(archive.get("path") or ingest.get("path") or ""),
            profile=profile,
            project_scope=scope,
            external_reference=str(ingest.get("path") or ""),
        )
        if receipt:
            receipts.append(receipt)

    voice = _mapping(reports.get("voice_smoke"))
    if (
        gates.get("voice_smoke_passed") is True
        and gates.get("voice_smoke_uses_live_tts") is True
        and voice.get("local_server") is False
    ):
        receipt = _customer_project_auto_onsite_evidence_receipt(
            evidence_type="voice_playback",
            label="Live voice playback",
            summary="Live TTS voice smoke passed against an existing deployment endpoint.",
            path=str(voice.get("path") or ""),
            profile=profile,
            project_scope=scope,
            external_reference=f"voice_profile={voice.get('voice_profile') or ''}",
        )
        if receipt:
            receipts.append(receipt)

    notification = _mapping(reports.get("notification_smoke"))
    if (
        gates.get("notification_smoke_passed") is True
        and gates.get("notification_smoke_uses_external_services") is True
        and notification.get("local_server") is False
    ):
        receipt = _customer_project_auto_onsite_evidence_receipt(
            evidence_type="notification_delivery",
            label="External notification delivery",
            summary="Real responder notification smoke used external services.",
            path=str(notification.get("path") or ""),
            profile=profile,
            project_scope=scope,
            external_reference=(
                f"collector_request_count={notification.get('collector_request_count') or 0}"
            ),
        )
        if receipt:
            receipts.append(receipt)

    runtime = _mapping(reports.get("runtime_roundtrip"))
    if (
        gates.get("runtime_roundtrip_smoke_passed") is True
        and gates.get("runtime_roundtrip_final_status_verified") is True
        and runtime.get("trusted_callbacks") is True
        and runtime.get("local_server") is False
    ):
        receipt = _customer_project_auto_onsite_evidence_receipt(
            evidence_type="runtime_roundtrip",
            label="Runtime roundtrip callback",
            summary=(
                "Runtime arbiter roundtrip produced trusted callbacks and a verified final status."
            ),
            path=str(runtime.get("path") or ""),
            profile=profile,
            project_scope=scope,
            external_reference=f"final_status={runtime.get('final_status') or ''}",
        )
        if receipt:
            receipts.append(receipt)

    return receipts


def _customer_project_auto_onsite_evidence_receipt(
    *,
    evidence_type: str,
    label: str,
    summary: str,
    path: str,
    profile: dict[str, Any],
    project_scope: dict[str, Any],
    external_reference: str = "",
) -> dict[str, Any] | None:
    path = str(path or "").strip()
    if not path:
        return None
    inventory = _evidence_file_inventory(path, evidence_url=_evidence_url(path))
    if inventory.get("exists") is not True or inventory.get("sha256") == "":
        return None
    sha = str(inventory.get("sha256") or "")
    recorded_at = _evidence_file_modified_at(path)
    receipt_id = _slug(f"auto-field-readiness-{evidence_type}-{sha[:16]}")
    return {
        "receipt_type": "askme.customer_project_onsite_evidence",
        "receipt_version": 1,
        "receipt_id": receipt_id,
        "recorded_at": recorded_at,
        "operator_id": "field_readiness.auto",
        "reason": "Auto-surfaced from verified real-link readiness evidence.",
        "evidence_type": evidence_type,
        "status": "passed",
        "source": "field_readiness_auto_backfill",
        "label": label,
        "summary": summary,
        "path": path,
        "evidence_url": str(inventory.get("evidence_url") or _evidence_url(path)),
        "exists": True,
        "size_bytes": int(inventory.get("size_bytes") or 0),
        "sha256": sha,
        "project_scope": project_scope,
        "managed_object_id": "",
        "event_id": "",
        "runtime_run_id": "",
        "external_reference": external_reference,
        "evidence_tier": "site_acceptance",
        "production_eligible": False,
        "auto_backfill": {
            "source": "field_readiness",
            "profile_scope": _delivery_scope_payload(profile),
        },
    }


def _customer_project_onsite_evidence_receipts(profile: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for item in _customer_project_raw_onsite_evidence(profile):
        evidence_type = _normalize_onsite_evidence_type(item.get("evidence_type") or item.get("type"))
        status = _normalize_onsite_evidence_status(item.get("status"))
        path = str(item.get("path") or item.get("evidence_path") or "").strip()
        inventory = _evidence_file_inventory(path, evidence_url=_evidence_url(path)) if path else {}
        receipt = {
            "receipt_type": str(item.get("receipt_type") or "askme.customer_project_onsite_evidence"),
            "receipt_version": int(item.get("receipt_version") or 1),
            "receipt_id": str(item.get("receipt_id") or _slug(f"{evidence_type}-{item.get('recorded_at') or ''}")),
            "recorded_at": _float_value(item.get("recorded_at")),
            "operator_id": str(item.get("operator_id") or "system"),
            "reason": str(item.get("reason") or ""),
            "evidence_type": evidence_type,
            "status": status,
            "source": str(item.get("source") or ""),
            "label": str(item.get("label") or evidence_type.replace("_", " ").title()),
            "summary": str(item.get("summary") or item.get("note") or ""),
            "path": path,
            "evidence_url": str(inventory.get("evidence_url") or item.get("evidence_url") or _evidence_url(path)),
            "exists": bool(inventory.get("exists")) if path else bool(item.get("exists")),
            "size_bytes": int(inventory.get("size_bytes") or item.get("size_bytes") or 0),
            "sha256": str(inventory.get("sha256") or item.get("sha256") or ""),
            "project_scope": _mapping(item.get("project_scope")),
            "managed_object_id": str(item.get("managed_object_id") or ""),
            "event_id": str(item.get("event_id") or ""),
            "runtime_run_id": str(item.get("runtime_run_id") or ""),
            "external_reference": str(item.get("external_reference") or ""),
            "evidence_tier": str(item.get("evidence_tier") or "acceptance_candidate"),
            "production_eligible": item.get("production_eligible") is True,
        }
        if inventory.get("error") or item.get("error"):
            receipt["error"] = str(inventory.get("error") or item.get("error") or "")
        receipts.append(receipt)
    receipts.sort(key=lambda receipt: float(receipt.get("recorded_at") or 0), reverse=True)
    return receipts


def _evidence_file_modified_at(path: str) -> float:
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
        return float(resolved.stat().st_mtime)
    except OSError:
        return 0.0
    except ValueError:
        return 0.0


def _customer_project_onsite_evidence_summary(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    latest_by_type: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        evidence_type = str(receipt.get("evidence_type") or "")
        if evidence_type and evidence_type not in latest_by_type:
            latest_by_type[evidence_type] = receipt
    passed_required = [
        item
        for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES
        if _mapping(latest_by_type.get(item)).get("status") == "passed"
    ]
    failed_required = [
        item
        for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES
        if _mapping(latest_by_type.get(item)).get("status") == "failed"
    ]
    manual_required = [
        item
        for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES
        if _mapping(latest_by_type.get(item)).get("status") == "manual_check"
    ]
    missing_required = [
        item for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES if item not in latest_by_type
    ]
    if failed_required:
        overall = "blocked"
        customer_status = "现场验收证据存在失败项，不能提交客户验收。"
        next_step = "重新执行失败的现场 smoke，并补充可核验的照片、回调或通知证据。"
    elif not missing_required and not manual_required:
        overall = "ready"
        customer_status = "必需现场证据已登记，可进入客户验收复核。"
        next_step = "导出验收 dossier，并由交付负责人提交客户签收。"
    else:
        overall = "manual_check"
        customer_status = "现场证据仍不完整，需要交付负责人补录或复核。"
        next_step = "补齐设备上报、语音播报、外部通知和运行回调的现场证据。"
    by_status = {
        status: len([receipt for receipt in receipts if receipt.get("status") == status])
        for status in sorted(ONSITE_ACCEPTANCE_STATUSES)
    }
    return {
        "overall_status": overall,
        "receipt_count": len(receipts),
        "passed_count": by_status.get("passed", 0),
        "failed_count": by_status.get("failed", 0),
        "manual_check_count": by_status.get("manual_check", 0),
        "required_count": len(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES),
        "passed_required_count": len(passed_required),
        "passed_required_types": passed_required,
        "failed_required_types": failed_required,
        "manual_check_required_types": manual_required,
        "missing_required_types": missing_required,
        "latest_receipt_id": str(receipts[0].get("receipt_id") or "") if receipts else "",
        "by_status": by_status,
        "customer_status": customer_status,
        "next_step": next_step,
    }


def _customer_project_raw_acceptance_reviews(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("acceptance_reviews")
    if isinstance(raw, dict):
        raw = raw.get("reviews") or raw.get("items") or []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _customer_project_acceptance_reviews(profile: dict[str, Any]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for item in _customer_project_raw_acceptance_reviews(profile):
        review = {
            "review_type": str(item.get("review_type") or "askme.customer_project_acceptance_review"),
            "review_version": int(item.get("review_version") or 1),
            "review_id": str(item.get("review_id") or _slug(f"{item.get('decision')}-{item.get('reviewed_at') or ''}")),
            "reviewed_at": _float_value(item.get("reviewed_at")),
            "operator_id": str(item.get("operator_id") or "system"),
            "decision": _normalize_acceptance_review_decision(item.get("decision")),
            "reason": str(item.get("reason") or ""),
            "risk_acknowledgement": item.get("risk_acknowledgement") is True,
            "evidence_refs": [
                str(ref)
                for ref in (item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else [])
                if str(ref).strip()
            ],
            "project_scope": _mapping(item.get("project_scope")),
        }
        reviews.append(review)
    reviews.sort(key=lambda item: float(item.get("reviewed_at") or 0), reverse=True)
    return reviews


def _customer_project_acceptance_review_gate(latest_review: dict[str, Any]) -> dict[str, Any]:
    decision = str(latest_review.get("decision") or "")
    if not latest_review:
        status = "manual_check"
        evidence = "No acceptance review submitted."
        next_step = "Delivery owner must review onsite evidence and submit a decision."
    elif decision == "accepted" and latest_review.get("risk_acknowledgement") is True:
        status = "ready"
        evidence = f"accepted by {latest_review.get('operator_id') or 'system'}"
        next_step = "Export dossier and submit customer signoff."
    elif decision == "rejected":
        status = "blocked"
        evidence = f"rejected by {latest_review.get('operator_id') or 'system'}"
        next_step = str(latest_review.get("reason") or "Resolve the rejected acceptance review.")
    else:
        status = "manual_check"
        evidence = f"{decision or 'review'} by {latest_review.get('operator_id') or 'system'}"
        next_step = str(latest_review.get("reason") or "Resolve review notes before customer signoff.")
    return {
        "gate_id": "manual_acceptance_review",
        "label": "Manual acceptance review",
        "status": status,
        "evidence": evidence,
        "next_step": next_step,
    }


def _customer_project_raw_customer_signoffs(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("customer_signoffs")
    if isinstance(raw, dict):
        raw = raw.get("signoffs") or raw.get("items") or []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _customer_project_customer_signoffs(profile: dict[str, Any]) -> list[dict[str, Any]]:
    signoffs: list[dict[str, Any]] = []
    for item in _customer_project_raw_customer_signoffs(profile):
        signoff = {
            "signoff_type": str(item.get("signoff_type") or "askme.customer_project_customer_signoff"),
            "signoff_version": int(item.get("signoff_version") or 1),
            "signoff_id": str(
                item.get("signoff_id")
                or _slug(f"{item.get('decision')}-{item.get('signed_at') or ''}")
            ),
            "signed_at": _float_value(item.get("signed_at")),
            "operator_id": str(item.get("operator_id") or "system"),
            "decision": _normalize_customer_signoff_decision(item.get("decision")),
            "signatory_name": str(item.get("signatory_name") or ""),
            "signatory_role": str(item.get("signatory_role") or ""),
            "organization": str(item.get("organization") or ""),
            "reason": str(item.get("reason") or ""),
            "risk_acknowledgement": item.get("risk_acknowledgement") is True,
            "credential_ref": str(item.get("credential_ref") or ""),
            "credential_sha256": _normalize_sha256_hex(item.get("credential_sha256")),
            "evidence_refs": [
                str(ref)
                for ref in (item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else [])
                if str(ref).strip()
            ],
            "gate_snapshot": _mapping(item.get("gate_snapshot")),
            "handoff_materials": _mapping(item.get("handoff_materials")),
            "project_scope": _mapping(item.get("project_scope")),
            "signoff_payload_sha256": str(item.get("signoff_payload_sha256") or ""),
        }
        expected_sha = _customer_project_customer_signoff_payload_sha256(signoff)
        stored_sha = str(signoff.get("signoff_payload_sha256") or "")
        signoff["integrity_valid"] = not stored_sha or stored_sha == expected_sha
        signoffs.append(signoff)
    signoffs.sort(key=lambda item: float(item.get("signed_at") or 0), reverse=True)
    return signoffs


def _customer_project_customer_signoff_gate_snapshot(closure: dict[str, Any]) -> dict[str, Any]:
    gates = closure.get("gates") if isinstance(closure.get("gates"), list) else []
    return {
        "snapshot_version": 1,
        "captured_at": time.time(),
        "overall_status": str(closure.get("overall_status") or ""),
        "customer_claim": str(closure.get("customer_claim") or ""),
        "next_step": str(closure.get("next_step") or ""),
        "gates": [
            {
                "gate_id": str(_mapping(gate).get("gate_id") or ""),
                "label": str(_mapping(gate).get("label") or ""),
                "status": str(_mapping(gate).get("status") or ""),
                "evidence": str(_mapping(gate).get("evidence") or ""),
                "next_step": str(_mapping(gate).get("next_step") or ""),
            }
            for gate in gates
            if _mapping(gate).get("gate_id")
        ],
        "acceptance_report": _mapping(closure.get("acceptance_report")),
    }


def _customer_project_customer_signoff_handoff_materials(
    closure: dict[str, Any],
    *,
    evidence_refs: list[str],
) -> dict[str, Any]:
    artifacts = _mapping(closure.get("artifact_verification"))
    dossier = _mapping(artifacts.get("acceptance_dossier"))
    proposal = _mapping(artifacts.get("proposal_bundle"))
    audit = _mapping(artifacts.get("audit_export"))
    return {
        "material_version": 1,
        "evidence_refs": list(evidence_refs),
        "acceptance_dossier": {
            "valid": bool(dossier.get("valid")),
            "reason": str(dossier.get("reason") or ""),
            "manifest": _mapping(dossier.get("manifest")),
        },
        "proposal_bundle": {
            "status": str(proposal.get("status") or ""),
            "evidence": str(proposal.get("evidence") or ""),
            "proposal_path": str(proposal.get("proposal_path") or ""),
        },
        "audit_export": {
            "status": str(audit.get("status") or ""),
            "evidence": str(audit.get("evidence") or ""),
            "manifest_path": str(audit.get("manifest_path") or ""),
            "sha256": str(audit.get("sha256") or ""),
        },
    }


def _customer_project_customer_signoff_payload_sha256(signoff: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in signoff.items()
        if key not in {"signoff_payload_sha256", "integrity_valid"}
    }
    return _sha256_json(payload)


def _customer_project_customer_signoff_gate(
    latest_signoff: dict[str, Any],
    *,
    base_ready_for_signoff: bool,
) -> dict[str, Any]:
    decision = str(latest_signoff.get("decision") or "")
    if not latest_signoff:
        status = "manual_check"
        evidence = "No customer signoff submitted."
        next_step = (
            "客户签收前仍需补齐内部交付门禁。"
            if not base_ready_for_signoff
            else "提交客户签收，并归档签收人、签收意见和证据引用。"
        )
    elif decision == "accepted" and latest_signoff.get("risk_acknowledgement") is True:
        if not latest_signoff.get("credential_sha256") or not latest_signoff.get("credential_ref"):
            status = "blocked"
            evidence = "Customer signoff is accepted but missing credential ref/hash."
            next_step = "补齐客户签收凭证引用和 SHA-256 哈希后重新登记签收。"
        elif latest_signoff.get("integrity_valid") is False:
            status = "blocked"
            evidence = "Customer signoff record hash mismatch."
            next_step = "签收记录完整性校验失败，必须重新核对签收材料。"
        else:
            status = "ready"
            evidence = (
                f"accepted by {latest_signoff.get('signatory_name') or 'customer'}; "
                f"credential={str(latest_signoff.get('credential_sha256') or '')[:12]}"
            )
            next_step = "客户签收已归档，保留验收包和审计证据。"
    elif decision == "accepted":
        status = "blocked"
        evidence = f"accepted by {latest_signoff.get('signatory_name') or 'customer'} but risk acknowledgement is missing"
        next_step = "补齐客户风险确认后重新登记签收。"
    elif decision == "rejected":
        status = "blocked"
        evidence = f"rejected by {latest_signoff.get('signatory_name') or 'customer'}"
        next_step = str(latest_signoff.get("reason") or "处理客户拒收意见后重新提交验收。")
    else:
        status = "manual_check"
        evidence = f"{decision or 'signoff'} by {latest_signoff.get('signatory_name') or 'customer'}"
        next_step = str(latest_signoff.get("reason") or "处理客户整改意见后重新提交签收。")
    return {
        "gate_id": "customer_signoff",
        "label": "Customer signoff",
        "status": status,
        "evidence": evidence,
        "next_step": next_step,
        "base_ready_for_signoff": base_ready_for_signoff,
    }


def _customer_project_acceptance_evidence_timeline(
    onsite: dict[str, Any],
    reviews: list[dict[str, Any]],
    signoffs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    receipts = onsite.get("receipts") if isinstance(onsite.get("receipts"), list) else []
    timeline: list[dict[str, Any]] = []
    for receipt in receipts:
        item = _mapping(receipt)
        timeline.append({
            "timestamp": item.get("recorded_at"),
            "type": "onsite_evidence",
            "status": item.get("status"),
            "label": item.get("label") or item.get("evidence_type"),
            "summary": item.get("summary") or item.get("path"),
            "ref": item.get("receipt_id"),
        })
    for review in reviews:
        item = _mapping(review)
        timeline.append({
            "timestamp": item.get("reviewed_at"),
            "type": "acceptance_review",
            "status": item.get("decision"),
            "label": f"review by {item.get('operator_id') or 'system'}",
            "summary": item.get("reason"),
            "ref": item.get("review_id"),
        })
    for signoff in signoffs or []:
        item = _mapping(signoff)
        timeline.append({
            "timestamp": item.get("signed_at"),
            "type": "customer_signoff",
            "status": item.get("decision"),
            "label": f"signoff by {item.get('signatory_name') or 'customer'}",
            "summary": item.get("reason"),
            "ref": item.get("signoff_id"),
        })
    timeline.sort(key=lambda item: float(item.get("timestamp") or 0), reverse=True)
    return timeline[:30]


def _customer_project_acceptance_closure_next_step(
    overall: str,
    gates: list[dict[str, Any]],
) -> str:
    if overall == "accepted_by_customer":
        return "客户签收已归档，后续只允许通过复盘或变更流程更新验收结论。"
    if overall == "ready_for_customer_signoff":
        return "导出验收 dossier，提交客户签收，并归档签收结果。"
    blocked = next((gate for gate in gates if gate.get("status") == "blocked"), {})
    if blocked:
        return str(blocked.get("next_step") or "先处理阻断项。")
    manual = next((gate for gate in gates if gate.get("status") == "manual_check"), {})
    return str(manual.get("next_step") or "补齐现场证据并提交人工复核。")


def _customer_project_latest_proposal_verification(profile: dict[str, Any]) -> dict[str, Any]:
    scope = _delivery_scope_payload(profile)
    root = PROJECT_ROOT / "artifacts" / "customer-project-proposals"
    latest: dict[str, Any] = {}
    for path in _recent_json_files(root, "*proposal-bundle.json"):
        payload = _read_json_file(path)
        if not payload:
            continue
        proposal_scope = _delivery_scope_payload_from_customer_site(
            _mapping(payload.get("customer")),
            _mapping(payload.get("site")),
        )
        if not _same_delivery_project_scope(scope, proposal_scope):
            continue
        verification = verify_customer_project_proposal_bundle(payload)
        latest = {
            "status": "ready" if verification.get("valid") else "blocked",
            "evidence": f"{path} valid={bool(verification.get('valid'))}",
            "next_step": (
                "Proposal bundle manifest is valid."
                if verification.get("valid")
                else "Regenerate the proposal bundle before sending it to the customer."
            ),
            "proposal_path": str(path),
            "verification": verification,
        }
        break
    if latest:
        return latest
    return {
        "status": "manual_check",
        "evidence": "No matching proposal bundle found under artifacts/customer-project-proposals.",
        "next_step": "Export and verify the proposal bundle before customer handoff.",
    }


def _customer_project_latest_audit_export(profile: dict[str, Any]) -> dict[str, Any]:
    scope = _delivery_scope_payload(profile)
    root = PROJECT_ROOT / "artifacts" / "audit_exports"
    for path in _recent_json_files(root, "*.manifest.json"):
        manifest = _read_json_file(path)
        if not manifest:
            continue
        filters = _mapping(manifest.get("filters"))
        if not _audit_manifest_matches_scope(scope, filters):
            continue
        records_path = str(manifest.get("records_path") or "")
        verified = _audit_records_hash_matches(manifest)
        return {
            "status": "ready" if verified else "blocked",
            "evidence": f"{path} records={manifest.get('record_count') or 0} hash={'ok' if verified else 'mismatch'}",
            "next_step": (
                "Audit export is available for the customer handoff."
                if verified
                else "Regenerate the scoped audit export; the records hash cannot be verified."
            ),
            "manifest_path": str(path),
            "records_path": records_path,
            "export_id": str(manifest.get("export_id") or ""),
            "record_count": int(manifest.get("record_count") or 0),
            "sha256": str(manifest.get("sha256") or ""),
        }
    return {
        "status": "manual_check",
        "evidence": "No matching audit export manifest found under artifacts/audit_exports.",
        "next_step": "Create a scoped audit export after onsite evidence and review are complete.",
    }


def _recent_json_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.glob(pattern) if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _audit_manifest_matches_scope(scope: dict[str, str], filters: dict[str, Any]) -> bool:
    for field in ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id"):
        expected = str(scope.get(field) or "")
        value = str(filters.get(field) or "")
        if expected and value and expected != value:
            return False
    return bool(
        str(filters.get("project_id") or "")
        or str(filters.get("customer_id") or "")
        or str(filters.get("site_id") or "")
    )


def _audit_records_hash_matches(manifest: dict[str, Any]) -> bool:
    records_path = str(manifest.get("records_path") or "")
    expected = str(manifest.get("sha256") or "")
    if not records_path or not expected:
        return False
    try:
        resolved = Path(records_path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        resolved = resolved.resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
        data = resolved.read_bytes()
    except (OSError, ValueError):
        return False
    return hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected)


def _normalize_onsite_evidence_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _normalize_onsite_evidence_status(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "manual_check").strip().lower()).strip("_")
    if text in {"pass", "success", "ok", "ready"}:
        return "passed"
    if text in {"fail", "error", "blocked"}:
        return "failed"
    if text in {"manual", "review", "pending", "unknown", ""}:
        return "manual_check"
    return text


def _normalize_acceptance_review_decision(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if text in {"accept", "pass", "passed", "approved", "approve"}:
        return "accepted"
    if text in {"fix", "need_fix", "needs_review", "manual_check"}:
        return "needs_fix"
    if text in {"reject", "fail", "failed", "blocked"}:
        return "rejected"
    return text


def _normalize_customer_signoff_decision(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if text in {"accept", "accepted", "approved", "approve", "signed", "sign"}:
        return "accepted"
    if text in {"fix", "need_fix", "needs_review", "needs_fix", "manual_check"}:
        return "needs_fix"
    if text in {"reject", "rejected", "fail", "failed", "blocked"}:
        return "rejected"
    return text


def _normalize_sha256_hex(value: Any) -> str:
    text = re.sub(r"[^a-fA-F0-9]+", "", str(value or "").strip())
    if len(text) != 64:
        return ""
    return text.lower()


def _float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _customer_project_field_readiness_gates(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    gates = _mapping(readiness.get("gates"))
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    warnings = readiness.get("warnings") if isinstance(readiness.get("warnings"), list) else []
    next_actions = readiness.get("next_actions") if isinstance(readiness.get("next_actions"), list) else []
    reports = _mapping(readiness.get("reports"))
    return [
        {
            "gate_id": "field_readiness",
            "label": "Field readiness",
            "status": _readiness_status(readiness.get("status")),
            "evidence": (
                f"status={readiness.get('status') or 'unknown'}; "
                f"blockers={len(blockers)} warnings={len(warnings)}"
            ),
            "next_step": str(next_actions[0] if next_actions else _mapping(readiness.get("delivery_brief")).get("release_claim") or "-"),
        },
        {
            "gate_id": "field_smoke_evidence",
            "label": "Scenario and ingest smoke",
            "status": _boolean_gate_status(
                gates,
                required=("scenario_eval_passed", "http_smoke_passed", "archive_has_events"),
                manual=("uses_real_hardware",),
            ),
            "evidence": _reports_evidence(reports, ("scenario", "ingest_smoke")),
            "next_step": "Run the bridge against real camera, sensor, and robot diagnostic input." if not gates.get("uses_real_hardware") else "Scenario and ingest smoke evidence is ready.",
        },
        {
            "gate_id": "voice_notification_evidence",
            "label": "Voice and notification smoke",
            "status": _boolean_gate_status(
                gates,
                required=("voice_smoke_passed", "notification_smoke_passed"),
                manual=("voice_smoke_uses_live_tts", "notification_smoke_uses_external_services"),
            ),
            "evidence": _reports_evidence(reports, ("voice_smoke", "notification_smoke")),
            "next_step": "Run live TTS and real DingTalk smoke before customer production acceptance.",
        },
        {
            "gate_id": "runtime_audit_trust",
            "label": "Runtime, audit, and device trust",
            "status": _boolean_gate_status(
                gates,
                required=(
                    "runtime_roundtrip_smoke_passed",
                    "runtime_roundtrip_final_status_verified",
                    "action_audit_integrity_verified",
                    "unified_audit_review_clear",
                ),
                manual=(
                    "action_audit_signed",
                    "all_registered_devices_signature_ready",
                    "trusted_device_events_observed",
                ),
            ),
            "evidence": _reports_evidence(reports, ("runtime_roundtrip",)),
            "next_step": "Resolve audit review, configure device secrets, and rerun signed live runtime smoke.",
        },
    ]


def _readiness_status(status: Any) -> str:
    text = str(status or "").strip()
    if text in {"production_ready", "ready_for_onsite_acceptance", "ready_for_acceptance", "ready"}:
        return "ready"
    if text == "ready_for_lab":
        return "manual_check"
    if text == "blocked":
        return "blocked"
    return "manual_check"


def _boolean_gate_status(
    gates: dict[str, Any],
    *,
    required: tuple[str, ...],
    manual: tuple[str, ...] = (),
) -> str:
    if any(gates.get(name) is not True for name in required):
        return "blocked"
    if any(gates.get(name) is not True for name in manual):
        return "manual_check"
    return "ready"


def _reports_evidence(reports: dict[str, Any], names: tuple[str, ...]) -> str:
    parts: list[str] = []
    for name in names:
        report = _mapping(reports.get(name))
        status = str(report.get("status") or "unknown")
        path = str(report.get("path") or "")
        if path:
            parts.append(f"{name}={status} ({path})")
        else:
            parts.append(f"{name}={status}")
    return "; ".join(parts)


def _customer_project_site_acceptance_checklist(
    *,
    profile: dict[str, Any],
    report: dict[str, Any],
    acceptance: dict[str, Any],
    field_readiness: dict[str, Any],
    onsite_evidence: dict[str, Any],
    missing_env: list[dict[str, Any]],
) -> dict[str, Any]:
    receipts = (
        onsite_evidence.get("receipts")
        if isinstance(onsite_evidence.get("receipts"), list)
        else []
    )
    latest_by_type = _latest_onsite_receipts_by_type(receipts)
    gates = _mapping(field_readiness.get("gates"))
    reports = _mapping(field_readiness.get("reports"))
    scope = _delivery_scope_payload(profile)
    items = [
        {
            "item_id": "site_profile",
            "label": "Site profile and customer scope",
            "owner": "delivery",
            "status": "ready" if report.get("status") == "passed" else "blocked",
            "evidence": str(report.get("profile_path") or ""),
            "next_step": "Fix site profile validation errors." if report.get("status") != "passed" else "Site profile is valid.",
            "required_for_customer_acceptance": True,
            "project_scope": scope,
        },
        {
            "item_id": "managed_object_acceptance",
            "label": "Managed object acceptance bindings",
            "owner": "product",
            "status": str(acceptance.get("overall_status") or "manual_check"),
            "evidence": (
                f"{acceptance.get('ready_object_count') or 0}/"
                f"{acceptance.get('object_count') or 0} objects ready"
            ),
            "next_step": str(acceptance.get("customer_status") or ""),
            "required_for_customer_acceptance": True,
            "project_scope": scope,
        },
        {
            "item_id": "deployment_credentials",
            "label": "Deployment credentials",
            "owner": "delivery",
            "status": "manual_check" if missing_env else "ready",
            "evidence": f"{len(missing_env)} missing required env reference(s)",
            "next_step": (
                "Configure DingTalk, device, voice, and runtime secrets before site acceptance."
                if missing_env
                else "Required deployment credentials are configured."
            ),
            "required_for_customer_acceptance": True,
            "missing_env": [
                str(item.get("env_name") or "")
                for item in missing_env
                if str(item.get("env_name") or "").strip()
            ],
            "project_scope": scope,
        },
        _onsite_acceptance_checklist_item(
            item_id="device_ingest",
            label="Trusted device ingest",
            owner="field-engineering",
            evidence_type="device_ingest",
            receipt=_mapping(latest_by_type.get("device_ingest")),
            fallback_evidence=_reports_evidence(reports, ("scenario", "ingest_smoke")),
            fallback_status=_boolean_gate_status(
                gates,
                required=("scenario_eval_passed", "http_smoke_passed", "trusted_device_events_observed"),
                manual=("uses_real_hardware",),
            ),
            next_step=(
                "Run deployed ingest against signed camera, sensor, or robot payloads and keep the field event archive."
            ),
            project_scope=scope,
        ),
        _onsite_acceptance_checklist_item(
            item_id="voice_playback",
            label="Live voice playback",
            owner="voice-engineering",
            evidence_type="voice_playback",
            receipt=_mapping(latest_by_type.get("voice_playback")),
            fallback_evidence=_reports_evidence(reports, ("voice_smoke",)),
            fallback_status=_boolean_gate_status(
                gates,
                required=("voice_smoke_passed",),
                manual=("voice_smoke_uses_live_tts",),
            ),
            next_step="Run live MiniMax/production TTS smoke against the deployed service.",
            project_scope=scope,
        ),
        _onsite_acceptance_checklist_item(
            item_id="notification_delivery",
            label="External notification delivery",
            owner="delivery",
            evidence_type="notification_delivery",
            receipt=_mapping(latest_by_type.get("notification_delivery")),
            fallback_evidence=_reports_evidence(reports, ("notification_smoke",)),
            fallback_status=_boolean_gate_status(
                gates,
                required=("notification_smoke_passed",),
                manual=("notification_smoke_uses_external_services",),
            ),
            next_step="Run DingTalk smoke with the real responder group webhook and secret.",
            project_scope=scope,
        ),
        _onsite_acceptance_checklist_item(
            item_id="runtime_roundtrip",
            label="Robot runtime roundtrip",
            owner="runtime-engineering",
            evidence_type="runtime_roundtrip",
            receipt=_mapping(latest_by_type.get("runtime_roundtrip")),
            fallback_evidence=_reports_evidence(reports, ("runtime_roundtrip",)),
            fallback_status=_boolean_gate_status(
                gates,
                required=("runtime_roundtrip_smoke_passed", "runtime_roundtrip_final_status_verified"),
                manual=("trusted_device_events_observed",),
            ),
            next_step="Run signed runtime callback roundtrip against the lab or site runtime arbiter.",
            project_scope=scope,
        ),
        {
            "item_id": "audit_and_operator_review",
            "label": "Audit integrity and operator review",
            "owner": "operations",
            "status": _boolean_gate_status(
                gates,
                required=("action_audit_integrity_verified", "unified_audit_review_clear"),
                manual=("action_audit_signed", "all_registered_devices_signature_ready"),
            ),
            "evidence": (
                f"action_audit_integrity={bool(gates.get('action_audit_integrity_verified'))}; "
                f"signed={bool(gates.get('action_audit_signed'))}; "
                f"device_signatures={bool(gates.get('all_registered_devices_signature_ready'))}"
            ),
            "next_step": "Clear audit review items and configure HMAC/device signatures before production acceptance.",
            "required_for_customer_acceptance": True,
            "project_scope": scope,
        },
    ]
    statuses = {str(item.get("status") or "") for item in items}
    if "blocked" in statuses:
        overall = "blocked"
    elif "manual_check" in statuses:
        overall = "manual_check"
    else:
        overall = "ready"
    return {
        "checklist_version": 1,
        "overall_status": overall,
        "ready_count": len([item for item in items if item.get("status") == "ready"]),
        "manual_check_count": len([item for item in items if item.get("status") == "manual_check"]),
        "blocked_count": len([item for item in items if item.get("status") == "blocked"]),
        "item_count": len(items),
        "items": items,
        "customer_message": {
            "ready": "All customer-site acceptance checklist items are ready.",
            "manual_check": "Customer-site acceptance still needs delivery review or real-link evidence.",
            "blocked": "Customer-site acceptance has blocked items and must not be claimed as ready.",
        }.get(overall, "Customer-site acceptance status is unknown."),
    }


def _latest_onsite_receipts_by_type(receipts: list[Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        item = _mapping(receipt)
        evidence_type = str(item.get("evidence_type") or "")
        if evidence_type and evidence_type not in latest:
            latest[evidence_type] = item
    return latest


def _onsite_acceptance_checklist_item(
    *,
    item_id: str,
    label: str,
    owner: str,
    evidence_type: str,
    receipt: dict[str, Any],
    fallback_evidence: str,
    fallback_status: str,
    next_step: str,
    project_scope: dict[str, Any],
) -> dict[str, Any]:
    receipt_status = str(receipt.get("status") or "")
    if receipt_status == "passed":
        status = "ready"
    elif receipt_status == "failed":
        status = "blocked"
    elif receipt_status == "manual_check":
        status = "manual_check"
    else:
        status = fallback_status if fallback_status == "blocked" else "manual_check"
    receipt_id = str(receipt.get("receipt_id") or "")
    source = str(receipt.get("source") or "")
    evidence = (
        f"receipt={receipt_id}; source={source}; sha={str(receipt.get('sha256') or '')[:16]}"
        if receipt_id
        else fallback_evidence
    )
    return {
        "item_id": item_id,
        "label": label,
        "owner": owner,
        "status": status,
        "evidence_type": evidence_type,
        "receipt_id": receipt_id,
        "source": source,
        "evidence": evidence,
        "next_step": "Evidence receipt is available." if status == "ready" else next_step,
        "required_for_customer_acceptance": True,
        "project_scope": project_scope,
    }


def upsert_managed_object(
    profile_root: Path,
    identifier: str,
    object_id: str,
    payload: dict[str, Any],
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Create or update one managed object inside a customer project profile."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    clean_object_id = _slug(object_id)
    if not clean_object_id or clean_object_id == "item":
        return {"accepted": False, "reason": "object_id_required"}
    if not isinstance(payload, dict):
        return {"accepted": False, "reason": "managed_object_must_be_mapping"}
    profile = load_field_site_profile(path)
    profile.setdefault("managed_objects", {})
    managed_objects = _mapping(profile.get("managed_objects"))
    before_object = _clean_nested_mapping(managed_objects.get(clean_object_id))
    managed_objects[clean_object_id] = _clean_nested_mapping(payload)
    profile["managed_objects"] = managed_objects
    report = validate_field_site_profile(profile)
    if report["status"] != "passed":
        return {"accepted": False, "reason": "profile_validation_failed", "report": report}
    _snapshot_customer_project_revision(
        profile_root,
        path,
        action="managed_object_upsert",
        operator_id=operator_id,
        reason=reason,
    )
    change = _append_object_change_log(
        profile,
        action="updated" if before_object else "created",
        object_id=clean_object_id,
        operator_id=operator_id,
        reason=reason,
        before=before_object,
        after=managed_objects[clean_object_id],
    )
    _write_yaml(path, profile)
    implementation_handoff = _customer_project_implementation_handoff(
        profile,
        template_id=str(_mapping(profile.get("template")).get("template_id") or ""),
        profile_path=path,
    )
    return {
        "accepted": True,
        "profile_path": str(path),
        "object_id": clean_object_id,
        "object_change": change,
        "managed_object": managed_object_catalog_from_site_profile(profile)["objects_by_id"][clean_object_id],
        "report": build_site_profile_report(path),
        "implementation_handoff": implementation_handoff,
        "next_step": implementation_handoff.get("customer_status", ""),
    }


def delete_managed_object(
    profile_root: Path,
    identifier: str,
    object_id: str,
    *,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Remove one managed object from a customer project profile."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    clean_object_id = _slug(object_id)
    clean_reason = str(reason or "").strip()
    if not clean_reason:
        return {
            "accepted": False,
            "reason": "delete_reason_required",
            "message": "Managed-object removal requires a customer-visible offline reason.",
        }
    profile = load_field_site_profile(path)
    managed_objects = _mapping(profile.get("managed_objects"))
    if clean_object_id not in managed_objects:
        return {"accepted": False, "reason": "managed_object_not_found"}
    deleted_object = _clean_nested_mapping(managed_objects.get(clean_object_id))
    managed_objects.pop(clean_object_id)
    profile["managed_objects"] = managed_objects
    report = validate_field_site_profile(profile)
    if report["status"] != "passed":
        return {"accepted": False, "reason": "profile_validation_failed", "report": report}
    _snapshot_customer_project_revision(
        profile_root,
        path,
        action="managed_object_delete",
        operator_id=operator_id,
        reason=clean_reason,
    )
    change = _append_object_change_log(
        profile,
        action="offline",
        object_id=clean_object_id,
        operator_id=operator_id,
        reason=clean_reason,
        before=deleted_object,
        after={},
    )
    _write_yaml(path, profile)
    implementation_handoff = _customer_project_implementation_handoff(
        profile,
        template_id=str(_mapping(profile.get("template")).get("template_id") or ""),
        profile_path=path,
    )
    return {
        "accepted": True,
        "profile_path": str(path),
        "object_id": clean_object_id,
        "offline_reason": clean_reason,
        "deleted_object": deleted_object,
        "object_change": change,
        "report": build_site_profile_report(path),
        "implementation_handoff": implementation_handoff,
        "next_step": implementation_handoff.get("customer_status", ""),
    }


def _append_object_change_log(
    profile: dict[str, Any],
    *,
    action: str,
    object_id: str,
    operator_id: str = "",
    reason: str = "",
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a compact customer-project object lifecycle record."""
    entries = profile.get("object_change_log")
    if not isinstance(entries, list):
        entries = []
    entry = {
        "timestamp": time.time(),
        "action": str(action or "updated"),
        "object_id": str(object_id or ""),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or ""),
        "before": _object_change_summary(before or {}),
        "after": _object_change_summary(after or {}),
    }
    profile["object_change_log"] = [*entries, entry][-100:]
    return entry


def _object_change_summary(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict) or not item:
        return {}
    bindings = _mapping(item.get("bindings"))
    return {
        "display_name": str(item.get("display_name") or ""),
        "category": str(item.get("category") or ""),
        "scenario_ids": _string_list(item.get("scenario_ids")),
        "zone_types": _string_list(item.get("zone_types")),
        "device_sources": _string_list(item.get("device_sources")),
        "responder_group": str(item.get("responder_group") or ""),
        "vision_models": _string_list(bindings.get("vision_models")),
        "sensor_protocols": _string_list(bindings.get("sensor_protocols")),
        "skill_packages": _string_list(bindings.get("skill_packages")),
        "acceptance_tests": _string_list(bindings.get("acceptance_tests")),
    }


def _object_change_log_payload(profile: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    entries = profile.get("object_change_log")
    if not isinstance(entries, list):
        return []
    clean_entries = [item for item in entries if isinstance(item, dict)]
    return clean_entries[-limit:]


def archive_customer_project_profile(
    profile_root: Path,
    identifier: str,
    *,
    archive_root: Path | None = None,
) -> dict[str, Any]:
    """Archive a customer project profile without deleting it permanently."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    root = archive_root or Path(profile_root) / "_archive"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = root / f"{stamp}-{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return {
        "accepted": True,
        "archived_path": str(target),
        "original_path": str(path),
        "next_step": "Review the archive before permanent cleanup.",
    }


def list_customer_project_revisions(
    profile_root: Path,
    identifier: str,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return saved customer project profile revisions for rollback review."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found", "revisions": [], "count": 0}
    profile = load_field_site_profile(path)
    revisions = _load_customer_project_revisions(profile_root, path, profile)
    clean_limit = max(1, min(int(limit or 20), 100))
    return {
        "found": True,
        "profile_path": str(path),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "current_profile_sha256": _sha256_json(profile),
        "revisions": revisions[:clean_limit],
        "count": len(revisions),
        "next_step": (
            "Use rollback dry-run before restoring a previous customer project revision."
            if revisions
            else "No saved revisions yet. A revision is created before every overwrite or object change."
        ),
    }


def rollback_customer_project_profile(
    profile_root: Path,
    identifier: str,
    revision_id: str,
    *,
    operator_id: str = "",
    reason: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore a customer project profile from a saved revision."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    current = load_field_site_profile(path)
    revision = _find_customer_project_revision(profile_root, path, current, revision_id)
    if revision is None:
        return {"accepted": False, "reason": "revision_not_found"}
    target_profile = revision.get("profile")
    if not isinstance(target_profile, dict):
        return {"accepted": False, "reason": "revision_missing_profile"}
    target_profile = _normalize_customer_project_profile(target_profile)
    if not _same_delivery_project_scope(
        _delivery_scope_payload(current),
        _delivery_scope_payload(target_profile),
    ):
        return {"accepted": False, "reason": "revision_scope_mismatch", "revision": revision}
    report = validate_field_site_profile(target_profile)
    if report["status"] != "passed":
        return {
            "accepted": False,
            "reason": "revision_profile_validation_failed",
            "report": report,
            "revision": _revision_public_payload(revision),
        }
    diff = _customer_project_profile_diff(current, target_profile)
    payload = {
        "accepted": True,
        "dry_run": bool(dry_run),
        "profile_path": str(path),
        "revision": _revision_public_payload(revision),
        "current_profile_sha256": _sha256_json(current),
        "target_profile_sha256": _sha256_json(target_profile),
        "field_changes": diff,
        "report": report,
    }
    if dry_run:
        payload["would_write"] = True
        return payload
    snapshot = _snapshot_customer_project_revision(
        profile_root,
        path,
        action="rollback_current",
        operator_id=operator_id,
        reason=reason or f"Rollback to revision {revision_id}.",
    )
    _write_yaml(path, target_profile)
    payload["rollback_snapshot"] = _revision_public_payload(snapshot)
    payload["next_step"] = "Run customer project acceptance report and onsite smoke checks after rollback."
    return payload


def _build_customer_project_acceptance_dossier(report: dict[str, Any]) -> dict[str, Any]:
    customer = _mapping(report.get("customer"))
    site = _mapping(report.get("site"))
    dossier = {
        "dossier_type": "askme.customer_project_acceptance",
        "dossier_version": 1,
        "exported_at": time.time(),
        "customer": customer,
        "site": site,
        "overall_status": report.get("overall_status"),
        "acceptance_summary": report.get("acceptance_summary"),
        "delivery_workflow": report.get("delivery_workflow"),
        "site_acceptance_checklist": report.get("site_acceptance_checklist"),
        "field_readiness": report.get("field_readiness"),
        "launch_readiness": report.get("launch_readiness"),
        "onsite_acceptance_evidence": report.get("onsite_acceptance_evidence"),
        "acceptance_reviews": report.get("acceptance_reviews"),
        "customer_signoffs": report.get("customer_signoffs"),
        "gates": report.get("gates"),
        "env_missing": report.get("env_missing"),
        "warnings": report.get("warnings"),
        "errors": report.get("errors"),
        "evidence_inventory": _customer_project_evidence_inventory(report),
        "release_claim": report.get("release_claim"),
        "customer_status": report.get("customer_status"),
        "handoff_boundary": (
            "This dossier is a customer handoff artifact. It records local and onsite evidence "
            "available at export time, but production launch still requires all gates to be ready."
        ),
    }
    dossier["manifest"] = _customer_project_acceptance_dossier_manifest(dossier)
    return dossier


def _customer_project_acceptance_dossier_verification(report: dict[str, Any]) -> dict[str, Any]:
    dossier = _build_customer_project_acceptance_dossier(report)
    verification = verify_customer_project_acceptance_dossier(dossier)
    return {
        **verification,
        "manifest": _mapping(verification.get("manifest")),
        "evidence_count": int(_mapping(dossier.get("manifest")).get("evidence_count") or 0),
        "onsite_evidence_status": str(
            _mapping(dossier.get("manifest")).get("onsite_evidence_status") or "manual_check"
        ),
        "manual_review_decision": str(
            _mapping(dossier.get("manifest")).get("manual_review_decision") or ""
        ),
    }


def export_customer_project_package(
    profile_root: Path,
    identifier: str,
    *,
    output_root: Path = Path("artifacts/customer-project-packages"),
) -> dict[str, Any]:
    """Export a reusable customer project handoff package."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"accepted": False, "reason": "profile_not_found"}
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    report = build_site_profile_report(path, check_env=True)
    customer = _customer_payload(profile)
    managed_object_catalog = managed_object_catalog_from_site_profile(profile)
    acceptance_summary = _customer_project_package_acceptance_summary(managed_object_catalog)
    env_references = site_profile_env_references(profile)
    reuse_assessment = _customer_project_package_reuse_assessment(
        profile=profile,
        report=report,
        managed_object_catalog=managed_object_catalog,
        acceptance_summary=acceptance_summary,
        env_references=env_references,
    )
    customer_delivery = _customer_delivery_surface(
        profile=profile,
        template=_mapping(profile.get("template")),
        customer=customer,
        managed_objects=managed_object_catalog,
        report=report,
        env_references=env_references,
        surface="project",
    )
    action_plan = _customer_project_package_action_plan(managed_object_catalog)
    package_delivery_gate = _customer_project_package_delivery_gate(
        action_plan=action_plan,
        acceptance_summary=acceptance_summary,
        binding_readiness=managed_object_catalog.get("binding_readiness_summary") or {},
        reuse_assessment=reuse_assessment,
        checked_at=_utc_timestamp(),
    )
    package = {
        "package_type": "askme.customer_project",
        "package_version": 1,
        "package_schema": "askme.customer_project.reusable_handoff.v1",
        "exported_at": time.time(),
        "source_profile_path": str(path),
        "customer": customer,
        "site": _mapping(profile.get("site")),
        "profile": profile,
        "readiness_report": report,
        "managed_object_catalog": managed_object_catalog,
        "acceptance_summary": acceptance_summary,
        "resource_catalog_summary": managed_object_catalog.get("resource_catalog_summary") or {},
        "binding_readiness_summary": managed_object_catalog.get("binding_readiness_summary") or {},
        "reuse_assessment": reuse_assessment,
        "deployment_dependencies": reuse_assessment.get("dependencies", {}),
        "applicability_scope": customer_delivery["applicability_scope"],
        "out_of_scope": customer_delivery["out_of_scope"],
        "customer_prerequisites": customer_delivery["customer_prerequisites"],
        "scenario_acceptance_criteria": customer_delivery["scenario_acceptance_criteria"],
        "dependency_matrix": customer_delivery["dependency_matrix"],
        "managed_object_action_plan": action_plan,
        "package_delivery_gate": package_delivery_gate,
        "env_template": render_site_profile_env_template(profile),
        "delivery_claim": (
            "This package can seed another customer project after site map, devices, credentials, "
            "and acceptance tests are revalidated."
        ),
    }
    package["manifest"] = _customer_project_package_manifest(package)
    output_root.mkdir(parents=True, exist_ok=True)
    filename_parts = [
        *_customer_delivery_filename_parts(customer),
        _slug(customer.get("customer_id")),
        _slug(customer.get("project_id")),
        "package",
    ]
    filename = "-".join(filename_parts) + ".json"
    output_path = output_root / filename
    output_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "accepted": True,
        "package_path": str(output_path),
        "package": package,
    }


def export_customer_project_acceptance_dossier(
    profile_root: Path,
    identifier: str,
    *,
    output_root: Path = Path("artifacts/customer-project-acceptance-dossiers"),
    check_env: bool = True,
    field_evidence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a signed/tamper-evident customer acceptance dossier."""
    report = customer_project_acceptance_report(
        profile_root,
        identifier,
        check_env=check_env,
        field_evidence_config=field_evidence_config,
    )
    if not report.get("found"):
        return {"accepted": False, "reason": report.get("reason") or "profile_not_found"}
    customer = _mapping(report.get("customer"))
    site = _mapping(report.get("site"))
    dossier = _build_customer_project_acceptance_dossier(report)
    output_root.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_slug(customer.get('customer_id'))}-"
        f"{_slug(customer.get('project_id') or site.get('site_id'))}-acceptance-dossier.json"
    )
    output_path = output_root / filename
    output_path.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = output_path.with_suffix(".html")
    html_path.write_text(
        _render_customer_project_acceptance_dossier_html(dossier),
        encoding="utf-8",
    )
    return {
        "accepted": True,
        "dossier_path": str(output_path),
        "html_path": str(html_path),
        "dossier": dossier,
    }


def export_customer_project_proposal_bundle(
    profile_root: Path,
    template_root: Path,
    identifier: str,
    *,
    output_root: Path = Path("artifacts/customer-project-proposals"),
    check_env: bool = True,
    release_limit: int = 50,
) -> dict[str, Any]:
    """Export a customer-facing proposal bundle bound to one customer project."""
    package_result = export_customer_project_package(profile_root, identifier)
    if not package_result.get("accepted"):
        return {"accepted": False, "reason": package_result.get("reason") or "package_export_failed"}
    dossier_result = export_customer_project_acceptance_dossier(
        profile_root,
        identifier,
        check_env=check_env,
    )
    if not dossier_result.get("accepted"):
        return {"accepted": False, "reason": dossier_result.get("reason") or "dossier_export_failed"}

    package = _mapping(package_result.get("package"))
    dossier = _mapping(dossier_result.get("dossier"))
    customer = _mapping(package.get("customer"))
    site = _mapping(package.get("site"))
    release_notes = export_customer_project_template_release_notes_bundle(
        template_root,
        customer_context={
            "customer_name": customer.get("customer_name"),
            "customer_id": customer.get("customer_id"),
            "project_name": customer.get("project_name") or customer.get("project_id"),
            "project_id": customer.get("project_id"),
            "site_name": site.get("name") or site.get("site_name") or site.get("site_id"),
            "industry": customer.get("industry"),
        },
        limit=release_limit,
    )
    release_bundle = _mapping(release_notes.get("bundle"))
    release_bundle_for_payload = {key: value for key, value in release_bundle.items() if key != "html"}
    proposal_base: dict[str, Any] = {
        "proposal_type": "askme.customer_project_proposal",
        "proposal_version": 1,
        "generated_at": time.time(),
        "customer": customer,
        "site": site,
        "project_context": {
            "profile_path": package.get("source_profile_path"),
            "tenant_id": customer.get("tenant_id"),
            "delivery_namespace": customer.get("delivery_namespace"),
            "industry": customer.get("industry"),
        },
        "customer_project_package": {
            "package_path": package_result.get("package_path"),
            "manifest": _mapping(package.get("manifest")),
            "acceptance_summary": _mapping(package.get("acceptance_summary")),
            "reuse_assessment": _mapping(package.get("reuse_assessment")),
            "applicability_scope": _mapping(package.get("applicability_scope")),
            "out_of_scope": package.get("out_of_scope") if isinstance(package.get("out_of_scope"), list) else [],
            "customer_prerequisites": (
                package.get("customer_prerequisites")
                if isinstance(package.get("customer_prerequisites"), list)
                else []
            ),
            "scenario_acceptance_criteria": (
                package.get("scenario_acceptance_criteria")
                if isinstance(package.get("scenario_acceptance_criteria"), list)
                else []
            ),
            "dependency_matrix": package.get("dependency_matrix") if isinstance(package.get("dependency_matrix"), list) else [],
            "delivery_claim": package.get("delivery_claim"),
        },
        "acceptance_dossier": {
            "dossier_path": dossier_result.get("dossier_path"),
            "html_path": dossier_result.get("html_path"),
            "manifest": _mapping(dossier.get("manifest")),
            "overall_status": dossier.get("overall_status"),
            "customer_status": dossier.get("customer_status"),
            "handoff_boundary": dossier.get("handoff_boundary"),
            "launch_readiness": _mapping(dossier.get("launch_readiness")),
            "delivery_workflow": _mapping(dossier.get("delivery_workflow")),
            "gates": dossier.get("gates") if isinstance(dossier.get("gates"), list) else [],
        },
        "customer_readable_delivery": {
            "applicability_scope": _mapping(package.get("applicability_scope")),
            "out_of_scope": package.get("out_of_scope") if isinstance(package.get("out_of_scope"), list) else [],
            "customer_prerequisites": (
                package.get("customer_prerequisites")
                if isinstance(package.get("customer_prerequisites"), list)
                else []
            ),
            "scenario_acceptance_criteria": (
                package.get("scenario_acceptance_criteria")
                if isinstance(package.get("scenario_acceptance_criteria"), list)
                else []
            ),
            "dependency_matrix": package.get("dependency_matrix") if isinstance(package.get("dependency_matrix"), list) else [],
        },
        "launch_readiness": _mapping(dossier.get("launch_readiness")),
        "approved_template_release_bundle": release_bundle_for_payload,
        "proposal_insert": _mapping(release_bundle_for_payload.get("proposal_insert")),
        "delivery_boundary": (
            "This proposal bundle is customer-facing planning material. Production launch still requires "
            "project-scope approval, onsite evidence, live notification tests, and robot runtime acceptance."
        ),
    }
    proposal = {
        **proposal_base,
        "manifest": {
            "manifest_version": 1,
            "payload_sha256": _sha256_json(proposal_base),
            "tenant_id": _delivery_tenant_id(customer),
            "delivery_namespace": _delivery_namespace(customer),
            "customer_id": str(customer.get("customer_id") or ""),
            "project_id": str(customer.get("project_id") or ""),
            "site_id": str(site.get("site_id") or ""),
            "package_sha256": str(_mapping(package.get("manifest")).get("payload_sha256") or ""),
            "dossier_sha256": str(_mapping(dossier.get("manifest")).get("payload_sha256") or ""),
            "release_notes_sha256": str(
                _mapping(release_bundle_for_payload.get("manifest")).get("bundle_sha256") or ""
            ),
            "launch_readiness_status": str(
                _mapping(dossier.get("launch_readiness")).get("overall_status") or "manual_check"
            ),
            "launch_stage": str(
                _mapping(dossier.get("launch_readiness")).get("launch_stage") or "demo_or_integration_only"
            ),
            "production_ready": bool(_mapping(dossier.get("launch_readiness")).get("production_ready")),
            "proposal_scenario_acceptance_criteria_count": len(
                package.get("scenario_acceptance_criteria")
                if isinstance(package.get("scenario_acceptance_criteria"), list)
                else []
            ),
            "proposal_customer_prerequisite_count": len(
                package.get("customer_prerequisites")
                if isinstance(package.get("customer_prerequisites"), list)
                else []
            ),
        },
    }
    proposal["html"] = _render_customer_project_proposal_bundle_html(proposal)

    output_root.mkdir(parents=True, exist_ok=True)
    filename_parts = [
        *_customer_delivery_filename_parts(customer),
        _slug(customer.get("customer_id")),
        _slug(customer.get("project_id") or site.get("site_id")),
        "proposal-bundle",
    ]
    filename = "-".join(part for part in filename_parts if part) + ".json"
    output_path = output_root / filename
    output_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = output_path.with_suffix(".html")
    html_path.write_text(proposal["html"], encoding="utf-8")
    return {
        "accepted": True,
        "proposal_path": str(output_path),
        "html_path": str(html_path),
        "proposal": proposal,
        "package_path": package_result.get("package_path"),
        "dossier_path": dossier_result.get("dossier_path"),
    }


def verify_customer_project_proposal_bundle(proposal: dict[str, Any]) -> dict[str, Any]:
    """Verify the integrity metadata of a customer project proposal bundle."""
    if not isinstance(proposal, dict) or proposal.get("proposal_type") != "askme.customer_project_proposal":
        return {"valid": False, "reason": "invalid_customer_project_proposal_bundle"}
    manifest = _mapping(proposal.get("manifest"))
    expected = str(manifest.get("payload_sha256") or "")
    actual = _customer_project_proposal_bundle_payload_sha256(proposal)
    errors: list[str] = []
    if not expected:
        errors.append("manifest.payload_sha256 missing")
    elif not hmac.compare_digest(expected, actual):
        errors.append("manifest.payload_sha256 mismatch")

    customer = _mapping(proposal.get("customer"))
    site = _mapping(proposal.get("site"))
    package = _mapping(proposal.get("customer_project_package"))
    dossier = _mapping(proposal.get("acceptance_dossier"))
    release_bundle = _mapping(proposal.get("approved_template_release_bundle"))
    proposal_scope = _delivery_scope_payload_from_customer_site(customer, site)
    manifest_scope = _delivery_scope_payload_from_customer_site(manifest, {"site_id": manifest.get("site_id")})
    for field in ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id"):
        manifest_value = str(manifest_scope.get(field) or "")
        proposal_value = str(proposal_scope.get(field) or "")
        if manifest_value and proposal_value and manifest_value != proposal_value:
            errors.append(f"manifest.{field} mismatch")

    package_sha = str(_mapping(package.get("manifest")).get("payload_sha256") or "")
    dossier_sha = str(_mapping(dossier.get("manifest")).get("payload_sha256") or "")
    release_notes_sha = str(_mapping(release_bundle.get("manifest")).get("bundle_sha256") or "")
    if str(manifest.get("package_sha256") or "") != package_sha:
        errors.append("manifest.package_sha256 mismatch")
    if str(manifest.get("dossier_sha256") or "") != dossier_sha:
        errors.append("manifest.dossier_sha256 mismatch")
    if str(manifest.get("release_notes_sha256") or "") != release_notes_sha:
        errors.append("manifest.release_notes_sha256 mismatch")
    if not _mapping(proposal.get("proposal_insert")).get("safe_claims"):
        errors.append("proposal_insert.safe_claims missing")
    if not proposal.get("delivery_boundary"):
        errors.append("delivery_boundary missing")
    readable_delivery = _mapping(proposal.get("customer_readable_delivery"))
    for manifest_key, field_key in (
        ("proposal_scenario_acceptance_criteria_count", "scenario_acceptance_criteria"),
        ("proposal_customer_prerequisite_count", "customer_prerequisites"),
    ):
        values = readable_delivery.get(field_key)
        actual_count = len(values) if isinstance(values, list) else 0
        manifest_value = manifest.get(manifest_key)
        if manifest_value not in (None, "") and int(manifest_value or 0) != actual_count:
            errors.append(f"manifest.{manifest_key} mismatch")

    return {
        "valid": not errors,
        "reason": "ok" if not errors else "integrity_errors",
        "errors": errors,
        "manifest": manifest,
        "proposal_scope": proposal_scope,
        "payload_sha256": actual,
        "package_sha256": package_sha,
        "dossier_sha256": dossier_sha,
        "release_notes_sha256": release_notes_sha,
    }


def verify_customer_project_acceptance_dossier(dossier: dict[str, Any]) -> dict[str, Any]:
    """Verify the manifest for a customer acceptance dossier."""
    if not isinstance(dossier, dict) or dossier.get("dossier_type") != "askme.customer_project_acceptance":
        return {"valid": False, "reason": "invalid_customer_acceptance_dossier"}
    manifest = _mapping(dossier.get("manifest"))
    expected = str(manifest.get("payload_sha256") or "")
    actual = _customer_project_acceptance_dossier_payload_sha256(dossier)
    errors: list[str] = []
    if not expected:
        errors.append("manifest.payload_sha256 missing")
    elif not hmac.compare_digest(expected, actual):
        errors.append("manifest.payload_sha256 mismatch")
    signature_expected = str(manifest.get("payload_signature") or "")
    signature_secret = _clean_secret(os.getenv("ASKME_CUSTOMER_ACCEPTANCE_DOSSIER_HMAC_SECRET"))
    signature_checked = False
    if signature_expected and signature_secret:
        signature_checked = True
        actual_signature = hmac.new(
            signature_secret.encode("utf-8"),
            actual.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature_expected, actual_signature):
            errors.append("manifest.payload_signature mismatch")
    elif signature_expected and not signature_secret:
        errors.append("signature present but verification secret is not configured")
    return {
        "valid": not errors,
        "reason": "ok" if not errors else "integrity_errors",
        "errors": errors,
        "manifest": manifest,
        "payload_sha256": actual,
        "signature_checked": signature_checked,
        "signature_key_id": str(manifest.get("signature_key_id") or ""),
    }


def _render_customer_project_acceptance_dossier_html(dossier: dict[str, Any]) -> str:
    customer = _mapping(dossier.get("customer"))
    site = _mapping(dossier.get("site"))
    manifest = _mapping(dossier.get("manifest"))
    readiness = _mapping(dossier.get("field_readiness"))
    launch_readiness = _mapping(dossier.get("launch_readiness"))
    delivery_brief = _mapping(readiness.get("delivery_brief"))
    delivery_workflow = _mapping(dossier.get("delivery_workflow"))
    gates = dossier.get("gates") if isinstance(dossier.get("gates"), list) else []
    launch_gates = (
        launch_readiness.get("gates")
        if isinstance(launch_readiness.get("gates"), list)
        else []
    )
    workflow_steps = (
        delivery_workflow.get("steps")
        if isinstance(delivery_workflow.get("steps"), list)
        else []
    )
    evidence = dossier.get("evidence_inventory") if isinstance(dossier.get("evidence_inventory"), list) else []
    env_missing = dossier.get("env_missing") if isinstance(dossier.get("env_missing"), list) else []
    warnings = dossier.get("warnings") if isinstance(dossier.get("warnings"), list) else []
    errors = dossier.get("errors") if isinstance(dossier.get("errors"), list) else []
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    next_actions = readiness.get("next_actions") if isinstance(readiness.get("next_actions"), list) else []
    workflow_rows = "\n".join(_dossier_workflow_row(_mapping(item)) for item in workflow_steps)
    if not workflow_rows:
        workflow_rows = "<tr><td colspan=\"4\">No delivery workflow was recorded.</td></tr>"
    gate_rows = "\n".join(_dossier_gate_row(_mapping(item)) for item in gates)
    launch_gate_rows = "\n".join(_dossier_gate_row(_mapping(item)) for item in launch_gates)
    if not launch_gate_rows:
        launch_gate_rows = "<tr><td colspan=\"4\">No launch readiness gates recorded.</td></tr>"
    evidence_rows = "\n".join(_dossier_evidence_row(_mapping(item)) for item in evidence)
    issue_rows = "\n".join(
        f"<li>{_h(item)}</li>"
        for item in [*errors, *blockers, *warnings[:10]]
    ) or "<li>No blocking evidence recorded in this dossier.</li>"
    next_action_rows = "\n".join(f"<li>{_h(item)}</li>" for item in next_actions) or "<li>No next action recorded.</li>"
    missing_env_rows = "\n".join(
        f"<li><strong>{_h(_mapping(item).get('env_name'))}</strong> - {_h(_mapping(item).get('purpose'))}</li>"
        for item in env_missing
    ) or "<li>No missing environment variable recorded.</li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer Acceptance Dossier - {_h(customer.get('project_id') or site.get('site_id'))}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #13211c;
      --muted: #60706b;
      --line: #dfe8e4;
      --soft: #f4f8f6;
      --ok: #0f8a57;
      --warn: #a76606;
      --bad: #b3261e;
      --accent: #0c6b4f;
    }}
    body {{
      margin: 0;
      background: #eef5f2;
      color: var(--ink);
      font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      padding: 28px;
      background: linear-gradient(135deg, #ffffff, #e7f3ee);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.2; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    section {{
      margin-top: 18px;
      padding: 22px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .meta, .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .card, .metric {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--soft);
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-weight: 700; word-break: break-word; }}
    .status {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-weight: 700;
      border: 1px solid currentColor;
    }}
    .ok {{ color: var(--ok); }}
    .manual_check, .ready_for_lab, .warn {{ color: var(--warn); }}
    .blocked, .err {{ color: var(--bad); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; vertical-align: top; padding: 10px 8px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: 12px; }}
    code {{ word-break: break-all; }}
    .boundary {{
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: #eef8f4;
      border-radius: 10px;
    }}
    @media print {{
      body {{ background: #fff; }}
      main {{ padding: 0; }}
      header, section {{ break-inside: avoid; border-radius: 0; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="label">AskMe Customer Acceptance Dossier</div>
      <h1>{_h(customer.get('customer_name') or customer.get('customer_id') or 'Customer Project')}</h1>
      <p>{_h(customer.get('project_name') or customer.get('project_id') or site.get('name') or '')}</p>
    </div>
    <div>
      <div class="label">Overall Status</div>
      <div class="status {_status_class(dossier.get('overall_status'))}">{_h(dossier.get('overall_status'))}</div>
    </div>
  </header>

  <section>
    <h2>交付结论</h2>
    <div class="boundary">{_h(dossier.get('customer_status') or delivery_brief.get('customer_status') or '')}</div>
    <p>{_h(dossier.get('handoff_boundary') or '')}</p>
    <p>{_h(dossier.get('release_claim') or delivery_brief.get('release_claim') or '')}</p>
  </section>

  <section>
    <h2>上线准入</h2>
    <div class="meta">
      {_metric('Launch Stage', launch_readiness.get('launch_stage'))}
      {_metric('Readiness', launch_readiness.get('overall_status'))}
      {_metric('Production Ready', launch_readiness.get('production_ready'))}
      {_metric('Next Step', launch_readiness.get('next_step'))}
    </div>
    <p>{_h(launch_readiness.get('customer_status'))}</p>
    <p>{_h(launch_readiness.get('release_claim'))}</p>
    <table>
      <thead><tr><th>Gate</th><th>Status</th><th>Evidence</th><th>Next Step</th></tr></thead>
      <tbody>{launch_gate_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>项目摘要</h2>
    <div class="meta">
      {_metric('Customer ID', customer.get('customer_id'))}
      {_metric('Project ID', customer.get('project_id'))}
      {_metric('Site ID', site.get('site_id'))}
      {_metric('Industry', customer.get('industry'))}
      {_metric('Field Readiness', readiness.get('status'))}
      {_metric('Manifest SHA-256', str(manifest.get('payload_sha256') or '')[:24])}
    </div>
  </section>

  <section>
    <h2>Delivery Workflow</h2>
    <table>
      <thead><tr><th>Step</th><th>Status</th><th>Evidence</th><th>Next Step</th></tr></thead>
      <tbody>{workflow_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>验收门禁</h2>
    <table>
      <thead><tr><th>Gate</th><th>Status</th><th>Evidence</th><th>Next Step</th></tr></thead>
      <tbody>{gate_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>证据文件清单</h2>
    <table>
      <thead><tr><th>Path</th><th>Status</th><th>Size</th><th>SHA-256</th></tr></thead>
      <tbody>{evidence_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>阻塞项和风险</h2>
    <ul>{issue_rows}</ul>
  </section>

  <section>
    <h2>缺失部署配置</h2>
    <ul>{missing_env_rows}</ul>
  </section>

  <section>
    <h2>下一步</h2>
    <ul>{next_action_rows}</ul>
  </section>

  <section>
    <h2>Manifest</h2>
    <div class="cards">
      {_metric('Evidence Count', manifest.get('evidence_count'))}
      {_metric('Missing Evidence', manifest.get('evidence_missing_count'))}
      {_metric('Signature', manifest.get('signature_alg') or 'unsigned')}
      {_metric('Signature Key', manifest.get('signature_key_id') or '-')}
    </div>
  </section>
</main>
</body>
</html>
"""


def _dossier_gate_row(gate: dict[str, Any]) -> str:
    status = str(gate.get("status") or "unknown")
    return (
        "<tr>"
        f"<td><strong>{_h(gate.get('label') or gate.get('gate_id'))}</strong></td>"
        f"<td><span class=\"status {_status_class(status)}\">{_h(status)}</span></td>"
        f"<td>{_h(gate.get('evidence'))}</td>"
        f"<td>{_h(gate.get('next_step'))}</td>"
        "</tr>"
    )


def _dossier_workflow_row(step: dict[str, Any]) -> str:
    status = str(step.get("status") or "unknown")
    return (
        "<tr>"
        f"<td><strong>{_h(step.get('label') or step.get('step_id'))}</strong></td>"
        f"<td><span class=\"status {_status_class(status)}\">{_h(status)}</span></td>"
        f"<td>{_h(step.get('evidence'))}</td>"
        f"<td>{_h(step.get('next_step'))}</td>"
        "</tr>"
    )


def _dossier_evidence_row(item: dict[str, Any]) -> str:
    status = "hashed" if item.get("exists") and item.get("sha256") else "missing"
    return (
        "<tr>"
        f"<td><code>{_h(item.get('path'))}</code></td>"
        f"<td><span class=\"status {'ok' if status == 'hashed' else 'blocked'}\">{status}</span></td>"
        f"<td>{_h(item.get('size_bytes') or 0)}</td>"
        f"<td><code>{_h(item.get('sha256'))}</code></td>"
        "</tr>"
    )


def _render_customer_project_proposal_bundle_html(proposal: dict[str, Any]) -> str:
    customer = _mapping(proposal.get("customer"))
    site = _mapping(proposal.get("site"))
    package = _mapping(proposal.get("customer_project_package"))
    dossier = _mapping(proposal.get("acceptance_dossier"))
    launch_readiness = _mapping(proposal.get("launch_readiness"))
    proposal_insert = _mapping(proposal.get("proposal_insert"))
    readable_delivery = _mapping(proposal.get("customer_readable_delivery"))
    applicability = _mapping(readable_delivery.get("applicability_scope"))
    prerequisites = (
        readable_delivery.get("customer_prerequisites")
        if isinstance(readable_delivery.get("customer_prerequisites"), list)
        else []
    )
    scenario_criteria = (
        readable_delivery.get("scenario_acceptance_criteria")
        if isinstance(readable_delivery.get("scenario_acceptance_criteria"), list)
        else []
    )
    dependency_matrix = (
        readable_delivery.get("dependency_matrix")
        if isinstance(readable_delivery.get("dependency_matrix"), list)
        else []
    )
    release_bundle = _mapping(proposal.get("approved_template_release_bundle"))
    release_notes = release_bundle.get("release_notes") if isinstance(release_bundle.get("release_notes"), list) else []
    gates = dossier.get("gates") if isinstance(dossier.get("gates"), list) else []
    launch_gates = launch_readiness.get("gates") if isinstance(launch_readiness.get("gates"), list) else []
    safe_claims = proposal_insert.get("safe_claims") if isinstance(proposal_insert.get("safe_claims"), list) else []
    boundaries = (
        proposal_insert.get("delivery_boundaries")
        if isinstance(proposal_insert.get("delivery_boundaries"), list)
        else []
    )
    note_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('template_id'))}</td>"
        f"<td>{_h(_mapping(item).get('version'))}</td>"
        f"<td>{_h(_mapping(item).get('product_status'))}</td>"
        f"<td>{_h(_mapping(item).get('customer_status') or _mapping(item).get('customer_claim'))}</td>"
        "</tr>"
        for item in release_notes
    ) or "<tr><td colspan=\"4\">No approved published template releases are available.</td></tr>"
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(gate).get('label') or _mapping(gate).get('gate_id'))}</td>"
        f"<td>{_h(_mapping(gate).get('status'))}</td>"
        f"<td>{_h(_mapping(gate).get('next_step'))}</td>"
        "</tr>"
        for gate in gates
    ) or "<tr><td colspan=\"3\">No acceptance gates recorded.</td></tr>"
    launch_gate_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(gate).get('label') or _mapping(gate).get('gate_id'))}</td>"
        f"<td>{_h(_mapping(gate).get('status'))}</td>"
        f"<td>{_h(_mapping(gate).get('next_step'))}</td>"
        "</tr>"
        for gate in launch_gates
        if isinstance(gate, dict)
    ) or "<tr><td colspan=\"3\">No launch readiness gates recorded.</td></tr>"
    claim_rows = "".join(f"<li>{_h(item)}</li>" for item in safe_claims)
    boundary_rows = "".join(f"<li>{_h(item)}</li>" for item in boundaries)
    prerequisite_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('label') or _mapping(item).get('prerequisite_id'))}</td>"
        f"<td>{_h(_mapping(item).get('status'))}</td>"
        f"<td>{_h(_mapping(item).get('owner'))}</td>"
        f"<td>{_h(_mapping(item).get('next_step'))}</td>"
        "</tr>"
        for item in prerequisites
        if isinstance(item, dict)
    ) or "<tr><td colspan=\"4\">No customer prerequisites recorded.</td></tr>"
    scenario_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('scenario_id'))}</td>"
        f"<td>{_h(', '.join(_string_list(_mapping(item).get('managed_object_labels'))))}</td>"
        f"<td>{_h(', '.join(_string_list(_mapping(item).get('required_evidence'))))}</td>"
        f"<td>{_h(_mapping(item).get('pass_condition'))}</td>"
        "</tr>"
        for item in scenario_criteria
        if isinstance(item, dict)
    ) or "<tr><td colspan=\"4\">No scenario acceptance criteria recorded.</td></tr>"
    package_manifest = _mapping(package.get("manifest"))
    dossier_manifest = _mapping(dossier.get("manifest"))
    proposal_manifest = _mapping(proposal.get("manifest"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AskMe Customer Project Proposal Bundle</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #17211f; margin: 32px; }}
    header {{ border-bottom: 1px solid #d9e5df; padding-bottom: 18px; margin-bottom: 22px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .muted {{ color: #64746e; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d9e5df; border-radius: 10px; padding: 14px; }}
    .metric b {{ display: block; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #e7efeb; text-align: left; padding: 9px 8px; vertical-align: top; }}
    th {{ color: #42534d; font-size: 12px; text-transform: uppercase; }}
    .boundary {{ margin-top: 18px; padding: 14px; background: #f4faf7; border: 1px solid #cfe4da; border-radius: 10px; }}
    code {{ background: #eef4f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>AskMe Customer Project Proposal Bundle</h1>
    <div class="muted">{_h(customer.get("customer_name") or customer.get("customer_id"))} / {_h(customer.get("project_name") or customer.get("project_id"))}</div>
    <div class="muted">{_h(site.get("name") or site.get("site_name") or site.get("site_id"))}</div>
  </header>
  <section class="metrics">
    <div class="metric"><b>{_h(package_manifest.get("managed_object_count"))}</b><span>managed objects</span></div>
    <div class="metric"><b>{_h(package_manifest.get("acceptance_overall_status"))}</b><span>package acceptance</span></div>
    <div class="metric"><b>{_h(dossier.get("overall_status"))}</b><span>dossier status</span></div>
    <div class="metric"><b>{_h(launch_readiness.get("launch_stage"))}</b><span>launch stage</span></div>
  </section>
  <section class="boundary">
    <strong>上线准入</strong>
    <p>{_h(launch_readiness.get("customer_status"))}</p>
    <p>{_h(launch_readiness.get("release_claim"))}</p>
    <table>
      <thead><tr><th>Gate</th><th>Status</th><th>Next step</th></tr></thead>
      <tbody>{launch_gate_rows}</tbody>
    </table>
  </section>
  <section class="boundary">
    <strong>{_h(proposal_insert.get("section_title") or "Approved reusable capabilities")}</strong>
    <p>{_h(proposal_insert.get("customer_message"))}</p>
    <ul>{claim_rows}</ul>
  </section>
  <h2>Customer Delivery Scope</h2>
  <section class="metrics">
    <div class="metric"><b>{_h(', '.join(_string_list(applicability.get("industries"))) or '-')}</b><span>industries</span></div>
    <div class="metric"><b>{_h(len(_string_list(applicability.get("scenarios"))))}</b><span>scenarios</span></div>
    <div class="metric"><b>{_h(len(dependency_matrix))}</b><span>dependencies</span></div>
    <div class="metric"><b>{_h(len(prerequisites))}</b><span>customer prerequisites</span></div>
  </section>
  <table>
    <thead><tr><th>Prerequisite</th><th>Status</th><th>Owner</th><th>Next step</th></tr></thead>
    <tbody>{prerequisite_rows}</tbody>
  </table>
  <table>
    <thead><tr><th>Scenario</th><th>Managed objects</th><th>Evidence</th><th>Pass condition</th></tr></thead>
    <tbody>{scenario_rows}</tbody>
  </table>
  <h2>Approved Template Release Notes</h2>
  <table>
    <thead><tr><th>Template</th><th>Version</th><th>Status</th><th>Customer status</th></tr></thead>
    <tbody>{note_rows}</tbody>
  </table>
  <h2>Acceptance Gates</h2>
  <table>
    <thead><tr><th>Gate</th><th>Status</th><th>Next step</th></tr></thead>
    <tbody>{gate_rows}</tbody>
  </table>
  <section class="boundary">
    <strong>Delivery boundary</strong>
    <p>{_h(proposal.get("delivery_boundary"))}</p>
    <ul>{boundary_rows}</ul>
    <p class="muted">Proposal SHA-256: <code>{_h(proposal_manifest.get("payload_sha256"))}</code></p>
    <p class="muted">Package SHA-256: <code>{_h(package_manifest.get("payload_sha256"))}</code></p>
    <p class="muted">Dossier SHA-256: <code>{_h(dossier_manifest.get("payload_sha256"))}</code></p>
  </section>
</body>
</html>
"""


def _metric(label: str, value: Any) -> str:
    return (
        "<div class=\"metric\">"
        f"<div class=\"label\">{_h(label)}</div>"
        f"<div class=\"value\">{_h(value if value not in (None, '') else '-')}</div>"
        "</div>"
    )


def _status_class(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"ready", "production_ready"}:
        return "ok"
    if text in {"manual_check", "ready_for_lab", "ready_for_onsite_acceptance"}:
        return "manual_check"
    if text in {"blocked", "failed", "missing", "invalid"}:
        return "blocked"
    return "warn"


def _h(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def import_customer_project_package(
    profile_root: Path,
    package: dict[str, Any],
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Import a previously exported customer project handoff package."""
    verification = verify_customer_project_package(package)
    if not verification.get("valid"):
        return {
            "accepted": False,
            "reason": "package_integrity_check_failed",
            "verification": verification,
        }
    profile = package.get("profile")
    if not isinstance(profile, dict):
        return {"accepted": False, "reason": "package_missing_profile"}
    profile = _normalize_customer_project_profile(profile)
    diff = diff_customer_project_package(profile_root, package)
    delivery_gate = _mapping(verification.get("delivery_gate"))
    import_allowed = bool(delivery_gate.get("import_allowed", True))
    import_gate_result = _customer_project_package_import_gate_result(delivery_gate)
    target = _customer_profile_target(profile_root, profile)
    implementation_handoff = _customer_project_implementation_handoff(
        profile,
        template_id=str(_mapping(profile.get("template")).get("template_id") or ""),
        profile_path=target,
    )
    if dry_run:
        return {
            "accepted": True,
            "dry_run": True,
            "verification": verification,
            "diff": diff,
            "delivery_gate": delivery_gate,
            "import_gate_result": import_gate_result,
            "would_write": bool(diff.get("change_type") in {"create", "replace"} and import_allowed),
            "implementation_handoff": implementation_handoff,
        }
    if not import_allowed:
        return {
            "accepted": False,
            "reason": "package_delivery_gate_blocked",
            "verification": verification,
            "diff": diff,
            "delivery_gate": delivery_gate,
            "import_gate_result": import_gate_result,
            "implementation_handoff": implementation_handoff,
        }
    result = upsert_customer_project_profile(
        profile_root,
        profile,
        overwrite=overwrite,
        operator_id=operator_id,
        reason=reason or "Import customer project package.",
        revision_action="package_import",
    )
    result["verification"] = verification
    result["diff"] = diff
    result["delivery_gate"] = delivery_gate
    result["import_gate_result"] = import_gate_result
    result["implementation_handoff"] = implementation_handoff
    return result


def verify_customer_project_package(package: dict[str, Any]) -> dict[str, Any]:
    """Verify the integrity metadata of a customer project package."""
    if not isinstance(package, dict) or package.get("package_type") != "askme.customer_project":
        return {"valid": False, "reason": "invalid_customer_project_package"}
    manifest = package.get("manifest") if isinstance(package.get("manifest"), dict) else {}
    expected = str(manifest.get("payload_sha256") or "")
    actual = _customer_project_package_payload_sha256(package)
    errors: list[str] = []
    if not expected:
        errors.append("manifest.payload_sha256 missing")
    elif not hmac.compare_digest(expected, actual):
        errors.append("manifest.payload_sha256 mismatch")

    profile = package.get("profile")
    profile_expected = str(manifest.get("profile_sha256") or "")
    profile_actual = _sha256_json(profile) if isinstance(profile, dict) else ""
    if not isinstance(profile, dict):
        errors.append("profile missing")
    elif profile_expected and not hmac.compare_digest(profile_expected, profile_actual):
        errors.append("manifest.profile_sha256 mismatch")
    profile_scope = _delivery_scope_payload(profile) if isinstance(profile, dict) else {}
    package_scope = _delivery_scope_payload_from_customer_site(
        _mapping(package.get("customer")),
        _mapping(package.get("site")),
    )
    manifest_scope = _delivery_scope_payload_from_customer_site(manifest, {"site_id": manifest.get("site_id")})
    for field in ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id"):
        manifest_value = str(manifest_scope.get(field) or "")
        package_value = str(package_scope.get(field) or "")
        profile_value = str(profile_scope.get(field) or "")
        if manifest_value and package_value and manifest_value != package_value:
            errors.append(f"manifest.{field} mismatch")
        if manifest_value and profile_value and manifest_value != profile_value:
            errors.append(f"manifest.{field} profile mismatch")
        if package_value and profile_value and package_value != profile_value:
            errors.append(f"package.{field} profile mismatch")
    acceptance = _mapping(package.get("acceptance_summary"))
    manifest_acceptance = str(manifest.get("acceptance_overall_status") or "")
    package_acceptance = str(acceptance.get("overall_status") or "")
    if manifest_acceptance and package_acceptance and manifest_acceptance != package_acceptance:
        errors.append("manifest.acceptance_overall_status mismatch")
    binding_readiness = _mapping(package.get("binding_readiness_summary"))
    manifest_binding_status = str(manifest.get("resource_binding_overall_status") or "")
    package_binding_status = str(binding_readiness.get("overall_status") or "")
    if manifest_binding_status and package_binding_status and manifest_binding_status != package_binding_status:
        errors.append("manifest.resource_binding_overall_status mismatch")
    for manifest_key, binding_key in (
        ("resource_binding_ready_object_count", "ready_object_count"),
        ("resource_binding_manual_check_object_count", "manual_check_object_count"),
        ("resource_binding_blocked_object_count", "blocked_object_count"),
        ("resource_binding_unregistered_resource_count", "unregistered_resource_count"),
    ):
        manifest_value = manifest.get(manifest_key)
        binding_value = binding_readiness.get(binding_key)
        if manifest_value not in (None, "") and binding_value not in (None, ""):
            if int(manifest_value or 0) != int(binding_value or 0):
                errors.append(f"manifest.{manifest_key} mismatch")
    resource_catalog = _mapping(package.get("resource_catalog_summary"))
    manifest_resource_count = manifest.get("delivery_resource_count")
    package_resource_count = resource_catalog.get("resource_count")
    if manifest_resource_count not in (None, "") and package_resource_count not in (None, ""):
        if int(manifest_resource_count or 0) != int(package_resource_count or 0):
            errors.append("manifest.delivery_resource_count mismatch")
    applicability = _mapping(package.get("applicability_scope"))
    out_of_scope = package.get("out_of_scope") if isinstance(package.get("out_of_scope"), list) else []
    prerequisites = (
        package.get("customer_prerequisites")
        if isinstance(package.get("customer_prerequisites"), list)
        else []
    )
    scenario_criteria = (
        package.get("scenario_acceptance_criteria")
        if isinstance(package.get("scenario_acceptance_criteria"), list)
        else []
    )
    dependency_matrix = (
        package.get("dependency_matrix")
        if isinstance(package.get("dependency_matrix"), list)
        else []
    )
    for manifest_key, package_count in (
        ("applicability_scenario_count", len(_string_list(applicability.get("scenarios")))),
        ("applicability_managed_object_type_count", len(_string_list(applicability.get("managed_object_types")))),
        ("out_of_scope_count", len(out_of_scope)),
        ("customer_prerequisite_count", len(prerequisites)),
        (
            "required_customer_prerequisite_count",
            len([item for item in prerequisites if isinstance(item, dict) and item.get("required") is True]),
        ),
        ("scenario_acceptance_criteria_count", len(scenario_criteria)),
        ("dependency_matrix_count", len(dependency_matrix)),
    ):
        manifest_value = manifest.get(manifest_key)
        if manifest_value not in (None, "") and int(manifest_value or 0) != int(package_count or 0):
            errors.append(f"manifest.{manifest_key} mismatch")
    reuse = _mapping(package.get("reuse_assessment"))
    reuse_status = str(reuse.get("status") or "")
    manifest_reuse_status = str(manifest.get("reuse_status") or "")
    if manifest_reuse_status and reuse_status and manifest_reuse_status != reuse_status:
        errors.append("manifest.reuse_status mismatch")
    for manifest_key, reuse_key in (
        ("reuse_blocker_count", "blocker_count"),
        ("reuse_manual_check_count", "manual_check_count"),
    ):
        manifest_value = manifest.get(manifest_key)
        reuse_value = reuse.get(reuse_key)
        if manifest_value not in (None, "") and reuse_value not in (None, ""):
            if int(manifest_value or 0) != int(reuse_value or 0):
                errors.append(f"manifest.{manifest_key} mismatch")

    action_plan = _customer_project_package_action_plan(_mapping(package.get("managed_object_catalog")))
    delivery_gate = _customer_project_package_delivery_gate(
        action_plan=action_plan,
        acceptance_summary=acceptance,
        binding_readiness=binding_readiness,
        reuse_assessment=reuse,
        checked_at=str(_mapping(package.get("package_delivery_gate")).get("delivery_gate_checked_at") or ""),
    )
    claimed_action_plan = _mapping(package.get("managed_object_action_plan"))
    claimed_gate = _mapping(package.get("package_delivery_gate"))
    if claimed_action_plan:
        for key in (
            "overall_status",
            "action_count",
            "blocked_action_count",
            "manual_check_action_count",
            "delivery_gate_source_version",
        ):
            if str(claimed_action_plan.get(key) or "") != str(action_plan.get(key) or ""):
                errors.append(f"managed_object_action_plan.{key} mismatch")
    if claimed_gate:
        for key in (
            "delivery_gate_status",
            "export_allowed",
            "import_allowed",
            "customer_handoff_ready",
            "action_count",
            "blocked_action_count",
            "manual_check_action_count",
            "delivery_gate_source_version",
        ):
            if str(claimed_gate.get(key) or "") != str(delivery_gate.get(key) or ""):
                errors.append(f"package_delivery_gate.{key} mismatch")
    for manifest_key, gate_key in (
        ("package_delivery_gate_status", "delivery_gate_status"),
        ("package_delivery_export_allowed", "export_allowed"),
        ("package_delivery_import_allowed", "import_allowed"),
        ("package_delivery_customer_handoff_ready", "customer_handoff_ready"),
        ("package_delivery_action_count", "action_count"),
        ("package_delivery_blocked_action_count", "blocked_action_count"),
        ("package_delivery_manual_check_action_count", "manual_check_action_count"),
        ("package_delivery_source_version", "delivery_gate_source_version"),
    ):
        manifest_value = manifest.get(manifest_key)
        gate_value = delivery_gate.get(gate_key)
        if manifest_value not in (None, "") and str(manifest_value) != str(gate_value):
            errors.append(f"manifest.{manifest_key} mismatch")

    signature_expected = str(manifest.get("payload_signature") or "")
    signature_key_id = str(manifest.get("signature_key_id") or "")
    signature_secret = _clean_secret(os.getenv("ASKME_CUSTOMER_PROJECT_PACKAGE_HMAC_SECRET"))
    signature_checked = False
    if signature_expected and signature_secret:
        signature_checked = True
        actual_signature = hmac.new(
            signature_secret.encode("utf-8"),
            actual.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature_expected, actual_signature):
            errors.append("manifest.payload_signature mismatch")
    elif signature_expected and not signature_secret:
        errors.append("signature present but verification secret is not configured")

    return {
        "valid": not errors,
        "reason": "ok" if not errors else "integrity_errors",
        "errors": errors,
        "manifest": manifest,
        "package_scope": package_scope,
        "profile_scope": profile_scope,
        "payload_sha256": actual,
        "profile_sha256": profile_actual,
        "managed_object_action_plan": action_plan,
        "delivery_gate": delivery_gate,
        "delivery_gate_status": delivery_gate.get("delivery_gate_status"),
        "customer_handoff_ready": bool(delivery_gate.get("customer_handoff_ready")),
        "import_allowed": bool(delivery_gate.get("import_allowed", True)),
        "export_allowed": bool(delivery_gate.get("export_allowed", True)),
        "signature_checked": signature_checked,
        "signature_key_id": signature_key_id,
    }


def diff_customer_project_package(profile_root: Path, package: dict[str, Any]) -> dict[str, Any]:
    """Return a dry-run diff for importing a customer project package."""
    profile = package.get("profile") if isinstance(package, dict) else None
    if not isinstance(profile, dict):
        return {"change_type": "invalid", "reason": "package_missing_profile"}
    profile = _normalize_customer_project_profile(profile)
    target = _customer_profile_target(profile_root, profile)
    incoming_scope = _delivery_scope_payload(profile)
    collisions = _customer_project_collision_candidates(profile_root, profile)
    incoming_report = validate_field_site_profile(profile)
    incoming_catalog = managed_object_catalog_from_site_profile(profile)
    incoming_acceptance = _customer_project_package_acceptance_summary(incoming_catalog)
    incoming_binding_readiness = _mapping(incoming_catalog.get("binding_readiness_summary"))
    incoming_reuse = _customer_project_package_reuse_assessment(
        profile=profile,
        report=incoming_report,
        managed_object_catalog=incoming_catalog,
        acceptance_summary=incoming_acceptance,
        env_references=site_profile_env_references(profile),
    )
    incoming_action_plan = _customer_project_package_action_plan(incoming_catalog)
    incoming_delivery_gate = _customer_project_package_delivery_gate(
        action_plan=incoming_action_plan,
        acceptance_summary=incoming_acceptance,
        binding_readiness=incoming_binding_readiness,
        reuse_assessment=incoming_reuse,
    )
    if not target.exists():
        return {
            "change_type": "create",
            "target_profile_path": str(target),
            "incoming_delivery_scope": incoming_scope,
            "collision_candidates": collisions,
            "incoming_valid": incoming_report.get("status") == "passed",
            "incoming_acceptance_summary": incoming_acceptance,
            "incoming_binding_readiness_summary": incoming_binding_readiness,
            "incoming_reuse_assessment": incoming_reuse,
            "incoming_managed_object_action_plan": incoming_action_plan,
            "incoming_delivery_gate": incoming_delivery_gate,
            "field_changes": [],
        }
    current = load_field_site_profile(target)
    current_scope = _delivery_scope_payload(current)
    field_changes = _customer_project_profile_diff(current, profile)
    current_acceptance = _customer_project_package_acceptance_summary(
        managed_object_catalog_from_site_profile(current)
    )
    current_catalog = managed_object_catalog_from_site_profile(current)
    current_binding_readiness = _mapping(current_catalog.get("binding_readiness_summary"))
    current_reuse = _customer_project_package_reuse_assessment(
        profile=current,
        report=validate_field_site_profile(current),
        managed_object_catalog=current_catalog,
        acceptance_summary=current_acceptance,
        env_references=site_profile_env_references(current),
    )
    current_action_plan = _customer_project_package_action_plan(current_catalog)
    current_delivery_gate = _customer_project_package_delivery_gate(
        action_plan=current_action_plan,
        acceptance_summary=current_acceptance,
        binding_readiness=current_binding_readiness,
        reuse_assessment=current_reuse,
    )
    return {
        "change_type": "noop" if not field_changes else "replace",
        "target_profile_path": str(target),
        "incoming_delivery_scope": incoming_scope,
        "current_delivery_scope": current_scope,
        "collision_candidates": collisions,
        "incoming_valid": incoming_report.get("status") == "passed",
        "incoming_acceptance_summary": incoming_acceptance,
        "current_acceptance_summary": current_acceptance,
        "incoming_binding_readiness_summary": incoming_binding_readiness,
        "current_binding_readiness_summary": current_binding_readiness,
        "incoming_reuse_assessment": incoming_reuse,
        "current_reuse_assessment": current_reuse,
        "incoming_managed_object_action_plan": incoming_action_plan,
        "current_managed_object_action_plan": current_action_plan,
        "incoming_delivery_gate": incoming_delivery_gate,
        "current_delivery_gate": current_delivery_gate,
        "field_changes": field_changes,
        "current_profile_sha256": _sha256_json(current),
        "incoming_profile_sha256": _sha256_json(profile),
    }


def _customer_project_package_reuse_assessment(
    *,
    profile: dict[str, Any],
    report: dict[str, Any],
    managed_object_catalog: dict[str, Any],
    acceptance_summary: dict[str, Any],
    env_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether a handoff package can be reused for another customer project."""
    objects = (
        managed_object_catalog.get("objects")
        if isinstance(managed_object_catalog.get("objects"), list)
        else []
    )
    missing_env = [
        item for item in env_references if item.get("required") and not item.get("configured")
    ]
    dependencies = _customer_project_reuse_dependencies(profile, objects, env_references)
    blockers: list[str] = []
    manual_checks: list[str] = []
    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    binding_readiness = _mapping(managed_object_catalog.get("binding_readiness_summary"))
    if errors:
        blockers.append(f"Site profile has {len(errors)} validation error(s).")
    if str(acceptance_summary.get("overall_status") or "") == "blocked":
        blockers.append("Managed-object acceptance evidence is blocked.")
    if str(binding_readiness.get("overall_status") or "") == "blocked":
        blockers.append("Managed-object resource bindings are blocked.")
    if not objects:
        blockers.append("Managed-object catalog is empty.")
    if missing_env:
        manual_checks.append(f"{len(missing_env)} live credential or device secret value(s) must be configured onsite.")
    if warnings:
        manual_checks.append(f"{len(warnings)} site warning(s) require delivery review.")
    if str(acceptance_summary.get("overall_status") or "") == "manual_check":
        manual_checks.append("Some managed-object acceptance references require manual review.")
    if str(binding_readiness.get("overall_status") or "") == "manual_check":
        manual_checks.append("Some managed-object resource bindings need catalog review.")
    if blockers:
        status = "blocked"
    elif manual_checks:
        status = "manual_check"
    else:
        status = "ready"
    return {
        "status": status,
        "customer_status": {
            "ready": "Package can seed a new customer project after onsite evidence is refreshed.",
            "manual_check": "Package is reusable, but delivery must rebind live credentials and review acceptance evidence.",
            "blocked": "Package should not be reused until profile or acceptance blockers are fixed.",
        }[status],
        "blocker_count": len(blockers),
        "manual_check_count": len(manual_checks),
        "blockers": blockers,
        "manual_checks": manual_checks,
        "dependencies": dependencies,
        "next_step": {
            "ready": "Import the package, then run onsite smoke and acceptance checks for the target customer.",
            "manual_check": "Resolve manual checks after import before customer signoff.",
            "blocked": "Fix blockers in the source project before using this as a reusable template.",
        }[status],
    }


def _utc_timestamp(value: float | None = None) -> str:
    current = time.time() if value is None else value
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current))


def _customer_project_package_action_plan(managed_object_catalog: dict[str, Any]) -> dict[str, Any]:
    objects = (
        managed_object_catalog.get("objects")
        if isinstance(managed_object_catalog.get("objects"), list)
        else []
    )
    actions: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        common = {
            "object_id": object_id,
            "display_name": str(item.get("display_name") or object_id),
            "object_type": str(item.get("category") or ""),
            "category": str(item.get("category") or ""),
        }
        resource_status = _mapping(item.get("resource_binding_status"))
        for check in resource_status.get("checks", []) if isinstance(resource_status.get("checks"), list) else []:
            check = _mapping(check)
            status = str(check.get("status") or "")
            if status == "linked":
                continue
            resource_type = str(check.get("resource_type") or "")
            resource_id = str(check.get("resource_id") or "")
            if status == "missing":
                actions.append({
                    **common,
                    "action": "bind_required_resource",
                    "reason_code": "resource_binding_missing",
                    "reason_label": f"{resource_type} binding is missing.",
                    "severity": "blocked",
                    "owner": "delivery_owner",
                    "target": resource_type,
                    "source": "resource_binding_status",
                    "next_step": f"Bind a {resource_type} resource before exporting as deliverable.",
                })
                continue
            if status == "unregistered":
                actions.append({
                    **common,
                    "action": "register_delivery_resource",
                    "reason_code": "resource_not_registered",
                    "reason_label": f"{resource_type} {resource_id} is not in the resource catalog.",
                    "severity": "manual_check",
                    "owner": "delivery_owner",
                    "target": resource_id,
                    "source": "resource_binding_status",
                    "next_step": "Register or replace the delivery resource, then re-evaluate the package.",
                })
                continue
            if status == "manual_check":
                actions.append({
                    **common,
                    "action": "review_delivery_resource",
                    "reason_code": "resource_manual_check_required",
                    "reason_label": str(check.get("message") or f"{resource_id} requires delivery review."),
                    "severity": "manual_check",
                    "owner": "delivery_owner",
                    "target": resource_id,
                    "source": "resource_binding_status",
                    "next_step": "Approve, replace, or publish the resource before customer signoff.",
                })
                continue
            actions.append({
                **common,
                "action": "replace_blocked_resource",
                "reason_code": "resource_binding_blocked",
                "reason_label": str(check.get("message") or f"{resource_id} cannot be used for delivery."),
                "severity": "blocked",
                "owner": "delivery_owner",
                "target": resource_id or resource_type,
                "source": "resource_binding_status",
                "next_step": "Replace the blocked resource before importing this package.",
            })

        acceptance = _mapping(item.get("acceptance_status"))
        acceptance_status = str(acceptance.get("status") or "blocked")
        for requirement in _string_list(acceptance.get("missing")):
            severity = "blocked" if acceptance_status == "blocked" else "manual_check"
            actions.append({
                **common,
                "action": "bind_acceptance_requirement",
                "reason_code": "acceptance_requirement_missing",
                "reason_label": f"Acceptance requirement {requirement} is missing.",
                "severity": severity,
                "owner": "delivery_owner",
                "target": requirement,
                "source": "acceptance_status",
                "next_step": "Complete required model, protocol, skill, and test bindings.",
            })
        for check in acceptance.get("acceptance_checks", []) if isinstance(acceptance.get("acceptance_checks"), list) else []:
            check = _mapping(check)
            status = str(check.get("status") or "")
            if status == "linked":
                continue
            reference = str(check.get("reference") or "")
            if status in {"node_unresolved", "read_error"}:
                actions.append({
                    **common,
                    "action": "resolve_acceptance_test_reference",
                    "reason_code": "acceptance_test_manual_check_required",
                    "reason_label": str(check.get("message") or "Acceptance test needs review."),
                    "severity": "manual_check",
                    "owner": "qa_owner",
                    "target": reference,
                    "source": "acceptance_status",
                    "next_step": "Resolve the pytest node or add a stable scenario alias.",
                })
                continue
            actions.append({
                **common,
                "action": "fix_acceptance_test_reference",
                "reason_code": "acceptance_test_blocked",
                "reason_label": str(check.get("message") or "Acceptance test reference is blocked."),
                "severity": "blocked",
                "owner": "qa_owner",
                "target": reference,
                "source": "acceptance_status",
                "next_step": "Fix the missing or unsafe acceptance test reference.",
            })

    if not objects:
        actions.append({
            "object_id": "",
            "display_name": "",
            "object_type": "",
            "category": "",
            "action": "add_managed_object_scope",
            "reason_code": "managed_object_catalog_empty",
            "reason_label": "Managed object catalog is empty.",
            "severity": "blocked",
            "owner": "delivery_owner",
            "target": "managed_objects",
            "source": "managed_object_catalog",
            "next_step": "Add customer-specific managed objects before exporting a handoff package.",
        })

    blocked_count = len([item for item in actions if item.get("severity") == "blocked"])
    manual_count = len([item for item in actions if item.get("severity") == "manual_check"])
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check_required"
        if manual_count
        else "deliverable"
    )
    source_version = _sha256_json({
        "objects": [
            {
                "object_id": str(item.get("object_id") or ""),
                "resource_binding_status": _mapping(item.get("resource_binding_status")).get("overall_status"),
                "acceptance_status": _mapping(item.get("acceptance_status")).get("status"),
            }
            for item in objects
            if isinstance(item, dict)
        ],
        "actions": [
            {
                "object_id": str(item.get("object_id") or ""),
                "action": str(item.get("action") or ""),
                "severity": str(item.get("severity") or ""),
                "target": str(item.get("target") or ""),
            }
            for item in actions
        ],
    })
    return {
        "plan_type": "askme.customer_project.managed_object_action_plan",
        "overall_status": overall_status,
        "object_count": len(objects),
        "object_action_count": len({str(item.get("object_id") or "") for item in actions if item.get("object_id")}),
        "action_count": len(actions),
        "blocked_action_count": blocked_count,
        "manual_check_action_count": manual_count,
        "delivery_gate_source_version": source_version,
        "actions": actions[:100],
        "next_step": {
            "deliverable": "No managed-object package action is open.",
            "manual_check_required": "Resolve manual checks before customer signoff.",
            "blocked": "Fix blocked object bindings before importing as a customer project.",
        }[overall_status],
    }


def _customer_project_package_delivery_gate(
    *,
    action_plan: dict[str, Any],
    acceptance_summary: dict[str, Any],
    binding_readiness: dict[str, Any],
    reuse_assessment: dict[str, Any],
    checked_at: str = "",
) -> dict[str, Any]:
    reasons = [
        {
            "object_id": str(item.get("object_id") or ""),
            "object_type": str(item.get("object_type") or item.get("category") or ""),
            "reason_code": str(item.get("reason_code") or "action_plan_present"),
            "reason_label": str(item.get("reason_label") or item.get("action") or ""),
            "severity": str(item.get("severity") or "manual_check"),
            "source": str(item.get("source") or "managed_object_action_plan"),
            "owner": str(item.get("owner") or ""),
            "target": str(item.get("target") or ""),
            "next_step": str(item.get("next_step") or ""),
        }
        for item in action_plan.get("actions", [])
        if isinstance(item, dict)
    ]
    blocked_count = int(action_plan.get("blocked_action_count") or 0)
    manual_count = int(action_plan.get("manual_check_action_count") or 0)
    for status_source, status in (
        ("acceptance_summary", str(_mapping(acceptance_summary).get("overall_status") or "")),
        ("binding_readiness_summary", str(_mapping(binding_readiness).get("overall_status") or "")),
        ("reuse_assessment", str(_mapping(reuse_assessment).get("status") or "")),
    ):
        if status == "blocked" and not any(item["source"] == status_source for item in reasons):
            blocked_count += 1
            reasons.append({
                "object_id": "",
                "object_type": "",
                "reason_code": f"{status_source}_blocked",
                "reason_label": f"{status_source} is blocked.",
                "severity": "blocked",
                "source": status_source,
                "owner": "delivery_owner",
                "target": status_source,
                "next_step": "Fix the blocked delivery gate before customer handoff.",
            })
        elif status == "manual_check" and not any(item["source"] == status_source for item in reasons):
            manual_count += 1
            reasons.append({
                "object_id": "",
                "object_type": "",
                "reason_code": f"{status_source}_manual_check_required",
                "reason_label": f"{status_source} requires manual review.",
                "severity": "manual_check",
                "source": status_source,
                "owner": "delivery_owner",
                "target": status_source,
                "next_step": "Review this gate before customer signoff.",
            })
    if blocked_count:
        status = "blocked"
    elif manual_count:
        status = "manual_check_required"
    else:
        status = "deliverable"
    export_allowed = status != "blocked"
    import_allowed = status != "blocked"
    return {
        "gate_type": "askme.customer_project.package_delivery_gate",
        "delivery_gate_status": status,
        "delivery_gate_reasons": reasons,
        "delivery_gate_checked_at": checked_at,
        "delivery_gate_source_version": str(action_plan.get("delivery_gate_source_version") or ""),
        "export_allowed": export_allowed,
        "import_allowed": import_allowed,
        "customer_handoff_ready": status == "deliverable",
        "action_count": len(reasons),
        "blocked_action_count": blocked_count,
        "manual_check_action_count": manual_count,
        "customer_status": {
            "deliverable": "Package has no open managed-object delivery actions.",
            "manual_check_required": "Package can be imported for pilot work, but it cannot be signed off until manual checks close.",
            "blocked": "Package is blocked and must not be imported as a customer project handoff.",
        }[status],
        "release_claim": (
            "This gate controls customer handoff readiness only. Production launch still requires "
            "onsite device, notification, voice, and robot runtime acceptance."
        ),
        "next_step": {
            "deliverable": "Import into the target customer namespace, then run onsite acceptance.",
            "manual_check_required": "Import only into a controlled pilot namespace and close manual checks before signoff.",
            "blocked": "Fix blocked object bindings or acceptance evidence, then export a new package.",
        }[status],
    }


def _customer_project_package_import_gate_result(delivery_gate: dict[str, Any]) -> str:
    status = str(delivery_gate.get("delivery_gate_status") or "blocked")
    if status == "deliverable":
        return "accepted"
    if status == "manual_check_required":
        return "accepted_with_manual_check"
    return "rejected"


def _customer_project_reuse_dependencies(
    profile: dict[str, Any],
    objects: list[dict[str, Any]],
    env_references: list[dict[str, Any]],
) -> dict[str, Any]:
    devices = _mapping(profile.get("devices"))
    responders = _mapping(profile.get("responder_groups"))
    bindings = [_mapping(item.get("bindings")) for item in objects]
    return {
        "device_count": len(devices),
        "device_sources": sorted({
            source
            for item in objects
            for source in _string_list(item.get("device_sources"))
        }),
        "responder_groups": sorted({
            str(item.get("responder_group") or "")
            for item in objects
            if str(item.get("responder_group") or "").strip()
        } | set(str(group) for group in responders)),
        "vision_models": sorted({
            value for binding in bindings for value in _string_list(binding.get("vision_models"))
        }),
        "sensor_protocols": sorted({
            value for binding in bindings for value in _string_list(binding.get("sensor_protocols"))
        }),
        "skill_packages": sorted({
            value for binding in bindings for value in _string_list(binding.get("skill_packages"))
        }),
        "acceptance_tests": sorted({
            value for binding in bindings for value in _string_list(binding.get("acceptance_tests"))
        }),
        "binding_readiness": _managed_object_binding_readiness_summary(objects),
        "env_reference_count": len(env_references),
        "required_env_count": len([item for item in env_references if item.get("required")]),
        "missing_env_count": len([
            item for item in env_references if item.get("required") and not item.get("configured")
        ]),
    }


def validate_field_site_profile(profile: dict[str, Any], *, check_env: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    site = _mapping(profile.get("site"))
    customer = _mapping(profile.get("customer"))
    zones = _mapping(profile.get("zones"))
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    thresholds = _mapping(profile.get("thresholds"))
    managed_objects = _mapping(profile.get("managed_objects"))

    if not site.get("site_id"):
        errors.append("site.site_id is required")
    if not site.get("name"):
        errors.append("site.name is required")
    if not zones:
        errors.append("zones must contain at least one zone")
    if not devices:
        errors.append("devices must contain at least one registered device")

    main_channels = _zones_by_type(zones, "main_channel")
    help_points = _zones_by_type(zones, "help_point")
    parking_restricted = [
        zone_id for zone_id, zone in main_channels.items() if zone.get("parking_allowed") is False
    ]
    if not main_channels:
        errors.append("zones must include at least one main_channel")
    if not parking_restricted:
        errors.append("at least one main_channel must set parking_allowed=false")
    if not help_points:
        errors.append("zones must include at least one help_point")
    for zone_id, zone in help_points.items():
        if not zone.get("help_point_id"):
            errors.append(f"zones.{zone_id}.help_point_id is required for help_point zones")
        if not zone.get("location"):
            errors.append(f"zones.{zone_id}.location is required for help_point zones")

    for group in REQUIRED_RESPONDER_GROUPS:
        responder = _mapping(responders.get(group))
        if not responder:
            errors.append(f"responder_groups.{group} is required")
            continue
        _require_env_reference(
            responder,
            key="webhook_env",
            errors=errors,
            warnings=warnings,
            label=f"responder_groups.{group}.webhook_env",
            check_env=check_env,
        )
        _require_env_reference(
            responder,
            key="secret_env",
            errors=errors,
            warnings=warnings,
            label=f"responder_groups.{group}.secret_env",
            check_env=check_env,
        )

    source_counts: dict[str, int] = {}
    zone_ids = set(zones)
    for device_id, device in devices.items():
        if not isinstance(device, dict):
            errors.append(f"devices.{device_id} must be a mapping")
            continue
        source = str(device.get("source") or "")
        if not source:
            errors.append(f"devices.{device_id}.source is required")
        source_counts[source] = source_counts.get(source, 0) + 1
        zone_id = str(device.get("zone_id") or "")
        if zone_id and zone_id not in zone_ids:
            errors.append(f"devices.{device_id}.zone_id references unknown zone {zone_id}")
        _require_env_reference(
            device,
            key="secret_env",
            errors=errors,
            warnings=warnings,
            label=f"devices.{device_id}.secret_env",
            check_env=check_env,
        )
    for source in REQUIRED_DEVICE_SOURCES:
        if source_counts.get(source, 0) <= 0:
            errors.append(f"devices must include at least one {source} device")

    _validate_thresholds(thresholds, errors)
    _validate_customer_project(customer, warnings)
    object_summary = _validate_managed_objects(
        managed_objects,
        zones=zones,
        responders=responders,
        source_counts=source_counts,
        errors=errors,
        warnings=warnings,
    )
    status = "passed" if not errors else "failed"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "tenant_id": _delivery_tenant_id(customer),
            "delivery_namespace": _delivery_namespace(customer),
            "site_id": str(site.get("site_id") or ""),
            "site_name": str(site.get("name") or ""),
            "map_version": str(site.get("map_version") or ""),
            "zone_count": len(zones),
            "main_channel_count": len(main_channels),
            "parking_restricted_count": len(parking_restricted),
            "help_point_count": len(help_points),
            "device_count": len(devices),
            "device_sources": source_counts,
            "responder_groups": sorted(responders),
            "customer_id": str(customer.get("customer_id") or ""),
            "customer_name": str(customer.get("customer_name") or ""),
            "industry": str(customer.get("industry") or ""),
            "project_id": str(customer.get("project_id") or ""),
            "project_name": str(customer.get("project_name") or ""),
            "managed_object_type_count": object_summary["object_type_count"],
            "managed_object_categories": object_summary["categories"],
        },
        "readiness": {
            "map_configured": bool(zones),
            "parking_policy_configured": bool(parking_restricted),
            "wayfinding_configured": bool(help_points),
            "responder_groups_configured": all(group in responders for group in REQUIRED_RESPONDER_GROUPS),
            "device_registry_configured": bool(devices),
            "customer_project_configured": bool(customer.get("customer_id") and customer.get("project_id")),
            "managed_objects_configured": object_summary["object_type_count"] > 0,
        },
        "managed_objects_summary": object_summary,
    }


def field_operations_config_from_site_profile(profile: dict[str, Any]) -> dict[str, Any]:
    zones = _mapping(profile.get("zones"))
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    thresholds = _mapping(profile.get("thresholds"))
    site = _mapping(profile.get("site"))
    customer = _customer_payload(profile)
    managed_objects = managed_object_catalog_from_site_profile(profile)
    config = {
        "customer_project": customer | {"site_id": site.get("site_id"), "site_name": site.get("name")},
        "customer_id": customer.get("customer_id"),
        "project_id": customer.get("project_id"),
        "industry": customer.get("industry"),
        "site_id": site.get("site_id"),
        "site_name": site.get("name"),
        "site_map": {"zones": zones},
        "device_registry": {
            device_id: _device_registry_entry(device)
            for device_id, device in devices.items()
            if isinstance(device, dict)
        },
        "dingtalk_webhooks": {
            group: _env_placeholder(responder.get("webhook_env"))
            for group, responder in responders.items()
            if isinstance(responder, dict)
        },
        "dingtalk_secrets": {
            group: _env_placeholder(responder.get("secret_env"))
            for group, responder in responders.items()
            if isinstance(responder, dict)
        },
        "thresholds": thresholds,
        "managed_objects": managed_objects["objects_by_id"],
    }
    config.update(_field_threshold_config(thresholds))
    return config


def managed_object_catalog_from_site_profile(
    profile: dict[str, Any],
    *,
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Return the customer-specific objects this project expects the robot to handle."""
    managed_objects = _mapping(profile.get("managed_objects"))
    resource_catalog = _delivery_resource_catalog(
        profile,
        delivery_resource_root=delivery_resource_root,
    )
    objects = [
        _managed_object_payload(object_id, item, resource_catalog=resource_catalog)
        for object_id, item in sorted(managed_objects.items())
        if isinstance(item, dict)
    ]
    categories = sorted({str(item.get("category") or "uncategorized") for item in objects})
    scenario_ids = sorted({
        str(scenario_id)
        for item in objects
        for scenario_id in item.get("scenario_ids", [])
        if scenario_id
    })
    return {
        "object_type_count": len(objects),
        "categories": categories,
        "scenario_ids": scenario_ids,
        "resource_catalog_summary": _delivery_resource_catalog_summary(resource_catalog),
        "binding_readiness_summary": _managed_object_binding_readiness_summary(objects),
        "acceptance_summary": _managed_object_acceptance_summary(objects),
        "objects": objects,
        "objects_by_id": {str(item["object_id"]): item for item in objects},
        "customer_claim": (
            "Managed object catalog is configured for this customer project."
            if objects
            else "Managed object catalog is not configured yet."
        ),
    }


def build_customer_project_acceptance_registry(
    profile_root: Path,
    *,
    template_root: Path | None = Path("deploy/customer-project-templates"),
) -> dict[str, Any]:
    """Return a delivery registry for all managed-object acceptance references."""
    consumers: list[dict[str, Any]] = []
    for path in _site_profile_paths(profile_root, pattern="*.yaml"):
        consumers.extend(_acceptance_registry_consumers_from_profile(path, scope_type="project"))
    if template_root is not None:
        for path in _site_profile_paths(template_root, pattern="*.yaml"):
            consumers.extend(_acceptance_registry_consumers_from_profile(path, scope_type="template"))
    references = _acceptance_registry_references(consumers)
    summary = _acceptance_registry_summary(consumers, references)
    return {
        "profile_root": str(profile_root),
        "template_root": str(template_root) if template_root is not None else "",
        "summary": summary,
        "references": references,
        "consumers": consumers,
        "customer_claim": (
            "Managed-object acceptance references are inspectable across projects and templates."
        ),
        "next_step": _acceptance_registry_next_step(summary),
    }


def build_customer_project_resource_catalog(
    profile_root: Path,
    *,
    template_root: Path | None = Path("deploy/customer-project-templates"),
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Return product resource bindings used by customer projects and templates."""
    consumers: list[dict[str, Any]] = []
    resource_catalog = _delivery_resource_catalog({}, delivery_resource_root=delivery_resource_root)
    for path in _site_profile_paths(profile_root, pattern="*.yaml"):
        consumers.extend(_delivery_resource_consumers_from_profile(path, scope_type="project"))
        try:
            resource_catalog = _merge_delivery_resource_catalog(
                resource_catalog,
                _delivery_resource_catalog(
                    load_field_site_profile(path),
                    include_defaults=False,
                    include_shared=False,
                ),
            )
        except Exception:
            continue
    if template_root is not None:
        for path in _site_profile_paths(template_root, pattern="*.yaml"):
            consumers.extend(_delivery_resource_consumers_from_profile(path, scope_type="template"))
            try:
                resource_catalog = _merge_delivery_resource_catalog(
                    resource_catalog,
                    _delivery_resource_catalog(
                        load_field_site_profile(path),
                        include_defaults=False,
                        include_shared=False,
                    ),
                )
            except Exception:
                continue
    resources = _delivery_resource_rows(resource_catalog, consumers)
    summary = _delivery_resource_registry_summary(resources, consumers)
    return {
        "profile_root": str(profile_root),
        "template_root": str(template_root) if template_root is not None else "",
        "delivery_resource_root": str(delivery_resource_root) if delivery_resource_root is not None else "",
        "summary": summary,
        "resources": resources,
        "consumers": consumers,
        "customer_claim": (
            "Managed object bindings are checked against a product resource catalog before delivery signoff."
        ),
        "next_step": _delivery_resource_registry_next_step(summary),
    }


def build_customer_project_execution_bindings(
    profile_root: Path,
    identifier: str,
    *,
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Return executable ingest/runtime binding plans for one customer project."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found"}
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    catalog = managed_object_catalog_from_site_profile(
        profile,
        delivery_resource_root=delivery_resource_root,
    )
    devices_by_source = _field_devices_by_source(profile)
    plans = [
        _managed_object_execution_binding_plan(item, profile, devices_by_source)
        for item in catalog.get("objects", [])
        if isinstance(item, dict)
    ]
    summary = _execution_binding_summary(plans)
    return {
        "found": True,
        "profile_path": str(path),
        "project_scope": _delivery_scope_payload(profile),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "summary": summary,
        "plans": plans,
        "plans_by_object_id": {str(item.get("object_id") or ""): item for item in plans},
        "customer_claim": _execution_binding_customer_claim(summary),
        "next_step": _execution_binding_next_step(summary),
    }


def site_profile_env_references(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deployment environment variables referenced by a field site profile."""
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    refs: list[dict[str, Any]] = []
    for group, responder in sorted(responders.items()):
        if not isinstance(responder, dict):
            continue
        group_name = str(responder.get("name") or group)
        refs.extend(
            [
                _env_reference(
                    responder.get("webhook_env"),
                    category="dingtalk_webhook",
                    reference=f"responder_groups.{group}.webhook_env",
                    owner=str(group),
                    owner_label=group_name,
                    purpose=f"{group_name} DingTalk robot webhook URL",
                ),
                _env_reference(
                    responder.get("secret_env"),
                    category="dingtalk_secret",
                    reference=f"responder_groups.{group}.secret_env",
                    owner=str(group),
                    owner_label=group_name,
                    purpose=f"{group_name} DingTalk robot signing secret",
                ),
            ]
        )
    for device_id, device in sorted(devices.items()):
        if not isinstance(device, dict):
            continue
        device_name = str(device.get("name") or device_id)
        refs.append(
            _env_reference(
                device.get("secret_env"),
                category="field_device_secret",
                reference=f"devices.{device_id}.secret_env",
                owner=str(device_id),
                owner_label=device_name,
                purpose=f"{device_name} field-device HMAC ingest secret",
                metadata={
                    "source": str(device.get("source") or ""),
                    "zone_id": str(device.get("zone_id") or ""),
                },
            )
        )
    return _dedupe_env_references(refs)


def render_site_profile_env_template(
    profile: dict[str, Any],
    *,
    include_comments: bool = True,
) -> str:
    """Render a deterministic .env template for site deployment handoff."""
    refs = site_profile_env_references(profile)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        grouped.setdefault(str(ref.get("category") or "other"), []).append(ref)

    lines: list[str] = []
    site = _mapping(profile.get("site"))
    if include_comments:
        lines.extend(
            [
                "# AskMe field site environment template",
                f"# Site: {site.get('site_id') or '-'} / {site.get('name') or '-'}",
                "# Fill values before running field-readiness --check-site-env.",
                "",
            ]
        )
    for category, title in (
        ("dingtalk_webhook", "DingTalk responder webhooks"),
        ("dingtalk_secret", "DingTalk responder signing secrets"),
        ("field_device_secret", "Field device HMAC ingest secrets"),
        ("other", "Other site secrets"),
    ):
        items = grouped.get(category) or []
        if not items:
            continue
        if include_comments:
            lines.append(f"# {title}")
        for item in items:
            env_name = str(item.get("env_name") or "").strip()
            if not env_name:
                continue
            if include_comments:
                lines.append(f"# {item.get('reference')}: {item.get('purpose')}")
            lines.append(f"{env_name}=")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _device_registry_entry(device: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "source": device.get("source"),
        "zone_id": device.get("zone_id"),
        "secret": _env_placeholder(device.get("secret_env")),
    }
    for key in ("name", "camera_id", "sensor_type", "robot_id"):
        if device.get(key):
            entry[key] = device.get(key)
    return {key: value for key, value in entry.items() if value not in ("", None)}


def _field_threshold_config(thresholds: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "parking_duration_s": "parking_duration_s",
        "night_stranger_dwell_s": "night_stranger_dwell_s",
        "fire_temperature_c": "fire_temperature_c",
        "smoke_level": "smoke_level",
        "trash_fill_ratio": "trash_fill_ratio",
        "crowd_person_count": "crowd_person_count",
        "crowd_duration_min": "crowd_duration_min",
    }
    return {
        target: thresholds[source]
        for source, target in mapping.items()
        if source in thresholds
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _site_profile_paths(root: Path, *, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    paths = list(root.rglob(pattern))
    if pattern != "*.yaml":
        return [path for path in paths if path.is_file()]
    paths.extend(root.rglob("*.yml"))
    return sorted({path.resolve() for path in paths if path.is_file()})


def _site_profile_catalog_item(path: Path, *, check_env: bool) -> dict[str, Any]:
    try:
        profile = load_field_site_profile(path)
        report = validate_field_site_profile(profile, check_env=check_env)
    except Exception as exc:
        return {
            "site_id": "",
            "site_name": path.stem,
            "profile_path": str(path),
            "status": "failed",
            "deployment_stage": "blocked",
            "customer_status": "Blocked",
            "errors": [str(exc)],
            "warnings": [],
            "readiness": {},
            "env_missing_count": 0,
            "delivery_workflow": _customer_project_delivery_workflow(
                profile={},
                report={"status": "failed", "errors": [str(exc)], "warnings": []},
                managed_objects={},
                env_missing=[],
            ),
            "next_step": "Fix the site profile file so it can be parsed and validated.",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    managed_objects = managed_object_catalog_from_site_profile(profile)
    env_refs = site_profile_env_references(profile)
    env_missing = [item for item in env_refs if item.get("required") and not item.get("configured")]
    status = str(report.get("status") or "failed")
    deployment_stage = _site_deployment_stage(status, warnings, env_missing, check_env=check_env)
    return {
        "site_id": str(summary.get("site_id") or ""),
        "site_name": str(summary.get("site_name") or path.stem),
        "map_version": str(summary.get("map_version") or ""),
        "profile_path": str(path),
        "status": status,
        "deployment_stage": deployment_stage,
        "customer_status": _site_customer_status(deployment_stage),
        "errors": errors,
        "warnings": warnings,
        "readiness": report.get("readiness") if isinstance(report.get("readiness"), dict) else {},
        "summary": summary,
        "customer": _customer_payload(profile),
        "managed_objects_summary": managed_objects | {
            "objects": managed_objects["objects"][:8],
            "objects_by_id": {},
        },
        "managed_objects": managed_objects["objects"],
        "object_change_log": _object_change_log_payload(profile),
        "delivery_workflow": _customer_project_delivery_workflow(
            profile=profile,
            report=report,
            managed_objects=managed_objects,
            env_missing=env_missing,
        ),
        "env_reference_count": len(env_refs),
        "env_missing_count": len(env_missing),
        "env_missing": [
            {
                "env_name": str(item.get("env_name") or ""),
                "category": str(item.get("category") or ""),
                "owner": str(item.get("owner") or ""),
                "purpose": str(item.get("purpose") or ""),
            }
            for item in env_missing[:12]
        ],
        "next_step": _site_profile_next_step(status, deployment_stage, errors, warnings, env_missing),
    }


def _customer_project_delivery_workflow(
    *,
    profile: dict[str, Any],
    report: dict[str, Any],
    managed_objects: dict[str, Any],
    env_missing: list[dict[str, Any]],
) -> dict[str, Any]:
    customer = _customer_payload(profile) if isinstance(profile, dict) else {}
    site = _mapping(profile.get("site")) if isinstance(profile, dict) else {}
    summary = _mapping(report.get("summary"))
    readiness = _mapping(report.get("readiness"))
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    acceptance = _mapping(managed_objects.get("acceptance_summary"))
    binding_missing_count = _managed_object_binding_missing_count(objects)
    vision_models = _unique_template_binding_values(objects, "vision_models")
    sensor_protocols = _unique_template_binding_values(objects, "sensor_protocols")
    skill_packages = _unique_template_binding_values(objects, "skill_packages")
    scope_fields = {
        "tenant_id": customer.get("tenant_id"),
        "delivery_namespace": customer.get("delivery_namespace"),
        "customer_id": customer.get("customer_id"),
        "project_id": customer.get("project_id"),
        "site_id": site.get("site_id") or summary.get("site_id"),
    }
    scope_ready = all(str(value or "").strip() for value in scope_fields.values())
    map_ready = bool(readiness.get("map_configured")) and bool(readiness.get("device_registry_configured"))
    responder_ready = bool(readiness.get("responder_groups_configured"))
    steps = [
        {
            "step_id": "customer_scope",
            "label": "Customer scope",
            "status": "ready" if scope_ready else "blocked",
            "evidence": " / ".join(str(value or "-") for value in scope_fields.values()),
            "next_step": "Set tenant, namespace, customer, project, and site identifiers.",
        },
        {
            "step_id": "managed_objects",
            "label": "Managed object catalog",
            "status": "ready" if objects else "blocked",
            "evidence": f"{len(objects)} object(s), {len(_string_list(managed_objects.get('scenario_ids')))} scenario(s).",
            "next_step": "Define the customer's real assets, areas, visitor services, and response ownership.",
        },
        {
            "step_id": "runtime_bindings",
            "label": "Runtime bindings",
            "status": "ready" if objects and binding_missing_count == 0 else "manual_check" if objects else "blocked",
            "evidence": (
                f"{len(vision_models)} vision model(s), {len(sensor_protocols)} sensor protocol(s), "
                f"{len(skill_packages)} skill package(s), {binding_missing_count} missing binding(s)."
            ),
            "next_step": "Bind every managed object to model, protocol, skill, and acceptance references.",
        },
        {
            "step_id": "site_map_devices",
            "label": "Site map and devices",
            "status": "ready" if report.get("status") == "passed" and map_ready else "blocked",
            "evidence": (
                f"{summary.get('zone_count', 0)} zone(s), {summary.get('device_count', 0)} device(s), "
                f"{summary.get('help_point_count', 0)} help point(s)."
            ),
            "next_step": "Complete map zones, help points, device registry, and source coverage.",
        },
        {
            "step_id": "responder_credentials",
            "label": "Responder and credentials",
            "status": "manual_check" if env_missing else "ready" if responder_ready else "blocked",
            "evidence": f"{len(env_missing)} missing required environment value(s).",
            "next_step": "Configure DingTalk responders, robot callbacks, and signed device secrets.",
        },
        {
            "step_id": "acceptance_evidence",
            "label": "Acceptance evidence",
            "status": str(acceptance.get("overall_status") or "blocked"),
            "evidence": (
                f"{acceptance.get('ready_object_count', 0)} ready / "
                f"{acceptance.get('manual_check_object_count', 0)} manual / "
                f"{acceptance.get('blocked_object_count', 0)} blocked object(s)."
            ),
            "next_step": "Run repository evidence and onsite smoke tests before customer signoff.",
        },
        {
            "step_id": "handoff_package",
            "label": "Handoff package",
            "status": "manual_check",
            "evidence": "Export package and acceptance dossier after the above steps are reviewed.",
            "next_step": "Export the customer package, acceptance report, and printable dossier.",
        },
    ]
    blocked_count = sum(1 for item in steps if item["status"] == "blocked")
    manual_count = sum(1 for item in steps if item["status"] == "manual_check")
    ready_count = sum(1 for item in steps if item["status"] == "ready")
    overall_status = "blocked" if blocked_count else "manual_check" if manual_count else "ready"
    next_step = next(
        (str(item.get("next_step") or "") for item in steps if item.get("status") != "ready"),
        "Project delivery workflow is ready for final handoff review.",
    )
    return {
        "overall_status": overall_status,
        "ready_count": ready_count,
        "manual_check_count": manual_count,
        "blocked_count": blocked_count,
        "steps": steps,
        "next_step": next_step,
        "customer_status": _customer_project_delivery_status(overall_status, blocked_count, manual_count),
        "release_claim": (
            "This workflow supports delivery review and customer pilot handoff. Production launch still requires "
            "onsite hardware, credential, safety, and operator-takeover acceptance."
        ),
    }


def _customer_project_product_acceptance_gate(project: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(project.get("managed_objects_summary"))
    acceptance = _mapping(summary.get("acceptance_summary"))
    binding = _mapping(summary.get("binding_readiness_summary"))
    workflow = _mapping(project.get("delivery_workflow"))
    objects = project.get("managed_objects") if isinstance(project.get("managed_objects"), list) else []
    scope_ready = all(
        str(project.get(key) or "").strip()
        for key in ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id")
    )
    object_count = int(summary.get("object_type_count") or len(objects))
    change_log_count = len(project.get("object_change_log") if isinstance(project.get("object_change_log"), list) else [])
    gates = [
        {
            "gate_id": "customer_scope",
            "label": "Customer and delivery scope",
            "status": "ready" if scope_ready else "blocked",
            "evidence": (
                f"tenant={project.get('tenant_id') or '-'}; "
                f"namespace={project.get('delivery_namespace') or '-'}; "
                f"customer={project.get('customer_id') or '-'}; "
                f"project={project.get('project_id') or '-'}; "
                f"site={project.get('site_id') or '-'}"
            ),
            "next_step": "Set tenant, delivery namespace, customer, project, and site identifiers.",
        },
        {
            "gate_id": "site_profile",
            "label": "Site profile",
            "status": "ready" if str(project.get("status") or "") == "passed" else "blocked",
            "evidence": str(project.get("status") or "failed"),
            "next_step": "Fix site profile validation before customer handoff.",
        },
        {
            "gate_id": "managed_object_catalog",
            "label": "Managed object catalog",
            "status": "ready" if object_count else "blocked",
            "evidence": f"{object_count} managed object(s)",
            "next_step": "Define customer-visible objects, scenarios, response owner, and evidence needs.",
        },
        {
            "gate_id": "resource_bindings",
            "label": "Vision, sensor, skill, and acceptance bindings",
            "status": str(binding.get("overall_status") or "blocked"),
            "evidence": (
                f"ready={binding.get('ready_object_count', 0)}; "
                f"manual={binding.get('manual_check_object_count', 0)}; "
                f"blocked={binding.get('blocked_object_count', 0)}; "
                f"unregistered={binding.get('unregistered_resource_count', 0)}"
            ),
            "next_step": "Register missing resources or replace disabled and blocked bindings.",
        },
        {
            "gate_id": "acceptance_references",
            "label": "Acceptance evidence references",
            "status": str(acceptance.get("overall_status") or "blocked"),
            "evidence": (
                f"ready={acceptance.get('ready_object_count', 0)}; "
                f"manual={acceptance.get('manual_check_object_count', 0)}; "
                f"blocked={acceptance.get('blocked_object_count', 0)}"
            ),
            "next_step": "Attach passing acceptance tests or onsite receipts to blocked objects.",
        },
        {
            "gate_id": "object_change_audit",
            "label": "Object change audit",
            "status": "ready",
            "evidence": f"{change_log_count} recorded object change(s)",
            "next_step": "Future object create, update, and offline actions are recorded with operator and reason.",
        },
        {
            "gate_id": "handoff_artifacts",
            "label": "Handoff artifacts",
            "status": "manual_check",
            "evidence": str(workflow.get("next_step") or "Acceptance report, dossier, proposal, and export package must be reviewed."),
            "next_step": "Generate and verify customer package, acceptance dossier, proposal bundle, and audit export.",
        },
    ]
    blocked_count = sum(1 for item in gates if item["status"] == "blocked")
    manual_count = sum(1 for item in gates if item["status"] == "manual_check")
    ready_count = sum(1 for item in gates if item["status"] == "ready")
    overall_status = "blocked" if blocked_count else "manual_check" if manual_count else "ready"
    return {
        "gate_type": "askme.solution_delivery_product_acceptance_gate",
        "overall_status": overall_status,
        "ready_count": ready_count,
        "manual_check_count": manual_count,
        "blocked_count": blocked_count,
        "can_enter_customer_signoff": overall_status == "ready",
        "gates": gates,
        "customer_status": _customer_project_delivery_status(overall_status, blocked_count, manual_count),
        "next_step": next(
            (str(item.get("next_step") or "") for item in gates if item.get("status") != "ready"),
            "Project can enter customer signoff review.",
        ),
    }


def _customer_project_catalog_delivery_acceptance_gate(
    projects: list[dict[str, Any]],
) -> dict[str, Any]:
    if not projects:
        return {
            "gate_type": "askme.solution_delivery_catalog_acceptance_gate",
            "overall_status": "blocked",
            "project_count": 0,
            "ready_count": 0,
            "manual_check_count": 0,
            "blocked_count": 0,
            "customer_status": "No customer projects are configured.",
            "next_step": "Create a customer project from an approved industry template.",
            "blocked_projects": [],
            "manual_check_projects": [],
        }
    blocked_projects: list[dict[str, str]] = []
    manual_projects: list[dict[str, str]] = []
    ready_count = 0
    for project in projects:
        gate = _mapping(project.get("product_acceptance_gate"))
        status = str(gate.get("overall_status") or "blocked")
        row = {
            "tenant_id": str(project.get("tenant_id") or ""),
            "delivery_namespace": str(project.get("delivery_namespace") or ""),
            "customer_id": str(project.get("customer_id") or ""),
            "project_id": str(project.get("project_id") or ""),
            "site_id": str(project.get("site_id") or ""),
            "next_step": str(gate.get("next_step") or ""),
        }
        if status == "blocked":
            blocked_projects.append(row)
        elif status == "manual_check":
            manual_projects.append(row)
        else:
            ready_count += 1
    overall = (
        "blocked"
        if blocked_projects
        else "manual_check"
        if manual_projects
        else "ready"
    )
    return {
        "gate_type": "askme.solution_delivery_catalog_acceptance_gate",
        "overall_status": overall,
        "project_count": len(projects),
        "ready_count": ready_count,
        "manual_check_count": len(manual_projects),
        "blocked_count": len(blocked_projects),
        "customer_status": _customer_project_delivery_status(
            overall,
            len(blocked_projects),
            len(manual_projects),
        ),
        "next_step": (
            blocked_projects[0]["next_step"]
            if blocked_projects
            else manual_projects[0]["next_step"]
            if manual_projects
            else "All visible customer projects can enter customer signoff review."
        ),
        "blocked_projects": blocked_projects[:20],
        "manual_check_projects": manual_projects[:20],
    }


def _managed_object_binding_missing_count(objects: list[dict[str, Any]]) -> int:
    missing = 0
    for item in objects:
        status = _mapping(item.get("acceptance_status"))
        for requirement in status.get("requirements", []):
            if isinstance(requirement, dict) and requirement.get("status") == "missing":
                missing += 1
    return missing


def _customer_project_delivery_status(
    overall_status: str,
    blocked_count: int,
    manual_count: int,
) -> str:
    if overall_status == "blocked":
        return f"Blocked: {blocked_count} delivery step(s) need configuration before customer handoff."
    if overall_status == "manual_check":
        return f"Review required: {manual_count} delivery step(s) need human evidence or onsite confirmation."
    return "Ready for final customer handoff review."


def _site_deployment_stage(
    status: str,
    warnings: list[Any],
    env_missing: list[dict[str, Any]],
    *,
    check_env: bool,
) -> str:
    if status != "passed":
        return "blocked"
    if check_env and (warnings or env_missing):
        return "site_config_ready"
    return "production_ready" if check_env else "site_config_ready"


def _site_customer_status(stage: str) -> str:
    return {
        "production_ready": "Production credentials configured",
        "site_config_ready": "Site configuration ready; live credentials still need validation",
        "blocked": "Blocked by missing site configuration",
    }.get(stage, "Unknown")


def _site_profile_next_step(
    status: str,
    deployment_stage: str,
    errors: list[Any],
    warnings: list[Any],
    env_missing: list[dict[str, Any]],
) -> str:
    if status != "passed":
        first = str(errors[0]) if errors else "missing required site profile fields"
        return f"Fix site profile validation error: {first}"
    if env_missing:
        return "Fill deployment environment variables for DingTalk responders and signed devices."
    if warnings:
        return "Resolve site profile warnings, then rerun readiness with environment checks."
    if deployment_stage == "production_ready":
        return "Run live field demo and device bridge smoke against the deployed service."
    return "Run readiness with check_env=true before customer production acceptance."


def _site_catalog_next_step(sites: list[dict[str, Any]], *, check_env: bool) -> str:
    if not sites:
        return "Create at least one site profile under deploy/site-profiles."
    blocked = [item for item in sites if item.get("status") != "passed"]
    if blocked:
        return "Fix blocked site profiles before scaling rollout."
    missing_env = sum(int(item.get("env_missing_count") or 0) for item in sites)
    if missing_env:
        return "Fill missing responder and device secrets for each configured site."
    if not check_env:
        return "Rerun the catalog with environment checks before production acceptance."
    return "Select a site and run live device, notification, voice, and runtime smokes."


def _validate_customer_project(customer: dict[str, Any], warnings: list[str]) -> None:
    missing = [
        field
        for field in ("customer_id", "customer_name", "industry", "project_id", "project_name")
        if not customer.get(field)
    ]
    if missing:
        warnings.append(
            "customer project metadata is incomplete: "
            + ", ".join(f"customer.{field}" for field in missing)
        )


def _validate_managed_objects(
    managed_objects: dict[str, Any],
    *,
    zones: dict[str, Any],
    responders: dict[str, Any],
    source_counts: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if not managed_objects:
        warnings.append("managed_objects is recommended for solution-provider customer projects")
        return {
            "object_type_count": 0,
            "categories": [],
            "scenario_ids": [],
            "customer_visible_count": 0,
        }
    zone_types = {
        str(zone.get("type") or "")
        for zone in zones.values()
        if isinstance(zone, dict) and zone.get("type")
    }
    categories: set[str] = set()
    scenario_ids: set[str] = set()
    skill_packages: set[str] = set()
    acceptance_tests: set[str] = set()
    binding_missing_count = 0
    visible_count = 0
    for object_id, item in managed_objects.items():
        if not isinstance(item, dict):
            errors.append(f"managed_objects.{object_id} must be a mapping")
            continue
        display_name = str(item.get("display_name") or "")
        category = str(item.get("category") or "")
        if not display_name:
            errors.append(f"managed_objects.{object_id}.display_name is required")
        if not category:
            errors.append(f"managed_objects.{object_id}.category is required")
        else:
            categories.add(category)
        object_scenarios = _string_list(item.get("scenario_ids"))
        if not object_scenarios:
            errors.append(f"managed_objects.{object_id}.scenario_ids must contain at least one scenario")
        scenario_ids.update(object_scenarios)
        object_zone_types = _string_list(item.get("zone_types"))
        if not object_zone_types:
            errors.append(f"managed_objects.{object_id}.zone_types must contain at least one zone type")
        unknown_zone_types = sorted(set(object_zone_types) - zone_types)
        if unknown_zone_types:
            errors.append(
                f"managed_objects.{object_id}.zone_types references unknown zone types: "
                + ", ".join(unknown_zone_types)
            )
        object_sources = _string_list(item.get("device_sources"))
        missing_sources = sorted(source for source in object_sources if source_counts.get(source, 0) <= 0)
        if missing_sources:
            warnings.append(
                f"managed_objects.{object_id}.device_sources have no registered device: "
                + ", ".join(missing_sources)
            )
        responder_group = str(item.get("responder_group") or "")
        if responder_group and responder_group not in responders:
            errors.append(
                f"managed_objects.{object_id}.responder_group references unknown responder group {responder_group}"
            )
        if item.get("customer_visible", True):
            visible_count += 1
        bindings = _mapping(item.get("bindings"))
        binding_report = _validate_managed_object_bindings(object_id, bindings, warnings)
        binding_missing_count += binding_report["missing_count"]
        skill_packages.update(binding_report["skill_packages"])
        acceptance_tests.update(binding_report["acceptance_tests"])
    return {
        "object_type_count": len([item for item in managed_objects.values() if isinstance(item, dict)]),
        "categories": sorted(categories),
        "scenario_ids": sorted(scenario_ids),
        "customer_visible_count": visible_count,
        "bound_object_type_count": len(managed_objects) - binding_missing_count,
        "binding_missing_count": binding_missing_count,
        "skill_packages": sorted(skill_packages),
        "acceptance_tests": sorted(acceptance_tests),
    }


def _validate_managed_object_bindings(
    object_id: str,
    bindings: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    required = ("vision_models", "sensor_protocols", "skill_packages", "acceptance_tests")
    missing = [key for key in required if not _string_list(bindings.get(key))]
    if missing:
        warnings.append(
            f"managed_objects.{object_id}.bindings missing product bindings: "
            + ", ".join(missing)
        )
    return {
        "missing_count": 1 if missing else 0,
        "skill_packages": _string_list(bindings.get("skill_packages")),
        "acceptance_tests": _string_list(bindings.get("acceptance_tests")),
    }


def _delivery_resource_catalog(
    profile: dict[str, Any],
    *,
    include_defaults: bool = True,
    include_shared: bool = True,
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, dict[str, dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {
        resource_type: {}
        for resource_type in DELIVERY_RESOURCE_TYPES
    }
    if include_defaults:
        catalog = _merge_delivery_resource_catalog(catalog, DEFAULT_DELIVERY_RESOURCE_CATALOG)
    if include_shared and delivery_resource_root is not None:
        try:
            catalog = _merge_delivery_resource_catalog(
                catalog,
                _delivery_resource_catalog(
                    load_delivery_resource_registry(delivery_resource_root),
                    include_defaults=False,
                    include_shared=False,
                ),
            )
        except Exception:
            pass
    configured = _mapping(profile.get("delivery_resources"))
    configured_source_default = (
        "shared_registry"
        if profile.get("registry_type") == "askme.delivery_resource_registry"
        else "profile"
    )
    for resource_type in DELIVERY_RESOURCE_TYPES:
        values = configured.get(resource_type)
        if isinstance(values, dict):
            for resource_id, metadata in values.items():
                catalog[resource_type][str(resource_id)] = _delivery_resource_descriptor(
                    str(resource_id),
                    resource_type,
                    metadata,
                    source_default=configured_source_default,
                )
        elif isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    resource_id = str(value.get("resource_id") or value.get("id") or value.get("name") or "").strip()
                    if resource_id:
                        catalog[resource_type][resource_id] = _delivery_resource_descriptor(
                            resource_id,
                            resource_type,
                            value,
                            source_default=configured_source_default,
                        )
                else:
                    resource_id = str(value or "").strip()
                    if resource_id:
                        catalog[resource_type][resource_id] = _delivery_resource_descriptor(
                            resource_id,
                            resource_type,
                            {},
                            source_default=configured_source_default,
                        )
    return catalog


def _merge_delivery_resource_catalog(
    base: dict[str, dict[str, dict[str, Any]]],
    incoming: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {
        resource_type: {
            resource_id: dict(metadata)
            for resource_id, metadata in _mapping(base.get(resource_type)).items()
            if isinstance(metadata, dict)
        }
        for resource_type in DELIVERY_RESOURCE_TYPES
    }
    for resource_type in DELIVERY_RESOURCE_TYPES:
        for resource_id, metadata in _mapping(incoming.get(resource_type)).items():
            if not isinstance(metadata, dict):
                continue
            merged.setdefault(resource_type, {})[str(resource_id)] = dict(metadata)
    return merged


def _delivery_resource_descriptor(
    resource_id: str,
    resource_type: str,
    metadata: Any,
    *,
    source_default: str,
) -> dict[str, Any]:
    data = _mapping(metadata)
    descriptor: dict[str, Any] = {
        "resource_id": str(resource_id),
        "resource_type": str(resource_type),
        "display_name": str(data.get("display_name") or data.get("name") or resource_id),
        "category": str(data.get("category") or ""),
        "version": str(data.get("version") or ""),
        "source": str(data.get("source") or source_default),
        "owner": str(data.get("owner") or ""),
        "description": str(data.get("description") or ""),
    }
    for key in (
        "tenant_id",
        "delivery_namespace",
        "customer_id",
        "project_id",
        "site_id",
        "publish_status",
        "release_channel",
        "created_at",
        "updated_at",
        "updated_by",
        "update_reason",
        "disabled_at",
        "disabled_by",
        "disable_reason",
    ):
        value = data.get(key)
        if value not in (None, ""):
            descriptor[key] = value
    return descriptor


def _delivery_resource_catalog_summary(
    resource_catalog: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "resource_type_count": len(DELIVERY_RESOURCE_TYPES),
        "resource_count": sum(len(_mapping(resource_catalog.get(resource_type))) for resource_type in DELIVERY_RESOURCE_TYPES),
        "vision_model_count": len(_mapping(resource_catalog.get("vision_models"))),
        "sensor_protocol_count": len(_mapping(resource_catalog.get("sensor_protocols"))),
        "skill_package_count": len(_mapping(resource_catalog.get("skill_packages"))),
        "acceptance_test_count": len(_mapping(resource_catalog.get("acceptance_tests"))),
    }


def _managed_object_payload(
    object_id: str,
    item: dict[str, Any],
    *,
    resource_catalog: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    bindings = _managed_object_binding_payload(_mapping(item.get("bindings")))
    resource_binding_status = _managed_object_resource_binding_status(
        bindings,
        resource_catalog or _delivery_resource_catalog({}),
    )
    return {
        "object_id": str(object_id),
        "display_name": str(item.get("display_name") or object_id),
        "category": str(item.get("category") or "uncategorized"),
        "object_labels": _string_list(item.get("object_labels")),
        "scenario_ids": _string_list(item.get("scenario_ids")),
        "zone_types": _string_list(item.get("zone_types")),
        "device_sources": _string_list(item.get("device_sources")),
        "tenant_ids": _string_list(item.get("tenant_ids") or item.get("tenant_id")),
        "delivery_namespaces": _string_list(
            item.get("delivery_namespaces") or item.get("delivery_namespace")
        ),
        "customer_ids": _string_list(item.get("customer_ids") or item.get("customer_id")),
        "project_ids": _string_list(item.get("project_ids") or item.get("project_id")),
        "site_ids": _string_list(item.get("site_ids") or item.get("site_id")),
        "responder_group": str(item.get("responder_group") or ""),
        "evidence_required": _string_list(item.get("evidence_required")),
        "bindings": bindings,
        "resource_binding_status": resource_binding_status,
        "acceptance_status": _managed_object_acceptance_status(bindings),
        "customer_visible": bool(item.get("customer_visible", True)),
    }


def _managed_object_binding_payload(bindings: dict[str, Any]) -> dict[str, Any]:
    return {
        "vision_models": _string_list(bindings.get("vision_models")),
        "sensor_protocols": _string_list(bindings.get("sensor_protocols")),
        "skill_packages": _string_list(bindings.get("skill_packages")),
        "acceptance_tests": _string_list(bindings.get("acceptance_tests")),
    }


def _managed_object_execution_binding_plan(
    obj: dict[str, Any],
    profile: dict[str, Any],
    devices_by_source: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    bindings = _mapping(obj.get("bindings"))
    required_sources = _string_list(obj.get("device_sources"))
    checks = _mapping(obj.get("resource_binding_status")).get("checks")
    check_by_key = {
        (str(_mapping(item).get("resource_type") or ""), str(_mapping(item).get("resource_id") or "")): _mapping(item)
        for item in (checks if isinstance(checks, list) else [])
    }
    blockers: list[str] = []
    manual_checks: list[str] = []
    source_plans = []
    for source in required_sources:
        matched_devices = devices_by_source.get(source, [])
        if not matched_devices:
            blockers.append(f"No registered {source} device can feed this object.")
        source_plans.append({
            "source": source,
            "status": "ready" if matched_devices else "blocked",
            "device_count": len(matched_devices),
            "devices": matched_devices,
        })

    adapters = []
    covered_sources: set[str] = set()
    for protocol in _string_list(bindings.get("sensor_protocols")):
        sources = _sensor_protocol_execution_sources(protocol)
        matched_adapter_devices = [
            device
            for source in sorted(set(sources).intersection(required_sources))
            for device in devices_by_source.get(source, [])
        ]
        covered_sources.update(source for source in sources if source in required_sources)
        check = check_by_key.get(("sensor_protocols", protocol), {})
        status = _execution_check_status(check, default="ready")
        if status == "blocked":
            blockers.append(f"Sensor protocol {protocol} is blocked.")
        elif status == "manual_check":
            manual_checks.append(f"Sensor protocol {protocol} requires review.")
        adapters.append({
            "protocol_id": protocol,
            "adapter": _sensor_protocol_adapter_name(protocol),
            "sources": sources,
            "matched_required_sources": sorted(set(sources).intersection(required_sources)),
            "status": status,
            "message": str(check.get("message") or ""),
            "adapter_contract": _field_ingest_adapter_contract(
                protocol,
                sources=sources,
                matched_devices=matched_adapter_devices,
            ),
        })
    for source in required_sources:
        if source == "camera" and _string_list(bindings.get("vision_models")):
            covered_sources.add(source)
        if source not in covered_sources:
            manual_checks.append(f"No explicit protocol covers source {source}.")

    vision_models = [
        _execution_resource_ref("vision_models", model_id, check_by_key)
        for model_id in _string_list(bindings.get("vision_models"))
    ]
    scenario_ids = _string_list(obj.get("scenario_ids"))
    scenario_id = next(iter(scenario_ids), "")
    required_inputs = _scenario_required_inputs_for_object(obj)
    skill_routes = []
    for route in field_capability_routes(
        _string_list(bindings.get("skill_packages")),
        scenario_id=scenario_id,
        required_inputs=required_inputs,
    ):
        skill_routes.append({
            **_execution_resource_ref("skill_packages", str(route.get("package_id") or ""), check_by_key),
            **route,
            "runtime_route": route.get("route"),
            "safety_boundary": route.get("hardware_boundary"),
        })
    acceptance_tests = [
        _execution_resource_ref("acceptance_tests", ref, check_by_key)
        for ref in _string_list(bindings.get("acceptance_tests"))
    ]
    for bucket_name, bucket in (
        ("vision model", vision_models),
        ("skill package", skill_routes),
        ("acceptance test", acceptance_tests),
    ):
        if not bucket:
            blockers.append(f"No {bucket_name} binding is configured.")
        for item in bucket:
            status = str(item.get("status") or "")
            if status == "blocked":
                blockers.append(f"{bucket_name} {item.get('resource_id')} is blocked.")
            elif status == "manual_check":
                manual_checks.append(f"{bucket_name} {item.get('resource_id')} requires review.")
            if bucket_name == "skill package" and not item.get("installed_contract"):
                manual_checks.append(f"skill package {item.get('resource_id')} has no installed executable contract.")

    resource_status = str(_mapping(obj.get("resource_binding_status")).get("overall_status") or "blocked")
    if resource_status == "blocked":
        blockers.append("Resource binding status is blocked.")
    elif resource_status == "manual_check":
        manual_checks.append("Resource binding status requires review.")

    overall_status = "blocked" if blockers else "manual_check" if manual_checks else "ready"
    return {
        "object_id": str(obj.get("object_id") or ""),
        "display_name": str(obj.get("display_name") or obj.get("object_id") or ""),
        "category": str(obj.get("category") or ""),
        "scenario_ids": scenario_ids,
        "overall_status": overall_status,
        "required_sources": required_sources,
        "scope_constraints": {
            "tenant_ids": _string_list(obj.get("tenant_ids")),
            "delivery_namespaces": _string_list(obj.get("delivery_namespaces")),
            "customer_ids": _string_list(obj.get("customer_ids")),
            "project_ids": _string_list(obj.get("project_ids")),
            "site_ids": _string_list(obj.get("site_ids")),
        },
        "source_plans": source_plans,
        "vision_models": vision_models,
        "input_adapters": adapters,
        "skill_routes": skill_routes,
        "acceptance_tests": acceptance_tests,
        "ingest_contract": _managed_object_ingest_contract(obj, profile),
        "runtime_contract": {
            "callback_endpoint": "/api/field/events/{event_id}/runtime-delivery",
            "handoff_boundary": "Field event actions are not direct hardware commands; runtime callbacks must be signed/trusted before they are recorded.",
        },
        "bridge_contract": _managed_object_bridge_contract(required_sources),
        "blockers": sorted(set(blockers)),
        "manual_checks": sorted(set(manual_checks)),
        "customer_status": {
            "ready": "对象绑定已能驱动现场接入：输入源、协议、模型、技能包和验收用例齐备。",
            "manual_check": "对象绑定可以接入，但仍有协议、资源发布状态或验收项需要交付复核。",
            "blocked": "对象绑定不能作为可执行接入，需要先补齐设备、协议、技能包或验收证据。",
        }[overall_status],
    }


def _scenario_required_inputs_for_object(obj: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for scenario_id in _string_list(obj.get("scenario_ids")):
        values.extend(_SCENARIO_REQUIRED_INPUTS.get(scenario_id, ()))
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _managed_object_ingest_contract(obj: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    source = next(iter(_string_list(obj.get("device_sources"))), "camera")
    zone_id = _first_zone_for_object(obj, profile)
    sample: dict[str, Any] = {
        "source": source,
        "observed_at": "ISO-8601 or unix timestamp",
        "zone_id": zone_id,
        "scenario_id": next(iter(_string_list(obj.get("scenario_ids"))), ""),
        "managed_object_id": str(obj.get("object_id") or ""),
    }
    labels = _string_list(obj.get("object_labels"))
    if source == "camera":
        sample["detections"] = [{"label": labels[0] if labels else "object", "confidence": 0.9}]
        sample["image_path"] = "artifacts/evidence/example.jpg"
    elif source == "sensor":
        sample["sensor"] = {"temperature_c": 25, "smoke_level": 0.0}
    elif source == "robot":
        sample["robot_id"] = "robot-id"
        sample["runtime_status"] = "reported"
    return {
        "endpoint": "/api/field/ingest",
        "method": "POST",
        "sample_payload": sample,
        "required_fields": ["source", "observed_at", "zone_id"],
        "normalizer": "askme.pipeline.field_ingest_adapters.normalize_field_ingest_payload",
        "bridge": "field-ingest-bridge",
    }


def _first_zone_for_object(obj: dict[str, Any], profile: dict[str, Any]) -> str:
    zones = _mapping(profile.get("zones"))
    wanted = set(_string_list(obj.get("zone_types")))
    for zone_id, zone in zones.items():
        if str(_mapping(zone).get("type") or "") in wanted:
            return str(zone_id)
    return ""


def _field_devices_by_source(profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    devices: dict[str, list[dict[str, Any]]] = {}
    for device_id, device in _mapping(profile.get("devices")).items():
        if not isinstance(device, dict):
            continue
        source = str(device.get("source") or "").strip()
        if not source:
            continue
        devices.setdefault(source, []).append({
            "device_id": str(device_id),
            "name": str(device.get("name") or device_id),
            "source": source,
            "zone_id": str(device.get("zone_id") or ""),
            "camera_id": str(device.get("camera_id") or ""),
            "sensor_type": str(device.get("sensor_type") or ""),
            "robot_id": str(device.get("robot_id") or ""),
            "secret_configured": bool(device.get("secret_env")),
            "secret_env": str(device.get("secret_env") or ""),
        })
    return devices


def _sensor_protocol_execution_sources(protocol_id: str) -> list[str]:
    text = str(protocol_id or "").lower()
    sources: list[str] = []
    if any(token in text for token in ("camera", "vision", "detection", "video")):
        sources.append("camera")
    if any(token in text for token in ("sensor", "smoke", "temperature", "mqtt", "iot")):
        sources.append("sensor")
    if any(token in text for token in ("robot", "route", "runtime", "status")):
        sources.append("robot")
    if "voice" in text:
        sources.append("robot")
    return sorted(set(sources)) or ["custom"]


def _sensor_protocol_adapter_name(protocol_id: str) -> str:
    sources = _sensor_protocol_execution_sources(protocol_id)
    if sources == ["camera"]:
        return "camera_detection_json"
    if "sensor" in sources:
        return "sensor_telemetry_json"
    if "robot" in sources:
        return "robot_runtime_json"
    return "custom_field_ingest_json"


def _field_ingest_adapter_contract(
    protocol_id: str,
    *,
    sources: list[str],
    matched_devices: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the delivery contract for running a real field ingest adapter."""
    secret_envs = sorted({
        str(device.get("secret_env") or "")
        for device in matched_devices
        if str(device.get("secret_env") or "").strip()
    })
    device_ids = sorted({
        str(device.get("device_id") or "")
        for device in matched_devices
        if str(device.get("device_id") or "").strip()
    })
    return {
        "protocol_id": str(protocol_id or ""),
        "adapter": _sensor_protocol_adapter_name(protocol_id),
        "normalizer": "askme.pipeline.field_ingest_adapters.normalize_field_ingest_payload",
        "bridge": "field-ingest-bridge",
        "bridge_runner": "scripts/runtime/bridges/field_ingest_bridge.py",
        "ingest_endpoint": "/api/field/ingest",
        "supported_formats": ["json", "jsonl", "ndjson"],
        "accepted_sources": sorted(set(_string_list(sources))),
        "matched_device_ids": device_ids,
        "device_signature_required": bool(secret_envs),
        "device_secret_envs": secret_envs,
        "dry_run_command": (
            "python -m askme runtime field-ingest-bridge <device-events.jsonl> "
            "--site-profile <site-profile.yaml> --dry-run --json"
        ),
        "live_command": (
            "python -m askme runtime field-ingest-bridge <device-events.jsonl> "
            "--server http://<askme-host>:8765 --site-profile <site-profile.yaml> --watch"
        ),
        "sign_command": (
            "python -m askme runtime field-sign-device-payload <device-event.json> "
            "--secret-env <DEVICE_SECRET_ENV> --output signed-device-event.json"
        ),
        "sample_fixture": "tests/fixtures/field_devices/site-a-device-events.jsonl",
        "verification_outputs": [
            "summary.posted",
            "summary.accepted",
            "summary.failed",
            "summary.signed",
            "results[].normalized.source",
            "results[].normalized.scenario_id",
            "results[].device_signing.reason",
            "results[].event.event_id",
        ],
        "customer_boundary": (
            "Dry-run proves parsing only. Customer signoff needs live post evidence, "
            "trusted device signatures, event archive entries, and runtime callback evidence."
        ),
    }


def _managed_object_bridge_contract(required_sources: list[str]) -> dict[str, Any]:
    return {
        "bridge": "field-ingest-bridge",
        "ingest_endpoint": "/api/field/ingest",
        "sources": sorted(set(_string_list(required_sources))),
        "dry_run_first": True,
        "live_post_required_for_customer_signoff": True,
        "trusted_device_signature_required_for_production": True,
        "state_file": "<bridge-state.json>",
        "sample_fixture": "tests/fixtures/field_devices/site-a-device-events.jsonl",
        "summary_fields": [
            "processed",
            "posted",
            "accepted",
            "failed",
            "signed",
            "events_created",
            "scenario_counts",
            "source_counts",
            "device_counts",
        ],
    }


def _execution_resource_ref(
    resource_type: str,
    resource_id: str,
    checks: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    check = checks.get((resource_type, resource_id), {})
    return {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "status": _execution_check_status(check, default="ready"),
        "publish_status": str(check.get("publish_status") or ""),
        "message": str(check.get("message") or ""),
        "reference_status": str(check.get("reference_status") or ""),
    }


def _execution_check_status(check: dict[str, Any], *, default: str) -> str:
    status = str(check.get("status") or default)
    if status in {"linked", "ready", "configured"}:
        return "ready"
    if status in {"manual_check", "unregistered", "node_unresolved", "draft", "pilot", "deprecated"}:
        return "manual_check"
    return "blocked"


def _execution_binding_summary(plans: list[dict[str, Any]]) -> dict[str, Any]:
    ready = len([item for item in plans if item.get("overall_status") == "ready"])
    manual = len([item for item in plans if item.get("overall_status") == "manual_check"])
    blocked = len([item for item in plans if item.get("overall_status") == "blocked"])
    overall = "blocked" if blocked else "manual_check" if manual else "ready" if plans else "blocked"
    return {
        "overall_status": overall,
        "object_count": len(plans),
        "ready_object_count": ready,
        "manual_check_object_count": manual,
        "blocked_object_count": blocked,
    }


def _execution_binding_customer_claim(summary: dict[str, Any]) -> str:
    status = str(summary.get("overall_status") or "blocked")
    if status == "ready":
        return "客户项目对象绑定已形成可执行接入计划，可进入现场接入验证。"
    if status == "manual_check":
        return "客户项目对象绑定已有接入计划，但仍需交付复核后才能对客户承诺。"
    return "客户项目对象绑定仍有阻断项，不能承诺真实现场接入。"


def _execution_binding_next_step(summary: dict[str, Any]) -> str:
    status = str(summary.get("overall_status") or "blocked")
    if status == "ready":
        return "使用每个对象的 ingest 示例接入真实设备 payload，并记录 runtime 回传。"
    if status == "manual_check":
        return "复核未覆盖的输入源、试点/草稿资源和验收用例后再接入现场。"
    return "先补齐缺失设备、资源注册、协议绑定或技能包绑定。"


def _managed_object_resource_binding_status(
    bindings: dict[str, Any],
    resource_catalog: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blocked_count = 0
    manual_check_count = 0
    linked_count = 0
    for resource_type in DELIVERY_RESOURCE_TYPES:
        values = _string_list(bindings.get(resource_type))
        if not values:
            checks.append({
                "resource_type": resource_type,
                "resource_id": "",
                "status": "missing",
                "message": f"{resource_type} binding is required.",
            })
            blocked_count += 1
            continue
        for value in values:
            if resource_type == "acceptance_tests":
                resource = _mapping(resource_catalog.get(resource_type)).get(value)
                link_status = (
                    _delivery_resource_link_status(_mapping(resource))
                    if resource
                    else {"bucket": "linked", "message": ""}
                )
                check = _acceptance_test_check(value)
                bucket = _acceptance_resource_bucket(str(check.get("status") or "unknown"))
                if str(link_status.get("bucket") or "linked") != "linked":
                    bucket = str(link_status.get("bucket") or bucket)
                checks.append({
                    "resource_type": resource_type,
                    "resource_id": value,
                    "status": bucket,
                    "reference_status": str(check.get("status") or "unknown"),
                    "source": str(check.get("resolved_by") or ""),
                    "publish_status": str(_mapping(resource).get("publish_status") or ""),
                    "message": str(link_status.get("message") or check.get("message") or ""),
                })
            else:
                resource = _mapping(resource_catalog.get(resource_type)).get(value)
                if resource:
                    link_status = _delivery_resource_link_status(_mapping(resource))
                    bucket = str(link_status.get("bucket") or "linked")
                    checks.append({
                        "resource_type": resource_type,
                        "resource_id": value,
                        "status": bucket,
                        "display_name": str(resource.get("display_name") or value),
                        "version": str(resource.get("version") or ""),
                        "source": str(resource.get("source") or ""),
                        "category": str(resource.get("category") or ""),
                        "publish_status": str(resource.get("publish_status") or ""),
                        "message": str(link_status.get("message") or ""),
                    })
                else:
                    bucket = "unregistered"
                    checks.append({
                        "resource_type": resource_type,
                        "resource_id": value,
                        "status": bucket,
                        "message": "Resource is not in the delivery resource catalog.",
                    })
            if bucket == "linked":
                linked_count += 1
            elif bucket in {"manual_check", "unregistered"}:
                manual_check_count += 1
            else:
                blocked_count += 1
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check"
        if manual_check_count
        else "ready"
    )
    return {
        "overall_status": overall_status,
        "linked_count": linked_count,
        "manual_check_count": manual_check_count,
        "blocked_count": blocked_count,
        "check_count": len(checks),
        "checks": checks,
        "customer_status": {
            "ready": "All object resources are linked to the delivery catalog.",
            "manual_check": "Some object resources need catalog review.",
            "blocked": "Some required object resources are missing or invalid.",
        }[overall_status],
    }


def _delivery_resource_link_status(resource: dict[str, Any]) -> dict[str, str]:
    publish_status = str(resource.get("publish_status") or "published").strip()
    if publish_status in {"disabled", "blocked"}:
        return {
            "bucket": "blocked",
            "message": f"Resource is {publish_status} and cannot be used for customer delivery.",
        }
    if publish_status in {"draft", "pilot", "deprecated"}:
        return {
            "bucket": "manual_check",
            "message": f"Resource is {publish_status}; delivery owner review is required.",
        }
    return {"bucket": "linked", "message": "Resource is available for delivery binding."}


def _acceptance_resource_bucket(status: str) -> str:
    if status in {"linked"}:
        return "linked"
    if status in {"node_unresolved", "read_error"}:
        return "manual_check"
    return "blocked"


def _managed_object_acceptance_status(bindings: dict[str, Any]) -> dict[str, Any]:
    requirements = []
    for key, label in (
        ("vision_models", "Vision model"),
        ("sensor_protocols", "Sensor protocol"),
        ("skill_packages", "Skill package"),
        ("acceptance_tests", "Acceptance test"),
    ):
        values = _string_list(bindings.get(key))
        requirements.append({
            "key": key,
            "label": label,
            "status": "configured" if values else "missing",
            "count": len(values),
            "items": values,
        })
    missing = [item["key"] for item in requirements if item["status"] == "missing"]
    acceptance_checks = [
        _acceptance_test_check(reference)
        for reference in _string_list(bindings.get("acceptance_tests"))
    ]
    blocked_checks = [
        item
        for item in acceptance_checks
        if item.get("status") in {"file_missing", "invalid_reference", "outside_project"}
    ]
    manual_checks = [
        item
        for item in acceptance_checks
        if item.get("status") in {"node_unresolved", "read_error"}
    ]
    if not missing and not blocked_checks and not manual_checks:
        status = "ready"
        customer_status = "Acceptance evidence linked"
        next_step = "Run this object's acceptance tests against the customer site."
    elif blocked_checks:
        status = "blocked"
        customer_status = "Acceptance test evidence missing"
        next_step = "Fix missing or unsafe acceptance test references before customer acceptance."
    elif manual_checks:
        status = "manual_check"
        customer_status = "Acceptance test evidence needs review"
        next_step = "Resolve the acceptance test node or add an explicit scenario alias."
    elif len(missing) == len(requirements):
        status = "blocked"
        customer_status = "Acceptance bindings missing"
        next_step = "Bind vision models, sensor protocols, skill packages, and acceptance tests."
    else:
        status = "manual_check"
        customer_status = "Acceptance bindings incomplete"
        next_step = "Complete missing bindings before customer acceptance."
    return {
        "status": status,
        "customer_status": customer_status,
        "missing": missing,
        "requirements": requirements,
        "acceptance_checks": acceptance_checks,
        "next_step": next_step,
    }


def _acceptance_test_check(reference: str) -> dict[str, Any]:
    ref = str(reference or "").strip()
    path_text, separator, node = ref.partition("::")
    if not ref or not path_text:
        return {
            "reference": ref,
            "status": "invalid_reference",
            "path": path_text,
            "node": node,
            "message": "Acceptance reference must use a local test file path.",
        }
    path = Path(path_text)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        resolved = resolved.resolve()
    except OSError as exc:
        return {
            "reference": ref,
            "status": "file_missing",
            "path": path_text,
            "node": node,
            "message": str(exc),
        }
    if not resolved.is_relative_to(PROJECT_ROOT):
        return {
            "reference": ref,
            "status": "outside_project",
            "path": path_text,
            "node": node,
            "message": "Acceptance test evidence must stay inside the project repository.",
        }
    if not resolved.exists() or not resolved.is_file():
        return {
            "reference": ref,
            "status": "file_missing",
            "path": path_text,
            "node": node,
            "message": "Acceptance test file was not found.",
        }
    if not separator:
        return {
            "reference": ref,
            "status": "linked",
            "path": path_text,
            "node": "",
            "resolved_by": "file",
            "matched": resolved.name,
        }
    try:
        text = resolved.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {
            "reference": ref,
            "status": "read_error",
            "path": path_text,
            "node": node,
            "message": str(exc),
        }
    match = _acceptance_node_match(text, node)
    if match:
        return {
            "reference": ref,
            "status": "linked",
            "path": path_text,
            "node": node,
            "resolved_by": match["resolved_by"],
            "matched": match["matched"],
        }
    return {
        "reference": ref,
        "status": "node_unresolved",
        "path": path_text,
        "node": node,
        "message": "Acceptance test file exists, but the referenced node or scenario alias was not found.",
    }


def _acceptance_node_match(text: str, node: str) -> dict[str, str] | None:
    node = str(node or "").strip()
    pytest_candidates = [node, f"test_{node}"] if node else []
    for candidate in [item for item in pytest_candidates if item]:
        if re.search(rf"\bdef\s+{re.escape(candidate)}\s*\(", text):
            return {"resolved_by": "pytest_node", "matched": candidate}
    for candidate in _ACCEPTANCE_TEST_ALIASES.get(node, ()):
        if candidate in text:
            return {"resolved_by": "scenario_alias", "matched": candidate}
    if node and node in text:
        return {"resolved_by": "literal", "matched": node}
    return None


def _acceptance_registry_consumers_from_profile(path: Path, *, scope_type: str) -> list[dict[str, Any]]:
    try:
        profile = load_field_site_profile(path)
    except Exception as exc:
        return [{
            "scope_type": scope_type,
            "profile_path": str(path),
            "reference": "",
            "status": "read_error",
            "message": str(exc),
        }]
    customer = _customer_payload(profile)
    site = _mapping(profile.get("site"))
    delivery_scope = _delivery_scope_payload_from_customer_site(customer, site)
    template = _mapping(profile.get("template"))
    objects = managed_object_catalog_from_site_profile(profile).get("objects")
    rows: list[dict[str, Any]] = []
    for item in objects if isinstance(objects, list) else []:
        bindings = _mapping(item.get("bindings"))
        references = _string_list(bindings.get("acceptance_tests"))
        if not references:
            rows.append({
                "scope_type": scope_type,
                "profile_path": str(path),
                "template_id": str(template.get("template_id") or path.stem) if scope_type == "template" else "",
                "tenant_id": delivery_scope["tenant_id"],
                "delivery_namespace": delivery_scope["delivery_namespace"],
                "customer_id": str(customer.get("customer_id") or ""),
                "customer_name": str(customer.get("customer_name") or ""),
                "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
                "project_name": str(customer.get("project_name") or site.get("name") or ""),
                "site_id": str(site.get("site_id") or ""),
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or ""),
                "reference": "",
                "status": "missing",
                "message": "Managed object has no acceptance test reference.",
            })
            continue
        for reference in references:
            check = _acceptance_test_check(reference)
            rows.append({
                "scope_type": scope_type,
                "profile_path": str(path),
                "template_id": str(template.get("template_id") or path.stem) if scope_type == "template" else "",
                "tenant_id": delivery_scope["tenant_id"],
                "delivery_namespace": delivery_scope["delivery_namespace"],
                "customer_id": str(customer.get("customer_id") or ""),
                "customer_name": str(customer.get("customer_name") or ""),
                "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
                "project_name": str(customer.get("project_name") or site.get("name") or ""),
                "site_id": str(site.get("site_id") or ""),
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or ""),
                "reference": reference,
                "status": str(check.get("status") or "unknown"),
                "path": str(check.get("path") or ""),
                "node": str(check.get("node") or ""),
                "resolved_by": str(check.get("resolved_by") or ""),
                "matched": str(check.get("matched") or ""),
                "message": str(check.get("message") or ""),
            })
    return rows


def _acceptance_registry_references(consumers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for consumer in consumers:
        reference = str(consumer.get("reference") or "").strip()
        if not reference:
            continue
        row = rows.setdefault(
            reference,
            {
                "reference": reference,
                "status": "linked",
                "consumer_count": 0,
                "linked_count": 0,
                "manual_check_count": 0,
                "blocked_count": 0,
                "consumers": [],
            },
        )
        row["consumer_count"] += 1
        status = str(consumer.get("status") or "unknown")
        bucket = _acceptance_registry_status_bucket(status)
        row[f"{bucket}_count"] += 1
        row["status"] = _merge_acceptance_registry_status(str(row.get("status") or "linked"), status)
        row["consumers"].append({
            "scope_type": consumer.get("scope_type"),
            "tenant_id": consumer.get("tenant_id"),
            "delivery_namespace": consumer.get("delivery_namespace"),
            "customer_id": consumer.get("customer_id"),
            "project_id": consumer.get("project_id"),
            "site_id": consumer.get("site_id"),
            "template_id": consumer.get("template_id"),
            "object_id": consumer.get("object_id"),
            "display_name": consumer.get("display_name"),
            "category": consumer.get("category"),
            "status": status,
            "resolved_by": consumer.get("resolved_by"),
            "matched": consumer.get("matched"),
        })
    return sorted(rows.values(), key=lambda item: (item["status"], item["reference"]))


def _acceptance_registry_summary(
    consumers: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {"linked": 0, "manual_check": 0, "blocked": 0}
    project_ids = set()
    template_ids = set()
    object_ids = set()
    for consumer in consumers:
        status = str(consumer.get("status") or "unknown")
        counts[_acceptance_registry_status_bucket(status)] += 1
        if consumer.get("project_id"):
            project_ids.add(str(consumer.get("project_id")))
        if consumer.get("template_id"):
            template_ids.add(str(consumer.get("template_id")))
        if consumer.get("object_id"):
            object_ids.add(str(consumer.get("object_id")))
    if not consumers or counts["blocked"]:
        overall = "blocked"
    elif counts["manual_check"]:
        overall = "manual_check"
    else:
        overall = "ready"
    return {
        "overall_status": overall,
        "reference_count": len(references),
        "consumer_count": len(consumers),
        "linked_count": counts["linked"],
        "manual_check_count": counts["manual_check"],
        "blocked_count": counts["blocked"],
        "project_count": len(project_ids),
        "template_count": len(template_ids),
        "object_count": len(object_ids),
    }


def _acceptance_registry_status_bucket(status: str) -> str:
    if status in {"linked", "passed", "configured"}:
        return "linked"
    if status in {"node_unresolved", "read_error", "manual_check", "not_run"}:
        return "manual_check"
    return "blocked"


def _merge_acceptance_registry_status(current: str, candidate: str) -> str:
    current_bucket = _acceptance_registry_status_bucket(current)
    candidate_bucket = _acceptance_registry_status_bucket(candidate)
    if "blocked" in {current_bucket, candidate_bucket}:
        return "blocked"
    if "manual_check" in {current_bucket, candidate_bucket}:
        return "manual_check"
    return "linked"


def _acceptance_registry_next_step(summary: dict[str, Any]) -> str:
    if summary.get("blocked_count"):
        return "Fix blocked acceptance references before customer signoff."
    if summary.get("manual_check_count"):
        return "Review unresolved acceptance nodes and convert them into scenario aliases or real pytest nodes."
    if summary.get("consumer_count"):
        return "Run onsite acceptance and attach live evidence to each customer project."
    return "Add acceptance tests to managed objects before customer delivery."


def _managed_object_binding_readiness_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"ready": 0, "manual_check": 0, "blocked": 0}
    unregistered: list[dict[str, str]] = []
    for item in objects:
        status_payload = _mapping(item.get("resource_binding_status"))
        status = str(status_payload.get("overall_status") or "blocked")
        counts[status if status in counts else "manual_check"] += 1
        for check in status_payload.get("checks", []) if isinstance(status_payload.get("checks"), list) else []:
            if not isinstance(check, dict) or check.get("status") != "unregistered":
                continue
            unregistered.append({
                "object_id": str(item.get("object_id") or ""),
                "resource_type": str(check.get("resource_type") or ""),
                "resource_id": str(check.get("resource_id") or ""),
            })
    if not objects or counts["blocked"]:
        overall_status = "blocked"
    elif counts["manual_check"]:
        overall_status = "manual_check"
    else:
        overall_status = "ready"
    return {
        "overall_status": overall_status,
        "ready_object_count": counts["ready"],
        "manual_check_object_count": counts["manual_check"],
        "blocked_object_count": counts["blocked"],
        "object_count": len(objects),
        "unregistered_resource_count": len(unregistered),
        "unregistered_resources": unregistered[:20],
    }


def _delivery_resource_consumers_from_profile(path: Path, *, scope_type: str) -> list[dict[str, Any]]:
    try:
        profile = load_field_site_profile(path)
    except Exception as exc:
        return [{
            "scope_type": scope_type,
            "profile_path": str(path),
            "resource_type": "",
            "resource_id": "",
            "status": "read_error",
            "message": str(exc),
        }]
    customer = _customer_payload(profile)
    site = _mapping(profile.get("site"))
    delivery_scope = _delivery_scope_payload_from_customer_site(customer, site)
    template = _mapping(profile.get("template"))
    resource_catalog = _delivery_resource_catalog(profile)
    objects = managed_object_catalog_from_site_profile(profile).get("objects")
    rows: list[dict[str, Any]] = []
    for item in objects if isinstance(objects, list) else []:
        binding_status = _managed_object_resource_binding_status(
            _mapping(item.get("bindings")),
            resource_catalog,
        )
        for check in binding_status.get("checks", []) if isinstance(binding_status.get("checks"), list) else []:
            if not isinstance(check, dict):
                continue
            resource_type = str(check.get("resource_type") or "")
            resource_id = str(check.get("resource_id") or "")
            if not resource_type or not resource_id:
                continue
            rows.append({
                "scope_type": scope_type,
                "profile_path": str(path),
                "template_id": str(template.get("template_id") or path.stem) if scope_type == "template" else "",
                "tenant_id": delivery_scope["tenant_id"],
                "delivery_namespace": delivery_scope["delivery_namespace"],
                "customer_id": str(customer.get("customer_id") or ""),
                "customer_name": str(customer.get("customer_name") or ""),
                "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
                "project_name": str(customer.get("project_name") or site.get("name") or ""),
                "site_id": str(site.get("site_id") or ""),
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or ""),
                "resource_type": resource_type,
                "resource_id": resource_id,
                "status": str(check.get("status") or "unknown"),
                "source": str(check.get("source") or ""),
                "message": str(check.get("message") or ""),
            })
    return rows


def _delivery_resource_rows(
    resource_catalog: dict[str, dict[str, dict[str, Any]]],
    consumers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for resource_type in DELIVERY_RESOURCE_TYPES:
        for resource_id, metadata in _mapping(resource_catalog.get(resource_type)).items():
            if not isinstance(metadata, dict):
                continue
            row = dict(metadata)
            row.setdefault("resource_id", str(resource_id))
            row.setdefault("resource_type", resource_type)
            row.update({
                "resource_id": str(resource_id),
                "resource_type": resource_type,
                "status": "registered",
                "consumer_count": 0,
                "project_count": 0,
                "template_count": 0,
                "unregistered_consumer_count": 0,
                "consumers": [],
            })
            rows[(resource_type, str(resource_id))] = row
    for consumer in consumers:
        resource_type = str(consumer.get("resource_type") or "")
        resource_id = str(consumer.get("resource_id") or "")
        if not resource_type or not resource_id:
            continue
        key = (resource_type, resource_id)
        inferred_registered = (
            resource_type == "acceptance_tests"
            and consumer.get("status") not in {"unregistered", "blocked", "missing"}
        )
        row = rows.setdefault(
            key,
            {
                "resource_id": resource_id,
                "resource_type": resource_type,
                "display_name": resource_id,
                "category": "",
                "version": "",
                "source": "acceptance_reference" if inferred_registered else "unregistered",
                "status": "registered" if inferred_registered else "unregistered",
                "consumer_count": 0,
                "project_count": 0,
                "template_count": 0,
                "unregistered_consumer_count": 0,
                "consumers": [],
            },
        )
        row["consumer_count"] = int(row.get("consumer_count") or 0) + 1
        if consumer.get("scope_type") == "template":
            row["template_count"] = int(row.get("template_count") or 0) + 1
        else:
            row["project_count"] = int(row.get("project_count") or 0) + 1
        if consumer.get("status") == "unregistered":
            row["unregistered_consumer_count"] = int(row.get("unregistered_consumer_count") or 0) + 1
            row["status"] = "unregistered"
        row["consumers"].append(consumer)
    return sorted(rows.values(), key=lambda item: (str(item.get("resource_type") or ""), str(item.get("resource_id") or "")))


def _delivery_resource_registry_summary(
    resources: list[dict[str, Any]],
    consumers: list[dict[str, Any]],
) -> dict[str, Any]:
    unregistered = [item for item in resources if item.get("status") == "unregistered"]
    used = [item for item in resources if int(item.get("consumer_count") or 0) > 0]
    return {
        "overall_status": "manual_check" if unregistered else "ready",
        "resource_count": len(resources),
        "used_resource_count": len(used),
        "consumer_count": len(consumers),
        "unregistered_resource_count": len(unregistered),
        "project_consumer_count": len([item for item in consumers if item.get("scope_type") == "project"]),
        "template_consumer_count": len([item for item in consumers if item.get("scope_type") == "template"]),
        "by_type": {
            resource_type: {
                "resource_count": len([item for item in resources if item.get("resource_type") == resource_type]),
                "used_resource_count": len([
                    item
                    for item in resources
                    if item.get("resource_type") == resource_type and int(item.get("consumer_count") or 0) > 0
                ]),
                "unregistered_resource_count": len([
                    item
                    for item in resources
                    if item.get("resource_type") == resource_type and item.get("status") == "unregistered"
                ]),
            }
            for resource_type in DELIVERY_RESOURCE_TYPES
        },
    }


def _delivery_resource_registry_next_step(summary: dict[str, Any]) -> str:
    if summary.get("unregistered_resource_count"):
        return "Register missing model, protocol, skill, or acceptance resources before customer signoff."
    if summary.get("consumer_count"):
        return "Use this catalog to audit object bindings before exporting a customer project package."
    return "Bind managed objects to product resources before project delivery."


def _managed_object_acceptance_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = {"ready": 0, "manual_check": 0, "blocked": 0}
    for item in objects:
        status = str(_mapping(item.get("acceptance_status")).get("status") or "blocked")
        status_counts[status if status in status_counts else "manual_check"] += 1
    if not objects or status_counts["blocked"]:
        overall_status = "blocked"
    elif status_counts["manual_check"]:
        overall_status = "manual_check"
    else:
        overall_status = "ready"
    return {
        "overall_status": overall_status,
        "ready_object_count": status_counts["ready"],
        "manual_check_object_count": status_counts["manual_check"],
        "blocked_object_count": status_counts["blocked"],
        "object_count": len(objects),
    }


def _customer_project_package_acceptance_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(catalog.get("acceptance_summary"))
    objects = catalog.get("objects") if isinstance(catalog.get("objects"), list) else []
    blocked = []
    manual = []
    for item in objects:
        status = str(_mapping(item.get("acceptance_status")).get("status") or "blocked")
        row = {
            "object_id": str(item.get("object_id") or ""),
            "display_name": str(item.get("display_name") or item.get("object_id") or ""),
            "status": status,
            "next_step": str(_mapping(item.get("acceptance_status")).get("next_step") or ""),
        }
        if status == "blocked":
            blocked.append(row)
        elif status == "manual_check":
            manual.append(row)
    overall = str(summary.get("overall_status") or "blocked")
    return {
        "overall_status": overall,
        "ready_object_count": int(summary.get("ready_object_count") or 0),
        "manual_check_object_count": int(summary.get("manual_check_object_count") or 0),
        "blocked_object_count": int(summary.get("blocked_object_count") or 0),
        "object_count": int(summary.get("object_count") or len(objects)),
        "blocked_objects": blocked,
        "manual_check_objects": manual,
        "customer_status": {
            "ready": "Local acceptance evidence is linked; onsite acceptance is still required.",
            "manual_check": "Some acceptance evidence requires delivery review before customer signoff.",
            "blocked": "Acceptance evidence is missing or unsafe; do not claim delivery readiness.",
        }.get(overall, "Acceptance status is unknown; review the package before delivery."),
        "release_claim": (
            "Do not claim production launch. This package only carries local delivery evidence "
            "until real devices, notifications, voice, and robot runtime pass onsite acceptance."
        ),
    }


def _normalize_customer_project_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a profile with explicit delivery scope defaults for product handoff."""
    result = copy.deepcopy(profile)
    customer = dict(_mapping(result.get("customer")))
    customer["tenant_id"] = _delivery_tenant_id(customer)
    customer["delivery_namespace"] = _delivery_namespace(customer)
    customer.setdefault("delivery_model", "solution_project")
    result["customer"] = customer
    return result


def _delivery_tenant_id(customer: dict[str, Any]) -> str:
    return _non_empty_text(
        customer.get("tenant_id")
        or customer.get("organization_id")
        or customer.get("org_id")
        or DEFAULT_DELIVERY_NAMESPACE
    )


def _delivery_namespace(customer: dict[str, Any]) -> str:
    return _non_empty_text(
        customer.get("delivery_namespace")
        or customer.get("tenant_namespace")
        or customer.get("namespace")
        or _delivery_tenant_id(customer)
    )


def _delivery_scope_payload(profile: dict[str, Any]) -> dict[str, str]:
    return _delivery_scope_payload_from_customer_site(
        _mapping(profile.get("customer")),
        _mapping(profile.get("site")),
    )


def _delivery_scope_payload_from_customer_site(
    customer: dict[str, Any],
    site: dict[str, Any],
) -> dict[str, str]:
    tenant_id = _delivery_tenant_id(customer)
    delivery_namespace = _delivery_namespace(customer)
    return {
        "tenant_id": tenant_id,
        "delivery_namespace": delivery_namespace,
        "customer_id": str(customer.get("customer_id") or ""),
        "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
        "site_id": str(site.get("site_id") or ""),
    }


def _customer_payload(profile: dict[str, Any]) -> dict[str, str]:
    customer = _mapping(profile.get("customer"))
    site = _mapping(profile.get("site"))
    delivery_scope = _delivery_scope_payload_from_customer_site(customer, site)
    return {
        "tenant_id": delivery_scope["tenant_id"],
        "delivery_namespace": delivery_scope["delivery_namespace"],
        "customer_id": str(customer.get("customer_id") or ""),
        "customer_name": str(customer.get("customer_name") or "Unassigned customer"),
        "industry": str(customer.get("industry") or "unspecified"),
        "project_id": delivery_scope["project_id"],
        "project_name": str(customer.get("project_name") or site.get("name") or ""),
        "delivery_model": str(customer.get("delivery_model") or "solution_project"),
        "object_scope_note": str(customer.get("object_scope_note") or ""),
    }


def _customer_project_summary(sites: list[dict[str, Any]]) -> dict[str, Any]:
    customers: set[str] = set()
    projects: set[str] = set()
    tenants: set[str] = set()
    namespaces: set[str] = set()
    industries: set[str] = set()
    object_categories: set[str] = set()
    scenario_ids: set[str] = set()
    object_count = 0
    for site in sites:
        customer = _mapping(site.get("customer"))
        customer_id = str(customer.get("customer_id") or "")
        project_id = str(customer.get("project_id") or "")
        tenant_id = str(customer.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE)
        delivery_namespace = str(customer.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE)
        industry = str(customer.get("industry") or "")
        tenants.add(tenant_id)
        namespaces.add(delivery_namespace)
        if customer_id:
            customers.add(customer_id)
        if project_id:
            projects.add(project_id)
        if industry:
            industries.add(industry)
        summary = _mapping(site.get("managed_objects_summary"))
        object_count += int(summary.get("object_type_count") or 0)
        object_categories.update(str(item) for item in summary.get("categories") or [] if item)
        scenario_ids.update(str(item) for item in summary.get("scenario_ids") or [] if item)
    return {
        "customer_count": len(customers),
        "project_count": len(projects),
        "tenant_count": len(tenants),
        "delivery_namespace_count": len(namespaces),
        "industry_count": len(industries),
        "managed_object_type_count": object_count,
        "managed_object_categories": sorted(object_categories),
        "covered_scenario_count": len(scenario_ids),
    }


def _customer_rows(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for project in projects:
        customer_id = str(project.get("customer_id") or "unassigned")
        tenant_id = str(project.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE)
        delivery_namespace = str(project.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE)
        row_key = f"{tenant_id}/{delivery_namespace}/{customer_id}"
        row = rows.setdefault(
            row_key,
            {
                "tenant_id": tenant_id,
                "delivery_namespace": delivery_namespace,
                "customer_id": customer_id,
                "customer_name": project.get("customer_name") or "Unassigned customer",
                "industries": set(),
                "project_count": 0,
                "site_count": 0,
                "managed_object_type_count": 0,
            },
        )
        row["industries"].add(project.get("industry") or "unspecified")
        row["project_count"] += 1
        row["site_count"] += 1
        summary = _mapping(project.get("managed_objects_summary"))
        row["managed_object_type_count"] += int(summary.get("object_type_count") or 0)
    return [
        {
            **row,
            "industries": sorted(row["industries"]),
        }
        for row in rows.values()
    ]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item or "").strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def find_site_profile_path(root: Path, identifier: str) -> Path | None:
    """Find a site profile by site_id, project_id, or filename stem."""
    target = str(identifier or "").strip()
    if not target:
        return None
    for path in _site_profile_paths(Path(root), pattern="*.yaml"):
        try:
            profile = load_field_site_profile(path)
        except Exception:
            continue
        site = _mapping(profile.get("site"))
        customer = _mapping(profile.get("customer"))
        scope = _delivery_scope_payload_from_customer_site(customer, site)
        candidates = {
            str(customer.get("customer_id") or ""),
            str(site.get("site_id") or ""),
            str(customer.get("project_id") or ""),
            path.stem,
        }
        candidates.update(_delivery_identifier_candidates(scope))
        if target in candidates:
            return path
    return None


def _find_template_path(root: Path, template_id: str) -> Path | None:
    target = str(template_id or "").strip()
    if not target:
        return None
    for path in _site_profile_paths(Path(root), pattern="*.yaml"):
        try:
            profile = load_field_site_profile(path)
        except Exception:
            continue
        template = _mapping(profile.get("template"))
        if target in {str(template.get("template_id") or ""), path.stem}:
            return path
    return None


def _is_semver(value: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?$", str(value or "")))


def _customer_profile_path(profile_root: Path, profile: dict[str, Any]) -> Path:
    customer = _mapping(profile.get("customer"))
    site = _mapping(profile.get("site"))
    scope = _delivery_scope_payload_from_customer_site(customer, site)
    customer_id = _slug(customer.get("customer_id") or "unassigned")
    project_id = _slug(customer.get("project_id") or site.get("site_id") or "project")
    root = Path(profile_root)
    tenant = _slug(scope.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE)
    namespace = _slug(scope.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE)
    if tenant != DEFAULT_DELIVERY_NAMESPACE or namespace != DEFAULT_DELIVERY_NAMESPACE:
        root = root / tenant
        if namespace != tenant:
            root = root / namespace
    return root / customer_id / f"{project_id}.yaml"


def _customer_profile_target(profile_root: Path, profile: dict[str, Any]) -> Path:
    existing = _find_customer_project_profile_path(profile_root, profile)
    if existing is not None:
        return existing
    return _customer_profile_path(profile_root, profile)


def _find_customer_project_profile_path(profile_root: Path, profile: dict[str, Any]) -> Path | None:
    incoming_scope = _delivery_scope_payload(profile)
    for path in _site_profile_paths(Path(profile_root), pattern="*.yaml"):
        try:
            current = load_field_site_profile(path)
        except Exception:
            continue
        current_scope = _delivery_scope_payload(current)
        if _same_delivery_project_scope(current_scope, incoming_scope):
            return path
    return None


def _customer_project_collision_candidates(profile_root: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    incoming_scope = _delivery_scope_payload(profile)
    collisions: list[dict[str, Any]] = []
    for path in _site_profile_paths(Path(profile_root), pattern="*.yaml"):
        try:
            current = load_field_site_profile(path)
        except Exception:
            continue
        current_scope = _delivery_scope_payload(current)
        if _same_customer_project_identity(current_scope, incoming_scope) and not _same_delivery_project_scope(
            current_scope,
            incoming_scope,
        ):
            collisions.append({"profile_path": str(path), "delivery_scope": current_scope})
    return collisions


def _same_delivery_project_scope(current: dict[str, str], incoming: dict[str, str]) -> bool:
    if str(current.get("tenant_id") or "") != str(incoming.get("tenant_id") or ""):
        return False
    if str(current.get("delivery_namespace") or "") != str(incoming.get("delivery_namespace") or ""):
        return False
    return _same_customer_project_identity(current, incoming)


def _same_customer_project_identity(current: dict[str, str], incoming: dict[str, str]) -> bool:
    current_customer = str(current.get("customer_id") or "")
    incoming_customer = str(incoming.get("customer_id") or "")
    current_project = str(current.get("project_id") or "")
    incoming_project = str(incoming.get("project_id") or "")
    current_site = str(current.get("site_id") or "")
    incoming_site = str(incoming.get("site_id") or "")
    if current_customer and incoming_customer and current_customer == incoming_customer:
        if current_project and incoming_project and current_project == incoming_project:
            return True
        if current_site and incoming_site and current_site == incoming_site:
            return True
    if current_project and incoming_project and current_site and incoming_site:
        return current_project == incoming_project and current_site == incoming_site
    return False


def _delivery_identifier_candidates(scope: dict[str, str]) -> set[str]:
    tenant_id = str(scope.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE)
    delivery_namespace = str(scope.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE)
    customer_id = str(scope.get("customer_id") or "")
    project_id = str(scope.get("project_id") or "")
    site_id = str(scope.get("site_id") or "")
    candidates: set[str] = set()
    for identifier in (project_id, site_id):
        if not identifier:
            continue
        candidates.update({
            f"{tenant_id}:{identifier}",
            f"{delivery_namespace}:{identifier}",
            f"{tenant_id}/{identifier}",
            f"{delivery_namespace}/{identifier}",
            f"{tenant_id}/{delivery_namespace}/{identifier}",
        })
        if customer_id:
            candidates.update({
                f"{tenant_id}/{customer_id}/{identifier}",
                f"{tenant_id}/{delivery_namespace}/{customer_id}/{identifier}",
            })
    return candidates


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
    profile_root: Path = Path("deploy/site-profiles"),
    template_root: Path | None = Path("deploy/customer-project-templates"),
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
    profile_root: Path = Path("deploy/site-profiles"),
    template_root: Path | None = Path("deploy/customer-project-templates"),
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
        catalog = build_customer_project_resource_catalog(
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


def _delivery_resource_registry_path(resource_root: Path) -> Path:
    return Path(resource_root) / "resources.yaml"


def _write_delivery_resource_registry(
    resource_root: Path,
    resources: dict[str, Any],
    *,
    operator_id: str = "",
) -> None:
    now = time.time()
    _write_yaml(
        _delivery_resource_registry_path(resource_root),
        {
            "registry_type": "askme.delivery_resource_registry",
            "registry_version": 1,
            "updated_at": now,
            "updated_by": str(operator_id or ""),
            "delivery_resources": resources,
        },
    )


def _snapshot_delivery_resource_registry_revision(
    resource_root: Path,
    registry: dict[str, Any],
    *,
    action: str,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Save the shared delivery-resource registry before a write."""
    payload = {
        "registry_type": "askme.delivery_resource_registry",
        "registry_version": int(registry.get("registry_version") or 1),
        "delivery_resources": _mapping(registry.get("delivery_resources")),
    }
    registry_hash = _sha256_json(payload)
    created_at = time.time()
    revision_id = _slug(
        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created_at))}-"
        f"{int(created_at * 1000)}-{action}-{registry_hash[:12]}"
    )
    revision = {
        "revision_type": "askme.delivery_resource_registry_revision",
        "revision_version": 1,
        "revision_id": revision_id,
        "created_at": created_at,
        "action": str(action or "resource_write"),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or ""),
        "registry_path": str(_delivery_resource_registry_path(resource_root)),
        "registry_sha256": registry_hash,
        "registry": payload,
    }
    target_dir = Path(resource_root) / "_resource_revisions"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{revision_id}.json"
    target.write_text(json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8")
    revision["revision_path"] = str(target)
    return revision


def _find_delivery_resource_revision(
    resource_root: Path,
    revision_id: str,
) -> dict[str, Any] | None:
    target_id = str(revision_id or "").strip()
    if not target_id:
        return None
    revision_dir = Path(resource_root) / "_resource_revisions"
    if not revision_dir.exists():
        return None
    for item in revision_dir.glob("*.json"):
        payload = _read_delivery_resource_revision_file(item)
        if payload and str(payload.get("revision_id") or "") == target_id:
            return payload
    return None


def _read_delivery_resource_revision_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("revision_type") != "askme.delivery_resource_registry_revision":
        return {}
    payload["revision_path"] = str(path)
    return payload


def _delivery_resource_revision_public_payload(revision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(revision, dict) or not revision:
        return {}
    return {
        "revision_id": str(revision.get("revision_id") or ""),
        "created_at": revision.get("created_at"),
        "action": str(revision.get("action") or ""),
        "operator_id": str(revision.get("operator_id") or ""),
        "reason": str(revision.get("reason") or ""),
        "registry_path": str(revision.get("registry_path") or ""),
        "revision_path": str(revision.get("revision_path") or ""),
        "registry_sha256": str(revision.get("registry_sha256") or ""),
    }


def _snapshot_customer_project_revision(
    profile_root: Path,
    path: Path,
    *,
    action: str,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Save the current profile before a customer-project write."""
    source = Path(path)
    if not source.exists():
        return {}
    profile = _normalize_customer_project_profile(load_field_site_profile(source))
    profile_hash = _sha256_json(profile)
    created_at = time.time()
    revision_id = _slug(
        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created_at))}-"
        f"{int(created_at * 1000)}-{action}-{profile_hash[:12]}"
    )
    revision = {
        "revision_type": "askme.customer_project_revision",
        "revision_version": 1,
        "revision_id": revision_id,
        "created_at": created_at,
        "action": str(action or "profile_write"),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or ""),
        "source_profile_path": str(source),
        "delivery_scope": _delivery_scope_payload(profile),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "profile_sha256": profile_hash,
        "profile": profile,
    }
    target_dir = _customer_project_revision_dir(profile_root, source, profile)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{revision_id}.json"
    target.write_text(json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8")
    revision["revision_path"] = str(target)
    return revision


def _load_customer_project_revisions(
    profile_root: Path,
    path: Path,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    revision_dir = _customer_project_revision_dir(profile_root, path, profile)
    if not revision_dir.exists():
        return []
    revisions = [
        _revision_public_payload(payload)
        for payload in (
            _read_customer_project_revision_file(item)
            for item in revision_dir.glob("*.json")
            if item.is_file()
        )
        if payload
    ]
    revisions.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
    return revisions


def _find_customer_project_revision(
    profile_root: Path,
    path: Path,
    profile: dict[str, Any],
    revision_id: str,
) -> dict[str, Any] | None:
    target_id = str(revision_id or "").strip()
    if not target_id:
        return None
    revision_dir = _customer_project_revision_dir(profile_root, path, profile)
    if not revision_dir.exists():
        return None
    for item in revision_dir.glob("*.json"):
        payload = _read_customer_project_revision_file(item)
        if payload and str(payload.get("revision_id") or "") == target_id:
            return payload
    return None


def _read_customer_project_revision_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("revision_type") != "askme.customer_project_revision":
        return {}
    payload["revision_path"] = str(path)
    return payload


def _revision_public_payload(revision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(revision, dict) or not revision:
        return {}
    return {
        "revision_id": str(revision.get("revision_id") or ""),
        "created_at": revision.get("created_at"),
        "action": str(revision.get("action") or ""),
        "operator_id": str(revision.get("operator_id") or ""),
        "reason": str(revision.get("reason") or ""),
        "source_profile_path": str(revision.get("source_profile_path") or ""),
        "revision_path": str(revision.get("revision_path") or ""),
        "delivery_scope": _mapping(revision.get("delivery_scope")),
        "customer": _mapping(revision.get("customer")),
        "site": _mapping(revision.get("site")),
        "profile_sha256": str(revision.get("profile_sha256") or ""),
    }


def _customer_project_revision_dir(profile_root: Path, path: Path, profile: dict[str, Any]) -> Path:
    scope = _delivery_scope_payload(profile)
    try:
        rel = Path(path).resolve().relative_to(Path(profile_root).resolve())
    except Exception:
        rel = Path(path).name
    rel_hash = hashlib.sha256(str(rel).encode("utf-8")).hexdigest()[:10]
    return Path(profile_root) / "_revisions" / _slug(scope.get("tenant_id")) / _slug(
        scope.get("delivery_namespace")
    ) / _slug(scope.get("customer_id")) / _slug(scope.get("project_id") or scope.get("site_id")) / rel_hash


def _snapshot_customer_project_template_revision(
    template_root: Path,
    path: Path,
    *,
    action: str,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Save the current template before a release-governance write."""
    source = Path(path)
    if not source.exists():
        return {}
    profile = load_field_site_profile(source)
    template = _mapping(profile.get("template"))
    template_id = str(template.get("template_id") or source.stem)
    profile_hash = _sha256_json(profile)
    created_at = time.time()
    revision_id = _slug(
        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created_at))}-"
        f"{int(created_at * 1000)}-{action}-{profile_hash[:12]}"
    )
    revision = {
        "revision_type": "askme.customer_project_template_revision",
        "revision_version": 1,
        "revision_id": revision_id,
        "created_at": created_at,
        "action": str(action or "template_release_write"),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or ""),
        "source_template_path": str(source),
        "template_id": template_id,
        "template_release": {
            "version": str(template.get("version") or "0.0.0"),
            "publish_status": str(template.get("publish_status") or "draft"),
            "release_channel": str(template.get("release_channel") or template.get("publish_status") or "draft"),
            "owner": str(template.get("owner") or ""),
            "upgrade_policy": str(template.get("upgrade_policy") or ""),
            "min_runtime_version": str(template.get("min_runtime_version") or ""),
        },
        "profile_sha256": profile_hash,
        "profile": profile,
    }
    target_dir = _customer_project_template_revision_dir(template_root, source, template)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{revision_id}.json"
    target.write_text(json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8")
    revision["revision_path"] = str(target)
    return revision


def _load_customer_project_template_revisions(
    template_root: Path,
    path: Path,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    revision_dir = _customer_project_template_revision_dir(
        template_root,
        path,
        _mapping(profile.get("template")),
    )
    if not revision_dir.exists():
        return []
    revisions = [
        _template_revision_public_payload(payload)
        for payload in (
            _read_customer_project_template_revision_file(item)
            for item in revision_dir.glob("*.json")
            if item.is_file()
        )
        if payload
    ]
    revisions.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
    return revisions


def _read_customer_project_template_revision_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("revision_type") != "askme.customer_project_template_revision":
        return {}
    payload["revision_path"] = str(path)
    return payload


def _template_revision_public_payload(revision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(revision, dict) or not revision:
        return {}
    return {
        "revision_id": str(revision.get("revision_id") or ""),
        "created_at": revision.get("created_at"),
        "action": str(revision.get("action") or ""),
        "operator_id": str(revision.get("operator_id") or ""),
        "reason": str(revision.get("reason") or ""),
        "source_template_path": str(revision.get("source_template_path") or ""),
        "revision_path": str(revision.get("revision_path") or ""),
        "template_id": str(revision.get("template_id") or ""),
        "template_release": _mapping(revision.get("template_release")),
        "profile_sha256": str(revision.get("profile_sha256") or ""),
    }


def _customer_project_template_revision_dir(
    template_root: Path,
    path: Path,
    template: dict[str, Any],
) -> Path:
    template_id = _slug(template.get("template_id") or Path(path).stem)
    try:
        rel = Path(path).resolve().relative_to(Path(template_root).resolve())
    except Exception:
        rel = Path(path).name
    rel_hash = hashlib.sha256(str(rel).encode("utf-8")).hexdigest()[:10]
    return Path(template_root) / "_template_revisions" / template_id / rel_hash


def _template_release_payload(release: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(release, dict):
        return {}
    payload = {
        field: str(release.get(field) or "").strip()
        for field in TEMPLATE_RELEASE_FIELDS
        if field in release and str(release.get(field) or "").strip()
    }
    if "publish_status" in release:
        payload["publish_status"] = str(release.get("publish_status") or "").strip()
    return payload


def _customer_project_template_release_request_dir(template_root: Path, template_id: str) -> Path:
    return Path(template_root) / "_template_release_requests" / _slug(template_id or "template")


def _iter_customer_project_template_release_requests(template_root: Path) -> list[dict[str, Any]]:
    root = Path(template_root) / "_template_release_requests"
    if not root.exists():
        return []
    return [
        payload
        for payload in (
            _read_customer_project_template_release_request_file(item)
            for item in root.glob("*/*.json")
            if item.is_file()
        )
        if payload
    ]


def _find_customer_project_template_release_request(
    template_root: Path,
    request_id: str,
) -> tuple[Path | None, dict[str, Any]]:
    target_id = str(request_id or "").strip()
    if not target_id:
        return None, {}
    root = Path(template_root) / "_template_release_requests"
    if not root.exists():
        return None, {}
    for item in root.glob("*/*.json"):
        payload = _read_customer_project_template_release_request_file(item)
        if payload and str(payload.get("request_id") or "") == target_id:
            return item, payload
    return None, {}


def _read_customer_project_template_release_request_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("request_type") != "askme.customer_project_template_release_request"
    ):
        return {}
    payload["request_path"] = str(path)
    return payload


def _template_release_request_public_payload(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or not request:
        return {}
    return {
        "request_id": str(request.get("request_id") or ""),
        "status": str(request.get("status") or ""),
        "template_id": str(request.get("template_id") or ""),
        "template_path": str(request.get("template_path") or ""),
        "request_path": str(request.get("request_path") or ""),
        "requested_by": str(request.get("requested_by") or ""),
        "requested_at": request.get("requested_at"),
        "reason": str(request.get("reason") or ""),
        "reviewed_by": str(request.get("reviewed_by") or ""),
        "reviewed_at": request.get("reviewed_at"),
        "review_reason": str(request.get("review_reason") or ""),
        "release": _mapping(request.get("release")),
        "current_template_sha256": str(request.get("current_template_sha256") or ""),
        "current_template_package": _mapping(request.get("current_template_package")),
        "proposed_template_package": _mapping(request.get("proposed_template_package")),
        "applied_template_package": _mapping(request.get("applied_template_package")),
        "applied_revision": _mapping(request.get("applied_revision")),
    }


def _release_notes_customer_context(payload: dict[str, Any]) -> dict[str, str]:
    clean = _clean_mapping(payload)
    return {
        "customer_name": str(clean.get("customer_name") or "Customer"),
        "customer_id": str(clean.get("customer_id") or ""),
        "project_name": str(clean.get("project_name") or "AskMe Robot Deployment"),
        "project_id": str(clean.get("project_id") or ""),
        "site_name": str(clean.get("site_name") or ""),
        "industry": str(clean.get("industry") or ""),
        "logo_url": str(clean.get("logo_url") or ""),
    }


def _release_notes_bundle_slug(context: dict[str, Any]) -> str:
    basis = (
        context.get("project_id")
        or context.get("project_name")
        or context.get("customer_id")
        or context.get("customer_name")
        or "askme"
    )
    return _slug(basis)


def _template_release_notes_proposal_insert(
    context: dict[str, Any],
    notes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    customer_name = str(context.get("customer_name") or "Customer")
    project_name = str(context.get("project_name") or "AskMe Robot Deployment")
    template_ids = [str(item.get("template_id") or "") for item in notes if item.get("template_id")]
    scenario_ids = sorted({
        scenario
        for item in notes
        for scenario in _string_list(_mapping(item.get("applicability_scope")).get("scenarios"))
    })
    dependency_count = sum(
        len(item.get("dependency_matrix") if isinstance(item.get("dependency_matrix"), list) else [])
        for item in notes
    )
    acceptance_coverage = [
        {
            "template_id": str(item.get("template_id") or ""),
            "scenario_count": len(_string_list(_mapping(item.get("applicability_scope")).get("scenarios"))),
            "acceptance_criteria_count": len(
                item.get("scenario_acceptance_criteria")
                if isinstance(item.get("scenario_acceptance_criteria"), list)
                else []
            ),
            "dependency_count": len(
                item.get("dependency_matrix")
                if isinstance(item.get("dependency_matrix"), list)
                else []
            ),
        }
        for item in notes
    ]
    safe_claims = [
        "AskMe can start this project from approved reusable robot-service templates.",
        "Each selected template still requires customer site binding, device credentials, and onsite acceptance evidence.",
        "Only templates approved by the product-owner release workflow are listed in this proposal insert.",
    ]
    if not notes:
        safe_claims = [
            "No reusable template package has been approved for customer-facing proposal use yet.",
            "Sales and delivery should not claim reusable template readiness until a published release is approved.",
        ]
    return {
        "section_title": f"{project_name} approved reusable capabilities",
        "customer_message": (
            f"For {customer_name}, AskMe currently has {int(summary.get('approved_release_count') or len(notes))} "
            "approved reusable robot-service template release(s) available for proposal discussion."
        ),
        "approved_template_ids": template_ids,
        "scenario_coverage": acceptance_coverage,
        "dependency_summary": {
            "approved_template_count": len(template_ids),
            "scenario_count": len(scenario_ids),
            "dependency_count": dependency_count,
        },
        "safe_claims": safe_claims,
        "delivery_boundaries": [
            "Not a production go-live certificate.",
            "Does not replace customer site survey, robot map validation, live sensor tests, notification tests, or runtime acceptance.",
            "Final delivery scope must be confirmed in the customer project acceptance dossier.",
        ],
    }


def _template_release_notes_bundle_html(bundle: dict[str, Any]) -> str:
    context = _mapping(bundle.get("customer_context"))
    notes = bundle.get("release_notes") if isinstance(bundle.get("release_notes"), list) else []
    summary = _mapping(bundle.get("summary"))
    proposal = _mapping(bundle.get("proposal_insert"))
    safe_claims = proposal.get("safe_claims") if isinstance(proposal.get("safe_claims"), list) else []
    boundaries = proposal.get("delivery_boundaries") if isinstance(proposal.get("delivery_boundaries"), list) else []
    logo_url = str(context.get("logo_url") or "")
    logo_html = (
        f'<img src="{html.escape(logo_url)}" alt="customer logo">'
        if logo_url
        else '<div class="logo-placeholder">AskMe</div>'
    )
    note_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('template_id') or '-'))}</td>"
        f"<td>{html.escape(str(item.get('version') or '-'))}</td>"
        f"<td>{html.escape(str(item.get('release_channel') or '-'))}</td>"
        f"<td>{html.escape(str(item.get('product_status') or '-'))}</td>"
        f"<td>{html.escape(str(item.get('customer_status') or item.get('customer_claim') or '-'))}</td>"
        f"<td>{html.escape(str(item.get('approved_by') or '-'))}</td>"
        "</tr>"
        for item in notes
    )
    if not note_rows:
        note_rows = '<tr><td colspan="6">No approved published template releases are available.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AskMe Template Release Notes</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #17211f; margin: 32px; }}
    header {{ display: flex; gap: 16px; align-items: center; border-bottom: 1px solid #d9e5df; padding-bottom: 18px; }}
    img {{ width: 72px; max-height: 72px; object-fit: contain; }}
    .logo-placeholder {{ width: 72px; height: 72px; border-radius: 14px; background: #0d513e; color: #fff; display: grid; place-items: center; font-weight: 700; }}
    h1 {{ margin: 0; font-size: 26px; }}
    .muted {{ color: #64746e; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }}
    .metric {{ border: 1px solid #d9e5df; border-radius: 10px; padding: 14px; }}
    .metric b {{ display: block; font-size: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #e7efeb; text-align: left; padding: 10px 8px; vertical-align: top; }}
    th {{ color: #42534d; font-size: 12px; text-transform: uppercase; }}
    .boundary {{ margin-top: 24px; padding: 14px; background: #f4faf7; border: 1px solid #cfe4da; border-radius: 10px; }}
    code {{ background: #eef4f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    {logo_html}
    <div>
      <h1>AskMe Template Release Notes</h1>
      <div class="muted">{html.escape(str(context.get("customer_name") or "Customer"))} / {html.escape(str(context.get("project_name") or "AskMe Robot Deployment"))}</div>
      <div class="muted">{html.escape(str(context.get("site_name") or ""))}</div>
    </div>
  </header>
  <section class="metrics">
    <div class="metric"><b>{html.escape(str(summary.get("approved_release_count") or 0))}</b><span>approved releases</span></div>
    <div class="metric"><b>{html.escape(str(summary.get("template_count") or 0))}</b><span>templates</span></div>
    <div class="metric"><b>{html.escape(str(summary.get("ready_count") or 0))}</b><span>ready</span></div>
    <div class="metric"><b>{html.escape(str(summary.get("manual_check_count") or 0))}</b><span>manual check</span></div>
  </section>
  <p>{html.escape(str(bundle.get("customer_claim") or ""))}</p>
  <section class="boundary">
    <strong>{html.escape(str(proposal.get("section_title") or "Approved reusable capabilities"))}</strong>
    <p>{html.escape(str(proposal.get("customer_message") or ""))}</p>
    <ul>
      {"".join(f"<li>{html.escape(str(item))}</li>" for item in safe_claims)}
    </ul>
  </section>
  <table>
    <thead>
      <tr><th>Template</th><th>Version</th><th>Channel</th><th>Status</th><th>Customer status</th><th>Approved by</th></tr>
    </thead>
    <tbody>
      {note_rows}
    </tbody>
  </table>
  <section class="boundary">
    <strong>Delivery boundary</strong>
    <p>{html.escape(str(bundle.get("delivery_boundary") or ""))}</p>
    <ul>
      {"".join(f"<li>{html.escape(str(item))}</li>" for item in boundaries)}
    </ul>
    <p class="muted">Bundle SHA-256: <code>{html.escape(str(_mapping(bundle.get("manifest")).get("bundle_sha256") or ""))}</code></p>
  </section>
</body>
</html>
"""


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _clean_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if value not in (None, "")
    }


def _clean_nested_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, ""):
            continue
        if isinstance(value, dict):
            result[str(key)] = _clean_nested_mapping(value)
        elif isinstance(value, list):
            result[str(key)] = [item for item in value if item not in (None, "")]
        else:
            result[str(key)] = value
    return result


def _customer_project_package_manifest(package: dict[str, Any]) -> dict[str, Any]:
    profile = package.get("profile") if isinstance(package.get("profile"), dict) else {}
    customer = package.get("customer") if isinstance(package.get("customer"), dict) else {}
    site = package.get("site") if isinstance(package.get("site"), dict) else {}
    acceptance = _mapping(package.get("acceptance_summary"))
    binding_readiness = _mapping(package.get("binding_readiness_summary"))
    resource_catalog = _mapping(package.get("resource_catalog_summary"))
    reuse = _mapping(package.get("reuse_assessment"))
    dependencies = _mapping(reuse.get("dependencies"))
    action_plan = _mapping(package.get("managed_object_action_plan"))
    delivery_gate = _mapping(package.get("package_delivery_gate"))
    applicability = _mapping(package.get("applicability_scope"))
    out_of_scope = package.get("out_of_scope") if isinstance(package.get("out_of_scope"), list) else []
    prerequisites = (
        package.get("customer_prerequisites")
        if isinstance(package.get("customer_prerequisites"), list)
        else []
    )
    scenario_criteria = (
        package.get("scenario_acceptance_criteria")
        if isinstance(package.get("scenario_acceptance_criteria"), list)
        else []
    )
    dependency_matrix = (
        package.get("dependency_matrix")
        if isinstance(package.get("dependency_matrix"), list)
        else []
    )
    payload_sha256 = _customer_project_package_payload_sha256(package)
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "payload_sha256": payload_sha256,
        "profile_sha256": _sha256_json(profile),
        "tenant_id": _delivery_tenant_id(customer),
        "delivery_namespace": _delivery_namespace(customer),
        "customer_id": str(customer.get("customer_id") or ""),
        "project_id": str(customer.get("project_id") or ""),
        "site_id": str(site.get("site_id") or ""),
        "managed_object_count": int(
            _mapping(package.get("managed_object_catalog")).get("object_type_count") or 0
        ),
        "acceptance_overall_status": str(acceptance.get("overall_status") or "blocked"),
        "acceptance_ready_object_count": int(acceptance.get("ready_object_count") or 0),
        "acceptance_manual_check_object_count": int(
            acceptance.get("manual_check_object_count") or 0
        ),
        "acceptance_blocked_object_count": int(acceptance.get("blocked_object_count") or 0),
        "resource_binding_overall_status": str(binding_readiness.get("overall_status") or "blocked"),
        "resource_binding_ready_object_count": int(binding_readiness.get("ready_object_count") or 0),
        "resource_binding_manual_check_object_count": int(
            binding_readiness.get("manual_check_object_count") or 0
        ),
        "resource_binding_blocked_object_count": int(binding_readiness.get("blocked_object_count") or 0),
        "resource_binding_unregistered_resource_count": int(
            binding_readiness.get("unregistered_resource_count") or 0
        ),
        "delivery_resource_count": int(resource_catalog.get("resource_count") or 0),
        "reuse_status": str(reuse.get("status") or "unknown"),
        "reuse_blocker_count": int(reuse.get("blocker_count") or 0),
        "reuse_manual_check_count": int(reuse.get("manual_check_count") or 0),
        "dependency_device_count": int(dependencies.get("device_count") or 0),
        "dependency_env_reference_count": int(dependencies.get("env_reference_count") or 0),
        "dependency_missing_env_count": int(dependencies.get("missing_env_count") or 0),
        "applicability_scenario_count": len(_string_list(applicability.get("scenarios"))),
        "applicability_managed_object_type_count": len(_string_list(applicability.get("managed_object_types"))),
        "out_of_scope_count": len(out_of_scope),
        "customer_prerequisite_count": len(prerequisites),
        "required_customer_prerequisite_count": len([
            item for item in prerequisites if isinstance(item, dict) and item.get("required") is True
        ]),
        "scenario_acceptance_criteria_count": len(scenario_criteria),
        "dependency_matrix_count": len(dependency_matrix),
        "package_delivery_gate_status": str(delivery_gate.get("delivery_gate_status") or "blocked"),
        "package_delivery_export_allowed": bool(delivery_gate.get("export_allowed")),
        "package_delivery_import_allowed": bool(delivery_gate.get("import_allowed")),
        "package_delivery_customer_handoff_ready": bool(delivery_gate.get("customer_handoff_ready")),
        "package_delivery_action_count": int(delivery_gate.get("action_count") or action_plan.get("action_count") or 0),
        "package_delivery_blocked_action_count": int(
            delivery_gate.get("blocked_action_count") or action_plan.get("blocked_action_count") or 0
        ),
        "package_delivery_manual_check_action_count": int(
            delivery_gate.get("manual_check_action_count") or action_plan.get("manual_check_action_count") or 0
        ),
        "package_delivery_source_version": str(
            action_plan.get("delivery_gate_source_version")
            or delivery_gate.get("delivery_gate_source_version")
            or ""
        ),
        "signature_alg": "",
        "signature_key_id": "",
        "payload_signature": "",
    }
    secret = _clean_secret(os.getenv("ASKME_CUSTOMER_PROJECT_PACKAGE_HMAC_SECRET"))
    if secret:
        manifest["signature_alg"] = "hmac-sha256"
        manifest["signature_key_id"] = os.getenv(
            "ASKME_CUSTOMER_PROJECT_PACKAGE_SIGNATURE_KEY_ID",
            "local-customer-project-package",
        )
        manifest["payload_signature"] = hmac.new(
            secret.encode("utf-8"),
            payload_sha256.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    return manifest


def _customer_project_package_payload_sha256(package: dict[str, Any]) -> str:
    payload = {key: value for key, value in package.items() if key != "manifest"}
    return _sha256_json(payload)


def _customer_project_acceptance_dossier_manifest(dossier: dict[str, Any]) -> dict[str, Any]:
    customer = _mapping(dossier.get("customer"))
    site = _mapping(dossier.get("site"))
    evidence = dossier.get("evidence_inventory") if isinstance(dossier.get("evidence_inventory"), list) else []
    onsite = _mapping(dossier.get("onsite_acceptance_evidence"))
    onsite_summary = _mapping(onsite.get("summary"))
    checklist = _mapping(dossier.get("site_acceptance_checklist"))
    launch_readiness = _mapping(dossier.get("launch_readiness"))
    reviews = dossier.get("acceptance_reviews") if isinstance(dossier.get("acceptance_reviews"), list) else []
    latest_review = _mapping(reviews[0]) if reviews else {}
    signoffs = dossier.get("customer_signoffs") if isinstance(dossier.get("customer_signoffs"), list) else []
    latest_signoff = _mapping(signoffs[0]) if signoffs else {}
    payload_sha256 = _customer_project_acceptance_dossier_payload_sha256(dossier)
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "payload_sha256": payload_sha256,
        "tenant_id": _delivery_tenant_id(customer),
        "delivery_namespace": _delivery_namespace(customer),
        "customer_id": str(customer.get("customer_id") or ""),
        "project_id": str(customer.get("project_id") or ""),
        "site_id": str(site.get("site_id") or ""),
        "overall_status": str(dossier.get("overall_status") or "unknown"),
        "launch_readiness_status": str(launch_readiness.get("overall_status") or "manual_check"),
        "launch_stage": str(launch_readiness.get("launch_stage") or "demo_or_integration_only"),
        "production_ready": bool(launch_readiness.get("production_ready")),
        "field_readiness_status": str(_mapping(dossier.get("field_readiness")).get("status") or "unknown"),
        "onsite_evidence_status": str(onsite_summary.get("overall_status") or "manual_check"),
        "onsite_evidence_count": int(onsite_summary.get("receipt_count") or 0),
        "onsite_required_evidence_ready": bool(
            onsite_summary.get("overall_status") == "ready"
        ),
        "site_acceptance_checklist_status": str(checklist.get("overall_status") or "manual_check"),
        "site_acceptance_checklist_ready_count": int(checklist.get("ready_count") or 0),
        "site_acceptance_checklist_manual_check_count": int(checklist.get("manual_check_count") or 0),
        "site_acceptance_checklist_blocked_count": int(checklist.get("blocked_count") or 0),
        "manual_review_decision": str(latest_review.get("decision") or ""),
        "manual_review_count": len(reviews),
        "customer_signoff_decision": str(latest_signoff.get("decision") or ""),
        "customer_signoff_count": len(signoffs),
        "customer_signoff_credential_sha256": str(latest_signoff.get("credential_sha256") or ""),
        "customer_signoff_payload_sha256": str(latest_signoff.get("signoff_payload_sha256") or ""),
        "customer_signoff_integrity_valid": bool(latest_signoff.get("integrity_valid")) if latest_signoff else False,
        "evidence_count": len(evidence),
        "evidence_missing_count": len([item for item in evidence if not _mapping(item).get("exists")]),
        "evidence_invalid_count": len([item for item in evidence if _mapping(item).get("sha256") == ""]),
        "signature_alg": "",
        "signature_key_id": "",
        "payload_signature": "",
    }
    secret = _clean_secret(os.getenv("ASKME_CUSTOMER_ACCEPTANCE_DOSSIER_HMAC_SECRET"))
    if secret:
        manifest["signature_alg"] = "hmac-sha256"
        manifest["signature_key_id"] = os.getenv(
            "ASKME_CUSTOMER_ACCEPTANCE_DOSSIER_SIGNATURE_KEY_ID",
            "local-customer-acceptance-dossier",
        )
        manifest["payload_signature"] = hmac.new(
            secret.encode("utf-8"),
            payload_sha256.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    return manifest


def _customer_project_acceptance_dossier_payload_sha256(dossier: dict[str, Any]) -> str:
    payload = {key: value for key, value in dossier.items() if key != "manifest"}
    return _sha256_json(payload)


def _customer_project_proposal_bundle_payload_sha256(proposal: dict[str, Any]) -> str:
    payload = {key: value for key, value in proposal.items() if key not in {"manifest", "html"}}
    return _sha256_json(payload)


def _customer_project_evidence_inventory(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _mapping(report.get("field_readiness"))
    reports = readiness.get("evidence_reports") if isinstance(readiness.get("evidence_reports"), list) else []
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reports:
        compact = _mapping(item)
        path = str(compact.get("path") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        inventory.append(_evidence_file_inventory(path, evidence_url=str(compact.get("evidence_url") or "")))
    archive_path = str(_mapping(readiness.get("archive")).get("path") or "")
    if archive_path and archive_path not in seen:
        inventory.append(_evidence_file_inventory(archive_path, evidence_url=_evidence_url(archive_path)))
    onsite = _mapping(report.get("onsite_acceptance_evidence"))
    onsite_receipts = onsite.get("receipts") if isinstance(onsite.get("receipts"), list) else []
    for item in onsite_receipts:
        receipt = _mapping(item)
        path = str(receipt.get("path") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        record = _evidence_file_inventory(path, evidence_url=str(receipt.get("evidence_url") or _evidence_url(path)))
        record.update({
            "evidence_type": "onsite_acceptance",
            "onsite_evidence_type": str(receipt.get("evidence_type") or ""),
            "receipt_id": str(receipt.get("receipt_id") or ""),
        })
        inventory.append(record)
    return inventory


def _evidence_file_inventory(path: str, *, evidence_url: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path,
        "evidence_url": evidence_url,
        "exists": False,
        "size_bytes": 0,
        "sha256": "",
    }
    try:
        resolved = Path(path).resolve()
    except OSError as exc:
        record["error"] = str(exc)
        return record
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        record["error"] = "outside_project"
        return record
    if not resolved.exists() or not resolved.is_file():
        return record
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        record["error"] = str(exc)
        return record
    record.update({
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    })
    return record


def _customer_project_profile_diff(current: dict[str, Any], incoming: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in (
        ("customer",),
        ("site",),
        ("zones",),
        ("devices",),
        ("responder_groups",),
        ("thresholds",),
        ("managed_objects",),
    ):
        current_value = _get_nested(current, path)
        incoming_value = _get_nested(incoming, path)
        if current_value != incoming_value:
            changes.append(
                {
                    "path": ".".join(path),
                    "current_sha256": _sha256_json(current_value),
                    "incoming_sha256": _sha256_json(incoming_value),
                }
            )
    return changes


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clean_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("${"):
        return ""
    return text


def _non_empty_text(value: Any) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_DELIVERY_NAMESPACE


def _customer_delivery_filename_parts(customer: dict[str, Any]) -> list[str]:
    tenant = _slug(_delivery_tenant_id(customer))
    namespace = _slug(_delivery_namespace(customer))
    if tenant == DEFAULT_DELIVERY_NAMESPACE and namespace == DEFAULT_DELIVERY_NAMESPACE:
        return []
    if namespace == tenant:
        return [tenant]
    return [tenant, namespace]


def _slug(value: Any) -> str:
    text = str(value or "item").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-_")
    return text or "item"


def _zones_by_type(zones: dict[str, Any], zone_type: str) -> dict[str, dict[str, Any]]:
    return {
        zone_id: zone
        for zone_id, zone in zones.items()
        if isinstance(zone, dict) and str(zone.get("type") or "") == zone_type
    }


def _require_env_reference(
    item: dict[str, Any],
    *,
    key: str,
    errors: list[str],
    warnings: list[str],
    label: str,
    check_env: bool,
) -> None:
    env_name = str(item.get(key) or "")
    if not env_name:
        errors.append(f"{label} is required")
        return
    if check_env and not os.getenv(env_name):
        warnings.append(f"{label} references unset environment variable {env_name}")


def _validate_thresholds(thresholds: dict[str, Any], errors: list[str]) -> None:
    required = {
        "parking_duration_s": 1,
        "night_stranger_dwell_s": 1,
        "fire_temperature_c": 1,
        "smoke_level": 0,
        "trash_fill_ratio": 0,
        "crowd_person_count": 1,
        "crowd_duration_min": 1,
    }
    for key, minimum in required.items():
        raw = thresholds.get(key)
        if raw is None:
            errors.append(f"thresholds.{key} is required")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"thresholds.{key} must be numeric")
            continue
        if value < minimum:
            errors.append(f"thresholds.{key} must be >= {minimum}")


def _env_placeholder(env_name: Any) -> str:
    return f"${{{env_name}}}" if env_name else ""


def _env_reference(
    env_name: Any,
    *,
    category: str,
    reference: str,
    owner: str,
    owner_label: str,
    purpose: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(env_name or "").strip()
    return {
        "env_name": name,
        "category": category,
        "reference": reference,
        "owner": owner,
        "owner_label": owner_label,
        "purpose": purpose,
        "required": True,
        "configured": bool(name and os.getenv(name)),
        "metadata": metadata or {},
    }


def _dedupe_env_references(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    by_env: dict[str, dict[str, Any]] = {}
    for ref in refs:
        env_name = str(ref.get("env_name") or "").strip()
        if not env_name:
            continue
        existing = by_env.get(env_name)
        if existing is None:
            clone = dict(ref)
            clone["references"] = [str(ref.get("reference") or "")]
            by_env[env_name] = clone
            deduped.append(clone)
            continue
        references = existing.setdefault("references", [])
        reference = str(ref.get("reference") or "")
        if reference and reference not in references:
            references.append(reference)
        if not existing.get("configured") and ref.get("configured"):
            existing["configured"] = True
    return deduped
