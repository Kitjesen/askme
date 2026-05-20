"""Delivery-resource registry, catalog, and revision kernel."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from askme.pipeline.field.paths import DEFAULT_DELIVERY_RESOURCE_ROOT

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


def load_delivery_resource_registry(
    resource_root: Path = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
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


def list_delivery_resource_registry(
    resource_root: Path = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
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
        return {
            "accepted": False,
            "reason": "resource_already_exists",
            "resource_type": clean_type,
            "resource_id": clean_id,
        }
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
    _write_yaml(
        path,
        {
            "registry_type": registry["registry_type"],
            "registry_version": registry["registry_version"],
            "updated_at": registry["updated_at"],
            "updated_by": registry["updated_by"],
            "delivery_resources": resources,
        },
    )
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
        return {
            "accepted": False,
            "reason": "resource_not_found",
            "resource_type": clean_type,
            "resource_id": clean_id,
        }
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
        "resource_count": sum(
            len(_mapping(resource_catalog.get(resource_type)))
            for resource_type in DELIVERY_RESOURCE_TYPES
        ),
        "vision_model_count": len(_mapping(resource_catalog.get("vision_models"))),
        "sensor_protocol_count": len(_mapping(resource_catalog.get("sensor_protocols"))),
        "skill_package_count": len(_mapping(resource_catalog.get("skill_packages"))),
        "acceptance_test_count": len(_mapping(resource_catalog.get("acceptance_tests"))),
    }


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
    return sorted(
        rows.values(),
        key=lambda item: (
            str(item.get("resource_type") or ""),
            str(item.get("resource_id") or ""),
        ),
    )


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
                    if item.get("resource_type") == resource_type
                    and int(item.get("consumer_count") or 0) > 0
                ]),
                "unregistered_resource_count": len([
                    item
                    for item in resources
                    if item.get("resource_type") == resource_type
                    and item.get("status") == "unregistered"
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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


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
    "DELIVERY_RESOURCE_GOVERNANCE_DUE_SOON_S",
    "DELIVERY_RESOURCE_PUBLISH_STATUSES",
    "DELIVERY_RESOURCE_TYPES",
    "DEFAULT_DELIVERY_RESOURCE_CATALOG",
    "DEFAULT_DELIVERY_RESOURCE_GOVERNANCE_SLA_S",
    "disable_delivery_resource",
    "list_delivery_resource_registry",
    "list_delivery_resource_revisions",
    "load_delivery_resource_registry",
    "rollback_delivery_resource_registry",
    "upsert_delivery_resource",
]
