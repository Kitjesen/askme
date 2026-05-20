"""Customer-project profile and managed-object mutation operations.

This module owns the product-facing CRUD operations for customer project
profiles. Lower-level storage, scope, validation, and object catalog helpers
stay in their dedicated modules.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_implementation_handoff import (
    _customer_project_implementation_handoff,
)
from askme.pipeline.field.customer_project_managed_objects import (
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_profiles import (
    _append_object_change_log,
    _customer_payload,
    _customer_profile_path,
    _customer_profile_target,
    _find_customer_project_revision,
    _normalize_customer_project_profile,
    _object_change_log_payload,
    _revision_public_payload,
    _snapshot_customer_project_revision,
    find_site_profile_path,
)
from askme.pipeline.field.customer_project_scope import (
    _customer_project_profile_diff,
    _delivery_scope_payload,
    _same_delivery_project_scope,
)
from askme.pipeline.field.customer_project_template_support import (
    _clean_mapping,
    _clean_nested_mapping,
    _find_template_path,
    _mapping,
    _sha256_json,
    _slug,
    _write_yaml,
    load_field_site_profile,
    site_profile_env_references,
)
from askme.pipeline.field.field_site_catalog import (
    _customer_project_delivery_workflow,
    build_site_profile_report,
)
from askme.pipeline.field.field_site_validation import validate_field_site_profile

__all__ = [
    "create_customer_project_from_template",
    "upsert_customer_project_profile",
    "get_customer_project_profile",
    "upsert_managed_object",
    "delete_managed_object",
    "rollback_customer_project_profile",
]


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


def get_customer_project_profile(
    profile_root: Path,
    identifier: str,
    *,
    check_env: bool = False,
) -> dict[str, Any]:
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
