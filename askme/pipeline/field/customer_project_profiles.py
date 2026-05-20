"""Customer-project profile identity, store, revision, and object-log helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_scope import (
    _delivery_scope_payload,
    _delivery_scope_payload_from_customer_site,
    _same_customer_project_identity,
    _same_delivery_project_scope,
)
from askme.pipeline.field.customer_project_template_support import (
    DEFAULT_DELIVERY_NAMESPACE,
    _delivery_namespace,
    _delivery_tenant_id,
    _mapping,
    _sha256_json,
    _site_profile_paths,
    _slug,
    _string_list,
    load_field_site_profile,
)


def _normalize_customer_project_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a profile with explicit delivery scope defaults for product handoff."""
    result = copy.deepcopy(profile)
    customer = dict(_mapping(result.get("customer")))
    customer["tenant_id"] = _delivery_tenant_id(customer)
    customer["delivery_namespace"] = _delivery_namespace(customer)
    customer.setdefault("delivery_model", "solution_project")
    result["customer"] = customer
    return result


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


def find_site_profile_path(root: Path, identifier: str) -> Path | None:
    """Find a site profile by site_id, project_id, delivery-scope alias, or filename stem."""
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


def _customer_project_collision_candidates(
    profile_root: Path,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    incoming_scope = _delivery_scope_payload(profile)
    collisions: list[dict[str, Any]] = []
    for path in _site_profile_paths(Path(profile_root), pattern="*.yaml"):
        try:
            current = load_field_site_profile(path)
        except Exception:
            continue
        current_scope = _delivery_scope_payload(current)
        if _same_customer_project_identity(
            current_scope,
            incoming_scope,
        ) and not _same_delivery_project_scope(current_scope, incoming_scope):
            collisions.append({"profile_path": str(path), "delivery_scope": current_scope})
    return collisions


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


__all__ = [
    "archive_customer_project_profile",
    "customer_project_catalog_acceptance_gate",
    "customer_project_catalog_summary_from_projects",
    "find_site_profile_path",
    "list_customer_project_revisions",
    "_append_object_change_log",
    "_customer_project_catalog_delivery_acceptance_gate",
    "_customer_project_catalog_filters",
    "_customer_project_delivery_status",
    "_customer_payload",
    "_customer_profile_path",
    "_customer_profile_target",
    "_customer_project_collision_candidates",
    "_customer_project_matches_filters",
    "_customer_project_product_acceptance_gate",
    "_customer_project_revision_dir",
    "_customer_rows",
    "_delivery_identifier_candidates",
    "_find_customer_project_profile_path",
    "_find_customer_project_revision",
    "_load_customer_project_revisions",
    "_normalize_customer_project_profile",
    "_object_change_log_payload",
    "_object_change_summary",
    "_read_customer_project_revision_file",
    "_revision_public_payload",
    "_snapshot_customer_project_revision",
    "_text_filter_matches",
]
