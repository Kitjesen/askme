"""Customer-project template release governance.

This module owns template release requests, revision history, approval review,
and customer-facing release-note bundles. Creating customer projects from
templates stays in the customer-project template facade until that write
workflow is split separately.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any


def list_customer_project_template_revisions(
    root: Path,
    template_id: str,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return release-governance history for one reusable industry template."""
    adapters = _site_profile_adapters()
    path = adapters["find_template_path"](root, template_id)
    if path is None:
        return {
            "found": False,
            "reason": "template_not_found",
            "template_id": str(template_id or ""),
            "revisions": [],
        }
    profile = adapters["load_field_site_profile"](path)
    template = _mapping(profile.get("template"))
    report = adapters["validate_field_site_profile"](profile)
    managed_objects = adapters["managed_object_catalog_from_site_profile"](profile)
    delivery_summary = adapters["template_delivery_summary"](
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
        "template_package": adapters["template_package_summary"](
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
    allow_published: bool = False,
    approval_request_id: str = "",
) -> dict[str, Any]:
    """Update template release metadata with a reversible audit snapshot."""
    adapters = _site_profile_adapters()
    path = adapters["find_template_path"](root, template_id)
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
    profile = adapters["load_field_site_profile"](path)
    template = _mapping(profile.get("template"))
    current_template_id = str(template.get("template_id") or path.stem)
    next_status = str(
        release.get("publish_status") or template.get("publish_status") or "draft"
    ).strip()
    if next_status not in adapters["template_publish_statuses"]:
        return {
            "accepted": False,
            "reason": "invalid_publish_status",
            "template_id": current_template_id,
            "template_path": str(path),
            "allowed_publish_statuses": sorted(adapters["template_publish_statuses"]),
        }
    next_version = str(release.get("version") or template.get("version") or "0.0.0").strip()
    if not adapters["is_semver"](next_version):
        return {
            "accepted": False,
            "reason": "invalid_template_version",
            "template_id": current_template_id,
            "template_path": str(path),
            "version": next_version,
        }
    if next_status == "published" and not dry_run and not allow_published:
        return {
            "accepted": False,
            "reason": "published_release_requires_approval_request",
            "template_id": current_template_id,
            "template_path": str(path),
            "next_step": (
                "Create a release request and approve it with a second product owner before publishing."
            ),
        }

    updated = copy.deepcopy(profile)
    updated_template = _mapping(updated.get("template"))
    updated_template.setdefault("template_id", current_template_id)
    for field in adapters["template_release_fields"]:
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
    if approval_request_id:
        updated_template["release_approval_request_id"] = str(approval_request_id)
    if reason or release.get("reason"):
        updated_template["release_reason"] = str(reason or release.get("reason") or "")
    updated["template"] = adapters["clean_nested_mapping"](updated_template)

    report = adapters["validate_field_site_profile"](updated)
    managed_objects = adapters["managed_object_catalog_from_site_profile"](updated)
    delivery_summary = adapters["template_delivery_summary"](
        template=_mapping(updated.get("template")),
        customer=_mapping(updated.get("customer")),
        managed_objects=managed_objects,
        report=report,
    )
    template_package = adapters["template_package_summary"](
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
        adapters["write_yaml"](path, updated)
    return {
        "accepted": True,
        "dry_run": bool(dry_run),
        "template_id": current_template_id,
        "template_path": str(path),
        "template": _mapping(updated.get("template")),
        "template_package": template_package,
        "delivery_summary": delivery_summary,
        "delivery_checklist": adapters["template_delivery_checklist"](delivery_summary),
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
    adapters = _site_profile_adapters()
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
    profile = adapters["load_field_site_profile"](Path(preview["template_path"]))
    created_at = time.time()
    request_id = adapters["slug"](
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
        "current_template_sha256": adapters["sha256_json"](profile),
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
    _write_release_request_file(target, request_payload)
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
            "applying_count": len([item for item in requests if item.get("status") == "applying"]),
            "apply_failed_count": len([item for item in requests if item.get("status") == "apply_failed"]),
            "approved_count": len([item for item in requests if item.get("status") == "approved"]),
            "rejected_count": len([item for item in requests if item.get("status") == "rejected"]),
        },
    }


def customer_project_template_release_notes(
    root: Path,
    *,
    limit: int = 50,
    template_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return customer-facing template release notes from approved requests."""
    visible_template_ids = (
        None
        if template_ids is None
        else {str(item).strip() for item in template_ids if str(item).strip()}
    )
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
        if visible_template_ids is not None and template_id not in visible_template_ids:
            continue
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
                "release_reason": str(
                    request.get("review_reason") or request.get("reason") or ""
                ),
                "customer_claim": (
                    "已批准的可复用模板包。生产上线前，交付仍需绑定客户范围、真实设备、"
                    "凭证和现场验收证据。"
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
        "customer_claim": "发布说明只包含已审批并发布的模板包。",
    }


def export_customer_project_template_release_notes_bundle(
    root: Path,
    *,
    customer_context: dict[str, Any] | None = None,
    limit: int = 50,
    template_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a portable proposal/handoff bundle from approved template releases."""
    adapters = _site_profile_adapters()
    notes_payload = customer_project_template_release_notes(
        root,
        limit=limit,
        template_ids=template_ids,
    )
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
        or "发布说明只包含已审批并发布的模板包。",
        "delivery_boundary": (
            "本包仅适用于方案和试点交接。生产上线前，交付仍需绑定客户范围、"
            "真实设备、凭证、运行证据和现场验收。"
        ),
        "files": {
            "json_filename": f"{_release_notes_bundle_slug(context)}-template-release-notes.json",
            "html_filename": f"{_release_notes_bundle_slug(context)}-template-release-notes.html",
        },
    }
    manifest = {
        "manifest_version": 1,
        "bundle_sha256": adapters["sha256_json"](bundle_base),
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
    adapters = _site_profile_adapters()
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
            _write_release_request_file(request_path, next_payload)
        return {
            "accepted": True,
            "dry_run": bool(dry_run),
            "request": _template_release_request_public_payload(next_payload),
            "next_step": "Release request rejected. The template YAML was not changed.",
        }

    template_path = adapters["find_template_path"](root, str(request_payload.get("template_id") or ""))
    if template_path is None:
        return {
            "accepted": False,
            "reason": "template_not_found",
            "request": _template_release_request_public_payload(request_payload),
        }
    current_profile = adapters["load_field_site_profile"](template_path)
    if adapters["sha256_json"](current_profile) != str(request_payload.get("current_template_sha256") or ""):
        return {
            "accepted": False,
            "reason": "template_changed_since_request",
            "request": _template_release_request_public_payload(request_payload),
        }
    if not dry_run:
        applying_payload = copy.deepcopy(next_payload)
        applying_payload["status"] = "applying"
        applying_payload["apply_started_at"] = reviewed_at
        _write_release_request_file(request_path, applying_payload)

    try:
        release_result = update_customer_project_template_release(
            root,
            str(request_payload.get("template_id") or ""),
            _mapping(request_payload.get("release")),
            operator_id=reviewer,
            reason=reason or str(request_payload.get("reason") or ""),
            dry_run=dry_run,
            allow_published=True,
            approval_request_id=str(request_payload.get("request_id") or request_id),
        )
    except Exception as exc:
        failed_payload = copy.deepcopy(next_payload)
        failed_payload["status"] = "apply_failed"
        failed_payload["apply_failed_at"] = time.time()
        failed_payload["apply_failure_reason"] = str(exc) or type(exc).__name__
        failed_payload["release_result"] = {
            "accepted": False,
            "reason": "release_apply_exception",
            "error": failed_payload["apply_failure_reason"],
        }
        if not dry_run:
            _write_release_request_file(request_path, failed_payload)
        return {
            "accepted": False,
            "reason": "release_apply_exception",
            "request": _template_release_request_public_payload(failed_payload),
            "release_result": failed_payload["release_result"],
        }
    if not release_result.get("accepted"):
        failed_payload = copy.deepcopy(next_payload)
        failed_payload["status"] = "apply_failed"
        failed_payload["apply_failed_at"] = time.time()
        failed_payload["apply_failure_reason"] = str(
            release_result.get("reason") or "release_apply_failed"
        )
        failed_payload["release_result"] = release_result
        if not dry_run:
            _write_release_request_file(request_path, failed_payload)
        return {
            "accepted": False,
            "reason": release_result.get("reason") or "release_apply_failed",
            "request": _template_release_request_public_payload(failed_payload),
            "release_result": release_result,
        }
    next_payload["status"] = "approved"
    next_payload["applied_template_package"] = release_result.get("template_package")
    next_payload["applied_revision"] = _template_revision_public_payload(
        _mapping(release_result.get("revision"))
    )
    if not dry_run:
        _write_release_request_file(request_path, next_payload)
    return {
        "accepted": True,
        "dry_run": bool(dry_run),
        "request": _template_release_request_public_payload(next_payload),
        "release_result": release_result,
        "next_step": "Template release approved and applied.",
    }


def _template_release_note_delivery_details(root: Path, template_id: str) -> dict[str, Any]:
    adapters = _site_profile_adapters()
    path = adapters["find_template_path"](root, template_id)
    if path is None:
        surface = adapters["customer_delivery_surface"](
            profile={},
            template={"template_id": template_id},
            customer={},
            managed_objects={},
            report={"status": "failed", "errors": ["template_not_found"], "warnings": []},
            surface="template",
        )
        return {
            "delivery_summary": adapters["template_delivery_summary"](
                template={"template_id": template_id},
                customer={},
                managed_objects={},
                report={"status": "failed", "errors": ["template_not_found"], "warnings": []},
            ) | surface,
            **surface,
        }
    profile = adapters["load_field_site_profile"](path)
    report = adapters["validate_field_site_profile"](profile)
    template = _mapping(profile.get("template"))
    customer = _mapping(profile.get("customer"))
    managed_objects = adapters["managed_object_catalog_from_site_profile"](profile)
    surface = adapters["customer_delivery_surface"](
        profile=profile,
        template=template,
        customer=customer,
        managed_objects=managed_objects,
        report=report,
        surface="template",
    )
    return {
        "delivery_summary": adapters["template_delivery_summary"](
            template=template,
            customer=customer,
            managed_objects=managed_objects,
            report=report,
        ) | surface,
        **surface,
    }


def _snapshot_customer_project_template_revision(
    template_root: Path,
    path: Path,
    *,
    action: str,
    operator_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Save the current template before a release-governance write."""
    adapters = _site_profile_adapters()
    source = Path(path)
    if not source.exists():
        return {}
    profile = adapters["load_field_site_profile"](source)
    template = _mapping(profile.get("template"))
    template_id = str(template.get("template_id") or source.stem)
    profile_hash = adapters["sha256_json"](profile)
    created_at = time.time()
    revision_id = adapters["slug"](
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
            "release_channel": str(
                template.get("release_channel") or template.get("publish_status") or "draft"
            ),
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
    adapters = _site_profile_adapters()
    template_id = adapters["slug"](template.get("template_id") or Path(path).stem)
    try:
        rel = Path(path).resolve().relative_to(Path(template_root).resolve())
    except Exception:
        rel = Path(path).name
    rel_hash = hashlib.sha256(str(rel).encode("utf-8")).hexdigest()[:10]
    return Path(template_root) / "_template_revisions" / template_id / rel_hash


def _template_release_payload(release: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(release, dict):
        return {}
    fields = _site_profile_adapters()["template_release_fields"]
    payload = {
        field: str(release.get(field) or "").strip()
        for field in fields
        if field in release and str(release.get(field) or "").strip()
    }
    if "publish_status" in release:
        payload["publish_status"] = str(release.get("publish_status") or "").strip()
    return payload


def _customer_project_template_release_request_dir(template_root: Path, template_id: str) -> Path:
    return Path(template_root) / "_template_release_requests" / _site_profile_adapters()["slug"](
        template_id or "template"
    )


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


def _write_release_request_file(path: Path, payload: dict[str, Any]) -> None:
    storage_payload = copy.deepcopy(payload)
    storage_payload.pop("request_path", None)
    _write_json_atomic(path, storage_payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{int(time.time() * 1000)}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


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
        "apply_started_at": request.get("apply_started_at"),
        "apply_failed_at": request.get("apply_failed_at"),
        "apply_failure_reason": str(request.get("apply_failure_reason") or ""),
        "release": _mapping(request.get("release")),
        "release_result": _mapping(request.get("release_result")),
        "current_template_sha256": str(request.get("current_template_sha256") or ""),
        "current_template_package": _mapping(request.get("current_template_package")),
        "proposed_template_package": _mapping(request.get("proposed_template_package")),
        "applied_template_package": _mapping(request.get("applied_template_package")),
        "applied_revision": _mapping(request.get("applied_revision")),
    }


def _release_notes_customer_context(payload: dict[str, Any]) -> dict[str, str]:
    clean = _site_profile_adapters()["clean_mapping"](payload)
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
    return _site_profile_adapters()["slug"](basis)


def _template_release_notes_proposal_insert(
    context: dict[str, Any],
    notes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    adapters = _site_profile_adapters()
    customer_name = str(context.get("customer_name") or "Customer")
    project_name = str(context.get("project_name") or "AskMe Robot Deployment")
    template_ids = [str(item.get("template_id") or "") for item in notes if item.get("template_id")]
    scenario_ids = sorted({
        scenario
        for item in notes
        for scenario in adapters["string_list"](
            _mapping(item.get("applicability_scope")).get("scenarios")
        )
    })
    dependency_count = sum(
        len(item.get("dependency_matrix") if isinstance(item.get("dependency_matrix"), list) else [])
        for item in notes
    )
    acceptance_coverage = [
        {
            "template_id": str(item.get("template_id") or ""),
            "scenario_count": len(
                adapters["string_list"](_mapping(item.get("applicability_scope")).get("scenarios"))
            ),
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
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #17211b; }}
    header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }}
    header img {{ max-height: 56px; max-width: 180px; }}
    .logo-placeholder {{ background: #0f5a3d; color: #fff; padding: 12px 16px; border-radius: 8px; font-weight: 700; }}
    h1 {{ margin: 0; font-size: 24px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d8e1dc; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #edf6f1; }}
    .claim {{ background: #f4faf7; border: 1px solid #d8e1dc; padding: 12px; border-radius: 8px; margin: 12px 0; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <header>
    {logo_html}
    <div>
      <h1>{html.escape(str(proposal.get("section_title") or "AskMe Template Release Notes"))}</h1>
      <div>{html.escape(str(context.get("customer_name") or "Customer"))}</div>
    </div>
  </header>
  <div class="claim">{html.escape(str(proposal.get("customer_message") or bundle.get("customer_claim") or ""))}</div>
  <h2>Approved templates</h2>
  <table>
    <thead><tr><th>Template</th><th>Version</th><th>Channel</th><th>Status</th><th>Customer status</th><th>Approved by</th></tr></thead>
    <tbody>{note_rows}</tbody>
  </table>
  <h2>Safe claims</h2>
  <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in safe_claims)}</ul>
  <h2>Delivery boundaries</h2>
  <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in boundaries)}</ul>
  <h2>Summary</h2>
  <p>Approved releases: {int(summary.get("approved_release_count") or 0)}; templates: {int(summary.get("template_count") or 0)}.</p>
</body>
</html>"""


def _site_profile_adapters() -> dict[str, Any]:
    from askme.pipeline.field.customer_project_managed_objects import (
        managed_object_catalog_from_site_profile,
    )
    from askme.pipeline.field.customer_project_template_delivery import (
        _customer_delivery_surface,
        _template_delivery_checklist,
        _template_delivery_summary,
        _template_package_summary,
    )
    from askme.pipeline.field.customer_project_template_support import (
        TEMPLATE_PUBLISH_STATUSES,
        TEMPLATE_RELEASE_FIELDS,
        _clean_mapping,
        _clean_nested_mapping,
        _find_template_path,
        _is_semver,
        _sha256_json,
        _slug,
        _string_list,
        _write_yaml,
        load_field_site_profile,
    )
    from askme.pipeline.field.field_site_validation import (
        validate_field_site_profile,
    )

    return {
        "clean_mapping": _clean_mapping,
        "clean_nested_mapping": _clean_nested_mapping,
        "customer_delivery_surface": _customer_delivery_surface,
        "find_template_path": _find_template_path,
        "is_semver": _is_semver,
        "load_field_site_profile": load_field_site_profile,
        "managed_object_catalog_from_site_profile": managed_object_catalog_from_site_profile,
        "sha256_json": _sha256_json,
        "slug": _slug,
        "string_list": _string_list,
        "template_delivery_checklist": _template_delivery_checklist,
        "template_delivery_summary": _template_delivery_summary,
        "template_package_summary": _template_package_summary,
        "template_publish_statuses": TEMPLATE_PUBLISH_STATUSES,
        "template_release_fields": TEMPLATE_RELEASE_FIELDS,
        "validate_field_site_profile": validate_field_site_profile,
        "write_yaml": _write_yaml,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "create_customer_project_template_release_request",
    "customer_project_template_release_notes",
    "export_customer_project_template_release_notes_bundle",
    "list_customer_project_template_release_requests",
    "list_customer_project_template_revisions",
    "review_customer_project_template_release_request",
    "update_customer_project_template_release",
]
