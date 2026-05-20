"""Public customer-project artifact import, export, and verification API.

This module owns reusable customer-project package, acceptance dossier,
proposal bundle, import, diff, and package verification flows. Keep it free of
``field_site_profile`` so product APIs can depend on artifact contracts without
pulling in the legacy compatibility surface.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_acceptance import (
    _build_customer_project_acceptance_dossier,
    _delivery_chain_step,
    _delivery_chain_summary,
    customer_project_acceptance_report,
    verify_customer_project_acceptance_dossier,
    verify_customer_project_proposal_bundle,
)
from askme.pipeline.field.customer_project_artifact_manifests import (
    _customer_project_package_manifest,
    _customer_project_package_payload_sha256,
)
from askme.pipeline.field.customer_project_implementation_handoff import (
    _customer_project_implementation_handoff,
)
from askme.pipeline.field.customer_project_managed_objects import (
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_package_assessment import (
    _customer_project_package_acceptance_summary,
    _customer_project_package_reuse_assessment,
)
from askme.pipeline.field.customer_project_package_html import (
    _render_customer_project_acceptance_dossier_html,
    _render_customer_project_proposal_bundle_html,
)
from askme.pipeline.field.customer_project_package_rules import (
    _customer_project_package_action_plan,
    _customer_project_package_delivery_gate,
    _customer_project_package_import_gate_result,
)
from askme.pipeline.field.customer_project_profile_operations import (
    upsert_customer_project_profile,
)
from askme.pipeline.field.customer_project_profiles import (
    _customer_payload,
    _customer_profile_target,
    _customer_project_collision_candidates,
    _normalize_customer_project_profile,
    find_site_profile_path,
)
from askme.pipeline.field.customer_project_scope import (
    _customer_delivery_filename_parts,
    _customer_project_profile_diff,
    _delivery_scope_payload,
    _delivery_scope_payload_from_customer_site,
)
from askme.pipeline.field.customer_project_template_delivery import _customer_delivery_surface
from askme.pipeline.field.customer_project_template_release import (
    export_customer_project_template_release_notes_bundle,
)
from askme.pipeline.field.customer_project_template_support import (
    _delivery_namespace,
    _delivery_tenant_id,
    _mapping,
    _sha256_json,
    _slug,
    _string_list,
    load_field_site_profile,
    site_profile_env_references,
)
from askme.pipeline.field.field_site_catalog import build_site_profile_report
from askme.pipeline.field.field_site_runtime_config import render_site_profile_env_template
from askme.pipeline.field.field_site_validation import validate_field_site_profile
from askme.pipeline.field.paths import (
    DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
    DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
    DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT,
)

__all__ = [
    "diff_customer_project_package",
    "export_customer_project_acceptance_dossier",
    "export_customer_project_package",
    "export_customer_project_proposal_bundle",
    "export_customer_project_template_release_notes_bundle",
    "import_customer_project_package",
    "verify_customer_project_acceptance_dossier",
    "verify_customer_project_package",
    "verify_customer_project_proposal_bundle",
]


def export_customer_project_package(
    profile_root: Path,
    identifier: str,
    *,
    output_root: Path = DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
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
    delivery_chain = _customer_project_package_delivery_chain(
        customer=customer,
        site=_mapping(profile.get("site")),
        profile=profile,
        acceptance_summary=acceptance_summary,
        binding_readiness=_mapping(managed_object_catalog.get("binding_readiness_summary")),
        resource_catalog=_mapping(managed_object_catalog.get("resource_catalog_summary")),
        package_delivery_gate=package_delivery_gate,
        report=report,
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
        "delivery_chain": delivery_chain,
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


def _customer_project_package_delivery_chain(
    *,
    customer: dict[str, Any],
    site: dict[str, Any],
    profile: dict[str, Any],
    acceptance_summary: dict[str, Any],
    binding_readiness: dict[str, Any],
    resource_catalog: dict[str, Any],
    package_delivery_gate: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return a customer-readable chain that travels with reusable export packages."""
    template = _mapping(profile.get("template"))
    runtime_blueprint = str(
        profile.get("runtime_blueprint")
        or profile.get("runtime_blueprint_name")
        or template.get("runtime_blueprint")
        or ""
    )
    scope_ready = all(
        str(value or "").strip()
        for value in (
            customer.get("customer_id"),
            customer.get("project_id"),
            site.get("site_id"),
        )
    )
    steps = [
        _delivery_chain_step(
            step_id="project_scope",
            label="Customer project scope",
            status="ready" if scope_ready else "blocked",
            customer_question="Which customer, project, and site does this export package cover?",
            evidence=(
                f"customer_id={customer.get('customer_id') or '-'}; "
                f"project_id={customer.get('project_id') or '-'}; "
                f"site_id={site.get('site_id') or '-'}"
            ),
            next_step="Set customer_id, project_id, and site_id before exporting the package."
            if not scope_ready
            else "Project scope is explicit in the package.",
            endpoint="/api/field/customer-projects",
            source_surface_id="customer_projects",
        ),
        _delivery_chain_step(
            step_id="template_market",
            label="Industry template",
            status="ready" if template.get("template_id") else "manual_check",
            customer_question="Which reusable industry template was this package derived from?",
            evidence=f"template_id={template.get('template_id') or '-'}; industry={customer.get('industry') or 'unspecified'}",
            next_step="Bind the customer project to a reusable template before reuse."
            if not template.get("template_id")
            else "Reusable template identity is recorded.",
            endpoint="/api/field/customer-project-templates",
            source_surface_id="template_market",
        ),
        _delivery_chain_step(
            step_id="managed_object_directory",
            label="Managed object directory",
            status=str(acceptance_summary.get("overall_status") or "blocked"),
            customer_question="Which real onsite objects are covered by this package?",
            evidence=(
                f"objects={acceptance_summary.get('object_count') or 0}; "
                f"ready={acceptance_summary.get('ready_object_count') or 0}; "
                f"manual={acceptance_summary.get('manual_check_object_count') or 0}; "
                f"blocked={acceptance_summary.get('blocked_object_count') or 0}"
            ),
            next_step=str(acceptance_summary.get("customer_status") or "完成现场对象验收。"),
            endpoint="/api/field/customer-projects/managed-object-directory",
            source_surface_id="managed_objects",
        ),
        _delivery_chain_step(
            step_id="capability_resource_binding",
            label="Capability and resource binding",
            status=str(binding_readiness.get("overall_status") or "blocked"),
            customer_question="Are every object's resources bound before customer handoff?",
            evidence=(
                f"resources={resource_catalog.get('resource_count') or 0}; "
                f"unregistered={binding_readiness.get('unregistered_resource_count') or 0}; "
                f"blocked={binding_readiness.get('blocked_object_count') or 0}"
            ),
            next_step=str(
                binding_readiness.get("customer_status")
                or "Register and bind vision models, sensor protocols, skill packages, and acceptance tests."
            ),
            endpoint="/api/field/customer-project-resource-catalog",
            source_surface_id="delivery_resources",
        ),
        _delivery_chain_step(
            step_id="runtime_blueprint",
            label="Runtime blueprint",
            status="ready" if runtime_blueprint else "manual_check",
            customer_question="Which robot runtime plan can execute this project package?",
            evidence=f"runtime_blueprint={runtime_blueprint or '-'}",
            next_step="Bind a customer-visible runtime blueprint before production claims."
            if not runtime_blueprint
            else "Runtime blueprint identity is recorded.",
            endpoint="/api/blueprints",
            source_surface_id="runtime_blueprint_binding",
        ),
        _delivery_chain_step(
            step_id="acceptance_package",
            label="Acceptance and handoff package",
            status=str(package_delivery_gate.get("delivery_gate_status") or report.get("status") or "manual_check"),
            customer_question="Can this package be handed to delivery without hidden blockers?",
            evidence=str(package_delivery_gate.get("customer_status") or package_delivery_gate.get("summary") or ""),
            next_step=str(
                package_delivery_gate.get("next_step")
                or "Resolve package preflight blockers before customer handoff."
            ),
            endpoint="/api/field/customer-projects/{identifier}/export",
            source_surface_id="package_delivery_gate",
        ),
    ]
    summary = _delivery_chain_summary(steps)
    return {
        "chain_type": "askme.customer_project.delivery_chain.v1",
        "overall_status": summary["overall_status"],
        "step_count": len(steps),
        "summary": summary,
        "steps": steps,
        "policy": {
            "runtime_blueprint_is_required_before_customer_claim": True,
            "capability_resources_must_be_bound_to_managed_objects": True,
            "acceptance_package_must_reference_real_evidence": True,
        },
    }


def export_customer_project_acceptance_dossier(
    profile_root: Path,
    identifier: str,
    *,
    output_root: Path = DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
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
    output_root: Path = DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT,
    package_output_root: Path | None = None,
    dossier_output_root: Path | None = None,
    check_env: bool = True,
    release_limit: int = 50,
) -> dict[str, Any]:
    """Export a customer-facing proposal bundle bound to one customer project."""
    if package_output_root is None and output_root != DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT:
        package_output_root = output_root.parent / DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT.name
    if dossier_output_root is None and output_root != DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT:
        dossier_output_root = output_root.parent / DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT.name
    package_result = export_customer_project_package(
        profile_root,
        identifier,
        output_root=package_output_root or DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
    )
    if not package_result.get("accepted"):
        return {"accepted": False, "reason": package_result.get("reason") or "package_export_failed"}
    dossier_result = export_customer_project_acceptance_dossier(
        profile_root,
        identifier,
        output_root=dossier_output_root or DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
        check_env=check_env,
    )
    if not dossier_result.get("accepted"):
        return {"accepted": False, "reason": dossier_result.get("reason") or "dossier_export_failed"}

    package = _mapping(package_result.get("package"))
    dossier = _mapping(dossier_result.get("dossier"))
    package_chain = _mapping(package.get("delivery_chain"))
    dossier_chain = _mapping(dossier.get("delivery_chain")) or package_chain
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
            "delivery_chain": package_chain,
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
            "delivery_chain": dossier_chain,
            "gates": dossier.get("gates") if isinstance(dossier.get("gates"), list) else [],
        },
        "customer_readable_delivery": {
            "delivery_chain": dossier_chain,
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
        "delivery_chain": dossier_chain,
        "launch_readiness": _mapping(dossier.get("launch_readiness")),
        "approved_template_release_bundle": release_bundle_for_payload,
        "proposal_insert": _mapping(release_bundle_for_payload.get("proposal_insert")),
        "delivery_boundary": (
            "本方案包是面向客户的规划材料；生产上线仍需要项目范围审批、现场证据、"
            "真实通知测试和机器人运行验收。"
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
            "proposal_delivery_chain_status": str(dossier_chain.get("overall_status") or "unknown"),
            "proposal_delivery_chain_step_count": len(
                dossier_chain.get("steps") if isinstance(dossier_chain.get("steps"), list) else []
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

    delivery_chain = _mapping(package.get("delivery_chain"))
    delivery_chain_steps = (
        delivery_chain.get("steps") if isinstance(delivery_chain.get("steps"), list) else []
    )
    manifest_chain_count = manifest.get("delivery_chain_step_count")
    if manifest_chain_count not in (None, "") and int(manifest_chain_count or 0) != len(delivery_chain_steps):
        errors.append("manifest.delivery_chain_step_count mismatch")
    manifest_chain_status = str(manifest.get("delivery_chain_status") or "")
    package_chain_status = str(delivery_chain.get("overall_status") or "")
    if manifest_chain_status and package_chain_status and manifest_chain_status != package_chain_status:
        errors.append("manifest.delivery_chain_status mismatch")

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


def _utc_timestamp(value: float | None = None) -> str:
    current = time.time() if value is None else value
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current))


def _clean_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("${"):
        return ""
    return text
