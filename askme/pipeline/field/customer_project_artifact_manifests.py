"""Customer-project artifact manifest and payload hash builders.

These helpers are leaf-level artifact contracts shared by package, acceptance
dossier, and proposal bundle flows. Keep them free of ``field_site_profile`` so
artifact import/export can be migrated without circular imports.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from askme.pipeline.field.customer_project_template_support import (
    _delivery_namespace,
    _delivery_tenant_id,
    _mapping,
    _sha256_json,
    _string_list,
)


def _clean_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("${"):
        return ""
    return text


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
    delivery_chain = _mapping(package.get("delivery_chain"))
    delivery_chain_steps = (
        delivery_chain.get("steps") if isinstance(delivery_chain.get("steps"), list) else []
    )
    delivery_chain_summary = _mapping(delivery_chain.get("summary"))
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
        "delivery_chain_status": str(
            delivery_chain.get("overall_status")
            or delivery_chain_summary.get("overall_status")
            or "unknown"
        ),
        "delivery_chain_step_count": len(delivery_chain_steps),
        "delivery_chain_blocked_count": int(delivery_chain_summary.get("blocked_count") or 0),
        "delivery_chain_manual_check_count": int(delivery_chain_summary.get("manual_check_count") or 0),
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
    delivery_chain = _mapping(dossier.get("delivery_chain"))
    delivery_chain_steps = (
        delivery_chain.get("steps") if isinstance(delivery_chain.get("steps"), list) else []
    )
    delivery_chain_summary = _mapping(delivery_chain.get("summary"))
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
        "delivery_chain_status": str(
            delivery_chain.get("overall_status")
            or delivery_chain_summary.get("overall_status")
            or "unknown"
        ),
        "delivery_chain_step_count": len(delivery_chain_steps),
        "delivery_chain_blocked_count": int(delivery_chain_summary.get("blocked_count") or 0),
        "delivery_chain_manual_check_count": int(delivery_chain_summary.get("manual_check_count") or 0),
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


__all__ = [
    "_customer_project_acceptance_dossier_manifest",
    "_customer_project_acceptance_dossier_payload_sha256",
    "_customer_project_package_manifest",
    "_customer_project_package_payload_sha256",
    "_customer_project_proposal_bundle_payload_sha256",
]
