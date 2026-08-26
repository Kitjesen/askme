"""Customer project onsite acceptance, review, and signoff workflow.

This module owns the customer-facing acceptance closure for a field deployment:
onsite evidence receipts, readiness projection, delivery-owner review, customer
signoff, and acceptance dossier integrity verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_artifact_manifests import (
    _customer_project_acceptance_dossier_manifest,
    _customer_project_acceptance_dossier_payload_sha256,
    _customer_project_proposal_bundle_payload_sha256,
)
from askme.pipeline.field.customer_project_evidence_inventory import (
    _customer_project_evidence_inventory,
    _evidence_file_inventory,
    _evidence_file_modified_at,
    _evidence_url,
)
from askme.pipeline.field.customer_project_execution_bindings import (
    build_customer_project_execution_bindings,
)
from askme.pipeline.field.customer_project_managed_objects import (
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_package_assessment import (
    _customer_project_package_acceptance_summary,
)
from askme.pipeline.field.customer_project_profiles import (
    _customer_payload,
    _normalize_customer_project_profile,
    _snapshot_customer_project_revision,
    find_site_profile_path,
)
from askme.pipeline.field.customer_project_scope import (
    _delivery_scope_payload,
    _delivery_scope_payload_from_customer_site,
    _same_delivery_project_scope,
)
from askme.pipeline.field.customer_project_template_support import (
    _sha256_json,
    _slug,
    _string_list,
    _write_yaml,
    load_field_site_profile,
    site_profile_env_references,
)
from askme.pipeline.field.field_site_catalog import (
    _customer_project_delivery_workflow,
    build_site_profile_report,
)
from askme.pipeline.field.field_site_runtime_config import (
    field_operations_config_from_site_profile,
)
from askme.pipeline.field.paths import PROJECT_ROOT

__all__ = [
    "_FIELD_READINESS_EVIDENCE_DEFAULTS",
    "ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES",
    "ONSITE_ACCEPTANCE_EVIDENCE_TYPES",
    "ONSITE_ACCEPTANCE_STATUSES",
    "ACCEPTANCE_REVIEW_DECISIONS",
    "CUSTOMER_SIGNOFF_DECISIONS",
    "list_customer_project_onsite_evidence",
    "register_customer_project_onsite_evidence",
    "customer_project_acceptance_closure",
    "register_customer_project_acceptance_review",
    "list_customer_project_customer_signoffs",
    "register_customer_project_customer_signoff",
    "customer_project_acceptance_report",
    "_customer_project_launch_readiness",
    "_launch_readiness_gate",
    "_customer_project_launch_gate_status",
    "_customer_project_launch_gate_next_step",
    "_customer_project_launch_rollup_status",
    "_execution_binding_report_contracts",
    "_customer_project_field_readiness",
    "_compact_field_readiness",
    "_compact_evidence_report",
    "_customer_project_raw_onsite_evidence",
    "_customer_project_onsite_evidence_payload",
    "_customer_project_onsite_evidence_payload_from_receipts",
    "_customer_project_auto_onsite_evidence_receipts",
    "_customer_project_auto_onsite_evidence_receipt",
    "_customer_project_onsite_evidence_receipts",
    "_customer_project_onsite_evidence_summary",
    "_customer_project_raw_acceptance_reviews",
    "_customer_project_acceptance_reviews",
    "_customer_project_acceptance_review_gate",
    "_customer_project_raw_customer_signoffs",
    "_customer_project_customer_signoffs",
    "_customer_project_customer_signoff_gate_snapshot",
    "_customer_project_customer_signoff_handoff_materials",
    "_customer_project_customer_signoff_payload_sha256",
    "_customer_project_customer_signoff_gate",
    "_customer_project_acceptance_evidence_timeline",
    "_customer_project_acceptance_closure_next_step",
    "_customer_project_latest_proposal_verification",
    "_customer_project_latest_audit_export",
    "_recent_json_files",
    "_read_json_file",
    "_audit_manifest_matches_scope",
    "_audit_records_hash_matches",
    "_normalize_onsite_evidence_type",
    "_normalize_onsite_evidence_status",
    "_normalize_acceptance_review_decision",
    "_normalize_customer_signoff_decision",
    "_normalize_sha256_hex",
    "_float_value",
    "_customer_project_field_readiness_gates",
    "_readiness_status",
    "_boolean_gate_status",
    "_reports_evidence",
    "_customer_project_site_acceptance_checklist",
    "_latest_onsite_receipts_by_type",
    "_onsite_acceptance_checklist_item",
    "_build_customer_project_acceptance_dossier",
    "_customer_project_acceptance_dossier_verification",
    "verify_customer_project_proposal_bundle",
    "verify_customer_project_acceptance_dossier",
]


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
        evidence_roots=_onsite_evidence_allowed_roots(profile_root),
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
    tier = _manual_onsite_evidence_tier(evidence.get("evidence_tier"))
    if tier == "":
        return {
            "accepted": False,
            "reason": "unsupported_onsite_evidence_trust_tier",
            "allowed_tiers": ["acceptance_candidate"],
        }
    evidence_path = str(evidence.get("path") or evidence.get("evidence_path") or "").strip()
    inventory = (
        _evidence_file_inventory(
            evidence_path,
            evidence_url=_evidence_url(evidence_path),
            allowed_roots=_onsite_evidence_allowed_roots(profile_root),
        )
        if evidence_path
        else {}
    )
    trust = _customer_project_onsite_evidence_trust(
        evidence_type=evidence_type,
        status=status,
        evidence_path=evidence_path,
        inventory=inventory,
    )
    if not trust["accepted"]:
        return {
            "accepted": False,
            "reason": trust["reason"],
            "trust": trust,
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
        "evidence_tier": tier,
        "production_eligible": False,
        "verified_evidence": trust["verified"],
        "trust_status": trust["status"],
        "trust_reason": trust["reason"],
        "acceptance_gate_eligible": trust["acceptance_gate_eligible"],
    }
    if inventory.get("error"):
        receipt["error"] = str(inventory.get("error") or "")
    receipts = _customer_project_raw_onsite_evidence(profile)
    receipts.append(receipt)
    profile["onsite_acceptance_evidence"] = receipts
    _write_yaml(path, profile)
    payload = _customer_project_onsite_evidence_payload(
        profile,
        evidence_roots=_onsite_evidence_allowed_roots(profile_root),
    )
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
            "next_step": str(
                site_checklist.get("customer_message") or "Review site acceptance checklist."
            ),
        },
        review_gate,
        {
            "gate_id": "dossier_verification",
            "label": "Dossier verification",
            "status": "ready" if dossier_verification.get("valid") else "blocked",
            "evidence": str(dossier_verification.get("reason") or "unknown"),
            "next_step": "Regenerate the acceptance dossier."
            if not dossier_verification.get("valid")
            else "Dossier manifest is self-consistent.",
        },
        {
            "gate_id": "proposal_verification",
            "label": "Proposal verification",
            "status": str(proposal_verification.get("status") or "manual_check"),
            "evidence": str(
                proposal_verification.get("evidence") or "No matching proposal bundle found."
            ),
            "next_step": str(
                proposal_verification.get("next_step")
                or "Export and verify the customer proposal bundle."
            ),
        },
        {
            "gate_id": "audit_export",
            "label": "Audit export",
            "status": str(audit_export.get("status") or "manual_check"),
            "evidence": str(audit_export.get("evidence") or "No matching audit export found."),
            "next_step": str(
                audit_export.get("next_step")
                or "Create a scoped audit export for customer handoff."
            ),
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
        "evidence_timeline": _customer_project_acceptance_evidence_timeline(
            onsite, reviews, signoffs
        ),
        "blocked_uses": (
            [
                "无人值守生产上线",
                "超出已签收项目、对象和证据范围的验收承诺",
            ]
            if overall == "accepted_by_customer"
            else [
                "无人值守生产上线",
                "无人工复核的最终验收承诺",
            ]
        ),
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
            for item in (
                review.get("evidence_refs") if isinstance(review.get("evidence_refs"), list) else []
            )
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
    evidence_refs = [
        str(item)
        for item in (
            signoff.get("evidence_refs") if isinstance(signoff.get("evidence_refs"), list) else []
        )
        if str(item).strip()
    ]
    evidence_ref_assessment = _customer_project_customer_signoff_evidence_ref_assessment(
        closure_before,
        evidence_refs,
        required=decision == "accepted",
    )
    if decision == "accepted" and not evidence_ref_assessment["valid"]:
        return {
            "accepted": False,
            "reason": evidence_ref_assessment["reason"],
            "evidence_ref_assessment": evidence_ref_assessment,
            "closure": closure_before,
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
        "evidence_ref_assessment": evidence_ref_assessment,
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
        evidence_roots=_onsite_evidence_allowed_roots(profile_root),
    )["onsite_acceptance_evidence"]
    onsite_summary = _mapping(onsite_evidence.get("summary"))
    env_refs = site_profile_env_references(profile)
    missing_env = [item for item in env_refs if item.get("required") and not item.get("configured")]
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
            "next_step": "Fix site profile validation errors."
            if errors
            else "Site profile schema is valid.",
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
            "label": "现场对象执行绑定",
            "status": str(execution_summary.get("overall_status") or "blocked"),
            "evidence": (
                f"{execution_summary.get('ready_object_count') or 0}/"
                f"{execution_summary.get('object_count') or 0} 个对象具备可执行接入计划"
            ),
            "next_step": str(execution_bindings.get("next_step") or "客户签收前先生成执行绑定。"),
        },
        {
            "gate_id": "deployment_credentials",
            "label": "部署凭证",
            "status": "manual_check" if missing_env else "ready",
            "evidence": f"{len(missing_env)} 个环境变量缺失",
            "next_step": (
                "现场验收前先补齐钉钉响应人和签名设备密钥。"
                if missing_env
                else "部署环境引用已配置。"
            ),
        },
        {
            "gate_id": "onsite_acceptance_boundary",
            "label": "现场验收边界",
            "status": str(onsite_summary.get("overall_status") or "manual_check"),
            "evidence": (
                f"{onsite_summary.get('passed_required_count') or 0}/"
                f"{onsite_summary.get('required_count') or len(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES)} "
                f"个必需现场证据回执已通过；回执总数={onsite_summary.get('receipt_count') or 0}"
            ),
            "next_step": str(
                onsite_summary.get("next_step") or "生产上线声明前先完成现场真实冒烟测试。"
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
            "ready_for_onsite_acceptance": "项目证据已具备，可进入现场验收。",
            "manual_check": "项目在客户签收前仍需交付复核。",
            "blocked": "项目存在阻断项，不能声明已具备验收条件。",
        }.get(overall, "项目验收状态未知。"),
        "release_claim": (
            "本报告仅支持演示/试点验收复核；生产上线仍需要设备、通知、语音和机器人运行的独立现场证据。"
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
    device_onboarding = _mapping(field_readiness.get("device_onboarding"))
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
            evidence=str(
                _mapping(gate_by_id.get("managed_object_execution_bindings")).get("evidence") or ""
            ),
            next_step=str(
                _mapping(gate_by_id.get("managed_object_execution_bindings")).get("next_step") or ""
            ),
        ),
        _launch_readiness_gate(
            gate_id="deployment_credentials",
            label="部署凭证和通知配置",
            status="manual_check" if missing_env else "ready",
            evidence=f"{len(missing_env)} 个必需部署值缺失",
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
                "类必需现场证据已通过"
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
            gate_id="field_device_onboarding",
            label="真实设备接入",
            status=_customer_project_device_onboarding_launch_status(device_onboarding),
            evidence=_customer_project_device_onboarding_evidence(device_onboarding),
            next_step=_customer_project_device_onboarding_next_step(device_onboarding),
        ),
        _launch_readiness_gate(
            gate_id="site_acceptance_checklist",
            label="客户现场验收清单",
            status=_customer_project_launch_gate_status(
                site_acceptance_checklist.get("overall_status")
            ),
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
            "blocked_count": len(
                [gate for gate in launch_gates if gate.get("status") == "blocked"]
            ),
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


def _customer_project_device_onboarding_launch_status(onboarding: dict[str, Any]) -> str:
    """Return the launch gate status for registered camera, sensor, and robot devices."""

    if onboarding.get("all_ready") is True:
        return "ready"
    if onboarding.get("available") is not True:
        return "blocked"
    if int(onboarding.get("blocked") or 0) > 0:
        return "blocked"
    if int(onboarding.get("total_device_count") or onboarding.get("registered") or 0) <= 0:
        return "blocked"
    if int(onboarding.get("ready") or 0) <= 0:
        return "blocked"
    return "manual_check"


def _customer_project_device_onboarding_acceptance_status(onboarding: dict[str, Any]) -> str:
    """Return a trial-acceptance status without overstating production readiness."""

    if onboarding.get("all_ready") is True:
        return "ready"
    return "manual_check"


def _customer_project_device_onboarding_evidence(onboarding: dict[str, Any]) -> str:
    if onboarding.get("available") is not True:
        return "device onboarding report is not available"
    return (
        f"ready={int(onboarding.get('ready') or 0)}, "
        f"manual={int(onboarding.get('manual_check') or 0)}, "
        f"blocked={int(onboarding.get('blocked') or 0)}, "
        f"total={int(onboarding.get('total_device_count') or onboarding.get('registered') or 0)}"
    )


def _customer_project_device_onboarding_next_step(onboarding: dict[str, Any]) -> str:
    if onboarding.get("all_ready") is True:
        return "已登记现场设备完成接入，并已绑定到客户现场对象。"
    if onboarding.get("available") is not True:
        return "通过现场运行就绪检查纳入设备接入状态。"
    if int(onboarding.get("blocked") or 0) > 0:
        return "客户生产上线前先修复阻断的现场设备。"
    if int(onboarding.get("ready") or 0) <= 0:
        return "发送带签名的现场上报，并至少将一个现场设备绑定到现场对象。"
    return "生产上线前完成所有已登记设备的接入。"


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
    if text in {
        "ready",
        "ok",
        "healthy",
        "passed",
        "production_ready",
        "ready_for_onsite_acceptance",
    }:
        return "ready"
    if text in {
        "manual_check",
        "ready_for_lab",
        "ready_for_acceptance",
        "ready_for_customer_signoff",
    }:
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
        for adapter in (
            plan.get("input_adapters") if isinstance(plan.get("input_adapters"), list) else []
        ):
            if not isinstance(adapter, dict):
                continue
            contract = _mapping(adapter.get("adapter_contract"))
            adapters.append(
                {
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
                }
            )
        skill_routes = []
        for route in plan.get("skill_routes") if isinstance(plan.get("skill_routes"), list) else []:
            if not isinstance(route, dict):
                continue
            skill_routes.append(
                {
                    "capability": str(route.get("capability") or route.get("resource_id") or ""),
                    "tool": str(route.get("tool") or ""),
                    "output_contract": str(route.get("output_contract") or ""),
                    "approval_policy": str(route.get("approval_policy") or ""),
                    "hardware_boundary": str(
                        route.get("hardware_boundary") or route.get("safety_boundary") or ""
                    ),
                }
            )
        contracts.append(
            {
                "object_id": str(plan.get("object_id") or ""),
                "display_name": str(plan.get("display_name") or plan.get("object_id") or ""),
                "overall_status": str(plan.get("overall_status") or ""),
                "input_adapters": adapters,
                "bridge_contract": _mapping(plan.get("bridge_contract")),
                "skill_routes": skill_routes,
            }
        )
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
        from askme.pipeline.field.field_operations import FieldOperationsService
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
    next_actions = (
        payload.get("next_actions") if isinstance(payload.get("next_actions"), list) else []
    )
    delivery_brief = (
        payload.get("delivery_brief") if isinstance(payload.get("delivery_brief"), dict) else {}
    )
    device_onboarding = _mapping(payload.get("device_onboarding"))
    device_onboarding_summary = _mapping(device_onboarding.get("summary"))
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
                "device_onboarding_report_available",
                "device_onboarding_no_blockers",
                "device_onboarding_has_ready_device",
                "device_onboarding_all_ready",
                "uses_real_hardware",
                "uses_external_services",
            )
        },
        "reports": {
            "scenario": _compact_evidence_report(payload.get("scenario_report")),
            "ingest_smoke": _compact_evidence_report(payload.get("smoke_report")),
            "voice_smoke": _compact_evidence_report(payload.get("voice_smoke_report")),
            "notification_smoke": _compact_evidence_report(
                payload.get("notification_smoke_report")
            ),
            "runtime_roundtrip": _compact_evidence_report(payload.get("runtime_roundtrip_report")),
        },
        "archive": {
            "path": str(_mapping(payload.get("archive")).get("path") or ""),
            "event_count": _mapping(payload.get("archive")).get("event_count") or 0,
            "scenario_ids": _mapping(payload.get("archive")).get("scenario_ids") or [],
            "sources": _mapping(payload.get("archive")).get("sources") or [],
            "trusted_device_event_count": _mapping(payload.get("archive")).get(
                "trusted_device_event_count"
            )
            or 0,
        },
        "device_trust": {
            "registered_device_count": _mapping(payload.get("device_trust")).get(
                "registered_device_count"
            )
            or 0,
            "signed_device_count": _mapping(payload.get("device_trust")).get("signed_device_count")
            or 0,
            "unsigned_device_count": _mapping(payload.get("device_trust")).get(
                "unsigned_device_count"
            )
            or 0,
            "all_registered_devices_signature_ready": _mapping(payload.get("device_trust")).get(
                "all_registered_devices_signature_ready"
            )
            is True,
        },
        "device_onboarding": {
            "available": device_onboarding.get("available") is True
            or device_onboarding.get("report_type") == "askme.field.device_onboarding_report.v1",
            "status": str(device_onboarding.get("status") or ""),
            "registered": int(
                device_onboarding_summary.get("registered")
                or device_onboarding_summary.get("registered_device_count")
                or 0
            ),
            "observed": int(device_onboarding_summary.get("observed") or 0),
            "ready": int(device_onboarding_summary.get("ready") or 0),
            "blocked": int(device_onboarding_summary.get("blocked") or 0),
            "manual_check": int(device_onboarding_summary.get("manual_check") or 0),
            "total_device_count": int(
                device_onboarding_summary.get("total_device_count")
                or device_onboarding_summary.get("registered")
                or device_onboarding_summary.get("registered_device_count")
                or 0
            ),
            "all_ready": device_onboarding_summary.get("all_ready") is True
            or bool(
                int(
                    device_onboarding_summary.get("total_device_count")
                    or device_onboarding_summary.get("registered")
                    or device_onboarding_summary.get("registered_device_count")
                    or 0
                )
                > 0
                and int(device_onboarding_summary.get("ready") or 0)
                == int(
                    device_onboarding_summary.get("total_device_count")
                    or device_onboarding_summary.get("registered")
                    or device_onboarding_summary.get("registered_device_count")
                    or 0
                )
                and int(device_onboarding_summary.get("blocked") or 0) == 0
                and int(device_onboarding_summary.get("manual_check") or 0) == 0
            ),
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
    evidence_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    receipts = _customer_project_onsite_evidence_receipts(profile, evidence_roots=evidence_roots)
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
        and gates.get("device_onboarding_no_blockers") is True
        and gates.get("device_onboarding_has_ready_device") is True
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
        "verified_evidence": True,
        "trust_status": "verified",
        "trust_reason": "field_readiness_auto_backfill_verified_real_link",
        "acceptance_gate_eligible": True,
        "auto_backfill": {
            "source": "field_readiness",
            "profile_scope": _delivery_scope_payload(profile),
        },
    }


def _customer_project_onsite_evidence_receipts(
    profile: dict[str, Any],
    *,
    evidence_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for item in _customer_project_raw_onsite_evidence(profile):
        evidence_type = _normalize_onsite_evidence_type(
            item.get("evidence_type") or item.get("type")
        )
        status = _normalize_onsite_evidence_status(item.get("status"))
        path = str(item.get("path") or item.get("evidence_path") or "").strip()
        inventory = (
            _evidence_file_inventory(
                path,
                evidence_url=_evidence_url(path),
                allowed_roots=evidence_roots,
            )
            if path
            else {}
        )
        receipt = {
            "receipt_type": str(
                item.get("receipt_type") or "askme.customer_project_onsite_evidence"
            ),
            "receipt_version": int(item.get("receipt_version") or 1),
            "receipt_id": str(
                item.get("receipt_id") or _slug(f"{evidence_type}-{item.get('recorded_at') or ''}")
            ),
            "recorded_at": _float_value(item.get("recorded_at")),
            "operator_id": str(item.get("operator_id") or "system"),
            "reason": str(item.get("reason") or ""),
            "evidence_type": evidence_type,
            "status": status,
            "source": str(item.get("source") or ""),
            "label": str(item.get("label") or evidence_type.replace("_", " ").title()),
            "summary": str(item.get("summary") or item.get("note") or ""),
            "path": path,
            "evidence_url": str(
                inventory.get("evidence_url") or item.get("evidence_url") or _evidence_url(path)
            ),
            "exists": bool(inventory.get("exists")) if path else bool(item.get("exists")),
            "size_bytes": int(inventory.get("size_bytes") or item.get("size_bytes") or 0),
            "sha256": str(inventory.get("sha256") or item.get("sha256") or ""),
            "project_scope": _mapping(item.get("project_scope")),
            "managed_object_id": str(item.get("managed_object_id") or ""),
            "event_id": str(item.get("event_id") or ""),
            "runtime_run_id": str(item.get("runtime_run_id") or ""),
            "external_reference": str(item.get("external_reference") or ""),
            "evidence_tier": _stored_onsite_evidence_tier(item),
            "production_eligible": False,
        }
        trust = _customer_project_onsite_evidence_trust(
            evidence_type=evidence_type,
            status=status,
            evidence_path=path,
            inventory=inventory,
        )
        receipt["verified_evidence"] = trust["verified"]
        receipt["trust_status"] = trust["status"]
        receipt["trust_reason"] = trust["reason"]
        receipt["acceptance_gate_eligible"] = trust["acceptance_gate_eligible"]
        if status == "passed" and not trust["accepted"]:
            receipt["original_status"] = "passed"
            receipt["status"] = "manual_check"
        if inventory.get("error") or item.get("error"):
            receipt["error"] = str(inventory.get("error") or item.get("error") or "")
        receipts.append(receipt)
    receipts.sort(key=lambda receipt: float(receipt.get("recorded_at") or 0), reverse=True)
    return receipts


def _manual_onsite_evidence_tier(value: Any) -> str:
    text = str(value or "acceptance_candidate").strip()
    if text in {"", "acceptance_candidate"}:
        return "acceptance_candidate"
    return ""


def _onsite_evidence_allowed_roots(profile_root: Path) -> tuple[Path, ...]:
    try:
        root = Path(profile_root).resolve()
    except OSError:
        return ()
    parent = root.parent
    if parent == root or parent == Path(root.anchor):
        return (root,)
    return (root, parent)


def _stored_onsite_evidence_tier(item: dict[str, Any]) -> str:
    text = str(item.get("evidence_tier") or "acceptance_candidate").strip()
    if text == "site_acceptance" and (
        item.get("source") == "field_readiness_auto_backfill"
        or isinstance(item.get("auto_backfill"), dict)
    ):
        return "site_acceptance"
    if text in {"acceptance_candidate", "lab_rehearsal"}:
        return text
    return "acceptance_candidate"


def _customer_project_onsite_evidence_trust(
    *,
    evidence_type: str,
    status: str,
    evidence_path: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    if status != "passed":
        return {
            "accepted": True,
            "verified": False,
            "status": "not_required",
            "reason": "non_passed_receipt_does_not_claim_acceptance",
            "acceptance_gate_eligible": False,
        }
    if evidence_type not in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES:
        return {
            "accepted": True,
            "verified": False,
            "status": "manual",
            "reason": "non_required_evidence_not_used_for_acceptance_gate",
            "acceptance_gate_eligible": False,
        }
    if not evidence_path:
        return {
            "accepted": False,
            "verified": False,
            "status": "unverified",
            "reason": "passed_required_onsite_evidence_requires_path",
            "acceptance_gate_eligible": False,
        }
    if inventory.get("exists") is not True:
        return {
            "accepted": False,
            "verified": False,
            "status": "unverified",
            "reason": "passed_required_onsite_evidence_path_not_found",
            "acceptance_gate_eligible": False,
        }
    if not str(inventory.get("sha256") or ""):
        return {
            "accepted": False,
            "verified": False,
            "status": "unverified",
            "reason": "passed_required_onsite_evidence_sha256_missing",
            "acceptance_gate_eligible": False,
        }
    return {
        "accepted": True,
        "verified": True,
        "status": "manual_check",
        "reason": "manual_onsite_evidence_requires_verified_readiness_auto_backfill",
        "acceptance_gate_eligible": False,
    }


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
        and _mapping(latest_by_type.get(item)).get("acceptance_gate_eligible") is True
    ]
    failed_required = [
        item
        for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES
        if _mapping(latest_by_type.get(item)).get("status") == "failed"
    ]
    manual_required = [
        item
        for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES
        if (
            _mapping(latest_by_type.get(item)).get("status") == "manual_check"
            or (
                _mapping(latest_by_type.get(item)).get("status") == "passed"
                and _mapping(latest_by_type.get(item)).get("acceptance_gate_eligible") is not True
            )
        )
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
            "review_type": str(
                item.get("review_type") or "askme.customer_project_acceptance_review"
            ),
            "review_version": int(item.get("review_version") or 1),
            "review_id": str(
                item.get("review_id")
                or _slug(f"{item.get('decision')}-{item.get('reviewed_at') or ''}")
            ),
            "reviewed_at": _float_value(item.get("reviewed_at")),
            "operator_id": str(item.get("operator_id") or "system"),
            "decision": _normalize_acceptance_review_decision(item.get("decision")),
            "reason": str(item.get("reason") or ""),
            "risk_acknowledgement": item.get("risk_acknowledgement") is True,
            "evidence_refs": [
                str(ref)
                for ref in (
                    item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
                )
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
        next_step = str(
            latest_review.get("reason") or "Resolve review notes before customer signoff."
        )
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
            "signoff_type": str(
                item.get("signoff_type") or "askme.customer_project_customer_signoff"
            ),
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
                for ref in (
                    item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
                )
                if str(ref).strip()
            ],
            "gate_snapshot": _mapping(item.get("gate_snapshot")),
            "handoff_materials": _mapping(item.get("handoff_materials")),
            "project_scope": _mapping(item.get("project_scope")),
            "signoff_payload_sha256": str(item.get("signoff_payload_sha256") or ""),
        }
        stored_evidence_assessment = (
            _mapping(item.get("evidence_ref_assessment"))
            if isinstance(item.get("evidence_ref_assessment"), dict)
            else {}
        )
        if stored_evidence_assessment:
            signoff["evidence_ref_assessment"] = stored_evidence_assessment
        expected_sha = _customer_project_customer_signoff_payload_sha256(signoff)
        stored_sha = str(signoff.get("signoff_payload_sha256") or "")
        signoff["integrity_valid"] = not stored_sha or stored_sha == expected_sha
        if signoff["decision"] == "accepted" and not stored_evidence_assessment:
            signoff["evidence_ref_assessment"] = _legacy_customer_signoff_evidence_ref_assessment(
                signoff
            )
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


def _customer_project_customer_signoff_evidence_ref_assessment(
    closure: dict[str, Any],
    evidence_refs: list[str],
    *,
    required: bool,
) -> dict[str, Any]:
    refs = [str(item).strip() for item in evidence_refs if str(item).strip()]
    available = _customer_project_customer_signoff_available_evidence_refs(closure)
    resolved = [
        {
            "ref": ref,
            **available[ref],
        }
        for ref in refs
        if ref in available
    ]
    unresolved = [ref for ref in refs if ref not in available]
    resolved_ref_types = {str(item.get("type") or "") for item in resolved}
    resolved_onsite_types = {
        str(item.get("evidence_type") or "")
        for item in resolved
        if item.get("type") == "onsite_receipt"
    }
    missing_onsite_types = [
        item
        for item in ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES
        if item not in resolved_onsite_types
    ]
    missing_material_types = [
        item
        for item in ("acceptance_dossier", "proposal_bundle", "audit_export")
        if item not in resolved_ref_types
    ]
    if required and not refs:
        valid = False
        reason = "customer_signoff_evidence_refs_required"
    elif required and unresolved:
        valid = False
        reason = "customer_signoff_evidence_refs_unresolved"
    elif required and (missing_onsite_types or missing_material_types):
        valid = False
        reason = "customer_signoff_evidence_refs_incomplete"
    elif required and not resolved:
        valid = False
        reason = "customer_signoff_verified_evidence_unavailable"
    else:
        valid = not unresolved
        reason = "ok" if valid else "customer_signoff_evidence_refs_unresolved"
    return {
        "valid": valid,
        "reason": reason,
        "required": required,
        "resolved_refs": resolved,
        "unresolved_refs": unresolved,
        "missing_onsite_evidence_types": missing_onsite_types if required else [],
        "missing_material_types": missing_material_types if required else [],
        "resolved_count": len(resolved),
        "available_ref_count": len(available),
        "available_ref_types": sorted({str(item.get("type") or "") for item in available.values()}),
    }


def _legacy_customer_signoff_evidence_ref_assessment(signoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": "customer_signoff_evidence_refs_legacy_unverified",
        "required": True,
        "legacy_unverified": True,
        "resolved_refs": [],
        "unresolved_refs": [
            str(item)
            for item in (
                signoff.get("evidence_refs")
                if isinstance(signoff.get("evidence_refs"), list)
                else []
            )
            if str(item).strip()
        ],
        "missing_onsite_evidence_types": list(ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES),
        "missing_material_types": [
            "acceptance_dossier",
            "proposal_bundle",
            "audit_export",
        ],
        "resolved_count": 0,
        "available_ref_count": 0,
        "available_ref_types": [],
    }


def _customer_project_customer_signoff_available_evidence_refs(
    closure: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    available: dict[str, dict[str, Any]] = {}

    def add(ref: Any, *, ref_type: str, evidence: str, status: str = "ready") -> None:
        key = str(ref or "").strip()
        if not key or key in available:
            return
        available[key] = {
            "type": ref_type,
            "status": status,
            "evidence": evidence,
        }

    onsite = _mapping(closure.get("onsite_acceptance_evidence"))
    receipts = onsite.get("receipts") if isinstance(onsite.get("receipts"), list) else []
    for raw in receipts:
        receipt = _mapping(raw)
        receipt_id = str(receipt.get("receipt_id") or "").strip()
        if (
            not receipt_id
            or receipt.get("status") != "passed"
            or receipt.get("acceptance_gate_eligible") is not True
        ):
            continue
        evidence = (
            f"type={receipt.get('evidence_type') or ''}; "
            f"source={receipt.get('source') or ''}; "
            f"sha={str(receipt.get('sha256') or '')[:16]}"
        )
        for ref in (
            receipt_id,
            f"receipt:{receipt_id}",
            f"onsite:{receipt_id}",
            f"onsite_receipt:{receipt_id}",
        ):
            add(ref, ref_type="onsite_receipt", evidence=evidence)
            available[ref]["evidence_type"] = str(receipt.get("evidence_type") or "")

    artifacts = _mapping(closure.get("artifact_verification"))
    dossier = _mapping(artifacts.get("acceptance_dossier"))
    if dossier.get("valid") is True:
        manifest = _mapping(dossier.get("manifest"))
        sha = str(manifest.get("payload_sha256") or "").strip()
        for ref in (
            "acceptance_dossier",
            "dossier",
            f"acceptance_dossier:{sha}" if sha else "",
            f"dossier:{sha}" if sha else "",
        ):
            add(ref, ref_type="acceptance_dossier", evidence=f"sha={sha[:16]}")

    proposal = _mapping(artifacts.get("proposal_bundle"))
    if proposal.get("status") == "ready":
        proposal_path = str(proposal.get("proposal_path") or "").strip()
        verification = _mapping(proposal.get("verification"))
        sha = str(verification.get("payload_sha256") or "").strip()
        for ref in (
            "proposal_bundle",
            "proposal",
            proposal_path,
            f"proposal_bundle:{sha}" if sha else "",
        ):
            add(ref, ref_type="proposal_bundle", evidence=proposal_path or f"sha={sha[:16]}")

    audit = _mapping(artifacts.get("audit_export"))
    if audit.get("status") == "ready":
        manifest_path = str(audit.get("manifest_path") or "").strip()
        sha = str(audit.get("sha256") or "").strip()
        export_id = str(audit.get("export_id") or "").strip()
        for ref in (
            "audit_export",
            "audit",
            manifest_path,
            f"audit_export:{sha}" if sha else "",
            f"audit_export:{export_id}" if export_id else "",
        ):
            add(ref, ref_type="audit_export", evidence=manifest_path or f"sha={sha[:16]}")

    return available


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
        elif _mapping(latest_signoff.get("evidence_ref_assessment")).get("valid") is not True:
            status = "manual_check"
            evidence = "Customer signoff is accepted but evidence refs are not verified."
            next_step = (
                "Customer signoff evidence refs must include the four onsite receipts, "
                "acceptance dossier, proposal bundle, and audit export."
            )
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
        evidence = (
            f"{decision or 'signoff'} by {latest_signoff.get('signatory_name') or 'customer'}"
        )
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
        timeline.append(
            {
                "timestamp": item.get("recorded_at"),
                "type": "onsite_evidence",
                "status": item.get("status"),
                "label": item.get("label") or item.get("evidence_type"),
                "summary": item.get("summary") or item.get("path"),
                "ref": item.get("receipt_id"),
            }
        )
    for review in reviews:
        item = _mapping(review)
        timeline.append(
            {
                "timestamp": item.get("reviewed_at"),
                "type": "acceptance_review",
                "status": item.get("decision"),
                "label": f"review by {item.get('operator_id') or 'system'}",
                "summary": item.get("reason"),
                "ref": item.get("review_id"),
            }
        )
    for signoff in signoffs or []:
        item = _mapping(signoff)
        timeline.append(
            {
                "timestamp": item.get("signed_at"),
                "type": "customer_signoff",
                "status": item.get("decision"),
                "label": f"signoff by {item.get('signatory_name') or 'customer'}",
                "summary": item.get("reason"),
                "ref": item.get("signoff_id"),
            }
        )
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
    next_actions = (
        readiness.get("next_actions") if isinstance(readiness.get("next_actions"), list) else []
    )
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
            "next_step": str(
                next_actions[0]
                if next_actions
                else _mapping(readiness.get("delivery_brief")).get("release_claim") or "-"
            ),
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
            "next_step": "Run the bridge against real camera, sensor, and robot diagnostic input."
            if not gates.get("uses_real_hardware")
            else "Scenario and ingest smoke evidence is ready.",
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
        {
            "gate_id": "field_device_onboarding",
            "label": "Field device onboarding",
            "status": _customer_project_device_onboarding_acceptance_status(
                _mapping(readiness.get("device_onboarding"))
            ),
            "evidence": _customer_project_device_onboarding_evidence(
                _mapping(readiness.get("device_onboarding"))
            ),
            "next_step": _customer_project_device_onboarding_next_step(
                _mapping(readiness.get("device_onboarding"))
            ),
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
        onsite_evidence.get("receipts") if isinstance(onsite_evidence.get("receipts"), list) else []
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
            "next_step": "Fix site profile validation errors."
            if report.get("status") != "passed"
            else "Site profile is valid.",
            "required_for_customer_acceptance": True,
            "project_scope": scope,
        },
        {
            "item_id": "managed_object_acceptance",
            "label": "现场对象验收绑定",
            "owner": "product",
            "status": str(acceptance.get("overall_status") or "manual_check"),
            "evidence": (
                f"{acceptance.get('ready_object_count') or 0}/"
                f"{acceptance.get('object_count') or 0} 个对象就绪"
            ),
            "next_step": str(acceptance.get("customer_status") or ""),
            "required_for_customer_acceptance": True,
            "project_scope": scope,
        },
        {
            "item_id": "deployment_credentials",
            "label": "部署凭证",
            "owner": "delivery",
            "status": "manual_check" if missing_env else "ready",
            "evidence": f"{len(missing_env)} 个必需环境引用缺失",
            "next_step": (
                "现场验收前配置钉钉、设备、语音和运行密钥。"
                if missing_env
                else "必需部署凭证已配置。"
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
                required=(
                    "scenario_eval_passed",
                    "http_smoke_passed",
                    "trusted_device_events_observed",
                    "device_onboarding_report_available",
                    "device_onboarding_no_blockers",
                    "device_onboarding_has_ready_device",
                ),
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
                required=(
                    "runtime_roundtrip_smoke_passed",
                    "runtime_roundtrip_final_status_verified",
                ),
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
    acceptance_gate_eligible = receipt.get("acceptance_gate_eligible") is True
    if receipt_status == "passed" and acceptance_gate_eligible:
        status = "ready"
    elif receipt_status == "failed":
        status = "blocked"
    elif receipt_status in {"passed", "manual_check"}:
        status = "manual_check"
    else:
        status = fallback_status if fallback_status == "blocked" else "manual_check"
    receipt_id = str(receipt.get("receipt_id") or "")
    source = str(receipt.get("source") or "")
    evidence = (
        f"receipt={receipt_id}; source={source}; sha={str(receipt.get('sha256') or '')[:16]}; "
        f"trust={str(receipt.get('trust_status') or '')}"
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
        "delivery_chain": _customer_project_acceptance_delivery_chain(report),
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
            "本 dossier 是客户交接材料。它记录导出时已有的本地和现场证据，"
            "但生产上线仍要求所有门禁均达到就绪状态。"
        ),
    }
    dossier["manifest"] = _customer_project_acceptance_dossier_manifest(dossier)
    return dossier


def _customer_project_acceptance_delivery_chain(report: dict[str, Any]) -> dict[str, Any]:
    """Return the customer-readable delivery chain embedded into handoff artifacts."""
    customer = _mapping(report.get("customer"))
    site = _mapping(report.get("site"))
    acceptance = _mapping(report.get("acceptance_summary"))
    execution_summary = _mapping(_mapping(report.get("execution_bindings")).get("summary"))
    launch_readiness = _mapping(report.get("launch_readiness"))
    site_checklist = _mapping(report.get("site_acceptance_checklist"))
    workflow_steps = (
        report.get("delivery_workflow", {}).get("steps")
        if isinstance(report.get("delivery_workflow"), dict)
        else []
    )
    workflow_by_id = {
        str(item.get("step_id") or ""): item for item in workflow_steps if isinstance(item, dict)
    }
    runtime_step = _mapping(workflow_by_id.get("runtime_bindings"))
    runtime_gate = _find_gate(launch_readiness, "managed_object_execution_bindings")
    scope_ready = all(
        str(value or "").strip()
        for value in (
            customer.get("customer_id"),
            customer.get("project_id"),
            site.get("site_id"),
        )
    )
    field_device_gate = _find_gate(launch_readiness, "field_device_onboarding")
    steps = [
        _delivery_chain_step(
            step_id="project_scope",
            label="Customer project scope",
            status="ready" if scope_ready else "blocked",
            customer_question="Which customer, project, and site does this handoff cover?",
            evidence=(
                f"customer_id={customer.get('customer_id') or '-'}; "
                f"project_id={customer.get('project_id') or '-'}; "
                f"site_id={site.get('site_id') or '-'}"
            ),
            next_step="Set customer_id, project_id, and site_id before exporting the handoff."
            if not scope_ready
            else "Project scope is explicit in the handoff.",
            endpoint="/api/field/customer-projects",
            source_surface_id="customer_projects",
        ),
        _delivery_chain_step(
            step_id="template_market",
            label="Industry template",
            status="manual_check",
            customer_question="Which reusable industry template was this project derived from?",
            evidence=f"industry={customer.get('industry') or 'unspecified'}",
            next_step="Verify the selected factory, park, warehouse, or scenic template before reuse.",
            endpoint="/api/field/customer-project-templates",
            source_surface_id="template_market",
        ),
        _delivery_chain_step(
            step_id="managed_object_directory",
            label="现场对象目录",
            status=str(acceptance.get("overall_status") or "blocked"),
            customer_question="本项目覆盖哪些真实现场对象？",
            evidence=(
                f"objects={acceptance.get('object_count') or 0}; "
                f"ready={acceptance.get('ready_object_count') or 0}; "
                f"manual={acceptance.get('manual_check_object_count') or 0}; "
                f"blocked={acceptance.get('blocked_object_count') or 0}"
            ),
            next_step=str(acceptance.get("customer_status") or "完成现场对象验收。"),
            endpoint="/api/field/customer-projects/managed-object-directory",
            source_surface_id="managed_objects",
        ),
        _delivery_chain_step(
            step_id="capability_resource_binding",
            label="Capability and resource binding",
            status=str(
                execution_summary.get("overall_status") or runtime_step.get("status") or "blocked"
            ),
            customer_question=(
                "Are every object's vision model, sensor protocol, skill package, "
                "and acceptance test bound?"
            ),
            evidence=str(
                runtime_step.get("evidence")
                or (
                    f"ready_objects={execution_summary.get('ready_object_count') or 0}; "
                    f"objects={execution_summary.get('object_count') or 0}"
                )
            ),
            next_step=str(
                execution_summary.get("next_step")
                or runtime_step.get("next_step")
                or "Bind executable resources before customer signoff."
            ),
            endpoint="/api/field/customer-project-resource-catalog",
            source_surface_id="delivery_resources",
        ),
        _delivery_chain_step(
            step_id="runtime_blueprint",
            label="Runtime blueprint",
            status=str(
                runtime_gate.get("status")
                or launch_readiness.get("overall_status")
                or "manual_check"
            ),
            customer_question="Which robot runtime plan can execute this customer project?",
            evidence=str(
                runtime_gate.get("evidence")
                or "Runtime evidence is taken from launch readiness gates."
            ),
            next_step=str(
                runtime_gate.get("next_step") or "Bind the customer project to a runtime blueprint."
            ),
            endpoint="/api/blueprints",
            source_surface_id="runtime_blueprint_binding",
        ),
        _delivery_chain_step(
            step_id="acceptance_package",
            label="Acceptance and handoff package",
            status=str(
                site_checklist.get("overall_status")
                or field_device_gate.get("status")
                or "manual_check"
            ),
            customer_question="Can the customer verify acceptance from evidence rather than internal code?",
            evidence=(
                f"site_checklist={site_checklist.get('overall_status') or 'unknown'}; "
                f"launch={launch_readiness.get('overall_status') or 'unknown'}"
            ),
            next_step=str(
                site_checklist.get("next_step")
                or launch_readiness.get("next_step")
                or "Export and verify the customer handoff package."
            ),
            endpoint="/api/field/customer-projects/{identifier}/acceptance-dossier",
            source_surface_id="acceptance_package",
        ),
    ]
    return {
        "chain_type": "askme.customer_project.delivery_chain.v1",
        "overall_status": _delivery_chain_summary(steps)["overall_status"],
        "step_count": len(steps),
        "summary": _delivery_chain_summary(steps),
        "steps": steps,
        "policy": {
            "runtime_blueprint_is_required_before_customer_claim": True,
            "capability_resources_must_be_bound_to_managed_objects": True,
            "acceptance_package_must_reference_real_evidence": True,
        },
    }


def _find_gate(parent: dict[str, Any], gate_id: str) -> dict[str, Any]:
    gates = parent.get("gates") if isinstance(parent.get("gates"), list) else []
    for gate in gates:
        if isinstance(gate, dict) and str(gate.get("gate_id") or "") == gate_id:
            return gate
    return {}


def _delivery_chain_step(
    *,
    step_id: str,
    label: str,
    status: str,
    customer_question: str,
    evidence: str,
    next_step: str,
    endpoint: str,
    source_surface_id: str,
) -> dict[str, str]:
    return {
        "step_id": step_id,
        "label": label,
        "status": _delivery_chain_status(status),
        "customer_question": customer_question,
        "evidence": evidence,
        "next_step": next_step,
        "endpoint": endpoint,
        "source_surface_id": source_surface_id,
    }


def _delivery_chain_status(status: Any) -> str:
    text = str(status or "unknown").strip()
    if text in {
        "ready",
        "passed",
        "production_acceptance_ready",
        "accepted_by_customer",
        "ready_for_onsite_acceptance",
    }:
        return "ready"
    if text in {"blocked", "failed", "missing", "file_missing", "invalid", "rejected"}:
        return "blocked"
    if text in {
        "manual_check",
        "needs_review",
        "ready_for_site_validation",
        "trial_or_demo_only",
        "pilot_or_site_trial",
    }:
        return "manual_check"
    return text or "unknown"


def _delivery_chain_summary(steps: list[dict[str, str]]) -> dict[str, Any]:
    ready_count = sum(1 for item in steps if item.get("status") == "ready")
    manual_count = sum(1 for item in steps if item.get("status") == "manual_check")
    blocked_count = sum(1 for item in steps if item.get("status") == "blocked")
    unknown_count = sum(1 for item in steps if item.get("status") == "unknown")
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check"
        if manual_count or unknown_count
        else "ready"
        if steps
        else "blocked"
    )
    first_gap = next(
        (
            item.get("next_step") or item.get("customer_question") or item.get("label")
            for item in steps
            if item.get("status") != "ready"
        ),
        "Customer project delivery chain is ready for handoff review.",
    )
    return {
        "overall_status": overall_status,
        "ready_count": ready_count,
        "manual_check_count": manual_count,
        "blocked_count": blocked_count,
        "unknown_count": unknown_count,
        "first_gap": first_gap,
    }


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


def verify_customer_project_proposal_bundle(proposal: dict[str, Any]) -> dict[str, Any]:
    """Verify the integrity metadata of a customer project proposal bundle."""
    if (
        not isinstance(proposal, dict)
        or proposal.get("proposal_type") != "askme.customer_project_proposal"
    ):
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
    manifest_scope = _delivery_scope_payload_from_customer_site(
        manifest, {"site_id": manifest.get("site_id")}
    )
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
    readable_chain = _mapping(readable_delivery.get("delivery_chain"))
    readable_chain_steps = (
        readable_chain.get("steps") if isinstance(readable_chain.get("steps"), list) else []
    )
    manifest_chain_count = manifest.get("proposal_delivery_chain_step_count")
    if manifest_chain_count not in (None, "") and int(manifest_chain_count or 0) != len(
        readable_chain_steps
    ):
        errors.append("manifest.proposal_delivery_chain_step_count mismatch")
    manifest_chain_status = str(manifest.get("proposal_delivery_chain_status") or "")
    readable_chain_status = str(readable_chain.get("overall_status") or "")
    if (
        manifest_chain_status
        and readable_chain_status
        and manifest_chain_status != readable_chain_status
    ):
        errors.append("manifest.proposal_delivery_chain_status mismatch")

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
    if (
        not isinstance(dossier, dict)
        or dossier.get("dossier_type") != "askme.customer_project_acceptance"
    ):
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
    delivery_chain = _mapping(dossier.get("delivery_chain"))
    delivery_chain_steps = (
        delivery_chain.get("steps") if isinstance(delivery_chain.get("steps"), list) else []
    )
    manifest_chain_count = manifest.get("delivery_chain_step_count")
    if manifest_chain_count not in (None, "") and int(manifest_chain_count or 0) != len(
        delivery_chain_steps
    ):
        errors.append("manifest.delivery_chain_step_count mismatch")
    manifest_chain_status = str(manifest.get("delivery_chain_status") or "")
    dossier_chain_status = str(delivery_chain.get("overall_status") or "")
    if (
        manifest_chain_status
        and dossier_chain_status
        and manifest_chain_status != dossier_chain_status
    ):
        errors.append("manifest.delivery_chain_status mismatch")
    return {
        "valid": not errors,
        "reason": "ok" if not errors else "integrity_errors",
        "errors": errors,
        "manifest": manifest,
        "payload_sha256": actual,
        "signature_checked": signature_checked,
        "signature_key_id": str(manifest.get("signature_key_id") or ""),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("${"):
        return ""
    return text
