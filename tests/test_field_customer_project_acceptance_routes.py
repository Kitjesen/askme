from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import askme.api.routes.field_customer_project_acceptance as acceptance_routes
from askme.pipeline.field import customer_project_acceptance as acceptance_module
from tests.support.field_route_app import field_route_test_app as _field_route_test_app
from tests.support.route_module_assertions import (
    function_names,
    imports_by_module,
    parse_python_module,
    route_method_counts,
)


def test_field_customer_project_acceptance_routes_are_registered_from_split_module() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    acceptance_tree = parse_python_module("askme/api/routes/field_customer_project_acceptance.py")

    field_route_defs = function_names(route_tree)
    acceptance_route_defs = function_names(acceptance_tree)
    field_imports = imports_by_module(route_tree)

    moved_routes = {
        "field_customer_project_acceptance_report",
        "field_customer_project_onsite_evidence",
        "field_customer_project_onsite_evidence_register",
        "field_customer_project_acceptance_closure",
        "field_customer_project_acceptance_review",
        "field_customer_project_customer_signoff",
        "field_customer_project_customer_signoff_register",
        "field_customer_project_acceptance_report_cors",
        "field_customer_project_onsite_evidence_cors",
        "field_customer_project_acceptance_closure_cors",
        "field_customer_project_acceptance_review_cors",
        "field_customer_project_customer_signoff_cors",
    }

    assert "register_customer_project_acceptance_routes" in field_imports[
        "askme.api.routes.field_customer_project_acceptance"
    ]
    assert moved_routes.isdisjoint(field_route_defs)
    assert moved_routes.issubset(acceptance_route_defs)


def test_customer_project_acceptance_route_module_owns_acceptance_domain_calls() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    acceptance_tree = parse_python_module("askme/api/routes/field_customer_project_acceptance.py")
    field_imports = imports_by_module(route_tree)
    acceptance_imports = imports_by_module(acceptance_tree)

    moved_domain_calls = {
        "customer_project_acceptance_closure",
        "customer_project_acceptance_report",
        "list_customer_project_customer_signoffs",
        "list_customer_project_onsite_evidence",
        "register_customer_project_acceptance_review",
        "register_customer_project_customer_signoff",
        "register_customer_project_onsite_evidence",
    }

    assert moved_domain_calls.isdisjoint(
        field_imports.get("askme.pipeline.field.customer_projects", set())
    )
    assert moved_domain_calls <= acceptance_imports["askme.pipeline.field.customer_projects"]
    assert "get_customer_project_profile" in acceptance_imports[
        "askme.pipeline.field.customer_projects"
    ]
    assert "askme.pipeline.field.field_ingest_adapters" not in acceptance_imports


def test_customer_project_acceptance_routes_have_no_duplicate_methods(tmp_path: Path) -> None:
    app = _field_route_test_app(tmp_path / "site-profiles")
    route_methods = route_method_counts(app, "/api/field/customer-projects")

    expected_routes = {
        ("/api/field/customer-projects/{identifier}/acceptance-report", "GET"),
        ("/api/field/customer-projects/{identifier}/acceptance-report", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/onsite-evidence", "GET"),
        ("/api/field/customer-projects/{identifier}/onsite-evidence", "POST"),
        ("/api/field/customer-projects/{identifier}/onsite-evidence", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/acceptance-closure", "GET"),
        ("/api/field/customer-projects/{identifier}/acceptance-closure", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/acceptance-review", "POST"),
        ("/api/field/customer-projects/{identifier}/acceptance-review", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/customer-signoff", "GET"),
        ("/api/field/customer-projects/{identifier}/customer-signoff", "POST"),
        ("/api/field/customer-projects/{identifier}/customer-signoff", "OPTIONS"),
    }

    for key in expected_routes:
        assert route_methods.get(key) == 1, key
    for key, count in route_methods.items():
        assert count == 1, key


@pytest.mark.parametrize(
    "reason",
    [
        "customer_signoff_evidence_refs_required",
        "customer_signoff_evidence_refs_unresolved",
        "customer_signoff_evidence_refs_incomplete",
    ],
)
def test_customer_signoff_route_maps_evidence_ref_errors_to_422(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    profile_root = tmp_path / "site-profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)

    def reject_signoff(*_args, **_kwargs) -> dict[str, object]:
        return {
            "accepted": False,
            "reason": reason,
            "evidence_ref_assessment": {
                "valid": False,
                "reason": reason,
                "unresolved_refs": ["arbitrary-string"]
                if reason == "customer_signoff_evidence_refs_unresolved"
                else [],
            },
        }

    monkeypatch.setattr(
        acceptance_routes,
        "register_customer_project_customer_signoff",
        reject_signoff,
    )
    client = TestClient(_field_route_test_app(profile_root))

    response = client.post(
        "/api/field/customer-projects/demo-field-ops/customer-signoff",
        json={
            "operator_id": "delivery.lead",
            "signoff": {
                "decision": "accepted",
                "signatory_name": "Fanmu Operator",
                "risk_acknowledgement": True,
                "credential_ref": "customer-signoff.pdf",
                "credential_sha256": "a" * 64,
                "evidence_refs": ["arbitrary-string"],
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["accepted"] is False
    assert response.json()["reason"] == reason
    assert response.json()["evidence_ref_assessment"]["reason"] == reason


def test_customer_signoff_route_exercises_real_missing_evidence_ref_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "site-profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_closure",
        lambda *_args, **_kwargs: {
            "found": True,
            "overall_status": "ready_for_customer_signoff",
            "customer_claim": "Ready for customer signoff.",
            "next_step": "Archive customer signoff.",
            "gates": [],
            "acceptance_report": {},
            "onsite_acceptance_evidence": {"receipts": []},
            "artifact_verification": {
                "acceptance_dossier": {
                    "valid": True,
                    "manifest": {"payload_sha256": "d" * 64},
                },
                "proposal_bundle": {"status": "manual_check"},
                "audit_export": {"status": "manual_check"},
            },
        },
    )
    client = TestClient(_field_route_test_app(profile_root))

    response = client.post(
        "/api/field/customer-projects/demo-field-ops/customer-signoff",
        json={
            "operator_id": "delivery.lead",
            "signoff": {
                "decision": "accepted",
                "signatory_name": "Fanmu Operator",
                "risk_acknowledgement": True,
                "credential_ref": "customer-signoff.pdf",
                "credential_sha256": "a" * 64,
            },
        },
    )

    payload = response.json()
    assert response.status_code == 422
    assert payload["accepted"] is False
    assert payload["reason"] == "customer_signoff_evidence_refs_required"
    assert payload["evidence_ref_assessment"]["valid"] is False
    assert payload["evidence_ref_assessment"]["required"] is True
    assert sorted(payload["evidence_ref_assessment"]["missing_onsite_evidence_types"]) == [
        "device_ingest",
        "notification_delivery",
        "runtime_roundtrip",
        "voice_playback",
    ]
    assert payload["evidence_ref_assessment"]["missing_material_types"] == [
        "acceptance_dossier",
        "proposal_bundle",
        "audit_export",
    ]
