"""HTTP guard tests for field admin and customer-project write routes."""

import pytest
from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.health_server import create_health_app
from tests.support.health_snapshots import runtime_snapshot as _runtime_snapshot


@pytest.mark.parametrize(
    "path",
    [
        "/api/field/notification-test",
        "/api/field/customer-projects/from-template",
        "/api/field/delivery-resource-registry",
        "/api/field/delivery-resource-registry/vision_models/vehicle-detection/disable",
        "/api/field/delivery-resource-registry/rollback",
        "/api/field/delivery-resource-governance-requests",
        "/api/field/delivery-resource-governance-requests/request-1/review",
        "/api/field/delivery-resource-governance-requests/escalate-overdue",
    ],
)
def test_field_admin_and_delivery_resource_write_routes_reject_non_object_json_body(
    path,
) -> None:
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post(path, json=["not-an-object"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/field/customer-projects"),
        ("POST", "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles"),
        ("DELETE", "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles"),
        ("POST", "/api/field/customer-projects/demo-field-ops/rollback"),
        ("POST", "/api/field/customer-projects/demo-field-ops/archive"),
        ("POST", "/api/field/customer-project-template-release-notes/export"),
        ("POST", "/api/field/customer-project-templates/park/release-requests"),
        ("POST", "/api/field/customer-project-template-release-requests/request-1/review"),
        ("POST", "/api/field/customer-project-templates/park/release"),
        ("POST", "/api/field/customer-projects/demo-field-ops/onsite-evidence"),
        ("POST", "/api/field/customer-projects/demo-field-ops/acceptance-review"),
        ("POST", "/api/field/customer-projects/demo-field-ops/customer-signoff"),
        ("POST", "/api/field/customer-projects/import"),
        ("POST", "/api/field/customer-projects/package/verify"),
        ("POST", "/api/field/customer-projects/package/diff"),
        ("POST", "/api/field/customer-projects/proposal-bundle/verify"),
        ("POST", "/api/field/customer-projects/acceptance-dossier/verify"),
        (
            "POST",
            "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
        ),
    ],
)
def test_field_customer_project_write_routes_reject_non_object_json_body(
    method,
    path,
) -> None:
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.request(method, path, json=["not-an-object"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_field_customer_project_write_endpoints_enforce_project_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        health_server,
        "get_config",
        lambda: {
            "field_operations": {
                "operator_directory": {"mode": "demo_config"},
                "operators": {
                    "supervisor-1": {"roles": ["supervisor"]},
                    "scoped-supervisor": {
                        "roles": ["supervisor"],
                        "project_scope": {
                            "customer_ids": ["other-customer"],
                            "project_ids": ["other-project"],
                            "site_ids": ["other-site"],
                        },
                    },
                    "scoped-owner": {
                        "roles": ["product_owner"],
                        "project_scope": {
                            "customer_ids": ["other-customer"],
                            "project_ids": ["other-project"],
                            "site_ids": ["other-site"],
                        },
                    },
                },
            }
        },
    )
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    exported = client.get(
        "/api/field/customer-projects/demo-field-ops/export",
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert exported.status_code == 200
    package = exported.json()["package"]
    forbidden_operator = {"operator_id": "scoped-supervisor"}

    denied_from_template = client.post(
        "/api/field/customer-projects/from-template",
        json={
            **forbidden_operator,
            "template_id": "park-visitor-service",
            "customer": {
                "customer_id": "demo-customer",
                "project_id": "demo-field-ops",
            },
            "site": {"site_id": "inovx-demo-park"},
        },
    )
    assert denied_from_template.status_code == 403
    assert denied_from_template.json()["reason"] == "project_scope_not_allowed"

    denied_upsert = client.post(
        "/api/field/customer-projects",
        json={
            **forbidden_operator,
            "profile": package["profile"],
        },
    )
    assert denied_upsert.status_code == 403
    assert denied_upsert.json()["reason"] == "project_scope_not_allowed"

    denied_import = client.post(
        "/api/field/customer-projects/import",
        json={**forbidden_operator, "dry_run": True, "package": package},
    )
    assert denied_import.status_code == 403
    assert denied_import.json()["reason"] == "project_scope_not_allowed"

    denied_resource = client.post(
        "/api/field/delivery-resource-registry",
        json={
            **forbidden_operator,
            "resource": {
                "resource_type": "vision_models",
                "resource_id": "scope-blocked-model",
                "display_name": "Scope blocked model",
                "project_id": "demo-field-ops",
            },
        },
    )
    assert denied_resource.status_code == 403
    assert denied_resource.json()["reason"] == "project_scope_not_allowed"

    denied_registry_rollback = client.post(
        "/api/field/delivery-resource-registry/rollback",
        json={
            **forbidden_operator,
            "revision_id": "missing-revision",
            "dry_run": True,
        },
    )
    assert denied_registry_rollback.status_code == 403
    assert denied_registry_rollback.json()["reason"] == "operator_missing_permission"
    assert denied_registry_rollback.json()["operator_auth"]["permission"] == (
        "resource:governance:approve"
    )

    scoped_registry_request = client.post(
        "/api/field/delivery-resource-governance-requests",
        json={
            "operator_id": "scoped-owner",
            "action": "rollback_registry",
            "operation": {"revision_id": "missing-revision"},
        },
    )
    assert scoped_registry_request.status_code == 403
    assert scoped_registry_request.json()["reason"] == (
        "resource_registry_rollback_requires_unrestricted_operator"
    )

    scoped_registry_rollback = client.post(
        "/api/field/delivery-resource-registry/rollback",
        json={
            "operator_id": "scoped-owner",
            "revision_id": "missing-revision",
            "dry_run": True,
        },
    )
    assert scoped_registry_rollback.status_code == 403
    assert scoped_registry_rollback.json()["reason"] == (
        "resource_registry_rollback_requires_unrestricted_operator"
    )

    denied_object = client.post(
        "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles",
        json={
            **forbidden_operator,
            "managed_object": {
                "display_name": "Vehicles",
                "category": "traffic",
            },
        },
    )
    assert denied_object.status_code == 403
    assert denied_object.json()["reason"] == "project_scope_not_allowed"

    denied_delete = client.request(
        "DELETE",
        "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles",
        json=forbidden_operator,
    )
    assert denied_delete.status_code == 403
    assert denied_delete.json()["reason"] == "project_scope_not_allowed"

    denied_archive = client.post(
        "/api/field/customer-projects/demo-field-ops/archive",
        json=forbidden_operator,
    )
    assert denied_archive.status_code == 403
    assert denied_archive.json()["reason"] == "project_scope_not_allowed"

    denied_rollback = client.post(
        "/api/field/customer-projects/demo-field-ops/rollback",
        json={**forbidden_operator, "revision_id": "missing"},
    )
    assert denied_rollback.status_code == 403
    assert denied_rollback.json()["reason"] == "project_scope_not_allowed"

    denied_onsite = client.post(
        "/api/field/customer-projects/demo-field-ops/onsite-evidence",
        json={
            **forbidden_operator,
            "evidence": {
                "evidence_type": "device_ingest",
                "status": "passed",
                "summary": "scope should block this receipt",
            },
        },
    )
    assert denied_onsite.status_code == 403
    assert denied_onsite.json()["reason"] == "project_scope_not_allowed"

    denied_review = client.post(
        "/api/field/customer-projects/demo-field-ops/acceptance-review",
        json={
            **forbidden_operator,
            "review": {
                "decision": "accepted",
                "risk_acknowledgement": True,
                "reason": "scope should block this review",
            },
        },
    )
    assert denied_review.status_code == 403
    assert denied_review.json()["reason"] == "project_scope_not_allowed"

    denied_execution = client.request(
        "GET",
        "/api/field/customer-projects/demo-field-ops/execution-bindings",
        headers={"X-Askme-Operator-Id": "scoped-supervisor"},
    )
    assert denied_execution.status_code == 403
    assert denied_execution.json()["reason"] == "project_scope_not_allowed"

    denied_rehearsal = client.post(
        "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
        json={**forbidden_operator, "mode": "dry_run"},
    )
    assert denied_rehearsal.status_code == 403
    assert denied_rehearsal.json()["reason"] == "project_scope_not_allowed"

    denied_signoff = client.post(
        "/api/field/customer-projects/demo-field-ops/customer-signoff",
        json={
            **forbidden_operator,
            "signoff": {
                "decision": "needs_fix",
                "signatory_name": "Scope blocked customer",
                "reason": "scope should block this signoff",
            },
        },
    )
    assert denied_signoff.status_code == 403
    assert denied_signoff.json()["reason"] == "project_scope_not_allowed"
