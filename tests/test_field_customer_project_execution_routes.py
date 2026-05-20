from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from tests.support.field_route_app import field_route_test_app as _field_route_test_app
from tests.support.health_snapshots import runtime_snapshot as _runtime_snapshot
from tests.support.route_module_assertions import (
    function_names,
    imports_by_module,
    parse_python_module,
    route_method_counts,
)


def test_field_customer_project_execution_routes_are_registered_from_split_module() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    acceptance_tree = parse_python_module("askme/api/routes/field_customer_project_acceptance.py")
    execution_tree = parse_python_module("askme/api/routes/field_customer_project_execution.py")

    field_route_defs = function_names(route_tree)
    acceptance_route_defs = function_names(acceptance_tree)
    execution_route_defs = function_names(execution_tree)
    field_imports = imports_by_module(route_tree)

    moved_routes = {
        "field_customer_project_execution_bindings",
        "field_customer_project_object_rehearsal",
        "field_customer_project_execution_bindings_cors",
        "field_customer_project_object_rehearsal_cors",
        "_object_rehearsal_plan_summary",
        "_first_object_device",
        "_object_rehearsal_payload",
        "_object_rehearsal_boundary",
        "_wants_rehearsal_onsite_evidence",
        "_rehearsal_onsite_evidence_rejection",
        "_rehearsal_onsite_evidence_candidate",
    }

    assert "register_customer_project_execution_routes" in field_imports[
        "askme.api.routes.field_customer_project_execution"
    ]
    assert moved_routes.isdisjoint(field_route_defs)
    assert moved_routes.isdisjoint(acceptance_route_defs)
    assert moved_routes.issubset(execution_route_defs)


def test_customer_project_execution_route_module_owns_execution_domain_calls() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    acceptance_tree = parse_python_module("askme/api/routes/field_customer_project_acceptance.py")
    execution_tree = parse_python_module("askme/api/routes/field_customer_project_execution.py")
    field_imports = imports_by_module(route_tree)
    acceptance_imports = imports_by_module(acceptance_tree)
    execution_imports = imports_by_module(execution_tree)

    assert "build_customer_project_execution_bindings" not in field_imports.get(
        "askme.pipeline.field.customer_projects",
        set(),
    )
    assert "build_customer_project_execution_bindings" not in acceptance_imports.get(
        "askme.pipeline.field.customer_projects",
        set(),
    )
    assert "build_customer_project_execution_bindings" in execution_imports[
        "askme.pipeline.field.customer_projects"
    ]
    assert "normalize_field_ingest_payload" not in field_imports.get(
        "askme.pipeline.field.field_ingest_adapters",
        set(),
    )
    assert "askme.pipeline.field.field_ingest_adapters" not in acceptance_imports
    assert "normalize_field_ingest_payload" in execution_imports[
        "askme.pipeline.field.field_ingest_adapters"
    ]


def test_customer_project_execution_routes_have_no_duplicate_methods(tmp_path: Path) -> None:
    app = _field_route_test_app(tmp_path / "site-profiles")
    route_methods = route_method_counts(app, "/api/field/customer-projects")

    expected_routes = {
        ("/api/field/customer-projects/{identifier}/execution-bindings", "GET"),
        ("/api/field/customer-projects/{identifier}/execution-bindings", "OPTIONS"),
        (
            "/api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal",
            "POST",
        ),
        (
            "/api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal",
            "OPTIONS",
        ),
    }

    for key in expected_routes:
        assert route_methods.get(key) == 1, key
    for key, count in route_methods.items():
        assert count == 1, key


def test_customer_project_object_rehearsal_rejects_unknown_mode() -> None:
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post(
        "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
        json={"operator_id": "supervisor-1", "mode": "live"},
    )

    assert response.status_code == 422
    assert response.json()["accepted"] is False
    assert response.json()["reason"] == "invalid_rehearsal_mode"
    assert response.json()["allowed_modes"] == ["dry_run", "shadow_post"]


def test_customer_project_object_rehearsal_uses_project_scope_over_payload_spoofing() -> None:
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post(
        "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
        json={
            "operator_id": "supervisor-1",
            "mode": "dry_run",
            "payload": {
                "tenant_id": "evil-tenant",
                "delivery_namespace": "evil-namespace",
                "customer_id": "evil-customer",
                "project_id": "evil-project",
                "site_id": "evil-site",
                "project_scope": {
                    "tenant_id": "evil-tenant",
                    "delivery_namespace": "evil-namespace",
                    "customer_id": "evil-customer",
                    "project_id": "evil-project",
                    "site_id": "evil-site",
                },
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["raw_payload"]["tenant_id"] == "default"
    assert payload["raw_payload"]["delivery_namespace"] == "default"
    assert payload["raw_payload"]["customer_id"] == "demo-customer"
    assert payload["raw_payload"]["project_id"] == "demo-field-ops"
    assert payload["raw_payload"]["site_id"] == "inovx-demo-park"
    assert payload["raw_payload"]["project_scope"] == {
        "tenant_id": "default",
        "delivery_namespace": "default",
        "customer_id": "demo-customer",
        "project_id": "demo-field-ops",
        "site_id": "inovx-demo-park",
    }
    assert payload["normalized"]["project_scope"] == payload["raw_payload"]["project_scope"]
