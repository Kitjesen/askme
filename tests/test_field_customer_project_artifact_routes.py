from __future__ import annotations

from pathlib import Path

from tests.support.field_route_app import field_route_test_app as _field_route_test_app
from tests.support.route_module_assertions import (
    function_names,
    imports_by_module,
    parse_python_module,
    route_method_counts,
)


def test_field_customer_project_artifact_routes_are_registered_from_split_module() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    artifact_tree = parse_python_module("askme/api/routes/field_customer_project_artifacts.py")

    field_route_defs = function_names(route_tree)
    artifact_route_defs = function_names(artifact_tree)
    field_imports = imports_by_module(route_tree)

    moved_routes = {
        "field_customer_project_import",
        "field_customer_project_package_verify",
        "field_customer_project_package_diff",
        "field_customer_project_proposal_bundle_verify",
        "field_customer_project_acceptance_dossier_verify",
        "field_customer_project_export",
        "field_customer_project_acceptance_dossier",
        "field_customer_project_proposal_bundle",
        "field_customer_project_import_cors",
        "field_customer_project_package_verify_cors",
        "field_customer_project_package_diff_cors",
        "field_customer_project_proposal_bundle_verify_cors",
        "field_customer_project_acceptance_dossier_verify_cors",
        "field_customer_project_export_cors",
        "field_customer_project_acceptance_dossier_cors",
        "field_customer_project_proposal_bundle_cors",
    }

    assert "register_customer_project_artifact_routes" in field_imports[
        "askme.api.routes.field_customer_project_artifacts"
    ]
    assert moved_routes.isdisjoint(field_route_defs)
    assert moved_routes.issubset(artifact_route_defs)
    assert "askme.pipeline.field.customer_project_artifacts" not in field_imports


def test_customer_project_artifact_route_module_owns_artifact_domain_calls() -> None:
    artifact_tree = parse_python_module("askme/api/routes/field_customer_project_artifacts.py")
    artifact_imports = imports_by_module(artifact_tree)

    assert artifact_imports["askme.pipeline.field.customer_project_artifacts"] == {
        "diff_customer_project_package",
        "export_customer_project_acceptance_dossier",
        "export_customer_project_package",
        "export_customer_project_proposal_bundle",
        "import_customer_project_package",
        "verify_customer_project_acceptance_dossier",
        "verify_customer_project_package",
        "verify_customer_project_proposal_bundle",
    }


def test_customer_project_artifact_routes_have_no_duplicate_methods(tmp_path: Path) -> None:
    app = _field_route_test_app(tmp_path / "site-profiles")
    route_methods = route_method_counts(app, "/api/field/customer-projects")

    expected_routes = {
        ("/api/field/customer-projects/import", "POST"),
        ("/api/field/customer-projects/import", "OPTIONS"),
        ("/api/field/customer-projects/package/verify", "POST"),
        ("/api/field/customer-projects/package/verify", "OPTIONS"),
        ("/api/field/customer-projects/package/diff", "POST"),
        ("/api/field/customer-projects/package/diff", "OPTIONS"),
        ("/api/field/customer-projects/proposal-bundle/verify", "POST"),
        ("/api/field/customer-projects/proposal-bundle/verify", "OPTIONS"),
        ("/api/field/customer-projects/acceptance-dossier/verify", "POST"),
        ("/api/field/customer-projects/acceptance-dossier/verify", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/export", "GET"),
        ("/api/field/customer-projects/{identifier}/export", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/acceptance-dossier", "GET"),
        ("/api/field/customer-projects/{identifier}/acceptance-dossier", "OPTIONS"),
        ("/api/field/customer-projects/{identifier}/proposal-bundle", "GET"),
        ("/api/field/customer-projects/{identifier}/proposal-bundle", "OPTIONS"),
    }

    for key in expected_routes:
        assert route_methods.get(key) == 1, key
    for key, count in route_methods.items():
        assert count == 1, key
