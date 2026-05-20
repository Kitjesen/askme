from __future__ import annotations

from pathlib import Path

from tests.support.field_route_app import field_route_test_app as _field_route_test_app
from tests.support.route_module_assertions import (
    function_names,
    imports_by_module,
    parse_python_module,
    route_method_counts,
    route_paths,
)


def test_field_customer_project_profile_routes_are_registered_from_split_module() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    profile_tree = parse_python_module("askme/api/routes/field_customer_project_profiles.py")

    field_route_defs = function_names(route_tree)
    profile_route_defs = function_names(profile_tree)
    field_imports = imports_by_module(route_tree)

    moved_routes = {
        "field_customer_project_detail",
        "field_customer_project_upsert",
        "field_customer_project_object_upsert",
        "field_customer_project_object_delete",
        "field_customer_project_history",
        "field_customer_project_rollback",
        "field_customer_project_archive",
        "field_customer_project_detail_cors",
        "field_customer_project_object_cors",
        "field_customer_project_history_cors",
        "field_customer_project_rollback_cors",
        "field_customer_project_archive_cors",
    }

    assert "register_customer_project_profile_routes" in field_imports[
        "askme.api.routes.field_customer_project_profiles"
    ]
    assert moved_routes.isdisjoint(field_route_defs)
    assert moved_routes.issubset(profile_route_defs)


def test_customer_project_profile_route_module_owns_profile_domain_calls() -> None:
    route_tree = parse_python_module("askme/api/routes/field.py")
    profile_tree = parse_python_module("askme/api/routes/field_customer_project_profiles.py")
    field_imports = imports_by_module(route_tree)
    profile_imports = imports_by_module(profile_tree)

    moved_domain_calls = {
        "archive_customer_project_profile",
        "delete_managed_object",
        "list_customer_project_revisions",
        "rollback_customer_project_profile",
        "upsert_customer_project_profile",
        "upsert_managed_object",
    }

    assert moved_domain_calls.isdisjoint(
        field_imports.get("askme.pipeline.field.customer_projects", set())
    )
    assert moved_domain_calls <= profile_imports["askme.pipeline.field.customer_projects"]
    assert "get_customer_project_profile" in profile_imports["askme.pipeline.field.customer_projects"]


def test_customer_project_profile_routes_have_no_duplicate_methods_and_safe_order(tmp_path: Path) -> None:
    app = _field_route_test_app(tmp_path / "site-profiles")
    route_methods = route_method_counts(app, "/api/field/customer-projects")
    paths = route_paths(app, "/api/field/customer-projects")

    for key, count in route_methods.items():
        assert count == 1, key

    assert paths.index("/api/field/customer-projects/managed-object-directory") < paths.index(
        "/api/field/customer-projects/{identifier}"
    )
    assert paths.index("/api/field/customer-projects/from-template") < paths.index(
        "/api/field/customer-projects/{identifier}"
    )
