from __future__ import annotations

import ast
from pathlib import Path

from askme.api.routes import field_project_scope as scope


def test_operator_project_scope_cleans_values_and_respects_unrestricted() -> None:
    assert scope.operator_project_scope({}) == {}
    assert scope.operator_project_scope(
        {"operator_auth": {"operator": {"project_scope": {"unrestricted": True}}}}
    ) == {}

    result = scope.operator_project_scope(
        {
            "operator_auth": {
                "operator": {
                    "project_scope": {
                        "tenant_ids": [" tenant-a ", "", None],
                        "delivery_namespaces": ["pilot"],
                        "customer_ids": ["customer-1"],
                        "project_ids": ["project-1"],
                        "site_ids": ["site-1"],
                    }
                }
            }
        }
    )

    assert result == {
        "tenant_ids": ["tenant-a"],
        "delivery_namespaces": ["pilot"],
        "customer_ids": ["customer-1"],
        "project_ids": ["project-1"],
        "site_ids": ["site-1"],
    }


def test_scope_allows_wildcard_and_blocks_cross_customer_project() -> None:
    item = {
        "tenant_id": "tenant-a",
        "delivery_namespace": "pilot",
        "customer_id": "customer-1",
        "project_id": "project-1",
        "site_id": "site-1",
    }

    assert scope.scope_allows({}, item) is True
    assert scope.scope_allows({"customer_ids": ["*"]}, item) is True
    assert scope.scope_allows({"customer_ids": ["customer-1"]}, item) is True
    assert scope.scope_allows({"customer_ids": ["customer-2"]}, item) is False
    assert scope.scoped_query_value("customer-2", {"customer_ids": ["customer-1"]}, "customer_ids") == (
        False,
        "customer-2",
    )
    assert scope.scoped_query_value("", {"customer_ids": ["customer-1"]}, "customer_ids") == (
        True,
        "",
    )


def test_payload_scope_detection_and_single_scope_defaults() -> None:
    payload = {"payload": {"project_scope": {"customer_id": "customer-1"}}}
    assert scope.has_explicit_project_scope(payload) is True
    assert scope.scope_item_from_event_payload(payload) == {
        "tenant_id": "default",
        "delivery_namespace": "default",
        "customer_id": "customer-1",
        "project_id": "",
        "site_id": "",
    }

    mutable: dict[str, str] = {}
    scope.apply_single_scope_defaults(
        mutable,
        {
            "tenant_ids": ["tenant-a"],
            "delivery_namespaces": ["pilot"],
            "customer_ids": ["customer-1"],
            "project_ids": ["project-1"],
            "site_ids": ["site-1"],
        },
    )
    assert mutable == {
        "tenant_id": "tenant-a",
        "delivery_namespace": "pilot",
        "customer_id": "customer-1",
        "project_id": "project-1",
        "site_id": "site-1",
    }


def test_multi_project_scope_uses_only_allowed_explicit_defaults() -> None:
    auth_body = {
        "operator_auth": {
            "operator": {
                "project_scope": {
                    "customer_ids": ["demo-customer", "site-customer"],
                    "project_ids": ["demo-project", "site-project"],
                    "site_ids": ["demo-site", "site-a"],
                    "default_customer_ids": ["site-customer"],
                    "default_project_ids": ["site-project"],
                    "default_site_ids": ["site-a"],
                }
            }
        }
    }
    normalized = scope.operator_project_scope(auth_body)
    payload: dict[str, str] = {}

    scope.apply_single_scope_defaults(payload, normalized)

    assert payload == {
        "customer_id": "site-customer",
        "project_id": "site-project",
        "site_id": "site-a",
    }

    invalid_default = dict(normalized)
    invalid_default["default_project_ids"] = ["other-project"]
    invalid_payload: dict[str, str] = {}
    scope.apply_single_scope_defaults(invalid_payload, invalid_default)
    assert "project_id" not in invalid_payload


def test_customer_project_artifact_and_resource_scope_extractors() -> None:
    package_payload = {
        "package": {
            "customer": {
                "tenant_id": "tenant-a",
                "delivery_namespace": "pilot",
                "customer_id": "customer-1",
                "project_id": "project-1",
            },
            "site": {"site_id": "site-1"},
        }
    }
    assert scope.scope_item_from_package(package_payload) == {
        "tenant_id": "tenant-a",
        "delivery_namespace": "pilot",
        "customer_id": "customer-1",
        "project_id": "project-1",
        "site_id": "site-1",
    }
    assert scope.scope_item_from_dossier({"dossier": package_payload["package"]})["site_id"] == "site-1"
    assert scope.scope_item_from_proposal({"proposal": package_payload["package"]})["project_id"] == "project-1"
    assert scope.scope_item_from_resource(
        {"project_scope": {"tenant_id": "tenant-a", "customer_id": "customer-1"}}
    ) == {
        "tenant_id": "tenant-a",
        "delivery_namespace": "",
        "customer_id": "customer-1",
        "project_id": "",
        "site_id": "",
    }
    assert scope.resource_has_explicit_scope({"project_scope": {"tenant_id": "tenant-a"}}) is True


def test_field_project_scope_is_leaf_and_route_imports_it() -> None:
    helper_path = Path("askme/api/routes/field_project_scope.py")
    route_path = Path("askme/api/routes/field.py")
    helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
    route_tree = ast.parse(route_path.read_text(encoding="utf-8"))
    helper_imports = {
        node.module
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    route_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    route_functions = {
        node.name
        for node in ast.walk(route_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "fastapi" not in helper_imports
    assert "askme.health_server" not in helper_imports
    assert "askme.api.routes.field_project_scope" in {module for module, _ in route_imports}
    assert "_operator_project_scope" not in route_functions
    assert "_scope_allows" not in route_functions
    assert "_resource_has_explicit_scope" not in route_functions
