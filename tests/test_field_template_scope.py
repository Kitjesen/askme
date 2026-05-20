from __future__ import annotations

import ast
from pathlib import Path

from askme.api.routes.field_template_scope import (
    scope_allows_template,
    template_release_request_summary,
)


def test_template_scope_allows_global_templates_and_blocks_cross_tenant() -> None:
    scoped = {"tenant_ids": ["tenant-a"], "delivery_namespaces": ["pilot"]}

    assert scope_allows_template(scoped, {"tenant_id": "default", "delivery_namespace": "default"})
    assert scope_allows_template(scoped, {"tenant_id": "tenant-a", "delivery_namespace": "pilot"})
    assert not scope_allows_template(scoped, {"tenant_id": "tenant-b", "delivery_namespace": "pilot"})
    assert scope_allows_template(
        {"tenant_ids": ["*"], "delivery_namespaces": ["pilot"]},
        {"tenant_id": "tenant-b", "delivery_namespace": "pilot"},
    )


def test_template_release_request_summary_counts_filtered_statuses() -> None:
    summary = template_release_request_summary(
        [
            {"status": "pending"},
            {"status": "approved"},
            {"status": "apply_failed"},
            {"status": "rejected"},
            {"status": "applying"},
        ],
        scope_filtered=True,
    )

    assert summary == {
        "pending_count": 1,
        "applying_count": 1,
        "apply_failed_count": 1,
        "approved_count": 1,
        "rejected_count": 1,
        "scope_filtered": True,
    }


def test_field_template_scope_helper_is_leaf_and_route_uses_it() -> None:
    helper_path = Path("askme/api/routes/field_template_scope.py")
    route_path = Path("askme/api/routes/field_customer_project_templates.py")
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
        node.module
        for node in ast.walk(route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    route_defs = {
        node.name
        for node in ast.walk(route_tree)
        if isinstance(node, ast.FunctionDef)
    }

    assert "fastapi" not in helper_imports
    assert "askme.health_server" not in helper_imports
    assert "askme.api.routes.field_template_scope" in route_imports
    assert "_scope_template_catalog" not in route_defs
    assert "_scope_allows_template" not in route_defs
    assert "_scope_template_release_requests_payload" not in route_defs


def test_field_template_routes_are_registered_from_split_module() -> None:
    route_path = Path("askme/api/routes/field.py")
    template_route_path = Path("askme/api/routes/field_customer_project_templates.py")
    route_tree = ast.parse(route_path.read_text(encoding="utf-8"))
    template_route_tree = ast.parse(template_route_path.read_text(encoding="utf-8"))

    field_route_defs = {
        node.name
        for node in ast.walk(route_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    field_imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    split_route_imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(template_route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    split_route_defs = {
        node.name
        for node in ast.walk(template_route_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "register_customer_project_template_routes" in field_imports[
        "askme.api.routes.field_customer_project_templates"
    ]
    assert "field_customer_project_templates" not in field_route_defs
    assert "field_customer_project_template_release" not in field_route_defs
    assert "field_customer_project_template_release" in split_route_defs
    assert {
        "list_customer_project_templates",
        "create_customer_project_template_release_request",
        "update_customer_project_template_release",
    } <= split_route_imports["askme.pipeline.field.customer_project_templates"]
