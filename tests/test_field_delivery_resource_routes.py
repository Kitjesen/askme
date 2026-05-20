from __future__ import annotations

import ast
from pathlib import Path


def test_field_delivery_resource_routes_are_registered_from_split_module() -> None:
    route_path = Path("askme/api/routes/field.py")
    delivery_route_path = Path("askme/api/routes/field_delivery_resources.py")
    route_tree = ast.parse(route_path.read_text(encoding="utf-8"))
    delivery_route_tree = ast.parse(delivery_route_path.read_text(encoding="utf-8"))

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
    split_route_defs = {
        node.name
        for node in ast.walk(delivery_route_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "register_delivery_resource_routes" in field_imports[
        "askme.api.routes.field_delivery_resources"
    ]
    assert "field_delivery_resource_registry" not in field_route_defs
    assert "field_delivery_resource_governance_request_review" not in field_route_defs
    assert "field_delivery_resource_registry" in split_route_defs
    assert "field_delivery_resource_governance_request_review" in split_route_defs
    assert "askme.pipeline.field.delivery_resources" not in field_imports


def test_field_delivery_resource_route_module_owns_delivery_resource_domain_calls() -> None:
    delivery_route_path = Path("askme/api/routes/field_delivery_resources.py")
    delivery_route_tree = ast.parse(delivery_route_path.read_text(encoding="utf-8"))

    imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(delivery_route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imports["askme.pipeline.field.delivery_resources"] == {
        "create_delivery_resource_governance_request",
        "disable_delivery_resource",
        "escalate_overdue_delivery_resource_governance_requests",
        "list_delivery_resource_governance_requests",
        "list_delivery_resource_registry",
        "list_delivery_resource_revisions",
        "review_delivery_resource_governance_request",
        "rollback_delivery_resource_registry",
        "upsert_delivery_resource",
    }
