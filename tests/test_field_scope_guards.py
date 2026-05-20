from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any

import askme.api.services.field_scope_guards as guards


def test_customer_project_scope_allows_unscoped_missing_and_scoped_profiles(monkeypatch) -> None:
    calls: list[str] = []

    def fake_profile(root: Path, identifier: str) -> dict[str, Any]:
        calls.append(identifier)
        if identifier == "missing":
            return {"found": False}
        return {
            "found": True,
            "profile": {"tenant_id": "tenant-a" if identifier == "allowed" else "tenant-b"},
        }

    monkeypatch.setattr(guards, "get_customer_project_profile", fake_profile)

    assert guards.customer_project_scope_allows(
        Path("deploy/site-profiles"),
        "allowed",
        {},
        scope_allows=_tenant_scope_allows,
        scope_item_from_detail=_profile_scope_item,
    ) is True
    assert calls == []
    assert guards.customer_project_scope_allows(
        Path("deploy/site-profiles"),
        "missing",
        {"tenant_ids": ["tenant-a"]},
        scope_allows=_tenant_scope_allows,
        scope_item_from_detail=_profile_scope_item,
    ) is True
    assert guards.customer_project_scope_allows(
        Path("deploy/site-profiles"),
        "allowed",
        {"tenant_ids": ["tenant-a"]},
        scope_allows=_tenant_scope_allows,
        scope_item_from_detail=_profile_scope_item,
    ) is True
    assert guards.customer_project_scope_allows(
        Path("deploy/site-profiles"),
        "denied",
        {"tenant_ids": ["tenant-a"]},
        scope_allows=_tenant_scope_allows,
        scope_item_from_detail=_profile_scope_item,
    ) is False


def test_field_event_scope_allows_unscoped_missing_and_scoped_events() -> None:
    calls: list[str] = []

    async def dispatch(name: str, event_id: str) -> dict[str, Any]:
        calls.append(f"{name}:{event_id}")
        if event_id == "missing":
            return {"found": False}
        return {
            "found": True,
            "event": {"tenant_id": "tenant-a" if event_id == "allowed" else "tenant-b"},
        }

    assert asyncio.run(
        guards.field_event_scope_allows(
            "allowed",
            {},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
        )
    ) is True
    assert calls == []
    assert asyncio.run(
        guards.field_event_scope_allows(
            "missing",
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
        )
    ) is True
    assert asyncio.run(
        guards.field_event_scope_allows(
            "allowed",
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
        )
    ) is True
    assert asyncio.run(
        guards.field_event_scope_allows(
            "denied",
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
        )
    ) is False


def test_field_scope_guards_service_is_leaf_and_route_imports_service() -> None:
    service_path = Path("askme/api/services/field_scope_guards.py")
    route_path = Path("askme/api/routes/field.py")
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))
    route_tree = ast.parse(route_path.read_text(encoding="utf-8"))

    service_imports = {
        node.module
        for node in ast.walk(service_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(service_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    route_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "fastapi" not in service_imports
    assert "askme.health_server" not in service_imports
    assert "get_customer_project_profile" not in {
        alias
        for module, aliases in route_imports
        if module == "askme.pipeline.field.customer_projects"
        for alias in aliases
    }
    assert (
        "askme.api.services.field_scope_guards",
        ("customer_project_scope_allows", "field_event_scope_allows"),
    ) in route_imports


def _tenant_scope_allows(scope: dict[str, list[str]], item: dict[str, Any]) -> bool:
    tenant_id = str(item.get("tenant_id") or "")
    return not scope.get("tenant_ids") or tenant_id in scope["tenant_ids"]


def _profile_scope_item(detail: dict[str, Any]) -> dict[str, Any]:
    return detail.get("profile", detail)


def _event_scope_item(detail: dict[str, Any]) -> dict[str, Any]:
    return detail.get("event", detail)
