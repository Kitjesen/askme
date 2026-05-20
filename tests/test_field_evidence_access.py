from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any

from askme.api.services.field_evidence_access import (
    field_evidence_candidate_paths,
    field_evidence_detail_references_path,
    field_evidence_scope_allows,
    resolve_field_evidence_path,
)


def test_resolve_field_evidence_path_only_allows_workspace_evidence_roots(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "evidence" / "unit.txt"
    outside = tmp_path / "askme" / "health_server.py"
    artifact.parent.mkdir(parents=True)
    outside.parent.mkdir(parents=True)
    artifact.write_text("ok", encoding="utf-8")
    outside.write_text("no", encoding="utf-8")

    assert resolve_field_evidence_path("artifacts/evidence/unit.txt", cwd=tmp_path) == artifact.resolve()
    assert resolve_field_evidence_path("askme/health_server.py", cwd=tmp_path) is None
    assert resolve_field_evidence_path("artifacts/../askme/health_server.py", cwd=tmp_path) is None
    assert resolve_field_evidence_path("https://example.com/evidence.txt", cwd=tmp_path) is None
    assert resolve_field_evidence_path("data:text/plain,no", cwd=tmp_path) is None


def test_field_evidence_candidate_paths_reads_encoded_evidence_urls() -> None:
    paths = field_evidence_candidate_paths(
        "/api/field/evidence?path=artifacts%2Fevidence%2Funit%20file.txt"
    )

    assert paths == [
        "/api/field/evidence?path=artifacts%2Fevidence%2Funit%20file.txt",
        "artifacts/evidence/unit file.txt",
    ]


def test_field_evidence_detail_references_nested_raw_or_encoded_path() -> None:
    raw_path = "artifacts/evidence/unit file.txt"
    resolved = Path(raw_path).resolve()
    detail = {
        "event": {
            "image_path": "/api/field/evidence?path=artifacts%2Fevidence%2Funit%20file.txt",
            "audit": [{"note": "other"}],
        }
    }

    assert field_evidence_detail_references_path(detail, raw_path, resolved) is True
    assert field_evidence_detail_references_path(detail, "artifacts/evidence/missing.txt", resolved) is False


def test_field_evidence_scope_allows_only_scoped_referenced_events() -> None:
    raw_path = "artifacts/evidence/tenant-a.txt"
    resolved = Path(raw_path).resolve()

    async def dispatch(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if name == "detail_payload":
            event_id = args[0]
            if event_id == "allowed":
                return {
                    "found": True,
                    "event": {
                        "tenant_id": "tenant-a",
                        "image_path": raw_path,
                    },
                }
            if event_id == "unreferenced":
                return {
                    "found": True,
                    "event": {
                        "tenant_id": "tenant-a",
                        "image_path": "artifacts/evidence/other.txt",
                    },
                }
            return {
                "found": True,
                "event": {
                    "tenant_id": "tenant-b",
                    "image_path": raw_path,
                },
            }
        if name == "list_payload":
            return {
                "events": [
                    {"tenant_id": "tenant-b", "image_path": raw_path},
                    {"tenant_id": "tenant-a", "image_path": raw_path},
                ]
            }
        raise AssertionError(name)

    allowed = asyncio.run(
        field_evidence_scope_allows(
            raw_path,
            resolved,
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
            event_id="allowed",
        )
    )
    wrong_tenant = asyncio.run(
        field_evidence_scope_allows(
            raw_path,
            resolved,
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
            event_id="denied",
        )
    )
    unreferenced = asyncio.run(
        field_evidence_scope_allows(
            raw_path,
            resolved,
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
            event_id="unreferenced",
        )
    )
    listed = asyncio.run(
        field_evidence_scope_allows(
            raw_path,
            resolved,
            {"tenant_ids": ["tenant-a"]},
            dispatch_field_operations=dispatch,
            scope_allows=_tenant_scope_allows,
            scope_item_from_event_detail=_event_scope_item,
        )
    )

    assert allowed is True
    assert wrong_tenant is False
    assert unreferenced is False
    assert listed is True


def test_field_evidence_access_service_is_leaf_and_route_imports_service() -> None:
    service_path = Path("askme/api/services/field_evidence_access.py")
    route_path = Path("askme/api/routes/field_events.py")
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
    route_functions = {
        node.name
        for node in ast.walk(route_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "fastapi" not in service_imports
    assert "askme.health_server" not in service_imports
    assert (
        "askme.api.services.field_evidence_access",
        ("field_evidence_scope_allows", "resolve_field_evidence_path"),
    ) in route_imports
    assert "_resolve_field_evidence_path" not in route_functions
    assert "_field_evidence_scope_allows" not in route_functions
    assert "_field_evidence_detail_references_path" not in route_functions


def _tenant_scope_allows(scope: dict[str, list[str]], item: dict[str, Any]) -> bool:
    tenant_id = str(item.get("tenant_id") or "")
    return not scope.get("tenant_ids") or tenant_id in scope["tenant_ids"]


def _event_scope_item(detail: dict[str, Any]) -> dict[str, Any]:
    return detail.get("event", detail)
