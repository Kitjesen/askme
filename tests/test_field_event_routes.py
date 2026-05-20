from __future__ import annotations

import ast
from pathlib import Path


def test_field_event_routes_are_extracted_from_main_field_router() -> None:
    field_path = Path("askme/api/routes/field.py")
    event_path = Path("askme/api/routes/field_events.py")
    field_tree = ast.parse(field_path.read_text(encoding="utf-8"))
    event_tree = ast.parse(event_path.read_text(encoding="utf-8"))

    field_functions = {
        node.name
        for node in ast.walk(field_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    event_functions = {
        node.name
        for node in ast.walk(event_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    field_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(field_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    event_imports = {
        node.module
        for node in ast.walk(event_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(event_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    extracted = {
        "field_scenarios",
        "field_scenario_acceptance",
        "field_events",
        "field_event_detail",
        "field_evidence",
        "field_event_trigger",
        "field_event_close",
        "field_event_request_close",
        "field_event_acknowledge",
        "field_event_resend_notification",
        "field_event_report",
    }

    assert extracted.isdisjoint(field_functions)
    assert extracted <= event_functions
    assert "_field_post" not in field_functions
    assert (
        "askme.api.routes.field_events",
        ("register_field_event_routes",),
    ) in field_imports
    assert "askme.health_server" not in event_imports
