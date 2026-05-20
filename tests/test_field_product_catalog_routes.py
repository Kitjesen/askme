from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from askme.api.routes.field_product_catalog import FieldProductCatalogPipeline
from askme.api.routes.field_product_catalog import create_field_product_catalog_router


def test_product_catalog_routes_are_extracted_from_main_field_router() -> None:
    field_path = Path("askme/api/routes/field.py")
    product_path = Path("askme/api/routes/field_product_catalog.py")
    field_tree = ast.parse(field_path.read_text(encoding="utf-8"))
    product_tree = ast.parse(product_path.read_text(encoding="utf-8"))

    field_functions = {
        node.name
        for node in ast.walk(field_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    product_functions = {
        node.name
        for node in ast.walk(product_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    field_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(field_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    product_imports = {
        node.module
        for node in ast.walk(product_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(product_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    extracted = {
        "field_site_profiles",
        "field_customer_projects",
        "field_customer_project_managed_object_directory",
        "field_customer_project_acceptance_registry",
        "field_customer_project_resource_catalog",
        "field_solution_delivery_readiness",
        "field_customer_project_workbench",
        "field_product_launch_readiness",
    }

    assert extracted.isdisjoint(field_functions)
    assert extracted <= product_functions
    assert "create_field_product_catalog_router" in product_functions
    assert (
        "askme.api.routes.field_product_catalog",
        ("register_field_product_catalog_routes",),
    ) in field_imports
    assert "askme.health_server" not in product_imports


def test_field_site_profiles_uses_injected_pipeline_builder() -> None:
    calls: list[dict[str, Any]] = []

    def build_site_profile_catalog(root: Path, *, check_env: bool = False) -> dict[str, Any]:
        calls.append({"root": root, "check_env": check_env})
        return {
            "root": str(root),
            "check_env": check_env,
            "summary": {"source": "injected"},
            "sites": [],
            "customer_claim": "injected catalog",
            "next_step": "continue",
        }

    app = FastAPI()
    app.include_router(
        create_field_product_catalog_router(
            dispatch_field_operations=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
            mission_json=lambda payload, **kwargs: JSONResponse(
                payload,
                status_code=kwargs.get("status_code", 200),
            ),
            project_read_auth=lambda _request: (None, {}),
            operator_project_scope=lambda _body: {
                "tenant_ids": [],
                "delivery_namespaces": [],
                "customer_ids": [],
                "project_ids": [],
                "site_ids": [],
            },
            scope_allows=lambda _scope, _item: True,
            scope_item_from_site=lambda item: item,
            scope_item_from_resource=lambda item: item,
            resource_has_explicit_scope=lambda _item: False,
            site_profile_root=lambda: Path("profiles"),
            template_root=lambda: Path("templates"),
            delivery_resource_root=lambda: Path("resources"),
            identity_readiness_payload=lambda: {"status": "demo_only"},
            pipeline=FieldProductCatalogPipeline(
                build_site_profile_catalog=build_site_profile_catalog,
            ),
            cors_options_response=lambda methods: JSONResponse({"methods": methods}),
            logger=logging.getLogger("tests.field_product_catalog_routes"),
        )
    )

    response = TestClient(app).get("/api/field/site-profiles?check_env=true")

    assert response.status_code == 200
    assert response.json()["summary"] == {"source": "injected"}
    assert calls == [{"root": Path("profiles"), "check_env": True}]
