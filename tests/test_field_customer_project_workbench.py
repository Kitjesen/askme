from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from askme.api.schemas.customer_projects import (
    CustomerProjectWorkbenchResponse,
    ManagedObjectDirectoryResponse,
)
from askme.api.services.field_customer_project_workbench import (
    build_customer_project_workbench_payload,
    customer_project_runtime_blueprint_binding,
)
from askme.api.services.field_managed_object_directory import (
    filter_managed_object_directory_rows,
    managed_object_directory_rows,
    managed_object_directory_summary,
)
from askme.api.services.field_project_catalog_scope import (
    customer_rows_for_projects,
    scope_project_catalog,
    scope_site_catalog,
)
from askme.api.services.field_resource_catalog_scope import (
    scope_acceptance_registry,
    scope_resource_catalog,
)


def test_customer_project_workbench_payload_is_customer_readable() -> None:
    payload = build_customer_project_workbench_payload(
        project_catalog={
            "summary": {
                "delivery_acceptance_gate_status": "ready",
                "project_count": 2,
            },
            "filters": {"industry": "park"},
        },
        template_catalog={
            "summary": {"overall_status": "manual_check", "template_count": 1},
            "templates": [{"template_id": f"tpl-{index}"} for index in range(25)],
        },
        resource_catalog={
            "summary": {"overall_status": "blocked", "resource_count": 3},
            "resources": [{"resource_id": f"res-{index}"} for index in range(60)],
        },
        object_summary={"overall_status": "ready", "object_count": 4},
        object_rows=[{"object_id": f"obj-{index}"} for index in range(60)],
        projects=[{"project_id": f"project-{index}"} for index in range(25)],
        readiness={
            "overall_status": "manual_check",
            "customer_status": "客户项目工作台已就绪",
            "release_claim": "仅声明可验收的交付范围。",
            "next_step": "核对对象目录。",
            "summary": {"project_count": 2},
        },
        scope_filtered=True,
    )

    schema_payload = CustomerProjectWorkbenchResponse.model_validate(payload)

    assert schema_payload.workbench_type == "askme.solution_provider_customer_project_workbench.v1"
    assert schema_payload.scope_filtered is True
    assert payload["workbench_type"] == "askme.solution_provider_customer_project_workbench.v1"
    assert payload["scope_filtered"] is True
    assert payload["filters"] == {"industry": "park"}
    assert payload["customer_status"] == "客户项目工作台已就绪"
    assert payload["delivery_surfaces"][0]["customer_label"] == "客户项目目录"
    assert payload["delivery_surfaces"][2]["customer_description"] == (
        "展示车辆、设备、游客、烟火、垃圾桶等现场对象及能力配置。"
    )
    vocabulary = {
        item["internal"]: item["customer_label"]
        for item in payload["customer_vocabulary"]
    }
    assert vocabulary["tenant_id"] == "客户空间"
    assert vocabulary["package_delivery_gate"] == "交付包准入检查"
    assert "客户只能看到自己的项目、对象、证据和交付包。" in {
        item["customer_value"] for item in payload["customer_acceptance_flow"]
    }
    assert payload["customer_readable_contract"]["positioning"] == (
        "面向多客户、多行业现场的可复用机器人方案交付平台。"
    )
    assert payload["runtime_blueprint_binding"]["binding_type"] == (
        "askme.customer_project.runtime_blueprint_binding.v1"
    )
    assert payload["runtime_blueprint_binding"]["policy"][
        "customer_project_must_select_runtime_blueprint"
    ] is True
    assert payload["delivery_chain"]["chain_type"] == "askme.customer_project.delivery_chain.v1"
    assert payload["delivery_chain"]["step_count"] == 6
    assert [item["step_id"] for item in payload["delivery_chain"]["steps"]] == [
        "project_scope",
        "template_market",
        "managed_object_directory",
        "capability_resource_binding",
        "runtime_blueprint",
        "acceptance_package",
    ]
    assert payload["delivery_chain"]["steps"][4]["source_surface_id"] == (
        "runtime_blueprint_binding"
    )
    assert payload["delivery_chain"]["policy"][
        "capability_resources_must_be_bound_to_managed_objects"
    ] is True
    assert len(payload["customer_projects"]["projects"]) == 20
    assert len(payload["template_market"]["templates"]) == 20
    assert len(payload["managed_object_directory"]["objects"]) == 50
    assert len(payload["delivery_resources"]["resources"]) == 50
    assert _mojibake_strings(payload) == []


def test_runtime_blueprint_binding_recommends_project_blueprint_and_blocks_missing_bindings() -> None:
    blueprints_payload = {
        "items": [
            {
                "name": "edge_robot",
                "title": "园区巡检机器人运行时",
                "product_stage": "pilot",
                "customer_visible": True,
                "primary_loop": "voice",
                "deployment_targets": ["robot_edge_pc", "customer_pilot_site"],
                "capabilities": ["现场事件接入", "任务运行交接"],
                "scenarios": ["车辆违停拍照取证", "访客问路和带路服务"],
                "external_services": ["DingTalk"],
                "safety_boundaries": ["LLM only plans; arbiter executes"],
                "validation_commands": ["python -m pytest tests/test_field.py -q"],
                "delivery_package": {
                    "package_id": "blueprint.edge_robot",
                    "status": "ready_for_site_validation",
                    "release_boundary": "pilot only",
                    "customer_claim": "ready for site validation",
                },
            },
            {
                "name": "mcp",
                "title": "MCP 工具服务",
                "customer_visible": False,
                "delivery_package": {"status": "ready_for_site_validation"},
            },
        ]
    }

    payload = customer_project_runtime_blueprint_binding(
        projects=[
            {
                "customer_id": "fanmu",
                "customer_name": "梵木创艺园",
                "project_id": "fanmu-park",
                "industry": "park",
            },
            {
                "customer_id": "demo",
                "project_id": "missing-bindings",
                "industry": "park",
            },
        ],
        object_rows=[
            {
                "project_id": "fanmu-park",
                "object_id": "parking",
                "delivery_status": "ready",
                "scenario_ids": ["illegal_parking"],
                "bindings": {
                    "vision_models": ["vehicle-detector"],
                    "sensor_protocols": ["camera-detection-json"],
                    "skill_packages": ["security-patrol"],
                    "acceptance_tests": ["tests/test_field.py::test_illegal_parking"],
                },
            },
            {
                "project_id": "missing-bindings",
                "object_id": "trash-bin",
                "delivery_status": "blocked",
                "scenario_ids": ["trash_bin_full"],
                "bindings": {"skill_packages": ["cleaning-alert"]},
            },
        ],
        blueprints_payload=blueprints_payload,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["summary"] == {
        "project_count": 2,
        "ready_project_count": 1,
        "manual_check_project_count": 0,
        "blocked_project_count": 1,
        "available_customer_blueprint_count": 1,
    }
    ready_project = payload["project_bindings"][0]
    blocked_project = payload["project_bindings"][1]
    assert ready_project["selected_blueprint"]["name"] == "edge_robot"
    assert ready_project["selected_blueprint"]["external_services"] == ["DingTalk"]
    assert ready_project["selected_blueprint"]["safety_boundaries"] == [
        "LLM only plans; arbiter executes"
    ]
    assert ready_project["selected_blueprint"]["validation_commands"] == [
        "python -m pytest tests/test_field.py -q"
    ]
    assert ready_project["selected_blueprint"]["release_boundary"] == "pilot only"
    assert ready_project["selected_blueprint"]["acceptance_boundary"] == "pilot only"
    assert ready_project["selected_blueprint"]["customer_claim"] == "ready for site validation"
    assert ready_project["selected_blueprint"]["customer_status"] == "可进入现场验证"
    assert ready_project["selected_blueprint"]["delivery_actions"] == [
        "运行现场验证用例。",
        "归档语音、通知、机器人运行和客户复核证据。",
        "签收前复核安全边界和人工接管方案。",
    ]
    assert ready_project["status"] == "ready"
    assert ready_project["match_reason"] == "按客户项目行业 park 推荐运行蓝图。"
    assert blocked_project["status"] == "blocked"
    assert blocked_project["missing_binding_types"] == ["识别模型", "传感器协议", "验收用例"]
    assert "现场对象仍处于阻断状态" in " ".join(blocked_project["blockers"])
    assert _mojibake_strings(payload) == []


def test_field_workbench_builder_is_leaf_and_route_imports_service() -> None:
    helper_path = Path("askme/api/services/field_customer_project_workbench.py")
    compatibility_path = Path("askme/api/routes/field_customer_project_workbench.py")
    route_path = Path("askme/api/routes/field_product_catalog.py")
    helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
    compatibility_tree = ast.parse(compatibility_path.read_text(encoding="utf-8"))
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
    compatibility_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(compatibility_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "fastapi" not in helper_imports
    assert "askme.health_server" not in helper_imports
    assert "askme.api.routes.field_customer_project_workbench" not in {
        module for module, _ in route_imports
    }
    assert (
        "askme.api.services.field_customer_project_workbench",
        ("build_customer_project_workbench_payload",),
    ) in route_imports
    assert (
        "askme.api.services.field_customer_project_workbench",
        (
            "build_customer_project_workbench_payload",
            "customer_project_delivery_surfaces",
            "customer_project_runtime_blueprint_binding",
            "customer_project_term_cards",
        ),
    ) in compatibility_imports


def test_managed_object_directory_service_builds_actions_filters_and_summary() -> None:
    rows = managed_object_directory_rows([
        {
            "tenant_id": "tenant-a",
            "delivery_namespace": "southwest",
            "customer_id": "fanmu",
            "customer_name": "梵木创艺园",
            "project_id": "fanmu-park",
            "site_id": "site-a",
            "managed_objects": [
                {
                    "object_id": "vehicles",
                    "display_name": "车辆违停检测",
                    "category": "vehicle",
                    "bindings": {
                        "acceptance_tests": ["tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"],
                    },
                    "resource_binding_status": {
                        "overall_status": "manual_check",
                        "check_count": 1,
                        "checks": [
                            {
                                "status": "unregistered",
                                "resource_type": "vision_models",
                                "resource_id": "vehicle-detection",
                            }
                        ],
                    },
                    "acceptance_status": {
                        "status": "file_missing",
                        "acceptance_checks": [
                            {
                                "status": "file_missing",
                                "reference": "missing.py::case",
                            }
                        ],
                    },
                }
            ],
        }
    ])

    assert len(rows) == 1
    row = rows[0]
    assert row["delivery_status"] == "blocked"
    assert row["blocked_action_count"] == 1
    assert row["manual_check_action_count"] == 1
    assert {item["action"] for item in row["action_plan"]} == {
        "register_delivery_resource",
        "fix_acceptance_test_reference",
    }
    filtered, filters = filter_managed_object_directory_rows(
        rows,
        delivery_status="blocked",
        category="vehicle",
        customer_visible="true",
    )
    summary = managed_object_directory_summary(
        filtered,
        projects=[{"project_id": "fanmu-park"}],
        base_summary={},
        filtered=bool(filters),
    )
    directory_payload = ManagedObjectDirectoryResponse.model_validate(
        {
            "directory_type": "askme.customer_project_managed_object_directory",
            "root": "deploy/site-profiles",
            "check_env": False,
            "filters": filters,
            "summary": summary,
            "objects": filtered,
            "customer_status": "对象目录可用于交付复核。",
            "next_step": "处理阻断对象。",
        }
    )

    assert filters == {
        "delivery_status": "blocked",
        "category": "vehicle",
        "customer_visible": True,
    }
    assert directory_payload.objects[0]["object_id"] == "vehicles"
    assert summary["object_count"] == 1
    assert summary["overall_status"] == "blocked"
    assert summary["action_count"] == 2
    assert _mojibake_strings({"rows": rows, "summary": summary}) == []


def test_managed_object_directory_service_is_leaf_and_route_imports_service() -> None:
    service_path = Path("askme/api/services/field_managed_object_directory.py")
    route_path = Path("askme/api/routes/field_product_catalog.py")
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
        "askme.api.services.field_managed_object_directory",
        (
            "filter_managed_object_directory_rows",
            "managed_object_directory_rows",
            "managed_object_directory_summary",
        ),
    ) in route_imports
    assert "_managed_object_directory_rows" not in route_functions
    assert "_managed_object_directory_summary" not in route_functions
    assert "_filter_managed_object_directory_rows" not in route_functions


def test_field_resource_catalog_scope_filters_resources_and_acceptance() -> None:
    scope = {"tenant_ids": ["tenant-a"]}
    resource_payload = {
        "consumers": [
            {"scope_type": "project", "tenant_id": "tenant-a", "status": "linked"},
            {"scope_type": "project", "tenant_id": "tenant-b", "status": "linked"},
            {"scope_type": "template", "template_id": "park", "status": "manual_check"},
        ],
        "resources": [
            {
                "resource_type": "vision_models",
                "resource_id": "vehicle-detector",
                "tenant_id": "tenant-a",
                "consumer_count": 2,
                "consumers": [
                    {"scope_type": "project", "tenant_id": "tenant-a", "status": "linked"},
                    {"scope_type": "project", "tenant_id": "tenant-b", "status": "unregistered"},
                ],
            },
            {
                "resource_type": "sensor_protocols",
                "resource_id": "smoke-sensor",
                "tenant_id": "tenant-b",
                "consumer_count": 1,
                "consumers": [
                    {"scope_type": "project", "tenant_id": "tenant-b", "status": "linked"},
                ],
            },
            {
                "resource_type": "skill_packages",
                "resource_id": "park-guide",
                "consumer_count": 1,
                "consumers": [
                    {"scope_type": "template", "template_id": "park", "status": "manual_check"},
                ],
            },
        ],
    }
    acceptance_payload = {
        "consumers": [
            {"scope_type": "project", "tenant_id": "tenant-a", "status": "passed", "project_id": "p1"},
            {"scope_type": "project", "tenant_id": "tenant-b", "status": "missing", "project_id": "p2"},
            {"scope_type": "template", "template_id": "park", "status": "not_run"},
        ],
        "references": [
            {
                "reference": "tests/test_field.py::case",
                "consumers": [
                    {"scope_type": "project", "tenant_id": "tenant-a", "status": "passed"},
                    {"scope_type": "project", "tenant_id": "tenant-b", "status": "missing"},
                    {"scope_type": "template", "template_id": "park", "status": "not_run"},
                ],
            }
        ],
    }

    scoped_resources = scope_resource_catalog(
        resource_payload,
        scope,
        scope_allows=_tenant_scope_allows,
        scope_item_from_resource=lambda resource: {"tenant_id": resource.get("tenant_id")},
        resource_has_explicit_scope=lambda resource: bool(resource.get("tenant_id")),
    )
    scoped_acceptance = scope_acceptance_registry(
        acceptance_payload,
        scope,
        scope_allows=_tenant_scope_allows,
    )

    assert [item["resource_id"] for item in scoped_resources["resources"]] == [
        "vehicle-detector",
        "park-guide",
    ]
    assert scoped_resources["resources"][0]["consumer_count"] == 1
    assert scoped_resources["summary"]["scope_filtered"] is True
    assert scoped_resources["summary"]["resource_count"] == 2
    assert scoped_resources["summary"]["consumer_count"] == 2
    assert scoped_acceptance["summary"]["scope_filtered"] is True
    assert scoped_acceptance["summary"]["consumer_count"] == 2
    assert scoped_acceptance["references"][0]["consumer_count"] == 2
    assert scoped_acceptance["references"][0]["status"] == "manual_check"


def test_field_resource_catalog_scope_service_is_leaf_and_route_imports_service() -> None:
    service_path = Path("askme/api/services/field_resource_catalog_scope.py")
    route_path = Path("askme/api/routes/field_product_catalog.py")
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
        "askme.api.services.field_resource_catalog_scope",
        ("scope_acceptance_registry", "scope_resource_catalog"),
    ) in route_imports
    assert "_scope_resource_catalog" not in route_functions
    assert "_resource_summary" not in route_functions
    assert "_registry_summary" not in route_functions


def test_field_project_catalog_scope_filters_projects_sites_and_customers() -> None:
    scope = {"tenant_ids": ["tenant-a"]}
    project_payload = {
        "summary": {"project_count": 3, "customer_count": 2},
        "projects": [
            {
                "tenant_id": "tenant-a",
                "delivery_namespace": "southwest",
                "customer_id": "fanmu",
                "customer_name": "梵木创艺园",
                "project_id": "fanmu-park",
                "industry": "park",
                "gate_status": "ready",
            },
            {
                "tenant_id": "tenant-a",
                "delivery_namespace": "southwest",
                "customer_id": "fanmu",
                "customer_name": "梵木创艺园",
                "project_id": "fanmu-stage2",
                "industry": "park",
                "gate_status": "manual_check",
            },
            {
                "tenant_id": "tenant-b",
                "delivery_namespace": "east",
                "customer_id": "factory",
                "customer_name": "示例工厂",
                "project_id": "factory-a",
                "industry": "factory",
                "gate_status": "ready",
            },
        ],
    }
    site_payload = {
        "summary": {"site_count": 3},
        "sites": [
            {"tenant_id": "tenant-a", "site_id": "site-a", "status": "passed", "deployment_stage": "production_ready"},
            {"tenant_id": "tenant-a", "site_id": "site-b", "status": "failed", "deployment_stage": "pilot"},
            {"tenant_id": "tenant-b", "site_id": "site-c", "status": "passed", "deployment_stage": "production_ready"},
        ],
    }

    scoped_projects = scope_project_catalog(
        project_payload,
        scope,
        scope_allows=_tenant_scope_allows,
    )
    scoped_sites = scope_site_catalog(
        site_payload,
        scope,
        scope_allows=_tenant_scope_allows,
        scope_item_from_site=lambda site: {"tenant_id": site.get("tenant_id")},
    )
    customer_rows = customer_rows_for_projects(scoped_projects["projects"])

    assert [item["project_id"] for item in scoped_projects["projects"]] == [
        "fanmu-park",
        "fanmu-stage2",
    ]
    assert scoped_projects["summary"]["scope_filtered"] is True
    assert scoped_projects["customers"] == customer_rows
    assert customer_rows == [
        {
            "tenant_id": "tenant-a",
            "delivery_namespace": "southwest",
            "customer_id": "fanmu",
            "customer_name": "梵木创艺园",
            "project_count": 2,
            "projects": ["fanmu-park", "fanmu-stage2"],
            "industries": ["park"],
        }
    ]
    assert [item["site_id"] for item in scoped_sites["sites"]] == ["site-a", "site-b"]
    assert scoped_sites["summary"]["site_count"] == 2
    assert scoped_sites["summary"]["configured_count"] == 1
    assert scoped_sites["summary"]["blocked_count"] == 1
    assert scoped_sites["summary"]["production_ready_count"] == 1


def test_field_project_catalog_scope_service_is_leaf_and_route_imports_service() -> None:
    service_path = Path("askme/api/services/field_project_catalog_scope.py")
    route_path = Path("askme/api/routes/field_product_catalog.py")
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
        "askme.api.services.field_project_catalog_scope",
        ("scope_project_catalog", "scope_site_catalog"),
    ) in route_imports
    assert "_scope_project_catalog" not in route_functions
    assert "_scope_site_catalog" not in route_functions
    assert "_customer_rows_for_projects" not in route_functions


def test_customer_project_workbench_source_has_no_common_mojibake() -> None:
    source = Path("askme/api/services/field_customer_project_workbench.py").read_text(
        encoding="utf-8"
    )

    assert _mojibake_strings(source) == []


def test_managed_object_directory_source_has_no_common_mojibake() -> None:
    source = Path("askme/api/services/field_managed_object_directory.py").read_text(
        encoding="utf-8"
    )

    assert _mojibake_strings(source) == []


def test_field_resource_catalog_scope_source_has_no_common_mojibake() -> None:
    source = Path("askme/api/services/field_resource_catalog_scope.py").read_text(
        encoding="utf-8"
    )

    assert _mojibake_strings(source) == []


def test_field_project_catalog_scope_source_has_no_common_mojibake() -> None:
    source = Path("askme/api/services/field_project_catalog_scope.py").read_text(
        encoding="utf-8"
    )

    assert _mojibake_strings(source) == []


def _tenant_scope_allows(scope: dict[str, list[str]], item: dict[str, Any]) -> bool:
    tenant_id = str(item.get("tenant_id") or "")
    return not scope.get("tenant_ids") or tenant_id in scope["tenant_ids"]


def _mojibake_strings(value: Any) -> list[str]:
    suspicious = _common_mojibake_tokens()
    found: list[str] = []
    if isinstance(value, str):
        if "\ufffd" in value or any("\ue000" <= char <= "\uf8ff" for char in value):
            found.append(value)
        elif any(token in value for token in suspicious):
            found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_mojibake_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_mojibake_strings(item))
    return found


def _common_mojibake_tokens() -> tuple[str, ...]:
    readable_tokens = ("客户", "项目", "对象", "交付", "验收", "范围", "模板")
    tokens: list[str] = []
    for token in readable_tokens:
        mojibake = token.encode("utf-8").decode("gbk", errors="ignore")
        if mojibake and mojibake != token:
            tokens.append(mojibake)
    return tuple(tokens)
