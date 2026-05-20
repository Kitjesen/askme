"""Customer-project workbench payload builders."""

from __future__ import annotations

from typing import Any

from askme.api.services.blueprint_payloads import (
    blueprint_runtime_summary,
    load_blueprints_payload,
)

_CUSTOMER_PROJECT_TERMS = {
    "tenant_id": "客户空间",
    "delivery_namespace": "交付空间",
    "customer_project": "客户项目",
    "managed_object": "现场对象",
    "managed_object_directory": "对象目录",
    "bindings": "能力配置",
    "vision_models": "识别能力",
    "sensor_protocols": "设备接入方式",
    "skill_packages": "业务能力",
    "acceptance_tests": "验收项",
    "delivery_resources": "交付资源",
    "package_delivery_gate": "交付包准入检查",
    "dry_run": "预检",
    "runtime": "执行服务",
    "operator_id": "操作人",
}

_CUSTOMER_PROJECT_ACCEPTANCE_FLOW = [
    {
        "step_id": "scope_isolated",
        "label": "确认客户范围",
        "customer_value": "客户只能看到自己的项目、对象、证据和交付包。",
        "acceptance_standard": "越权读取、导出、导入预检都会被拒绝或返回空结果。",
    },
    {
        "step_id": "template_selected",
        "label": "选择行业模板",
        "customer_value": "新项目从厂区、园区、仓储、景区模板复制，不从空白配置开始。",
        "acceptance_standard": "模板展示适用场景、默认对象、交付边界和客户准备项。",
    },
    {
        "step_id": "object_directory_ready",
        "label": "核对对象目录",
        "customer_value": "客户能看懂本项目覆盖哪些车辆、设备、游客、烟火、垃圾桶或通道对象。",
        "acceptance_standard": "每个对象展示识别能力、设备接入方式、业务能力、验收项和未完成原因。",
    },
    {
        "step_id": "package_preflight",
        "label": "交付包预检",
        "customer_value": "导入前先看到新增、覆盖、冲突和阻断项，避免误覆盖客户现场。",
        "acceptance_standard": "预检不写入项目；冲突和越权包不能导入。",
    },
    {
        "step_id": "customer_acceptance",
        "label": "输出验收材料",
        "customer_value": "客户按项目范围、对象覆盖、未完成项和证据结论验收。",
        "acceptance_standard": "报告不要求客户阅读 YAML、接口路径或测试节点即可判断交付状态。",
    },
]


def customer_project_term_cards() -> list[dict[str, str]]:
    return [
        {"internal": key, "customer_label": value}
        for key, value in _CUSTOMER_PROJECT_TERMS.items()
    ]


def customer_project_delivery_surfaces(
    *,
    project_catalog: dict[str, Any],
    template_catalog: dict[str, Any],
    resource_catalog: dict[str, Any],
    object_summary: dict[str, Any],
    readiness: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return customer-readable delivery surfaces for the workbench."""
    project_summary = _mapping(project_catalog.get("summary"))
    template_summary = _mapping(template_catalog.get("summary"))
    resource_summary = _mapping(resource_catalog.get("summary"))
    readiness_summary = _mapping(readiness.get("summary"))
    return [
        {
            "surface_id": "customer_projects",
            "label": "客户项目目录",
            "customer_label": "客户项目目录",
            "customer_description": "按客户、项目、现场和交付阶段管理项目范围。",
            "customer_count_label": "项目",
            "customer_action": "选择客户项目并核对对象范围。",
            "status": project_summary.get("delivery_acceptance_gate_status", "unknown"),
            "count": project_summary.get("project_count", 0),
            "api": "/api/field/customer-projects",
        },
        {
            "surface_id": "template_market",
            "label": "行业模板市场",
            "customer_label": "行业模板市场",
            "customer_description": "提供厂区、园区、仓储、景区等可复用方案模板。",
            "customer_count_label": "模板",
            "customer_action": "从合适模板创建客户项目。",
            "status": template_summary.get("overall_status", "unknown"),
            "count": template_summary.get("template_count", 0),
            "api": "/api/field/customer-project-templates",
        },
        {
            "surface_id": "managed_objects",
            "label": "对象目录",
            "customer_label": "对象目录",
            "customer_description": "展示车辆、设备、游客、烟火、垃圾桶等现场对象及能力配置。",
            "customer_count_label": "对象",
            "customer_action": "补齐对象的识别、设备接入、业务能力和验收项。",
            "status": object_summary.get("overall_status", "unknown"),
            "count": object_summary.get("object_count", 0),
            "api": "/api/field/customer-projects/managed-object-directory",
        },
        {
            "surface_id": "delivery_resources",
            "label": "交付资源",
            "customer_label": "交付资源",
            "customer_description": "统一检查识别模型、设备接入方式、业务能力和验收项是否可交付。",
            "customer_count_label": "资源",
            "customer_action": "替换未注册或被阻断的交付资源。",
            "status": resource_summary.get("overall_status", "unknown"),
            "count": resource_summary.get("resource_count", 0),
            "api": "/api/field/customer-project-resource-catalog",
        },
        {
            "surface_id": "package_delivery_gate",
            "label": "交付包准入",
            "customer_label": "交付包准入",
            "customer_description": "导出、导入前统一检查范围、对象、资源、证据和验收风险。",
            "customer_count_label": "项目",
            "customer_action": "通过预检后再导出或导入客户项目包。",
            "status": readiness.get("overall_status", "unknown"),
            "count": readiness_summary.get("project_count", 0),
            "api": "/api/field/customer-projects/{identifier}/export",
        },
    ]


def build_customer_project_workbench_payload(
    *,
    project_catalog: dict[str, Any],
    template_catalog: dict[str, Any],
    resource_catalog: dict[str, Any],
    object_summary: dict[str, Any],
    object_rows: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    readiness: dict[str, Any],
    scope_filtered: bool,
    blueprints_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the customer-facing workbench response from prepared catalogs."""
    runtime_blueprint_binding = customer_project_runtime_blueprint_binding(
        projects=projects,
        object_rows=object_rows,
        blueprints_payload=blueprints_payload,
    )
    delivery_surfaces = customer_project_delivery_surfaces(
        project_catalog=project_catalog,
        template_catalog=template_catalog,
        resource_catalog=resource_catalog,
        object_summary=object_summary,
        readiness=readiness,
    )
    return {
        "workbench_type": "askme.solution_provider_customer_project_workbench.v1",
        "overall_status": readiness.get("overall_status", "unknown"),
        "customer_status": readiness.get("customer_status", ""),
        "release_claim": readiness.get("release_claim", ""),
        "next_step": readiness.get("next_step", ""),
        "scope_filtered": scope_filtered,
        "filters": project_catalog.get("filters") or {},
        "delivery_surfaces": delivery_surfaces,
        "delivery_chain": _customer_project_delivery_chain(
            delivery_surfaces=delivery_surfaces,
            runtime_blueprint_binding=runtime_blueprint_binding,
            object_summary=object_summary,
            resource_catalog=resource_catalog,
            readiness=readiness,
        ),
        "customer_vocabulary": customer_project_term_cards(),
        "customer_acceptance_flow": _CUSTOMER_PROJECT_ACCEPTANCE_FLOW,
        "runtime_blueprint_binding": runtime_blueprint_binding,
        "customer_readable_contract": {
            "contract_type": "askme.solution_provider_customer_delivery_contract.v1",
            "positioning": "面向多客户、多行业现场的可复用机器人方案交付平台。",
            "customer_can_verify": [
                "客户项目是否按客户范围隔离",
                "行业模板是否能复制成新项目",
                "对象目录是否覆盖客户购买的现场对象",
                "每个对象是否绑定识别能力、设备接入方式、业务能力和验收项",
                "交付包导出、预检、导入、报告是否完整",
            ],
            "not_claimed": [
                "不把演示或试点状态承诺为无人值守生产上线",
                "不让客户阅读接口、YAML 或测试路径来理解验收结论",
                "不绕过现场验收、权限隔离和交付包准入检查",
            ],
        },
        "solution_delivery_readiness": readiness,
        "customer_projects": {
            "summary": project_catalog.get("summary") or {},
            "project_count": len(projects),
            "projects": projects[:20],
        },
        "template_market": {
            "summary": template_catalog.get("summary") or {},
            "templates": (
                template_catalog.get("templates", [])
                if isinstance(template_catalog.get("templates"), list)
                else []
            )[:20],
        },
        "managed_object_directory": {
            "summary": object_summary,
            "objects": object_rows[:50],
        },
        "delivery_resources": {
            "summary": resource_catalog.get("summary") or {},
            "resources": (
                resource_catalog.get("resources", [])
                if isinstance(resource_catalog.get("resources"), list)
                else []
            )[:50],
        },
    }


def _customer_project_delivery_chain(
    *,
    delivery_surfaces: list[dict[str, Any]],
    runtime_blueprint_binding: dict[str, Any],
    object_summary: dict[str, Any],
    resource_catalog: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    surfaces = {
        str(item.get("surface_id") or ""): item
        for item in delivery_surfaces
        if isinstance(item, dict)
    }
    resource_summary = _mapping(resource_catalog.get("summary"))
    runtime_summary = _mapping(runtime_blueprint_binding.get("summary"))
    steps = [
        _delivery_chain_step(
            step_id="project_scope",
            label="客户项目范围",
            status=str(surfaces.get("customer_projects", {}).get("status") or "unknown"),
            customer_question="这个页面现在看的是哪个客户、哪个项目、哪个现场？",
            evidence=(
                f"项目 {surfaces.get('customer_projects', {}).get('count', 0)} 个；"
                f"范围过滤={bool(readiness.get('scope_filtered'))}"
            ),
            next_step=str(surfaces.get("customer_projects", {}).get("customer_action") or ""),
            endpoint=str(surfaces.get("customer_projects", {}).get("api") or "/api/field/customer-projects"),
            source_surface_id="customer_projects",
        ),
        _delivery_chain_step(
            step_id="template_market",
            label="行业模板",
            status=str(surfaces.get("template_market", {}).get("status") or "unknown"),
            customer_question="这个客户项目是从哪个行业模板复制出来的？",
            evidence=f"模板 {surfaces.get('template_market', {}).get('count', 0)} 个",
            next_step=str(surfaces.get("template_market", {}).get("customer_action") or ""),
            endpoint=str(surfaces.get("template_market", {}).get("api") or "/api/field/customer-project-templates"),
            source_surface_id="template_market",
        ),
        _delivery_chain_step(
            step_id="managed_object_directory",
            label="现场对象目录",
            status=str(
                object_summary.get("overall_status")
                or surfaces.get("managed_objects", {}).get("status")
                or "unknown"
            ),
            customer_question="本项目覆盖哪些车辆、设备、垃圾桶、烟感、服务点或通道？",
            evidence=(
                f"对象 {object_summary.get('object_count', surfaces.get('managed_objects', {}).get('count', 0))} 个；"
                f"阻塞 {object_summary.get('blocked_count', 0)} 个"
            ),
            next_step=str(surfaces.get("managed_objects", {}).get("customer_action") or ""),
            endpoint=str(
                surfaces.get("managed_objects", {}).get("api")
                or "/api/field/customer-projects/managed-object-directory"
            ),
            source_surface_id="managed_objects",
        ),
        _delivery_chain_step(
            step_id="capability_resource_binding",
            label="能力和资源绑定",
            status=str(
                resource_summary.get("overall_status")
                or surfaces.get("delivery_resources", {}).get("status")
                or "unknown"
            ),
            customer_question="每个对象是否绑定了视觉模型、传感器协议、技能包和验收用例？",
            evidence=(
                f"资源 {resource_summary.get('resource_count', surfaces.get('delivery_resources', {}).get('count', 0))} 个；"
                f"未注册 {resource_summary.get('unregistered_count', 0)} 个"
            ),
            next_step=str(surfaces.get("delivery_resources", {}).get("customer_action") or ""),
            endpoint=str(
                surfaces.get("delivery_resources", {}).get("api")
                or "/api/field/customer-project-resource-catalog"
            ),
            source_surface_id="delivery_resources",
        ),
        _delivery_chain_step(
            step_id="runtime_blueprint",
            label="运行蓝图",
            status=str(runtime_blueprint_binding.get("overall_status") or "unknown"),
            customer_question="这个客户项目将由哪套机器人运行方案承载？",
            evidence=(
                f"项目 {runtime_summary.get('project_count', 0)} 个；"
                f"已绑定 {runtime_summary.get('ready_project_count', 0)} 个；"
                f"客户可见蓝图 {runtime_summary.get('available_customer_blueprint_count', 0)} 个"
            ),
            next_step=str(runtime_blueprint_binding.get("next_step") or ""),
            endpoint="/api/blueprints",
            source_surface_id="runtime_blueprint_binding",
        ),
        _delivery_chain_step(
            step_id="acceptance_package",
            label="验收和交付包",
            status=str(
                readiness.get("overall_status")
                or surfaces.get("package_delivery_gate", {}).get("status")
                or "unknown"
            ),
            customer_question="客户能否按证据包判断这个版本可以验收？",
            evidence=str(
                readiness.get("customer_status")
                or surfaces.get("package_delivery_gate", {}).get("customer_description")
                or ""
            ),
            next_step=str(
                readiness.get("next_step")
                or surfaces.get("package_delivery_gate", {}).get("customer_action")
                or ""
            ),
            endpoint=str(
                surfaces.get("package_delivery_gate", {}).get("api")
                or "/api/field/customer-projects/{identifier}/export"
            ),
            source_surface_id="package_delivery_gate",
        ),
    ]
    summary = _delivery_chain_summary(steps)
    return {
        "chain_type": "askme.customer_project.delivery_chain.v1",
        "overall_status": summary["overall_status"],
        "step_count": len(steps),
        "summary": summary,
        "steps": steps,
        "policy": {
            "runtime_blueprint_is_required_before_customer_claim": True,
            "capability_resources_must_be_bound_to_managed_objects": True,
            "acceptance_package_must_reference_real_evidence": True,
        },
    }


def _delivery_chain_step(
    *,
    step_id: str,
    label: str,
    status: str,
    customer_question: str,
    evidence: str,
    next_step: str,
    endpoint: str,
    source_surface_id: str,
) -> dict[str, str]:
    normalized_status = _delivery_chain_status(status)
    return {
        "step_id": step_id,
        "label": label,
        "status": normalized_status,
        "customer_question": customer_question,
        "evidence": evidence,
        "next_step": next_step,
        "endpoint": endpoint,
        "source_surface_id": source_surface_id,
    }


def _delivery_chain_status(status: str) -> str:
    text = str(status or "unknown")
    if text in {"ready", "passed", "production_acceptance_ready", "accepted_by_customer"}:
        return "ready"
    if text in {"blocked", "failed", "missing", "file_missing", "invalid"}:
        return "blocked"
    if text in {"manual_check", "needs_review", "ready_for_site_validation", "trial_or_demo_only"}:
        return "manual_check"
    return text or "unknown"


def _delivery_chain_summary(steps: list[dict[str, str]]) -> dict[str, Any]:
    ready_count = sum(1 for item in steps if item.get("status") == "ready")
    manual_count = sum(1 for item in steps if item.get("status") == "manual_check")
    blocked_count = sum(1 for item in steps if item.get("status") == "blocked")
    unknown_count = sum(1 for item in steps if item.get("status") == "unknown")
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check"
        if manual_count or unknown_count
        else "ready"
        if steps
        else "blocked"
    )
    first_gap = next(
        (
            item.get("next_step") or item.get("customer_question") or item.get("label")
            for item in steps
            if item.get("status") != "ready"
        ),
        "客户项目交付链路已具备验收条件。",
    )
    return {
        "overall_status": overall_status,
        "ready_count": ready_count,
        "manual_check_count": manual_count,
        "blocked_count": blocked_count,
        "unknown_count": unknown_count,
        "first_gap": first_gap,
    }


def customer_project_runtime_blueprint_binding(
    *,
    projects: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    blueprints_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind customer projects to runtime blueprints in a customer-readable shape."""

    payload = blueprints_payload if isinstance(blueprints_payload, dict) else load_blueprints_payload(None)
    runtime_blueprints = blueprint_runtime_summary(payload)
    blueprint_items = [
        item
        for item in runtime_blueprints.get("items", [])
        if isinstance(item, dict) and item.get("customer_visible")
    ]
    object_rows_by_project = _object_rows_by_project(object_rows)
    project_bindings = [
        _project_runtime_blueprint_binding(project, object_rows_by_project, blueprint_items)
        for project in projects
        if isinstance(project, dict)
    ]
    ready_count = sum(1 for item in project_bindings if item.get("status") == "ready")
    manual_count = sum(1 for item in project_bindings if item.get("status") == "manual_check")
    blocked_count = sum(1 for item in project_bindings if item.get("status") == "blocked")
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check"
        if manual_count
        else "ready"
        if project_bindings
        else "blocked"
    )
    return {
        "binding_type": "askme.customer_project.runtime_blueprint_binding.v1",
        "overall_status": overall_status,
        "summary": {
            "project_count": len(project_bindings),
            "ready_project_count": ready_count,
            "manual_check_project_count": manual_count,
            "blocked_project_count": blocked_count,
            "available_customer_blueprint_count": len(blueprint_items),
        },
        "policy": {
            "customer_project_must_select_runtime_blueprint": True,
            "managed_objects_must_bind_resources_before_customer_claim": True,
            "acceptance_tests_are_project_scope": True,
        },
        "blueprints": blueprint_items,
        "project_bindings": project_bindings,
        "next_step": _runtime_blueprint_binding_next_step(overall_status),
    }


def _project_runtime_blueprint_binding(
    project: dict[str, Any],
    object_rows_by_project: dict[str, list[dict[str, Any]]],
    blueprint_items: list[dict[str, Any]],
) -> dict[str, Any]:
    project_id = str(project.get("project_id") or "")
    rows = object_rows_by_project.get(project_id, [])
    if not rows and len(object_rows_by_project) == 1 and len([item for item in object_rows_by_project.values() if item]) == 1:
        rows = next(iter(object_rows_by_project.values()))

    explicit = _explicit_blueprint_name(project)
    selected = _select_blueprint_for_project(
        project=project,
        rows=rows,
        explicit_blueprint=explicit,
        blueprint_items=blueprint_items,
    )
    blockers: list[str] = []
    manual_checks: list[str] = []
    if not selected:
        blockers.append("没有匹配到可交付运行蓝图。")
    elif explicit and selected.get("name") != explicit:
        manual_checks.append(f"项目配置的运行蓝图 {explicit} 未在客户可见蓝图中找到，已使用推荐蓝图。")

    missing_binding_types = _missing_binding_types(rows)
    if missing_binding_types:
        blockers.append("对象目录还缺少绑定：" + "、".join(missing_binding_types))

    object_statuses = [str(item.get("delivery_status") or item.get("overall_status") or "") for item in rows]
    if any(status == "blocked" for status in object_statuses):
        blockers.append("至少一个现场对象仍处于阻断状态。")
    elif any(status == "manual_check" for status in object_statuses):
        manual_checks.append("至少一个现场对象需要交付复核。")

    status = "blocked" if blockers else "manual_check" if manual_checks else "ready"
    return {
        "project_id": project_id,
        "customer_id": str(project.get("customer_id") or ""),
        "customer_name": str(project.get("customer_name") or ""),
        "industry": str(project.get("industry") or "unspecified"),
        "selected_blueprint": _blueprint_public_binding(selected),
        "match_reason": _blueprint_match_reason(project, rows, explicit, selected),
        "managed_object_count": len(rows),
        "scenario_ids": _project_scenario_ids(project, rows),
        "missing_binding_types": missing_binding_types,
        "status": status,
        "blockers": sorted(set(blockers)),
        "manual_checks": sorted(set(manual_checks)),
        "customer_claim": _runtime_blueprint_project_claim(status, selected),
        "next_step": _runtime_blueprint_project_next_step(status, missing_binding_types),
    }


def _object_rows_by_project(object_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in object_rows:
        if not isinstance(row, dict):
            continue
        project_id = str(row.get("project_id") or "")
        grouped.setdefault(project_id, []).append(row)
    return grouped


def _explicit_blueprint_name(project: dict[str, Any]) -> str:
    for key in (
        "runtime_blueprint",
        "runtime_blueprint_id",
        "blueprint",
        "blueprint_id",
    ):
        value = str(project.get(key) or "").strip()
        if value:
            return value.replace("blueprint.", "")
    runtime = _mapping(project.get("runtime"))
    for key in ("blueprint", "blueprint_id"):
        value = str(runtime.get(key) or "").strip()
        if value:
            return value.replace("blueprint.", "")
    return ""


def _select_blueprint_for_project(
    *,
    project: dict[str, Any],
    rows: list[dict[str, Any]],
    explicit_blueprint: str,
    blueprint_items: list[dict[str, Any]],
) -> dict[str, Any]:
    by_name = {str(item.get("name") or ""): item for item in blueprint_items}
    if explicit_blueprint and explicit_blueprint in by_name:
        return by_name[explicit_blueprint]

    scenario_ids = set(_project_scenario_ids(project, rows))
    industry = str(project.get("industry") or "").lower()
    if industry in {"park", "campus", "scenic", "factory", "warehouse"}:
        if "edge_robot" in by_name:
            return by_name["edge_robot"]
    if scenario_ids and scenario_ids.issubset({"wayfinding_help_point", "visitor_escort"}):
        if "lingtu_voice" in by_name:
            return by_name["lingtu_voice"]
    if any("voice" in str(item.get("primary_loop") or "").lower() for item in blueprint_items):
        return next(
            item
            for item in blueprint_items
            if "voice" in str(item.get("primary_loop") or "").lower()
        )
    return blueprint_items[0] if blueprint_items else {}


def _blueprint_public_binding(item: dict[str, Any]) -> dict[str, Any]:
    if not item:
        return {}
    return {
        "name": str(item.get("name") or ""),
        "title": str(item.get("title") or ""),
        "status": str(item.get("status") or "unknown"),
        "customer_status": str(item.get("customer_status") or ""),
        "product_stage": str(item.get("product_stage") or ""),
        "package_id": str(item.get("package_id") or ""),
        "deployment_targets": _string_list(item.get("deployment_targets")),
        "capabilities": _string_list(item.get("capabilities")),
        "scenarios": _string_list(item.get("scenarios")),
        "missing_config": _string_list(item.get("missing_config")),
        "external_services": _string_list(item.get("external_services")),
        "safety_boundaries": _string_list(item.get("safety_boundaries")),
        "validation_commands": _string_list(item.get("validation_commands")),
        "release_boundary": str(item.get("release_boundary") or ""),
        "acceptance_boundary": str(item.get("acceptance_boundary") or ""),
        "customer_claim": str(item.get("customer_claim") or ""),
        "customer_next_step": str(item.get("customer_next_step") or ""),
        "delivery_actions": _string_list(item.get("delivery_actions")),
    }


def _blueprint_match_reason(
    project: dict[str, Any],
    rows: list[dict[str, Any]],
    explicit: str,
    selected: dict[str, Any],
) -> str:
    if not selected:
        return "没有可用蓝图。"
    if explicit and str(selected.get("name") or "") == explicit:
        return "项目显式配置了该运行蓝图。"
    industry = str(project.get("industry") or "").strip()
    if industry:
        return f"按客户项目行业 {industry} 推荐运行蓝图。"
    if _project_scenario_ids(project, rows):
        return "按对象目录中的场景覆盖推荐运行蓝图。"
    return "使用默认客户可见运行蓝图。"


def _project_scenario_ids(project: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    summary = _mapping(project.get("managed_objects_summary"))
    values.extend(_string_list(summary.get("scenario_ids")))
    for row in rows:
        values.extend(_string_list(row.get("scenario_ids")))
    return _unique_strings(values)


def _missing_binding_types(rows: list[dict[str, Any]]) -> list[str]:
    labels = {
        "vision_models": "识别模型",
        "sensor_protocols": "传感器协议",
        "skill_packages": "能力包",
        "acceptance_tests": "验收用例",
    }
    missing: set[str] = set()
    if not rows:
        return ["对象目录"]
    for row in rows:
        bindings = _mapping(row.get("bindings"))
        for key, label in labels.items():
            if not _string_list(bindings.get(key)):
                missing.add(label)
    return [labels[key] for key in labels if labels[key] in missing]


def _runtime_blueprint_binding_next_step(status: str) -> str:
    if status == "ready":
        return "按项目绑定的运行蓝图启动试点环境，并执行对象级验收用例。"
    if status == "manual_check":
        return "复核项目推荐蓝图、对象目录和资源绑定，再进入客户演示。"
    return "先补齐项目运行蓝图、对象目录、能力包和验收用例绑定。"


def _runtime_blueprint_project_claim(status: str, selected: dict[str, Any]) -> str:
    title = str(selected.get("title") or selected.get("name") or "运行蓝图")
    if status == "ready":
        return f"{title} 已与该客户项目形成可验收绑定。"
    if status == "manual_check":
        return f"{title} 已推荐给该客户项目，但交付前仍需复核。"
    return "该客户项目还不能声明运行蓝图已可交付。"


def _runtime_blueprint_project_next_step(status: str, missing_binding_types: list[str]) -> str:
    if missing_binding_types:
        return "补齐：" + "、".join(missing_binding_types)
    if status == "ready":
        return "启动对应 runtime blueprint，并跑现场验收用例。"
    if status == "manual_check":
        return "确认推荐蓝图是否符合客户项目现场范围。"
    return "选择客户可见运行蓝图并绑定对象目录。"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = [
    "build_customer_project_workbench_payload",
    "customer_project_delivery_surfaces",
    "customer_project_runtime_blueprint_binding",
    "customer_project_term_cards",
]
