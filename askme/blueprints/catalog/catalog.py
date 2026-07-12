"""Product-facing catalog logic for askme runtime blueprints.

Blueprint files assemble runtime modules. Static blueprint metadata lives in
``askme.blueprints.catalog.data``; this module loads, inspects, checks, and
formats that data for CLI, API, Dashboard, QA, and delivery surfaces.
"""

from __future__ import annotations

import importlib
from typing import Any

from askme.blueprints.catalog.data import (
    ALIASES,
    BLUEPRINTS,
)
from askme.blueprints.catalog.models import BlueprintSpec
from askme.runtime.core.module import Runtime


def list_blueprints(*, customer_visible: bool | None = None) -> list[BlueprintSpec]:
    """Return blueprint specs, optionally filtered by customer visibility."""
    items = list(BLUEPRINTS)
    if customer_visible is not None:
        items = [item for item in items if item.customer_visible is customer_visible]
    return items


def get_blueprint_spec(name: str) -> BlueprintSpec:
    """Return one blueprint spec by name or alias."""
    key = ALIASES.get(str(name or "").strip(), str(name or "").strip())
    for spec in BLUEPRINTS:
        if spec.name == key:
            return spec
    raise KeyError(f"unknown blueprint: {name}")


def load_blueprint_runtime(name: str) -> Runtime:
    """Import and return the Runtime object for a blueprint."""
    spec = get_blueprint_spec(name)
    module = importlib.import_module(spec.import_path)
    runtime = getattr(module, spec.object_name)
    if not isinstance(runtime, Runtime):
        raise TypeError(f"{spec.import_path}.{spec.object_name} is not a Runtime")
    return runtime


def resolve_runtime_blueprint_for_modes(*, voice_mode: bool, robot_mode: bool) -> str:
    """Resolve legacy runtime mode flags to a catalog blueprint name."""
    if voice_mode and robot_mode:
        return "edge_robot"
    if voice_mode:
        return "voice"
    return "text"


def load_runtime_blueprint_for_modes(*, voice_mode: bool, robot_mode: bool) -> Runtime:
    """Load the catalog runtime selected by legacy runtime mode flags."""
    name = resolve_runtime_blueprint_for_modes(
        voice_mode=voice_mode,
        robot_mode=robot_mode,
    )
    return load_blueprint_runtime(name)


def inspect_blueprint(name: str) -> dict[str, Any]:
    """Inspect a blueprint composition without starting services."""
    spec = get_blueprint_spec(name)
    runtime = load_blueprint_runtime(spec.name)
    module_names = runtime.module_names()
    duplicate_modules = _duplicates(module_names)
    missing_declared = [item for item in spec.modules if item not in module_names]
    undeclared_runtime_modules = [item for item in module_names if item not in spec.modules]
    order_findings = _order_findings(module_names)
    valid = not duplicate_modules and not missing_declared and not undeclared_runtime_modules
    return {
        "name": spec.name,
        "title": spec.title,
        "valid": valid,
        "module_count": len(module_names),
        "modules": list(module_names),
        "declared_modules": list(spec.modules),
        "duplicate_modules": duplicate_modules,
        "missing_declared_modules": missing_declared,
        "undeclared_runtime_modules": undeclared_runtime_modules,
        "order_findings": order_findings,
        "startup_command": spec.startup_command,
        "product_stage": spec.product_stage,
        "customer_visible": spec.customer_visible,
        "safety_boundaries": list(spec.safety_boundaries),
    }


def blueprint_readiness(
    name: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return delivery readiness gates for one product runtime blueprint.

    This is intentionally a static preflight. It checks the blueprint contract
    and the current configuration shape; live credentials, hardware, and site
    smoke tests must still be verified by the validation commands.
    """
    spec = get_blueprint_spec(name)
    inspection = inspect_blueprint(spec.name)
    cfg = config if isinstance(config, dict) else {}
    config_evidence = [
        _required_config_evidence(cfg, item, aliases=spec.config_aliases)
        for item in spec.required_config
    ]
    missing_config = [
        item["requirement"] for item in config_evidence if not item["present"]
    ]
    gates = [
        {
            "gate_id": "runtime_composition",
            "status": "pass" if inspection["valid"] else "fail",
            "message": "运行模块符合产品蓝图约定。",
            "evidence": {
                "module_count": inspection["module_count"],
                "missing_declared_modules": inspection["missing_declared_modules"],
                "undeclared_runtime_modules": inspection["undeclared_runtime_modules"],
                "duplicate_modules": inspection["duplicate_modules"],
            },
        },
        {
            "gate_id": "required_config",
            "status": "pass" if not missing_config else "fail",
            "message": (
                "必需配置项已经填写。"
                if not missing_config
                else "蓝图交付前仍缺少必需配置。"
            ),
            "missing": missing_config,
            "required": list(spec.required_config),
            "evidence": config_evidence,
        },
        {
            "gate_id": "external_services",
            "status": "manual_check" if spec.external_services else "pass",
            "message": "外部服务需要完成凭证配置和现场冒烟测试。",
            "services": list(spec.external_services),
        },
        {
            "gate_id": "validation_commands",
            "status": "manual_check" if spec.validation_commands else "fail",
            "message": "声明蓝图可交付前，需要运行这些验证命令并归档结果。",
            "commands": list(spec.validation_commands),
        },
    ]
    runtime_profile = _runtime_profile_evidence(spec, cfg)
    if runtime_profile["applies"]:
        gates.append({
            "gate_id": "runtime_profile",
            "status": runtime_profile["status"],
            "message": runtime_profile["message"],
            "profile": runtime_profile["profile"],
            "allowed_for_site_validation": runtime_profile["allowed_for_site_validation"],
        })
    if not inspection["valid"]:
        status = "blocked"
        claim = "运行模块契约修复前不能交付。"
    elif missing_config:
        status = "configuration_incomplete"
        claim = "可以作为产品包介绍，但补齐必需配置前不能部署。"
    elif runtime_profile["status"] == "fail":
        status = "runtime_profile_not_site_ready"
        claim = runtime_profile["message"]
    else:
        status = "ready_for_validation"
        claim = "可进入实验室或现场验证；生产可用声明仍需要真实服务和硬件证据。"
    return {
        "name": spec.name,
        "status": status,
        "customer_visible": spec.customer_visible,
        "product_stage": spec.product_stage,
        "production_ready": False,
        "customer_claim": claim,
        "runtime_profile": runtime_profile,
        "missing_config": missing_config,
        "config_evidence": config_evidence,
        "external_services": list(spec.external_services),
        "validation_commands": list(spec.validation_commands),
        "gates": gates,
    }


def blueprint_delivery_package(
    name: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a concrete delivery package for one product runtime blueprint.

    The package is the product handoff surface: product can explain it, QA can
    test it, delivery can deploy it, and a customer pilot can be accepted or
    rejected against the listed evidence.
    """
    spec = get_blueprint_spec(name)
    inspection = inspect_blueprint(spec.name)
    readiness = blueprint_readiness(spec.name, config=config)
    config_items = [
        {
            "requirement": item["requirement"],
            "status": "ready" if item["present"] else "missing",
            "matched_path": item.get("matched_path") or "",
            "paths_checked": item.get("paths_checked") or [],
        }
        for item in readiness.get("config_evidence", [])
        if isinstance(item, dict)
    ]
    external_items = [
        {
            "service": service,
            "status": "manual_check",
            "evidence_required": f"{service} 的凭证配置和冒烟测试证据",
        }
        for service in spec.external_services
    ]
    scenario_items = [
        {
            "scenario_id": _scenario_id(spec.name, scenario),
            "customer_scenario": scenario,
            "acceptance": (
                "在选定部署目标中运行该场景，记录操作员可见结果，"
                "并附上审计或事件证据。"
            ),
        }
        for scenario in spec.scenarios
    ]
    package_status = _delivery_package_status(readiness)
    release_boundary = _release_boundary(package_status, spec.product_stage)
    return {
        "package_id": f"blueprint.{spec.name}",
        "blueprint": spec.name,
        "title": spec.title,
        "product_stage": spec.product_stage,
        "customer_visible": spec.customer_visible,
        "status": package_status,
        "customer_status": _delivery_customer_status(package_status),
        "customer_claim": readiness["customer_claim"],
        "release_boundary": release_boundary,
        "acceptance_boundary": release_boundary,
        "customer_next_step": _delivery_customer_next_step(package_status, readiness),
        "delivery_actions": _delivery_actions(package_status, readiness),
        "startup_command": spec.startup_command,
        "deployment_targets": list(spec.deployment_targets),
        "deliverables": {
            "runtime_composition": {
                "status": "ready" if inspection["valid"] else "blocked",
                "module_count": inspection["module_count"],
                "modules": inspection["modules"],
                "findings": {
                    "missing_declared_modules": inspection["missing_declared_modules"],
                    "undeclared_runtime_modules": inspection["undeclared_runtime_modules"],
                    "duplicate_modules": inspection["duplicate_modules"],
                    "order_findings": inspection["order_findings"],
                },
            },
            "configuration_checklist": config_items,
            "external_service_checklist": external_items,
            "scenario_acceptance": scenario_items,
            "validation_commands": list(spec.validation_commands),
            "safety_boundaries": list(spec.safety_boundaries),
        },
        "handoff_steps": _handoff_steps(spec, readiness),
        "stop_conditions": _stop_conditions(readiness),
        "operator_runbook": _operator_runbook(spec),
        "customer_questions_to_answer": _customer_questions_to_answer(spec),
        "notes": list(spec.notes),
    }


def catalog_payload(*, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a serializable catalog for UI, CLI, and documentation surfaces."""
    inspections = [inspect_blueprint(spec.name) for spec in BLUEPRINTS]
    readiness = [blueprint_readiness(spec.name, config=config) for spec in BLUEPRINTS]
    summary = blueprint_configuration_summary(config=config, readiness=readiness)
    return {
        "summary": {
            **summary,
            "valid_count": sum(1 for item in inspections if item["valid"]),
        },
        "items": [
            {
                **spec.to_dict(),
                "inspection": next(item for item in inspections if item["name"] == spec.name),
                "readiness": next(item for item in readiness if item["name"] == spec.name),
                "delivery_package": blueprint_delivery_package(spec.name, config=config),
            }
            for spec in BLUEPRINTS
        ],
    }


def blueprint_configuration_summary(
    *,
    config: dict[str, Any] | None = None,
    readiness: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the small readiness summary used by product UI and CLI surfaces."""
    readiness_items = readiness or [
        blueprint_readiness(spec.name, config=config) for spec in BLUEPRINTS
    ]
    ready_names = [
        str(item["name"])
        for item in readiness_items
        if item["status"] == "ready_for_validation"
    ]
    incomplete = [
        {
            "name": str(item["name"]),
            "missing_config": list(item.get("missing_config") or []),
        }
        for item in readiness_items
        if item["status"] == "configuration_incomplete"
    ]
    return {
        "blueprint_count": len(BLUEPRINTS),
        "customer_visible_count": sum(1 for item in BLUEPRINTS if item.customer_visible),
        "ready_for_validation_count": len(ready_names),
        "configuration_incomplete_count": len(incomplete),
        "ready_for_validation_blueprints": ready_names,
        "configuration_incomplete_blueprints": incomplete,
        "pilot_blueprints": [
            item.name for item in BLUEPRINTS if item.product_stage in {"pilot", "lab"}
        ],
    }


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _order_findings(module_names: tuple[str, ...]) -> list[str]:
    findings: list[str] = []
    if "cognition" in module_names and "runtime_handoff" in module_names:
        if module_names.index("runtime_handoff") < module_names.index("cognition"):
            findings.append("runtime_handoff_before_cognition")
    if "executor" in module_names and "voice" in module_names:
        if module_names.index("voice") < module_names.index("executor"):
            findings.append("voice_before_executor")
    if "executor" in module_names and "text" in module_names:
        if module_names.index("text") < module_names.index("executor"):
            findings.append("text_before_executor")
    return findings


def _config_path_present(config: dict[str, Any], path: str) -> bool:
    if not path:
        return False
    if path.isupper():
        import os

        return _concrete_config_value(os.getenv(path))
    current: Any = config
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
    return _concrete_config_value(current)


def _concrete_config_value(value: Any) -> bool:
    """Return whether a config value is usable as delivery readiness evidence.

    Product readiness cannot be satisfied by a placeholder, an empty service
    map, or a service explicitly disabled for local demos. This keeps runtime
    blueprints from looking customer-ready just because a YAML section exists.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and not (text.startswith("${") and text.endswith("}"))
    if isinstance(value, dict):
        if value.get("enabled") is False:
            return False
        ignored_keys = {"enabled", "operator_id", "robot_id", "site_id", "description", "remark"}
        return any(
            _concrete_config_value(child)
            for key, child in value.items()
            if str(key) not in ignored_keys
        )
    if isinstance(value, (list, tuple, set)):
        return any(_concrete_config_value(item) for item in value)
    return True


def _required_config_evidence(
    config: dict[str, Any],
    requirement: str,
    *,
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    paths = _candidate_config_paths(requirement, aliases=aliases)
    for path in paths:
        if _config_path_present(config, path):
            return {
                "requirement": requirement,
                "present": True,
                "matched_path": path,
                "paths_checked": list(paths),
                "matched_by": "primary" if path == requirement else "alias",
            }
    return {
        "requirement": requirement,
        "present": False,
        "matched_path": "",
        "paths_checked": list(paths),
        "matched_by": "",
    }


def _candidate_config_paths(
    requirement: str,
    *,
    aliases: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in (requirement, *aliases.get(requirement, ())):
        text = str(item or "").strip()
        if text and text not in ordered:
            ordered.append(text)
    return tuple(ordered)


def _delivery_package_status(readiness: dict[str, Any]) -> str:
    status = str(readiness.get("status") or "")
    if status == "ready_for_validation":
        return "ready_for_site_validation"
    if status == "configuration_incomplete":
        return "missing_configuration"
    if status == "runtime_profile_not_site_ready":
        return "demo_or_shadow_only"
    if status == "blocked":
        return "blocked"
    return "needs_review"


def _release_boundary(package_status: str, product_stage: str) -> str:
    if package_status == "ready_for_site_validation":
        return (
            "可用于实验室或客户试点验证。真实凭证、硬件、场景证据和客户验收记录"
            "齐备前，不能声明无人值守生产运行。"
        )
    if package_status == "demo_or_shadow_only":
        return (
            "当前 runtime profile 只支持演示、仿真或影子验证；切换到受控 lab/prod "
            "profile 并完成现场证据前，不能声明客户现场验证通过。"
        )
    if package_status == "missing_configuration":
        return (
            "可以作为产品包销售或介绍，但补齐必需配置前，"
            f"不能按 {product_stage} 阶段部署。"
        )
    return "阻断性的运行时或契约问题解决前不能交付。"


def _delivery_customer_status(package_status: str) -> str:
    if package_status == "ready_for_site_validation":
        return "可进入现场验证"
    if package_status == "demo_or_shadow_only":
        return "仅可演示或影子验证"
    if package_status == "missing_configuration":
        return "运行配置未补齐"
    if package_status == "blocked":
        return "蓝图存在阻断项"
    return "需要交付复核"


def _delivery_customer_next_step(
    package_status: str,
    readiness: dict[str, Any],
) -> str:
    missing = readiness.get("missing_config") or []
    if missing:
        return "补齐运行配置：" + "、".join(str(item) for item in missing)
    if package_status == "demo_or_shadow_only":
        return "切换 runtime_handoff.profile 到 lab 或 prod 后再进入客户现场验证。"
    if package_status == "ready_for_site_validation":
        return "运行现场验证用例，并归档客户可查证据。"
    if package_status == "blocked":
        return "先修复运行组合或安全边界阻断项。"
    return "由交付负责人复核蓝图状态和客户验收边界。"


def _delivery_actions(
    package_status: str,
    readiness: dict[str, Any],
) -> list[str]:
    missing = readiness.get("missing_config") or []
    if missing:
        return [
            "补齐运行配置：" + "、".join(str(item) for item in missing),
            "完成外部服务凭证配置和冒烟测试。",
            "重新生成蓝图交付包并复核验收边界。",
        ]
    if package_status == "demo_or_shadow_only":
        return [
            "将 runtime_handoff.profile 切换为 lab 或 prod。",
            "确认对应 profile 已显式启用并绑定真实机器人运行边界。",
            "重新运行现场验证命令并归档证据。",
        ]
    if package_status == "ready_for_site_validation":
        return [
            "运行现场验证用例。",
            "归档语音、通知、机器人运行和客户复核证据。",
            "签收前复核安全边界和人工接管方案。",
        ]
    if package_status == "blocked":
        return [
            "修复运行组合或安全边界阻断项。",
            "重新运行蓝图 readiness 检查。",
            "确认客户验收场景仍在本蓝图范围内。",
        ]
    return ["复核运行蓝图状态、配置缺口和客户验收边界。"]


def _handoff_steps(spec: BlueprintSpec, readiness: dict[str, Any]) -> list[dict[str, Any]]:
    steps = [
        {
            "step": "select_deployment_target",
            "owner": "delivery",
            "action": f"选择一个部署目标：{', '.join(spec.deployment_targets)}",
        },
        {
            "step": "fill_required_config",
            "owner": "delivery",
            "action": "启动前补齐所有缺失的必需配置。",
            "missing": readiness.get("missing_config") or [],
        },
    ]
    if spec.name == "edge_robot":
        steps.append({
            "step": "generate_site_env_template",
            "owner": "delivery",
            "action": (
                "Run python -m askme runtime field-site-env-template --site-profile "
                "deploy/site-profiles/park-demo.yaml --output .env.site"
            ),
        })
    steps.extend(
        [
            {
                "step": "run_validation_commands",
                "owner": "qa",
                "action": "运行每条验证命令并归档输出。",
                "commands": list(spec.validation_commands),
            },
            {
                "step": "start_runtime",
                "owner": "delivery",
                "action": spec.startup_command,
            },
            {
                "step": "customer_acceptance",
                "owner": "product_and_delivery",
                "action": "演示选定客户场景，并附上审计证据。",
            },
        ]
    )
    return steps


def _stop_conditions(readiness: dict[str, Any]) -> list[str]:
    conditions = [
        "如果运行组合门禁失败，停止交付。",
        "如果物理机器人任务会绕过运行交接或安全预检，停止交付。",
        "如果选定客户场景无法附上验证证据，停止交付。",
    ]
    missing = readiness.get("missing_config") or []
    if missing:
        conditions.append(f"补齐这些必需配置前停止交付：{', '.join(missing)}。")
    return conditions


def _operator_runbook(spec: BlueprintSpec) -> dict[str, Any]:
    return {
        "start": spec.startup_command,
        "health": "启动后打开 Dashboard 健康页或调用 /health。",
        "validate": list(spec.validation_commands),
        "rollback": "停止该运行时，回退到上一个已批准的蓝图或交付包。",
        "audit": "试点或事件关闭后导出统一审计记录。",
    }


def _customer_questions_to_answer(spec: BlueprintSpec) -> list[str]:
    questions = [
        "本次试点要验收哪个客户场景？",
        "本次范围包含哪个部署目标和现场配置？",
        "哪些外部服务已经完成凭证配置和冒烟测试？",
        "哪些审计证据能证明场景已经完成？",
    ]
    if spec.name == "edge_robot":
        questions.append("真实机器人、摄像头、传感器、钉钉和导航网关分别绑定到哪里？")
    return questions


def _scenario_id(blueprint: str, scenario: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in scenario)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{blueprint}.{safe[:48]}"


def _runtime_profile_evidence(spec: BlueprintSpec, config: dict[str, Any]) -> dict[str, Any]:
    """Return product evidence for whether the selected runtime profile can support site validation."""
    allowed = ("lab", "prod")
    applies = _requires_site_runtime_profile(spec)
    profile = str(_config_path_value(config, "runtime_handoff.profile") or "").strip()
    if not applies:
        return {
            "applies": False,
            "status": "not_applicable",
            "profile": profile,
            "allowed_for_site_validation": [],
            "message": "该蓝图不需要物理现场 runtime profile 门禁。",
        }
    if profile in allowed:
        return {
            "applies": True,
            "status": "pass",
            "profile": profile,
            "allowed_for_site_validation": list(allowed),
            "message": f"runtime_handoff.profile={profile} 可进入受控客户现场验证。",
        }
    shown = profile or "missing"
    return {
        "applies": True,
        "status": "fail",
        "profile": profile,
        "allowed_for_site_validation": list(allowed),
        "message": (
            f"runtime_handoff.profile={shown} 只适合本地演示、仿真或影子验证；"
            "客户现场验证需要 lab 或 prod profile。"
        ),
    }


def _requires_site_runtime_profile(spec: BlueprintSpec) -> bool:
    return (
        "robot_control" in spec.required_config
        or "customer_pilot_site" in spec.deployment_targets
    )


def _config_path_value(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current
