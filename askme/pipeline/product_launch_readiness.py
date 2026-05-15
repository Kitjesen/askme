"""Customer-facing product launch readiness rollup."""

from __future__ import annotations

from typing import Any


def build_product_launch_readiness(
    *,
    identity_readiness: dict[str, Any],
    field_readiness: dict[str, Any],
    solution_delivery_readiness: dict[str, Any],
    customer_project_workbench: dict[str, Any],
) -> dict[str, Any]:
    """Roll up launch gates into one customer-readable delivery decision."""

    gates = [
        _identity_gateway_gate(identity_readiness),
        _field_operations_gate(field_readiness),
        _solution_delivery_gate(solution_delivery_readiness),
        _customer_project_workbench_gate(customer_project_workbench),
    ]
    overall_status = _rollup_status(gates)
    blockers = [
        str(gate.get("next_step") or gate.get("evidence") or "")
        for gate in gates
        if gate.get("status") == "blocked"
    ]
    manual_checks = [
        str(gate.get("next_step") or gate.get("evidence") or "")
        for gate in gates
        if gate.get("status") == "manual_check"
    ]
    production_ready = overall_status == "ready"
    launch_stage = {
        "ready": "production_acceptance_ready",
        "manual_check": "pilot_or_site_trial",
        "blocked": "demo_or_integration_only",
    }[overall_status]
    customer_status = {
        "ready": "可申请客户现场上线验收；上线前仍需客户签收、接管预案和现场值守确认。",
        "manual_check": "可用于客户试点、现场联调或方案演示；不能承诺无人值守生产上线。",
        "blocked": "暂不能进入客户上线验收；只能说明当前处于演示、研发或现场联调阶段。",
    }[overall_status]
    release_claim = {
        "ready": "可以声明具备受控现场上线验收条件，但不替代客户最终签收。",
        "manual_check": "只能声明试点或现场联调能力，不能声明生产上线或无人值守运行。",
        "blocked": "不能声明客户可上线、可验收通过或可无人值守运行。",
    }[overall_status]
    return {
        "readiness_type": "askme.product_launch_readiness.v1",
        "overall_status": overall_status,
        "launch_stage": launch_stage,
        "production_ready": production_ready,
        "customer_status": customer_status,
        "release_claim": release_claim,
        "next_step": (
            blockers[0]
            if blockers
            else manual_checks[0]
            if manual_checks
            else "安排客户现场验收、签署交付单，并确认人工接管和回滚预案。"
        ),
        "gates": gates,
        "summary": {
            "gate_count": len(gates),
            "ready_count": len([gate for gate in gates if gate.get("status") == "ready"]),
            "manual_check_count": len(
                [gate for gate in gates if gate.get("status") == "manual_check"]
            ),
            "blocked_count": len(
                [gate for gate in gates if gate.get("status") == "blocked"]
            ),
            "project_count": _int(
                _mapping(customer_project_workbench.get("customer_projects"))
                .get("summary", {})
                .get("project_count")
            ),
            "template_count": _int(
                _mapping(customer_project_workbench.get("template_market"))
                .get("summary", {})
                .get("template_count")
            ),
            "field_blocker_count": len(field_readiness.get("blockers") or []),
            "identity_blocker_count": len(identity_readiness.get("blockers") or []),
        },
        "blockers": blockers,
        "manual_checks": manual_checks,
        "evidence_sources": [
            {
                "source_id": "identity_gateway",
                "endpoint": "/api/governance/identity-readiness",
                "status": str(identity_readiness.get("status") or "unknown"),
            },
            {
                "source_id": "field_readiness",
                "endpoint": "/api/field/readiness",
                "status": str(field_readiness.get("status") or "unknown"),
            },
            {
                "source_id": "solution_delivery",
                "endpoint": "/api/field/solution-delivery-readiness",
                "status": str(solution_delivery_readiness.get("overall_status") or "unknown"),
            },
            {
                "source_id": "customer_project_workbench",
                "endpoint": "/api/field/customer-project-workbench",
                "status": str(customer_project_workbench.get("overall_status") or "unknown"),
            },
        ],
        "source_snapshots": {
            "identity_gateway": {
                "status": identity_readiness.get("status"),
                "production_ready": bool(identity_readiness.get("production_ready")),
                "customer_status": identity_readiness.get("customer_status"),
                "release_claim": identity_readiness.get("release_claim"),
            },
            "field_readiness": {
                "status": field_readiness.get("status"),
                "stage_code": _mapping(field_readiness.get("delivery_brief")).get("stage_code"),
                "release_scope": _mapping(field_readiness.get("delivery_brief")).get(
                    "release_scope"
                ),
            },
            "solution_delivery": {
                "overall_status": solution_delivery_readiness.get("overall_status"),
                "customer_status": solution_delivery_readiness.get("customer_status"),
                "release_claim": solution_delivery_readiness.get("release_claim"),
            },
            "customer_project_workbench": {
                "overall_status": customer_project_workbench.get("overall_status"),
                "scope_filtered": bool(customer_project_workbench.get("scope_filtered")),
                "surface_count": len(customer_project_workbench.get("delivery_surfaces") or []),
            },
        },
    }


def _identity_gateway_gate(readiness: dict[str, Any]) -> dict[str, Any]:
    if readiness.get("production_ready"):
        status = "ready"
    elif readiness.get("demo_operator_directory") or readiness.get("production_binding_required"):
        status = "manual_check"
    else:
        status = _normalize_gate_status(readiness.get("status"))
    return {
        "gate_id": "identity_gateway",
        "label": "企业身份与租户边界",
        "status": status,
        "evidence": (
            f"{readiness.get('identity_mode') or 'unknown'} / "
            f"{readiness.get('identity_provider') or 'unknown'}"
        ),
        "customer_message": str(readiness.get("customer_status") or ""),
        "next_step": str(
            readiness.get("next_step")
            or "接入企业 IAM/SSO 网关，并用受信 headers 注入 operator、role 和租户/项目作用域。"
        ),
    }


def _field_operations_gate(readiness: dict[str, Any]) -> dict[str, Any]:
    status = _normalize_field_status(readiness.get("status"))
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    warnings = readiness.get("warnings") if isinstance(readiness.get("warnings"), list) else []
    brief = _mapping(readiness.get("delivery_brief"))
    stakeholder_messages = _mapping(brief.get("stakeholder_messages"))
    return {
        "gate_id": "field_operations",
        "label": "现场运行与真实服务",
        "status": status,
        "evidence": (
            f"status={readiness.get('status') or 'unknown'}, "
            f"blockers={len(blockers)}, warnings={len(warnings)}"
        ),
        "customer_message": str(
            stakeholder_messages.get("customer") or brief.get("customer_status") or ""
        ),
        "next_step": str(
            blockers[0]
            if blockers
            else warnings[0]
            if warnings
            else "保持现场运行证据，并完成客户现场验收。"
        ),
    }


def _solution_delivery_gate(readiness: dict[str, Any]) -> dict[str, Any]:
    status = _normalize_gate_status(readiness.get("overall_status"))
    summary = _mapping(readiness.get("summary"))
    return {
        "gate_id": "solution_delivery",
        "label": "方案交付包",
        "status": status,
        "evidence": (
            f"projects={summary.get('project_count', 0)}, "
            f"templates={summary.get('template_count', 0)}, "
            f"resources={summary.get('resource_count', 0)}"
        ),
        "customer_message": str(readiness.get("customer_status") or ""),
        "next_step": str(readiness.get("next_step") or "生成并验证客户交付包和验收材料。"),
    }


def _customer_project_workbench_gate(workbench: dict[str, Any]) -> dict[str, Any]:
    status = _normalize_gate_status(workbench.get("overall_status"))
    surfaces = workbench.get("delivery_surfaces") if isinstance(
        workbench.get("delivery_surfaces"), list
    ) else []
    blocked = len([item for item in surfaces if item.get("status") == "blocked"])
    manual = len([item for item in surfaces if item.get("status") == "manual_check"])
    if blocked:
        status = "blocked"
    elif manual and status == "ready":
        status = "manual_check"
    return {
        "gate_id": "customer_project_workbench",
        "label": "客户项目工作台",
        "status": status,
        "evidence": f"{len(surfaces)} surface(s), blocked={blocked}, manual={manual}",
        "customer_message": str(workbench.get("customer_status") or ""),
        "next_step": str(workbench.get("next_step") or "补齐客户项目、对象目录和交付资源。"),
    }


def _normalize_field_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "production_ready":
        return "ready"
    if text == "ready_for_lab":
        return "manual_check"
    if text in {"ready", "ok", "healthy", "passed"}:
        return "ready"
    if text in {"manual_check", "warning", "degraded", "insufficient_evidence"}:
        return "manual_check"
    return "blocked"


def _normalize_gate_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"ready", "ok", "healthy", "passed", "production_ready"}:
        return "ready"
    if text in {"manual_check", "ready_for_lab", "warning", "degraded"}:
        return "manual_check"
    return "blocked"


def _rollup_status(gates: list[dict[str, Any]]) -> str:
    statuses = {str(gate.get("status") or "blocked") for gate in gates}
    if "blocked" in statuses:
        return "blocked"
    if "manual_check" in statuses:
        return "manual_check"
    return "ready"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
