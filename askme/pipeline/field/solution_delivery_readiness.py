"""Solution-provider delivery readiness gates.

This module is the product-level rollup for customer project, template market,
delivery-resource, and governance readiness. It consumes already built catalog
payloads and intentionally has no dependency on ``field_site_profile``.
"""

from __future__ import annotations

from typing import Any


def build_solution_delivery_readiness(
    *,
    project_catalog: dict[str, Any],
    template_catalog: dict[str, Any],
    resource_catalog: dict[str, Any],
    governance_requests: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one product-facing readiness gate for solution-provider delivery."""

    governance_requests = governance_requests or {}
    gates = [
        _solution_delivery_customer_project_gate(project_catalog),
        _solution_delivery_template_market_gate(template_catalog),
        _solution_delivery_resource_binding_gate(resource_catalog),
        _solution_delivery_resource_governance_gate(governance_requests),
    ]
    overall_status = _delivery_gate_rollup_status(gates)
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
    return {
        "readiness_type": "askme.solution_delivery_readiness",
        "overall_status": overall_status,
        "production_ready": overall_status == "ready",
        "customer_status": {
            "ready": "可进入客户试点验收；仍需按项目执行现场验收和客户签收。",
            "manual_check": "可用于客户方案演示或试点准备，但仍有交付负责人需要复核的项目。",
            "blocked": "不能对客户承诺验收通过；需要先处理阻塞项。",
        }[overall_status],
        "release_claim": {
            "ready": "可以声明具备受控试点交付能力，不能替代现场最终验收。",
            "manual_check": "只能声明演示或试点准备能力，不能声明客户验收完成。",
            "blocked": "不能声明客户可验收或可上线。",
        }[overall_status],
        "gates": gates,
        "summary": {
            "gate_count": len(gates),
            "ready_count": len([gate for gate in gates if gate.get("status") == "ready"]),
            "manual_check_count": len(
                [gate for gate in gates if gate.get("status") == "manual_check"]
            ),
            "blocked_count": len([gate for gate in gates if gate.get("status") == "blocked"]),
            "project_count": int(
                _mapping(project_catalog.get("summary")).get("project_count") or 0
            ),
            "template_count": int(
                _mapping(template_catalog.get("summary")).get("template_count") or 0
            ),
            "resource_count": int(
                _mapping(resource_catalog.get("summary")).get("resource_count") or 0
            ),
        },
        "blockers": blockers,
        "manual_checks": manual_checks,
        "next_step": (
            blockers[0]
            if blockers
            else manual_checks[0]
            if manual_checks
            else "Continue with onsite acceptance evidence, customer review, and signed handoff."
        ),
    }


def _solution_delivery_customer_project_gate(project_catalog: dict[str, Any]) -> dict[str, Any]:
    gate = _mapping(project_catalog.get("delivery_acceptance_gate"))
    project_count = int(
        gate.get("project_count")
        or _mapping(project_catalog.get("summary")).get("project_count")
        or 0
    )
    status = str(gate.get("overall_status") or "blocked")
    if project_count <= 0:
        status = "blocked"
    return {
        "gate_id": "customer_project_acceptance",
        "label": "客户项目验收门禁",
        "status": status,
        "evidence": (
            f"{project_count} project(s), ready={gate.get('ready_count', 0)}, "
            f"manual={gate.get('manual_check_count', 0)}, blocked={gate.get('blocked_count', 0)}"
        ),
        "next_step": str(
            gate.get("next_step")
            or "Create and verify at least one customer project before delivery."
        ),
    }


def _solution_delivery_template_market_gate(template_catalog: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(template_catalog.get("summary"))
    template_count = int(summary.get("template_count") or 0)
    ready = int(summary.get("product_ready_count") or 0)
    manual = int(summary.get("manual_check_count") or 0)
    blocked = int(summary.get("blocked_count") or 0)
    if template_count <= 0 or blocked >= template_count:
        status = "blocked"
    elif manual or blocked or ready <= 0:
        status = "manual_check"
    else:
        status = "ready"
    return {
        "gate_id": "template_market",
        "label": "行业模板市场",
        "status": status,
        "evidence": (
            f"{template_count} template(s), ready={ready}, manual={manual}, blocked={blocked}"
        ),
        "next_step": (
            "Publish or approve at least one reusable template before customer rollout."
            if status == "blocked"
            else "Review pilot/manual template releases before creating customer projects."
            if status == "manual_check"
            else "Use published templates to seed customer projects."
        ),
    }


def _solution_delivery_resource_binding_gate(resource_catalog: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(resource_catalog.get("summary"))
    resource_count = int(summary.get("resource_count") or 0)
    unregistered = int(summary.get("unregistered_resource_count") or 0)
    overall = str(summary.get("overall_status") or "blocked")
    if resource_count <= 0:
        status = "blocked"
    elif unregistered or overall != "ready":
        status = "manual_check"
    else:
        status = "ready"
    return {
        "gate_id": "delivery_resource_bindings",
        "label": "模型/协议/技能/验收资源绑定",
        "status": status,
        "evidence": (
            f"{resource_count} resource(s), consumers={summary.get('consumer_count', 0)}, "
            f"unregistered={unregistered}"
        ),
        "next_step": str(
            resource_catalog.get("next_step")
            or "Register missing delivery resources before project signoff."
        ),
    }


def _solution_delivery_resource_governance_gate(
    governance_requests: dict[str, Any],
) -> dict[str, Any]:
    if governance_requests.get("skipped"):
        return {
            "gate_id": "delivery_resource_governance",
            "label": "共享资源治理队列",
            "status": "manual_check",
            "evidence": str(governance_requests.get("reason") or "governance queue not visible"),
            "next_step": "Use an unrestricted delivery owner to review shared resource governance queue.",
        }
    summary = _mapping(governance_requests.get("summary"))
    pending = int(summary.get("pending_count") or 0)
    due_soon = int(summary.get("due_soon_count") or 0)
    overdue = int(summary.get("overdue_count") or 0)
    if overdue:
        status = "blocked"
    elif pending or due_soon:
        status = "manual_check"
    else:
        status = "ready"
    return {
        "gate_id": "delivery_resource_governance",
        "label": "共享资源治理队列",
        "status": status,
        "evidence": f"pending={pending}, due_soon={due_soon}, overdue={overdue}",
        "next_step": (
            "Escalate overdue resource governance requests before customer signoff."
            if overdue
            else "Review pending shared resource governance requests."
            if pending or due_soon
            else "No open shared-resource governance blockers."
        ),
    }


def _delivery_gate_rollup_status(gates: list[dict[str, Any]]) -> str:
    statuses = {str(gate.get("status") or "blocked") for gate in gates}
    if "blocked" in statuses:
        return "blocked"
    if "manual_check" in statuses:
        return "manual_check"
    return "ready"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "build_solution_delivery_readiness",
]
