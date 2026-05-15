from askme.pipeline.product_launch_readiness import build_product_launch_readiness


def _ready_identity() -> dict:
    return {
        "status": "production_ready",
        "production_ready": True,
        "identity_mode": "enterprise_gateway",
        "identity_provider": "oidc",
        "customer_status": "企业身份已接入",
        "release_claim": "可进入生产准入测试",
    }


def _demo_identity() -> dict:
    return {
        "status": "blocked",
        "production_ready": False,
        "identity_mode": "demo_operator_directory",
        "identity_provider": "local_config",
        "production_binding_required": True,
        "demo_operator_directory": {"allowed_for": ["demo", "lab", "customer_pilot"]},
        "customer_status": "当前只能用于演示、实验室或客户试点",
        "release_claim": "只能承诺演示或试点能力",
        "next_step": "接入企业 IAM/SSO",
    }


def _ready_field() -> dict:
    return {
        "status": "production_ready",
        "blockers": [],
        "warnings": [],
        "delivery_brief": {"customer": "现场链路已具备验收证据"},
    }


def _ready_solution() -> dict:
    return {
        "overall_status": "ready",
        "customer_status": "客户交付材料已就绪",
        "release_claim": "可以声明受控试点交付能力",
        "summary": {"project_count": 1, "template_count": 4, "resource_count": 10},
    }


def _ready_workbench() -> dict:
    return {
        "overall_status": "ready",
        "customer_status": "客户项目工作台已就绪",
        "next_step": "安排现场验收",
        "delivery_surfaces": [
            {"surface_id": "customer_projects", "status": "ready"},
            {"surface_id": "template_market", "status": "ready"},
        ],
        "customer_projects": {"summary": {"project_count": 1}},
        "template_market": {"summary": {"template_count": 4}},
    }


def test_product_launch_readiness_allows_production_when_all_gates_ready() -> None:
    payload = build_product_launch_readiness(
        identity_readiness=_ready_identity(),
        field_readiness=_ready_field(),
        solution_delivery_readiness=_ready_solution(),
        customer_project_workbench=_ready_workbench(),
    )

    assert payload["overall_status"] == "ready"
    assert payload["launch_stage"] == "production_acceptance_ready"
    assert payload["production_ready"] is True
    assert payload["summary"]["blocked_count"] == 0


def test_product_launch_readiness_demo_identity_blocks_only_production_claim() -> None:
    payload = build_product_launch_readiness(
        identity_readiness=_demo_identity(),
        field_readiness=_ready_field(),
        solution_delivery_readiness=_ready_solution(),
        customer_project_workbench=_ready_workbench(),
    )

    assert payload["overall_status"] == "manual_check"
    assert payload["launch_stage"] == "pilot_or_site_trial"
    assert payload["production_ready"] is False
    assert "不能承诺无人值守生产上线" in payload["customer_status"]
    assert {
        "identity_gateway",
        "field_operations",
        "solution_delivery",
        "customer_project_workbench",
    } <= {gate["gate_id"] for gate in payload["gates"]}


def test_product_launch_readiness_blocks_on_field_blocker() -> None:
    field = _ready_field()
    field["status"] = "blocked"
    field["blockers"] = ["missing real DingTalk notification smoke"]

    payload = build_product_launch_readiness(
        identity_readiness=_ready_identity(),
        field_readiness=field,
        solution_delivery_readiness=_ready_solution(),
        customer_project_workbench=_ready_workbench(),
    )

    assert payload["overall_status"] == "blocked"
    assert payload["launch_stage"] == "demo_or_integration_only"
    assert payload["blockers"][0] == "missing real DingTalk notification smoke"
    assert payload["production_ready"] is False
