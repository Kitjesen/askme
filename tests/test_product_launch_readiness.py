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
    snapshot = payload["customer_acceptance_snapshot"]
    assert snapshot["snapshot_type"] == "askme.customer_project_acceptance_snapshot.v1"
    assert snapshot["overall_status"] == "ready"
    assert snapshot["production_ready"] is True
    assert snapshot["metrics"]["gate_count"] == 4
    assert snapshot["metrics"]["ready_gate_count"] == 4
    assert snapshot["metrics"]["project_count"] == 1
    assert snapshot["primary_gap"]
    assert "identity_gateway" in {source["source_id"] for source in snapshot["evidence_sources"]}
    assert snapshot["customer_can_verify"]
    assert snapshot["not_claimed"]


def test_product_launch_readiness_includes_dashboard_page_gate_when_available() -> None:
    payload = build_product_launch_readiness(
        identity_readiness=_ready_identity(),
        field_readiness=_ready_field(),
        solution_delivery_readiness=_ready_solution(),
        customer_project_workbench=_ready_workbench(),
        dashboard_pages={
            "summary": {
                "page_count": 11,
                "internal_page_count": 0,
                "primary_endpoint_missing_count": 0,
                "primary_endpoint_internal_count": 0,
                "primary_endpoint_unclassified_count": 0,
            },
            "policy": {
                "internal_runtime_is_not_a_customer_page": True,
                "dashboard_shell_uses_registered_pages": True,
                "new_pages_must_have_audience_section_and_primary_endpoint": True,
                "primary_endpoints_must_exist_in_route_inventory": True,
                "customer_pages_must_not_point_to_internal_or_unclassified_routes": True,
            },
        },
    )

    assert payload["overall_status"] == "ready"
    assert "dashboard_pages" in {gate["gate_id"] for gate in payload["gates"]}
    assert payload["summary"]["gate_count"] == 5
    assert payload["summary"]["dashboard_page_count"] == 11
    assert payload["summary"]["dashboard_endpoint_missing_count"] == 0
    assert "/api/dashboard/pages" in {source["endpoint"] for source in payload["evidence_sources"]}
    snapshot = payload["customer_acceptance_snapshot"]
    assert snapshot["metrics"]["gate_count"] == 5
    assert snapshot["metrics"]["dashboard_endpoint_missing_count"] == 0
    assert "dashboard_pages" in {source["source_id"] for source in snapshot["evidence_sources"]}


def test_product_launch_readiness_blocks_when_customer_page_endpoint_is_missing() -> None:
    payload = build_product_launch_readiness(
        identity_readiness=_ready_identity(),
        field_readiness=_ready_field(),
        solution_delivery_readiness=_ready_solution(),
        customer_project_workbench=_ready_workbench(),
        dashboard_pages={
            "summary": {
                "page_count": 11,
                "internal_page_count": 0,
                "primary_endpoint_missing_count": 1,
                "primary_endpoint_internal_count": 0,
                "primary_endpoint_unclassified_count": 0,
            },
            "policy": {
                "internal_runtime_is_not_a_customer_page": True,
                "dashboard_shell_uses_registered_pages": True,
                "new_pages_must_have_audience_section_and_primary_endpoint": True,
                "primary_endpoints_must_exist_in_route_inventory": False,
                "customer_pages_must_not_point_to_internal_or_unclassified_routes": True,
            },
        },
    )

    dashboard_gate = next(gate for gate in payload["gates"] if gate["gate_id"] == "dashboard_pages")
    assert payload["overall_status"] == "blocked"
    assert payload["production_ready"] is False
    assert dashboard_gate["status"] == "blocked"
    assert "Dashboard" in dashboard_gate["next_step"]
    assert payload["summary"]["dashboard_endpoint_missing_count"] == 1
    assert payload["customer_acceptance_snapshot"]["metrics"]["dashboard_endpoint_missing_count"] == 1
    assert payload["customer_acceptance_snapshot"]["primary_gap"] == dashboard_gate["next_step"]


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
