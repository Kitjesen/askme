from __future__ import annotations

from askme.pipeline.field.customer_project_package_rules import (
    _customer_project_package_action_plan,
    _customer_project_package_delivery_gate,
    _customer_project_package_import_gate_result,
)


def test_package_action_plan_blocks_missing_resources_and_acceptance() -> None:
    plan = _customer_project_package_action_plan(
        {
            "objects": [
                {
                    "object_id": "trash-bin-01",
                    "display_name": "1号垃圾桶",
                    "category": "trash_bin",
                    "resource_binding_status": {
                        "checks": [
                            {
                                "status": "missing",
                                "resource_type": "vision_model",
                            }
                        ]
                    },
                    "acceptance_status": {
                        "status": "blocked",
                        "missing": ["sensor_protocol"],
                    },
                }
            ]
        }
    )

    assert plan["overall_status"] == "blocked"
    assert plan["blocked_action_count"] == 2
    assert plan["manual_check_action_count"] == 0
    assert plan["delivery_gate_source_version"]
    assert [item["action"] for item in plan["actions"]] == [
        "bind_required_resource",
        "bind_acceptance_requirement",
    ]


def test_package_delivery_gate_allows_manual_check_but_rejects_blocked() -> None:
    manual_plan = {
        "blocked_action_count": 0,
        "manual_check_action_count": 1,
        "delivery_gate_source_version": "v1",
        "actions": [
            {
                "object_id": "gate-01",
                "reason_code": "resource_manual_check_required",
                "severity": "manual_check",
            }
        ],
    }
    manual_gate = _customer_project_package_delivery_gate(
        action_plan=manual_plan,
        acceptance_summary={"overall_status": "ready"},
        binding_readiness={"overall_status": "ready"},
        reuse_assessment={"status": "ready"},
    )

    assert manual_gate["delivery_gate_status"] == "manual_check_required"
    assert manual_gate["export_allowed"] is True
    assert manual_gate["import_allowed"] is True
    assert manual_gate["customer_handoff_ready"] is False
    assert manual_gate["customer_status"] == "交付包可导入试点项目，但人工检查关闭前不能签收。"
    assert manual_gate["release_claim"] == "该门禁只控制客户交接就绪度；生产上线仍需要现场设备、通知、语音和机器人运行验收。"
    assert manual_gate["next_step"] == "仅导入受控试点命名空间，并在签收前关闭人工检查。"
    assert _customer_project_package_import_gate_result(manual_gate) == "accepted_with_manual_check"

    blocked_gate = _customer_project_package_delivery_gate(
        action_plan={"blocked_action_count": 1, "manual_check_action_count": 0, "actions": []},
        acceptance_summary={"overall_status": "ready"},
        binding_readiness={"overall_status": "ready"},
        reuse_assessment={"status": "ready"},
    )

    assert blocked_gate["delivery_gate_status"] == "blocked"
    assert blocked_gate["export_allowed"] is False
    assert blocked_gate["import_allowed"] is False
    assert blocked_gate["customer_status"] == "交付包存在阻断项，不能作为客户项目交接包导入。"
    assert blocked_gate["next_step"] == "修复阻断的对象绑定或验收证据后重新导出交付包。"
    assert _customer_project_package_import_gate_result(blocked_gate) == "rejected"
