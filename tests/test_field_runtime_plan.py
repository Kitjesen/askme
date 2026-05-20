from askme.api.services.field_runtime_plan import (
    build_field_runtime_plan_from_event,
    field_runtime_area_id,
    field_runtime_delivery_status,
    field_runtime_risk_tier,
    field_runtime_task_type,
)


def test_build_field_runtime_plan_from_event_uses_high_level_contract() -> None:
    event = {
        "event_id": "evt-1",
        "scenario_id": "illegal_parking",
        "priority": "P1",
        "severity": "warning",
        "location": "Main Road",
        "notification_group": "security",
        "payload": {"zone_id": "zone-main-road"},
        "playbook": {"robot_motion_policy": "retreat_to_safe_distance"},
    }

    plan = build_field_runtime_plan_from_event(event, operator_id="operator-1")

    assert plan["plan_id"] == "field-evt-1"
    assert plan["handoff_ready"] is True
    assert plan["operator_id"] == "operator-1"
    assert plan["reference"]["resolved"] == {
        "area_id": "zone-main-road",
        "label": "Main Road",
        "field_event_id": "evt-1",
        "scenario_id": "illegal_parking",
    }
    mission = plan["mission"]["mission"]
    assert mission["mission_type"] == "field_incident_response"
    assert mission["risk_tier"] == "medium"
    assert mission["steps"] == [
        {"target": "zone-main-road", "policy": "retreat_to_safe_distance"}
    ]
    assert mission["field_event"]["notification_group"] == "security"
    assert "Do not execute low-level motor commands" in " ".join(plan["safety_constraints"])


def test_field_runtime_area_id_prefers_explicit_and_slugs_free_text() -> None:
    assert field_runtime_area_id({}, {"help_point_id": "checkpoint-west-gate"}) == (
        "checkpoint-west-gate"
    )
    assert field_runtime_area_id({"location": "Main Road / North"}, {}) == (
        "zone-main-road-north"
    )
    assert field_runtime_area_id({}, {}) == "zone-field-event"


def test_field_runtime_task_type_and_risk_tier() -> None:
    assert field_runtime_task_type("illegal_parking", "observe_then_continue") == (
        "field_incident_response"
    )
    assert field_runtime_task_type("", "") == "status_report"
    assert field_runtime_risk_tier({"priority": "P0"}) == "high"
    assert field_runtime_risk_tier({"severity": "error"}) == "high"
    assert field_runtime_risk_tier({"priority": "P2"}) == "medium"
    assert field_runtime_risk_tier({}) == "low"


def test_field_runtime_delivery_status_maps_runtime_result() -> None:
    assert field_runtime_delivery_status({"accepted": False}, {}) == "rejected"
    assert field_runtime_delivery_status({"state": "queued"}, {}) == "queued"
    assert field_runtime_delivery_status({}, {"current_state": "completed"}) == "completed"
    assert field_runtime_delivery_status({}, {}) == "submitted"
