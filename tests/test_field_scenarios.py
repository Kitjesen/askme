"""Tests for the product field-scenario registry."""

from __future__ import annotations

from askme.pipeline.field_scenarios import FIELD_SCENARIOS, get_field_scenario


def test_required_customer_scenarios_are_registered():
    required = {
        "robot_abnormal_incident",
        "night_stranger_photo",
        "illegal_parking",
        "fire_or_smoke",
        "trash_bin_full",
        "urgent_patrol_dispatch",
        "crowd_gathering",
        "wayfinding_help_point",
        "visitor_escort",
    }
    assert required <= set(FIELD_SCENARIOS)


def test_incident_scenarios_have_evidence_notification_and_archive():
    incident_ids = {
        "robot_abnormal_incident",
        "night_stranger_photo",
        "illegal_parking",
        "fire_or_smoke",
        "trash_bin_full",
        "crowd_gathering",
    }
    for scenario_id in incident_ids:
        scenario = FIELD_SCENARIOS[scenario_id]
        assert scenario.required_evidence
        assert scenario.notification_group in {"security", "cleaning"}
        assert scenario.archive_required is True
        assert scenario.acceptance_criteria


def test_urgent_dispatch_interrupts_and_requires_operator():
    scenario = get_field_scenario("urgent_patrol_dispatch")
    assert scenario is not None
    assert scenario.interrupts_current_task is True
    assert scenario.requires_operator_approval is True


def test_wayfinding_is_service_not_security_alarm():
    scenario = get_field_scenario("wayfinding_help_point")
    assert scenario is not None
    assert scenario.category == "visitor_service"
    assert scenario.notification_group == "none"
    assert scenario.archive_required is False
    assert "未知地名不编路线" in scenario.acceptance_criteria
