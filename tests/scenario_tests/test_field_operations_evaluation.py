from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval.evaluate_field_operations_scenarios import evaluate_scenarios, write_report


@pytest.mark.asyncio
async def test_field_operations_scenario_evaluation_suite_passes(tmp_path: Path) -> None:
    payload = await evaluate_scenarios()
    report = write_report(payload, tmp_path / "scenario-evaluation.json")

    names = {item["name"] for item in payload["scenarios"]}
    assert payload["status"] == "passed"
    assert payload["external_services"] is False
    assert payload["hardware_dispatch"] is False
    assert payload["failed"] == 0
    assert report.exists()
    assert {
        "robot_immobilized_notifies_security",
        "night_stranger_photo_archived",
        "illegal_parking_camera_ingest",
        "fire_sensor_notifies_security",
        "trash_bin_full_notifies_cleaning",
        "urgent_patrol_dispatch_notifies_operations",
        "crowd_gathering_records_security_event",
        "wayfinding_help_point_does_not_notify_security",
        "visitor_escort_is_archived_without_alert",
        "notification_smoke_test_reports_delivery",
    }.issubset(names)

    product_demo = payload["product_demo"]
    customer_names = {item["customer_name"] for item in product_demo["customer_scenarios"]}
    assert product_demo["suite_name"] == "园区机器狗场景演示包"
    assert product_demo["demo_ready"] is True
    assert product_demo["real_integration_ready"] is False
    assert product_demo["passed"] == payload["scenario_count"]
    assert {
        "机器人摔倒/卡住无法恢复",
        "夜间陌生人拍照",
        "车辆违停检测",
        "火灾及烟雾监测",
        "垃圾桶满溢监测",
        "突发任务巡检",
        "人群聚集检测",
        "路人指路",
        "路人带路",
    }.issubset(customer_names)
    assert any("真实摄像头" in item for item in product_demo["blocked_on_real_integrations"])
    assert any("生产钉钉" in item for item in product_demo["blocked_on_real_integrations"])
    assert any("真实机器狗 runtime" in item for item in product_demo["blocked_on_real_integrations"])
    assert any("MiniMax" in item for item in product_demo["blocked_on_real_integrations"])

    parking = next(
        item
        for item in product_demo["customer_scenarios"]
        if item["customer_name"] == "车辆违停检测"
    )
    assert parking["actual"]["notification_group"] == "security"
    assert parking["actual"]["delivery_status"] == "sent"
    assert parking["actual"]["archive_required"] is True
    assert parking["evidence"]["event_id"].startswith("field-")

    wayfinding = next(
        item
        for item in product_demo["customer_scenarios"]
        if item["customer_name"] == "路人指路"
    )
    assert wayfinding["actual"]["notification_group"] == "none"
    assert wayfinding["expected_robot_action"].endswith("不误触发巡检任务。")
