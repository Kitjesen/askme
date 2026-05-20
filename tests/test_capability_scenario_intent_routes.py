import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from askme.api.routes.capabilities import register_capability_routes


def _client() -> TestClient:
    app = FastAPI()

    register_capability_routes(
        app,
        capabilities_provider=lambda: {
            "skills": {
                "catalog": [
                    {"name": "lookup_place", "enabled": True},
                    {"name": "answer_wayfinding", "enabled": True},
                    {"name": "detect_illegal_parking", "enabled": True},
                    {"name": "report_malicious_blocking", "enabled": True},
                ]
            }
        },
        blueprints_provider=lambda: {},
        logger=logging.getLogger("tests.capability_scenario_intents"),
    )
    return TestClient(app)


def test_scenario_intent_catalog_exposes_enabled_auditable_rules() -> None:
    response = _client().get("/api/scenario-intents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["summary"]["enabled_rule_count"] >= 3
    by_rule = {item["rule_id"]: item for item in payload["rules"]}
    assert by_rule["robot_malicious_blocking_report"]["enabled"] is True
    assert by_rule["robot_malicious_blocking_report"]["skill_name"] == "report_malicious_blocking"
    assert by_rule["robot_malicious_blocking_report"]["risk_level"] == "dangerous"


def test_scenario_intent_preview_routes_without_executing_skill() -> None:
    response = _client().post(
        "/api/scenario-intents/preview",
        json={"text": "有人恶意挡住机器狗"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is True
    assert payload["decision"]["skill_name"] == "report_malicious_blocking"
    assert payload["decision"]["scenario_id"] == "robot_abnormal_incident"
    assert payload["policy"]["does_not_execute_skill"] is True


def test_scenario_intent_preview_keeps_wayfinding_separate_from_parking_incident() -> None:
    response = _client().post(
        "/api/scenario-intents/preview",
        json={"text": "停车场在哪里？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is True
    assert payload["decision"]["skill_name"] == "lookup_place"
    assert payload["decision"]["scenario_id"] == "wayfinding"


def test_scenario_intent_preview_covers_customer_scenario_matrix_without_execution() -> None:
    available_skills = [
        "lookup_place",
        "answer_wayfinding",
        "escort_visitor",
        "detect_illegal_parking",
        "detect_fire_smoke",
        "inspect_trash_bin",
        "detect_night_intruder",
        "detect_crowd_gathering",
        "patrol_scan",
        "report_stuck",
    ]
    cases = [
        ("咖啡店在哪", "lookup_place", "wayfinding"),
        ("请带路去服务中心", "escort_visitor", "visitor_escort"),
        ("主通道有车违停", "detect_illegal_parking", "illegal_parking"),
        ("三号楼有烟味", "detect_fire_smoke", "fire_or_smoke"),
        ("西门垃圾桶满了", "inspect_trash_bin", "trash_bin_full"),
        ("北侧窗边有陌生人拍照", "detect_night_intruder", "night_stranger_photo"),
        ("中央广场人群聚集", "detect_crowd_gathering", "crowd_gathering"),
        ("请派机器狗去A区北门巡检", "patrol_scan", "urgent_patrol_dispatch"),
        ("机器狗卡住无法运动", "report_stuck", "robot_abnormal_incident"),
    ]

    client = _client()
    for text, skill_name, scenario_id in cases:
        response = client.post(
            "/api/scenario-intents/preview",
            json={"text": text, "available_skills": available_skills},
        )
        assert response.status_code == 200, text
        payload = response.json()
        assert payload["matched"] is True, text
        assert payload["decision"]["skill_name"] == skill_name, text
        assert payload["decision"]["scenario_id"] == scenario_id, text
        assert payload["policy"]["does_not_execute_skill"] is True


def test_scenario_intent_preview_does_not_misroute_visitor_question_to_patrol() -> None:
    response = _client().post(
        "/api/scenario-intents/preview",
        json={
            "text": "咖啡店在哪",
            "available_skills": ["lookup_place", "patrol_scan"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is True
    assert payload["decision"]["skill_name"] == "lookup_place"
    assert payload["decision"]["scenario_id"] == "wayfinding"
