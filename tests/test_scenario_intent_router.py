from askme.robot_interaction import IntentRouter, IntentType, intent_route_payload
from askme.robot_interaction.scenario_intents import classify_scenario_intent

FIELD_SCENARIO_SKILLS = {
    "问路": "lookup_place",
    "怎么走": "answer_wayfinding",
    "带我去": "escort_visitor",
    "发现违停": "detect_illegal_parking",
    "发现烟雾": "detect_fire_smoke",
    "垃圾桶满了": "inspect_trash_bin",
    "夜间陌生人拍照": "detect_night_intruder",
    "摔倒无法恢复": "report_fall_unrecoverable",
    "卡住无法运动": "report_stuck",
    "人为恶意挡路": "report_malicious_blocking",
    "关节电机故障": "report_motor_fault",
    "人群聚集": "detect_crowd_gathering",
    "突发巡检": "patrol_scan",
}


def _router() -> IntentRouter:
    return IntentRouter(voice_triggers=FIELD_SCENARIO_SKILLS)


def test_semantic_scenario_intents_route_common_customer_wording() -> None:
    cases = [
        ("那辆车停在主通道中间了", "detect_illegal_parking", "illegal_parking"),
        ("这边好像冒烟了", "detect_fire_smoke", "fire_or_smoke"),
        ("三号垃圾桶快满了", "inspect_trash_bin", "trash_bin_full"),
        ("窗户旁边有人在拍照", "detect_night_intruder", "night_stranger_photo"),
        ("机器狗倒地起不来了", "report_fall_unrecoverable", "robot_abnormal_incident"),
        ("有人恶意挡住机器狗", "report_malicious_blocking", "robot_abnormal_incident"),
        ("机器狗走不动了", "report_stuck", "robot_abnormal_incident"),
        ("机器狗关节报警", "report_motor_fault", "robot_abnormal_incident"),
        ("厕所在哪？", "lookup_place", "wayfinding"),
        ("去咖啡店怎么走？", "answer_wayfinding", "wayfinding"),
        ("请派机器狗去A区北门巡检", "patrol_scan", "urgent_patrol_dispatch"),
        ("这里人太多了", "detect_crowd_gathering", "crowd_gathering"),
    ]

    router = _router()
    for utterance, skill_name, scenario_id in cases:
        intent = router.route(utterance)
        assert intent.type == IntentType.VOICE_TRIGGER, utterance
        assert intent.skill_name == skill_name
        assert intent.reason in {"voice_trigger", "scenario_intent"}
        if intent.reason == "scenario_intent":
            assert intent.scenario_id == scenario_id
            assert intent.confidence is not None
            assert intent.route_evidence


def test_dangerous_scenario_question_does_not_trigger_action() -> None:
    intent = _router().route("违停事件怎么处理吗")

    assert intent.type == IntentType.GENERAL


def test_wayfinding_question_is_allowed_but_parking_lot_query_is_not_illegal_parking() -> None:
    intent = _router().route("停车场在哪？")

    assert intent.type == IntentType.VOICE_TRIGGER
    assert intent.skill_name == "lookup_place"
    assert intent.scenario_id != "illegal_parking"


def test_visitor_destination_question_does_not_become_urgent_patrol() -> None:
    intent = _router().route("咖啡店在哪？")

    assert intent.type == IntentType.VOICE_TRIGGER
    assert intent.skill_name == "lookup_place"
    assert intent.scenario_id == "wayfinding"


def test_scenario_intent_requires_installed_skill() -> None:
    decision = classify_scenario_intent(
        "那辆车停在主通道中间了",
        available_skills={"lookup_place"},
    )

    assert decision is None


def test_scenario_route_payload_is_auditable() -> None:
    intent = _router().route("这边好像冒烟了")
    payload = intent_route_payload(intent, source="voice")

    assert payload["type"] == "voice_trigger"
    assert payload["reason"] == "scenario_intent"
    assert payload["skill_name"] == "detect_fire_smoke"
    assert payload["scenario_id"] == "fire_or_smoke"
    assert payload["confidence"] >= 0.8
    assert payload["route_evidence"]["rule_id"] == "fire_or_smoke_report"
