import json
from pathlib import Path

from askme.skills.skill_manager import SkillManager
from askme.tools.builtin_tools import register_builtin_tools
from askme.tools.field_event_tool import FieldEventTriggerTool
from askme.tools.tool_registry import ToolRegistry


def test_field_event_trigger_tool_creates_archived_event(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    tool = FieldEventTriggerTool(
        config={
            "archive_path": str(archive),
            "action_audit": {"enabled": False},
        }
    )

    payload = json.loads(
        tool.execute(
            scenario_id="robot_abnormal_incident",
            location="Zone A east road",
            fault_type="fall_unrecoverable",
            operator_id="operator-1",
            description="robot fell and cannot recover",
        )
    )

    assert payload["accepted"] is True
    assert payload["scenario_id"] == "robot_abnormal_incident"
    assert payload["priority"] == "P0"
    assert payload["notification_group"] == "security"
    assert payload["event_id"]
    assert archive.exists()
    archived = json.loads(archive.read_text(encoding="utf-8").splitlines()[0])
    assert archived["event_id"] == payload["event_id"]
    assert archived["payload"]["fault_type"] == "fall_unrecoverable"


def test_field_event_trigger_tool_rejects_unknown_scenario(tmp_path: Path) -> None:
    tool = FieldEventTriggerTool(config={"archive_path": str(tmp_path / "events.jsonl")})

    payload = json.loads(tool.execute(scenario_id="unknown_scenario", location="Zone A"))

    assert payload["accepted"] is False
    assert payload["status"] == "rejected"
    assert payload["reason"] == "unknown_scenario"


def test_builtin_tools_register_field_event_trigger() -> None:
    registry = ToolRegistry(config={"default_timeout": 1.0})

    register_builtin_tools(registry, production_mode=True)

    assert "field_event_trigger" in registry
    tool = registry.get("field_event_trigger")
    assert tool is not None
    assert tool.safety_level == "dangerous"


def test_field_scenario_skills_are_installed_and_customer_visible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = SkillManager(project_dir=tmp_path)
    manager.load()

    for skill_name in [
        "report_fall_unrecoverable",
        "report_stuck",
        "report_motor_fault",
        "detect_night_intruder",
        "detect_illegal_parking",
        "detect_fire_smoke",
        "inspect_trash_bin",
        "detect_crowd_gathering",
        "offer_wayfinding_help",
        "escort_visitor",
    ]:
        skill = manager.get(skill_name)
        assert skill is not None, skill_name
        assert skill.source == "builtin"
        assert "field_event_trigger" in skill.tools_section

    center = manager.get_capability_center()
    by_skill = {
        skill["skill_name"]: skill
        for group in center["groups"]
        for skill in group["skills"]
    }
    assert by_skill["report_fall_unrecoverable"]["installed"] is True
    assert by_skill["detect_illegal_parking"]["installed"] is True
    assert by_skill["detect_crowd_gathering"]["installed"] is True
    assert by_skill["escort_visitor"]["requires_approval"] is True


def test_crowd_gathering_skill_can_trigger_security_event(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    tool = FieldEventTriggerTool(
        config={
            "archive_path": str(archive),
            "action_audit": {"enabled": False},
        }
    )

    payload = json.loads(
        tool.execute(
            scenario_id="crowd_gathering",
            location="主通道广场",
            operator_id="operator-1",
            description="同一区域超过 5 人停留超过 30 分钟",
            payload={
                "person_count": 8,
                "duration_min": 35,
                "image_path": "artifacts/evidence/crowd.jpg",
            },
        )
    )

    assert payload["accepted"] is True
    assert payload["scenario_id"] == "crowd_gathering"
    assert payload["priority"] == "P1"
    assert payload["notification_group"] == "security"
