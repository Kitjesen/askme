from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.pipeline.skill_gate import SkillGate
from askme.skills.audit import SkillAuditLog
from askme.skills.capability_center import build_capability_center
from askme.skills.skill_manager import SkillManager
from askme.skills.skill_model import SkillDefinition


def test_capability_center_groups_customer_facing_skills() -> None:
    skills = [
        SkillDefinition(
            name="patrol_scan",
            description="scan area",
            safety_level="normal",
            voice_trigger="巡检A区",
            tags=["robot"],
            enabled=True,
        ),
        SkillDefinition(
            name="agent_task",
            description="agent task",
            safety_level="dangerous",
            execution="agent_shell",
            tags=["agent"],
            enabled=True,
        ),
    ]

    payload = build_capability_center(
        skills,
        voice_triggers={"巡检A区": "patrol_scan"},
    )

    assert payload["title"] == "园区巡检机器人能力中心"
    groups = {group["group_id"]: group for group in payload["groups"]}
    patrol = groups["patrol"]
    assert patrol["display_name"] == "巡检任务"
    patrol_scan = next(item for item in patrol["skills"] if item["skill_name"] == "patrol_scan")
    assert patrol_scan["display_name"] == "巡检指定区域"
    assert patrol_scan["status"] == "enabled"
    assert patrol_scan["voice_triggers"] == ["巡检A区"]

    agent = groups["agent"]
    agent_task = next(item for item in agent["skills"] if item["skill_name"] == "agent_task")
    assert agent_task["requires_approval"] is True
    assert payload["online_growth"]["recommended_lifecycle"] == [
        "observe",
        "candidate",
        "draft",
        "review",
        "approve",
        "assign_package",
        "enable",
        "audit",
    ]
    assert payload["scenario_blueprints"]["summary"]["scenario_count"] >= 9


def test_skill_manager_exposes_capability_center() -> None:
    manager = SkillManager()
    manager.load()

    payload = manager.get_capability_center()

    assert payload["summary"]["available_count"] >= 1
    assert any(group["display_name"] == "语音交互" for group in payload["groups"])
    assert payload["summary"]["scenario_count"] >= 9
    scenario_items = {
        item["scenario_id"]: item
        for item in payload["scenario_blueprints"]["items"]
    }
    assert scenario_items["wayfinding_help_point"]["coverage_status"] == "ready"
    assert scenario_items["crowd_gathering"]["coverage_status"] == "ready"
    assert scenario_items["visitor_escort"]["coverage_status"] == "partial"
    assert scenario_items["visitor_escort"]["disabled_skill_names"] == ["navigate"]


def test_capability_center_maps_field_scenarios_to_required_skills() -> None:
    skills = [
        SkillDefinition(
            name="detect_illegal_parking",
            description="detect parking",
            safety_level="dangerous",
            tags=["field", "security", "parking"],
            enabled=True,
        ),
        SkillDefinition(
            name="offer_wayfinding_help",
            description="offer wayfinding",
            tags=["field", "visitor"],
            enabled=True,
        ),
    ]

    payload = build_capability_center(skills)
    items = {
        item["scenario_id"]: item
        for item in payload["scenario_blueprints"]["items"]
    }

    illegal_parking = items["illegal_parking"]
    assert illegal_parking["coverage_status"] == "ready"
    assert illegal_parking["runtime_entry"] == "field_event_trigger"
    assert illegal_parking["required_skills"][0]["skill_name"] == "detect_illegal_parking"
    assert illegal_parking["requires_operator_approval"] is True

    wayfinding = items["wayfinding_help_point"]
    assert wayfinding["coverage_status"] == "partial"
    assert wayfinding["missing_skill_names"] == [
        "lookup_place",
        "recommend_route",
        "answer_wayfinding",
    ]
    assert "InteractionGate" in wayfinding["dependencies"]


def test_skill_audit_log_appends_recent_records(tmp_path: Path) -> None:
    log = SkillAuditLog(tmp_path / "skill-audit.jsonl")

    log.append(skill_name="patrol_scan", status="succeeded", user_text="巡检A区")

    records = log.recent()
    assert records[-1]["skill_name"] == "patrol_scan"
    assert records[-1]["status"] == "succeeded"
    assert records[-1]["user_text_preview"] == "巡检A区"


@pytest.mark.asyncio
async def test_skill_gate_audits_disabled_skill(tmp_path: Path, monkeypatch) -> None:
    audit = SkillAuditLog(tmp_path / "skill-audit.jsonl")
    monkeypatch.setattr("askme.pipeline.skill_gate.SkillAuditLog", lambda: audit)

    skill = SkillDefinition(name="disabled_skill", enabled=False)
    manager = MagicMock()
    manager.get.return_value = skill
    gate = SkillGate(
        skill_manager=manager,
        skill_executor=AsyncMock(),
        audio=MagicMock(),
        conversation=SimpleNamespace(add_user_message=MagicMock(), add_assistant_message=MagicMock()),
    )

    result = await gate.execute_skill("disabled_skill", "执行一下", source="text")

    assert result == "[Skill] Disabled: disabled_skill"
    records = audit.recent()
    assert records[-1]["skill_name"] == "disabled_skill"
    assert records[-1]["status"] == "blocked"
    assert records[-1]["reason"] == "disabled"
