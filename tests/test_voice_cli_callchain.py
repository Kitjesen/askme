"""Call-chain tests for voice trigger routing and generated skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from askme.robot_interaction import IntentRouter, IntentType
from askme.skills.skill_manager import SkillManager
from askme.tools.skill_tools import CreateSkillTool


@pytest.fixture()
def skill_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SkillManager:
    import askme.skills.skill_manager as skill_manager_module

    monkeypatch.setattr(skill_manager_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        skill_manager_module,
        "_SETTINGS_FILE",
        tmp_path / "skills_settings.json",
    )
    (tmp_path / "skills").mkdir(exist_ok=True)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    return manager


@pytest.fixture()
def router(skill_manager: SkillManager) -> IntentRouter:
    return IntentRouter(voice_triggers=skill_manager.get_voice_triggers())


def test_builtin_agent_task_has_voice_triggers(skill_manager: SkillManager) -> None:
    triggers = skill_manager.get_voice_triggers()
    agent_task_triggers = [phrase for phrase, skill in triggers.items() if skill == "agent_task"]

    assert len(agent_task_triggers) >= 1


def test_builtin_voice_trigger_routes_to_expected_skill(
    skill_manager: SkillManager,
    router: IntentRouter,
) -> None:
    phrase, skill_name = next(
        (phrase, skill)
        for phrase, skill in skill_manager.get_voice_triggers().items()
        if skill == "agent_task"
    )

    intent = router.route(phrase)

    assert intent.type == IntentType.VOICE_TRIGGER
    assert intent.skill_name == skill_name


def test_unknown_phrase_falls_back_to_general(router: IntentRouter) -> None:
    intent = router.route("this phrase should not match any skill trigger")

    assert intent.type == IntentType.GENERAL


def test_create_skill_writes_disabled_draft_then_approval_enables_trigger(
    skill_manager: SkillManager,
    router: IntentRouter,
) -> None:
    tool = CreateSkillTool()
    tool.set_context(skill_manager, router)
    trigger = "test-battery-trigger-xyz"

    result = tool.execute(
        name="check_battery",
        description="Check Thunder battery and health",
        voice_trigger=trigger,
        prompt="Use robot_api to check battery_percent.",
        tools_section="robot_api",
        tags="robot,sensor",
    )

    assert "[Error]" not in result
    assert "待审批" in result
    assert (skill_manager.generated_skills_dir / "check_battery" / "SKILL.md").exists()
    assert trigger not in router._voice_triggers

    queue = skill_manager.get_generated_skill_governance()
    record = next(item for item in queue["records"] if item["skill_name"] == "check_battery")
    assert record["status"] == "pending_approval"
    assert record["enabled"] is False

    approved = skill_manager.review_generated_skill(
        "check_battery",
        action="approve",
        operator_id="test.operator",
        router=router,
    )

    assert approved["ok"] is True
    assert approved["enabled"] is True
    assert router.route(trigger).type == IntentType.VOICE_TRIGGER
    assert router.route(trigger).skill_name == "check_battery"


def test_rejected_generated_skill_does_not_route(
    skill_manager: SkillManager,
    router: IntentRouter,
) -> None:
    tool = CreateSkillTool()
    tool.set_context(skill_manager, router)
    trigger = "unsafe-generated-trigger"
    tool.execute(
        name="unsafe_generated",
        description="Unsafe generated skill",
        voice_trigger=trigger,
        prompt="Do something unsafe.",
    )

    rejected = skill_manager.review_generated_skill(
        "unsafe_generated",
        action="reject",
        operator_id="safety.operator",
        router=router,
    )

    assert rejected["ok"] is True
    assert rejected["enabled"] is False
    assert router.route(trigger).type != IntentType.VOICE_TRIGGER
