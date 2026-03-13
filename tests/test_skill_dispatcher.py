"""Tests for the SkillDispatcher orchestration layer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.skill_dispatcher import MissionContext, SkillDispatcher
from askme.tools.builtin_tools import DispatchSkillTool


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.execute_skill = AsyncMock(return_value="技能执行完成")
    pipeline.process = AsyncMock(return_value="LLM回复")
    return pipeline


@pytest.fixture()
def mock_skill_manager():
    mgr = MagicMock()
    skill = MagicMock()
    skill.name = "navigate"
    skill.description = "导航技能"
    mgr.get.return_value = skill
    mgr.get_enabled.return_value = [skill]
    mgr.get_skill_catalog.return_value = "navigate, get_time"
    return mgr


@pytest.fixture()
def mock_audio():
    return MagicMock()


@pytest.fixture()
def dispatcher(mock_pipeline, mock_skill_manager, mock_audio):
    return SkillDispatcher(
        pipeline=mock_pipeline,
        skill_manager=mock_skill_manager,
        audio=mock_audio,
    )


# ── MissionContext tests ──────────────────────────────────────────


class TestMissionContext:
    def test_new_mission_has_no_steps(self):
        ctx = MissionContext()
        assert ctx.step_count == 0
        assert ctx.history_for_context() == ""

    def test_add_step(self):
        ctx = MissionContext(source="voice")
        ctx.add_step("navigate", "去仓库", "已开始导航")
        assert ctx.step_count == 1
        assert "navigate" in ctx.summary()
        assert "voice" in ctx.summary()

    def test_history_for_context(self):
        ctx = MissionContext()
        ctx.add_step("navigate", "去仓库", "已开始导航")
        ctx.add_step("environment_report", "查温度", "温度28度")
        history = ctx.history_for_context()
        assert "步骤1" in history
        assert "步骤2" in history
        assert "navigate" in history
        assert "environment_report" in history

    def test_mission_id_is_unique(self):
        a = MissionContext()
        b = MissionContext()
        assert a.mission_id != b.mission_id


# ── SkillDispatcher tests ─────────────────────────────────────────


class TestSkillDispatcher:
    async def test_dispatch_creates_mission(self, dispatcher):
        assert not dispatcher.has_active_mission
        await dispatcher.dispatch("navigate", "去仓库", source="voice")
        assert dispatcher.has_active_mission
        assert dispatcher.current_mission.step_count == 1

    async def test_dispatch_tracks_steps(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库")
        await dispatcher.dispatch("environment_report", "查温度")
        mission = dispatcher.current_mission
        assert mission.step_count == 2
        assert mission.steps[0].skill_name == "navigate"
        assert mission.steps[1].skill_name == "environment_report"

    async def test_handle_general_completes_mission(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库")
        assert dispatcher.has_active_mission
        await dispatcher.handle_general("今天天气怎么样")
        assert not dispatcher.has_active_mission

    async def test_complete_mission_returns_context(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库")
        mission = dispatcher.complete_mission()
        assert mission is not None
        assert mission.step_count == 1
        assert not dispatcher.has_active_mission

    async def test_complete_empty_returns_none(self, dispatcher):
        assert dispatcher.complete_mission() is None

    async def test_dispatch_calls_pipeline(self, dispatcher, mock_pipeline):
        await dispatcher.dispatch("navigate", "去仓库")
        mock_pipeline.execute_skill.assert_called_once_with("navigate", "去仓库", "")

    async def test_handle_general_calls_pipeline(self, dispatcher, mock_pipeline):
        await dispatcher.handle_general("你好", memory_task=None)
        mock_pipeline.process.assert_called_once_with("你好", memory_task=None, source="voice")

    async def test_source_tracking(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库", source="text")
        assert dispatcher.current_mission.source == "text"

    async def test_dispatch_with_extra_context(self, dispatcher, mock_pipeline):
        await dispatcher.dispatch(
            "navigate", "去仓库", extra_context="紧急任务"
        )
        mock_pipeline.execute_skill.assert_called_once()

    async def test_skill_catalog_for_prompt(self, dispatcher):
        catalog = dispatcher.get_skill_catalog_for_prompt()
        assert "navigate" in catalog
        assert "导航技能" in catalog


# ── DispatchSkillTool tests ───────────────────────────────────────


class TestDispatchSkillTool:
    def test_no_dispatcher_returns_error(self):
        tool = DispatchSkillTool()
        result = tool.execute(skill_name="navigate")
        assert "[Error]" in result

    def test_empty_skill_name_returns_error(self):
        tool = DispatchSkillTool()
        tool.set_dispatcher(MagicMock())
        result = tool.execute(skill_name="")
        assert "[Error]" in result

    def test_tool_definition_format(self):
        tool = DispatchSkillTool()
        defn = tool.get_definition()
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "dispatch_skill"
        assert "skill_name" in defn["function"]["parameters"]["properties"]

    def test_dispatches_to_dispatcher(self):
        tool = DispatchSkillTool()
        mock_dispatcher = MagicMock()
        mock_dispatcher.execute_skill_sync.return_value = "导航已开始"
        tool.set_dispatcher(mock_dispatcher)
        result = tool.execute(skill_name="navigate", reason="用户想去仓库")
        assert result == "导航已开始"
        mock_dispatcher.execute_skill_sync.assert_called_once_with(
            "navigate", "用户想去仓库"
        )

    def test_nonexistent_skill(self):
        tool = DispatchSkillTool()
        mock_dispatcher = MagicMock()
        mock_dispatcher.execute_skill_sync.return_value = "[Error] 技能不存在: foo"
        tool.set_dispatcher(mock_dispatcher)
        result = tool.execute(skill_name="foo")
        assert "技能不存在" in result
