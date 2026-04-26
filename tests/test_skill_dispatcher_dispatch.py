"""Integration tests for SkillDispatcher.dispatch() behavior.

Covers:
- Unknown skill_name returns [Error] without polluting mission history
- Known skill uses skill.timeout + 10 as step_timeout (passed to wait_for)
- First dispatch() call creates a mission automatically
- Combined context (mission history + extra_context) is forwarded to the pipeline
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.skill_dispatcher import SkillDispatcher

# ── Helpers ────────────────────────────────────────────────────────


def _make_skill(name: str = "navigate", timeout: int = 30) -> MagicMock:
    skill = MagicMock()
    skill.name = name
    skill.description = f"{name} 技能"
    skill.timeout = timeout
    return skill


def _make_dispatcher(
    *,
    skill: MagicMock | None = None,
    execute_skill_result: str = "技能执行完成",
) -> tuple[SkillDispatcher, MagicMock, MagicMock]:
    """Return (dispatcher, mock_pipeline, mock_skill_manager)."""
    mock_pipeline = MagicMock()
    mock_pipeline.execute_skill = AsyncMock(return_value=execute_skill_result)
    mock_pipeline.process = AsyncMock(return_value="LLM回复")

    mock_skill_manager = MagicMock()
    if skill is not None:
        mock_skill_manager.get.return_value = skill
    else:
        mock_skill_manager.get.return_value = None  # unknown by default
    mock_skill_manager.get_skill_catalog.return_value = "navigate, get_time"
    mock_skill_manager.get_enabled.return_value = [skill] if skill else []

    mock_audio = MagicMock()

    dispatcher = SkillDispatcher(
        pipeline=mock_pipeline,
        skill_manager=mock_skill_manager,
        audio=mock_audio,
    )
    return dispatcher, mock_pipeline, mock_skill_manager


# ── Tests ──────────────────────────────────────────────────────────


class TestDispatchUnknownSkill:
    """dispatch() with an unknown skill_name."""

    async def test_returns_error_string(self):
        dispatcher, _, _ = _make_dispatcher(skill=None)
        result = await dispatcher.dispatch("ghost_skill", "执行它")
        assert "[Error]" in result
        assert "ghost_skill" in result

    async def test_does_not_create_mission(self):
        """Error path must bail out before touching mission state."""
        dispatcher, _, _ = _make_dispatcher(skill=None)
        assert not dispatcher.has_active_mission
        await dispatcher.dispatch("ghost_skill", "执行它")
        # Mission should NOT have been created for an unknown skill
        assert not dispatcher.has_active_mission

    async def test_does_not_call_pipeline(self):
        dispatcher, mock_pipeline, _ = _make_dispatcher(skill=None)
        await dispatcher.dispatch("ghost_skill", "执行它")
        mock_pipeline.execute_skill.assert_not_called()

    async def test_available_skills_mentioned_in_error(self):
        dispatcher, _, mock_skill_manager = _make_dispatcher(skill=None)
        mock_skill_manager.get_skill_catalog.return_value = "navigate, get_time"
        result = await dispatcher.dispatch("ghost_skill", "执行它")
        # The catalog string should appear in the error message
        assert "navigate" in result or "get_time" in result


class TestDispatchTimeout:
    """dispatch() uses skill.timeout + 10 as step_timeout."""

    async def test_step_timeout_is_skill_timeout_plus_ten(self):
        skill = _make_skill(timeout=45)
        dispatcher, _, _ = _make_dispatcher(skill=skill)

        captured_timeouts: list[float] = []

        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, timeout):
            captured_timeouts.append(timeout)
            return await original_wait_for(coro, timeout)

        with patch("askme.pipeline.skill_dispatcher.asyncio.wait_for", patched_wait_for):
            await dispatcher.dispatch("navigate", "去仓库")

        assert len(captured_timeouts) == 1
        assert captured_timeouts[0] == pytest.approx(55.0)  # 45 + 10

    async def test_step_timeout_is_reasonable(self):
        """Sanity: timeout for a 30s skill must be > 30 and < 120."""
        skill = _make_skill(timeout=30)
        dispatcher, _, _ = _make_dispatcher(skill=skill)

        captured_timeouts: list[float] = []
        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, timeout):
            captured_timeouts.append(timeout)
            return await original_wait_for(coro, timeout)

        with patch("askme.pipeline.skill_dispatcher.asyncio.wait_for", patched_wait_for):
            await dispatcher.dispatch("navigate", "去仓库")

        assert captured_timeouts[0] > 30
        assert captured_timeouts[0] < 120

    async def test_step_timeout_reflects_skill_definition(self):
        """Longer skill timeout → proportionally longer step_timeout."""
        skill_short = _make_skill(timeout=10)
        skill_long = _make_skill(timeout=120)

        dispatcher_short, _, _ = _make_dispatcher(skill=skill_short)
        dispatcher_long, _, _ = _make_dispatcher(skill=skill_long)

        captured_short: list[float] = []
        captured_long: list[float] = []
        original_wait_for = asyncio.wait_for

        async def patch_short(coro, timeout):
            captured_short.append(timeout)
            return await original_wait_for(coro, timeout)

        async def patch_long(coro, timeout):
            captured_long.append(timeout)
            return await original_wait_for(coro, timeout)

        with patch("askme.pipeline.skill_dispatcher.asyncio.wait_for", patch_short):
            await dispatcher_short.dispatch("navigate", "短")
        with patch("askme.pipeline.skill_dispatcher.asyncio.wait_for", patch_long):
            await dispatcher_long.dispatch("navigate", "长")

        assert captured_short[0] == pytest.approx(20.0)   # 10 + 10
        assert captured_long[0] == pytest.approx(130.0)   # 120 + 10
        assert captured_long[0] > captured_short[0]


class TestDispatchCreatesMission:
    """dispatch() auto-creates a mission on the first call."""

    async def test_no_mission_before_dispatch(self):
        skill = _make_skill()
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        assert dispatcher.has_active_mission is False
        assert dispatcher.current_mission is None

    async def test_mission_created_after_first_dispatch(self):
        skill = _make_skill()
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库", source="voice")
        assert dispatcher.has_active_mission
        assert dispatcher.current_mission is not None

    async def test_mission_has_correct_source(self):
        skill = _make_skill()
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库", source="text")
        assert dispatcher.current_mission.source == "text"

    async def test_mission_has_one_step_after_first_dispatch(self):
        skill = _make_skill()
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库")
        assert dispatcher.current_mission.step_count == 1

    async def test_second_dispatch_reuses_same_mission(self):
        skill = _make_skill()
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库")
        mission_id_first = dispatcher.current_mission.mission_id
        await dispatcher.dispatch("navigate", "再去一次")
        assert dispatcher.current_mission.mission_id == mission_id_first
        assert dispatcher.current_mission.step_count == 2

    async def test_mission_step_records_skill_name(self):
        skill = _make_skill(name="navigate")
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库")
        step = dispatcher.current_mission.steps[0]
        assert step.skill_name == "navigate"
        assert step.user_text == "去仓库"

    async def test_complete_mission_clears_state(self):
        skill = _make_skill()
        dispatcher, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库")
        assert dispatcher.has_active_mission
        completed = dispatcher.complete_mission()
        assert completed is not None
        assert not dispatcher.has_active_mission


class TestDispatchCombinedContext:
    """dispatch() combines mission history and extra_context before calling pipeline."""

    async def test_first_dispatch_passes_empty_context_when_no_extra(self):
        skill = _make_skill()
        dispatcher, mock_pipeline, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库")
        mock_pipeline.execute_skill.assert_called_once_with("navigate", "去仓库", "", source="voice")

    async def test_extra_context_forwarded_to_pipeline(self):
        skill = _make_skill()
        dispatcher, mock_pipeline, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库", extra_context="紧急任务")
        _, _, combined = mock_pipeline.execute_skill.call_args[0]
        assert "紧急任务" in combined

    async def test_second_step_includes_prior_mission_history(self):
        """On the second dispatch, the pipeline receives the first step's result."""
        skill = _make_skill()
        dispatcher, mock_pipeline, _ = _make_dispatcher(
            skill=skill,
            execute_skill_result="导航已开始",
        )
        await dispatcher.dispatch("navigate", "去仓库")
        # Second call: combined_context should mention prior step
        await dispatcher.dispatch("navigate", "去下一个点")

        second_call_args = mock_pipeline.execute_skill.call_args_list[1]
        _, _, combined = second_call_args[0]
        assert "navigate" in combined
        assert "导航已开始" in combined

    async def test_combined_context_includes_both_history_and_extra(self):
        """Mission history AND extra_context are both present in the combined string."""
        skill = _make_skill()
        dispatcher, mock_pipeline, _ = _make_dispatcher(
            skill=skill,
            execute_skill_result="步骤一结果",
        )
        await dispatcher.dispatch("navigate", "第一步")
        # Second dispatch with extra_context
        await dispatcher.dispatch(
            "navigate", "第二步", extra_context="附加上下文XYZ"
        )
        second_call_args = mock_pipeline.execute_skill.call_args_list[1]
        _, _, combined = second_call_args[0]
        assert "步骤一结果" in combined
        assert "附加上下文XYZ" in combined

    async def test_no_extra_context_second_step_still_has_history(self):
        skill = _make_skill()
        dispatcher, mock_pipeline, _ = _make_dispatcher(
            skill=skill,
            execute_skill_result="已完成第一步",
        )
        await dispatcher.dispatch("navigate", "第一步")
        await dispatcher.dispatch("navigate", "第二步")

        second_call_args = mock_pipeline.execute_skill.call_args_list[1]
        _, _, combined = second_call_args[0]
        # History should be there, but no trailing newline junk
        assert "已完成第一步" in combined
