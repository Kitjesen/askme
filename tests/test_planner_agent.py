"""Tests for PlannerAgent — multi-step skill plan generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from askme.pipeline.planner_agent import PlannerAgent


def _make_skill(name: str, description: str = "") -> MagicMock:
    s = MagicMock()
    s.name = name
    s.description = description or name
    s.enabled = True
    return s


def _make_skill_manager(*skill_names: str) -> MagicMock:
    skills = [_make_skill(n) for n in skill_names]
    sm = MagicMock()
    sm.get_enabled.return_value = skills
    sm.get = lambda name: next((s for s in skills if s.name == name), None)
    return sm


def _make_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=response)
    return llm


class TestPlannerAgentMultiStep:
    async def test_two_step_plan_returned(self):
        llm = _make_llm('{"plan": [{"skill": "navigate", "intent": "前往仓库"}, {"skill": "robot_grab", "intent": "抓取货物"}]}')
        sm = _make_skill_manager("navigate", "robot_grab")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        steps = await agent.plan("先去仓库取货")

        assert steps is not None
        assert len(steps) == 2
        assert steps[0].skill_name == "navigate"
        assert steps[0].intent == "前往仓库"
        assert steps[1].skill_name == "robot_grab"

    async def test_three_step_plan(self):
        llm = _make_llm('{"plan": [{"skill": "navigate", "intent": "去A"}, {"skill": "robot_grab", "intent": "抓取"}, {"skill": "navigate", "intent": "去B"}]}')
        sm = _make_skill_manager("navigate", "robot_grab")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        steps = await agent.plan("先去A抓货再送到B")

        assert steps is not None
        assert len(steps) == 3
        assert steps[2].skill_name == "navigate"
        assert steps[2].intent == "去B"

    async def test_max_5_steps_enforced(self):
        import json as _json
        many = [{"skill": "navigate", "intent": f"步骤{i}"} for i in range(8)]
        llm = _make_llm(_json.dumps({"plan": many}))
        sm = _make_skill_manager("navigate")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        steps = await agent.plan("很多步骤的任务")

        # Should be capped at MAX_STEPS=5
        assert steps is not None
        assert len(steps) <= 5


class TestPlannerAgentSingleStep:
    async def test_null_plan_returns_none(self):
        llm = _make_llm('{"plan": null}')
        sm = _make_skill_manager("navigate", "robot_grab")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        result = await agent.plan("去仓库")

        assert result is None

    async def test_single_step_plan_returns_none(self):
        llm = _make_llm('{"plan": [{"skill": "navigate", "intent": "去仓库"}]}')
        sm = _make_skill_manager("navigate")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        # Single step adds no value over normal routing
        result = await agent.plan("去仓库")

        assert result is None

    async def test_empty_skill_catalog_returns_none(self):
        llm = _make_llm('{"plan": []}')
        sm = MagicMock()
        sm.get_enabled.return_value = []
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        result = await agent.plan("帮我完成任务")

        assert result is None
        llm.chat.assert_not_called()  # should short-circuit before calling LLM


class TestPlannerAgentRobustness:
    async def test_unknown_skill_in_plan_skipped(self):
        llm = _make_llm('{"plan": [{"skill": "navigate", "intent": "去仓库"}, {"skill": "unknown_skill", "intent": "做某事"}, {"skill": "robot_grab", "intent": "抓取"}]}')
        sm = _make_skill_manager("navigate", "robot_grab")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        steps = await agent.plan("复杂任务")

        # unknown_skill filtered out, remaining 2 valid steps returned
        assert steps is not None
        assert all(s.skill_name in ("navigate", "robot_grab") for s in steps)
        assert len(steps) == 2

    async def test_llm_error_returns_none(self):
        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("network error"))
        sm = _make_skill_manager("navigate")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        result = await agent.plan("任何任务")

        assert result is None

    async def test_invalid_json_returns_none(self):
        llm = _make_llm("not json at all")
        sm = _make_skill_manager("navigate")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        result = await agent.plan("任何任务")

        assert result is None

    async def test_markdown_fences_stripped(self):
        llm = _make_llm('```json\n{"plan": [{"skill": "navigate", "intent": "去A"}, {"skill": "robot_grab", "intent": "抓取"}]}\n```')
        sm = _make_skill_manager("navigate", "robot_grab")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        steps = await agent.plan("去A抓取")

        assert steps is not None
        assert len(steps) == 2

    async def test_empty_plan_array_returns_none(self):
        llm = _make_llm('{"plan": []}')
        sm = _make_skill_manager("navigate")
        agent = PlannerAgent(llm_client=llm, skill_manager=sm)

        result = await agent.plan("任务")

        assert result is None
