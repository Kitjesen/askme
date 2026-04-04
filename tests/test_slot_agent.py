"""Tests for SlotCollectorAgent and slot_utils.slot_present."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.proactive.base import ProactiveContext
from askme.pipeline.proactive.slot_agent import SlotCollectorAgent, _MIN_SLOT_CHARS, _MAX_ATTEMPTS
from askme.pipeline.proactive.slot_utils import slot_present
from askme.skills.skill_model import SkillDefinition, SlotSpec


# ── slot_present ──────────────────────────────────────────────────────────────

class TestSlotPresent:
    def test_navigate_without_pipeline_returns_false(self):
        skill = SkillDefinition(name="navigate")
        assert slot_present(skill, "去仓库A", pipeline=None) is False

    def test_navigate_with_pipeline_extracts_target(self):
        skill = SkillDefinition(name="navigate")
        pipeline = MagicMock()
        pipeline.extract_semantic_target.return_value = "仓库A"
        assert slot_present(skill, "去仓库A", pipeline=pipeline) is True

    def test_navigate_same_text_means_no_slot(self):
        skill = SkillDefinition(name="navigate")
        pipeline = MagicMock()
        pipeline.extract_semantic_target.return_value = "去仓库A"  # same as user text
        assert slot_present(skill, "去仓库A", pipeline=pipeline) is False

    def test_voice_trigger_with_remainder(self):
        skill = SkillDefinition(name="patrol", voice_trigger="开始巡逻")
        # "开始巡逻 仓库区" has a remainder "仓库区" (3 chars ≥ 2)
        with patch("askme.pipeline.proactive.slot_analyst.is_vague", return_value=False):
            result = slot_present(skill, "开始巡逻 仓库区")
        assert result is True

    def test_voice_trigger_without_enough_remainder(self):
        skill = SkillDefinition(name="patrol", voice_trigger="开始巡逻")
        # Only "开始巡逻" — no remainder
        with patch("askme.pipeline.proactive.slot_analyst.is_vague", return_value=False):
            result = slot_present(skill, "开始巡逻")
        assert result is False

    def test_no_trigger_falls_back_to_length(self):
        skill = SkillDefinition(name="my_skill", voice_trigger=None)
        # user_text longer than empty trigger
        assert slot_present(skill, "detailed request text") is True

    def test_short_text_with_no_trigger_returns_false(self):
        skill = SkillDefinition(name="my_skill", voice_trigger="执行任务")
        # user_text shorter than or equal to trigger length with no trigger found
        assert slot_present(skill, "执行") is False


# ── SlotCollectorAgent.should_activate ───────────────────────────────────────

class TestShouldActivate:
    def test_no_required_prompt_does_not_activate(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="x", required_prompt="")
        assert agent.should_activate(skill, "text", ProactiveContext()) is False

    def test_has_required_slots_does_not_activate(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(
            name="x",
            required_prompt="问题？",
            required_slots=[SlotSpec(name="dest")],
        )
        assert agent.should_activate(skill, "text", ProactiveContext()) is False

    def test_activates_when_slot_missing(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="navigate", required_prompt="去哪里？")
        ctx = ProactiveContext(pipeline=None)
        # navigate without pipeline → slot_present=False → activate
        assert agent.should_activate(skill, "导航", ctx) is True

    def test_does_not_activate_when_slot_present(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="navigate", required_prompt="去哪里？")
        pipeline = MagicMock()
        pipeline.extract_semantic_target.return_value = "仓库A"
        ctx = ProactiveContext(pipeline=pipeline)
        assert agent.should_activate(skill, "去仓库A", ctx) is False


# ── SlotCollectorAgent.interact ───────────────────────────────────────────────

class TestInteract:
    @pytest.mark.asyncio
    async def test_collects_slot_on_first_attempt(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="navigate", required_prompt="去哪里？")

        with patch("askme.pipeline.proactive.slot_agent.ask_and_listen",
                   AsyncMock(return_value="仓库A")):
            result = await agent.interact(skill, "导航", MagicMock(), ProactiveContext())

        assert result.proceed is True
        assert "仓库A" in result.enriched_text

    @pytest.mark.asyncio
    async def test_retries_on_short_answer(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="patrol", required_prompt="巡逻哪里？")
        call_count = []

        async def fake_listen(question, audio):
            call_count.append(question)
            if len(call_count) == 1:
                return "嗯"  # too short (1 char < _MIN_SLOT_CHARS=2)
            return "仓库B区"

        with patch("askme.pipeline.proactive.slot_agent.ask_and_listen", side_effect=fake_listen):
            result = await agent.interact(skill, "开始", MagicMock(), ProactiveContext())

        assert len(call_count) == 2
        assert "仓库B区" in result.enriched_text

    @pytest.mark.asyncio
    async def test_proceeds_after_max_attempts_exhausted(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="x", required_prompt="问？")

        with patch("askme.pipeline.proactive.slot_agent.ask_and_listen",
                   AsyncMock(return_value="好")):  # 1 char, always too short
            result = await agent.interact(skill, "original", MagicMock(), ProactiveContext())

        assert result.proceed is True
        # Falls back to original text
        assert result.enriched_text == "original"

    @pytest.mark.asyncio
    async def test_empty_answer_retries(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="x", required_prompt="哪里？")
        answers = [None, "仓库C"]

        with patch("askme.pipeline.proactive.slot_agent.ask_and_listen",
                   AsyncMock(side_effect=answers)):
            result = await agent.interact(skill, "go", MagicMock(), ProactiveContext())

        assert "仓库C" in result.enriched_text

    @pytest.mark.asyncio
    async def test_none_answer_all_attempts_still_proceeds(self):
        agent = SlotCollectorAgent()
        skill = SkillDefinition(name="x", required_prompt="问？")

        with patch("askme.pipeline.proactive.slot_agent.ask_and_listen",
                   AsyncMock(return_value=None)):
            result = await agent.interact(skill, "original", MagicMock(), ProactiveContext())

        assert result.proceed is True


# ── _build_question ───────────────────────────────────────────────────────────

class TestBuildQuestion:
    def test_no_hint_uses_required_prompt(self):
        skill = SkillDefinition(name="x", required_prompt="去哪里？")
        q = SlotCollectorAgent._build_question(skill, "")
        assert q == "去哪里？"

    def test_with_hint_mentions_previous(self):
        skill = SkillDefinition(name="x", required_prompt="去哪里？")
        q = SlotCollectorAgent._build_question(skill, "仓库A")
        assert "仓库A" in q
        assert "去哪里" in q
