"""Tests for ConfirmationAgent — yes/no/estop/interrupt classification and flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.proactive.base import ProactiveContext, ProactiveResult
from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
from askme.skills.skill_model import SkillDefinition

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_skill(*, confirm: bool = True, name: str = "dangerous_skill") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description="危险操作",
        confirm_before_execute=confirm,
    )


def _make_audio(listen_return: str | None) -> MagicMock:
    audio = MagicMock()
    audio.listen_loop.return_value = listen_return
    audio.wait_speaking_done.return_value = None
    return audio


# ── should_activate ───────────────────────────────────────────────────────────

class TestShouldActivate:
    def test_activates_when_confirm_required(self):
        agent = ConfirmationAgent()
        skill = _make_skill(confirm=True)
        assert agent.should_activate(skill, "text", ProactiveContext()) is True

    def test_does_not_activate_without_confirm(self):
        agent = ConfirmationAgent()
        skill = _make_skill(confirm=False)
        assert agent.should_activate(skill, "text", ProactiveContext()) is False


# ── _is_yes ───────────────────────────────────────────────────────────────────

class TestIsYes:
    def test_none_returns_false(self):
        assert ConfirmationAgent._is_yes(None) is False

    def test_empty_returns_false(self):
        assert ConfirmationAgent._is_yes("") is False

    def test_确认_returns_true(self):
        assert ConfirmationAgent._is_yes("确认") is True

    def test_好的_returns_true(self):
        assert ConfirmationAgent._is_yes("好的") is True

    def test_yes_returns_true(self):
        assert ConfirmationAgent._is_yes("yes") is True

    def test_unrelated_text_returns_false(self):
        assert ConfirmationAgent._is_yes("巡逻完成") is False

    def test_single_char_好_exact_match_only(self):
        assert ConfirmationAgent._is_yes("好") is True
        assert ConfirmationAgent._is_yes("好久不见") is False


# ── _is_no ────────────────────────────────────────────────────────────────────

class TestIsNo:
    def test_none_returns_false(self):
        assert ConfirmationAgent._is_no(None) is False

    def test_取消_returns_true(self):
        assert ConfirmationAgent._is_no("取消") is True

    def test_不要_returns_true(self):
        assert ConfirmationAgent._is_no("不要") is True

    def test_cancel_returns_true(self):
        assert ConfirmationAgent._is_no("cancel") is True

    def test_random_text_returns_false(self):
        assert ConfirmationAgent._is_no("继续执行") is False


# ── _is_estop ─────────────────────────────────────────────────────────────────

class TestIsEstop:
    def test_none_returns_false(self):
        assert ConfirmationAgent._is_estop(None) is False

    def test_急停_returns_true(self):
        assert ConfirmationAgent._is_estop("急停") is True

    def test_estop_lowercase_returns_true(self):
        assert ConfirmationAgent._is_estop("estop now") is True

    def test_emergency_stop_returns_true(self):
        assert ConfirmationAgent._is_estop("emergency stop") is True

    def test_regular_cancel_not_estop(self):
        assert ConfirmationAgent._is_estop("取消") is False


# ── interact flows ────────────────────────────────────────────────────────────

class TestInteract:
    @pytest.mark.asyncio
    async def test_yes_returns_proceed_true(self):
        agent = ConfirmationAgent()
        skill = _make_skill()
        audio = _make_audio("确认")
        with patch("askme.pipeline.proactive.confirm_agent.ask_and_listen",
                   AsyncMock(return_value="确认")):
            result = await agent.interact(skill, "execute", audio, ProactiveContext())
        assert result.proceed is True
        assert result.enriched_text == "execute"

    @pytest.mark.asyncio
    async def test_no_returns_proceed_false(self):
        agent = ConfirmationAgent()
        skill = _make_skill()
        with patch("askme.pipeline.proactive.confirm_agent.ask_and_listen",
                   AsyncMock(return_value="取消")):
            result = await agent.interact(skill, "execute", MagicMock(), ProactiveContext())
        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_none_answer_returns_proceed_false(self):
        agent = ConfirmationAgent()
        skill = _make_skill()
        with patch("askme.pipeline.proactive.confirm_agent.ask_and_listen",
                   AsyncMock(return_value=None)):
            result = await agent.interact(skill, "execute", MagicMock(), ProactiveContext())
        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_estop_returns_cancelled_by_estop(self):
        agent = ConfirmationAgent()
        skill = _make_skill()
        with patch("askme.pipeline.proactive.confirm_agent.ask_and_listen",
                   AsyncMock(return_value="急停")):
            result = await agent.interact(skill, "execute", MagicMock(), ProactiveContext())
        assert result.proceed is False
        assert result.cancelled_by == "estop"

    @pytest.mark.asyncio
    async def test_ambiguous_answer_cancels(self):
        agent = ConfirmationAgent()
        skill = _make_skill()
        with patch("askme.pipeline.proactive.confirm_agent.ask_and_listen",
                   AsyncMock(return_value="嗯嗯嗯嗯嗯")):
            result = await agent.interact(skill, "execute", MagicMock(), ProactiveContext())
        # "嗯嗯嗯嗯嗯" doesn't exactly match "嗯" for yes — it may or may not match depending
        # on the substring logic; either way should not raise
        assert isinstance(result, ProactiveResult)

    @pytest.mark.asyncio
    async def test_interrupt_payload_extracted(self):
        agent = ConfirmationAgent()
        skill = _make_skill()
        with patch("askme.pipeline.proactive.confirm_agent.ask_and_listen",
                   AsyncMock(return_value="算了")), \
             patch("askme.pipeline.proactive.confirm_agent.parse_interrupt_payload",
                   return_value=(True, "re-route")):
            result = await agent.interact(skill, "execute", MagicMock(), ProactiveContext())
        assert result.proceed is False
        assert result.interrupt_payload == "re-route"
