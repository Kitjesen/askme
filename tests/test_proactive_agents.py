"""Tests for the proactive multi-agent system:
  - SlotCollectorAgent
  - ConfirmationAgent
  - ProactiveOrchestrator

All audio I/O is mocked; asyncio.to_thread calls run the mock synchronously.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.proactive.base import ProactiveContext, ProactiveResult
from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.pipeline.proactive.slot_agent import SlotCollectorAgent
from askme.pipeline.proactive.slot_utils import slot_present
from askme.skills.skill_model import SkillDefinition


# ── Helpers ───────────────────────────────────────────────────────────────────


class MockAudio:
    """Minimal audio stub — records speak() calls, replays canned listen answers."""

    def __init__(self, answers: list[str | None] | None = None) -> None:
        self.spoken: list[str] = []
        self._answers = list(answers or [])
        self._idx = 0

    # --- TTS methods ---
    def drain_buffers(self) -> None: ...
    def speak(self, text: str) -> None:
        self.spoken.append(text)
    def start_playback(self) -> None: ...
    def stop_playback(self) -> None: ...
    def wait_speaking_done(self) -> None: ...  # blocking stub

    # --- ASR method ---
    def listen_loop(self) -> str | None:
        if self._idx < len(self._answers):
            answer = self._answers[self._idx]
            self._idx += 1
            return answer
        return None


class FakePipeline:
    def extract_semantic_target(self, text: str) -> str:
        import re
        for pat in [r"去(.{1,10}?)(?:吧|了|$)", r"导航到(.{1,10}?)(?:去|了|$)"]:
            m = re.search(pat, text)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return text


def _skill(
    name: str,
    *,
    voice_trigger: str = "",
    required_prompt: str = "",
    confirm: bool = False,
    description: str = "测试技能",
) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=description,
        voice_trigger=voice_trigger or None,
        required_prompt=required_prompt,
        confirm_before_execute=confirm,
    )


def _context(
    *,
    pipeline: object = None,
    dispatcher: object = None,
) -> ProactiveContext:
    return ProactiveContext(pipeline=pipeline or FakePipeline(), dispatcher=dispatcher)


# ── slot_utils ───────────────────────────────────────────────────────────────


class TestSlotUtils:
    NAV = _skill("navigate", voice_trigger="导航到,去,带我去")
    WS  = _skill("web_search", voice_trigger="搜索一下,查一下")

    def test_navigate_bare_missing(self):
        assert not slot_present(self.NAV, "导航到", FakePipeline())

    def test_navigate_with_destination(self):
        assert slot_present(self.NAV, "去厨房", FakePipeline())

    def test_web_search_bare_missing(self):
        assert not slot_present(self.WS, "搜索一下")

    def test_web_search_with_query(self):
        assert slot_present(self.WS, "搜索一下明天天气")

    def test_no_trigger_empty(self):
        sk = _skill("no_trigger")
        assert not slot_present(sk, "")


# ── SlotCollectorAgent ────────────────────────────────────────────────────────


class TestSlotCollectorAgent:
    agent = SlotCollectorAgent()

    # -- should_activate --

    def test_no_required_prompt_skip(self):
        sk = _skill("navigate", voice_trigger="导航,去")
        assert not self.agent.should_activate(sk, "去厨房", _context())

    def test_slot_already_present_skip(self):
        sk = _skill("navigate", voice_trigger="去", required_prompt="去哪里？")
        assert not self.agent.should_activate(sk, "去仓库B", _context())

    def test_slot_missing_activates(self):
        sk = _skill("web_search", voice_trigger="搜索一下", required_prompt="搜索什么？")
        assert self.agent.should_activate(sk, "搜索一下", _context())

    # -- interact --

    async def test_successful_slot_collection(self):
        sk = _skill("web_search", voice_trigger="搜索一下", required_prompt="搜索什么内容？")
        audio = MockAudio(["明天北京天气"])
        ctx = _context()
        result = await self.agent.interact(sk, "搜索一下", audio, ctx)
        assert result.proceed is True
        assert "明天北京天气" in result.enriched_text
        assert "搜索什么内容？" in audio.spoken[0]

    async def test_retry_on_short_answer(self):
        """First answer too short → retry with plain required_prompt."""
        sk = _skill("web_search", voice_trigger="搜索一下", required_prompt="搜索什么？")
        audio = MockAudio(["", "穹沛科技"])  # first empty, then valid
        ctx = _context()
        result = await self.agent.interact(sk, "搜索一下", audio, ctx)
        assert result.proceed is True
        assert "穹沛科技" in result.enriched_text
        assert len(audio.spoken) == 2  # asked twice

    async def test_all_attempts_fail_proceeds_anyway(self):
        """After MAX_ATTEMPTS empty answers, proceeds with original text."""
        sk = _skill("web_search", voice_trigger="搜索一下", required_prompt="搜索什么？")
        audio = MockAudio([None, None])
        ctx = _context()
        result = await self.agent.interact(sk, "搜索一下", audio, ctx)
        assert result.proceed is True
        assert result.enriched_text == "搜索一下"

    async def test_memory_hint_from_mission(self):
        """Previous step in current mission provides a hint for navigate."""
        sk = _skill("navigate", voice_trigger="去,导航到", required_prompt="去哪里？")

        # Build a fake mission with a prior navigate step
        prior_step = MagicMock()
        prior_step.skill_name = "navigate"
        prior_step.user_text = "去仓库B"

        mission = MagicMock()
        mission.steps = [prior_step]

        dispatcher = MagicMock()
        dispatcher.current_mission = mission

        audio = MockAudio(["是"])  # user confirms the hint
        ctx = _context(dispatcher=dispatcher)
        result = await self.agent.interact(sk, "导航到", audio, ctx)

        assert result.proceed is True
        # The spoken question should reference the hint
        assert "仓库B" in audio.spoken[0]

    async def test_no_hint_when_no_mission(self):
        sk = _skill("web_search", voice_trigger="搜索一下", required_prompt="搜什么？")
        dispatcher = MagicMock()
        dispatcher.current_mission = None
        audio = MockAudio(["AI新闻"])
        ctx = _context(dispatcher=dispatcher)
        result = await self.agent.interact(sk, "搜索一下", audio, ctx)
        assert result.proceed is True
        assert audio.spoken[0] == "搜什么？"  # no hint prefix


# ── ConfirmationAgent ─────────────────────────────────────────────────────────


class TestConfirmationAgent:
    agent = ConfirmationAgent()

    def test_does_not_activate_without_flag(self):
        sk = _skill("navigate", voice_trigger="去", description="导航")
        assert not self.agent.should_activate(sk, "去仓库", _context())

    def test_activates_with_flag(self):
        sk = _skill("robot_grab", confirm=True, description="机械臂抓取")
        assert self.agent.should_activate(sk, "抓取瓶子", _context())

    async def test_yes_answer_proceeds(self):
        sk = _skill("robot_grab", confirm=True, description="机械臂抓取")
        audio = MockAudio(["确认"])
        result = await self.agent.interact(sk, "抓取瓶子", audio, _context())
        assert result.proceed is True

    async def test_no_answer_cancels(self):
        sk = _skill("robot_grab", confirm=True, description="机械臂抓取")
        audio = MockAudio(["取消"])
        result = await self.agent.interact(sk, "抓取瓶子", audio, _context())
        assert result.proceed is False
        assert result.cancelled_by == "用户取消"

    async def test_empty_answer_cancels(self):
        """Safe default: empty/timed-out answer = cancel."""
        sk = _skill("robot_move", confirm=True, description="机械臂移动")
        audio = MockAudio([None])
        result = await self.agent.interact(sk, "往前走", audio, _context())
        assert result.proceed is False
        assert result.cancelled_by == "未确认"

    async def test_ambiguous_answer_cancels(self):
        sk = _skill("robot_move", confirm=True, description="机械臂移动")
        audio = MockAudio(["也许吧"])
        result = await self.agent.interact(sk, "往前走", audio, _context())
        assert result.proceed is False

    async def test_cancelled_speech_played(self):
        """On cancel, the agent should speak a cancellation message."""
        sk = _skill("robot_grab", confirm=True, description="机械臂抓取")
        audio = MockAudio(["不要"])
        result = await self.agent.interact(sk, "抓取", audio, _context())
        assert not result.proceed
        assert any("取消" in s for s in audio.spoken)

    async def test_various_yes_words(self):
        sk = _skill("robot_grab", confirm=True, description="机械臂抓取")
        for yes in ["好的", "执行", "继续", "yes"]:
            audio = MockAudio([yes])
            result = await self.agent.interact(sk, "抓取", audio, _context())
            assert result.proceed, f"Expected proceed=True for answer={yes!r}"


# ── ProactiveOrchestrator ─────────────────────────────────────────────────────


class TestProactiveOrchestrator:

    def _make_dispatcher(self, skill: SkillDefinition) -> MagicMock:
        d = MagicMock()
        d.get_skill.return_value = skill
        d.current_mission = None
        return d

    async def test_no_dispatcher_passes_through(self):
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=None)
        audio = MockAudio()
        result = await orch.run("navigate", "去厨房", audio)
        assert result.proceed is True
        assert result.enriched_text == "去厨房"
        assert len(audio.spoken) == 0

    async def test_unknown_skill_passes_through(self):
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = None
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=dispatcher)
        audio = MockAudio()
        result = await orch.run("unknown_skill", "做点什么", audio)
        assert result.proceed is True

    async def test_slot_collected_then_confirmed(self):
        """Full chain: slot missing → ask → confirm → proceed."""
        sk = _skill(
            "robot_grab",
            voice_trigger="抓取",
            required_prompt="抓取什么物体？",
            confirm=True,
            description="机械臂抓取",
        )
        dispatcher = self._make_dispatcher(sk)
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=dispatcher)

        audio = MockAudio(["红色瓶子", "确认"])  # slot answer, then confirm
        result = await orch.run("robot_grab", "抓取", audio)

        assert result.proceed is True
        assert "红色瓶子" in result.enriched_text

    async def test_slot_collected_then_user_cancels(self):
        """Slot collected but user says 取消 → chain stops, proceed=False."""
        sk = _skill(
            "robot_grab",
            voice_trigger="抓取",
            required_prompt="抓取什么？",
            confirm=True,
            description="机械臂抓取",
        )
        dispatcher = self._make_dispatcher(sk)
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=dispatcher)

        audio = MockAudio(["红色瓶子", "取消"])
        result = await orch.run("robot_grab", "抓取", audio)

        assert result.proceed is False

    async def test_no_agents_activate_for_plain_skill(self):
        """Skill with no required_prompt and no confirm — chain is a no-op."""
        sk = _skill("get_time", voice_trigger="几点了", description="查询时间")
        dispatcher = self._make_dispatcher(sk)
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=dispatcher)

        audio = MockAudio()
        result = await orch.run("get_time", "现在几点了", audio)

        assert result.proceed is True
        assert result.enriched_text == "现在几点了"
        assert len(audio.spoken) == 0

    async def test_slot_only_no_confirm(self):
        """Slot agent fires but confirmation agent does not (confirm=False)."""
        sk = _skill(
            "web_search",
            voice_trigger="搜索一下",
            required_prompt="搜索什么？",
            confirm=False,
        )
        dispatcher = self._make_dispatcher(sk)
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=dispatcher)

        audio = MockAudio(["AI机器人"])
        result = await orch.run("web_search", "搜索一下", audio)

        assert result.proceed is True
        assert "AI机器人" in result.enriched_text
        # Only one question spoken (slot), no confirm question
        assert len(audio.spoken) == 1

    async def test_chain_order_slot_before_confirm(self):
        """Confirm question must appear after the slot question."""
        sk = _skill(
            "robot_grab",
            voice_trigger="抓取",
            required_prompt="抓什么？",
            confirm=True,
            description="机械臂抓取",
        )
        dispatcher = self._make_dispatcher(sk)
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=dispatcher)

        audio = MockAudio(["瓶子", "确认"])
        await orch.run("robot_grab", "抓取", audio)

        assert "抓什么？" in audio.spoken[0]
        # Confirm question should mention the skill description
        assert any("机械臂抓取" in s for s in audio.spoken)
