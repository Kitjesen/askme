"""Tests for ClarificationPlannerAgent and the _combine_prompts helper."""

import pytest
from unittest.mock import MagicMock

from askme.pipeline.proactive.base import ProactiveContext
from askme.pipeline.proactive.clarification_agent import (
    ClarificationPlannerAgent,
    _combine_prompts,
)
from askme.skills.skill_model import SkillDefinition, SlotSpec


# ── helpers ───────────────────────────────────────────────────────────────────


class FakePipeline:
    def extract_semantic_target(self, text: str) -> str:
        import re
        m = re.search(r"去(.{1,10}?)(?:吧|了|$)", text)
        if m and m.group(1).strip():
            return m.group(1).strip()
        return text


class MockAudio:
    def __init__(self, answers: list[str | None]) -> None:
        self.spoken: list[str] = []
        self._answers = list(answers)
        self._idx = 0

    def drain_buffers(self) -> None: ...
    def speak(self, text: str) -> None: self.spoken.append(text)
    def start_playback(self) -> None: ...
    def stop_playback(self) -> None: ...
    def wait_speaking_done(self) -> None: ...
    def listen_loop(self) -> str | None:
        if self._idx < len(self._answers):
            ans = self._answers[self._idx]
            self._idx += 1
            return ans
        return None


def _ctx(dispatcher=None) -> ProactiveContext:
    return ProactiveContext(pipeline=FakePipeline(), dispatcher=dispatcher)


def _nav_skill() -> SkillDefinition:
    return SkillDefinition(
        name="navigate",
        description="导航",
        voice_trigger="导航到,去",
        required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
    )


def _grab_skill() -> SkillDefinition:
    return SkillDefinition(
        name="robot_grab",
        description="机械臂抓取",
        voice_trigger="抓取,拿起",
        required_slots=[
            SlotSpec(name="object", type="referent", prompt="抓取什么物体？"),
            SlotSpec(name="place_target", type="location", prompt="放到哪里？", optional=True),
        ],
        confirm_before_execute=True,
    )


def _search_skill() -> SkillDefinition:
    return SkillDefinition(
        name="web_search",
        voice_trigger="搜索一下,查一下",
        required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么内容？")],
    )


# ── _combine_prompts ──────────────────────────────────────────────────────────


class TestCombinePrompts:
    def test_single_prompt_unchanged(self):
        assert _combine_prompts(["搜索什么内容？"]) == "搜索什么内容？"

    def test_two_prompts_merged(self):
        result = _combine_prompts(["抓取什么物体？", "放到哪里？"])
        assert "抓取什么物体" in result
        assert "放到哪里" in result
        # Last prompt keeps trailing question mark
        assert result.endswith("放到哪里？")

    def test_three_prompts_merged(self):
        result = _combine_prompts(["A？", "B？", "C？"])
        assert result.endswith("C？")
        assert "A" in result and "B" in result

    def test_empty_list(self):
        assert _combine_prompts([]) == ""


# ── ClarificationPlannerAgent ─────────────────────────────────────────────────


class TestClarificationPlannerAgentShouldActivate:
    agent = ClarificationPlannerAgent()

    def test_no_required_slots_skip(self):
        sk = SkillDefinition(name="get_time", voice_trigger="几点了")
        assert not self.agent.should_activate(sk, "几点了", _ctx())

    def test_skill_with_slots_slot_already_filled_skip(self):
        """If navigate has a destination, should NOT activate."""
        sk = _nav_skill()
        assert not self.agent.should_activate(sk, "去仓库B", _ctx())

    def test_skill_with_slots_slot_missing_activates(self):
        sk = _nav_skill()
        assert self.agent.should_activate(sk, "导航到", _ctx())

    def test_skill_with_slots_vague_value_activates(self):
        sk = _search_skill()
        assert self.agent.should_activate(sk, "搜索一下那个", _ctx())


class TestClarificationPlannerAgentInteract:
    agent = ClarificationPlannerAgent()

    async def test_single_slot_collected(self):
        sk = _search_skill()
        audio = MockAudio(["穹沛科技"])
        result = await self.agent.interact(sk, "搜索一下", audio, _ctx())
        assert result.proceed is True
        assert "穹沛科技" in result.enriched_text
        assert "搜索什么内容？" in audio.spoken[0]

    async def test_slot_already_present_no_question(self):
        """If analysis says ready (all slots filled), no question should be asked."""
        sk = _search_skill()
        audio = MockAudio([])  # no answers needed
        result = await self.agent.interact(sk, "搜索一下穹沛科技", audio, _ctx())
        assert result.proceed is True
        assert len(audio.spoken) == 0

    async def test_retry_on_empty_first_answer(self):
        sk = _search_skill()
        audio = MockAudio(["", "北京天气"])  # first empty, then valid
        result = await self.agent.interact(sk, "搜索一下", audio, _ctx())
        assert result.proceed is True
        assert "北京天气" in result.enriched_text

    async def test_vague_answer_triggers_retry(self):
        sk = _search_skill()
        audio = MockAudio(["那个", "北京明日天气"])  # "那个" is vague → retry
        result = await self.agent.interact(sk, "搜索一下", audio, _ctx())
        assert result.proceed is True
        assert "北京明日天气" in result.enriched_text

    async def test_multi_slot_single_required_asked_once(self):
        """robot_grab: required=object (optional=place_target). Only one question."""
        sk = _grab_skill()
        audio = MockAudio(["红色瓶子"])
        result = await self.agent.interact(sk, "抓取", audio, _ctx())
        assert result.proceed is True
        assert len(audio.spoken) == 1  # only object asked (place_target is optional)
        assert "抓取什么物体" in audio.spoken[0]

    async def test_memory_hint_in_question(self):
        """Previous navigate step in mission → hint appears in question."""
        sk = _nav_skill()

        step = MagicMock()
        step.skill_name = "navigate"
        step.user_text = "去仓库B"

        mission = MagicMock()
        mission.steps = [step]

        dispatcher = MagicMock()
        dispatcher.current_mission = mission

        audio = MockAudio(["是"])  # confirm the hint
        result = await self.agent.interact(sk, "导航到", audio, _ctx(dispatcher=dispatcher))
        assert result.proceed is True
        assert "仓库B" in audio.spoken[0]

    async def test_no_hint_without_mission(self):
        sk = _nav_skill()
        dispatcher = MagicMock()
        dispatcher.current_mission = None
        audio = MockAudio(["仓库C"])
        result = await self.agent.interact(sk, "导航到", audio, _ctx(dispatcher=dispatcher))
        assert result.proceed is True
        assert audio.spoken[0] == "导航去哪里？"  # plain prompt, no hint

    async def test_proceed_true_even_when_all_attempts_fail(self):
        """If user never gives a good answer, we still proceed (LLM handles it)."""
        sk = _search_skill()
        audio = MockAudio([None, None])  # both turns empty
        result = await self.agent.interact(sk, "搜索一下", audio, _ctx())
        assert result.proceed is True
