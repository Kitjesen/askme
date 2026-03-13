"""Category 1: Semantic Ambiguity Tests

Focus: the system should NEVER silently proceed when the user's intent is
ambiguous. A vague pronoun, an unresolved referent, or an unclear target
must trigger clarification — not a confident (and possibly wrong) execution.

What we're verifying here is NOT just pass/fail, but:
  - Does clarification fire when it should?
  - Does the system give back a question (not a wrong action)?
  - Does it NOT fire when the request is already specific?
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from askme.pipeline.proactive.slot_analyst import analyze_slots, is_vague
from askme.pipeline.proactive.clarification_agent import ClarificationPlannerAgent
from askme.pipeline.proactive.base import ProactiveContext
from askme.skills.skill_model import SkillDefinition, SlotSpec


# ── fixtures ──────────────────────────────────────────────────────────────────


class FakePipeline:
    """Simulates extract_semantic_target: returns specific destination or empty."""

    def extract_semantic_target(self, text: str) -> str:
        import re
        # Matches: "去仓库B", "导航到控制室", "带我去A区"
        patterns = [
            r"去(.{1,15}?)(?:吧|了|吗|一下|$)",
            r"导航到(.{1,15}?)(?:去|了|吧|$)",
            r"带我去(.{1,15}?)(?:吧|了|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return text  # unchanged = no specific target found


class MockAudio:
    def __init__(self, answers=None):
        self.spoken: list[str] = []
        self._answers = list(answers or [])
        self._idx = 0
    def drain_buffers(self): ...
    def speak(self, t): self.spoken.append(t)
    def start_playback(self): ...
    def stop_playback(self): ...
    def wait_speaking_done(self): ...
    def listen_loop(self):
        if self._idx < len(self._answers):
            r = self._answers[self._idx]; self._idx += 1; return r
        return None


def _nav_skill():
    return SkillDefinition(
        name="navigate", description="导航",
        voice_trigger="导航到,去,带我去,前往",
        required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
    )

def _inspect_skill():
    return SkillDefinition(
        name="inspect", description="巡检",
        voice_trigger="检查,巡查,看看,检查一下",
        required_slots=[SlotSpec(name="target", type="text", prompt="检查哪里或什么？")],
    )

def _ctx(pipeline=None):
    return ProactiveContext(pipeline=pipeline or FakePipeline(), dispatcher=None)


agent = ClarificationPlannerAgent()


# ── Category 1a: Pronoun / deictic ambiguity ─────────────────────────────────
# "那里" "那边" "过去" "这里" → unresolved spatial reference, must ask


class TestDeiticAmbiguity:
    """Vague spatial pronouns should always trigger clarification."""

    def test_去那里_requires_clarification(self):
        sk = _nav_skill()
        assert agent.should_activate(sk, "去那里", _ctx())

    def test_去那边_requires_clarification(self):
        sk = _nav_skill()
        assert agent.should_activate(sk, "去那边", _ctx())

    def test_过来_requires_clarification(self):
        sk = _nav_skill()
        assert agent.should_activate(sk, "走过来", _ctx())

    def test_去那里看看_requires_clarification(self):
        sk = _nav_skill()
        assert agent.should_activate(sk, "去那里看看", _ctx())

    def test_具体目标_does_not_require_clarification(self):
        """Specific named destination must NOT trigger clarification."""
        sk = _nav_skill()
        assert not agent.should_activate(sk, "去仓库B", _ctx())

    def test_去厨房_does_not_require_clarification(self):
        sk = _nav_skill()
        assert not agent.should_activate(sk, "去厨房", _ctx())


# ── Category 1b: Referent ambiguity ──────────────────────────────────────────
# "那个东西" "这件" "它" → no concrete object named

class TestReferentAmbiguity:
    def _grab_skill(self):
        return SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取,拿起",
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓取什么物体？")],
        )

    def test_抓那个_requires_clarification(self):
        sk = self._grab_skill()
        assert agent.should_activate(sk, "抓那个", _ctx())

    def test_抓这个东西_requires_clarification(self):
        sk = self._grab_skill()
        assert agent.should_activate(sk, "抓这个东西", _ctx())

    def test_拿起它_requires_clarification(self):
        sk = self._grab_skill()
        assert agent.should_activate(sk, "拿起", _ctx())  # bare trigger, nothing else

    def test_抓红色瓶子_does_not_require_clarification(self):
        sk = self._grab_skill()
        assert not agent.should_activate(sk, "抓红色瓶子", _ctx())


# ── Category 1c: Semantically-complete command passes through ─────────────────


class TestNoFalsePositiveClarification:
    """Specific, complete commands MUST NOT trigger unnecessary clarification."""

    def test_navigate_with_full_destination(self):
        sk = _nav_skill()
        assert not agent.should_activate(sk, "导航到仓库B", _ctx())

    def test_search_with_full_query(self):
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么？")],
        )
        assert not agent.should_activate(sk, "搜索一下明天北京天气", _ctx())

    def test_plain_skill_no_slots_never_asks(self):
        """A skill with no required_slots never activates ClarificationPlannerAgent."""
        sk = SkillDefinition(name="get_time", voice_trigger="几点了")
        assert not agent.should_activate(sk, "几点了", _ctx())


# ── Category 1d: Question vs command disambiguation ──────────────────────────
# These verify the IntentRouter correctly separates questions from commands.
# A question about navigation should NOT trigger the navigate skill.


class TestQuestionVsCommand:
    """Intent routing: questions must go to GENERAL, not fire voice triggers."""

    def test_navigate_question_routes_to_general(self):
        from askme.brain.intent_router import IntentRouter, IntentType
        router = IntentRouter(voice_triggers={"导航": "navigate"})
        # "导航会失败吗" ends with 吗 → question → GENERAL
        intent = router.route("导航会失败吗")
        assert intent.type == IntentType.GENERAL, (
            "Question about navigation should NOT fire the navigate skill"
        )

    def test_navigation_command_fires_skill(self):
        from askme.brain.intent_router import IntentRouter, IntentType
        router = IntentRouter(voice_triggers={"导航": "navigate", "导航到仓库": "navigate"})
        intent = router.route("帮我导航到仓库")
        assert intent.type == IntentType.VOICE_TRIGGER

    def test_ambiguous_probe_does_not_navigate(self):
        from askme.brain.intent_router import IntentRouter, IntentType
        router = IntentRouter(voice_triggers={"导航到仓库": "navigate"})
        # "有没有仓库可以导航到" → embedded trigger but it's a question
        intent = router.route("有没有仓库可以导航到吗")
        assert intent.type == IntentType.GENERAL


# ── Category 1e: Clarification question quality ───────────────────────────────
# When clarification IS triggered, verify the question is meaningful.


class TestClarificationQuestionQuality:
    async def test_vague_navigation_asks_destination(self):
        sk = _nav_skill()
        audio = MockAudio(["仓库B"])
        await agent.interact(sk, "去那里", audio, _ctx())
        # Must have asked something related to destination
        assert len(audio.spoken) >= 1
        assert audio.spoken[0], "Should have asked a non-empty question"
        # The question should reference the prompt
        assert "导航去哪里" in audio.spoken[0] or "哪里" in audio.spoken[0]

    async def test_vague_grab_asks_object(self):
        sk = SkillDefinition(
            name="robot_grab", voice_trigger="抓取",
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓取什么物体？")],
        )
        audio = MockAudio(["红色瓶子"])
        await agent.interact(sk, "抓那个", audio, _ctx())
        assert "抓取什么物体" in audio.spoken[0]
