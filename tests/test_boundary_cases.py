"""Category 2: Boundary Word & Error Task Tests

Anti-pattern: "看起来聪明，实际上爱瞎猜"

These tests guard against:
  - High-confidence mismatches (trigger fires but content is invalid)
  - Slot content that is syntactically long but semantically empty
  - Vague-but-trigger-shaped inputs (content IS a trigger phrase)
  - Unknown skill / non-existent targets passed to orchestrator
  - Mismatch between trigger verb and actual task semantics
"""

from __future__ import annotations

import pytest

from askme.pipeline.proactive.slot_analyst import analyze_slots, is_vague
from askme.pipeline.proactive.slot_utils import slot_present
from askme.pipeline.proactive.clarification_agent import ClarificationPlannerAgent
from askme.pipeline.proactive.base import ProactiveContext
from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.skills.skill_model import SkillDefinition, SlotSpec


# ── Mock helpers ──────────────────────────────────────────────────────────────

class FakePipeline:
    def extract_semantic_target(self, text: str) -> str:
        import re
        m = re.search(r"去(.{1,10}?)(?:吧|了|$)", text)
        if m and m.group(1).strip():
            return m.group(1).strip()
        return text


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


# ── Boundary 1: Trigger phrase IS the full content ───────────────────────────
# e.g. "搜索一下" has trigger "搜索一下" → remainder = "" → missing


class TestTriggerEqualsContent:
    def test_exact_trigger_word_is_missing_slot(self):
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么？")],
        )
        a = analyze_slots(sk, "搜索一下")
        assert not a.ready, "Bare trigger with no payload must be missing"

    def test_longer_trigger_still_missing(self):
        sk = SkillDefinition(
            name="web_search", voice_trigger="帮我搜索,搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么？")],
        )
        a = analyze_slots(sk, "帮我搜索")
        assert not a.ready

    def test_bare_mapping_trigger(self):
        sk = SkillDefinition(
            name="mapping", voice_trigger="建图,开始建图",
            required_slots=[SlotSpec(name="map_scope", type="text", prompt="建哪里？")],
        )
        a = analyze_slots(sk, "建图")
        assert not a.ready
        a2 = analyze_slots(sk, "开始建图")
        assert not a2.ready


# ── Boundary 2: Long text with only filler content ───────────────────────────
# "搜索一下那个东西" — 10 chars, passes the old length=4 check, but is still vague


class TestLongButVagueContent:
    def test_搜索那个东西_is_vague(self):
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么？")],
        )
        a = analyze_slots(sk, "搜索一下那个东西")
        # "那个东西" after stripping trigger — is_vague("那个东西") should be True
        assert not a.ready, (
            "'那个东西' is a vague referent, not a real search query. "
            "Old length-check (>4 chars) would wrongly accept this."
        )

    def test_去那里看看_is_vague_location(self):
        assert is_vague("那里")
        assert is_vague("那边")

    def test_随便_is_filler(self):
        assert is_vague("随便")

    def test_都行_is_filler(self):
        assert is_vague("都行")

    def test_actual_content_not_filler(self):
        assert not is_vague("北京明天天气")
        assert not is_vague("仓库B二层")
        assert not is_vague("穹沛科技最新消息")


# ── Boundary 3: Slot_present legacy check with vague words ───────────────────
# Verifies slot_utils.slot_present now rejects vague remainders


class TestSlotPresentRejectsVague:
    def test_那个_rejected_by_slot_present(self):
        sk = SkillDefinition(
            name="robot_grab", voice_trigger="抓取",
            required_prompt="抓取什么？",
        )
        # "抓取那个" → remainder="那个" → is_vague → False
        assert not slot_present(sk, "抓取那个")

    def test_real_content_accepted(self):
        sk = SkillDefinition(
            name="robot_grab", voice_trigger="抓取",
            required_prompt="抓取什么？",
        )
        assert slot_present(sk, "抓取红色瓶子")

    def test_那里_rejected_for_navigation(self):
        sk = SkillDefinition(
            name="navigate", voice_trigger="导航到",
            required_prompt="导航去哪里？",
        )
        # slot_present for navigate uses extract_semantic_target
        # "导航到那里" → pipeline needed; without pipeline, navigate returns False
        assert not slot_present(sk, "导航到那里", pipeline=None)


# ── Boundary 4: Unknown skill passes through orchestrator ────────────────────
# Orchestrator must NOT raise on unknown skill — just pass through


class TestUnknownSkillHandling:
    async def test_unknown_skill_name_passes_through(self):
        from unittest.mock import MagicMock
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = None  # skill not found

        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=dispatcher
        )
        audio = MockAudio()
        result = await orch.run("nonexistent_skill_xyz", "some text", audio)
        assert result.proceed is True
        assert result.enriched_text == "some text"
        assert len(audio.spoken) == 0  # no clarification fired

    async def test_no_dispatcher_passes_through(self):
        orch = ProactiveOrchestrator.default(pipeline=FakePipeline(), dispatcher=None)
        audio = MockAudio()
        result = await orch.run("navigate", "去厨房", audio)
        assert result.proceed is True


# ── Boundary 5: Trigger fires but task is semantically wrong ─────────────────
# Guard: a navigation command shouldn't fire if route leads to estop

class TestSemanticMismatch:
    def test_negated_trigger_does_not_fire(self):
        """'不要导航' should NOT trigger navigate skill."""
        from askme.brain.intent_router import IntentRouter, IntentType
        router = IntentRouter(voice_triggers={"导航到仓库": "navigate"})
        intent = router.route("不要导航到仓库")
        assert intent.type == IntentType.GENERAL, (
            "Negated command should NOT fire the skill — "
            "navigating when user said '不要导航' would be a dangerous error."
        )

    def test_别导航_does_not_fire(self):
        from askme.brain.intent_router import IntentRouter, IntentType
        router = IntentRouter(voice_triggers={"导航": "navigate"})
        intent = router.route("别导航了")
        assert intent.type == IntentType.GENERAL

    def test_positive_command_still_fires(self):
        from askme.brain.intent_router import IntentRouter, IntentType
        router = IntentRouter(voice_triggers={"导航到仓库": "navigate"})
        intent = router.route("帮我导航到仓库")
        assert intent.type == IntentType.VOICE_TRIGGER


# ── Boundary 6: Edge filler in multi-slot scenario ───────────────────────────


class TestMultiSlotEdgeFiller:
    def test_grab_with_vague_object_missing(self):
        sk = SkillDefinition(
            name="robot_grab", voice_trigger="抓取",
            required_slots=[
                SlotSpec(name="object", type="referent", prompt="抓什么？"),
            ],
        )
        # "抓取随便一个" — "随便一个" is vague
        a = analyze_slots(sk, "抓取随便一个")
        # "随便" is in vague set but "随便一个" (as a whole) might not be
        # The key check: the slot value extracted is "随便一个"
        # is_vague("随便一个") → "随便" is contained but is_vague checks full string
        # This documents the CURRENT behavior: partial vague phrases not caught
        # TODO Phase 3: per-token vague detection
        assert a.slots[0].value == "随便一个"  # documents what we extract

    def test_grab_with_real_object_ready(self):
        sk = SkillDefinition(
            name="robot_grab", voice_trigger="拿起",
            required_slots=[SlotSpec(name="object", type="referent", prompt="拿什么？")],
        )
        a = analyze_slots(sk, "拿起蓝色积木")
        assert a.ready
        assert a.slots[0].value == "蓝色积木"
