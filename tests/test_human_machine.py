"""Category 8: Human-Machine Collaboration Tests

These are the most product-quality tests. They verify the END-TO-END
conversational loop: from vague user input to correctly executed skill.

Test scenarios:
  - Full clarification round-trip: "搜索一下" → ask → answer → dispatch
  - User provides specific content upfront: no unnecessary questions
  - User changes mind mid-clarification (escape hatch)
  - Dangerous skill: slot fill → confirm → dispatch
  - Dangerous skill: slot fill → user cancels → no execution
  - Multi-turn completion: first answer still vague, retry, second answer good
  - Memory hint surfaces: system references previous dispatches

These are integration-style tests — they exercise the full ProactiveOrchestrator
chain with realistic audio mock sequences.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.skills.skill_model import SkillDefinition, SlotSpec

# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _make_dispatcher(skill: SkillDefinition) -> MagicMock:
    d = MagicMock()
    d.get_skill.return_value = skill
    d.current_mission = None
    return d


# ── Scenario 1: Single slot, bare trigger ────────────────────────────────────
# User: "搜索一下" → System: "搜索什么内容？" → User: "北京天气" → dispatch


class TestSingleSlotFlows:
    async def test_bare_trigger_asks_question_then_enriches(self):
        sk = SkillDefinition(
            name="web_search", description="搜索",
            voice_trigger="搜索一下,帮我搜索",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么内容？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio(["北京天气"])
        result = await orch.run("web_search", "搜索一下", audio)

        assert result.proceed is True
        assert "北京天气" in result.enriched_text
        assert len(audio.spoken) == 1
        assert "搜索什么内容" in audio.spoken[0]

    async def test_specific_query_no_question_asked(self):
        """User says "搜索一下穹沛科技" — no clarification needed."""
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么内容？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio()  # no answers needed
        result = await orch.run("web_search", "搜索一下穹沛科技", audio)

        assert result.proceed is True
        assert len(audio.spoken) == 0, (
            "No clarification should fire when content is already specific. "
            "Unnecessary questions degrade UX."
        )


# ── Scenario 2: Navigation with vague pronoun ────────────────────────────────
# User: "去那里" → System: "导航去哪里？" → User: "仓库B" → dispatch


class TestNavigationWithPronoun:
    async def test_vague_navigate_asks_destination(self):
        sk = SkillDefinition(
            name="navigate", description="导航",
            voice_trigger="导航到,去,带我去",
            required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio(["仓库B"])
        result = await orch.run("navigate", "去那里", audio)

        assert result.proceed is True
        assert "仓库B" in result.enriched_text
        assert "导航去哪里" in audio.spoken[0]

    async def test_named_destination_no_question(self):
        sk = SkillDefinition(
            name="navigate", voice_trigger="去",
            required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio()
        result = await orch.run("navigate", "去厨房", audio)
        assert result.proceed is True
        assert len(audio.spoken) == 0


# ── Scenario 3: Multi-turn — first vague, retry succeeds ─────────────────────
# User: "查一下那个" → System: "搜索什么内容？" → User: "那个" (still vague) →
# System: "搜索什么内容？" again → User: "AI发展趋势" → dispatch


class TestMultiTurnClarification:
    async def test_vague_retry_then_specific(self):
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下,查一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么内容？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        # First answer "那个" is vague → retry; second "AI发展趋势" is specific
        audio = MockAudio(["那个", "AI发展趋势"])
        result = await orch.run("web_search", "查一下", audio)

        assert result.proceed is True
        assert "AI发展趋势" in result.enriched_text
        # Should have asked twice (once for initial, once for retry)
        assert len(audio.spoken) == 2

    async def test_all_vague_answers_proceeds_anyway(self):
        """After MAX_TURNS vague answers, must proceed — not hang."""
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio([None, None])
        result = await orch.run("web_search", "搜索一下", audio)
        assert result.proceed is True  # MUST NOT hang or return False


# ── Scenario 4: Dangerous skill — slot + confirm → execute ───────────────────
# User: "抓取" → System: "抓什么？" → User: "红色瓶子" →
# System: "即将执行…确认吗？" → User: "确认" → dispatch


class TestDangerousSkillHappyPath:
    async def test_grab_full_flow_with_confirm(self):
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取,拿起",
            confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓取什么物体？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio(["红色瓶子", "确认"])  # slot then confirm
        result = await orch.run("robot_grab", "抓取", audio)

        assert result.proceed is True
        assert "红色瓶子" in result.enriched_text
        # Two questions: one for slot, one for confirmation
        assert len(audio.spoken) == 2

    async def test_slot_then_question_order(self):
        """Slot question must come BEFORE confirmation question."""
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取",
            confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓什么？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio(["绿色积木", "好的"])
        await orch.run("robot_grab", "抓取", audio)

        assert "抓什么" in audio.spoken[0], "Slot question must come first"
        assert "机械臂抓取" in audio.spoken[1], "Confirm question must come second"


# ── Scenario 5: Dangerous skill — user cancels ───────────────────────────────
# User: "抓取" → System: "抓什么？" → User: "那个红杯子" →
# System: "确认吗？" → User: "取消" → NO dispatch


class TestDangerousSkillCancelPath:
    async def test_cancel_after_slot_no_dispatch(self):
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取",
            confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓什么？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio(["红色杯子", "不要"])
        result = await orch.run("robot_grab", "抓取", audio)

        assert result.proceed is False, (
            "User said '不要' at confirmation — MUST NOT dispatch. "
            "Robot arm movements cannot be undone."
        )

    async def test_cancel_speaks_cancellation_message(self):
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取",
            confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓什么？")],
        )
        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=_make_dispatcher(sk)
        )
        audio = MockAudio(["红杯子", "取消"])
        await orch.run("robot_grab", "抓取", audio)
        assert any("取消" in s for s in audio.spoken), (
            "System must verbally acknowledge the cancellation."
        )


# ── Scenario 6: Memory hint surfaces previous value ──────────────────────────


class TestMemoryHintInClarification:
    async def test_previous_navigate_destination_offered(self):
        sk = SkillDefinition(
            name="navigate", description="导航",
            voice_trigger="导航到,去",
            required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
        )
        # Simulate previous mission step
        prior_step = MagicMock()
        prior_step.skill_name = "navigate"
        prior_step.user_text = "去仓库B"

        mission = MagicMock()
        mission.steps = [prior_step]

        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = mission

        orch = ProactiveOrchestrator.default(
            pipeline=FakePipeline(), dispatcher=dispatcher
        )
        audio = MockAudio(["是"])  # user confirms hint
        result = await orch.run("navigate", "导航到", audio)

        # Question should reference the previous destination
        assert "仓库B" in audio.spoken[0], (
            "System must surface memory hint for returning users. "
            "This reduces friction for repeated tasks."
        )
        assert result.proceed is True
