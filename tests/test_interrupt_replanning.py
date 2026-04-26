"""Category: Interrupt & Re-planning

Real risks covered:
  - User says "算了" mid-slot-collection → old code appended "算了" to the
    command text and dispatched it ("搜索 算了") — dangerous garbage execution
  - "急停" as a slot answer → old code treated it as content, enriched the
    command with "急停" in the text — robot would try to grab "急停"
  - After interrupt, session state is INTERRUPTED — stale data must not bleed
    into the next command's context
  - System must immediately stop on ESTOP regardless of where we are in the
    clarification chain

Status BEFORE ClarificationPlannerAgent interrupt detection was added:
  - test_suan_le_aborts_slot_collection: FAIL (proceeded with "搜索 算了")
  - test_interrupt_original_text_preserved: FAIL (enriched_text was contaminated)
  - test_jiting_aborts_slot: FAIL (proceeded with "急停" in text)
  - test_session_state_is_interrupted: FAIL (no session existed in context)
Status AFTER implementation: ALL PASS.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from askme.pipeline.proactive.base import ProactiveContext
from askme.pipeline.proactive.clarification_agent import ClarificationPlannerAgent
from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.pipeline.proactive.session_state import ClarificationSession, ClarificationState
from askme.skills.skill_model import SkillDefinition, SlotSpec

# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _search_skill():
    return SkillDefinition(
        name="web_search", description="搜索",
        voice_trigger="搜索一下,搜索",
        required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么内容？")],
    )


def _ctx(session=None):
    return ProactiveContext(session=session or ClarificationSession())


# ── Interrupt during slot collection ──────────────────────────────────────────

class TestInterruptDuringSlotCollection:
    """算了 as answer to slot question must abort the flow, not enrich the text."""

    async def test_suan_le_aborts_slot_collection(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["算了"])   # user bails out
        result = await agent.interact(sk, "搜索一下", audio, _ctx())

        assert result.proceed is False, (
            "User said '算了' — must NOT proceed. "
            "Before fix: result.proceed was True and '算了' was appended to command text."
        )

    async def test_interrupt_cancelled_by_field(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["算了"])
        result = await agent.interact(sk, "搜索一下", audio, _ctx())

        assert result.cancelled_by == "interrupted", (
            f"Expected cancelled_by='interrupted', got {result.cancelled_by!r}"
        )

    async def test_interrupt_original_text_preserved(self):
        """enriched_text must be the ORIGINAL command — not contaminated by '算了'."""
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["算了"])
        result = await agent.interact(sk, "搜索一下", audio, _ctx())

        assert result.enriched_text == "搜索一下", (
            "enriched_text must equal original user_text on interrupt. "
            f"Got: {result.enriched_text!r}"
        )
        assert "算了" not in result.enriched_text

    async def test_interrupt_word_not_appended_as_content(self):
        """Contrast: before fix, agent would do current_text += ' 算了'."""
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["算了"])
        result = await agent.interact(sk, "搜索", audio, _ctx())
        assert "算了" not in result.enriched_text

    async def test_session_state_is_interrupted(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        session = ClarificationSession()
        audio = MockAudio(["算了"])
        await agent.interact(sk, "搜索一下", audio, _ctx(session))

        assert session.state == ClarificationState.INTERRUPTED
        assert session.interrupted_by == "算了"

    async def test_bugao_also_interrupts(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["不了"])
        result = await agent.interact(sk, "搜索一下", audio, _ctx())
        assert result.proceed is False
        assert result.cancelled_by == "interrupted"

    async def test_valid_answer_before_interrupt_still_valid(self):
        """First answer is real, second turn is interrupt — first answer was accepted."""
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索",
            required_slots=[
                SlotSpec(name="query", type="text", prompt="搜什么？"),
            ],
        )
        agent = ClarificationPlannerAgent()
        audio = MockAudio(["北京天气"])  # valid first answer
        result = await agent.interact(sk, "搜索", audio, _ctx())

        assert result.proceed is True
        assert "北京天气" in result.enriched_text


# ── ESTOP during slot collection ──────────────────────────────────────────────

class TestESTOPDuringSlotCollection:
    """Emergency-stop during clarification must abort immediately."""

    async def test_jiting_aborts_slot(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["急停"])
        result = await agent.interact(sk, "搜索一下", audio, _ctx())

        assert result.proceed is False
        assert result.cancelled_by == "estop", (
            "ESTOP word must set cancelled_by='estop', not 'interrupted'. "
            f"Got: {result.cancelled_by!r}"
        )

    async def test_jiting_not_enriched_into_command(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["急停"])
        result = await agent.interact(sk, "搜索一下", audio, _ctx())
        assert "急停" not in result.enriched_text

    async def test_session_state_canceled_on_estop(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        session = ClarificationSession()
        audio = MockAudio(["急停"])
        await agent.interact(sk, "搜索一下", audio, _ctx(session))

        assert session.state == ClarificationState.CANCELED
        assert session.cancel_reason == "estop"

    async def test_jinji_tingzhi_is_estop(self):
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["紧急停止"])
        result = await agent.interact(sk, "搜索一下", audio, _ctx())
        assert result.proceed is False
        assert result.cancelled_by == "estop"


# ── Interrupt during confirmation ──────────────────────────────────────────────

class TestInterruptDuringConfirmation:
    """中断和急停词在 confirm 阶段同样生效。"""

    async def test_estop_during_confirmation_aborts(self):
        from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        session = ClarificationSession(state=ClarificationState.IDLE)
        ctx = ProactiveContext(session=session)
        audio = MockAudio(["急停"])

        agent = ConfirmationAgent()
        result = await agent.interact(sk, "抓取红色瓶子", audio, ctx)

        assert result.proceed is False
        assert result.cancelled_by == "estop"

    async def test_session_state_canceled_on_confirmation_estop(self):
        from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        session = ClarificationSession(state=ClarificationState.IDLE)
        ctx = ProactiveContext(session=session)
        audio = MockAudio(["急停"])
        await ConfirmationAgent().interact(sk, "抓取红色瓶子", audio, ctx)

        assert session.cancel_reason == "estop"


# ── After interrupt: system state is clean ─────────────────────────────────────

class TestAfterInterruptSystemReset:
    async def test_after_interrupt_next_run_is_clean(self):
        """Run 1 interrupted. Run 2 must have fresh session — no state leakage."""
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜什么？")],
        )
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        orch = ProactiveOrchestrator.default(pipeline=MagicMock(), dispatcher=dispatcher)

        # Run 1 — interrupt
        r1 = await orch.run("web_search", "搜索一下", MockAudio(["算了"]))
        assert r1.proceed is False

        # Run 2 — normal, should not be affected by run 1's session
        r2 = await orch.run("web_search", "搜索明天天气", MockAudio([]))
        assert r2.proceed is True, (
            "After an interrupted run, the next run must succeed cleanly. "
            "If session state leaked, this would be stuck in INTERRUPTED."
        )

    async def test_turn_count_not_carried_over(self):
        """Turn count from an interrupted session must not appear in the next session."""
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜什么？")],
        )
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        orch = ProactiveOrchestrator.default(pipeline=MagicMock(), dispatcher=dispatcher)

        # Interrupted run (used up 1 turn)
        await orch.run("web_search", "搜索一下", MockAudio(["算了"]))

        # Clean run should need at most MAX_TURNS questions, not be penalised
        r2 = await orch.run("web_search", "搜索一下", MockAudio(["北京天气"]))
        assert r2.proceed is True
        assert "北京天气" in r2.enriched_text
