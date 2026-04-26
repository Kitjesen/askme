"""Category: Safety & ESTOP Linkage

Real risks covered:
  - Robot arm moves because "好像是" was misread as YES ("好" substring match) —
    BUG: old _is_yes used `any(w in text)`, so single-char "好" matched "好像是"
  - "嗯哼" (hesitation noise) treated as "嗯" (yes) → physical action fires
  - ESTOP word "急停" during confirmation — old code returned cancelled_by="未确认",
    not "estop"; monitoring/logging couldn't distinguish safety abort from timeout
  - No confirmation bypass: even if slots are perfectly filled, a
    confirm_before_execute skill MUST ask for confirmation before executing

Status BEFORE _is_yes single-char fix and ESTOP detection:
  - test_hao_xiang_shi_not_confirmed:  FAIL  (was True, should be False)
  - test_estop_cancelled_by_is_estop:  FAIL  (was "未确认", should be "estop")
  - test_regular_cancel_not_estop:     PASS  (cancel logic unchanged)
Status AFTER implementation: ALL PASS.
"""

from __future__ import annotations

from askme.pipeline.proactive.base import ProactiveContext
from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
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


def _grab_skill(confirm=True):
    return SkillDefinition(
        name="robot_grab", description="机械臂抓取",
        voice_trigger="抓取,拿起",
        confirm_before_execute=confirm,
    )


def _ctx():
    return ProactiveContext(session=ClarificationSession())


# ── Ambiguous answers default to cancel (safe) ────────────────────────────────

class TestAmbiguousAnswerDefaultsCancel:
    """The core safety principle: uncertain input = do NOT execute."""

    async def test_hao_xiang_shi_not_confirmed(self):
        """'好像是' contains '好' but is NOT a confirmation.

        Before fix: _is_yes used any(w in text) → '好' in '好像是' → True → BUG.
        After fix: single-char '好' requires exact match → False → safe cancel.
        """
        agent = ConfirmationAgent()
        audio = MockAudio(["好像是"])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False, (
            "'好像是' (maybe/seems like) must NOT confirm execution. "
            "Before the single-char fix, '好' in '好像是' triggered True."
        )

    async def test_en_heng_not_confirmed(self):
        """'嗯哼' (hesitation) should NOT match single-char '嗯' (yes)."""
        agent = ConfirmationAgent()
        audio = MockAudio(["嗯哼"])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False

    async def test_silence_cancels(self):
        agent = ConfirmationAgent()
        audio = MockAudio([None])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False

    async def test_empty_string_cancels(self):
        agent = ConfirmationAgent()
        audio = MockAudio([""])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False

    async def test_partial_yes_word_cancels(self):
        """'确' alone (1 char) must NOT confirm — only '确认' or '确定' do."""
        agent = ConfirmationAgent()
        audio = MockAudio(["确"])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False


# ── Exact YES words DO confirm ─────────────────────────────────────────────────

class TestExactYESWordsConfirm:
    async def test_queren_confirms(self):
        agent = ConfirmationAgent()
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", MockAudio(["确认"]), _ctx())
        assert result.proceed is True

    async def test_hao_de_confirms(self):
        agent = ConfirmationAgent()
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", MockAudio(["好的"]), _ctx())
        assert result.proceed is True

    async def test_single_hao_confirms(self):
        """Single '好' by itself IS a valid confirmation (exact match)."""
        agent = ConfirmationAgent()
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", MockAudio(["好"]), _ctx())
        assert result.proceed is True

    async def test_single_shi_confirms(self):
        agent = ConfirmationAgent()
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", MockAudio(["是"]), _ctx())
        assert result.proceed is True


# ── ESTOP in confirmation ──────────────────────────────────────────────────────

class TestESTOPInConfirmation:
    async def test_jiting_cancels_with_estop_flag(self):
        """'急停' must set cancelled_by='estop' for monitoring to distinguish it."""
        agent = ConfirmationAgent()
        audio = MockAudio(["急停"])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False
        assert result.cancelled_by == "estop", (
            f"Expected 'estop', got {result.cancelled_by!r}. "
            "ESTOP must be distinguishable from a regular cancel in logs/monitoring."
        )

    async def test_jinji_tingzhi_is_estop(self):
        agent = ConfirmationAgent()
        audio = MockAudio(["紧急停止"])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.cancelled_by == "estop"

    async def test_regular_cancel_is_not_estop(self):
        """'取消' is a regular cancel, NOT an ESTOP — different severity."""
        agent = ConfirmationAgent()
        audio = MockAudio(["取消"])
        result = await agent.interact(_grab_skill(), "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False
        assert result.cancelled_by == "用户取消", (
            f"'取消' must be '用户取消', not 'estop'. Got: {result.cancelled_by!r}"
        )

    async def test_estop_session_state_is_canceled(self):
        agent = ConfirmationAgent()
        session = ClarificationSession()
        ctx = ProactiveContext(session=session)
        await agent.interact(_grab_skill(), "抓取红色瓶子", MockAudio(["急停"]), ctx)
        assert session.state == ClarificationState.CANCELED
        assert session.cancel_reason == "estop"


# ── Confirm required even with complete slot data ──────────────────────────────

class TestConfirmBeforePhysicalAction:
    async def test_full_slot_data_still_requires_confirm(self):
        """Even when enriched_text already contains the object, confirm fires."""
        from unittest.mock import MagicMock
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取", confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓什么？")],
        )
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        orch = ProactiveOrchestrator.default(pipeline=MagicMock(), dispatcher=dispatcher)
        # "抓取红色瓶子" — object slot IS filled, but confirm is still required
        audio = MockAudio(["取消"])   # user cancels at confirm stage
        result = await orch.run("robot_grab", "抓取红色瓶子", audio)

        assert result.proceed is False, (
            "Confirm must still fire even when all slots are filled. "
            "The '取消' response at confirm should block execution."
        )

    async def test_slot_asked_before_confirm_not_after(self):
        """Ordering: slot question MUST come before confirmation question."""
        from unittest.mock import MagicMock
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取", confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓什么？")],
        )
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        orch = ProactiveOrchestrator.default(pipeline=MagicMock(), dispatcher=dispatcher)
        audio = MockAudio(["红色瓶子", "确认"])
        await orch.run("robot_grab", "抓取", audio)

        assert len(audio.spoken) == 2
        assert "抓什么" in audio.spoken[0], (
            f"First question should be about the slot, got: {audio.spoken[0]!r}"
        )
        assert "确认" in audio.spoken[1] or "取消" in audio.spoken[1], (
            f"Second question should be the confirm prompt, got: {audio.spoken[1]!r}"
        )
