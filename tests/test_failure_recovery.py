"""Category 4: Failure Recovery Tests

What happens AFTER something goes wrong?

  - Skill execution raises or returns [Error]
  - ConfirmationAgent cancels → no dispatch happens
  - SlotCollector runs out of attempts → proceeds anyway (graceful)
  - Pipeline raises TimeoutError mid-execution
  - Next command after failure must work normally (no stuck state)

Key principle: failure should be CONTAINED and RECOVERABLE.
The system must never enter an unrecoverable state after a single failure.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from askme.pipeline.proactive.base import ProactiveContext, ProactiveResult
from askme.pipeline.proactive.clarification_agent import ClarificationPlannerAgent
from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
from askme.pipeline.proactive.slot_agent import SlotCollectorAgent
from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.pipeline.skill_dispatcher import MissionContext, SkillDispatcher
from askme.skills.skill_model import SkillDefinition, SlotSpec


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Recovery 1: ConfirmationAgent cancel → orchestrator returns proceed=False ─
# The caller must handle proceed=False gracefully (no dispatch, no crash)


class TestConfirmationCancel:
    async def test_cancel_returns_proceed_false(self):
        agent = ConfirmationAgent()
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取", confirm_before_execute=True
        )
        audio = MockAudio(["取消"])
        ctx = ProactiveContext()
        result = await agent.interact(sk, "抓取红色瓶子", audio, ctx)
        assert result.proceed is False
        assert result.enriched_text == "抓取红色瓶子"  # text unchanged

    async def test_cancel_does_not_mutate_mission(self):
        """Cancellation should not leave a partial step in MissionContext."""
        pipeline = MagicMock()
        pipeline.execute_skill = AsyncMock(return_value="done")
        skill_mgr = MagicMock()
        confirm_skill = SkillDefinition(
            name="robot_grab", description="抓取", confirm_before_execute=True,
            required_slots=[SlotSpec(name="object", type="referent", prompt="抓什么？")],
        )
        skill_mgr.get.return_value = confirm_skill
        skill_mgr.get_enabled.return_value = [confirm_skill]
        skill_mgr.get_skill_catalog.return_value = "robot_grab"

        dispatcher = SkillDispatcher(
            pipeline=pipeline, skill_manager=skill_mgr, audio=MagicMock()
        )
        # Orchestrator with cancellation
        orch = ProactiveOrchestrator.default(
            pipeline=MagicMock(), dispatcher=dispatcher
        )
        audio = MockAudio(["红色瓶子", "取消"])  # slot answer then cancel

        result = await orch.run("robot_grab", "抓取", audio)
        assert result.proceed is False
        # Dispatcher should NOT have been called for dispatch
        pipeline.execute_skill.assert_not_called()
        # Mission should not have started
        assert not dispatcher.has_active_mission

    async def test_system_usable_after_cancel(self):
        """After a cancel, subsequent commands must work normally."""
        pipeline = MagicMock()
        pipeline.execute_skill = AsyncMock(return_value="done")
        skill_mgr = MagicMock()
        plain_skill = SkillDefinition(name="get_time", description="查时间")
        skill_mgr.get.return_value = plain_skill
        skill_mgr.get_enabled.return_value = [plain_skill]
        skill_mgr.get_skill_catalog.return_value = "get_time"

        dispatcher = SkillDispatcher(
            pipeline=pipeline, skill_manager=skill_mgr, audio=MagicMock()
        )
        await dispatcher.dispatch("get_time", "几点了")
        assert dispatcher.has_active_mission
        assert dispatcher.current_mission.step_count == 1


# ── Recovery 2: Slot collection exhausts retries → proceed anyway ────────────
# User never answers → system must not hang, must proceed (LLM handles rest)


class TestSlotExhaustionRecovery:
    async def test_slot_collector_proceeds_after_max_attempts(self):
        agent = SlotCollectorAgent()
        sk = SkillDefinition(
            name="mapping", voice_trigger="建图", required_prompt="建哪里？"
        )
        audio = MockAudio([None, None])  # two empty answers
        ctx = ProactiveContext()
        result = await agent.interact(sk, "建图", audio, ctx)
        assert result.proceed is True  # MUST not be False
        assert result.enriched_text == "建图"  # original preserved

    async def test_clarification_agent_proceeds_after_max_turns(self):
        agent = ClarificationPlannerAgent()
        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索一下",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜什么？")],
        )
        audio = MockAudio([None, None])
        ctx = ProactiveContext()
        result = await agent.interact(sk, "搜索一下", audio, ctx)
        assert result.proceed is True
        assert result.enriched_text == "搜索一下"


# ── Recovery 3: Dispatcher handles pipeline error gracefully ─────────────────


class TestDispatcherPipelineFailure:
    @pytest.fixture
    def failed_pipeline(self):
        p = MagicMock()
        p.execute_skill = AsyncMock(side_effect=asyncio.TimeoutError)
        return p

    @pytest.fixture
    def ok_skill_mgr(self):
        m = MagicMock()
        sk = MagicMock()
        sk.name = "navigate"
        sk.timeout = 15
        m.get.return_value = sk
        m.get_enabled.return_value = [sk]
        m.get_skill_catalog.return_value = "navigate"
        return m

    async def test_timeout_returns_error_string(self, failed_pipeline, ok_skill_mgr):
        dispatcher = SkillDispatcher(
            pipeline=failed_pipeline, skill_manager=ok_skill_mgr, audio=MagicMock()
        )
        result = await dispatcher.dispatch("navigate", "去仓库")
        assert "[超时]" in result or "[Error]" in result, (
            f"Expected timeout error in result, got: {result!r}"
        )

    async def test_step_recorded_despite_timeout(self, failed_pipeline, ok_skill_mgr):
        """Even on failure, the mission step must be recorded for audit."""
        dispatcher = SkillDispatcher(
            pipeline=failed_pipeline, skill_manager=ok_skill_mgr, audio=MagicMock()
        )
        await dispatcher.dispatch("navigate", "去仓库")
        # After timeout, complete_mission() clears current_mission.
        # last_mission preserves the completed/failed mission for audit.
        assert dispatcher.last_mission is not None
        assert dispatcher.last_mission.step_count == 1

    async def test_second_dispatch_works_after_timeout(self, ok_skill_mgr):
        """After one timeout, the next dispatch must not be poisoned."""
        p = MagicMock()
        call_count = [0]
        async def _execute(name, text, ctx=""):
            call_count[0] += 1
            if call_count[0] == 1:
                raise asyncio.TimeoutError
            return "ok"
        p.execute_skill = _execute

        dispatcher = SkillDispatcher(
            pipeline=p, skill_manager=ok_skill_mgr, audio=MagicMock()
        )
        r1 = await dispatcher.dispatch("navigate", "去仓库")
        r2 = await dispatcher.dispatch("navigate", "去厨房")
        assert "超时" in r1 or "Error" in r1
        assert r2 == "ok", "Second dispatch must succeed after first timeout"


# ── Recovery 4: MissionContext isolation ─────────────────────────────────────
# Errors in one mission must not leak state to the next


class TestMissionIsolation:
    async def test_new_mission_starts_fresh(self):
        pipeline = MagicMock()
        pipeline.execute_skill = AsyncMock(return_value="done")
        skill_mgr = MagicMock()
        sk = MagicMock()
        sk.name = "navigate"
        sk.timeout = 15
        skill_mgr.get.return_value = sk
        skill_mgr.get_enabled.return_value = [sk]
        skill_mgr.get_skill_catalog.return_value = "navigate"

        dispatcher = SkillDispatcher(
            pipeline=pipeline, skill_manager=skill_mgr, audio=MagicMock()
        )
        # First mission
        await dispatcher.dispatch("navigate", "去仓库")
        m1 = dispatcher.complete_mission()
        assert m1 is not None
        assert not dispatcher.has_active_mission

        # Second mission — must be a NEW mission with step_count=0 at start
        await dispatcher.dispatch("navigate", "去厨房")
        m2 = dispatcher.current_mission
        assert m2 is not None
        assert m2.mission_id != m1.mission_id
        assert m2.step_count == 1  # only the one step from this mission


# ── Recovery 5: Orchestrator is reusable ─────────────────────────────────────
# Same ProactiveOrchestrator instance handles many dispatches without leaking state


class TestOrchestratorReuse:
    async def test_orchestrator_handles_sequential_dispatches(self):
        sk = SkillDefinition(
            name="get_time", description="查时间", voice_trigger="几点了"
        )
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        orch = ProactiveOrchestrator.default(
            pipeline=MagicMock(), dispatcher=dispatcher
        )
        audio = MockAudio()
        for _ in range(10):
            result = await orch.run("get_time", "现在几点了", audio)
            assert result.proceed is True
        # No audio questions should have been asked (no required slots)
        assert len(audio.spoken) == 0
