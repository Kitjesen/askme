"""Tests for BrainPipeline handle_estop, reset_estop, and injectable protocol path."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.hooks import PipelineHooks

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_injectable_pipeline(
    *,
    cancel_token=None,
    hooks=None,
    arm=None,
    dog_safety=None,
):
    """Create a BrainPipeline using the protocol injection path (no internal construction)."""
    stream_processor = MagicMock()
    stream_processor.set_audio = MagicMock()
    stream_processor.reset = MagicMock()

    skill_gate = MagicMock()
    skill_gate.last_spoken_text = ""
    skill_gate.set_audio = MagicMock()
    skill_gate.set_skill_manager = MagicMock()
    skill_gate.set_skill_executor = MagicMock()
    skill_gate.set_agent_shell = MagicMock()

    turn_executor = MagicMock()
    turn_executor.last_spoken_text = ""
    turn_executor.process = AsyncMock(return_value="ok")
    turn_executor.set_audio = MagicMock()
    turn_executor.shutdown = AsyncMock()

    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.has_pending_approval.return_value = False

    pipeline = BrainPipeline(
        llm=MagicMock(),
        conversation=MagicMock(),
        memory=MagicMock(),
        tools=tools,
        skill_manager=MagicMock(),
        skill_executor=MagicMock(),
        audio=MagicMock(),
        splitter=MagicMock(),
        # protocol injection
        stream_processor=stream_processor,
        skill_gate=skill_gate,
        turn_executor=turn_executor,
        cancel_token=cancel_token,
        hooks=hooks,
        arm_controller=arm,
        dog_safety_client=dog_safety,
    )
    return pipeline


# ── handle_estop ──────────────────────────────────────────────────────────────

class TestHandleEstop:
    def test_sets_cancel_token(self):
        token = asyncio.Event()
        pipeline = _make_injectable_pipeline(cancel_token=token)
        assert not token.is_set()
        pipeline.handle_estop()
        assert token.is_set()

    def test_calls_arm_emergency_stop(self):
        arm = MagicMock()
        pipeline = _make_injectable_pipeline(arm=arm)
        pipeline.handle_estop()
        arm.emergency_stop.assert_called_once()

    def test_no_arm_no_crash(self):
        pipeline = _make_injectable_pipeline()
        pipeline.handle_estop()  # should not raise

    def test_notifies_dog_safety(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        pipeline = _make_injectable_pipeline(dog_safety=dog_safety)
        pipeline.handle_estop()
        dog_safety.notify_estop.assert_called_once()

    def test_skips_dog_safety_when_not_configured(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = False
        pipeline = _make_injectable_pipeline(dog_safety=dog_safety)
        pipeline.handle_estop()
        dog_safety.notify_estop.assert_not_called()

    def test_fires_estop_hooks(self):
        hooks = PipelineHooks()
        fired = []
        hooks.on_estop(lambda: fired.append(True))
        pipeline = _make_injectable_pipeline(hooks=hooks)
        pipeline.handle_estop()
        assert len(fired) == 1

    def test_no_hooks_no_crash(self):
        pipeline = _make_injectable_pipeline(hooks=None)
        pipeline.handle_estop()


# ── reset_estop ───────────────────────────────────────────────────────────────

class TestResetEstop:
    def test_clears_cancel_token(self):
        token = asyncio.Event()
        token.set()
        pipeline = _make_injectable_pipeline(cancel_token=token)
        pipeline.reset_estop()
        assert not token.is_set()

    def test_noop_when_not_set(self):
        token = asyncio.Event()
        pipeline = _make_injectable_pipeline(cancel_token=token)
        pipeline.reset_estop()  # should not raise; no-op
        assert not token.is_set()


# ── has_pending_tool_approval ─────────────────────────────────────────────────

class TestPendingToolApproval:
    def test_delegates_to_tools(self):
        pipeline = _make_injectable_pipeline()
        pipeline._tools.has_pending_approval.return_value = True
        assert pipeline.has_pending_tool_approval() is True

    def test_false_by_default(self):
        pipeline = _make_injectable_pipeline()
        pipeline._tools.has_pending_approval.return_value = False
        assert pipeline.has_pending_tool_approval() is False


# ── process delegation ────────────────────────────────────────────────────────

class TestProcessDelegation:
    @pytest.mark.asyncio
    async def test_process_delegates_to_turn_executor(self):
        pipeline = _make_injectable_pipeline()
        pipeline._turn_executor.process = AsyncMock(return_value="response text")
        result = await pipeline.process("hello")
        assert result == "response text"

    @pytest.mark.asyncio
    async def test_execute_skill_delegates_to_skill_gate(self):
        pipeline = _make_injectable_pipeline()
        pipeline._skill_gate.execute_skill = AsyncMock(return_value="skill result")
        result = await pipeline.execute_skill("navigate", "去仓库A")
        assert result == "skill result"


# ── set_audio / set_skill_manager ─────────────────────────────────────────────

class TestTurnScopedCancellation:
    @pytest.mark.asyncio
    async def test_external_voice_token_closes_pre_ownership_cancel_window(self):
        import threading

        pipeline = _make_injectable_pipeline()
        token = threading.Event()
        token.set()

        result = await pipeline.process("hello", turn_cancel_token=token)

        assert result == ""
        pipeline._turn_executor.process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_external_voice_token_is_wrapped_by_the_active_turn_lease(self):
        import threading

        pipeline = _make_injectable_pipeline()
        token = threading.Event()
        observed = []

        async def _process(user_text, **kwargs):
            del user_text
            observed.append(kwargs["turn_cancel_token"])
            return "ok"

        pipeline._turn_executor.process = AsyncMock(side_effect=_process)

        assert await pipeline.process("hello", turn_cancel_token=token) == "ok"
        assert len(observed) == 1
        assert observed[0].cancel_event is token
        assert observed[0].is_set() is False

    @pytest.mark.asyncio
    async def test_barge_in_cancels_only_the_active_turn(self):
        pipeline = _make_injectable_pipeline()
        started = asyncio.Event()
        observed: list[tuple[str, int, asyncio.Event]] = []

        async def _process(
            user_text,
            *,
            memory_task=None,
            source="voice",
            conversation_session_id=None,
            voice_turn_id=None,
            turn_epoch=None,
            turn_cancel_token=None,
        ):
            del user_text, memory_task, source, conversation_session_id
            observed.append((voice_turn_id, turn_epoch, turn_cancel_token))
            started.set()
            while not turn_cancel_token.is_set():
                await asyncio.sleep(0.01)
            return ""

        pipeline._turn_executor.process = AsyncMock(side_effect=_process)
        task = asyncio.create_task(
            pipeline.process("hello", voice_turn_id="voice-turn-1")
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        assert pipeline.cancel_active_turn(reason="barge_in") is True
        assert pipeline._cancel_token.is_set() is False
        assert await asyncio.wait_for(task, timeout=1) == ""
        assert observed[0][0] == "voice-turn-1"
        assert observed[0][1] == 1
        assert observed[0][2].is_set() is True

    def test_barge_in_without_active_turn_is_a_noop(self):
        pipeline = _make_injectable_pipeline()

        assert pipeline.cancel_active_turn(reason="barge_in") is False
        assert pipeline._cancel_token.is_set() is False


class TestSetters:
    def test_set_audio_wires_sub_components(self):
        pipeline = _make_injectable_pipeline()
        new_audio = MagicMock()
        pipeline.set_audio(new_audio)
        pipeline._stream_processor.set_audio.assert_called_with(new_audio)
        pipeline._skill_gate.set_audio.assert_called_with(new_audio)
        pipeline._turn_executor.set_audio.assert_called_with(new_audio)

    def test_set_skill_manager(self):
        pipeline = _make_injectable_pipeline()
        mgr = MagicMock()
        pipeline.set_skill_manager(mgr)
        pipeline._skill_gate.set_skill_manager.assert_called_with(mgr)
