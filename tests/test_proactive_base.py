"""Tests for proactive base types and ProactiveOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.pipeline.proactive.base import (
    ESTOP_SIGNALS,
    ProactiveAgent,
    ProactiveContext,
    ProactiveResult,
    ask_and_listen,
)
from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.skills.skill_model import SkillDefinition


# ── ProactiveResult ───────────────────────────────────────────────────────────

class TestProactiveResult:
    def test_defaults(self):
        r = ProactiveResult(enriched_text="hello")
        assert r.proceed is True
        assert r.cancelled_by == ""
        assert r.interrupt_payload == ""

    def test_custom_values(self):
        r = ProactiveResult(enriched_text="abc", proceed=False, cancelled_by="agent_x")
        assert r.proceed is False
        assert r.cancelled_by == "agent_x"


# ── ProactiveContext ──────────────────────────────────────────────────────────

class TestProactiveContext:
    def test_defaults(self):
        ctx = ProactiveContext()
        assert ctx.pipeline is None
        assert ctx.dispatcher is None
        assert ctx.source == "voice"
        assert ctx.session is None

    def test_custom_values(self):
        ctx = ProactiveContext(source="text")
        assert ctx.source == "text"


# ── ESTOP_SIGNALS ─────────────────────────────────────────────────────────────

class TestEstopSignals:
    def test_is_frozenset(self):
        assert isinstance(ESTOP_SIGNALS, frozenset)

    def test_contains_chinese_estop(self):
        assert "急停" in ESTOP_SIGNALS

    def test_contains_english_estop(self):
        assert "estop" in ESTOP_SIGNALS


# ── ask_and_listen ────────────────────────────────────────────────────────────

class TestAskAndListen:
    @pytest.mark.asyncio
    async def test_returns_transcription(self):
        audio = MagicMock()
        audio.listen_loop.return_value = "好的"
        result = await ask_and_listen("请说话", audio)
        assert result == "好的"

    @pytest.mark.asyncio
    async def test_sets_awaiting_confirmation(self):
        """Ensures awaiting_confirmation is set True then reset to False."""
        states = []
        audio = MagicMock()

        def listen_and_record():
            states.append(audio.awaiting_confirmation)
            return "收到"

        audio.listen_loop.side_effect = listen_and_record
        await ask_and_listen("问题", audio)
        assert states == [True]
        assert audio.awaiting_confirmation is False

    @pytest.mark.asyncio
    async def test_drain_buffers_failure_does_not_raise(self):
        audio = MagicMock()
        audio.drain_buffers.side_effect = OSError("buffer error")
        audio.listen_loop.return_value = "yes"
        result = await ask_and_listen("question", audio)
        assert result == "yes"

    @pytest.mark.asyncio
    async def test_tts_failure_still_returns_listen_result(self):
        audio = MagicMock()
        audio.speak.side_effect = OSError("tts error")
        audio.listen_loop.return_value = "text"
        result = await ask_and_listen("question", audio)
        assert result == "text"

    @pytest.mark.asyncio
    async def test_listen_failure_returns_none(self):
        audio = MagicMock()
        audio.listen_loop.side_effect = OSError("mic error")
        result = await ask_and_listen("question", audio)
        assert result is None


# ── ProactiveOrchestrator ─────────────────────────────────────────────────────

class TestProactiveOrchestrator:
    @pytest.mark.asyncio
    async def test_no_dispatcher_skips_chain(self):
        orch = ProactiveOrchestrator(agents=[], pipeline=None, dispatcher=None)
        result = await orch.run("some_skill", "user text", audio=MagicMock())
        assert result.enriched_text == "user text"
        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_skill_not_found_skips_chain(self):
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = None
        orch = ProactiveOrchestrator(agents=[], dispatcher=dispatcher)
        result = await orch.run("missing_skill", "hello", audio=MagicMock())
        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_agents_run_in_order(self):
        order = []

        class FakeAgent(ProactiveAgent):
            def __init__(self, name):
                self._name = name

            def should_activate(self, skill, text, ctx):
                return True

            async def interact(self, skill, text, audio, ctx):
                order.append(self._name)
                return ProactiveResult(enriched_text=text + f"+{self._name}")

        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = SkillDefinition(name="test")
        orch = ProactiveOrchestrator(
            agents=[FakeAgent("A"), FakeAgent("B")],
            dispatcher=dispatcher,
        )
        result = await orch.run("test", "start", audio=MagicMock())
        assert order == ["A", "B"]
        assert "start+A+B" == result.enriched_text

    @pytest.mark.asyncio
    async def test_chain_short_circuits_on_cancel(self):
        order = []

        class CancelAgent(ProactiveAgent):
            def should_activate(self, skill, text, ctx):
                return True

            async def interact(self, skill, text, audio, ctx):
                order.append("cancel")
                return ProactiveResult(enriched_text=text, proceed=False, cancelled_by="cancel")

        class NeverRunAgent(ProactiveAgent):
            def should_activate(self, skill, text, ctx):
                return True

            async def interact(self, skill, text, audio, ctx):
                order.append("never")
                return ProactiveResult(enriched_text=text)

        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = SkillDefinition(name="test")
        orch = ProactiveOrchestrator(
            agents=[CancelAgent(), NeverRunAgent()],
            dispatcher=dispatcher,
        )
        result = await orch.run("test", "hi", audio=MagicMock())
        assert result.proceed is False
        assert "never" not in order

    @pytest.mark.asyncio
    async def test_agent_not_activated_is_skipped(self):
        class InactiveAgent(ProactiveAgent):
            def should_activate(self, skill, text, ctx):
                return False

            async def interact(self, skill, text, audio, ctx):
                raise RuntimeError("should not be called")

        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = SkillDefinition(name="test")
        orch = ProactiveOrchestrator(agents=[InactiveAgent()], dispatcher=dispatcher)
        result = await orch.run("test", "input", audio=MagicMock())
        assert result.proceed is True
        assert result.enriched_text == "input"

    def test_default_creates_three_agents(self):
        orch = ProactiveOrchestrator.default()
        assert len(orch._agents) == 3

    @pytest.mark.asyncio
    async def test_enriched_text_flows_between_agents(self):
        class AddSuffix(ProactiveAgent):
            def __init__(self, suffix):
                self._suffix = suffix

            def should_activate(self, skill, text, ctx):
                return True

            async def interact(self, skill, text, audio, ctx):
                return ProactiveResult(enriched_text=text + self._suffix)

        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = SkillDefinition(name="nav")
        orch = ProactiveOrchestrator(
            agents=[AddSuffix(" [A]"), AddSuffix(" [B]")],
            dispatcher=dispatcher,
        )
        result = await orch.run("nav", "go", audio=MagicMock())
        assert result.enriched_text == "go [A] [B]"
