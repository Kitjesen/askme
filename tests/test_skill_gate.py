"""Tests for SkillGate — skill execution gate with safety checks and routing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.skill_gate import SkillGate


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_skill_def(
    name: str = "patrol",
    execution: str = "skill_executor",
    enabled: bool = True,
    depends: list[str] | None = None,
) -> MagicMock:
    skill = MagicMock()
    skill.name = name
    skill.execution = execution
    skill.enabled = enabled
    skill.depends = depends or []
    return skill


def _make_gate(
    *,
    skill=None,
    cancel_token: asyncio.Event | None = None,
    hooks=None,
    dog_safety=None,
    agent_shell=None,
    executor_result: str = "Done.",
    max_response_chars: int = 500,
) -> SkillGate:
    skill_manager = MagicMock()
    skill_manager.get.return_value = skill

    skill_executor = MagicMock()
    skill_executor.execute = AsyncMock(return_value=executor_result)

    audio = MagicMock()
    audio.speak = MagicMock()
    audio.wait_speaking_done = MagicMock()
    audio.play_thinking = MagicMock()
    audio.drain_buffers = MagicMock()
    audio.start_playback = MagicMock()
    audio.stop_playback = MagicMock()

    conversation = MagicMock()

    gate = SkillGate(
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        conversation=conversation,
        dog_safety=dog_safety,
        cancel_token=cancel_token,
        hooks=hooks,
        agent_shell=agent_shell,
        max_response_chars=max_response_chars,
    )
    return gate


# ── TestCancelToken ───────────────────────────────────────────────────────────

class TestCancelToken:
    async def test_cancel_token_set_returns_empty(self):
        token = asyncio.Event()
        token.set()
        gate = _make_gate(cancel_token=token)
        result = await gate.execute_skill("patrol", "go patrol")
        assert result == ""

    async def test_cancel_token_not_set_proceeds(self):
        token = asyncio.Event()
        skill = _make_skill_def("patrol")
        gate = _make_gate(cancel_token=token, skill=skill)
        result = await gate.execute_skill("patrol", "go patrol")
        assert result == "Done."

    async def test_no_cancel_token_proceeds_normally(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(cancel_token=None, skill=skill)
        result = await gate.execute_skill("patrol", "go patrol")
        assert result == "Done."


# ── TestSkillNotFound ─────────────────────────────────────────────────────────

class TestSkillNotFound:
    async def test_returns_error_message_for_missing_skill(self):
        gate = _make_gate(skill=None)
        result = await gate.execute_skill("ghost_skill", "trigger")
        assert "ghost_skill" in result

    async def test_error_message_contains_not_found(self):
        gate = _make_gate(skill=None)
        result = await gate.execute_skill("missing", "text")
        assert "Not found" in result or "missing" in result


# ── TestEstop ─────────────────────────────────────────────────────────────────

class TestEstop:
    async def test_estop_active_blocks_skill(self):
        skill = _make_skill_def("navigate")
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.query_estop_state = MagicMock(return_value={"enabled": True})
        gate = _make_gate(skill=skill, dog_safety=dog_safety)
        result = await gate.execute_skill("navigate", "go")
        assert "急停" in result or "安全锁定" in result

    async def test_estop_inactive_allows_skill(self):
        skill = _make_skill_def("navigate")
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.query_estop_state = MagicMock(return_value={"enabled": False})
        gate = _make_gate(skill=skill, dog_safety=dog_safety)
        result = await gate.execute_skill("navigate", "go")
        assert result == "Done."

    async def test_estop_none_state_allows_skill(self):
        skill = _make_skill_def("navigate")
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.query_estop_state = MagicMock(return_value=None)
        gate = _make_gate(skill=skill, dog_safety=dog_safety)
        result = await gate.execute_skill("navigate", "go")
        assert result == "Done."

    async def test_unconfigured_safety_skips_estop_check(self):
        skill = _make_skill_def("navigate")
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = False
        gate = _make_gate(skill=skill, dog_safety=dog_safety)
        result = await gate.execute_skill("navigate", "go")
        assert result == "Done."
        dog_safety.query_estop_state.assert_not_called()


# ── TestSkillExecution ────────────────────────────────────────────────────────

class TestSkillExecution:
    async def test_successful_execution_returns_result(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill, executor_result="Patrol complete.")
        result = await gate.execute_skill("patrol", "start patrol")
        assert result == "Patrol complete."

    async def test_conversation_updated_after_execution(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill, executor_result="Done.")
        await gate.execute_skill("patrol", "go patrol")
        gate._conversation.add_user_message.assert_called_once_with("go patrol")
        gate._conversation.add_assistant_message.assert_called_once()

    async def test_audio_speak_called(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill, executor_result="Result text.")
        await gate.execute_skill("patrol", "patrol now", source="text")
        gate._audio.speak.assert_called_once_with("Result text.")

    async def test_last_spoken_text_updated(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill, executor_result="Spoken output.")
        await gate.execute_skill("patrol", "go", source="text")
        assert gate.last_spoken_text == "Spoken output."

    async def test_think_blocks_stripped_from_result(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(
            skill=skill,
            executor_result="<think>reasoning</think>Final answer."
        )
        result = await gate.execute_skill("patrol", "go", source="text")
        assert "<think>" not in result
        assert "Final answer." in result

    async def test_skill_exception_returns_error_string(self):
        skill = _make_skill_def("patrol")
        skill_executor = MagicMock()
        skill_executor.execute = AsyncMock(side_effect=RuntimeError("sensor fail"))
        gate = _make_gate(skill=skill)
        gate._skill_executor = skill_executor
        result = await gate.execute_skill("patrol", "go")
        assert "[Skill Error]" in result or "sensor fail" in result

    async def test_stop_playback_called_on_success(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill)
        await gate.execute_skill("patrol", "go", source="text")
        gate._audio.stop_playback.assert_called()

    async def test_stop_playback_called_on_exception(self):
        skill = _make_skill_def("patrol")
        skill_executor = MagicMock()
        skill_executor.execute = AsyncMock(side_effect=RuntimeError("boom"))
        gate = _make_gate(skill=skill)
        gate._skill_executor = skill_executor
        await gate.execute_skill("patrol", "go")
        gate._audio.stop_playback.assert_called()


# ── TestAgentShellRouting ─────────────────────────────────────────────────────

class TestAgentShellRouting:
    async def test_agent_shell_skill_routed_to_shell(self):
        skill = _make_skill_def("agent_task", execution="agent_shell")
        agent_shell = MagicMock()
        agent_shell.run_task = AsyncMock(return_value="Agent done.")
        gate = _make_gate(skill=skill, agent_shell=agent_shell)
        result = await gate.execute_skill("agent_task", "do the task", source="text")
        assert result == "Agent done."
        agent_shell.run_task.assert_called_once()

    async def test_agent_shell_result_in_conversation(self):
        skill = _make_skill_def("agent_task", execution="agent_shell")
        agent_shell = MagicMock()
        agent_shell.run_task = AsyncMock(return_value="Agent result.")
        gate = _make_gate(skill=skill, agent_shell=agent_shell)
        await gate.execute_skill("agent_task", "do task", source="text")
        gate._conversation.add_user_message.assert_called_once_with("do task")

    async def test_agent_shell_exception_returns_error(self):
        skill = _make_skill_def("agent_task", execution="agent_shell")
        agent_shell = MagicMock()
        agent_shell.run_task = AsyncMock(side_effect=RuntimeError("timeout"))
        gate = _make_gate(skill=skill, agent_shell=agent_shell)
        result = await gate.execute_skill("agent_task", "do task", source="text")
        assert "[AgentShell Error]" in result

    async def test_agent_shell_none_falls_through_to_executor(self):
        skill = _make_skill_def("agent_task", execution="agent_shell")
        # agent_shell=None → should fall through to skill executor
        gate = _make_gate(skill=skill, agent_shell=None, executor_result="Fallback.")
        result = await gate.execute_skill("agent_task", "do task", source="text")
        assert result == "Fallback."


# ── TestPrepareAgentResult ────────────────────────────────────────────────────

class TestPrepareAgentResult:
    def test_short_result_returned_unchanged(self):
        gate = _make_gate(max_response_chars=500)
        spoken, stored = gate._prepare_agent_result("Short result.")
        assert spoken == "Short result."
        assert stored == "Short result."

    def test_long_result_truncated_for_spoken(self):
        gate = _make_gate(max_response_chars=20)
        long_text = "This is a very long result that exceeds the limit."
        spoken, stored = gate._prepare_agent_result(long_text)
        assert len(spoken) <= len(long_text)
        assert stored == long_text  # stored is always full

    def test_sentence_boundary_preferred(self):
        gate = _make_gate(max_response_chars=30)
        # Contains sentence-ending punctuation before the limit
        text = "First sentence。Second part of long result here."
        spoken, stored = gate._prepare_agent_result(text)
        assert "。" in spoken or len(spoken) <= 30

    def test_spoken_has_workspace_note_when_truncated(self):
        gate = _make_gate(max_response_chars=10)
        long_text = "A" * 100
        spoken, _ = gate._prepare_agent_result(long_text)
        assert "工作区" in spoken

    def test_exact_limit_not_truncated(self):
        gate = _make_gate(max_response_chars=10)
        text = "1234567890"  # exactly 10 chars
        spoken, stored = gate._prepare_agent_result(text)
        assert spoken == text
        assert stored == text


# ── TestExtractSemanticTarget ─────────────────────────────────────────────────

class TestExtractSemanticTarget:
    def setup_method(self):
        self.gate = _make_gate()

    def test_daohang_pattern(self):
        result = self.gate.extract_semantic_target("导航到仓库A吧")
        assert result == "仓库A"

    def test_daiwoqu_pattern(self):
        result = self.gate.extract_semantic_target("带我去大厅")
        assert result == "大厅"

    def test_qianwang_pattern(self):
        result = self.gate.extract_semantic_target("前往充电站啊")
        assert result == "充电站"

    def test_zoudo_pattern(self):
        result = self.gate.extract_semantic_target("走到入口处嗯")
        assert result == "入口处"

    def test_qu_pattern(self):
        result = self.gate.extract_semantic_target("去走廊B")
        assert result == "走廊B"

    def test_no_pattern_returns_original(self):
        text = "start patrol mode"
        result = self.gate.extract_semantic_target(text)
        assert result == text

    def test_extracts_before_sentence_end(self):
        result = self.gate.extract_semantic_target("导航到房间3。")
        assert result == "房间3"


# ── TestPostToolHook ──────────────────────────────────────────────────────────

class TestPostToolHook:
    async def test_post_tool_hook_can_transform_result(self):
        from askme.pipeline.hooks import PipelineHooks
        hooks = PipelineHooks()

        async def transform(record):
            return "TRANSFORMED"

        hooks.on_post_tool(transform)
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill, hooks=hooks, executor_result="original")
        result = await gate.execute_skill("patrol", "go", source="text")
        assert result == "TRANSFORMED"

    async def test_no_hook_result_unchanged(self):
        skill = _make_skill_def("patrol")
        gate = _make_gate(skill=skill, hooks=None, executor_result="original")
        result = await gate.execute_skill("patrol", "go", source="text")
        assert result == "original"


# ── TestLateBindingSetters ────────────────────────────────────────────────────

class TestLateBindingSetters:
    def test_set_audio(self):
        gate = _make_gate()
        new_audio = MagicMock()
        gate.set_audio(new_audio)
        assert gate._audio is new_audio

    def test_set_skill_manager(self):
        gate = _make_gate()
        new_sm = MagicMock()
        gate.set_skill_manager(new_sm)
        assert gate._skill_manager is new_sm

    def test_set_skill_executor(self):
        gate = _make_gate()
        new_exec = MagicMock()
        gate.set_skill_executor(new_exec)
        assert gate._skill_executor is new_exec

    def test_set_agent_shell(self):
        gate = _make_gate()
        new_shell = MagicMock()
        gate.set_agent_shell(new_shell)
        assert gate._agent_shell is new_shell


# ── TestEpisodicLogging ───────────────────────────────────────────────────────

class TestEpisodicLogging:
    async def test_episodic_logs_action_and_outcome(self):
        skill = _make_skill_def("patrol")
        episodic = MagicMock()
        gate = _make_gate(skill=skill, executor_result="Done.")
        gate._episodic = episodic
        gate._mem = None
        await gate.execute_skill("patrol", "patrol now", source="text")
        assert episodic.log.call_count >= 1

    async def test_mem_system_preferred_over_episodic(self):
        skill = _make_skill_def("patrol")
        mem_system = MagicMock()
        gate = _make_gate(skill=skill, executor_result="Done.")
        gate._mem = mem_system
        gate._episodic = MagicMock()
        await gate.execute_skill("patrol", "go", source="text")
        mem_system.log_event.assert_called()
        gate._episodic.log.assert_not_called()
