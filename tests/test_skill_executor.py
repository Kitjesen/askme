"""Tests for askme.skills.skill_executor safety gating."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import pytest
from askme.skills.skill_executor import SkillExecutor
from askme.skills.skill_model import SkillDefinition
from askme.tools.tool_registry import BaseTool, ToolRegistry

from askme.conversation import InteractionTurnContext
from askme.llm.core.contracts import LLMCallContext


class DangerousCommandTool(BaseTool):
    name = "run_command"
    description = "Dangerous shell execution"
    parameters = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def execute(self, **kwargs):
        return "command executed"


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Normal echo tool"
    parameters = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs):
        return "echo result"


def _tool_call_response(name: str, arguments: str = "{}"):
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _text_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeLLM:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeResilientLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self._responses.pop(0)


class _RecordingToolRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
        self.execution_contexts: list[InteractionTurnContext | None] = []
        self.pending_contexts: list[InteractionTurnContext | None] = []

    def execute(self, *args, interaction_context=None, **kwargs):
        self.execution_contexts.append(interaction_context)
        return super().execute(
            *args,
            interaction_context=interaction_context,
            **kwargs,
        )

    def has_pending_approval(self, interaction_context=None):
        self.pending_contexts.append(interaction_context)
        return super().has_pending_approval(interaction_context)


class _StrictLegacyToolRegistry:
    def __init__(self) -> None:
        self.execute_calls = 0
        self.pending_checks = 0

    def get_definitions(self, *, max_safety_level):
        assert max_safety_level == "normal"
        return [
            {
                "type": "function",
                "function": {
                    "name": "echo_tool",
                    "description": "Normal echo tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def execute(
        self,
        name,
        args_json,
        *,
        allowed_names,
        max_safety_level,
    ):
        self.execute_calls += 1
        assert name == "echo_tool"
        assert args_json == "{}"
        assert allowed_names == {"echo_tool"}
        assert max_safety_level == "normal"
        return "echo result"

    def has_pending_approval(self):
        self.pending_checks += 1
        return False


@pytest.mark.asyncio
async def test_normal_skill_cannot_use_dangerous_tool():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    registry.register(DangerousCommandTool())
    llm = _FakeLLM(
        [
            _tool_call_response("run_command"),
            _text_response("fallback reply"),
        ]
    )
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="web_search",
        safety_level="normal",
        tools_section="run_command",
        prompt_template="Use tools if needed.",
    )

    result = await executor.execute(skill, {"user_input": "search logs"})

    assert result == "fallback reply"
    assert "tools" not in llm.completions.calls[0]
    tool_messages = [msg for msg in llm.completions.calls[1]["messages"] if msg["role"] == "tool"]
    assert "not enabled for this request" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_dangerous_skill_returns_approval_request_for_allowed_dangerous_tool():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    registry.register(DangerousCommandTool())
    llm = _FakeLLM([_tool_call_response("run_command")])
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="run_command",
        safety_level="dangerous",
        tools_section="run_command",
        prompt_template="Use tools if needed.",
    )

    result = await executor.execute(skill, {"user_input": "dir"})

    assert result.startswith("[Approval Required]")
    assert llm.completions.calls[0]["tools"][0]["function"]["name"] == "run_command"
    assert len(llm.completions.calls) == 1


@pytest.mark.asyncio
async def test_skill_executor_supports_llm_client_style_interface():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    registry.register(EchoTool())
    llm = _FakeResilientLLM(
        [
            _tool_call_response("echo_tool"),
            _text_response("done"),
        ]
    )
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="echo",
        safety_level="normal",
        tools_section="echo_tool",
        prompt_template="Use tools if needed.",
    )

    result = await executor.execute(skill, {"user_input": "echo this"})

    assert result == "done"
    assert llm.calls[0]["tools"][0]["function"]["name"] == "echo_tool"
    contexts = [call["context"] for call in llm.calls]
    assert all(isinstance(context, LLMCallContext) for context in contexts)
    assert all(context.purpose == "tool_followup" for context in contexts)
    assert contexts[0].call_id != contexts[1].call_id
    tool_messages = [msg for msg in llm.calls[1]["messages"] if msg["role"] == "tool"]
    assert tool_messages[0]["content"] == "echo result"


@pytest.mark.asyncio
async def test_skill_executor_projects_llm_context_into_every_tool_boundary():
    registry = _RecordingToolRegistry()
    registry.register(EchoTool())
    tool_response = _tool_call_response("echo_tool")
    tool_response.choices[0].message.tool_calls.append(
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="echo_tool", arguments="{}"),
        )
    )
    llm = _FakeResilientLLM(
        [
            tool_response,
            _text_response("done"),
        ]
    )
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="echo",
        safety_level="normal",
        tools_section="echo_tool",
        prompt_template="Use tools if needed.",
    )
    cancel_token = Event()
    llm_context = LLMCallContext(
        session_id="session-7",
        turn_id="turn-9",
        channel="voice",
        operator_id="operator-3",
    )
    object.__setattr__(llm_context, "cancel_token", cancel_token)

    result = await executor.execute(
        skill,
        {"user_input": "echo this"},
        llm_call_context=llm_context,
    )

    assert result == "done"
    assert len(registry.execution_contexts) == 2
    assert registry.pending_contexts == registry.execution_contexts
    interaction_context = registry.execution_contexts[0]
    assert all(context is interaction_context for context in registry.execution_contexts)
    assert isinstance(interaction_context, InteractionTurnContext)
    assert interaction_context.thread_id == "session-7"
    assert interaction_context.turn_id == "turn-9"
    assert interaction_context.channel == "voice"
    assert interaction_context.source == "voice"
    assert interaction_context.operator_id == "operator-3"
    assert interaction_context.person_id is None
    assert interaction_context.cancel_token is cancel_token


@pytest.mark.asyncio
async def test_scoped_pending_check_does_not_fall_back_to_legacy_queue():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    registry.register(DangerousCommandTool())
    registry.register(EchoTool())
    queued = registry.execute(
        "run_command",
        "{}",
        max_safety_level="dangerous",
    )
    assert queued.startswith("[Approval Required]")
    assert registry.has_pending_approval() is True
    llm = _FakeResilientLLM(
        [
            _tool_call_response("echo_tool"),
            _text_response("done"),
        ]
    )
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="echo",
        safety_level="normal",
        tools_section="echo_tool",
        prompt_template="Use tools if needed.",
    )

    result = await executor.execute(
        skill,
        {"user_input": "echo this"},
        llm_call_context=LLMCallContext(
            session_id="session-scoped",
            turn_id="turn-scoped",
            channel="text",
        ),
    )

    assert result == "done"
    assert len(llm.calls) == 2
    assert registry.has_pending_approval() is True


@pytest.mark.asyncio
async def test_anonymous_voice_skill_hits_dangerous_tool_fail_closed():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    registry.register(DangerousCommandTool())
    llm = _FakeResilientLLM(
        [
            _tool_call_response("run_command"),
            _text_response("blocked"),
        ]
    )
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="run_command",
        safety_level="dangerous",
        tools_section="run_command",
        prompt_template="Use tools if needed.",
    )

    result = await executor.execute(
        skill,
        {"user_input": "dir"},
        llm_call_context=LLMCallContext(
            session_id="voice-session",
            turn_id="voice-turn",
            channel="voice",
        ),
    )

    assert result == "blocked"
    tool_messages = [message for message in llm.calls[1]["messages"] if message["role"] == "tool"]
    assert "需要已认证操作员" in tool_messages[0]["content"]
    assert registry.has_pending_approval() is False


@pytest.mark.asyncio
async def test_skill_executor_preserves_strict_legacy_tool_registry_signatures():
    registry = _StrictLegacyToolRegistry()
    llm = _FakeResilientLLM(
        [
            _tool_call_response("echo_tool"),
            _text_response("done"),
        ]
    )
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="echo",
        safety_level="normal",
        tools_section="echo_tool",
        prompt_template="Use tools if needed.",
    )

    result = await executor.execute(
        skill,
        {"user_input": "echo this"},
        llm_call_context=LLMCallContext(
            session_id="session-legacy",
            turn_id="turn-legacy",
            channel="text",
        ),
    )

    assert result == "done"
    assert registry.execute_calls == 1
    assert registry.pending_checks == 1


@pytest.mark.asyncio
async def test_skill_executor_returns_error_when_llm_fails():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    llm = _FakeResilientLLM([])

    async def _boom(messages, **kwargs):
        raise RuntimeError("relay unavailable")

    llm.chat_completion = _boom
    executor = SkillExecutor(llm, registry)
    skill = SkillDefinition(
        name="echo",
        safety_level="normal",
        prompt_template="Reply directly.",
    )

    result = await executor.execute(skill, {"user_input": "echo this"})

    assert result == "[Error] Skill 'echo' execution failed: relay unavailable"


def test_build_prompt_strips_unresolved_placeholders() -> None:
    """Unresolved {{vars}} should not reach the LLM as literal template syntax."""
    skill = SkillDefinition(
        name="patrol_report",
        prompt_template="Time: {{current_time}}\nData: {{patrol_data}}\nInput: {{user_input}}",
    )
    result = skill.build_prompt({"current_time": "2026-01-01 12:00:00"})
    assert "{{" not in result
    assert "2026-01-01 12:00:00" in result
    assert "Data: " in result  # placeholder stripped, label preserved
    assert "Input: " in result  # placeholder stripped, label preserved


def test_build_prompt_with_full_context() -> None:
    """All placeholders should be substituted when context is complete."""
    skill = SkillDefinition(
        name="test",
        prompt_template="Hello {{name}}, time is {{time}}.",
    )
    result = skill.build_prompt({"name": "Thunder", "time": "noon"})
    assert result == "Hello Thunder, time is noon."
