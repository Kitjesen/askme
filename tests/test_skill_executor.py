"""Tests for askme.skills.skill_executor safety gating."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from askme.skills.skill_executor import SkillExecutor
from askme.skills.skill_model import SkillDefinition
from askme.tools.tool_registry import BaseTool, ToolRegistry


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


@pytest.mark.asyncio
async def test_normal_skill_cannot_use_dangerous_tool():
    registry = ToolRegistry(config={"default_timeout": 1.0, "timeout_cooldown": 0.0})
    registry.register(DangerousCommandTool())
    llm = _FakeLLM([
        _tool_call_response("run_command"),
        _text_response("fallback reply"),
    ])
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
    tool_messages = [
        msg for msg in llm.completions.calls[1]["messages"]
        if msg["role"] == "tool"
    ]
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
    llm = _FakeResilientLLM([
        _tool_call_response("echo_tool"),
        _text_response("done"),
    ])
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
    tool_messages = [
        msg for msg in llm.calls[1]["messages"]
        if msg["role"] == "tool"
    ]
    assert tool_messages[0]["content"] == "echo result"


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
    assert "Data: " in result   # placeholder stripped, label preserved
    assert "Input: " in result  # placeholder stripped, label preserved


def test_build_prompt_with_full_context() -> None:
    """All placeholders should be substituted when context is complete."""
    skill = SkillDefinition(
        name="test",
        prompt_template="Hello {{name}}, time is {{time}}.",
    )
    result = skill.build_prompt({"name": "Thunder", "time": "noon"})
    assert result == "Hello Thunder, time is noon."
