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
