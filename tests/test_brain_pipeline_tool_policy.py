from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_general_chat_only_exposes_normal_safety_tools():
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.tools.tool_registry import BaseTool, ToolRegistry

    class _SafeTool(BaseTool):
        name = "safe_tool"
        description = "safe tool"
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        safety_level = "normal"

        def execute(self, **kwargs: Any) -> str:
            return "safe"

    class _DangerousTool(BaseTool):
        name = "dangerous_tool"
        description = "dangerous tool"
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        safety_level = "dangerous"

        def execute(self, **kwargs: Any) -> str:
            return "dangerous"

    llm = AsyncMock()
    captured: dict[str, list[dict[str, Any]]] = {}

    async def fake_stream(messages, **kwargs):
        captured["tools"] = kwargs.get("tools", [])
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "reply"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    llm.chat_stream = fake_stream

    conversation = MagicMock()
    conversation.history = []
    conversation.maybe_compress = AsyncMock()
    conversation.get_messages.return_value = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]

    memory = AsyncMock()
    memory.retrieve = AsyncMock(return_value="")

    tools = ToolRegistry()
    tools.register(_SafeTool())
    tools.register(_DangerousTool())

    skill_manager = MagicMock()
    skill_manager.get_skill_catalog.return_value = "none"
    skill_executor = MagicMock()
    audio = MagicMock()
    splitter = MagicMock()
    splitter.reset.return_value = None
    splitter.feed.return_value = []
    splitter.flush.return_value = None

    pipeline = BrainPipeline(
        llm=llm,
        conversation=conversation,
        memory=memory,
        tools=tools,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        splitter=splitter,
    )

    await pipeline.process("hello")

    tool_names = [tool["function"]["name"] for tool in captured["tools"]]
    assert tool_names == ["safe_tool"]


@pytest.mark.asyncio
async def test_pending_high_risk_input_is_handled_without_llm():
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.tools.tool_registry import BaseTool, ToolRegistry

    class _DangerousTool(BaseTool):
        name = "dangerous_tool"
        description = "dangerous tool"
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        safety_level = "dangerous"

        def execute(self, **kwargs: Any) -> str:
            return "dangerous"

    llm = AsyncMock()
    llm.chat_stream = AsyncMock()

    conversation = MagicMock()
    conversation.history = []

    memory = AsyncMock()
    memory.retrieve = AsyncMock(return_value="")

    tools = ToolRegistry(
        config={
            "default_timeout": 0.2,
            "dangerous_timeout": 0.2,
            "critical_timeout": 0.2,
            "timeout_cooldown": 0.0,
            "approval_timeout_seconds": 30.0,
            "require_confirmation_levels": ["dangerous", "critical"],
            "confirmation_phrases": ["approve", "confirm"],
            "rejection_phrases": ["cancel", "deny"],
        }
    )
    tools.register(_DangerousTool())
    tools.execute("dangerous_tool", '{"target": "bin-a"}', max_safety_level="dangerous")

    skill_manager = MagicMock()
    skill_manager.get_skill_catalog.return_value = "none"
    skill_executor = MagicMock()
    audio = MagicMock()
    splitter = MagicMock()

    pipeline = BrainPipeline(
        llm=llm,
        conversation=conversation,
        memory=memory,
        tools=tools,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        splitter=splitter,
    )

    reply = await pipeline.handle_pending_tool_response("status update")

    assert reply is not None
    assert reply.startswith("[Approval Pending]")
    assert "dangerous_tool" in reply
    llm.chat_stream.assert_not_called()
    audio.speak.assert_called_once_with(reply)
    conversation.add_user_message.assert_called_once_with("status update")
    conversation.add_assistant_message.assert_called_once_with(reply)
