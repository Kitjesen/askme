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
