"""Tests for MCP memory tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _import_tools():
    from askme.mcp.tools.memory_tools import memory_save, memory_search

    return memory_search, memory_save


def _make_ctx(*, memory_bridge=None, episodic_memory=None):
    app = MagicMock()
    app.memory_bridge = memory_bridge
    app.episodic_memory = episodic_memory
    ctx = MagicMock()
    ctx.request_context = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestMemorySearch:
    @pytest.mark.asyncio
    async def test_no_memory_bridge_returns_empty(self):
        memory_search, _ = _import_tools()
        ctx = _make_ctx(memory_bridge=None)

        result = await memory_search("test query", ctx=ctx)

        data = json.loads(result)
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_conversation_layer_returns_results(self):
        memory_search, _ = _import_tools()
        bridge = AsyncMock()
        bridge.retrieve = AsyncMock(return_value="- memory1\n- memory2")
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_search("query", layer="conversation", ctx=ctx)

        data = json.loads(result)
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_conversation_layer_strips_prefix(self):
        memory_search, _ = _import_tools()
        bridge = AsyncMock()
        bridge.retrieve = AsyncMock(return_value="- fact one")
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_search("query", layer="conversation", ctx=ctx)

        data = json.loads(result)
        assert data["results"][0]["text"] == "fact one"
        assert data["results"][0]["source"] == "L4_conversation"

    @pytest.mark.asyncio
    async def test_bridge_exception_returns_empty(self):
        memory_search, _ = _import_tools()
        bridge = AsyncMock()
        bridge.retrieve = AsyncMock(side_effect=OSError("bridge error"))
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_search("query", layer="conversation", ctx=ctx)

        data = json.loads(result)
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_episodic_knowledge_included_in_all_layer(self):
        memory_search, _ = _import_tools()
        episodic = MagicMock()
        episodic.get_knowledge_context.return_value = "device info: sensor-a"
        ctx = _make_ctx(episodic_memory=episodic)

        with patch(
            "askme.memory.retrieval.semantic_index.SemanticIndex",
            side_effect=ImportError,
            create=True,
        ):
            result = await memory_search("sensor-a", layer="all", ctx=ctx)

        data = json.loads(result)
        l3_results = [item for item in data["results"] if "L3" in item["source"]]
        assert len(l3_results) > 0

    @pytest.mark.asyncio
    async def test_returns_at_most_n_results(self):
        memory_search, _ = _import_tools()
        bridge = AsyncMock()
        lines = "\n".join(f"- item{i}" for i in range(10))
        bridge.retrieve = AsyncMock(return_value=lines)
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_search("query", n=3, layer="conversation", ctx=ctx)

        data = json.loads(result)
        assert len(data["results"]) <= 3


class TestMemorySave:
    @pytest.mark.asyncio
    async def test_no_memory_bridge_returns_error(self):
        _, memory_save = _import_tools()
        ctx = _make_ctx(memory_bridge=None)

        result = await memory_save("fact", ctx=ctx)

        data = json.loads(result)
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_saves_to_bridge(self):
        _, memory_save = _import_tools()
        bridge = AsyncMock()
        bridge.save = AsyncMock()
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_save("warehouse-a temperature normal", ctx=ctx)

        data = json.loads(result)
        assert data["status"] == "ok"
        bridge.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_exception_returns_error(self):
        _, memory_save = _import_tools()
        bridge = AsyncMock()
        bridge.save = AsyncMock(side_effect=OSError("save failed"))
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_save("fact", ctx=ctx)

        data = json.loads(result)
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_result_includes_saved_text_truncated(self):
        _, memory_save = _import_tools()
        bridge = AsyncMock()
        bridge.save = AsyncMock()
        ctx = _make_ctx(memory_bridge=bridge)

        result = await memory_save("short fact", ctx=ctx)

        data = json.loads(result)
        assert "short fact" in data["message"]
