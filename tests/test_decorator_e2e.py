"""End-to-end test: @tool decorator → auto_discover → ToolRegistry → execute."""

from __future__ import annotations

import importlib

import pytest

from askme.core.decorators import _TOOL_REGISTRY, _SKILL_REGISTRY
from askme.core import services
from askme.core.registry import auto_discover, get_agent_allowed_tools, get_tool_voice_labels
from askme.tools.tool_registry import ToolRegistry


@pytest.fixture(autouse=True)
def clean_and_reimport():
    """Clear registries, then re-trigger @tool decorators by reloading module."""
    _TOOL_REGISTRY.clear()
    _SKILL_REGISTRY.clear()
    services.clear()

    import askme.tools.time_tool as mod
    importlib.reload(mod)

    yield

    _TOOL_REGISTRY.clear()
    _SKILL_REGISTRY.clear()
    services.clear()


class TestDecoratorE2E:
    def test_time_tool_discovered_and_executable(self):
        """@tool get_current_time → auto_discover → ToolRegistry → execute."""
        registry = ToolRegistry()
        tools_count, _ = auto_discover(tool_registry=registry)
        assert tools_count >= 1

        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert "get_current_time" in names

        result = registry.execute("get_current_time", "{}")
        assert "202" in result
        assert "-" in result

    def test_agent_allowed_propagates(self):
        allowed = get_agent_allowed_tools()
        assert "get_current_time" in allowed

    def test_voice_label_propagates(self):
        labels = get_tool_voice_labels()
        assert labels["get_current_time"] == "获取时间"

    def test_schema_auto_generated(self):
        registry = ToolRegistry()
        auto_discover(tool_registry=registry)

        definitions = registry.get_definitions()
        time_def = [d for d in definitions if d["function"]["name"] == "get_current_time"]
        assert len(time_def) == 1
        params = time_def[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}

    def test_coexists_with_old_tools(self):
        """Decorator tools coexist with legacy BaseTool classes."""
        registry = ToolRegistry()
        from askme.tools.builtin_tools import ReadFileTool
        registry.register(ReadFileTool())

        auto_discover(tool_registry=registry)

        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert "read_file" in names
        assert "get_current_time" in names
