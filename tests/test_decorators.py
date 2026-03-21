"""Tests for core decorator system — @tool, @skill, auto-discover, services."""

from __future__ import annotations

import pytest

from askme.core.decorators import (
    ToolSpec, SkillSpec, _TOOL_REGISTRY, _SKILL_REGISTRY,
    tool, skill, get_registered_tools, get_registered_skills,
    _fn_to_schema,
)
from askme.core import services
from askme.core.registry import (
    _spec_to_tool, get_agent_allowed_tools, get_tool_voice_labels,
    get_agent_shell_skills,
)


# Clean registries between tests
@pytest.fixture(autouse=True)
def clean_registries():
    _TOOL_REGISTRY.clear()
    _SKILL_REGISTRY.clear()
    services.clear()
    yield
    _TOOL_REGISTRY.clear()
    _SKILL_REGISTRY.clear()
    services.clear()


# ---------- @tool ----------

class TestToolDecorator:
    def test_basic_registration(self):
        @tool(name="test_tool", description="A test tool")
        def my_tool(query: str = "") -> str:
            return f"result: {query}"

        specs = get_registered_tools()
        assert len(specs) == 1
        assert specs[0].name == "test_tool"
        assert specs[0].description == "A test tool"
        assert specs[0].fn is my_tool

    def test_agent_allowed(self):
        @tool(name="agent_tool", description="x", agent_allowed=True, voice_label="测试")
        def my_tool() -> str:
            return ""

        assert get_agent_allowed_tools() == {"agent_tool"}
        labels = get_tool_voice_labels()
        assert labels["agent_tool"] == "测试"

    def test_safety_level(self):
        @tool(name="dangerous_tool", description="x", safety="dangerous")
        def my_tool() -> str:
            return ""

        assert get_registered_tools()[0].safety_level == "dangerous"

    def test_requires(self):
        @tool(name="vision_tool", description="x", requires=["vision_bridge"])
        def my_tool(*, _vision_bridge=None) -> str:
            return str(_vision_bridge)

        assert get_registered_tools()[0].requires == ["vision_bridge"]

    def test_function_still_callable(self):
        @tool(name="callable_tool", description="x")
        def my_tool(x: int = 0) -> str:
            return f"value={x}"

        assert my_tool(x=42) == "value=42"

    def test_docstring_as_description(self):
        @tool(name="doc_tool")
        def my_tool() -> str:
            """这是工具描述"""
            return ""

        assert get_registered_tools()[0].description == "这是工具描述"


# ---------- @skill ----------

class TestSkillDecorator:
    def test_basic_registration(self):
        @skill(name="test_skill", triggers=["你好", "嗨"])
        async def my_skill(user_input: str, context: dict) -> str:
            return "hello"

        specs = get_registered_skills()
        assert len(specs) == 1
        assert specs[0].name == "test_skill"
        assert specs[0].triggers == ["你好", "嗨"]

    def test_agent_shell(self):
        @skill(name="agent_skill", agent_shell=True, timeout=90)
        async def my_skill(user_input: str, context: dict) -> str:
            return ""

        assert get_agent_shell_skills() == {"agent_skill"}
        assert get_registered_skills()[0].timeout == 90

    def test_tools_list(self):
        @skill(name="vis_skill", tools=["look_around", "find_target"])
        async def my_skill(user_input: str, context: dict) -> str:
            return ""

        assert get_registered_skills()[0].tools == ["look_around", "find_target"]


# ---------- Schema generation ----------

class TestSchemaGeneration:
    def test_string_param(self):
        def fn(query: str = "") -> str:
            return ""

        schema = _fn_to_schema(fn)
        assert schema["properties"]["query"]["type"] == "string"
        assert "query" not in schema.get("required", [])

    def test_required_param(self):
        def fn(query: str) -> str:
            return ""

        schema = _fn_to_schema(fn)
        assert "query" in schema["required"]

    def test_int_param(self):
        def fn(count: int = 0) -> str:
            return ""

        schema = _fn_to_schema(fn)
        assert schema["properties"]["count"]["type"] == "integer"

    def test_bool_param(self):
        def fn(full: bool = False) -> str:
            return ""

        schema = _fn_to_schema(fn)
        assert schema["properties"]["full"]["type"] == "boolean"

    def test_skip_underscore_params(self):
        def fn(query: str = "", *, _vision=None) -> str:
            return ""

        schema = _fn_to_schema(fn)
        assert "_vision" not in schema["properties"]
        assert "query" in schema["properties"]

    def test_skip_kwargs(self):
        def fn(query: str = "", **kwargs) -> str:
            return ""

        schema = _fn_to_schema(fn)
        assert len(schema["properties"]) == 1


# ---------- Spec to Tool ----------

class TestSpecToTool:
    def test_basic_conversion(self):
        @tool(name="simple", description="简单工具")
        def simple_tool(text: str = "") -> str:
            return f"got: {text}"

        spec = get_registered_tools()[0]
        tool_instance = _spec_to_tool(spec)

        assert tool_instance is not None
        assert tool_instance.name == "simple"
        assert tool_instance.description == "简单工具"
        result = tool_instance.execute(text="hello")
        assert result == "got: hello"

    def test_with_dependency(self):
        services.register("my_service", {"key": "value"})

        @tool(name="dep_tool", description="x", requires=["my_service"])
        def dep_tool(query: str = "", *, _my_service=None) -> str:
            return str(_my_service)

        spec = get_registered_tools()[0]
        tool_instance = _spec_to_tool(spec)
        result = tool_instance.execute(query="test")
        assert "value" in result


# ---------- Services ----------

class TestServices:
    def test_register_and_get(self):
        services.register("vision", "fake_vision")
        assert services.get("vision") == "fake_vision"

    def test_get_missing(self):
        assert services.get("nonexistent") is None

    def test_resolve(self):
        services.register("a", 1)
        services.register("b", 2)
        result = services.resolve(["a", "b", "missing"])
        assert result == {"a": 1, "b": 2}

    def test_clear(self):
        services.register("x", 1)
        services.clear()
        assert services.get("x") is None
