"""Extended tests for ToolRegistry — registration, validation, exposure rules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from askme.tools.tool_registry import (
    BaseTool,
    ToolRegistry,
    _json_type_matches,
    _normalize_safety_level,
)

# ── Test doubles ─────────────────────────────────────────────────────────────

def _make_tool(
    name: str = "test_tool",
    description: str = "A test tool",
    safety_level: str = "normal",
    dev_only: bool = False,
    agent_allowed: bool = False,
    voice_label: str = "",
    result: str = "ok",
    parameters: dict | None = None,
) -> BaseTool:
    _result = result
    _parameters = parameters or {}

    class _Tool(BaseTool):
        def execute(self, **kwargs):
            return _result

    t = _Tool()
    t.name = name
    t.description = description
    t.safety_level = safety_level
    t.dev_only = dev_only
    t.agent_allowed = agent_allowed
    t.voice_label = voice_label
    t.parameters = _parameters
    t.execute = MagicMock(return_value=result)
    return t


def _make_registry(config: dict | None = None) -> ToolRegistry:
    return ToolRegistry(config or {})


# ── TestRegister ──────────────────────────────────────────────────────────────

class TestRegister:
    def test_register_tool(self):
        reg = _make_registry()
        tool = _make_tool("my_tool")
        reg.register(tool)
        assert "my_tool" in reg

    def test_register_overwrites_existing(self):
        reg = _make_registry()
        tool1 = _make_tool("t")
        tool2 = _make_tool("t", description="new")
        reg.register(tool1)
        reg.register(tool2)
        assert reg.get("t").description == "new"

    def test_register_empty_name_raises(self):
        reg = _make_registry()
        tool = _make_tool("")
        with pytest.raises(ValueError, match="name"):
            reg.register(tool)

    def test_len_increases_after_register(self):
        reg = _make_registry()
        assert len(reg) == 0
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        assert len(reg) == 2

    def test_contains_true_after_register(self):
        reg = _make_registry()
        reg.register(_make_tool("tool_x"))
        assert "tool_x" in reg

    def test_contains_false_for_unknown(self):
        reg = _make_registry()
        assert "unknown_tool" not in reg


# ── TestUnregister ────────────────────────────────────────────────────────────

class TestUnregister:
    def test_unregister_existing_returns_true(self):
        reg = _make_registry()
        reg.register(_make_tool("t"))
        assert reg.unregister("t") is True
        assert "t" not in reg

    def test_unregister_nonexistent_returns_false(self):
        reg = _make_registry()
        assert reg.unregister("ghost") is False

    def test_unregister_clears_pending_approval(self):
        reg = _make_registry()
        tool = _make_tool("danger", safety_level="dangerous")
        reg.register(tool)
        reg.execute("danger", "{}")  # queues pending approval
        reg.unregister("danger")
        assert reg._pending_approval is None


# ── TestGet ───────────────────────────────────────────────────────────────────

class TestGet:
    def test_get_registered_tool(self):
        reg = _make_registry()
        tool = _make_tool("my_tool")
        reg.register(tool)
        assert reg.get("my_tool") is tool

    def test_get_nonexistent_returns_none(self):
        reg = _make_registry()
        assert reg.get("ghost") is None


# ── TestGetAgentAllowed ───────────────────────────────────────────────────────

class TestGetAgentAllowed:
    def test_agent_allowed_true_included(self):
        reg = _make_registry()
        reg.register(_make_tool("agent_tool", agent_allowed=True))
        assert "agent_tool" in reg.get_agent_allowed_names()

    def test_agent_allowed_false_excluded(self):
        reg = _make_registry()
        reg.register(_make_tool("normal_tool", agent_allowed=False))
        assert "normal_tool" not in reg.get_agent_allowed_names()

    def test_mixed_tools(self):
        reg = _make_registry()
        reg.register(_make_tool("a", agent_allowed=True))
        reg.register(_make_tool("b", agent_allowed=False))
        allowed = reg.get_agent_allowed_names()
        assert "a" in allowed
        assert "b" not in allowed


# ── TestGetVoiceLabels ────────────────────────────────────────────────────────

class TestGetVoiceLabels:
    def test_tool_with_voice_label_included(self):
        reg = _make_registry()
        reg.register(_make_tool("env_tool", voice_label="观察环境"))
        labels = reg.get_voice_labels()
        assert labels["env_tool"] == "观察环境"

    def test_tool_without_voice_label_excluded(self):
        reg = _make_registry()
        reg.register(_make_tool("silent_tool", voice_label=""))
        assert "silent_tool" not in reg.get_voice_labels()

    def test_multiple_voice_labels(self):
        reg = _make_registry()
        reg.register(_make_tool("a", voice_label="扫描"))
        reg.register(_make_tool("b", voice_label="导航"))
        reg.register(_make_tool("c", voice_label=""))
        labels = reg.get_voice_labels()
        assert len(labels) == 2


# ── TestListNames ─────────────────────────────────────────────────────────────

class TestListNames:
    def test_returns_sorted_names(self):
        reg = _make_registry()
        reg.register(_make_tool("zebra"))
        reg.register(_make_tool("alpha"))
        names = reg.list_names()
        assert names == sorted(names)

    def test_filters_by_allowed_names(self):
        reg = _make_registry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        reg.register(_make_tool("c"))
        names = reg.list_names(allowed_names=["a", "c"])
        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_filters_by_max_safety_level(self):
        reg = _make_registry()
        reg.register(_make_tool("safe", safety_level="normal"))
        reg.register(_make_tool("danger", safety_level="dangerous"))
        names = reg.list_names(max_safety_level="normal")
        assert "safe" in names
        assert "danger" not in names


# ── TestGetDefinitions ────────────────────────────────────────────────────────

class TestGetDefinitions:
    def test_definition_format(self):
        reg = _make_registry()
        reg.register(_make_tool("my_tool", description="Does things"))
        defs = reg.get_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "my_tool"

    def test_dangerous_tool_excluded_at_normal_level(self):
        reg = _make_registry()
        reg.register(_make_tool("safe", safety_level="normal"))
        reg.register(_make_tool("risky", safety_level="dangerous"))
        defs = reg.get_definitions(max_safety_level="normal")
        names = [d["function"]["name"] for d in defs]
        assert "safe" in names
        assert "risky" not in names

    def test_allowed_names_filter(self):
        reg = _make_registry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        defs = reg.get_definitions(allowed_names=["b"])
        names = [d["function"]["name"] for d in defs]
        assert names == ["b"]


# ── TestValidateArgs ──────────────────────────────────────────────────────────

class TestValidateArgs:
    def test_valid_args_returns_none(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        result = ToolRegistry._validate_args("tool", {"name": "hello"}, schema)
        assert result is None

    def test_missing_required_arg(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        result = ToolRegistry._validate_args("tool", {}, schema)
        assert result is not None
        assert "name" in result

    def test_wrong_type_returns_error(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        result = ToolRegistry._validate_args("tool", {"count": "five"}, schema)
        assert result is not None
        assert "count" in result

    def test_extra_key_blocked_when_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        result = ToolRegistry._validate_args("tool", {"name": "ok", "extra": "bad"}, schema)
        assert result is not None
        assert "extra" in result

    def test_extra_key_allowed_by_default(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        result = ToolRegistry._validate_args("tool", {"name": "ok", "extra": "fine"}, schema)
        assert result is None

    def test_empty_schema_no_error(self):
        result = ToolRegistry._validate_args("tool", {"anything": 42}, {})
        assert result is None


# ── TestJsonTypeMatches ───────────────────────────────────────────────────────

class TestJsonTypeMatches:
    def test_string_matches_string(self):
        assert _json_type_matches("hello", "string") is True

    def test_integer_matches_integer(self):
        assert _json_type_matches(42, "integer") is True

    def test_float_does_not_match_integer(self):
        assert _json_type_matches(3.14, "integer") is False

    def test_bool_does_not_match_integer(self):
        assert _json_type_matches(True, "integer") is False

    def test_bool_does_not_match_number(self):
        assert _json_type_matches(False, "number") is False

    def test_number_matches_float(self):
        assert _json_type_matches(3.14, "number") is True

    def test_number_matches_int(self):
        assert _json_type_matches(5, "number") is True

    def test_bool_matches_boolean(self):
        assert _json_type_matches(True, "boolean") is True

    def test_list_matches_array(self):
        assert _json_type_matches([1, 2, 3], "array") is True

    def test_dict_matches_object(self):
        assert _json_type_matches({"a": 1}, "object") is True

    def test_none_matches_null(self):
        assert _json_type_matches(None, "null") is True

    def test_unknown_type_returns_true(self):
        assert _json_type_matches("anything", "custom_type") is True


# ── TestNormalizeSafetyLevel ──────────────────────────────────────────────────

class TestNormalizeSafetyLevel:
    def test_normal_stays_normal(self):
        assert _normalize_safety_level("normal") == "normal"

    def test_dangerous_stays_dangerous(self):
        assert _normalize_safety_level("dangerous") == "dangerous"

    def test_critical_stays_critical(self):
        assert _normalize_safety_level("critical") == "critical"

    def test_unknown_defaults_to_critical(self):
        assert _normalize_safety_level("unknown") == "critical"

    def test_none_defaults_to_critical(self):
        assert _normalize_safety_level(None) == "critical"


# ── TestNormalizePhrase ───────────────────────────────────────────────────────

class TestNormalizePhrase:
    def test_strips_punctuation(self):
        result = ToolRegistry._normalize_phrase("确认！")
        assert "！" not in result

    def test_strips_spaces(self):
        result = ToolRegistry._normalize_phrase("  yes  ")
        assert result == "yes"

    def test_lowercases_ascii(self):
        result = ToolRegistry._normalize_phrase("YES")
        assert result == "yes"

    def test_empty_string(self):
        assert ToolRegistry._normalize_phrase("") == ""


# ── TestExecuteNormalTool ─────────────────────────────────────────────────────

class TestExecuteNormalTool:
    def test_successful_execution_returns_result(self):
        reg = _make_registry()
        tool = _make_tool("calc", result="42", safety_level="normal")
        reg.register(tool)
        result = reg.execute("calc", "{}")
        assert result == "42"

    def test_unknown_tool_returns_error(self):
        reg = _make_registry()
        result = reg.execute("ghost", "{}")
        assert "not found" in result.lower() or "error" in result.lower()

    def test_tool_execute_called_with_kwargs(self):
        reg = _make_registry()
        tool = _make_tool("add", result="3", safety_level="normal")
        reg.register(tool)
        reg.execute("add", '{"a": 1, "b": 2}')
        tool.execute.assert_called_once_with(a=1, b=2)

    def test_execute_exception_returns_error_string(self):
        reg = _make_registry()
        tool = _make_tool("broken", safety_level="normal")
        tool.execute = MagicMock(side_effect=RuntimeError("crash"))
        reg.register(tool)
        result = reg.execute("broken", "{}")
        assert "crash" in result or "[Error]" in result

    def test_validation_error_returned_before_execute(self):
        reg = _make_registry()
        tool = _make_tool(
            "strict",
            safety_level="normal",
            parameters={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        )
        reg.register(tool)
        result = reg.execute("strict", "{}")  # missing 'name'
        assert "missing" in result.lower()
        tool.execute.assert_not_called()
