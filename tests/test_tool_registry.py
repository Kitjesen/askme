from __future__ import annotations

import time
from typing import Any

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

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "dangerous"


class _CriticalTool(BaseTool):
    name = "critical_tool"
    description = "critical tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "critical"

    def execute(self, **kwargs: Any) -> str:
        return "critical"


class _SecondDangerousTool(BaseTool):
    name = "second_dangerous_tool"
    description = "another dangerous tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "second-dangerous"


class _EmergencyStopTool(BaseTool):
    name = "robot_emergency_stop"
    description = "emergency stop"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "critical"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "estop"


class _SlowTool(BaseTool):
    name = "slow_tool"
    description = "slow tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        time.sleep(0.05)
        return "slow"


def _make_registry(**overrides: Any) -> ToolRegistry:
    config = {
        "default_timeout": 0.2,
        "dangerous_timeout": 0.2,
        "critical_timeout": 0.2,
        "timeout_cooldown": 0.0,
        "approval_timeout_seconds": 30.0,
        "require_confirmation_levels": ["dangerous", "critical"],
        "confirmation_bypass_tools": ["robot_emergency_stop"],
        "confirmation_phrases": ["确认执行", "approve"],
        "rejection_phrases": ["取消", "cancel"],
        **overrides,
    }
    return ToolRegistry(config=config)


def test_get_definitions_filters_by_max_safety_level() -> None:
    registry = _make_registry()
    registry.register(_SafeTool())
    registry.register(_DangerousTool())
    registry.register(_CriticalTool())

    definitions = registry.get_definitions(max_safety_level="normal")
    names = [definition["function"]["name"] for definition in definitions]

    assert names == ["safe_tool"]


def test_execute_blocks_tool_outside_context_policy() -> None:
    registry = _make_registry()
    registry.register(_DangerousTool())

    result = registry.execute("dangerous_tool", "{}", max_safety_level="normal")

    assert result == (
        "[Error] Tool 'dangerous_tool' requires safety level 'dangerous', "
        "but this request only allows 'normal'."
    )


def test_execute_respects_explicit_allowlist() -> None:
    registry = _make_registry()
    registry.register(_SafeTool())
    registry.register(_DangerousTool())

    result = registry.execute(
        "dangerous_tool",
        "{}",
        allowed_names={"dangerous_tool"},
        max_safety_level="dangerous",
    )

    assert result.startswith("[Approval Required]")


def test_dangerous_tool_requires_operator_approval() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)

    result = registry.execute(
        "dangerous_tool",
        '{"target": "bin-a"}',
        max_safety_level="dangerous",
    )

    assert result.startswith("[Approval Required]")
    assert registry.has_pending_approval() is True
    assert tool.calls == 0


def test_approve_pending_executes_dangerous_tool() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    result = registry.approve_pending()

    assert result == "dangerous"
    assert registry.has_pending_approval() is False
    assert tool.calls == 1


def test_reject_pending_cancels_dangerous_tool() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    result = registry.reject_pending()

    assert result.startswith("[Approval Cancelled]")
    assert registry.has_pending_approval() is False
    assert tool.calls == 0


def test_pending_approval_expires() -> None:
    registry = _make_registry(approval_timeout_seconds=0.01)
    registry.register(_DangerousTool())
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    time.sleep(0.02)
    result = registry.approve_pending()

    assert result.startswith("[Approval Expired]")
    assert registry.has_pending_approval() is False


def test_pending_approval_blocks_new_high_risk_request() -> None:
    registry = _make_registry()
    first = _DangerousTool()
    second = _SecondDangerousTool()
    registry.register(first)
    registry.register(second)

    registry.execute("dangerous_tool", '{"target": "bin-a"}', max_safety_level="dangerous")
    result = registry.execute(
        "second_dangerous_tool",
        '{"target": "bin-b"}',
        max_safety_level="dangerous",
    )

    assert result.startswith("[Approval Pending]")
    assert "dangerous_tool" in result

    approved = registry.approve_pending()
    assert approved == "dangerous"
    assert first.calls == 1
    assert second.calls == 0


def test_handle_pending_input_requires_explicit_resolution() -> None:
    registry = _make_registry()
    registry.register(_DangerousTool())
    registry.execute("dangerous_tool", '{"target": "bin-a"}', max_safety_level="dangerous")

    result = registry.handle_pending_input("status update")

    assert result is not None
    assert result.startswith("[Approval Pending]")
    assert "dangerous_tool" in result
    assert registry.has_pending_approval() is True


def test_handle_pending_input_reports_expired_request() -> None:
    registry = _make_registry(approval_timeout_seconds=0.01)
    registry.register(_DangerousTool())
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    time.sleep(0.02)
    result = registry.handle_pending_input("approve")

    assert result is not None
    assert result.startswith("[Approval Expired]")
    assert registry.has_pending_approval() is False


def test_confirmation_phrase_matching_requires_pending_request() -> None:
    registry = _make_registry()
    registry.register(_DangerousTool())

    assert registry.matches_confirmation("确认执行") is False

    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")
    assert registry.matches_confirmation("确认执行。") is True
    assert registry.matches_rejection("取消") is True


def test_critical_bypass_tool_executes_immediately() -> None:
    registry = _make_registry()
    tool = _EmergencyStopTool()
    registry.register(tool)

    result = registry.execute("robot_emergency_stop", "{}", max_safety_level="critical")

    assert result == "estop"
    assert registry.has_pending_approval() is False
    assert tool.calls == 1


def test_timeout_places_tool_on_cooldown() -> None:
    registry = _make_registry(default_timeout=0.01, timeout_cooldown=1.0)
    registry.register(_SlowTool())

    first = registry.execute("slow_tool", "{}")
    second = registry.execute("slow_tool", "{}")

    assert first.startswith("[Timeout]")
    assert second.startswith("[Error] Tool 'slow_tool' is temporarily unavailable")
