"""Core tool registry, built-in tools, and execution-control contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "BaseTool": ("askme.tools.core.tool_registry", "BaseTool"),
    "CircuitBreaker": ("askme.tools.core.execution_control", "CircuitBreaker"),
    "DispatchSkillTool": ("askme.tools.core.builtin_tools", "DispatchSkillTool"),
    "DogControlDispatchTool": (
        "askme.tools.core.builtin_tools",
        "DogControlDispatchTool",
    ),
    "EditFileTool": ("askme.tools.core.builtin_tools", "EditFileTool"),
    "GetTimeTool": ("askme.tools.core.builtin_tools", "GetTimeTool"),
    "HttpRequestTool": ("askme.tools.core.builtin_tools", "HttpRequestTool"),
    "ListDirectoryTool": ("askme.tools.core.builtin_tools", "ListDirectoryTool"),
    "NavDispatchTool": ("askme.tools.core.builtin_tools", "NavDispatchTool"),
    "NavStatusTool": ("askme.tools.core.builtin_tools", "NavStatusTool"),
    "PendingToolApproval": ("askme.tools.core.tool_registry", "PendingToolApproval"),
    "ReadFileTool": ("askme.tools.core.builtin_tools", "ReadFileTool"),
    "RunCommandTool": ("askme.tools.core.builtin_tools", "RunCommandTool"),
    "SandboxedBashTool": ("askme.tools.core.builtin_tools", "SandboxedBashTool"),
    "ScheduledWork": ("askme.tools.core.execution_control", "ScheduledWork"),
    "SpeakProgressTool": ("askme.tools.core.builtin_tools", "SpeakProgressTool"),
    "ToolExecutionScheduler": (
        "askme.tools.core.execution_control",
        "ToolExecutionScheduler",
    ),
    "ToolExecutionTimeoutError": (
        "askme.tools.core.tool_registry",
        "ToolExecutionTimeoutError",
    ),
    "ToolQueueFullError": ("askme.tools.core.execution_control", "ToolQueueFullError"),
    "ToolRegistry": ("askme.tools.core.tool_registry", "ToolRegistry"),
    "WebFetchTool": ("askme.tools.core.builtin_tools", "WebFetchTool"),
    "WebSearchTool": ("askme.tools.core.builtin_tools", "WebSearchTool"),
    "WindowRateLimiter": ("askme.tools.core.execution_control", "WindowRateLimiter"),
    "WriteFileTool": ("askme.tools.core.builtin_tools", "WriteFileTool"),
    "register_builtin_tools": (
        "askme.tools.core.builtin_tools",
        "register_builtin_tools",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve core tool contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
