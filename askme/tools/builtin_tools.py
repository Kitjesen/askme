"""
Built-in tools for askme.

Ports the four original tools (get_time, run_command, read_file, list_directory)
from the prototype as proper BaseTool subclasses.
"""

from __future__ import annotations

import datetime
import os
import shlex
import subprocess
from typing import Any

from .tool_registry import BaseTool, ToolRegistry


class GetTimeTool(BaseTool):
    """Return the current system date and time."""

    name = "get_current_time"
    description = "获取当前系统时间"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunCommandTool(BaseTool):
    """Execute a shell command and return its output."""

    name = "run_command"
    description = "在本地执行一条 shell 命令并返回输出（最多 2000 字符）"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令",
            },
        },
        "required": ["command"],
    }
    safety_level = "dangerous"

    def execute(self, *, command: str = "", **kwargs: Any) -> str:
        if not command:
            return "[Error] No command provided."
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return f"[Error] Invalid command syntax: {exc}"
        try:
            result = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout or result.stderr or "(no output)"
            return output[:2000]
        except subprocess.TimeoutExpired:
            return "[Error] Command timed out (10s)."
        except Exception as exc:
            return f"[Error] {exc}"


class ReadFileTool(BaseTool):
    """Read the contents of a local file."""

    name = "read_file"
    description = "读取本地文件内容（最多 3000 字符）"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
        },
        "required": ["path"],
    }
    safety_level = "normal"

    def execute(self, *, path: str = "", **kwargs: Any) -> str:
        if not path:
            return "[Error] No path provided."
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(3000)
        except Exception as exc:
            return f"[Error] Reading file failed: {exc}"


class ListDirectoryTool(BaseTool):
    """List files and directories in a given path."""

    name = "list_directory"
    description = "列出目录中的文件和文件夹"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径，默认当前目录",
            },
        },
    }
    safety_level = "normal"

    def execute(self, *, path: str = ".", **kwargs: Any) -> str:
        target = path or "."
        try:
            entries = os.listdir(target)
            return "\n".join(entries[:50])
        except Exception as exc:
            return f"[Error] {exc}"


class DispatchSkillTool(BaseTool):
    """Meta-tool: lets the LLM invoke a named skill during general conversation.

    This enables natural language → skill composition without voice triggers.
    The LLM sees the skill catalog in its system prompt and can decide to
    dispatch skills based on user intent.
    """

    name = "dispatch_skill"
    description = (
        "执行一个机器人技能（如导航、检查、建图等）。"
        "可用技能列表见系统提示。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "要执行的技能名称",
            },
            "reason": {
                "type": "string",
                "description": "调用原因/传给技能的上下文",
            },
        },
        "required": ["skill_name"],
    }
    safety_level = "normal"

    def __init__(self) -> None:
        # Dispatcher is set after construction via set_dispatcher()
        self._dispatcher: Any = None

    def set_dispatcher(self, dispatcher: Any) -> None:
        """Wire the SkillDispatcher after both are constructed."""
        self._dispatcher = dispatcher

    def execute(self, *, skill_name: str = "", reason: str = "", **kwargs: Any) -> str:
        if not skill_name:
            return "[Error] 未指定技能名称"
        if self._dispatcher is None:
            return "[Error] SkillDispatcher 未初始化"
        return self._dispatcher.execute_skill_sync(skill_name, reason)


# ── Convenience registration ────────────────────────────────────

_BUILTIN_TOOLS: list[type[BaseTool]] = [
    GetTimeTool,
    RunCommandTool,
    ReadFileTool,
    ListDirectoryTool,
]


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Instantiate and register all built-in tools into the given registry."""
    for tool_cls in _BUILTIN_TOOLS:
        registry.register(tool_cls())
    # dispatch_skill is registered separately in app.py after SkillDispatcher
    # is constructed (circular dependency: tool needs dispatcher, dispatcher
    # needs pipeline which needs tools).
