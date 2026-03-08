"""
Tool registry system for askme.

Provides an abstract BaseTool class and a ToolRegistry that manages
tool registration, OpenAI-format definition export, and execution dispatch.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must define:
      - name: unique tool identifier
      - description: human-readable description
      - parameters: JSON Schema dict for the tool's parameters
      - execute(**kwargs) -> str: the tool's implementation
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    safety_level: str = "normal"  # normal | dangerous | critical

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given keyword arguments.

        Returns:
            A string result to feed back to the LLM.
        """
        ...

    def get_definition(self) -> dict[str, Any]:
        """Return the OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ToolRegistry:
    """Registry that holds tools and dispatches execution requests."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── Registration ────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance. Overwrites if name already exists."""
        if not tool.name:
            raise ValueError("Tool must have a non-empty 'name'.")
        logger.debug("Registered tool: %s", tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        removed = self._tools.pop(name, None)
        if removed:
            logger.debug("Unregistered tool: %s", name)
        return removed is not None

    # ── Querying ────────────────────────────────────────────────

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name, or None."""
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions in OpenAI function-calling format."""
        return [tool.get_definition() for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ── Execution ───────────────────────────────────────────────

    def execute(self, name: str, args_json: str | None = None) -> str:
        """Execute a tool by name with JSON-encoded arguments.

        Args:
            name: The tool name to execute.
            args_json: A JSON string of keyword arguments (may be None or empty).

        Returns:
            The tool's string result, or an error message.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"[Error] Tool not found: {name}"

        try:
            kwargs = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return f"[Error] Invalid JSON arguments: {exc}"

        try:
            logger.info("Executing tool: %s(%s)", name, kwargs)
            result = tool.execute(**kwargs)
            return result
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            return f"[Error] Tool '{name}' execution failed: {exc}"
