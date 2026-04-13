"""ToolsModule — wraps ToolRegistry construction as a declarative module.

Mirrors the tool registration logic from ``assembly.py`` lines 433-441::

    tools = ToolRegistry()
    register_builtin_tools(tools, production_mode=...)
    tools.register(RobotApiTool())
    register_vision_tools(tools, vision)
    register_move_tools(tools)
    register_scan_tools(tools, vision)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.module import Module, ModuleRegistry, Out
from askme.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolsModule(Module):
    """Provides a ToolRegistry with all builtin tools registered."""

    name = "tools"
    provides = ("tools",)

    tool_registry: Out[ToolRegistry]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.tools.builtin_tools import register_builtin_tools
        from askme.tools.move_tool import register_move_tools
        from askme.tools.robot_api_tool import RobotApiTool
        from askme.tools.scan_tool import register_scan_tools
        from askme.tools.temporal_query_tool import register_temporal_tools

        tools_cfg = cfg.get("tools", {})
        production_mode = bool(tools_cfg.get("production_mode", False))

        self.registry = ToolRegistry()
        register_builtin_tools(self.registry, production_mode=production_mode)
        self.registry.register(RobotApiTool())
        register_move_tools(self.registry)
        register_scan_tools(self.registry)
        register_temporal_tools(self.registry)

        logger.info("ToolsModule: built (%d tools)", len(self.registry))

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "tool_count": len(self.registry)}
