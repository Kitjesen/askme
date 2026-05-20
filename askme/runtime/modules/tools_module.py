"""ToolsModule — wraps ToolRegistry construction as a declarative module.

Canonical wiring::

    tools = ToolRegistry()
    register_builtin_tools(tools, production_mode=...)
    tools.register(RobotApiTool())
    register_vision_tools(tools, vision)
    register_move_tools(tools, robot_control_client=..., navigation_client=...)
    register_scan_tools(tools, vision, robot_control_client=...)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.core.module import Module, ModuleRegistry, Out
from askme.tools.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolsModule(Module):
    """Provides a ToolRegistry with all builtin tools registered."""

    name = "tools"
    provides = ("tools",)

    tool_registry: Out[ToolRegistry]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.providers import build_navigation, build_temporal_memory
        from askme.tools.core.builtin_tools import register_builtin_tools
        from askme.tools.robot.move_tool import register_move_tools
        from askme.tools.robot.robot_api_tool import RobotApiTool
        from askme.tools.spatial.scan_tool import register_scan_tools
        from askme.tools.spatial.temporal_query_tool import register_temporal_tools

        tools_cfg = cfg.get("tools", {})
        production_mode = bool(tools_cfg.get("production_mode", False))
        navigation_cfg = cfg.get("runtime", {}).get("navigation", {})
        navigation_client = build_navigation(navigation_cfg)
        temporal_memory_client = build_temporal_memory(navigation_cfg)
        self.navigation_client = navigation_client
        self.temporal_memory_client = temporal_memory_client

        self.registry = ToolRegistry()
        register_builtin_tools(
            self.registry,
            production_mode=production_mode,
            navigation_client=navigation_client,
        )
        self.registry.register(RobotApiTool())
        control_mod = registry.get("control")
        control_client = (
            getattr(control_mod, "control_client", None)
            or getattr(control_mod, "client", None)
            if control_mod is not None
            else None
        )
        register_move_tools(
            self.registry,
            robot_control_client=control_client,
            navigation_client=navigation_client,
        )
        register_scan_tools(self.registry, robot_control_client=control_client)
        register_temporal_tools(
            self.registry,
            temporal_memory_client=temporal_memory_client,
        )
        if control_client is not None:
            self.bind_robot_control_client(control_client)

        logger.info("ToolsModule: built (%d tools)", len(self.registry))

    def bind_robot_control_client(self, robot_control_client: Any) -> bool:
        """Inject the runtime robot-control port into tools that need motion."""

        bound = False
        for tool_name in ("dog_control_dispatch", "move_robot", "scan_around"):
            tool = self.registry.get(tool_name)
            setter = getattr(tool, "set_robot_control_client", None)
            if callable(setter):
                setter(robot_control_client)
                bound = True
        return bound

    def bind_navigation_client(self, navigation_client: Any) -> bool:
        """Inject the runtime navigation port into tools that need navigation."""

        self.navigation_client = navigation_client
        bound = False
        for tool_name in ("nav_dispatch", "nav_status", "move_robot"):
            tool = self.registry.get(tool_name)
            setter = getattr(tool, "set_navigation_client", None)
            if callable(setter):
                setter(navigation_client)
                bound = True
        return bound

    def bind_temporal_memory_client(self, temporal_memory_client: Any) -> bool:
        """Inject the runtime temporal-memory port into spatial memory tools."""

        self.temporal_memory_client = temporal_memory_client
        tool = self.registry.get("temporal_query")
        setter = getattr(tool, "set_temporal_memory_client", None)
        if callable(setter):
            setter(temporal_memory_client)
            return True
        return False

    async def stop(self) -> None:
        registry = getattr(self, "registry", None)
        shutdown = getattr(registry, "shutdown", None)
        if callable(shutdown):
            shutdown(wait=False, cancel_futures=True)

    def health(self) -> dict[str, Any]:
        diagnostics = self.registry.diagnostics()
        return {
            "status": "ok",
            "tool_count": diagnostics["tool_count"],
            "executor": diagnostics["executor"],
            "cooldown_count": diagnostics["cooldown_count"],
            "pending_approval": diagnostics["pending_approval"],
            "rate_limit": diagnostics.get("rate_limit", {}),
            "circuit_breakers": diagnostics.get("circuit_breakers", {}),
            "background_jobs": diagnostics.get("background_jobs", {}),
        }
