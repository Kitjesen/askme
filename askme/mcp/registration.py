"""Shared FastMCP instance and explicit tool/resource module registration."""

from __future__ import annotations

from importlib import import_module

from mcp.server.fastmcp import FastMCP

from askme.mcp.context import app_lifespan

mcp = FastMCP(
    "askme",
    instructions="Voice AI assistant with robot arm control and skills",
    lifespan=app_lifespan,
)

_REGISTERED = False

MCP_MODULES: tuple[str, ...] = (
    "askme.mcp.resources.contract_resources",
    "askme.mcp.resources.health_resources",
    "askme.mcp.resources.perception_resources",
    "askme.mcp.resources.robot_resources",
    "askme.mcp.resources.skill_resources",
    "askme.mcp.tools.memory_tools",
    "askme.mcp.tools.robot_tools",
    "askme.mcp.tools.skill_tools",
    "askme.mcp.tools.vision_tools",
    "askme.mcp.tools.voice_tools",
)


def register_mcp_modules() -> None:
    """Import MCP tool/resource modules once so decorators bind to ``mcp``."""
    global _REGISTERED
    if _REGISTERED:
        return

    for module_name in MCP_MODULES:
        import_module(module_name)

    _REGISTERED = True


def mcp_module_manifest() -> list[str]:
    """Return the MCP tool/resource module registration manifest."""

    return list(MCP_MODULES)


__all__ = ["MCP_MODULES", "mcp", "mcp_module_manifest", "register_mcp_modules"]
