"""Auto-discovery and registration of @tool and @skill decorated functions.

Scans specified packages for decorated functions and registers them
with the tool registry and skill manager automatically.

Usage::

    from askme.core.registry import auto_discover
    auto_discover(tool_registry, skill_manager, config)
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Any, TYPE_CHECKING

from askme.core.decorators import ToolSpec, SkillSpec, get_registered_tools, get_registered_skills
from askme.core import services

if TYPE_CHECKING:
    from askme.tools.tool_registry import BaseTool, ToolRegistry
    from askme.skills.skill_manager import SkillManager

logger = logging.getLogger(__name__)


def auto_discover(
    tool_registry: ToolRegistry | None = None,
    skill_manager: SkillManager | None = None,
    *,
    scan_packages: list[str] | None = None,
) -> tuple[int, int]:
    """Scan packages for @tool/@skill decorators and register them.

    Args:
        tool_registry: ToolRegistry to register tools into.
        skill_manager: SkillManager to register skills into (Phase 2).
        scan_packages: List of package names to scan. Defaults to
            ["askme.tools", "askme.skills"].

    Returns:
        (tools_registered, skills_registered) count tuple.
    """
    packages = scan_packages or ["askme.tools", "askme.skills"]

    for pkg_name in packages:
        _import_package_recursive(pkg_name)

    tools_count = 0
    skills_count = 0

    # Register tools
    if tool_registry is not None:
        for spec in get_registered_tools():
            try:
                tool_instance = _spec_to_tool(spec)
                if tool_instance is not None:
                    tool_registry.register(tool_instance)
                    tools_count += 1
                    logger.debug("[AutoDiscover] Tool registered: %s", spec.name)
            except Exception as exc:
                logger.warning("[AutoDiscover] Failed to register tool '%s': %s", spec.name, exc)

    # Register skills (Phase 2 — SkillSpec → SkillDefinition)
    if skill_manager is not None:
        for spec in get_registered_skills():
            try:
                _register_skill(spec, skill_manager)
                skills_count += 1
                logger.debug("[AutoDiscover] Skill registered: %s", spec.name)
            except Exception as exc:
                logger.warning("[AutoDiscover] Failed to register skill '%s': %s", spec.name, exc)

    if tools_count or skills_count:
        logger.info(
            "[AutoDiscover] Registered %d tools, %d skills from decorators",
            tools_count, skills_count,
        )

    return tools_count, skills_count


def _import_package_recursive(package_name: str) -> None:
    """Import all modules in a package to trigger decorator registration."""
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        logger.debug("[AutoDiscover] Package '%s' not found, skipping", package_name)
        return

    if not hasattr(package, "__path__"):
        return  # not a package, just a module

    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, prefix=package_name + "."
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.debug("[AutoDiscover] Could not import '%s': %s", module_name, exc)


def _spec_to_tool(spec: ToolSpec) -> BaseTool | None:
    """Convert a ToolSpec (from @tool decorator) into a BaseTool instance."""
    from askme.tools.tool_registry import BaseTool

    # Resolve dependencies
    deps = services.resolve(spec.requires) if spec.requires else {}

    # Create a dynamic BaseTool subclass wrapping the decorated function
    fn = spec.fn

    class _DecoratedTool(BaseTool):
        name = spec.name
        description = spec.description
        parameters: dict[str, Any] = spec.parameters or {"type": "object", "properties": {}}
        safety_level = spec.safety_level
        dev_only = spec.dev_only

        def execute(self, **kwargs: Any) -> str:
            # Inject dependencies as _prefixed kwargs
            for dep_name, dep_instance in deps.items():
                kwargs[f"_{dep_name}"] = dep_instance
            result = fn(**kwargs)
            return str(result) if result is not None else ""

    return _DecoratedTool()


def _register_skill(spec: SkillSpec, skill_manager: Any) -> None:
    """Register a SkillSpec with the SkillManager. Phase 2 placeholder."""
    # For now, @skill decorated functions are logged but registration
    # into SkillManager requires SkillDefinition format — will be
    # implemented in Phase 2 when we migrate existing skills.
    logger.debug(
        "[AutoDiscover] Skill '%s' found (triggers=%d, agent_shell=%s) — "
        "Phase 2: will auto-register into SkillManager",
        spec.name, len(spec.triggers), spec.agent_shell,
    )


def get_agent_allowed_tools() -> set[str]:
    """Return names of all @tool functions with agent_allowed=True."""
    return {spec.name for spec in get_registered_tools() if spec.agent_allowed}


def get_tool_voice_labels() -> dict[str, str | None]:
    """Return {tool_name: voice_label} for all @tool functions."""
    return {spec.name: spec.voice_label for spec in get_registered_tools()}


def get_agent_shell_skills() -> set[str]:
    """Return names of all @skill functions with agent_shell=True."""
    return {spec.name for spec in get_registered_skills() if spec.agent_shell}
