"""Dependency surface for MCP resource handlers.

Resource handlers should stay as small JSON presenters. This module owns the
few legacy fallbacks that still need config, provider, or skill-manager access
when resources are called without an MCP lifespan context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

ConfigProvider = Callable[[], Mapping[str, Any]]
SectionProvider = Callable[[str], Mapping[str, Any]]
PayloadProvider = Callable[[], Mapping[str, Any]]
SkillManagerProvider = Callable[[], Any]


def _default_config_provider() -> Mapping[str, Any]:
    from askme.config import get_config

    return get_config()


def _default_section_provider(name: str) -> Mapping[str, Any]:
    from askme.config import get_section

    return get_section(name)


def _default_depth_info_provider() -> Mapping[str, Any]:
    from askme.providers import read_depth_info

    return read_depth_info()


def _default_arm_safety_defaults_provider() -> Mapping[str, Any]:
    from askme.providers import get_arm_safety_defaults

    return get_arm_safety_defaults()


def _default_skill_manager_provider() -> Any:
    from askme.skills.core.skill_manager import SkillManager

    manager = SkillManager()
    manager.load()
    return manager


@dataclass(frozen=True)
class MCPResourceSurface:
    """Callable dependencies and payload builders for MCP resources."""

    config_provider: ConfigProvider = _default_config_provider
    section_provider: SectionProvider = _default_section_provider
    depth_info_provider: PayloadProvider = _default_depth_info_provider
    arm_safety_defaults_provider: PayloadProvider = _default_arm_safety_defaults_provider
    skill_manager_provider: SkillManagerProvider = _default_skill_manager_provider

    def config(self) -> dict[str, Any]:
        return dict(self.config_provider() or {})

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.section_provider(name) or {})

    def depth_info_payload(self) -> dict[str, Any]:
        return dict(self.depth_info_provider() or {})

    def robot_status_payload(self) -> dict[str, Any]:
        robot_cfg = self.section("robot")
        return {
            "enabled": robot_cfg.get("enabled", False),
            "simulate": robot_cfg.get("simulate", True),
            "serial_port": robot_cfg.get("serial_port", "COM3"),
            "message": "Use robot_state() tool for live joint data",
        }

    def robot_safety_config_payload(self) -> dict[str, Any]:
        safety_defaults = dict(self.arm_safety_defaults_provider() or {})
        return {
            "arm_joint_limits_rad": ["-pi", "pi"],
            "finger_limits_rad": [0.0, 1.5],
            "arm_max_velocity_rad_per_step": 0.5,
            "finger_max_velocity_rad_per_step": 0.3,
            "estop_keywords": safety_defaults.get("estop_words", []),
        }

    def health_payload(
        self,
        *,
        version: str,
        python_version: str,
        uptime_seconds: float,
    ) -> dict[str, Any]:
        cfg = self.config()
        robot_cfg = self.section("robot")
        voice_cfg = self.section("voice")
        return {
            "status": "ok",
            "version": version,
            "python": python_version,
            "subsystems": {
                "brain": True,
                "robot": robot_cfg.get("enabled", False),
                "voice": bool(voice_cfg),
                "memory": cfg.get("memory", {}).get("enabled", False),
            },
            "uptime_seconds": uptime_seconds,
        }

    def skills_catalog_payload(self) -> dict[str, Any]:
        manager = self.skill_manager_provider()
        skills = manager.get_contract_catalog()
        return {"skills": skills, "count": len(skills)}

    def skills_openapi_payload(self) -> dict[str, Any]:
        manager = self.skill_manager_provider()
        return dict(manager.openapi_document())

    def sanitized_config_payload(self) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for section_name, section_val in self.config().items():
            if isinstance(section_val, dict):
                sanitized[section_name] = {
                    key: value
                    for key, value in section_val.items()
                    if "key" not in key.lower() and "secret" not in key.lower()
                }
            else:
                sanitized[section_name] = section_val
        return sanitized


_CURRENT_SURFACE = MCPResourceSurface()


def default_resource_surface() -> MCPResourceSurface:
    """Return a fresh default surface backed by legacy fallback suppliers."""

    return MCPResourceSurface()


def get_resource_surface() -> MCPResourceSurface:
    """Return the active resource dependency surface."""

    return _CURRENT_SURFACE


def set_resource_surface(surface: MCPResourceSurface) -> MCPResourceSurface:
    """Set the active resource surface and return the previous one."""

    global _CURRENT_SURFACE
    previous = _CURRENT_SURFACE
    _CURRENT_SURFACE = surface
    return previous


def resource_surface_from_context(ctx: Any) -> MCPResourceSurface:
    """Create a resource surface backed by an MCP context object."""

    return MCPResourceSurface(
        config_provider=lambda: dict(getattr(ctx, "config", None) or {}),
        section_provider=lambda name: _section_from_context(ctx, name),
        skill_manager_provider=lambda: _skill_manager_from_context(ctx),
    )


def _section_from_context(ctx: Any, name: str) -> Mapping[str, Any]:
    config = getattr(ctx, "config", None)
    if isinstance(config, dict):
        section = config.get(name, {})
        if isinstance(section, dict):
            return section
    return {}


def _skill_manager_from_context(ctx: Any) -> Any:
    manager = getattr(ctx, "skill_manager", None)
    if manager is not None:
        return manager
    return _default_skill_manager_provider()


__all__ = [
    "MCPResourceSurface",
    "default_resource_surface",
    "get_resource_surface",
    "resource_surface_from_context",
    "set_resource_surface",
]
