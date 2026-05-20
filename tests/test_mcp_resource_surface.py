"""Tests for the MCP resource dependency surface."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from askme.mcp.resource_surface import (
    MCPResourceSurface,
    resource_surface_from_context,
    set_resource_surface,
)


@contextmanager
def _using_surface(surface: MCPResourceSurface) -> Iterator[None]:
    previous = set_resource_surface(surface)
    try:
        yield
    finally:
        set_resource_surface(previous)


def test_robot_status_uses_injected_resource_surface_config() -> None:
    from askme.mcp.resources.robot_resources import robot_status

    surface = MCPResourceSurface(
        section_provider=lambda name: {
            "robot": {"enabled": True, "simulate": False, "serial_port": "COM9"},
        }.get(name, {}),
    )

    with _using_surface(surface):
        payload = json.loads(robot_status())

    assert payload["enabled"] is True
    assert payload["simulate"] is False
    assert payload["serial_port"] == "COM9"


def test_robot_safety_config_uses_injected_arm_defaults() -> None:
    from askme.mcp.resources.robot_resources import robot_safety_config

    surface = MCPResourceSurface(
        arm_safety_defaults_provider=lambda: {"estop_words": ["stop-now"]}
    )

    with _using_surface(surface):
        payload = json.loads(robot_safety_config())

    assert payload["estop_keywords"] == ["stop-now"]


def test_depth_info_uses_injected_depth_reader() -> None:
    from askme.mcp.resources.perception_resources import depth_info

    surface = MCPResourceSurface(
        depth_info_provider=lambda: {
            "daemon_alive": True,
            "center_depth_m": 1.25,
        }
    )

    with _using_surface(surface):
        payload = json.loads(depth_info())

    assert payload == {"daemon_alive": True, "center_depth_m": 1.25}


def test_resource_surface_from_context_uses_context_config_and_skill_manager() -> None:
    class SkillManager:
        def get_contract_catalog(self) -> list[dict[str, Any]]:
            return [{"name": "inspect"}]

        def openapi_document(self) -> dict[str, Any]:
            return {"paths": {"/skills/inspect": {}}}

    ctx = SimpleNamespace(
        config={"robot": {"enabled": True}, "memory": {"enabled": True}},
        skill_manager=SkillManager(),
    )

    surface = resource_surface_from_context(ctx)

    assert surface.section("robot") == {"enabled": True}
    assert surface.health_payload(version="1", python_version="3", uptime_seconds=0)[
        "subsystems"
    ]["memory"] is True
    assert surface.skills_catalog_payload()["count"] == 1
