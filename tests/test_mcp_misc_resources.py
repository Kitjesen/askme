"""Tests for health/config/skill MCP resources."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from askme.mcp.resource_surface import MCPResourceSurface, set_resource_surface


@contextmanager
def _using_surface(surface: MCPResourceSurface) -> Iterator[None]:
    previous = set_resource_surface(surface)
    try:
        yield
    finally:
        set_resource_surface(previous)


def _surface(
    *,
    config: dict[str, Any] | None = None,
    sections: dict[str, dict[str, Any]] | None = None,
    skill_manager: Any | None = None,
) -> MCPResourceSurface:
    return MCPResourceSurface(
        config_provider=lambda: config or {},
        section_provider=lambda name: (sections or {}).get(name, {}),
        skill_manager_provider=lambda: skill_manager,
    )


class _FakeSkillManager:
    def get_contract_catalog(self) -> list[dict[str, Any]]:
        return [{"name": "inspect"}, {"name": "navigate"}]

    def openapi_document(self) -> dict[str, Any]:
        return {"openapi": "3.1.0", "paths": {"/skills/inspect": {}}}


class TestHealthCheck:
    def test_returns_json_string(self) -> None:
        from askme.mcp.resources.health_resources import health_check

        with _using_surface(_surface()):
            result = health_check()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_status_ok(self) -> None:
        from askme.mcp.resources.health_resources import health_check

        with _using_surface(_surface()):
            result = health_check()
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_contains_version(self) -> None:
        from askme.mcp.resources.health_resources import health_check

        with _using_surface(_surface()):
            result = health_check()
        data = json.loads(result)
        assert "version" in data

    def test_contains_uptime(self) -> None:
        from askme.mcp.resources.health_resources import health_check

        with _using_surface(_surface()):
            result = health_check()
        data = json.loads(result)
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_robot_enabled_from_resource_surface(self) -> None:
        from askme.mcp.resources.health_resources import health_check

        with _using_surface(_surface(sections={"robot": {"enabled": True}})):
            result = health_check()
        data = json.loads(result)
        assert data["subsystems"]["robot"] is True

    def test_memory_enabled_from_resource_surface(self) -> None:
        from askme.mcp.resources.health_resources import health_check

        with _using_surface(_surface(config={"memory": {"enabled": True}})):
            result = health_check()
        data = json.loads(result)
        assert data["subsystems"]["memory"] is True


class TestAskmeConfig:
    def test_returns_json(self) -> None:
        from askme.mcp.resources.skill_resources import askme_config

        with _using_surface(_surface(config={"brain": {"model": "gpt-4"}})):
            result = askme_config()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_sanitizes_api_keys(self) -> None:
        from askme.mcp.resources.skill_resources import askme_config

        with _using_surface(
            _surface(config={"brain": {"model": "gpt-4", "api_key": "secret"}})
        ):
            result = askme_config()
        data = json.loads(result)
        assert "api_key" not in data.get("brain", {})

    def test_sanitizes_secret_fields(self) -> None:
        from askme.mcp.resources.skill_resources import askme_config

        with _using_surface(
            _surface(config={"service": {"url": "http://localhost", "secret_token": "abc"}})
        ):
            result = askme_config()
        data = json.loads(result)
        assert "secret_token" not in data.get("service", {})

    def test_preserves_non_sensitive_fields(self) -> None:
        from askme.mcp.resources.skill_resources import askme_config

        with _using_surface(_surface(config={"robot": {"enabled": True, "simulate": False}})):
            result = askme_config()
        data = json.loads(result)
        assert data["robot"]["enabled"] is True
        assert data["robot"]["simulate"] is False


class TestSkillResources:
    def test_skills_catalog_uses_injected_skill_manager(self) -> None:
        from askme.mcp.resources.skill_resources import skills_catalog

        with _using_surface(_surface(skill_manager=_FakeSkillManager())):
            result = skills_catalog()
        data = json.loads(result)
        assert data["count"] == 2
        assert data["skills"][0]["name"] == "inspect"

    def test_skills_openapi_uses_injected_skill_manager(self) -> None:
        from askme.mcp.resources.skill_resources import skills_openapi

        with _using_surface(_surface(skill_manager=_FakeSkillManager())):
            result = skills_openapi()
        data = json.loads(result)
        assert data["openapi"] == "3.1.0"
        assert "/skills/inspect" in data["paths"]
