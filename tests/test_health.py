"""Tests for the health check MCP resource."""

import json


class TestHealthResource:
    def test_health_returns_valid_json(self):
        from askme.mcp_resources.health_resources import health_check

        result = health_check()
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_health_has_version(self):
        from askme.mcp_resources.health_resources import health_check

        data = json.loads(health_check())
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_subsystems(self):
        from askme.mcp_resources.health_resources import health_check

        data = json.loads(health_check())
        assert "subsystems" in data
        assert "brain" in data["subsystems"]
        assert "robot" in data["subsystems"]
        assert "voice" in data["subsystems"]

    def test_health_has_uptime(self):
        from askme.mcp_resources.health_resources import health_check

        data = json.loads(health_check())
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0
