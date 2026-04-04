"""Tests for health_check MCP resource and skill/config resources."""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest


# ── health_check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_json_string(self):
        from askme.mcp.resources.health_resources import health_check
        with patch("askme.mcp.resources.health_resources.get_config", return_value={}), \
             patch("askme.mcp.resources.health_resources.get_section", return_value={}):
            result = health_check()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_status_ok(self):
        from askme.mcp.resources.health_resources import health_check
        with patch("askme.mcp.resources.health_resources.get_config", return_value={}), \
             patch("askme.mcp.resources.health_resources.get_section", return_value={}):
            result = health_check()
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_contains_version(self):
        from askme.mcp.resources.health_resources import health_check
        with patch("askme.mcp.resources.health_resources.get_config", return_value={}), \
             patch("askme.mcp.resources.health_resources.get_section", return_value={}):
            result = health_check()
        data = json.loads(result)
        assert "version" in data

    def test_contains_uptime(self):
        from askme.mcp.resources.health_resources import health_check
        with patch("askme.mcp.resources.health_resources.get_config", return_value={}), \
             patch("askme.mcp.resources.health_resources.get_section", return_value={}):
            result = health_check()
        data = json.loads(result)
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_robot_enabled_from_config(self):
        from askme.mcp.resources.health_resources import health_check
        with patch("askme.mcp.resources.health_resources.get_config", return_value={}), \
             patch("askme.mcp.resources.health_resources.get_section", return_value={"enabled": True}):
            result = health_check()
        data = json.loads(result)
        assert data["subsystems"]["robot"] is True

    def test_memory_enabled_from_config(self):
        from askme.mcp.resources.health_resources import health_check
        with patch("askme.mcp.resources.health_resources.get_config",
                   return_value={"memory": {"enabled": True}}), \
             patch("askme.mcp.resources.health_resources.get_section", return_value={}):
            result = health_check()
        data = json.loads(result)
        assert data["subsystems"]["memory"] is True


# ── askme_config ──────────────────────────────────────────────────────────────

class TestAskmeConfig:
    def test_returns_json(self):
        from askme.mcp.resources.skill_resources import askme_config
        with patch("askme.config.get_config",
                   return_value={"brain": {"model": "gpt-4", "api_key": "secret"}}):
            result = askme_config()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_sanitizes_api_keys(self):
        from askme.mcp.resources.skill_resources import askme_config
        with patch("askme.config.get_config",
                   return_value={"brain": {"model": "gpt-4", "api_key": "my_secret_key"}}):
            result = askme_config()
        data = json.loads(result)
        assert "api_key" not in data.get("brain", {})

    def test_sanitizes_secret_fields(self):
        from askme.mcp.resources.skill_resources import askme_config
        with patch("askme.config.get_config",
                   return_value={"service": {"url": "http://localhost", "secret_token": "abc"}}):
            result = askme_config()
        data = json.loads(result)
        assert "secret_token" not in data.get("service", {})

    def test_preserves_non_sensitive_fields(self):
        from askme.mcp.resources.skill_resources import askme_config
        with patch("askme.config.get_config",
                   return_value={"robot": {"enabled": True, "simulate": False}}):
            result = askme_config()
        data = json.loads(result)
        assert data["robot"]["enabled"] is True
        assert data["robot"]["simulate"] is False
