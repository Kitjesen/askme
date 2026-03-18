"""Tests for NavDispatchTool and DogControlDispatchTool."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from askme.tools.builtin_tools import (
    DogControlDispatchTool,
    NavDispatchTool,
    register_builtin_tools,
)
from askme.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a mock urllib response context-manager."""
    body = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Class 1: TestNavDispatchTool
# ---------------------------------------------------------------------------

class TestNavDispatchTool:
    def setup_method(self) -> None:
        self.tool = NavDispatchTool()

    def test_no_env_var_returns_unconfigured_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NAV_GATEWAY_URL", raising=False)
        result = self.tool.execute(destination="仓库A")
        assert "未配置" in result or "NAV_GATEWAY_URL" in result

    def test_empty_destination_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        result = self.tool.execute(destination="", task_type="navigate")
        assert result.startswith("[Error]")

    def test_successful_dispatch_returns_task_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        mock_resp = _make_mock_response({"session": {"mission_id": "abc123", "state": "submitted"}})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.execute(destination="仓库A")
        assert "abc123" in result

    def test_service_unreachable_returns_readable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = self.tool.execute(destination="仓库A")
        assert "不可达" in result or "NAV_GATEWAY_URL" in result

    def test_mapping_task_type_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        mock_resp = _make_mock_response({"session": {"mission_id": "map001", "state": "submitted"}})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.execute(destination="全区", task_type="mapping")
        assert "任务已下发" in result


# ---------------------------------------------------------------------------
# Class 2: TestDogControlDispatchTool
# ---------------------------------------------------------------------------

class TestDogControlDispatchTool:
    def setup_method(self) -> None:
        self.tool = DogControlDispatchTool()

    def test_no_service_url_returns_unconfigured_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOG_CONTROL_SERVICE_URL", raising=False)
        result = self.tool.execute(capability="stand")
        assert "未配置" in result

    def test_empty_capability_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://localhost:5080")
        result = self.tool.execute(capability="")
        assert result.startswith("[Error]")

    def test_successful_dispatch_stand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://localhost:5080")
        with patch(
            "askme.dog_control_client.DogControlClient.dispatch_capability",
            return_value={"status": "ok", "execution_id": "xyz"},
        ):
            result = self.tool.execute(capability="stand")
        assert "xyz" in result

    def test_service_returns_error_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://localhost:5080")
        with patch(
            "askme.dog_control_client.DogControlClient.dispatch_capability",
            return_value={"error": "service down"},
        ):
            result = self.tool.execute(capability="stand")
        assert "下发失败" in result

    def test_sit_capability_dispatched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://localhost:5080")
        with patch(
            "askme.dog_control_client.DogControlClient.dispatch_capability",
            return_value={"status": "ok", "execution_id": "sit-001"},
        ) as mock_dispatch:
            result = self.tool.execute(capability="sit")
        mock_dispatch.assert_called_once_with("sit", None)
        assert "sit" in result


# ---------------------------------------------------------------------------
# Class 3: TestToolRegistration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_nav_dispatch_registered(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        assert registry.get("nav_dispatch") is not None

    def test_dog_control_dispatch_registered(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        assert registry.get("dog_control_dispatch") is not None
