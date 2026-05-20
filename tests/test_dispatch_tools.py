"""Tests for NavDispatchTool and DogControlDispatchTool."""

from __future__ import annotations

import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from askme.tools.builtin_tools import (
    DogControlDispatchTool,
    NavDispatchTool,
    register_builtin_tools,
)
from askme.tools.tool_registry import ToolRegistry


def _make_mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a mock urllib response context-manager."""

    body = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class _FakeDogControlClient:
    def __init__(
        self,
        *,
        configured: bool = True,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.configured = configured
        self.result = result or {"status": "ok", "execution_id": "xyz"}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def is_configured(self) -> bool:
        return self.configured

    def dispatch_capability(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((capability, params))
        return self.result


class _FakeNavigationClient:
    def __init__(
        self,
        *,
        configured: bool = True,
        result: dict[str, Any] | None = None,
        status: dict[str, Any] | None = None,
    ) -> None:
        self.configured = configured
        self.result = result or {"session": {"mission_id": "nav-001", "state": "submitted"}}
        self.status_result = status or {"sessions": []}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def is_configured(self) -> bool:
        return self.configured

    def dispatch_navigation(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((capability, params))
        return self.result

    def status(self) -> dict[str, Any]:
        return self.status_result


class TestNavDispatchTool:
    def setup_method(self) -> None:
        self.tool = NavDispatchTool()

    def test_no_env_var_returns_unconfigured_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NAV_GATEWAY_URL", raising=False)
        result = self.tool.execute(destination="warehouse-a")
        assert "NAV_GATEWAY_URL" in result

    def test_empty_destination_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        result = self.tool.execute(destination="", task_type="navigate")
        assert result.startswith("[Error]")

    def test_successful_dispatch_returns_task_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        mock_resp = _make_mock_response({"session": {"mission_id": "abc123", "state": "submitted"}})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.execute(destination="warehouse-a")
        assert "abc123" in result

    def test_service_unreachable_returns_readable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = self.tool.execute(destination="warehouse-a")
        assert "NAV_GATEWAY_URL" in result

    def test_mapping_task_type_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
        mock_resp = _make_mock_response({"session": {"mission_id": "map001", "state": "submitted"}})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.execute(destination="full-area", task_type="mapping")
        assert "map001" in result

    def test_injected_navigation_client_handles_dispatch_without_env(self) -> None:
        client = _FakeNavigationClient()
        self.tool.set_navigation_client(client)

        with patch("urllib.request.urlopen") as mock:
            result = self.tool.execute(destination="warehouse-a")

        mock.assert_not_called()
        assert "nav-001" in result
        assert client.calls == [("nav.semantic.execute", {"semantic_target": "warehouse-a"})]

    def test_injected_navigation_client_error_maps_to_config_message(self) -> None:
        client = _FakeNavigationClient(result={"error": "NAV_GATEWAY_URL not configured"})
        self.tool.set_navigation_client(client)

        result = self.tool.execute(destination="warehouse-a")

        assert "NAV_GATEWAY_URL" in result


class TestDogControlDispatchTool:
    def setup_method(self) -> None:
        self.tool = DogControlDispatchTool()

    def test_no_service_url_returns_unconfigured_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOG_CONTROL_SERVICE_URL", raising=False)
        result = self.tool.execute(capability="stand")
        assert "DOG_CONTROL_SERVICE_URL" in result

    def test_empty_capability_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://localhost:5080")
        result = self.tool.execute(capability="")
        assert result.startswith("[Error]")

    def test_successful_dispatch_stand(self) -> None:
        self.tool.set_robot_control_client(_FakeDogControlClient())
        result = self.tool.execute(capability="stand")
        assert "xyz" in result

    def test_service_returns_error_propagated(self) -> None:
        self.tool.set_robot_control_client(
            _FakeDogControlClient(result={"error": "service down"})
        )
        result = self.tool.execute(capability="stand")
        assert "service down" in result

    def test_sit_capability_dispatched(self) -> None:
        client = _FakeDogControlClient(result={"status": "ok", "execution_id": "sit-001"})
        self.tool.set_robot_control_client(client)
        result = self.tool.execute(capability="sit")
        assert client.calls == [("sit", None)]
        assert "sit" in result

    def test_unconfigured_injected_client_returns_configuration_message(self) -> None:
        self.tool.set_robot_control_client(_FakeDogControlClient(configured=False))
        result = self.tool.execute(capability="stand")
        assert "DOG_CONTROL_SERVICE_URL" in result

    def test_env_var_alone_does_not_create_control_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://localhost:5080")

        result = self.tool.execute(capability="stand")

        assert "RobotControlPort" in result


class TestToolRegistration:
    def test_nav_dispatch_registered(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        assert registry.get("nav_dispatch") is not None

    def test_dog_control_dispatch_registered(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        assert registry.get("dog_control_dispatch") is not None

    def test_dog_control_client_can_be_injected_at_registration(self) -> None:
        registry = ToolRegistry()
        client = _FakeDogControlClient(result={"status": "ok", "execution_id": "reg-001"})
        register_builtin_tools(registry, robot_control_client=client)

        tool = registry.get("dog_control_dispatch")

        assert tool is not None
        assert "reg-001" in tool.execute(capability="stand")

    def test_navigation_client_can_be_injected_at_registration(self) -> None:
        registry = ToolRegistry()
        client = _FakeNavigationClient()
        register_builtin_tools(registry, navigation_client=client)

        tool = registry.get("nav_dispatch")
        status_tool = registry.get("nav_status")

        assert tool is not None
        assert status_tool is not None
        assert "nav-001" in tool.execute(destination="dock")
        assert "sessions" in status_tool.execute()
