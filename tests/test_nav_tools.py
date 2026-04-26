"""Unit tests for NavDispatchTool and NavStatusTool.

Tests verify:
1. Capability mapping (task_type → requested_capability)
2. Parameter building per task type
3. Request body format (mission_id, mission_type, requested_capability, parameters)
4. POST endpoint path (/api/v1/navigation/dispatch)
5. GET endpoint path (/api/v1/navigation/status)
6. Error handling (no URL, empty destination, HTTP errors)
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from askme.tools.builtin_tools import NavDispatchTool, NavStatusTool

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tool() -> NavDispatchTool:
    return NavDispatchTool()


@pytest.fixture()
def status_tool() -> NavStatusTool:
    return NavStatusTool()


# ── Capability mapping ────────────────────────────────────────────────────────

class TestCapabilityMapping:
    def test_navigate_capability(self, tool: NavDispatchTool) -> None:
        assert tool._CAPABILITY_MAP["navigate"] == "nav.semantic.execute"

    def test_mapping_capability(self, tool: NavDispatchTool) -> None:
        assert tool._CAPABILITY_MAP["mapping"] == "nav.mapping.start"

    def test_follow_person_capability(self, tool: NavDispatchTool) -> None:
        assert tool._CAPABILITY_MAP["follow_person"] == "nav.follow_person.start"

    def test_unknown_type_falls_back_to_semantic(self, tool: NavDispatchTool) -> None:
        cap = tool._CAPABILITY_MAP.get("unknown_type", "nav.semantic.execute")
        assert cap == "nav.semantic.execute"


# ── Parameter building ────────────────────────────────────────────────────────

class TestParameterBuilding:
    def test_navigate_uses_semantic_target(self, tool: NavDispatchTool) -> None:
        p = tool._build_parameters("navigate", "仓库A", None)
        assert p == {"semantic_target": "仓库A"}

    def test_navigate_ignores_extra_params(self, tool: NavDispatchTool) -> None:
        p = tool._build_parameters("navigate", "出口", {"extra": "ignored"})
        assert p == {"semantic_target": "出口"}

    def test_mapping_uses_map_scope(self, tool: NavDispatchTool) -> None:
        p = tool._build_parameters("mapping", "全区", {"map_scope": "B仓库"})
        assert p == {"map_name": "B仓库", "save_on_complete": True}

    def test_mapping_falls_back_to_destination(self, tool: NavDispatchTool) -> None:
        p = tool._build_parameters("mapping", "全厂区", None)
        assert p == {"map_name": "全厂区", "save_on_complete": True}

    def test_mapping_empty_map_scope_falls_back(self, tool: NavDispatchTool) -> None:
        p = tool._build_parameters("mapping", "A区", {"map_scope": ""})
        assert p == {"map_name": "A区", "save_on_complete": True}

    def test_follow_person_empty_params(self, tool: NavDispatchTool) -> None:
        p = tool._build_parameters("follow_person", "", None)
        assert p == {}


# ── Request body format ───────────────────────────────────────────────────────

class TestRequestBody:
    def _capture_request(self, tool: NavDispatchTool, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with a mock server and capture the sent body."""
        captured: dict[str, Any] = {}

        response_mock = MagicMock()
        response_mock.__enter__ = lambda s: s
        response_mock.__exit__ = MagicMock(return_value=False)
        response_mock.read.return_value = json.dumps(
            {"session": {"mission_id": "abc123", "state": "submitted"}}
        ).encode()
        response_mock.status = 201

        def fake_urlopen(req: urllib.request.Request, timeout: int) -> Any:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return response_mock

        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                tool.execute(**kwargs)

        return captured

    def test_navigate_body_fields(self, tool: NavDispatchTool) -> None:
        cap = self._capture_request(tool, destination="充电桩", task_type="navigate")
        body = cap["body"]
        assert "mission_id" in body
        assert body["mission_type"] == "voice_command"
        assert body["requested_capability"] == "nav.semantic.execute"
        assert body["parameters"] == {"semantic_target": "充电桩"}

    def test_mapping_body_fields(self, tool: NavDispatchTool) -> None:
        cap = self._capture_request(
            tool, destination="全区", task_type="mapping",
            params={"map_scope": "仓库B"}
        )
        body = cap["body"]
        assert body["requested_capability"] == "nav.mapping.start"
        assert body["parameters"]["map_name"] == "仓库B"
        assert body["parameters"]["save_on_complete"] is True

    def test_follow_person_body_fields(self, tool: NavDispatchTool) -> None:
        cap = self._capture_request(tool, destination="", task_type="follow_person")
        body = cap["body"]
        assert body["requested_capability"] == "nav.follow_person.start"
        assert body["parameters"] == {}

    def test_post_endpoint_path(self, tool: NavDispatchTool) -> None:
        cap = self._capture_request(tool, destination="出口", task_type="navigate")
        assert cap["url"] == "http://localhost:5070/api/v1/navigation/dispatch"

    def test_mission_id_is_generated(self, tool: NavDispatchTool) -> None:
        cap = self._capture_request(tool, destination="出口", task_type="navigate")
        mid = cap["body"]["mission_id"]
        assert isinstance(mid, str) and len(mid) == 16

    def test_response_parsed_correctly(self, tool: NavDispatchTool) -> None:
        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            response_mock = MagicMock()
            response_mock.__enter__ = lambda s: s
            response_mock.__exit__ = MagicMock(return_value=False)
            response_mock.read.return_value = json.dumps(
                {"session": {"mission_id": "test123", "state": "running"}}
            ).encode()
            with patch("urllib.request.urlopen", return_value=response_mock):
                result = tool.execute(destination="充电桩", task_type="navigate")
        assert "test123" in result
        assert "running" in result


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_no_url_returns_config_error(self, tool: NavDispatchTool) -> None:
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("NAV_GATEWAY_URL", None)
            result = tool.execute(destination="仓库A")
        assert "NAV_GATEWAY_URL" in result

    def test_empty_destination_for_navigate(self, tool: NavDispatchTool) -> None:
        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            result = tool.execute(destination="", task_type="navigate")
        assert "[Error]" in result

    def test_empty_destination_ok_for_follow_person(self, tool: NavDispatchTool) -> None:
        response_mock = MagicMock()
        response_mock.__enter__ = lambda s: s
        response_mock.__exit__ = MagicMock(return_value=False)
        response_mock.read.return_value = json.dumps(
            {"session": {"mission_id": "f1", "state": "submitted"}}
        ).encode()
        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            with patch("urllib.request.urlopen", return_value=response_mock):
                result = tool.execute(destination="", task_type="follow_person")
        assert "[Error]" not in result

    def test_http_error_reported(self, tool: NavDispatchTool) -> None:
        err = urllib.error.HTTPError(
            url="http://x", code=422, msg="Unprocessable",
            hdrs=MagicMock(), fp=MagicMock(read=lambda n: b"invalid payload"),
        )
        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            with patch("urllib.request.urlopen", side_effect=err):
                result = tool.execute(destination="X", task_type="navigate")
        assert "422" in result

    def test_url_error_reported(self, tool: NavDispatchTool) -> None:
        err = urllib.error.URLError(reason="Connection refused")
        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            with patch("urllib.request.urlopen", side_effect=err):
                result = tool.execute(destination="X", task_type="navigate")
        assert "不可达" in result


# ── NavStatusTool ─────────────────────────────────────────────────────────────

class TestNavStatusTool:
    def test_status_endpoint_path(self, status_tool: NavStatusTool) -> None:
        captured_url: list[str] = []

        response_mock = MagicMock()
        response_mock.__enter__ = lambda s: s
        response_mock.__exit__ = MagicMock(return_value=False)
        response_mock.read.return_value = json.dumps({"sessions": []}).encode()

        def fake_urlopen(url: str, timeout: int) -> Any:
            captured_url.append(url)
            return response_mock

        with patch.dict("os.environ", {"NAV_GATEWAY_URL": "http://localhost:5070"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                status_tool.execute()

        assert captured_url[0] == "http://localhost:5070/api/v1/navigation/status"

    def test_no_url_returns_error(self, status_tool: NavStatusTool) -> None:
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("NAV_GATEWAY_URL", None)
            result = status_tool.execute()
        assert "未配置" in result
