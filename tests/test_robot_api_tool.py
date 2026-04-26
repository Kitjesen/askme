"""Tests for RobotApiTool — service routing, HTTP error handling, metadata."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

from askme.tools.robot_api_tool import _SERVICE_PORTS, RobotApiTool

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tool() -> RobotApiTool:
    return RobotApiTool()


# ── Tool metadata ─────────────────────────────────────────────────────────────

class TestMetadata:
    def test_name(self):
        assert RobotApiTool.name == "robot_api"

    def test_agent_allowed(self):
        assert RobotApiTool.agent_allowed is True

    def test_required_params(self):
        required = RobotApiTool.parameters["required"]
        assert "service" in required
        assert "method" in required
        assert "path" in required

    def test_all_services_in_enum(self):
        enum_vals = RobotApiTool.parameters["properties"]["service"]["enum"]
        for svc in _SERVICE_PORTS:
            assert svc in enum_vals


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_unknown_service_returns_error(self):
        tool = _make_tool()
        result = tool.execute(service="unknown_svc", method="GET", path="/api/test")
        assert "[Error]" in result
        assert "unknown_svc" in result

    def test_empty_path_returns_error(self):
        tool = _make_tool()
        result = tool.execute(service="nav", method="GET", path="")
        assert "[Error]" in result


# ── HTTP success responses ────────────────────────────────────────────────────

class TestSuccessResponses:
    def _mock_urlopen(self, status: int, body: str, content_type: str = "application/json"):
        mock_resp = MagicMock()
        mock_resp.read = MagicMock(return_value=body.encode("utf-8"))
        mock_resp.status = status
        mock_resp.headers.get = MagicMock(return_value=content_type)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_json_response_parsed(self):
        tool = _make_tool()
        mock_resp = self._mock_urlopen(200, '{"ok": true}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(service="nav", method="GET", path="/api/v1/tasks")
        data = json.loads(result)
        assert data["status"] == 200
        assert data["body"]["ok"] is True

    def test_non_json_response_returned_as_text(self):
        tool = _make_tool()
        mock_resp = self._mock_urlopen(200, "plain text response", "text/plain")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(service="telemetry", method="GET", path="/health")
        data = json.loads(result)
        assert data["body"] == "plain text response"

    def test_post_with_body_sends_json(self):
        tool = _make_tool()
        mock_resp = self._mock_urlopen(201, '{"created": true}')
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            tool.execute(service="control", method="POST", path="/api/v1/posture",
                        body={"posture": "stand"})
        request = mock_open.call_args[0][0]
        assert request.data is not None
        assert b"posture" in request.data


# ── HTTP error responses ──────────────────────────────────────────────────────

class TestErrorResponses:
    def test_url_error_returns_unreachable_message(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            result = tool.execute(service="safety", method="GET", path="/api/v1/estop")
        assert "[Error]" in result
        assert "不可达" in result

    def test_timeout_error_returns_timeout_message(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            result = tool.execute(service="arbiter", method="GET", path="/api/v1/missions")
        assert "[Error]" in result
        assert "超时" in result

    def test_http_error_returns_status_and_body(self):
        tool = _make_tool()
        exc = urllib.error.HTTPError(
            url="http://localhost:5090/api/test",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MagicMock(read=MagicMock(return_value=b'{"detail": "not found"}')),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            result = tool.execute(service="nav", method="GET", path="/api/v1/nonexistent")
        data = json.loads(result)
        assert data["status"] == 404

    def test_generic_exception_returns_error(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = tool.execute(service="ops", method="GET", path="/api/v1/config")
        assert "[Error]" in result


# ── Service port mapping ──────────────────────────────────────────────────────

class TestServicePortMapping:
    def test_all_known_services_have_ports(self):
        expected_services = ["arbiter", "telemetry", "safety", "control", "nav", "arm", "ops"]
        for svc in expected_services:
            assert svc in _SERVICE_PORTS

    def test_ports_are_in_expected_range(self):
        for svc, port in _SERVICE_PORTS.items():
            assert 5000 <= port <= 6000, f"{svc} port {port} out of range"
