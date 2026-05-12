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

    def test_robot_api_stays_dangerous(self):
        assert RobotApiTool.safety_level == "dangerous"

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
        assert request.get_method() == "POST"
        assert request.full_url == "http://localhost:5080/api/v1/posture"
        assert request.data is not None
        assert b"posture" in request.data

    def test_prefers_canonical_service_env_urls(self, monkeypatch):
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://control-host:5080")
        monkeypatch.setenv("DOG_SAFETY_SERVICE_URL", "http://safety-host:5070")
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://nav-host:8088")

        tool = _make_tool()
        mock_resp = self._mock_urlopen(200, '{"ok": true}')
        captured: list[str] = []

        def fake_urlopen(req, timeout):
            captured.append(req.full_url)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            tool.execute(service="control", method="GET", path="/health")
            tool.execute(service="safety", method="GET", path="/health")
            tool.execute(service="nav", method="GET", path="/health")

        assert captured == [
            "http://control-host:5080/health",
            "http://safety-host:5070/health",
            "http://nav-host:8088/health",
        ]

    def test_prefers_runtime_bearer_token_for_auth(self, monkeypatch):
        monkeypatch.setenv("RUNTIME_BEARER_TOKEN", "runtime-token")
        monkeypatch.setenv("NOVA_DOG_RUNTIME_API_KEY", "nova-token")
        tool = _make_tool()
        mock_resp = self._mock_urlopen(200, '{"ok": true}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            tool.execute(service="nav", method="GET", path="/health")

        request = mock_open.call_args[0][0]
        assert request.headers["Authorization"] == "Bearer runtime-token"

    def test_legacy_runtime_api_key_still_used(self, monkeypatch):
        monkeypatch.delenv("RUNTIME_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("NOVA_DOG_RUNTIME_API_KEY", raising=False)
        monkeypatch.setenv("RUNTIME_API_KEY", "legacy-token")
        tool = _make_tool()
        mock_resp = self._mock_urlopen(200, '{"ok": true}')

        with patch("askme.tools.robot_api_tool.get_section", return_value={}):
            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
                tool.execute(service="nav", method="GET", path="/health")

        request = mock_open.call_args[0][0]
        assert request.headers["Authorization"] == "Bearer legacy-token"


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
            url="http://localhost:8088/api/test",
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

    def test_ports_match_canonical_runtime_map(self):
        assert _SERVICE_PORTS == {
            "arbiter": 5050,
            "telemetry": 5060,
            "safety": 5070,
            "control": 5080,
            "nav": 8088,
            "arm": 5100,
            "ops": 5110,
        }
