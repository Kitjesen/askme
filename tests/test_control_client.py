"""Tests for DogControlClient — config, dispatch, error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from askme.robot.control_client import DogControlClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(**kwargs) -> DogControlClient:
    cfg = {"base_url": "http://localhost:5080", "bearer_token": "test-token", **kwargs}
    return DogControlClient(config=cfg)


# ── Init ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_base_url_from_config(self):
        client = _make_client()
        assert client._base_url == "http://localhost:5080"

    def test_trailing_slash_stripped(self):
        client = DogControlClient(config={"base_url": "http://localhost:5080/"})
        assert client._base_url == "http://localhost:5080"

    def test_bearer_token_from_config(self):
        client = _make_client(bearer_token="my-token")
        assert client._token == "my-token"

    def test_operator_id_default(self):
        client = _make_client()
        assert client._operator_id == "askme"

    def test_operator_id_custom(self):
        client = DogControlClient(config={"base_url": "http://host", "operator_id": "operator-1"})
        assert client._operator_id == "operator-1"

    def test_env_url_override(self, monkeypatch):
        monkeypatch.setenv("DOG_CONTROL_SERVICE_URL", "http://envhost:9999")
        client = DogControlClient()
        assert client._base_url == "http://envhost:9999"

    def test_env_token_override(self, monkeypatch):
        monkeypatch.setenv("RUNTIME_BEARER_TOKEN", "env-token")
        client = DogControlClient(config={"base_url": "http://host"})
        assert client._token == "env-token"

    def test_env_operator_id(self, monkeypatch):
        monkeypatch.setenv("RUNTIME_OPERATOR_ID", "env-operator")
        client = DogControlClient(config={"base_url": "http://host"})
        assert client._operator_id == "env-operator"


# ── is_configured ─────────────────────────────────────────────────────────────

class TestIsConfigured:
    def test_configured_when_url_set(self):
        client = _make_client()
        assert client.is_configured() is True

    def test_not_configured_when_no_url(self):
        client = DogControlClient(config={})
        assert client.is_configured() is False

    def test_not_configured_empty_url(self):
        client = DogControlClient(config={"base_url": ""})
        assert client.is_configured() is False


# ── dispatch_capability ───────────────────────────────────────────────────────

class TestDispatchCapability:
    def test_not_configured_returns_error(self):
        client = DogControlClient(config={})
        result = client.dispatch_capability("stand")
        assert "error" in result
        assert "not configured" in result["error"]

    def test_success_returns_json(self):
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "accepted", "execution_id": "abc"}
        with patch("requests.post", return_value=mock_resp):
            result = client.dispatch_capability("stand")
        assert result["status"] == "accepted"

    def test_sends_correct_headers(self):
        client = _make_client(bearer_token="my-bearer")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.dispatch_capability("sit")
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-bearer"
        assert "X-Request-Id" in headers
        assert "Idempotency-Key" in headers

    def test_no_token_no_auth_header(self):
        client = DogControlClient(config={"base_url": "http://host:5080"})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.dispatch_capability("sit")
        headers = mock_post.call_args[1]["headers"]
        assert "Authorization" not in headers

    def test_timeout_returns_error(self):
        client = _make_client()
        with patch("requests.post", side_effect=requests.Timeout()):
            result = client.dispatch_capability("stand")
        assert "error" in result
        assert "timeout" in result["error"]

    def test_request_exception_returns_error(self):
        client = _make_client()
        with patch("requests.post", side_effect=requests.RequestException("connection error")):
            result = client.dispatch_capability("move")
        assert "error" in result

    def test_capability_in_request_body(self):
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.dispatch_capability("patrol", params={"speed": 0.5})
        body = mock_post.call_args[1]["json"]
        assert body["requested_capability"] == "patrol"
        assert body["parameters"] == {"speed": 0.5}

    def test_no_params_defaults_empty_dict(self):
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.dispatch_capability("stand")
        body = mock_post.call_args[1]["json"]
        assert body["parameters"] == {}

    def test_non_json_response_returns_ok(self):
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("no json")
        with patch("requests.post", return_value=mock_resp):
            result = client.dispatch_capability("stand")
        assert result["status"] == "ok"
        assert result["http_status"] == 200

    def test_http_error_raises_request_exception(self):
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
        mock_resp.status_code = 500
        with patch("requests.post", return_value=mock_resp):
            result = client.dispatch_capability("stand")
        assert "error" in result
