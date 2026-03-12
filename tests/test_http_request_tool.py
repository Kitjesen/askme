"""Tests for HttpRequestTool — allowlist enforcement and basic request logic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from askme.tools.builtin_tools import HttpRequestTool


@pytest.fixture()
def tool() -> HttpRequestTool:
    return HttpRequestTool()


# ── Allowlist enforcement ────────────────────────────────────────────────────


def test_blocks_url_not_in_allowlist(tool: HttpRequestTool) -> None:
    with patch("askme.tools.builtin_tools._http_allowlist", return_value=["http://localhost:8080"]):
        result = tool.execute(method="GET", url="http://evil.com/steal")
    assert "[Error]" in result
    assert "白名单" in result


def test_allows_url_matching_prefix(tool: HttpRequestTool) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.headers.get.return_value = "application/json"
    mock_resp.read.return_value = b'{"ok": true}'
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("askme.tools.builtin_tools._http_allowlist", return_value=["http://localhost:8080"]):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(method="GET", url="http://localhost:8080/api/v1/status")

    data = json.loads(result)
    assert data["status"] == 200
    assert data["body"]["ok"] is True


def test_empty_allowlist_blocks_external(tool: HttpRequestTool) -> None:
    with patch("askme.tools.builtin_tools._http_allowlist", return_value=[]):
        result = tool.execute(method="GET", url="http://192.168.1.100:8080/api")
    assert "[Error]" in result


def test_empty_allowlist_allows_localhost(tool: HttpRequestTool) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.headers.get.return_value = "text/plain"
    mock_resp.read.return_value = b"pong"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("askme.tools.builtin_tools._http_allowlist", return_value=[]):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(method="GET", url="http://localhost/ping")

    data = json.loads(result)
    assert data["status"] == 200


# ── POST with body ────────────────────────────────────────────────────────────


def test_post_sends_json_body(tool: HttpRequestTool) -> None:
    import urllib.request as _ureq

    mock_resp = MagicMock()
    mock_resp.status = 201
    mock_resp.headers.get.return_value = "application/json"
    mock_resp.read.return_value = b'{"created": true}'
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured: list[_ureq.Request] = []

    def _fake_urlopen(req: _ureq.Request, timeout: float) -> MagicMock:
        captured.append(req)
        return mock_resp

    with patch("askme.tools.builtin_tools._http_allowlist", return_value=["http://localhost"]):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = tool.execute(
                method="POST",
                url="http://localhost:8080/api/v1/action",
                body={"action": "stand"},
            )

    assert len(captured) == 1
    req = captured[0]
    assert req.get_method() == "POST"
    sent_body = json.loads(req.data)
    assert sent_body["action"] == "stand"
    data = json.loads(result)
    assert data["body"]["created"] is True


# ── HTTP error handling ───────────────────────────────────────────────────────


def test_http_error_returns_status_and_body(tool: HttpRequestTool) -> None:
    import urllib.error

    err = urllib.error.HTTPError(
        url="http://localhost/bad",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=MagicMock(read=lambda n: b"not found"),
    )
    with patch("askme.tools.builtin_tools._http_allowlist", return_value=["http://localhost"]):
        with patch("urllib.request.urlopen", side_effect=err):
            result = tool.execute(method="GET", url="http://localhost/bad")

    data = json.loads(result)
    assert data["status"] == 404
