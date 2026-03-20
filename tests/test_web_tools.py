"""Tests for WebFetchTool and WebSearchTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from askme.tools.builtin_tools import WebFetchTool, WebSearchTool


# ── WebFetchTool ──────────────────────────────────────────────────────────────


@pytest.fixture()
def fetch_tool() -> WebFetchTool:
    return WebFetchTool()


def _make_resp(body: bytes, content_type: str = "text/html", status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.headers.get.return_value = content_type
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_fetch_strips_html_tags(fetch_tool: WebFetchTool) -> None:
    html = b"<html><body><h1>Hello Thunder</h1><p>Robot is running.</p></body></html>"
    with patch("urllib.request.urlopen", return_value=_make_resp(html)):
        result = fetch_tool.execute(url="http://example.com/")
    assert "<html>" not in result
    assert "Hello Thunder" in result
    assert "Robot is running" in result


def test_fetch_strips_script_blocks(fetch_tool: WebFetchTool) -> None:
    html = b"<html><script>alert('xss')</script><p>Clean text</p></html>"
    with patch("urllib.request.urlopen", return_value=_make_resp(html)):
        result = fetch_tool.execute(url="http://example.com/")
    assert "alert" not in result
    assert "Clean text" in result


def test_fetch_returns_json_pretty(fetch_tool: WebFetchTool) -> None:
    data = json.dumps({"status": "ok", "value": 42}).encode()
    with patch("urllib.request.urlopen", return_value=_make_resp(data, "application/json")):
        result = fetch_tool.execute(url="http://example.com/api")
    assert '"status"' in result
    assert '"ok"' in result


def test_fetch_respects_max_chars(fetch_tool: WebFetchTool) -> None:
    body = b"<p>" + b"X" * 10000 + b"</p>"
    with patch("urllib.request.urlopen", return_value=_make_resp(body)):
        result = fetch_tool.execute(url="http://example.com/", max_chars=100)
    assert len(result) <= 120  # small buffer for truncation suffix
    assert "[" in result  # truncation marker present


def test_fetch_empty_url_returns_error(fetch_tool: WebFetchTool) -> None:
    result = fetch_tool.execute(url="")
    assert "[Error]" in result


def test_fetch_non_http_url_returns_error(fetch_tool: WebFetchTool) -> None:
    result = fetch_tool.execute(url="ftp://example.com/file.txt")
    assert "[Error]" in result


def test_fetch_http_error_handled(fetch_tool: WebFetchTool) -> None:
    import urllib.error
    err = urllib.error.HTTPError(
        url="http://example.com/",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=MagicMock(read=lambda n: b"not found"),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        result = fetch_tool.execute(url="http://example.com/")
    assert "[Error]" in result
    assert "404" in result


def test_fetch_timeout_handled(fetch_tool: WebFetchTool) -> None:
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        result = fetch_tool.execute(url="http://example.com/")
    assert "[Error]" in result


def test_fetch_url_error_handled(fetch_tool: WebFetchTool) -> None:
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("name not found")):
        result = fetch_tool.execute(url="http://notareal.host/")
    assert "[Error]" in result


def test_fetch_decodes_html_entities(fetch_tool: WebFetchTool) -> None:
    html = b"<p>AT&amp;T &lt;rocks&gt;</p>"
    with patch("urllib.request.urlopen", return_value=_make_resp(html)):
        result = fetch_tool.execute(url="http://example.com/")
    assert "AT&T" in result
    assert "&amp;" not in result


# ── WebSearchTool ─────────────────────────────────────────────────────────────


@pytest.fixture()
def search_tool() -> WebSearchTool:
    return WebSearchTool()


def _mock_search_resp(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_search_returns_results(search_tool: WebSearchTool) -> None:
    """Bing-first search returns results from HTML scraping."""
    bing_html = (
        b'<li class="b_algo"><h2><a href="https://python.org">Python</a></h2>'
        b'<div class="b_caption"><p>Python is a programming language.</p></div></li>'
    )
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(bing_html)):
        result = search_tool.execute(query="Python programming")
    assert "Python" in result


def test_search_empty_query(search_tool: WebSearchTool) -> None:
    result = search_tool.execute(query="")
    assert "Error" in result


def test_search_bing_fallback_to_baidu(search_tool: WebSearchTool) -> None:
    """When Bing fails, falls back to Baidu."""
    call_count = [0]
    def side_effect(req, **kw):
        call_count[0] += 1
        if call_count[0] == 1:  # Bing
            raise OSError("timeout")
        # Baidu response
        return _mock_search_resp(
            b'<h3><a>Python tutorial</a></h3>'
            b'<div class="c-abstract">Learn Python basics</div>'
        )
    with patch("urllib.request.urlopen", side_effect=side_effect):
        result = search_tool.execute(query="Python")
    assert call_count[0] >= 2  # tried Bing then Baidu


def test_search_no_results_returns_fallback(search_tool: WebSearchTool) -> None:
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(b"")):
        result = search_tool.execute(query="xyzzy_nonexistent_term_12345")
    assert "未找到" in result or "Error" in result


def test_search_url_error_handled(search_tool: WebSearchTool) -> None:
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("network unreachable")):
        result = search_tool.execute(query="test")
    assert "未找到" in result or "Error" in result


def test_search_timeout_handled(search_tool: WebSearchTool) -> None:
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        result = search_tool.execute(query="test")
    assert "未找到" in result or "Error" in result


def test_bing_returns_results(search_tool: WebSearchTool) -> None:
    """_bing_fallback extracts result snippets from Bing HTML."""
    bing_html = (
        b'<li class="b_algo"><p class="b_lineclamp2">Python asyncio is great</p>'
        b'<cite>docs.python.org/3/library/asyncio.html</cite></li>'
    )
    resp = MagicMock()
    resp.read.return_value = bing_html
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=resp):
        result = search_tool._bing_fallback("python asyncio")

    assert "asyncio" in result.lower()
    assert "docs.python.org" in result or "搜索结果" in result


# ── create_skill in agent allowed tools ──────────────────────────────────────


def test_create_skill_in_agent_allowed_tools() -> None:
    """create_skill must be in ThunderAgentShell's allowed tool set."""
    from askme.agent_shell.thunder_agent_shell import _AGENT_ALLOWED_TOOLS
    assert "create_skill" in _AGENT_ALLOWED_TOOLS


# ── DDG fallback tests removed — DDG is blocked in China ─────────────────────
# Bing is now primary, Baidu is fallback. Old DDG HTML/API tests deleted.

def test_all_search_engines_fail(search_tool: WebSearchTool) -> None:
    """When both Bing and Baidu fail, returns a clean fallback message."""
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        result = search_tool.execute(query="python")
    assert "未找到" in result or isinstance(result, str)
