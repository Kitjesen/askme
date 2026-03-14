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


def _ddg_resp(abstract: str = "", answer: str = "", topics: list | None = None) -> bytes:
    return json.dumps({
        "AbstractText": abstract,
        "AbstractURL": "https://en.wikipedia.org/wiki/Test" if abstract else "",
        "Answer": answer,
        "RelatedTopics": topics or [],
    }, ensure_ascii=False).encode("utf-8")


def _mock_search_resp(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_search_returns_abstract(search_tool: WebSearchTool) -> None:
    body = _ddg_resp(abstract="Python is a programming language.", answer="")
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(body)):
        result = search_tool.execute(query="Python programming")
    assert "Python is a programming language" in result
    assert "摘要" in result


def test_search_returns_direct_answer(search_tool: WebSearchTool) -> None:
    body = _ddg_resp(answer="42")
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(body)):
        result = search_tool.execute(query="meaning of life")
    assert "42" in result


def test_search_returns_related_topics(search_tool: WebSearchTool) -> None:
    topics = [
        {"Text": "Python - a high-level language", "FirstURL": "https://en.wikipedia.org/wiki/Python"},
        {"Text": "Python 3 features", "FirstURL": "https://docs.python.org/3/"},
    ]
    body = _ddg_resp(topics=topics)
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(body)):
        result = search_tool.execute(query="Python")
    assert "Python" in result
    assert "wikipedia" in result.lower() or "docs.python" in result.lower()


def test_search_no_results_returns_fallback(search_tool: WebSearchTool) -> None:
    body = _ddg_resp()  # all empty
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(body)):
        result = search_tool.execute(query="xyzzy_nonexistent_term_12345")
    assert "未找到" in result or "建议" in result


def test_search_empty_query_returns_error(search_tool: WebSearchTool) -> None:
    result = search_tool.execute(query="")
    assert "[Error]" in result


def test_search_url_error_handled(search_tool: WebSearchTool) -> None:
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("network unreachable")):
        result = search_tool.execute(query="test")
    assert "[Error]" in result


def test_search_timeout_handled(search_tool: WebSearchTool) -> None:
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        result = search_tool.execute(query="test")
    assert "[Error]" in result
    assert "超时" in result


def test_search_limits_related_topics_to_five(search_tool: WebSearchTool) -> None:
    topics = [{"Text": f"Topic {i}", "FirstURL": f"https://example.com/{i}"} for i in range(10)]
    body = _ddg_resp(topics=topics)
    with patch("urllib.request.urlopen", return_value=_mock_search_resp(body)):
        result = search_tool.execute(query="test")
    # Should include at most 5 topics
    assert result.count("example.com") <= 5


# ── HTML fallback ─────────────────────────────────────────────────────────────


def test_search_html_fallback_called_on_empty_instant_answer(search_tool: WebSearchTool) -> None:
    """When DDG API returns empty, _html_fallback is invoked automatically."""
    empty_body = _ddg_resp()  # no abstract, no answer, no topics

    ddg_html = (
        b'<a class="result__snippet">Python asyncio is a library for async I/O</a>'
        b'<a class="result__url">docs.python.org/3/library/asyncio.html</a>'
    )

    html_resp = MagicMock()
    html_resp.read.return_value = ddg_html
    html_resp.__enter__ = lambda s: s
    html_resp.__exit__ = MagicMock(return_value=False)

    call_count = [0]

    def _side_effect(req, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_search_resp(empty_body)  # first call: Instant Answer API
        return html_resp  # second call: HTML fallback

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        result = search_tool.execute(query="Python asyncio")

    assert call_count[0] == 2, "HTML fallback should make a second request"
    assert "asyncio" in result.lower()
    assert "docs.python.org" in result or "搜索结果" in result


def test_search_html_fallback_returns_default_on_parse_failure(search_tool: WebSearchTool) -> None:
    """If HTML fallback also fails, returns a clean fallback message (no crash)."""
    import urllib.error
    empty_body = _ddg_resp()

    call_count = [0]

    def _side_effect(req, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_search_resp(empty_body)
        raise urllib.error.URLError("html fallback unreachable")

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        result = search_tool.execute(query="xyzzy_nonexistent")

    assert "[Error]" not in result  # must not crash with an error prefix
    assert isinstance(result, str)
    assert len(result) > 0


def test_search_html_fallback_limits_to_five_results(search_tool: WebSearchTool) -> None:
    """HTML fallback returns at most 5 results."""
    empty_body = _ddg_resp()

    snippets = "".join(
        f'<a class="result__snippet">Result {i}</a>'
        f'<a class="result__url">example.com/{i}</a>'
        for i in range(10)
    ).encode()

    html_resp = MagicMock()
    html_resp.read.return_value = snippets
    html_resp.__enter__ = lambda s: s
    html_resp.__exit__ = MagicMock(return_value=False)

    call_count = [0]

    def _side_effect(req, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_search_resp(empty_body)
        return html_resp

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        result = search_tool.execute(query="test query")

    assert result.count("example.com") <= 5


# ── create_skill in agent allowed tools ──────────────────────────────────────


def test_create_skill_in_agent_allowed_tools() -> None:
    """create_skill must be in ThunderAgentShell's allowed tool set."""
    from askme.agent_shell.thunder_agent_shell import _AGENT_ALLOWED_TOOLS
    assert "create_skill" in _AGENT_ALLOWED_TOOLS


# ── Bing fallback (tier 3) ────────────────────────────────────────────────────


def test_bing_fallback_returns_results(search_tool: WebSearchTool) -> None:
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


def test_bing_fallback_returns_default_on_network_error(search_tool: WebSearchTool) -> None:
    """_bing_fallback returns a clean default if Bing is unreachable."""
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("unreachable")):
        result = search_tool._bing_fallback("anything")
    assert "[Error]" not in result
    assert isinstance(result, str) and len(result) > 0


def test_bing_fallback_called_when_ddg_html_fails(search_tool: WebSearchTool) -> None:
    """Third-tier Bing fallback is called when both DDG API and DDG HTML fail."""
    import urllib.error
    empty_body = _ddg_resp()
    bing_html = (
        b'<li class="b_algo"><p>Result from Bing</p>'
        b'<cite>example.com/result</cite></li>'
    )

    call_count = [0]

    def _side_effect(req, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_search_resp(empty_body)   # DDG API: empty
        if call_count[0] == 2:
            raise urllib.error.URLError("ddg html down")  # DDG HTML: fail
        # Bing call
        resp = MagicMock()
        resp.read.return_value = bing_html
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        result = search_tool.execute(query="python")

    assert call_count[0] == 3  # DDG API + DDG HTML + Bing
    assert isinstance(result, str)


# ── DDG HTML real URL extraction ──────────────────────────────────────────────


def test_ddg_html_extracts_real_urls_from_result_a_hrefs(search_tool: WebSearchTool) -> None:
    """_html_fallback decodes real URLs from DDG redirect hrefs (uddg= param).

    The old approach used result__url display text (e.g. "docs.python.org"),
    which is not a valid URL for web_fetch.  New approach reads result__a href
    and decodes the uddg= query parameter to get the real URL.
    """
    ddg_html = (
        b'<a class="result__a" href="/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Fasyncio.html">'
        b"asyncio docs</a>"
        b'<a class="result__snippet">asyncio is a library for async I/O in Python</a>'
    )
    resp = MagicMock()
    resp.read.return_value = ddg_html
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=resp):
        result = search_tool._html_fallback("python asyncio")

    # Must contain the real decoded URL, not the display text
    assert "https://docs.python.org/3/asyncio.html" in result
    assert "asyncio is a library" in result


def test_ddg_html_direct_http_href_used_as_url(search_tool: WebSearchTool) -> None:
    """_html_fallback uses direct http href when no DDG redirect encoding present."""
    ddg_html = (
        b'<a class="result__a" href="https://www.example.com/page">Example</a>'
        b'<a class="result__snippet">An example page with content</a>'
    )
    resp = MagicMock()
    resp.read.return_value = ddg_html
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=resp):
        result = search_tool._html_fallback("example")

    assert "https://www.example.com/page" in result
    assert "example page" in result.lower()


def test_ddg_html_snippet_shown_even_without_href(search_tool: WebSearchTool) -> None:
    """If no result__a hrefs found, snippets are still shown (no URL line appended)."""
    ddg_html = b'<a class="result__snippet">Python is a great language</a>'
    resp = MagicMock()
    resp.read.return_value = ddg_html
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=resp):
        result = search_tool._html_fallback("python language")

    assert "Python is a great language" in result
    # No URL line since no result__a was present
    assert "http" not in result
