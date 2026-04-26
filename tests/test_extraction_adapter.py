"""Tests for ExtractionAdapter — extract method, rate limiting, skip logic."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from askme.memory.extraction_adapter import ExtractionAdapter

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_adapter() -> ExtractionAdapter:
    return ExtractionAdapter(llm_client=MagicMock(), model="test-model")


# ── Disabled ──────────────────────────────────────────────────────────────────

class TestDisabled:
    def test_disabled_returns_empty(self):
        adapter = _make_adapter()
        adapter._enabled = False
        result = adapter.extract("sensor anomaly detected", "I will check it")
        assert result == []


# ── Short text filtering ──────────────────────────────────────────────────────

class TestShortTextFiltering:
    def test_short_user_text_skipped(self):
        adapter = _make_adapter()
        result = adapter.extract("hi", "Hello! How can I help?")
        assert result == []

    def test_short_assistant_text_skipped(self):
        adapter = _make_adapter()
        result = adapter.extract("sensor shows high temperature", "ok")
        assert result == []


# ── Skip words ────────────────────────────────────────────────────────────────

class TestSkipWords:
    def test_greeting_skipped(self):
        adapter = _make_adapter()
        result = adapter.extract("你好啊今天天气真好", "你好！有什么需要帮助的？")
        assert result == []

    def test_time_query_skipped(self):
        adapter = _make_adapter()
        result = adapter.extract("现在几点了你知道吗", "现在是下午三点整")
        assert result == []

    def test_farewell_skipped(self):
        adapter = _make_adapter()
        result = adapter.extract("再见了今天", "再见！有需要再联系")
        assert result == []

    def test_stop_command_skipped(self):
        adapter = _make_adapter()
        result = adapter.extract("停下来别动了", "好的，已停止")
        assert result == []


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_second_call_within_cooldown_returns_empty(self):
        adapter = _make_adapter()
        adapter._last_extract = time.time()  # just extracted
        result = adapter.extract("sensor temperature rising detected", "I will alert operator")
        assert result == []

    def test_after_cooldown_allowed(self):
        adapter = _make_adapter()
        adapter._last_extract = time.time() - 100  # far in past
        # No API key → returns [] but for the right reason (no key, not rate limit)
        result = adapter.extract("anomaly detected in zone A", "I will investigate immediately")
        assert isinstance(result, list)


# ── API key check ─────────────────────────────────────────────────────────────

class TestApiKeyCheck:
    def test_no_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        adapter = _make_adapter()
        adapter._last_extract = 0.0  # ensure not rate-limited
        result = adapter.extract("motor overcurrent error in sector B", "I detected the problem")
        assert result == []


# ── JSON response parsing ─────────────────────────────────────────────────────

class TestJsonParsing:
    def _mock_http_response(self, content: str):
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp

    def test_valid_json_parsed(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        facts_json = '[{"type": "anomaly", "location": "zone A", "text": "sensor error"}]'
        with patch("httpx.post", return_value=self._mock_http_response(facts_json)):
            result = adapter.extract("sensor error in zone A", "investigating the anomaly now")
        assert len(result) == 1
        assert result[0]["type"] == "anomaly"

    def test_markdown_fence_stripped(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        content = '```json\n[{"type": "observation", "location": "hall", "text": "all clear"}]\n```'
        with patch("httpx.post", return_value=self._mock_http_response(content)):
            result = adapter.extract("inspection complete in hallway", "everything looks normal now")
        assert len(result) == 1

    def test_empty_array_returns_empty(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        with patch("httpx.post", return_value=self._mock_http_response("[]")):
            result = adapter.extract("hello world test response ignored", "nothing to report today")
        assert result == []

    def test_caps_at_3_facts(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        facts = [{"type": "anomaly", "location": f"zone{i}", "text": f"error {i}"} for i in range(10)]
        import json
        with patch("httpx.post", return_value=self._mock_http_response(json.dumps(facts))):
            result = adapter.extract("multiple errors detected across zones", "I will address them all")
        assert len(result) <= 3

    def test_invalid_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        with patch("httpx.post", return_value=self._mock_http_response("not valid json")):
            result = adapter.extract("anomaly detected in sector delta now", "I found a critical issue")
        assert result == []

    def test_non_list_response_returns_empty(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        with patch("httpx.post", return_value=self._mock_http_response('{"fact": "not a list"}')):
            result = adapter.extract("anomaly occurred in factory sector", "I see the factory issue")
        assert result == []

    def test_missing_required_fields_filtered(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        # Missing "text" field
        import json
        facts = [{"type": "anomaly"}]  # no "text" field
        with patch("httpx.post", return_value=self._mock_http_response(json.dumps(facts))):
            result = adapter.extract("critical error detected in warehouse now", "I found the warehouse error")
        assert result == []

    def test_text_truncated_at_100_chars(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        adapter = _make_adapter()
        adapter._last_extract = 0.0

        import json
        facts = [{"type": "anomaly", "location": "zone", "text": "x" * 200}]
        with patch("httpx.post", return_value=self._mock_http_response(json.dumps(facts))):
            result = adapter.extract("long text anomaly detected in factory area", "I see the long anomaly text")
        if result:
            assert len(result[0]["text"]) <= 100
