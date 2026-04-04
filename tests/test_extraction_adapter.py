"""Tests for ExtractionAdapter — LLM-based fact extraction from conversation turns."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from askme.memory.extraction_adapter import ExtractionAdapter


def _make_adapter(**kwargs) -> ExtractionAdapter:
    return ExtractionAdapter(llm_client=MagicMock(), **kwargs)


class TestInit:
    def test_enabled_by_default(self):
        a = _make_adapter()
        assert a._enabled is True

    def test_default_model(self):
        a = _make_adapter()
        assert a._model == "qwen-turbo"

    def test_custom_model(self):
        a = _make_adapter(model="gpt-4")
        assert a._model == "gpt-4"


class TestEarlyReturns:
    def test_disabled_returns_empty(self):
        a = _make_adapter()
        a._enabled = False
        result = a.extract("仓库A出现漏水", "已记录，立即派人处理")
        assert result == []

    def test_short_user_text_skipped(self):
        a = _make_adapter()
        result = a.extract("嗯", "好的我知道了")
        assert result == []

    def test_short_assistant_text_skipped(self):
        a = _make_adapter()
        result = a.extract("仓库A出现漏水现象需要处理", "好")
        assert result == []

    def test_greeting_skip_word_你好(self):
        a = _make_adapter()
        a._last_extract = 0.0  # ensure cooldown not active
        result = a.extract("你好机器人", "你好主人很高兴见到您")
        assert result == []

    def test_skip_word_几点(self):
        a = _make_adapter()
        a._last_extract = 0.0
        result = a.extract("现在几点了请告诉我", "现在是下午三点整")
        assert result == []

    def test_skip_word_静音(self):
        a = _make_adapter()
        a._last_extract = 0.0
        result = a.extract("请你静音谢谢", "好的我会静音的")
        assert result == []

    def test_cooldown_returns_empty(self):
        a = _make_adapter()
        a._last_extract = time.time()  # just extracted
        result = a.extract("仓库A出现漏水现象需要处理", "已记录立即处理")
        assert result == []

    def test_missing_api_key_returns_empty(self):
        a = _make_adapter()
        a._last_extract = 0.0
        with patch.dict("os.environ", {}, clear=True):
            # No DASHSCOPE_API_KEY
            result = a.extract("仓库A出现漏水现象需要处理", "已记录立即联系维修人员")
        assert result == []


class TestJsonParsing:
    """Test JSON cleaning logic by patching the HTTP call."""

    def _fake_http(self, response_text: str):
        """Return a context manager that patches httpx.post."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": response_text}}]
        }
        return patch("httpx.post", return_value=mock_resp)

    def test_clean_json_array_parsed(self):
        a = _make_adapter()
        a._last_extract = 0.0
        facts_json = '[{"type": "anomaly", "location": "仓库A", "text": "漏水"}]'
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http(facts_json):
                result = a.extract("仓库A有漏水现象需要处理", "已记录立即联系维修人员")
        assert len(result) == 1
        assert result[0]["type"] == "anomaly"
        assert result[0]["location"] == "仓库A"
        assert "漏水" in result[0]["text"]

    def test_markdown_fenced_json_parsed(self):
        a = _make_adapter()
        a._last_extract = 0.0
        fenced = '```json\n[{"type": "observation", "location": "general", "text": "ok"}]\n```'
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http(fenced):
                result = a.extract("一切正常没有问题状态良好", "系统运行正常所有传感器工作")
        assert len(result) == 1
        assert result[0]["type"] == "observation"

    def test_empty_array_returns_empty(self):
        a = _make_adapter()
        a._last_extract = 0.0
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http("[]"):
                result = a.extract("一切都很好今天天气不错很开心", "好的没有发现异常情况")
        assert result == []

    def test_max_three_facts_enforced(self):
        a = _make_adapter()
        a._last_extract = 0.0
        many = [
            {"type": "anomaly", "location": f"loc{i}", "text": f"issue{i}"}
            for i in range(5)
        ]
        import json
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http(json.dumps(many)):
                result = a.extract("发现多个问题需要处理立即行动", "已经记录所有问题立即处理各个位置")
        assert len(result) <= 3

    def test_invalid_json_returns_empty(self):
        a = _make_adapter()
        a._last_extract = 0.0
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http("not valid json at all"):
                result = a.extract("仓库A出现漏水现象非常严重", "已经记录派人立即处理问题")
        assert result == []

    def test_non_list_response_returns_empty(self):
        a = _make_adapter()
        a._last_extract = 0.0
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http('{"type": "anomaly"}'):
                result = a.extract("仓库有问题需要紧急处理", "好的立即安排人员处理")
        assert result == []

    def test_text_truncated_to_100_chars(self):
        a = _make_adapter()
        a._last_extract = 0.0
        long_text = "x" * 200
        import json
        facts = [{"type": "observation", "location": "A", "text": long_text}]
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with self._fake_http(json.dumps(facts)):
                result = a.extract("状态报告所有设备运行正常无异常", "已记录设备状态一切正常运行良好")
        assert len(result) == 1
        assert len(result[0]["text"]) <= 100

    def test_network_error_returns_empty(self):
        a = _make_adapter()
        a._last_extract = 0.0
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            with patch("httpx.post", side_effect=Exception("connection refused")):
                result = a.extract("仓库A发现设备故障需要立即处理", "已记录故障情况立即联系维修")
        assert result == []
