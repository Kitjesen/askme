"""Tests for PunctuationRestorer."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from askme.voice.punctuation import PunctuationRestorer


# ── Init ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_disabled_when_model_not_found(self, tmp_path):
        cfg = {"model_path": str(tmp_path / "nonexistent_model.onnx")}
        r = PunctuationRestorer(cfg)
        assert r.available is False

    def test_disabled_when_sherpa_not_installed(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(b"fake")
        cfg = {"model_path": str(model)}
        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            r = PunctuationRestorer(cfg)
        assert r.available is False

    def test_disabled_when_sherpa_raises(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(b"fake")
        cfg = {"model_path": str(model)}
        mock_sherpa = MagicMock()
        mock_sherpa.OfflinePunctuation.side_effect = RuntimeError("init failed")
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            r = PunctuationRestorer(cfg)
        assert r.available is False


# ── restore ───────────────────────────────────────────────────────────────────

class TestRestore:
    def _make_disabled(self):
        r = PunctuationRestorer.__new__(PunctuationRestorer)
        r._punct = None
        return r

    def _make_enabled(self, restored="你好，世界。"):
        r = PunctuationRestorer.__new__(PunctuationRestorer)
        mock_punct = MagicMock()
        mock_punct.add_punctuation.return_value = restored
        r._punct = mock_punct
        return r

    def test_returns_original_when_disabled(self):
        r = self._make_disabled()
        assert r.restore("你好世界") == "你好世界"

    def test_returns_original_when_empty_text(self):
        r = self._make_enabled()
        assert r.restore("") == ""

    def test_calls_add_punctuation(self):
        r = self._make_enabled("结果文本。")
        result = r.restore("原始文本")
        assert result == "结果文本。"
        r._punct.add_punctuation.assert_called_once_with("原始文本")

    def test_strips_whitespace_before_processing(self):
        r = self._make_enabled("text")
        r.restore("  text  ")
        r._punct.add_punctuation.assert_called_once_with("text")

    def test_returns_original_on_exception(self):
        r = self._make_enabled()
        r._punct.add_punctuation.side_effect = RuntimeError("inference error")
        result = r.restore("test text")
        assert result == "test text"

    def test_whitespace_only_returns_unchanged(self):
        r = self._make_enabled()
        result = r.restore("   ")
        # Empty after strip — should return original
        assert result == "   "
