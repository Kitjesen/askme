"""Tests for NavStatusTool — navigation status query tool."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from askme.tools.builtin_tools import NavStatusTool


class TestNavStatusTool:
    def test_not_configured(self, monkeypatch):
        """NAV_GATEWAY_URL が未設定のとき '未配置' を含む文字列を返す。"""
        monkeypatch.delenv("NAV_GATEWAY_URL", raising=False)
        tool = NavStatusTool()
        result = tool.execute()
        assert "未配置" in result

    def test_configured_success(self, monkeypatch):
        """urlopen が成功したとき JSON 文字列を返す。"""
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:9000")

        nav_data = {"status": "idle", "position": {"x": 1.0, "y": 2.0}}
        encoded = json.dumps(nav_data).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = encoded
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tool = NavStatusTool()
            result = tool.execute()

        # Result should be parseable JSON containing our data
        parsed = json.loads(result)
        assert parsed["status"] == "idle"
        assert parsed["position"]["x"] == 1.0

    def test_configured_failure(self, monkeypatch):
        """urlopen が例外を投げたとき '查询失败' を含む文字列を返す。"""
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:9000")

        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            tool = NavStatusTool()
            result = tool.execute()

        assert "查询失败" in result
