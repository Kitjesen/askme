"""Tests for DogSafetyClient — fire-and-forget E-STOP notification."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from askme.dog_safety_client import DogSafetyClient, _ESTOP_PATH


class TestDogSafetyClientConfiguration:
    def test_not_configured(self):
        """base_url が空のとき is_configured() は False を返す。"""
        client = DogSafetyClient(config={"base_url": ""})
        assert client.is_configured() is False

    def test_not_configured_notify_no_crash(self):
        """未設定時に notify_estop() を呼んでも例外が起きない。"""
        client = DogSafetyClient(config={"base_url": ""})
        # Should simply return without raising
        client.notify_estop()

    def test_configured(self):
        """base_url が非空のとき is_configured() は True を返す。"""
        client = DogSafetyClient(config={"base_url": "http://localhost:5070"})
        assert client.is_configured() is True


class TestDogSafetyClientNotifyEstop:
    def test_notify_estop_sends_request(self):
        """notify_estop() が正しい URL と body で POST を送る。"""
        client = DogSafetyClient(
            config={"base_url": "http://localhost:5070", "bearer_token": ""}
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.notify_estop()
            # Give the daemon thread time to finish
            time.sleep(0.2)

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args

            # Verify URL contains the estop path
            called_url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs.get("url", "")
            # requests.post positional first arg is the URL
            assert _ESTOP_PATH in called_url

            # Verify body
            sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert sent_json == {"enabled": True}

    def test_notify_estop_timeout_no_crash(self):
        """POST がタイムアウトしても notify_estop() は例外を送出しない。"""
        client = DogSafetyClient(config={"base_url": "http://localhost:5070"})

        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            client.notify_estop()
            time.sleep(0.2)  # let daemon thread run
        # No assertion needed — test passes if no exception propagates

    def test_notify_estop_http_error_no_crash(self):
        """POST が RequestException を投げても notify_estop() はクラッシュしない。"""
        client = DogSafetyClient(config={"base_url": "http://localhost:5070"})

        with patch("requests.post", side_effect=requests.RequestException("conn refused")):
            client.notify_estop()
            time.sleep(0.2)

    def test_bearer_token_in_header(self):
        """bearer_token が設定されているとき Authorization ヘッダーに含まれる。"""
        client = DogSafetyClient(
            config={
                "base_url": "http://localhost:5070",
                "bearer_token": "my-secret-token",
            }
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.notify_estop()
            time.sleep(0.2)

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
            assert "Authorization" in headers
            assert "Bearer my-secret-token" == headers["Authorization"]
