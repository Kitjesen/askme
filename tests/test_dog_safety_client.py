"""Tests for DogSafetyClient — fire-and-forget E-STOP notification + DDS integration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from askme.robot.safety_client import DogSafetyClient, _ESTOP_PATH


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


class TestPulseEstopIntegration:
    """Tests for Pulse bus integration in is_estop_active()."""

    def _make_pulse(self, *, estop_data=None) -> MagicMock:
        """Create a mock Pulse bus with optional estop data."""
        client = MagicMock()
        if estop_data is not None:
            client.get_latest.return_value = {
                "active": estop_data.get("active", False),
                "_ts": 1.0,
            }
        else:
            client.get_latest.return_value = None
        client.is_estop_active.return_value = bool(
            estop_data and estop_data.get("active", False)
        )
        return client

    def test_pulse_estop_active(self):
        """When Pulse has estop data with active=True, is_estop_active returns True."""
        pulse = self._make_pulse(estop_data={"active": True})
        client = DogSafetyClient(config={"base_url": ""}, pulse=pulse)
        assert client.is_estop_active() is True

    def test_pulse_estop_inactive(self):
        """When Pulse has estop data with active=False, is_estop_active returns False."""
        pulse = self._make_pulse(estop_data={"active": False})
        client = DogSafetyClient(config={"base_url": ""}, pulse=pulse)
        assert client.is_estop_active() is False

    def test_pulse_no_data_falls_back_to_http(self):
        """When Pulse has no estop data, falls back to HTTP cache."""
        pulse = self._make_pulse(estop_data=None)
        client = DogSafetyClient(
            config={"base_url": "http://localhost:5070"},
            pulse=pulse,
        )
        # No HTTP cache either → False
        assert client.is_estop_active() is False
        # Pulse is_estop_active should NOT have been called (no data)
        pulse.is_estop_active.assert_not_called()

    def test_no_pulse_uses_http_cache(self):
        """When pulse is None, uses HTTP cache only."""
        client = DogSafetyClient(
            config={"base_url": "http://localhost:5070"},
            pulse=None,
        )
        # No cache → False
        assert client.is_estop_active() is False

    def test_pulse_default_none(self):
        """Default pulse is None."""
        client = DogSafetyClient(config={})
        assert client._pulse is None

    def test_pulse_takes_priority_over_http_cache(self):
        """Pulse data takes priority even when HTTP cache has data."""
        pulse = self._make_pulse(estop_data={"active": False})
        client = DogSafetyClient(
            config={"base_url": "http://localhost:5070"},
            pulse=pulse,
        )
        # Manually set HTTP cache to active
        import time as _time
        client._cached_estop = {"enabled": True}
        client._cache_ts = _time.monotonic()

        # Pulse says inactive → should return False (Pulse wins)
        assert client.is_estop_active() is False
        pulse.is_estop_active.assert_called_once()
