"""Tests for LedStateKind, NullLedController, HttpLedController."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from askme.robot.led_controller import (
    HttpLedController,
    LedController,
    LedStateKind,
    NullLedController,
)


# ── LedStateKind ──────────────────────────────────────────────────────────────

class TestLedStateKind:
    def test_all_expected_states(self):
        names = {s.name for s in LedStateKind}
        assert "IDLE" in names
        assert "LISTENING" in names
        assert "PROCESSING" in names
        assert "SPEAKING" in names
        assert "MUTED" in names
        assert "AGENT_TASK" in names
        assert "ESTOP" in names

    def test_values_are_strings(self):
        for s in LedStateKind:
            assert isinstance(s.value, str)

    def test_idle_value(self):
        assert LedStateKind.IDLE.value == "idle"

    def test_estop_value(self):
        assert LedStateKind.ESTOP.value == "estop"


# ── NullLedController ─────────────────────────────────────────────────────────

class TestNullLedController:
    def test_implements_protocol(self):
        ctrl = NullLedController()
        assert hasattr(ctrl, "set_state")

    def test_set_state_does_not_raise(self):
        ctrl = NullLedController()
        ctrl.set_state(LedStateKind.IDLE)  # should not raise
        ctrl.set_state(LedStateKind.ESTOP)

    def test_set_state_all_states(self):
        ctrl = NullLedController()
        for state in LedStateKind:
            ctrl.set_state(state)  # no exception


# ── HttpLedController ─────────────────────────────────────────────────────────

class TestHttpLedController:
    def test_base_url_trailing_slash_stripped(self):
        ctrl = HttpLedController("http://localhost:5080/")
        assert ctrl._base_url == "http://localhost:5080"

    def test_default_timeout(self):
        ctrl = HttpLedController("http://localhost:5080")
        assert ctrl._timeout == 1.0

    def test_custom_timeout(self):
        ctrl = HttpLedController("http://localhost:5080", timeout=2.5)
        assert ctrl._timeout == 2.5

    def test_set_state_posts_in_background_thread(self):
        """set_state fires a thread and returns immediately; no direct HTTP call in test."""
        ctrl = HttpLedController("http://localhost:9999")
        called = threading.Event()

        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: called.set()):
            ctrl.set_state(LedStateKind.LISTENING)
            called.wait(timeout=1.0)

        assert called.is_set()

    def test_set_state_sends_correct_state_value(self):
        ctrl = HttpLedController("http://localhost:9999")
        requests_seen = []

        def fake_urlopen(req, timeout=None):
            import json
            body = json.loads(req.data.decode())
            requests_seen.append(body)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            ctrl.set_state(LedStateKind.MUTED)
            # Wait for background thread
            time.sleep(0.1)

        assert len(requests_seen) == 1
        assert requests_seen[0]["state"] == "muted"

    def test_set_state_does_not_raise_on_connection_error(self):
        ctrl = HttpLedController("http://localhost:9999")

        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            ctrl.set_state(LedStateKind.PROCESSING)  # must not raise
            time.sleep(0.1)  # let thread complete

    def test_set_state_sends_correct_url(self):
        ctrl = HttpLedController("http://robot:5080")
        urls_seen = []

        def fake_urlopen(req, timeout=None):
            urls_seen.append(req.full_url)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            ctrl.set_state(LedStateKind.IDLE)
            time.sleep(0.1)

        assert len(urls_seen) == 1
        assert "/api/v1/control/led" in urls_seen[0]

    def test_set_state_content_type_header(self):
        ctrl = HttpLedController("http://robot:5080")
        headers_seen = []

        def fake_urlopen(req, timeout=None):
            headers_seen.append(req.get_header("Content-type"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            ctrl.set_state(LedStateKind.SPEAKING)
            time.sleep(0.1)

        assert headers_seen[0] == "application/json"
