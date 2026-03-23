"""Tests for askme.robot.dds_bridge_client."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from askme.robot.dds_bridge_client import DdsBridgeClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msg(topic: str, data: dict, ts: float = 1234567890.0) -> bytes:
    """Encode a single bridge message as newline-delimited JSON bytes."""
    return json.dumps({"topic": topic, "data": data, "ts": ts}).encode() + b"\n"


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


class TestHandleMessage:
    """Unit tests for _handle_message (message parsing + caching)."""

    async def test_valid_message_cached(self):
        client = DdsBridgeClient({"enabled": True})
        raw = json.dumps(
            {"topic": "/thunder/estop", "data": {"active": True}, "ts": 100.0}
        ).encode()
        await client._handle_message(raw)

        result = client.get_latest("/thunder/estop")
        assert result is not None
        assert result["topic"] == "/thunder/estop"
        assert result["data"]["active"] is True
        assert result["ts"] == 100.0
        assert client.message_count == 1

    async def test_malformed_json_ignored(self):
        client = DdsBridgeClient({"enabled": True})
        await client._handle_message(b"not-json{{{")
        assert client.get_latest("anything") is None
        assert client.message_count == 0

    async def test_empty_bytes_ignored(self):
        client = DdsBridgeClient({"enabled": True})
        await client._handle_message(b"")
        assert client.message_count == 0

    async def test_latest_overwrites_previous(self):
        client = DdsBridgeClient({"enabled": True})
        msg1 = json.dumps(
            {"topic": "/t", "data": {"v": 1}, "ts": 1.0}
        ).encode()
        msg2 = json.dumps(
            {"topic": "/t", "data": {"v": 2}, "ts": 2.0}
        ).encode()
        await client._handle_message(msg1)
        await client._handle_message(msg2)

        result = client.get_latest("/t")
        assert result is not None
        assert result["data"]["v"] == 2
        assert client.message_count == 2


# ---------------------------------------------------------------------------
# get_latest
# ---------------------------------------------------------------------------


class TestGetLatest:
    """Tests for get_latest()."""

    def test_returns_none_when_no_data(self):
        client = DdsBridgeClient()
        assert client.get_latest("/thunder/estop") is None

    def test_returns_none_for_unknown_topic(self):
        client = DdsBridgeClient()
        assert client.get_latest("/nonexistent") is None

    async def test_returns_copy_not_reference(self):
        client = DdsBridgeClient({"enabled": True})
        raw = json.dumps(
            {"topic": "/t", "data": {"x": 1}, "ts": 1.0}
        ).encode()
        await client._handle_message(raw)

        r1 = client.get_latest("/t")
        r2 = client.get_latest("/t")
        assert r1 is not r2  # distinct dicts


# ---------------------------------------------------------------------------
# is_estop_active
# ---------------------------------------------------------------------------


class TestIsEstopActive:
    """Tests for the is_estop_active() convenience method."""

    def test_defaults_false_when_no_data(self):
        client = DdsBridgeClient()
        assert client.is_estop_active() is False

    async def test_true_when_active(self):
        client = DdsBridgeClient({"enabled": True})
        raw = json.dumps(
            {"topic": "/thunder/estop", "data": {"active": True}, "ts": 1.0}
        ).encode()
        await client._handle_message(raw)
        assert client.is_estop_active() is True

    async def test_false_when_inactive(self):
        client = DdsBridgeClient({"enabled": True})
        raw = json.dumps(
            {"topic": "/thunder/estop", "data": {"active": False}, "ts": 1.0}
        ).encode()
        await client._handle_message(raw)
        assert client.is_estop_active() is False

    async def test_false_when_data_missing_active_key(self):
        client = DdsBridgeClient({"enabled": True})
        raw = json.dumps(
            {"topic": "/thunder/estop", "data": {}, "ts": 1.0}
        ).encode()
        await client._handle_message(raw)
        assert client.is_estop_active() is False


# ---------------------------------------------------------------------------
# get_detections
# ---------------------------------------------------------------------------


class TestGetDetections:
    """Tests for the get_detections() convenience method."""

    def test_returns_none_when_no_data(self):
        client = DdsBridgeClient()
        assert client.get_detections() is None

    async def test_returns_data_dict(self):
        client = DdsBridgeClient({"enabled": True})
        det_data = {"detections": [{"label": "person", "score": 0.9}], "frame_id": 42}
        raw = json.dumps(
            {"topic": "/thunder/detections", "data": det_data, "ts": 1.0}
        ).encode()
        await client._handle_message(raw)

        result = client.get_detections()
        assert result is not None
        assert result["frame_id"] == 42
        assert len(result["detections"]) == 1


# ---------------------------------------------------------------------------
# Callback registration and dispatch
# ---------------------------------------------------------------------------


class TestCallbacks:
    """Tests for on() callback registration and dispatch."""

    async def test_sync_callback_called(self):
        client = DdsBridgeClient({"enabled": True})
        received = []

        def cb(topic: str, data: dict, ts: float) -> None:
            received.append((topic, data, ts))

        client.on("/thunder/estop", cb)
        raw = json.dumps(
            {"topic": "/thunder/estop", "data": {"active": True}, "ts": 5.0}
        ).encode()
        await client._handle_message(raw)

        assert len(received) == 1
        assert received[0] == ("/thunder/estop", {"active": True}, 5.0)

    async def test_async_callback_called(self):
        client = DdsBridgeClient({"enabled": True})
        received = []

        async def cb(topic: str, data: dict, ts: float) -> None:
            received.append((topic, data, ts))

        client.on("/t", cb)
        raw = json.dumps({"topic": "/t", "data": {"v": 1}, "ts": 3.0}).encode()
        await client._handle_message(raw)

        assert len(received) == 1
        assert received[0][1] == {"v": 1}

    async def test_multiple_callbacks_same_topic(self):
        client = DdsBridgeClient({"enabled": True})
        calls_a = []
        calls_b = []

        client.on("/t", lambda t, d, ts: calls_a.append(d))
        client.on("/t", lambda t, d, ts: calls_b.append(d))

        raw = json.dumps({"topic": "/t", "data": {"n": 1}, "ts": 1.0}).encode()
        await client._handle_message(raw)

        assert len(calls_a) == 1
        assert len(calls_b) == 1

    async def test_callback_not_called_for_other_topic(self):
        client = DdsBridgeClient({"enabled": True})
        received = []
        client.on("/thunder/estop", lambda t, d, ts: received.append(d))

        raw = json.dumps(
            {"topic": "/thunder/detections", "data": {}, "ts": 1.0}
        ).encode()
        await client._handle_message(raw)

        assert len(received) == 0

    async def test_callback_exception_does_not_crash(self):
        client = DdsBridgeClient({"enabled": True})

        def bad_cb(topic, data, ts):
            raise ValueError("boom")

        good_results = []

        def good_cb(topic, data, ts):
            good_results.append(data)

        client.on("/t", bad_cb)
        client.on("/t", good_cb)

        raw = json.dumps({"topic": "/t", "data": {"ok": 1}, "ts": 1.0}).encode()
        await client._handle_message(raw)

        # Good callback still ran despite bad_cb raising
        assert len(good_results) == 1


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Tests for start() and stop()."""

    async def test_start_disabled_is_noop(self):
        client = DdsBridgeClient({"enabled": False})
        await client.start()
        assert client._task is None

    async def test_start_enabled_creates_task(self):
        client = DdsBridgeClient({"enabled": True, "socket_path": "/tmp/nonexistent_test.sock"})

        async def mock_open(path):
            raise FileNotFoundError("test")

        with patch("askme.robot.dds_bridge_client.asyncio.open_unix_connection", mock_open, create=True):
            await client.start()
            assert client._task is not None
            await client.stop()
            assert client._task is None

    async def test_stop_idempotent(self):
        client = DdsBridgeClient({"enabled": False})
        # stop() on a client that was never started should not raise
        await client.stop()
        await client.stop()

    async def test_start_idempotent(self):
        client = DdsBridgeClient({"enabled": True, "socket_path": "/tmp/nonexistent_test.sock"})

        async def mock_open(path):
            raise FileNotFoundError("test")

        with patch("askme.robot.dds_bridge_client.asyncio.open_unix_connection", mock_open, create=True):
            await client.start()
            task1 = client._task
            await client.start()  # second call is no-op
            assert client._task is task1
            await client.stop()


# ---------------------------------------------------------------------------
# Reconnect behavior (mocked socket)
# ---------------------------------------------------------------------------


class TestReconnect:
    """Tests for auto-reconnect logic."""

    async def test_reconnects_on_connection_refused(self):
        """Client retries after ConnectionRefusedError."""
        client = DdsBridgeClient({
            "enabled": True,
            "socket_path": "/tmp/nonexistent_dds_test.sock",
            "reconnect_interval": 0.05,
        })

        attempt_count = 0

        async def mock_open(path):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionRefusedError("not up yet")
            # On 3rd attempt, simulate connection then immediate close
            reader = AsyncMock()
            reader.read = AsyncMock(return_value=b"")
            writer = MagicMock()
            writer.close = MagicMock()
            return reader, writer

        with patch("askme.robot.dds_bridge_client.asyncio.open_unix_connection", mock_open, create=True):
            await client.start()
            # Give it time to attempt reconnects
            await asyncio.sleep(0.3)
            await client.stop()

        assert attempt_count >= 3

    async def test_reconnects_on_file_not_found(self):
        """Client retries after FileNotFoundError."""
        client = DdsBridgeClient({
            "enabled": True,
            "socket_path": "/tmp/nonexistent_dds_test.sock",
            "reconnect_interval": 0.05,
        })

        attempt_count = 0

        async def mock_open(path):
            nonlocal attempt_count
            attempt_count += 1
            raise FileNotFoundError("no socket file")

        with patch("askme.robot.dds_bridge_client.asyncio.open_unix_connection", mock_open, create=True):
            await client.start()
            await asyncio.sleep(0.2)
            await client.stop()

        assert attempt_count >= 2

    async def test_reads_messages_after_connect(self):
        """Client processes messages from the socket stream."""
        client = DdsBridgeClient({
            "enabled": True,
            "socket_path": "/tmp/test.sock",
            "reconnect_interval": 0.05,
        })

        msg1 = _make_msg("/thunder/estop", {"active": True}, ts=1.0)
        msg2 = _make_msg("/thunder/detections", {"detections": []}, ts=2.0)
        stream_data = msg1 + msg2

        read_call_count = 0

        async def mock_read(n):
            nonlocal read_call_count
            read_call_count += 1
            if read_call_count == 1:
                return stream_data
            return b""  # signal close

        reader = AsyncMock()
        reader.read = mock_read
        writer = MagicMock()
        writer.close = MagicMock()

        async def mock_open(path):
            return reader, writer

        with patch("askme.robot.dds_bridge_client.asyncio.open_unix_connection", mock_open, create=True):
            await client.start()
            await asyncio.sleep(0.15)
            await client.stop()

        assert client.is_estop_active() is True
        assert client.get_detections() is not None
        assert client.message_count == 2


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for configuration handling."""

    def test_defaults(self):
        client = DdsBridgeClient()
        assert client._socket_path == "/tmp/askme_dds_bridge.sock"
        assert client._reconnect_interval == 2.0
        assert client._enabled is False

    def test_custom_config(self):
        client = DdsBridgeClient({
            "enabled": True,
            "socket_path": "/var/run/custom.sock",
            "reconnect_interval": 5.0,
        })
        assert client._socket_path == "/var/run/custom.sock"
        assert client._reconnect_interval == 5.0
        assert client._enabled is True

    def test_none_config_uses_defaults(self):
        client = DdsBridgeClient(None)
        assert client._socket_path == "/tmp/askme_dds_bridge.sock"
        assert client._enabled is False
