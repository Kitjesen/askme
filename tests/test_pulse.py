"""Tests for Pulse — 脉搏数据总线."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

from askme.robot.pulse import Pulse


# ── Construction ─────────────────────────────────────


def test_default_disabled_without_rclpy():
    with patch("askme.robot.pulse._RCLPY_AVAILABLE", False):
        bus = Pulse()
        assert not bus.available


def test_enabled_with_config():
    bus = Pulse({"enabled": True})
    # available depends on _RCLPY_AVAILABLE at import time
    assert isinstance(bus.available, bool)


def test_disabled_with_config():
    bus = Pulse({"enabled": False})
    assert not bus.available


# ── get_latest / cache ───────────────────────────────


def test_get_latest_none_when_empty():
    bus = Pulse({"enabled": False})
    assert bus.get_latest("/thunder/detections") is None


def test_get_latest_returns_copy():
    bus = Pulse({"enabled": False})
    bus._latest["/thunder/detections"] = {"frame_id": 1, "_ts": 1.0}
    result = bus.get_latest("/thunder/detections")
    assert result == {"frame_id": 1, "_ts": 1.0}
    result["frame_id"] = 999
    assert bus._latest["/thunder/detections"]["frame_id"] == 1


def test_get_latest_thread_safe():
    bus = Pulse({"enabled": False})
    errors = []

    def writer():
        for i in range(100):
            with bus._latest_lock:
                bus._latest["/t"] = {"v": i}

    def reader():
        for _ in range(100):
            r = bus.get_latest("/t")
            if r is not None and "v" not in r:
                errors.append("missing key")

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(errors) == 0


# ── ESTOP ────────────────────────────────────────────


def test_estop_false_when_no_data():
    bus = Pulse({"enabled": False})
    assert bus.is_estop_active() is False


def test_estop_true_when_active():
    bus = Pulse({"enabled": False})
    bus._latest["/thunder/estop"] = {"active": True, "_ts": 1.0}
    assert bus.is_estop_active() is True


def test_estop_false_when_inactive():
    bus = Pulse({"enabled": False})
    bus._latest["/thunder/estop"] = {"active": False, "_ts": 1.0}
    assert bus.is_estop_active() is False


# ── Detections ───────────────────────────────────────


def test_get_detections_none():
    bus = Pulse({"enabled": False})
    assert bus.get_detections() is None


def test_get_detections_returns_data():
    bus = Pulse({"enabled": False})
    bus._latest["/thunder/detections"] = {"detections": [{"label": "person"}], "_ts": 1.0}
    result = bus.get_detections()
    assert result["detections"][0]["label"] == "person"


# ── Joint States / IMU ───────────────────────────────


def test_get_joint_states_none():
    bus = Pulse({"enabled": False})
    assert bus.get_joint_states() is None


def test_get_imu_none():
    bus = Pulse({"enabled": False})
    assert bus.get_imu() is None


# ── Callbacks ────────────────────────────────────────


def test_on_registers_callback():
    bus = Pulse({"enabled": False})
    cb = MagicMock()
    bus.on("/thunder/detections", cb)
    assert cb in bus._callbacks["/thunder/detections"]


def test_multiple_callbacks_per_topic():
    bus = Pulse({"enabled": False})
    cb1 = MagicMock()
    cb2 = MagicMock()
    bus.on("/t", cb1)
    bus.on("/t", cb2)
    assert len(bus._callbacks["/t"]) == 2


def test_callbacks_isolated_by_topic():
    bus = Pulse({"enabled": False})
    cb1 = MagicMock()
    cb2 = MagicMock()
    bus.on("/a", cb1)
    bus.on("/b", cb2)
    assert "/a" in bus._callbacks
    assert "/b" in bus._callbacks
    assert cb1 not in bus._callbacks.get("/b", [])


# ── _on_message ──────────────────────────────────────


def test_on_message_updates_cache():
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/estop", lambda m: {"active": True}, None)
    assert bus._latest["/thunder/estop"]["active"] is True
    assert bus._msg_count == 1


def test_on_message_bad_parser_ignored():
    bus = Pulse({"enabled": False})
    bus._on_message("/t", lambda m: 1 / 0, None)  # ZeroDivisionError
    assert bus._msg_count == 0
    assert "/t" not in bus._latest


# ── Health ───────────────────────────────────────────


def test_health_disabled():
    bus = Pulse({"enabled": False})
    h = bus.health()
    assert h["status"] == "disabled"
    assert h["available"] is False


def test_health_not_started():
    bus = Pulse({"enabled": True})
    h = bus.health()
    # If rclpy not available, status is "disabled"; otherwise "disconnected"
    assert h["status"] in ("disabled", "disconnected")


# ── Lifecycle (no rclpy needed) ──────────────────────


async def test_start_disabled_is_noop():
    bus = Pulse({"enabled": False})
    await bus.start()
    assert not bus.connected


async def test_stop_without_start_is_safe():
    bus = Pulse({"enabled": False})
    await bus.stop()  # should not raise


async def test_start_stop_idempotent():
    bus = Pulse({"enabled": False})
    await bus.start()
    await bus.start()  # double start
    await bus.stop()
    await bus.stop()  # double stop
