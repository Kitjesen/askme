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
    with patch("askme.robot.pulse._CYCLONE_AVAILABLE", False):
        bus = Pulse()
        assert not bus.available


def test_enabled_with_config():
    bus = Pulse({"enabled": True})
    # available depends on _CYCLONE_AVAILABLE at import time
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


# ── Health with per-topic freshness ─────────────────


def test_health_topics_is_dict():
    """health()['topics'] is now a dict keyed by topic name."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/estop", lambda m: {"active": True}, None)
    h = bus.health()
    assert isinstance(h["topics"], dict)
    assert "/thunder/estop" in h["topics"]


def test_health_topic_freshness_fields():
    """Each topic entry has the expected freshness fields."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/estop", lambda m: {"active": False}, None)
    h = bus.health()
    info = h["topics"]["/thunder/estop"]
    assert "last_ts" in info
    assert "age_ms" in info
    assert "stale" in info
    assert "msg_count" in info
    assert "rate_hz" in info
    assert info["msg_count"] == 1
    assert info["last_ts"] > 0


def test_health_stale_detection():
    """A topic with an old timestamp should be marked stale."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/detections", lambda m: {"detections": []}, None)
    # Force the last_ts to be old (detections threshold is 2000ms)
    bus._topic_last_ts["/thunder/detections"] = time.time() - 10.0
    h = bus.health()
    info = h["topics"]["/thunder/detections"]
    assert info["stale"] is True
    assert info["age_ms"] > 2000


def test_health_not_stale_when_fresh():
    """A topic with a recent timestamp should not be stale."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/detections", lambda m: {"detections": []}, None)
    h = bus.health()
    info = h["topics"]["/thunder/detections"]
    assert info["stale"] is False
    assert info["age_ms"] < 2000


def test_health_estop_stale_threshold_is_long():
    """ESTOP has a 60s threshold -- should not be stale after a few seconds."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/estop", lambda m: {"active": False}, None)
    # Force 5 seconds old -- well within 60s threshold
    bus._topic_last_ts["/thunder/estop"] = time.time() - 5.0
    h = bus.health()
    info = h["topics"]["/thunder/estop"]
    assert info["stale"] is False


def test_health_rate_hz_calculation():
    """rate_hz should reflect messages in the sliding window."""
    bus = Pulse({"enabled": False})
    now = time.time()
    # Simulate 20 messages in 10 seconds
    for i in range(20):
        bus._record_topic_msg("/thunder/detections", now - 9.5 + i * 0.5)
    # Also put something in _latest so topic shows in health
    bus._latest["/thunder/detections"] = {"detections": [], "_ts": now}
    h = bus.health()
    info = h["topics"]["/thunder/detections"]
    assert info["rate_hz"] == 2.0  # 20 msgs / 10s
    assert info["msg_count"] == 20


def test_health_rate_hz_prunes_old_entries():
    """Messages older than the 10s window are pruned from rate calculation."""
    bus = Pulse({"enabled": False})
    now = time.time()
    # 5 old messages (15s ago) + 5 recent messages
    for i in range(5):
        bus._record_topic_msg("/thunder/imu", now - 15.0 + i * 0.1)
    for i in range(5):
        bus._record_topic_msg("/thunder/imu", now - 1.0 + i * 0.1)
    bus._latest["/thunder/imu"] = {"angular_velocity": {}, "_ts": now}
    h = bus.health()
    info = h["topics"]["/thunder/imu"]
    # Only the 5 recent messages should be in the window
    assert info["rate_hz"] == 0.5  # 5 msgs / 10s


def test_health_msg_count_per_topic():
    """Per-topic msg_count increments correctly."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/estop", lambda m: {"active": True}, None)
    bus._on_message("/thunder/estop", lambda m: {"active": False}, None)
    bus._on_message("/thunder/detections", lambda m: {"detections": []}, None)
    h = bus.health()
    assert h["topics"]["/thunder/estop"]["msg_count"] == 2
    assert h["topics"]["/thunder/detections"]["msg_count"] == 1
    assert h["msg_count"] == 3


def test_health_last_message_type_for_known_topics():
    """Well-known topics include last_message_type in health info."""
    bus = Pulse({"enabled": False})
    bus._on_message("/thunder/estop", lambda m: {"active": False}, None)
    h = bus.health()
    info = h["topics"]["/thunder/estop"]
    assert info["last_message_type"] == "EstopState"


def test_health_no_message_type_for_unknown_topic():
    """Unknown topics do not include last_message_type."""
    bus = Pulse({"enabled": False})
    bus._on_message("/custom/topic", lambda m: {"v": 1}, None)
    h = bus.health()
    info = h["topics"]["/custom/topic"]
    assert "last_message_type" not in info


# ── Publish registry ────────────────────────────────


def test_publish_without_dp_is_noop_for_any_topic():
    """Publishing without DomainParticipant is a no-op."""
    bus = Pulse({"enabled": False})
    bus._dp = None
    bus.publish("/any/topic", {"data": 1})
    assert "/any/topic" not in bus._writers


def test_publish_without_dp_is_noop():
    """Publish without DomainParticipant does nothing (no crash)."""
    bus = Pulse({"enabled": False})
    bus._dp = None
    bus.publish("/thunder/world_state", {"test": True})
    # No crash, no writer created
