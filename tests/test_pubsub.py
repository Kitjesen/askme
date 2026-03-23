"""Tests for PubSubBase ABC and MockPulse implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

from askme.robot.pubsub import PubSubBase
from askme.robot.mock_pulse import MockPulse


# ── PubSubBase ABC ──────────────────────────────────


def test_pubsub_cannot_instantiate():
    """ABC should not be directly instantiable."""
    try:
        PubSubBase()  # type: ignore[abstract]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_pubsub_convenience_methods_use_get_latest():
    """Convenience methods delegate to get_latest correctly."""
    mock = MockPulse()
    assert mock.is_estop_active() is False
    assert mock.get_detections() is None
    assert mock.get_joint_states() is None
    assert mock.get_imu() is None


def test_pubsub_is_estop_active_true():
    mock = MockPulse()
    mock.publish("/thunder/estop", {"active": True})
    assert mock.is_estop_active() is True


def test_pubsub_is_estop_active_false():
    mock = MockPulse()
    mock.publish("/thunder/estop", {"active": False})
    assert mock.is_estop_active() is False


def test_pubsub_get_detections():
    mock = MockPulse()
    mock.publish("/thunder/detections", {"detections": [{"label": "person"}]})
    result = mock.get_detections()
    assert result is not None
    assert result["detections"][0]["label"] == "person"


def test_pubsub_get_joint_states():
    mock = MockPulse()
    mock.publish("/thunder/joint_states", {"name": ["j1"], "position": [0.5]})
    result = mock.get_joint_states()
    assert result is not None
    assert result["name"] == ["j1"]


def test_pubsub_get_imu():
    mock = MockPulse()
    mock.publish("/thunder/imu", {"angular_velocity": {"x": 0.1}})
    result = mock.get_imu()
    assert result is not None
    assert result["angular_velocity"]["x"] == 0.1


# ── MockPulse ───────────────────────────────────────


def test_mock_pulse_publish_and_get_latest():
    mock = MockPulse()
    mock.publish("/test/topic", {"key": "value"})
    result = mock.get_latest("/test/topic")
    assert result == {"key": "value"}


def test_mock_pulse_get_latest_returns_copy():
    mock = MockPulse()
    mock.publish("/t", {"a": 1})
    result = mock.get_latest("/t")
    result["a"] = 999
    assert mock.get_latest("/t")["a"] == 1


def test_mock_pulse_get_latest_none_when_empty():
    mock = MockPulse()
    assert mock.get_latest("/nonexistent") is None


def test_mock_pulse_on_callback_fires():
    mock = MockPulse()
    cb = MagicMock()
    mock.on("/t", cb)
    mock.publish("/t", {"x": 1})
    cb.assert_called_once_with("/t", {"x": 1})


def test_mock_pulse_multiple_callbacks():
    mock = MockPulse()
    cb1 = MagicMock()
    cb2 = MagicMock()
    mock.on("/t", cb1)
    mock.on("/t", cb2)
    mock.publish("/t", {"x": 1})
    cb1.assert_called_once()
    cb2.assert_called_once()


def test_mock_pulse_callbacks_isolated_by_topic():
    mock = MockPulse()
    cb_a = MagicMock()
    cb_b = MagicMock()
    mock.on("/a", cb_a)
    mock.on("/b", cb_b)
    mock.publish("/a", {"v": 1})
    cb_a.assert_called_once()
    cb_b.assert_not_called()


def test_mock_pulse_publish_overwrites_latest():
    mock = MockPulse()
    mock.publish("/t", {"v": 1})
    mock.publish("/t", {"v": 2})
    assert mock.get_latest("/t")["v"] == 2


# ── MockPulse lifecycle ─────────────────────────────


async def test_mock_pulse_start_stop():
    mock = MockPulse()
    assert not mock.connected
    await mock.start()
    assert mock.connected
    await mock.stop()
    assert not mock.connected


def test_mock_pulse_health_disconnected():
    mock = MockPulse()
    h = mock.health()
    assert h["status"] == "disconnected"
    assert h["connected"] is False


async def test_mock_pulse_health_connected():
    mock = MockPulse()
    await mock.start()
    h = mock.health()
    assert h["status"] == "ok"
    assert h["connected"] is True


def test_mock_pulse_health_shows_topics():
    mock = MockPulse()
    mock.publish("/thunder/estop", {"active": False})
    mock.publish("/thunder/imu", {"x": 0})
    h = mock.health()
    assert "/thunder/estop" in h["topics"]
    assert "/thunder/imu" in h["topics"]


# ── MockPulse is_estop_active with mock data ────────


def test_mock_pulse_estop_workflow():
    """Full workflow: no data -> publish inactive -> publish active."""
    mock = MockPulse()
    assert mock.is_estop_active() is False

    mock.publish("/thunder/estop", {"active": False})
    assert mock.is_estop_active() is False

    mock.publish("/thunder/estop", {"active": True})
    assert mock.is_estop_active() is True

    mock.publish("/thunder/estop", {"active": False})
    assert mock.is_estop_active() is False


def test_mock_pulse_isinstance_pubsub():
    """MockPulse is a proper PubSubBase implementation."""
    mock = MockPulse()
    assert isinstance(mock, PubSubBase)
