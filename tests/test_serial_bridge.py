"""Tests for SerialBridge — simulate-mode serial communication with robot arm."""

from __future__ import annotations

import numpy as np
import pytest

from askme.robot.serial_bridge import SerialBridge


def _make_sim() -> SerialBridge:
    return SerialBridge(simulate=True)


class TestInit:
    def test_not_connected_on_init(self):
        sb = SerialBridge()
        assert sb.is_connected is False

    def test_defaults(self):
        sb = SerialBridge()
        assert sb._port == "COM3"
        assert sb._baudrate == 115200
        assert sb._simulate is True  # default is simulate=True

    def test_custom_port_and_baudrate(self):
        sb = SerialBridge(port="/dev/ttyUSB0", baudrate=9600, simulate=True)
        assert sb._port == "/dev/ttyUSB0"
        assert sb._baudrate == 9600


class TestConnectSimulate:
    def test_connect_returns_true_in_simulate(self):
        sb = _make_sim()
        assert sb.connect() is True

    def test_is_connected_after_connect(self):
        sb = _make_sim()
        sb.connect()
        assert sb.is_connected is True

    def test_disconnect_clears_connected(self):
        sb = _make_sim()
        sb.connect()
        sb.disconnect()
        assert sb.is_connected is False

    def test_disconnect_noop_when_not_connected(self):
        sb = _make_sim()
        sb.disconnect()  # should not raise
        assert sb.is_connected is False


class TestSendActionSimulate:
    def test_send_action_returns_true(self):
        sb = _make_sim()
        sb.connect()
        action = np.ones(16, dtype=np.float32)
        assert sb.send_action(action) is True

    def test_send_action_updates_sim_state(self):
        sb = _make_sim()
        sb.connect()
        action = np.arange(16, dtype=np.float32)
        sb.send_action(action)
        np.testing.assert_array_equal(sb._sim_state, action)

    def test_send_action_casts_to_float32(self):
        sb = _make_sim()
        sb.connect()
        action = np.ones(16, dtype=np.float64)
        assert sb.send_action(action) is True
        assert sb._sim_state.dtype == np.float32

    def test_send_action_fails_when_not_connected(self):
        sb = _make_sim()
        # Don't call connect()
        action = np.ones(16, dtype=np.float32)
        assert sb.send_action(action) is False

    def test_send_action_rejects_wrong_dim(self):
        sb = _make_sim()
        sb.connect()
        action = np.ones(8, dtype=np.float32)
        assert sb.send_action(action) is False

    def test_send_action_rejects_empty(self):
        sb = _make_sim()
        sb.connect()
        assert sb.send_action(np.zeros(0)) is False

    def test_send_action_accepts_2d_if_flat(self):
        """A (1,16) shaped array can be squeezed — but shape[0] would be 1 != 16.
        The bridge rejects it explicitly.
        """
        sb = _make_sim()
        sb.connect()
        action = np.ones((1, 16), dtype=np.float32)
        # shape[0] == 1 != 16
        assert sb.send_action(action) is False


class TestGetStateSimulate:
    def test_get_state_returns_zeros_initially(self):
        sb = _make_sim()
        sb.connect()
        state = sb.get_state()
        assert state is not None
        assert state.shape == (16,)
        np.testing.assert_array_equal(state, np.zeros(16))

    def test_get_state_reflects_last_action(self):
        sb = _make_sim()
        sb.connect()
        action = np.arange(16, dtype=np.float32) * 0.1
        sb.send_action(action)
        state = sb.get_state()
        np.testing.assert_array_almost_equal(state, action)

    def test_get_state_returns_copy_not_reference(self):
        sb = _make_sim()
        sb.connect()
        state1 = sb.get_state()
        state1[0] = 999.0
        state2 = sb.get_state()
        assert state2[0] != 999.0

    def test_get_state_returns_none_when_not_connected(self):
        sb = _make_sim()
        assert sb.get_state() is None


class TestRealSerialFallback:
    def test_connect_fails_gracefully_without_pyserial(self):
        """connect() returns False when pyserial is unavailable (not simulate)."""
        import sys
        from unittest.mock import patch

        sb = SerialBridge(port="/dev/fake", simulate=False)
        with patch.dict("sys.modules", {"serial": None}):
            result = sb.connect()
        assert result is False
        assert sb.is_connected is False
