"""Tests for SafetyChecker — joint limits, velocity limits, e-stop detection."""

from __future__ import annotations

import numpy as np

from askme.robot.safety import SafetyChecker


def _make_checker(**cfg_overrides) -> SafetyChecker:
    return SafetyChecker(config=None if not cfg_overrides else cfg_overrides)


class TestInit:
    def test_not_estopped_on_init(self):
        s = _make_checker()
        assert s.is_estopped is False

    def test_no_last_action_on_init(self):
        s = _make_checker()
        assert s._last_action is None


class TestCheckActionJointLimits:
    def test_safe_action_not_clipped(self):
        s = _make_checker()
        action = np.zeros(16, dtype=np.float32)
        was_clipped, result = s.check_action(action)
        assert was_clipped is False
        np.testing.assert_array_almost_equal(result, action)

    def test_action_above_max_clipped(self):
        s = _make_checker()
        action = np.full(16, 10.0, dtype=np.float32)  # 10 rad >> 3.14 max
        was_clipped, result = s.check_action(action)
        assert was_clipped is True
        assert np.all(result[:6] <= 3.14 + 1e-5)

    def test_action_below_min_clipped(self):
        s = _make_checker()
        action = np.full(16, -10.0, dtype=np.float32)
        was_clipped, result = s.check_action(action)
        assert was_clipped is True
        assert np.all(result[:6] >= -3.14 - 1e-5)

    def test_finger_joints_clipped_to_nonnegative(self):
        s = _make_checker()
        action = np.zeros(16, dtype=np.float32)
        action[6:10] = -1.0  # fingers below 0
        was_clipped, result = s.check_action(action)
        assert was_clipped is True
        assert np.all(result[6:10] >= 0.0)

    def test_result_is_float32(self):
        s = _make_checker()
        action = np.zeros(16, dtype=np.float64)
        _, result = s.check_action(action)
        # result inherits dtype from clips
        assert result.dtype in (np.float32, np.float64)


class TestCheckActionVelocityLimits:
    def test_small_delta_not_clipped(self):
        s = _make_checker()
        s.check_action(np.zeros(16, dtype=np.float32))  # set last_action
        action = np.full(16, 0.1, dtype=np.float32)  # small delta
        was_clipped, _ = s.check_action(action)
        assert was_clipped is False

    def test_large_delta_clipped_to_max_delta(self):
        s = _make_checker()
        s.check_action(np.zeros(16, dtype=np.float32))  # set last_action
        action = np.full(16, 3.0, dtype=np.float32)  # huge jump
        _, result = s.check_action(action)
        # First 6 joints: max_delta = 0.5
        assert np.all(result[:6] <= 0.5 + 1e-5)

    def test_last_action_updated_after_check(self):
        s = _make_checker()
        action = np.zeros(16, dtype=np.float32)
        _, result = s.check_action(action)
        np.testing.assert_array_almost_equal(s._last_action, result)


class TestCheckActionEstopped:
    def test_estopped_returns_zeros(self):
        s = _make_checker()
        s.emergency_stop()
        action = np.ones(16, dtype=np.float32)
        was_clipped, result = s.check_action(action)
        assert was_clipped is True
        np.testing.assert_array_equal(result, np.zeros(16))

    def test_estopped_was_clipped_true(self):
        s = _make_checker()
        s.emergency_stop()
        was_clipped, _ = s.check_action(np.ones(16, dtype=np.float32))
        assert was_clipped is True


class TestEmergencyStop:
    def test_emergency_stop_sets_flag(self):
        s = _make_checker()
        s.emergency_stop()
        assert s.is_estopped is True

    def test_emergency_stop_clears_last_action(self):
        s = _make_checker()
        s.check_action(np.ones(16, dtype=np.float32))
        s.emergency_stop()
        assert s._last_action is None

    def test_reset_clears_estop(self):
        s = _make_checker()
        s.emergency_stop()
        s.reset()
        assert s.is_estopped is False

    def test_reset_clears_last_action(self):
        s = _make_checker()
        s.check_action(np.zeros(16, dtype=np.float32))
        s.reset()
        assert s._last_action is None


class TestEstopCommand:
    def test_停_exact_match(self):
        s = _make_checker()
        assert s.is_estop_command("停") is True

    def test_停_with_trailing_punctuation(self):
        s = _make_checker()
        assert s.is_estop_command("停!") is True
        assert s.is_estop_command("停。") is True

    def test_停止播放_not_estop(self):
        """Single-char '停' should not trigger on partial match."""
        s = _make_checker()
        assert s.is_estop_command("停止播放") is False

    def test_stop_exact_match(self):
        s = _make_checker()
        assert s.is_estop_command("stop") is True

    def test_急停_substring(self):
        s = _make_checker()
        assert s.is_estop_command("请急停") is True

    def test_emergency_substring(self):
        s = _make_checker()
        assert s.is_estop_command("this is an emergency") is True

    def test_normal_text_not_estop(self):
        s = _make_checker()
        assert s.is_estop_command("去仓库检查温度") is False

    def test_empty_text_not_estop(self):
        s = _make_checker()
        assert s.is_estop_command("") is False

    def test_case_insensitive_stop(self):
        s = _make_checker()
        assert s.is_estop_command("STOP") is True
        assert s.is_estop_command("Stop") is True
