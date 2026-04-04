"""Tests for ArmController — high-level robot arm orchestration."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_controller(**kwargs):
    """Build an ArmController with simulate=True (no hardware needed)."""
    from askme.robot.arm_controller import ArmController
    config = {"simulate": True, **kwargs}
    return ArmController(config=config)


class TestInit:
    def test_bridge_connected_on_init(self):
        ctrl = _make_controller()
        assert ctrl._bridge.is_connected is True

    def test_not_estopped_on_init(self):
        ctrl = _make_controller()
        assert ctrl._safety.is_estopped is False

    def test_policy_not_loaded_without_model(self):
        ctrl = _make_controller()
        assert ctrl._policy.is_loaded is False


class TestGetState:
    def test_returns_dict_with_required_keys(self):
        ctrl = _make_controller()
        state = ctrl.get_state()
        assert "connected" in state
        assert "estopped" in state
        assert "policy_loaded" in state
        assert "joint_angles" in state

    def test_connected_is_true_in_simulate(self):
        ctrl = _make_controller()
        assert ctrl.get_state()["connected"] is True

    def test_joint_angles_is_list(self):
        ctrl = _make_controller()
        angles = ctrl.get_state()["joint_angles"]
        assert isinstance(angles, list)
        assert len(angles) == 16

    def test_estopped_reflects_safety_state(self):
        ctrl = _make_controller()
        ctrl.emergency_stop()
        assert ctrl.get_state()["estopped"] is True


class TestIsConnected:
    def test_is_connected_true_in_simulate(self):
        ctrl = _make_controller()
        assert ctrl.is_connected() is True


class TestEmergencyStop:
    @pytest.mark.asyncio
    async def test_estop_prevents_execute(self):
        ctrl = _make_controller()
        ctrl.emergency_stop()
        result = await ctrl.execute("home")
        assert result["status"] == "error"
        assert "Emergency stop" in result["message"]

    def test_estop_sets_safety_flag(self):
        ctrl = _make_controller()
        ctrl.emergency_stop()
        assert ctrl._safety.is_estopped is True

    def test_reset_clears_estop(self):
        ctrl = _make_controller()
        ctrl.emergency_stop()
        ctrl.reset()
        assert ctrl._safety.is_estopped is False

    @pytest.mark.asyncio
    async def test_execute_works_after_reset(self):
        ctrl = _make_controller()
        ctrl.emergency_stop()
        ctrl.reset()
        result = await ctrl.execute("home")
        assert result["status"] == "ok"


class TestExecuteDirectCommands:
    @pytest.mark.asyncio
    async def test_home_command(self):
        ctrl = _make_controller()
        result = await ctrl.execute("home")
        assert result["status"] == "ok"
        assert result["label"] == "home"

    @pytest.mark.asyncio
    async def test_wave_command(self):
        ctrl = _make_controller()
        result = await ctrl.execute("wave")
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_result_contains_action_list(self):
        ctrl = _make_controller()
        result = await ctrl.execute("home")
        assert isinstance(result["action"], list)
        assert len(result["action"]) == 16

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self):
        ctrl = _make_controller()
        result = await ctrl.execute("fly_to_moon")
        assert result["status"] == "error"
        assert "Unknown action" in result["message"]


class TestExecuteMove:
    @pytest.mark.asyncio
    async def test_move_with_xyz_returns_ok(self):
        ctrl = _make_controller()
        result = await ctrl.execute("move", {"x": 0.1, "y": 0.2, "z": 0.3})
        assert result["status"] == "ok"
        assert "move" in result["label"]

    @pytest.mark.asyncio
    async def test_move_without_params_uses_zero(self):
        ctrl = _make_controller()
        result = await ctrl.execute("move")
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_move_with_policy_loaded(self):
        ctrl = _make_controller()
        # Inject a mock policy
        mock_policy = MagicMock()
        mock_policy.is_loaded = True
        mock_policy.infer.return_value = np.ones(16, dtype=np.float32)
        ctrl._policy = mock_policy
        ctrl._obs_history = np.zeros(285, dtype=np.float32)
        result = await ctrl.execute("move", {"x": 0.5, "y": 0.0, "z": 0.0})
        assert result["status"] == "ok"
        mock_policy.infer.assert_called_once()


class TestExecutePolicy:
    @pytest.mark.asyncio
    async def test_policy_without_model_returns_error(self):
        ctrl = _make_controller()
        result = await ctrl.execute("policy", {"observation": list(range(285))})
        assert result["status"] == "error"
        assert "not loaded" in result["message"]

    @pytest.mark.asyncio
    async def test_policy_missing_observation_returns_error(self):
        ctrl = _make_controller()
        mock_policy = MagicMock()
        mock_policy.is_loaded = True
        ctrl._policy = mock_policy
        result = await ctrl.execute("policy", {})
        assert result["status"] == "error"
        assert "observation" in result["message"]

    @pytest.mark.asyncio
    async def test_policy_runs_inference(self):
        ctrl = _make_controller()
        mock_policy = MagicMock()
        mock_policy.is_loaded = True
        mock_policy.infer.return_value = np.zeros(16, dtype=np.float32)
        ctrl._policy = mock_policy
        obs = list(range(285))
        result = await ctrl.execute("policy", {"observation": obs})
        assert result["status"] == "ok"
        mock_policy.infer.assert_called_once()


class TestClose:
    def test_close_disconnects_bridge(self):
        ctrl = _make_controller()
        ctrl.close()
        assert ctrl._bridge.is_connected is False
