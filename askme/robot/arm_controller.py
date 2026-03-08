"""
High-level arm controller for askme robot.

Orchestrates PolicyRunner, DirectCommands, SafetyChecker, and SerialBridge
to provide a unified async interface for robot control.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import direct_commands
from .policy_runner import PolicyRunner
from .safety import SafetyChecker
from .serial_bridge import SerialBridge

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_CONFIG: dict[str, Any] = {
    "policy_model_path": "models/policy/policy_pose0111.onnx",
    "obs_dim": 285,
    "action_dim": 16,
    "serial_port": "COM3",
    "serial_baudrate": 115200,
    "simulate": True,
    "safety": None,  # Use default safety config
}


class ArmController:
    """Unified robot arm controller.

    Dispatches commands to either the RL policy or direct joint commands,
    applies safety checks, and sends actions via the serial bridge.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Configuration dict. Uses defaults if not provided.
                Keys: policy_model_path, obs_dim, action_dim,
                      serial_port, serial_baudrate, simulate, safety.
        """
        cfg = {**_DEFAULT_CONFIG, **(config or {})}

        # Initialize sub-components
        self._policy = PolicyRunner(
            model_path=cfg["policy_model_path"],
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
        )
        self._safety = SafetyChecker(cfg.get("safety"))
        self._bridge = SerialBridge(
            port=cfg["serial_port"],
            baudrate=cfg["serial_baudrate"],
            simulate=cfg["simulate"],
        )

        # Observation history buffer for policy inference
        self._obs_history: np.ndarray | None = None

        # Connect the serial bridge
        self._bridge.connect()

    # ── Public API ──────────────────────────────────────────────

    async def execute(
        self,
        action_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a robot action by name.

        Supported actions:
          - Any direct command name (home, wave, grab, release, point_forward, rest)
          - 'move': uses policy or direct positioning with params {x, y, z}
          - 'policy': runs the RL policy with given observation

        Args:
            action_name: Name of the action.
            params: Optional parameters for the action.

        Returns:
            A dict with 'status', 'action' (the sent joint angles), and
            optional 'clipped' flag.
        """
        if self._safety.is_estopped:
            return {
                "status": "error",
                "message": "Emergency stop is active. Call reset() first.",
            }

        params = params or {}

        # Check if it's a direct command
        direct_action = direct_commands.get_command(action_name)
        if direct_action is not None:
            return await self._send_action(direct_action, action_name)

        # Special actions
        if action_name == "move":
            return await self._handle_move(params)
        elif action_name == "policy":
            return await self._handle_policy(params)
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action_name}. "
                           f"Available: {direct_commands.list_commands() + ['move', 'policy']}",
            }

    def emergency_stop(self) -> None:
        """Immediately stop all robot motion."""
        self._safety.emergency_stop()
        # Send zero action to halt
        zero_action = np.zeros(16, dtype=np.float32)
        self._bridge.send_action(zero_action)
        logger.critical("ArmController: EMERGENCY STOP executed.")

    def reset(self) -> None:
        """Reset the emergency stop, allowing actions again."""
        self._safety.reset()
        logger.info("ArmController: Reset from emergency stop.")

    def get_state(self) -> dict[str, Any]:
        """Get the current robot state.

        Returns:
            Dict with 'connected', 'estopped', 'joint_angles', and 'policy_loaded'.
        """
        state = self._bridge.get_state()
        return {
            "connected": self._bridge.is_connected,
            "estopped": self._safety.is_estopped,
            "policy_loaded": self._policy.is_loaded,
            "joint_angles": state.tolist() if state is not None else None,
        }

    def is_connected(self) -> bool:
        """Whether the serial bridge is connected."""
        return self._bridge.is_connected

    def close(self) -> None:
        """Clean up resources."""
        self._policy.close()
        self._bridge.disconnect()

    # ── Internal Dispatch ───────────────────────────────────────

    async def _send_action(
        self,
        action: np.ndarray,
        label: str = "action",
    ) -> dict[str, Any]:
        """Apply safety checks and send action to robot."""
        was_clipped, safe_action = self._safety.check_action(action)
        success = self._bridge.send_action(safe_action)

        if not success:
            return {"status": "error", "message": "Failed to send action via serial bridge."}

        return {
            "status": "ok",
            "action": safe_action.tolist(),
            "clipped": was_clipped,
            "label": label,
        }

    async def _handle_move(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle 'move' action: position-based move using policy or interpolation."""
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)
        z = params.get("z", 0.0)

        if self._policy.is_loaded and self._obs_history is not None:
            # Use RL policy with position as part of observation
            obs = self._obs_history.copy()
            # Inject target position into observation (convention: last 3 dims)
            obs[-3:] = [x, y, z]
            action = self._policy.infer(obs)
            return await self._send_action(action, f"move({x},{y},{z})")

        # Fallback: simple direct positioning via home + offsets
        # This is a placeholder; real IK would go here
        base_action = direct_commands.get_command("home")
        if base_action is None:
            base_action = np.zeros(16, dtype=np.float32)

        # Crude mapping: scale x/y/z to joint offsets
        base_action[0] += float(x) * 0.001  # shoulder pan
        base_action[1] += float(z) * 0.001  # shoulder lift
        base_action[2] += float(y) * 0.001  # elbow

        return await self._send_action(base_action, f"move({x},{y},{z})")

    async def _handle_policy(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle 'policy' action: run RL policy inference."""
        if not self._policy.is_loaded:
            return {"status": "error", "message": "Policy model not loaded."}

        obs = params.get("observation")
        if obs is None:
            return {"status": "error", "message": "Missing 'observation' parameter."}

        obs_array = np.array(obs, dtype=np.float32)
        self._obs_history = obs_array

        action = self._policy.infer(obs_array)
        return await self._send_action(action, "policy")
