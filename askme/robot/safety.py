"""
Safety checker for robot actions.

Enforces joint limits, velocity limits, and emergency-stop detection
from voice commands. All actions are clipped to safe ranges before
being sent to the hardware.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default safety configuration
_DEFAULT_CONFIG: dict[str, Any] = {
    # Per-joint min/max limits (radians), 16 joints
    "joint_limits": {
        "min": [-3.14] * 6 + [0.0] * 10,   # arm joints +/- pi, fingers 0+
        "max": [3.14] * 6 + [1.5] * 10,     # fingers max grip
    },
    # Maximum change per step (radians)
    "velocity_limits": {
        "max_delta": [0.5] * 6 + [0.3] * 10,
    },
    # Words that trigger emergency stop
    "estop_words": [
        "停", "stop", "急停", "紧急停止", "emergency",
        "别动", "不要动", "halt", "freeze", "危险",
    ],
}


class SafetyChecker:
    """Validates and clips robot actions to safe ranges."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Safety configuration dict. Uses defaults if not provided.
                Expected keys: joint_limits, velocity_limits, estop_words.
        """
        cfg = config or _DEFAULT_CONFIG

        jl = cfg.get("joint_limits", _DEFAULT_CONFIG["joint_limits"])
        self._joint_min = np.array(jl["min"], dtype=np.float32)
        self._joint_max = np.array(jl["max"], dtype=np.float32)

        vl = cfg.get("velocity_limits", _DEFAULT_CONFIG["velocity_limits"])
        self._max_delta = np.array(vl["max_delta"], dtype=np.float32)

        self._estop_words: list[str] = cfg.get(
            "estop_words", _DEFAULT_CONFIG["estop_words"]
        )

        self._estopped: bool = False
        self._last_action: np.ndarray | None = None

    @property
    def is_estopped(self) -> bool:
        """Whether the emergency stop is currently active."""
        return self._estopped

    def check_action(self, action: np.ndarray) -> tuple[bool, np.ndarray]:
        """Validate and clip an action to safe ranges.

        Args:
            action: A 16-dim numpy array of target joint angles.

        Returns:
            (was_clipped, safe_action): True if any values were clipped,
            plus the clipped action array.
        """
        if self._estopped:
            logger.warning("E-stop active, returning zero action.")
            return (True, np.zeros_like(action))

        original = action.copy()

        # 1. Clip to joint limits
        safe = np.clip(action, self._joint_min[: len(action)], self._joint_max[: len(action)])

        # 2. Clip velocity (delta from last action)
        if self._last_action is not None:
            delta = safe - self._last_action
            max_d = self._max_delta[: len(action)]
            delta_clipped = np.clip(delta, -max_d, max_d)
            safe = self._last_action + delta_clipped

            # Re-clip to joint limits after velocity adjustment
            safe = np.clip(safe, self._joint_min[: len(action)], self._joint_max[: len(action)])

        was_clipped = not np.allclose(original, safe, atol=1e-6)
        self._last_action = safe.copy()

        if was_clipped:
            logger.debug("Action was clipped by safety checker.")

        return (was_clipped, safe)

    def is_estop_command(self, text: str) -> bool:
        """Check if text contains an emergency-stop keyword.

        Args:
            text: User input text to check.

        Returns:
            True if an e-stop word is found.
        """
        text_lower = text.lower().strip()
        for word in self._estop_words:
            if word.lower() in text_lower:
                return True
        return False

    def emergency_stop(self) -> None:
        """Activate emergency stop. All subsequent actions will be zeroed."""
        self._estopped = True
        self._last_action = None
        logger.critical("EMERGENCY STOP activated!")

    def reset(self) -> None:
        """Reset the emergency stop. Allows actions to proceed again."""
        self._estopped = False
        self._last_action = None
        logger.info("Safety checker reset, e-stop cleared.")
