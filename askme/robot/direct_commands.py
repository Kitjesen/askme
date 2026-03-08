"""
Predefined direct joint-position commands for common robot actions.

Each command is a 16-dimensional numpy array representing target joint
angles for the robot arm. These bypass the RL policy and move the arm
to known safe positions directly.
"""

from __future__ import annotations

import numpy as np

# 16-dim joint angle arrays for predefined poses (radians)
# Joints: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3,
#           finger_1, finger_2, finger_3, finger_4, ..., padding]

_COMMANDS: dict[str, np.ndarray] = {
    "home": np.array([
        0.0, -0.5, 0.5, -0.5, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32),

    "wave": np.array([
        0.0, -0.8, 1.2, -1.0, 0.0, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32),

    "grab": np.array([
        0.0, -0.5, 0.5, -0.5, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32),

    "release": np.array([
        0.0, -0.5, 0.5, -0.5, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32),

    "point_forward": np.array([
        0.0, -0.3, 0.8, -0.5, 0.0, 0.0,
        0.8, 0.0, 0.8, 0.8,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32),

    "rest": np.array([
        0.0, -1.0, 1.5, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32),
}


def get_command(name: str) -> np.ndarray | None:
    """Get a predefined joint-angle command by name.

    Args:
        name: Command name (e.g. 'home', 'wave', 'grab').

    Returns:
        A copy of the 16-dim numpy array, or None if name not found.
    """
    cmd = _COMMANDS.get(name)
    return cmd.copy() if cmd is not None else None


def list_commands() -> list[str]:
    """Return a sorted list of available command names."""
    return sorted(_COMMANDS.keys())
