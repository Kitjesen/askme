"""MCP resources for robot status and safety configuration."""

from __future__ import annotations

import json
import logging

from askme.mcp.server import mcp

logger = logging.getLogger(__name__)

# ── Module-level helpers ──────────────────────────────────────

_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
    "finger_1", "finger_2", "finger_3", "finger_4",
    "pad_10", "pad_11", "pad_12", "pad_13", "pad_14", "pad_15",
]


# ── Resources ─────────────────────────────────────────────────

@mcp.resource("robot://status")
def robot_status() -> str:
    """Robot arm connection status, mode, and e-stop state."""
    # Note: resources cannot easily receive lifespan context in all MCP SDK
    # versions, so we import the config directly for a lightweight check.
    from askme.config import get_section

    robot_cfg = get_section("robot")
    return json.dumps({
        "enabled": robot_cfg.get("enabled", False),
        "simulate": robot_cfg.get("simulate", True),
        "serial_port": robot_cfg.get("serial_port", "COM3"),
        "message": "Use robot_state() tool for live joint data",
    })


@mcp.resource("robot://joint/{joint_id}/state")
def robot_joint_info(joint_id: str) -> str:
    """Static info about a specific joint (name, limits)."""
    try:
        jid = int(joint_id)
    except ValueError:
        return json.dumps({"error": f"Invalid joint_id: {joint_id}"})

    if jid < 0 or jid >= 16:
        return json.dumps({"error": f"joint_id out of range [0,15]: {jid}"})

    import math

    is_arm = jid < 6
    is_finger = 6 <= jid < 10
    return json.dumps({
        "joint_id": jid,
        "name": _JOINT_NAMES[jid],
        "type": "arm" if is_arm else ("finger" if is_finger else "padding"),
        "limit_min_rad": -math.pi if is_arm else 0.0,
        "limit_max_rad": math.pi if is_arm else (1.5 if is_finger else 0.0),
        "max_velocity_rad_per_step": 0.5 if is_arm else (0.3 if is_finger else 0.0),
    })


@mcp.resource("robot://safety/config")
def robot_safety_config() -> str:
    """Safety system configuration: joint limits, velocity limits, e-stop keywords."""
    from askme.robot.safety import _DEFAULT_CONFIG as safety_defaults

    return json.dumps({
        "arm_joint_limits_rad": ["-pi", "pi"],
        "finger_limits_rad": [0.0, 1.5],
        "arm_max_velocity_rad_per_step": 0.5,
        "finger_max_velocity_rad_per_step": 0.3,
        "estop_keywords": safety_defaults["estop_words"],
    }, ensure_ascii=False)
