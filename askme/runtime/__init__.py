"""Runtime assembly primitives for askme."""

from .profiles import (
    EDGE_ROBOT_MODE,
    EDGE_ROBOT_PROFILE,
    MCP_MODE,
    MCP_PROFILE,
    RuntimeMode,
    TEXT_PROFILE,
    TEXT_MODE,
    VOICE_PROFILE,
    VOICE_MODE,
    RuntimeProfile,
    legacy_profile_for,
    mode_for,
)

__all__ = [
    "EDGE_ROBOT_MODE",
    "EDGE_ROBOT_PROFILE",
    "MCP_MODE",
    "MCP_PROFILE",
    "RuntimeMode",
    "RuntimeProfile",
    "TEXT_PROFILE",
    "TEXT_MODE",
    "VOICE_PROFILE",
    "VOICE_MODE",
    "legacy_profile_for",
    "mode_for",
]
