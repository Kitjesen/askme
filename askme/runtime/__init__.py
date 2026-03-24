"""Runtime assembly primitives for askme."""

from .profiles import (
    EDGE_ROBOT_PROFILE,
    MCP_PROFILE,
    TEXT_PROFILE,
    VOICE_PROFILE,
    RuntimeProfile,
    legacy_profile_for,
)

__all__ = [
    "EDGE_ROBOT_PROFILE",
    "MCP_PROFILE",
    "RuntimeProfile",
    "TEXT_PROFILE",
    "VOICE_PROFILE",
    "legacy_profile_for",
]
