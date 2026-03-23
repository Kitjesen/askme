"""Runtime assembly primitives for askme."""

from .assembly import RuntimeAssembly, RuntimeServices, build_legacy_runtime, build_runtime
from .components import CallableComponent, RuntimeComponent, resolve_start_order
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
    "CallableComponent",
    "EDGE_ROBOT_MODE",
    "EDGE_ROBOT_PROFILE",
    "MCP_MODE",
    "MCP_PROFILE",
    "RuntimeMode",
    "RuntimeAssembly",
    "RuntimeComponent",
    "RuntimeProfile",
    "RuntimeServices",
    "TEXT_PROFILE",
    "TEXT_MODE",
    "VOICE_PROFILE",
    "VOICE_MODE",
    "build_legacy_runtime",
    "build_runtime",
    "legacy_profile_for",
    "mode_for",
    "resolve_start_order",
]
