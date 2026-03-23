"""Runtime assembly primitives for askme."""

from .assembly import RuntimeAssembly, RuntimeServices, build_legacy_runtime
from .components import CallableComponent, RuntimeComponent, resolve_start_order
from .profiles import (
    EDGE_ROBOT_PROFILE,
    MCP_PROFILE,
    TEXT_PROFILE,
    VOICE_PROFILE,
    RuntimeProfile,
    legacy_profile_for,
)

__all__ = [
    "CallableComponent",
    "EDGE_ROBOT_PROFILE",
    "MCP_PROFILE",
    "RuntimeAssembly",
    "RuntimeComponent",
    "RuntimeProfile",
    "RuntimeServices",
    "TEXT_PROFILE",
    "VOICE_PROFILE",
    "build_legacy_runtime",
    "legacy_profile_for",
    "resolve_start_order",
]
