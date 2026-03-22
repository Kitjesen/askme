"""Named runtime profiles used to assemble askme."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    """A named composition profile for an askme runtime."""

    name: str
    description: str
    primary_loop: str
    voice_io: bool
    text_io: bool
    mcp: bool
    robot_api: bool
    health_http: bool
    proactive: bool
    led_bridge: bool
    change_detector: bool
    http_chat: bool

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable view for health/introspection endpoints."""
        return asdict(self)


VOICE_PROFILE = RuntimeProfile(
    name="voice",
    description="Interactive voice runtime with health endpoints and proactive perception.",
    primary_loop="voice",
    voice_io=True,
    text_io=True,
    mcp=False,
    robot_api=False,
    health_http=True,
    proactive=True,
    led_bridge=False,
    change_detector=True,
    http_chat=True,
)

TEXT_PROFILE = RuntimeProfile(
    name="text",
    description="Interactive text runtime with shared services and health endpoints.",
    primary_loop="text",
    voice_io=False,
    text_io=True,
    mcp=False,
    robot_api=False,
    health_http=True,
    proactive=True,
    led_bridge=False,
    change_detector=False,
    http_chat=True,
)

MCP_PROFILE = RuntimeProfile(
    name="mcp",
    description="Tool and resource serving profile for MCP transports.",
    primary_loop="mcp",
    voice_io=True,
    text_io=False,
    mcp=True,
    robot_api=True,
    health_http=False,
    proactive=False,
    led_bridge=False,
    change_detector=False,
    http_chat=False,
)

EDGE_ROBOT_PROFILE = RuntimeProfile(
    name="edge_robot",
    description="Voice-first edge runtime with robot APIs, LED state, and event-driven perception.",
    primary_loop="voice",
    voice_io=True,
    text_io=True,
    mcp=False,
    robot_api=True,
    health_http=True,
    proactive=True,
    led_bridge=True,
    change_detector=True,
    http_chat=True,
)


def legacy_profile_for(*, voice_mode: bool, robot_mode: bool) -> RuntimeProfile:
    """Resolve the legacy CLI flags into a named runtime profile."""
    if voice_mode and robot_mode:
        return EDGE_ROBOT_PROFILE
    if robot_mode:
        return replace(
            TEXT_PROFILE,
            robot_api=True,
            description="Interactive text runtime with robot APIs and health endpoints.",
        )
    if voice_mode:
        return VOICE_PROFILE
    return TEXT_PROFILE
