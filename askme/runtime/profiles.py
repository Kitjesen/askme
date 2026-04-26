"""Named runtime modes used to assemble askme."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

# ---------------------------------------------------------------------------
# Component bundle constants — used by RuntimeProfile.components frozenset
# ---------------------------------------------------------------------------

_COMPONENT_ALIASES = {
    "voice_io": "operator_io",
    "skill_runtime": "skills",
    "agent_shell": "executor",
    "proactive_runtime": "supervisor",
    "pulse": "telemetry",
    "robot_api": "robot_io",
    "perception_runtime": "change_monitor",
    "signal_runtime": "indicators",
    "control_plane": "diagnostics",
}

_COMMON_COMPONENTS = frozenset({
    "memory", "skills", "executor", "vision",
})

_VOICE_COMPONENTS = _COMMON_COMPONENTS | frozenset({
    "telemetry", "operator_io", "diagnostics", "supervisor",
})

_TEXT_COMPONENTS = _COMMON_COMPONENTS | frozenset({
    "telemetry", "diagnostics", "supervisor",
})

_MCP_COMPONENTS = _COMMON_COMPONENTS | frozenset({
    "telemetry", "operator_io", "robot_io",
})

_EDGE_ROBOT_COMPONENTS = _VOICE_COMPONENTS | frozenset({
    "robot_io", "indicators", "change_monitor",
})


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
    components: frozenset[str] = field(default_factory=frozenset)

    def has(self, component_name: str) -> bool:
        """Check whether *component_name* is included in this profile.

        Falls back to the legacy bool fields when the components set is
        empty (backward-compatible with profiles built via ``replace()``).
        """
        component_name = _COMPONENT_ALIASES.get(component_name, component_name)
        if self.components:
            return component_name in self.components
        # Legacy fallback: map component names to bool fields
        _FIELD_MAP: dict[str, str] = {
            "operator_io": "voice_io",
            "robot_io": "robot_api",
            "diagnostics": "health_http",
            "supervisor": "proactive",
            "indicators": "led_bridge",
            "change_monitor": "change_detector",
        }
        field_name = _FIELD_MAP.get(component_name)
        if field_name is not None:
            return bool(getattr(self, field_name, False))
        # Components not gated by a bool are always included
        return True

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable view for health/introspection endpoints."""
        data = asdict(self)
        # Convert frozenset to sorted list for JSON serialization
        data["components"] = sorted(data.get("components", []))
        return data


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
    components=_VOICE_COMPONENTS | frozenset({"change_monitor"}),
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
    components=_TEXT_COMPONENTS,
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
    components=_MCP_COMPONENTS,
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
    components=_EDGE_ROBOT_COMPONENTS,
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
            components=_TEXT_COMPONENTS | frozenset({"robot_io"}),
        )
    if voice_mode:
        return VOICE_PROFILE
    return TEXT_PROFILE
