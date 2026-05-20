"""Read-only tool catalog for Agent Profile policy validation."""

from __future__ import annotations

AGENT_PROFILE_SYSTEM_TOOLS = frozenset(
    {
        "spawn_agent",
        "dispatch_skill",
        "create_skill",
        "create_agent_profile",
    }
)

AGENT_PROFILE_RUNTIME_TOOLS = frozenset(
    {
        "bash",
        "edit_file",
        "field_event_trigger",
        "find_target",
        "get_current_time",
        "http_request",
        "list_directory",
        "look_around",
        "move_robot",
        "read_file",
        "robot_api",
        "scan_around",
        "space_lookup_place",
        "space_recommend_route",
        "speak_progress",
        "temporal_query",
        "web_fetch",
        "web_search",
        "write_file",
    }
)

AGENT_PROFILE_KNOWN_TOOLS = frozenset(
    AGENT_PROFILE_SYSTEM_TOOLS | AGENT_PROFILE_RUNTIME_TOOLS
)


def agent_profile_known_tools() -> set[str]:
    """Return a copy of known tool names allowed in Agent Profile policies."""

    return set(AGENT_PROFILE_KNOWN_TOOLS)


__all__ = [
    "AGENT_PROFILE_KNOWN_TOOLS",
    "AGENT_PROFILE_RUNTIME_TOOLS",
    "AGENT_PROFILE_SYSTEM_TOOLS",
    "agent_profile_known_tools",
]
