"""Diagnostics plane: health server, live endpoints, metrics, capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from askme.runtime.components import CallableComponent, RuntimeComponent

if TYPE_CHECKING:
    from askme.runtime.assembly import RuntimeAssembly


def build_diagnostics_plane(runtime: RuntimeAssembly) -> dict[str, RuntimeComponent]:
    """Build diagnostics-plane components (health HTTP, metrics, capabilities)."""
    services = runtime.services
    profile = runtime.profile
    components: dict[str, RuntimeComponent] = {}

    if services.health_server is not None:
        components["diagnostics"] = CallableComponent(
            name="diagnostics",
            description="Embedded diagnostics endpoints for health, metrics, chat, and capabilities.",
            start_hook=services.health_server.start,
            stop_hook=services.health_server.stop,
            health_hook=lambda: {
                "status": "ok" if services.health_server.enabled else "disabled",
                "url": services.health_server.url if services.health_server.enabled else "",
            },
            capabilities_hook=lambda: {
                "health_http": profile.health_http,
                "http_chat": profile.http_chat,
            },
            default_status="disabled",
            depends_on=(),
            provides=("health_http", "http_chat", "capabilities"),
            profiles=("voice", "text", "edge_robot"),
        )

    return components


build_control_plane = build_diagnostics_plane
