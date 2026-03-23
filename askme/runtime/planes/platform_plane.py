"""Platform plane: telemetry, robot IO, vision, change monitor, indicators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from askme.runtime.components import CallableComponent, RuntimeComponent

if TYPE_CHECKING:
    from askme.runtime.assembly import RuntimeAssembly


def build_platform_plane(runtime: RuntimeAssembly) -> dict[str, RuntimeComponent]:
    """Build platform-plane components (telemetry, robot IO, vision, monitors)."""
    services = runtime.services
    profile = runtime.profile

    def robot_io_health() -> dict[str, Any]:
        configured = (
            services.arm_controller is not None
            or services.dog_control.is_configured()
            or services.dog_safety.is_configured()
        )
        status = "ok" if configured else "disabled"
        if profile.robot_api and not configured:
            status = "degraded"
        telemetry_health = services.pulse.health()
        return {
            "status": status,
            "arm_controller": services.arm_controller is not None,
            "dog_control_service": services.dog_control.is_configured(),
            "dog_safety_service": services.dog_safety.is_configured(),
            "telemetry_connected": telemetry_health.get("connected", False),
            "telemetry_messages": telemetry_health.get("msg_count", 0),
        }

    def robot_io_capabilities() -> dict[str, Any]:
        telemetry_health = services.pulse.health()
        return {
            "local_arm": services.arm_controller is not None,
            "runtime_control_plane": services.dog_control.is_configured(),
            "runtime_safety_plane": services.dog_safety.is_configured(),
            "telemetry_enabled": telemetry_health.get("available", False),
            "profile_enabled": profile.robot_api,
        }

    def vision_health() -> dict[str, Any]:
        enabled = getattr(services.vision, "_enabled", False)
        return {
            "status": "ok" if enabled else "disabled",
            "vision_enabled": enabled,
            "change_monitor_running": runtime._task_running("change-detector"),
        }

    def vision_capabilities() -> dict[str, Any]:
        return {
            "describe_scene": True,
            "find_object": True,
            "change_monitor": services.change_detector is not None,
            "capture_backend": getattr(services.vision, "_capture_backend", "unknown"),
        }

    components: dict[str, RuntimeComponent] = {
        "telemetry": CallableComponent(
            name="telemetry",
            description="In-process DDS telemetry bus for real-time robot state and events.",
            start_hook=services.pulse.start,
            stop_hook=services.pulse.stop,
            health_hook=lambda: services.pulse.health(),
            capabilities_hook=lambda: {
                "enabled": services.pulse.health().get("available", False),
                "available": services.pulse.health().get("available", False),
            },
            default_status="disabled",
            depends_on=(),
            provides=("telemetry", "dds"),
            profiles=("*",),
        ),
        "robot_io": CallableComponent(
            name="robot_io",
            description="Robot control, safety, and local platform integration APIs.",
            health_hook=robot_io_health,
            capabilities_hook=robot_io_capabilities,
            default_status="disabled",
            depends_on=("telemetry",),
            provides=("arm_control", "dog_control", "dog_safety"),
            profiles=("mcp", "edge_robot"),
        ),
        "vision": CallableComponent(
            name="vision",
            description="Vision bridge, scene understanding, and optional change monitoring.",
            health_hook=vision_health,
            capabilities_hook=vision_capabilities,
            default_status="disabled",
            depends_on=(),
            provides=("vision", "scene_understanding"),
            profiles=("*",),
        ),
    }

    if services.change_detector is not None:
        components["change_monitor"] = CallableComponent(
            name="change_monitor",
            description="Event-driven change monitoring backed by ChangeDetector.",
            start_hook=lambda: runtime._start_task(
                "change-detector",
                services.change_detector.run(runtime.stop_event),
            ),
            stop_hook=lambda: runtime._cancel_task("change-detector"),
            health_hook=lambda: {
                "status": "ok" if runtime._task_running("change-detector") else "degraded",
                "running": runtime._task_running("change-detector"),
            },
            capabilities_hook=lambda: {"change_monitor": True},
            default_status="disabled",
            depends_on=("telemetry", "vision"),
            provides=("change_monitor",),
            profiles=("voice", "edge_robot"),
        )

    if services.led_bridge is not None:
        components["indicators"] = CallableComponent(
            name="indicators",
            description="Status indicators driven from executor and safety state.",
            start_hook=lambda: runtime._start_task("led-bridge", services.led_bridge.run()),
            stop_hook=lambda: runtime._cancel_task("led-bridge"),
            health_hook=lambda: {
                "status": "ok" if runtime._task_running("led-bridge") else "degraded",
                "running": runtime._task_running("led-bridge"),
            },
            capabilities_hook=lambda: {"led_bridge": profile.led_bridge},
            default_status="disabled",
            depends_on=("telemetry",),
            provides=("indicators",),
            profiles=("edge_robot",),
        )

    return components


build_robot_plane = build_platform_plane
