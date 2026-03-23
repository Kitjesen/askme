"""Robot plane: Pulse, safety, control, perception, LED, OTA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from askme.runtime.components import CallableComponent, RuntimeComponent

if TYPE_CHECKING:
    from askme.runtime.assembly import RuntimeAssembly


def build_robot_plane(runtime: RuntimeAssembly) -> dict[str, RuntimeComponent]:
    """Build robot-plane components (Pulse, safety, control, perception, LED)."""
    services = runtime.services
    profile = runtime.profile

    def robot_health() -> dict[str, Any]:
        configured = (
            services.arm_controller is not None
            or services.dog_control.is_configured()
            or services.dog_safety.is_configured()
        )
        status = "ok" if configured else "disabled"
        if profile.robot_api and not configured:
            status = "degraded"
        pulse_health = services.pulse.health()
        return {
            "status": status,
            "arm_controller": services.arm_controller is not None,
            "dog_control_service": services.dog_control.is_configured(),
            "dog_safety_service": services.dog_safety.is_configured(),
            "pulse_connected": pulse_health.get("connected", False),
            "pulse_messages": pulse_health.get("msg_count", 0),
        }

    def robot_capabilities() -> dict[str, Any]:
        pulse_health = services.pulse.health()
        return {
            "local_arm": services.arm_controller is not None,
            "runtime_control_plane": services.dog_control.is_configured(),
            "runtime_safety_plane": services.dog_safety.is_configured(),
            "pulse_enabled": pulse_health.get("available", False),
            "profile_enabled": profile.robot_api,
        }

    def vision_health() -> dict[str, Any]:
        enabled = getattr(services.vision, "_enabled", False)
        return {
            "status": "ok" if enabled else "disabled",
            "vision_enabled": enabled,
            "change_detector_running": runtime._task_running("change-detector"),
        }

    def vision_capabilities() -> dict[str, Any]:
        return {
            "describe_scene": True,
            "find_object": True,
            "change_detector": services.change_detector is not None,
            "capture_backend": getattr(services.vision, "_capture_backend", "unknown"),
        }

    components: dict[str, RuntimeComponent] = {
        "pulse": CallableComponent(
            name="pulse",
            description="Pulse in-process DDS bus for real-time robot telemetry.",
            start_hook=services.pulse.start,
            stop_hook=services.pulse.stop,
            health_hook=lambda: services.pulse.health(),
            capabilities_hook=lambda: {
                "enabled": services.pulse.health().get("available", False),
                "available": services.pulse.health().get("available", False),
            },
            default_status="disabled",
            depends_on=(),
            provides=("pulse", "dds"),
            profiles=("*",),
        ),
        "robot_api": CallableComponent(
            name="robot_api",
            description="Robot arm integration and remote runtime control/safety APIs.",
            health_hook=robot_health,
            capabilities_hook=robot_capabilities,
            default_status="disabled",
            depends_on=("pulse",),
            provides=("arm_control", "dog_control", "dog_safety"),
            profiles=("mcp", "edge_robot"),
        ),
        "vision": CallableComponent(
            name="vision",
            description="Vision bridge, scene understanding, and optional change detection.",
            health_hook=vision_health,
            capabilities_hook=vision_capabilities,
            default_status="disabled",
            depends_on=(),
            provides=("vision", "scene_understanding"),
            profiles=("*",),
        ),
    }

    if services.change_detector is not None:
        components["perception_runtime"] = CallableComponent(
            name="perception_runtime",
            description="Event-driven perception loop backed by ChangeDetector.",
            start_hook=lambda: runtime._start_task(
                "change-detector",
                services.change_detector.run(runtime.stop_event),
            ),
            stop_hook=lambda: runtime._cancel_task("change-detector"),
            health_hook=lambda: {
                "status": "ok" if runtime._task_running("change-detector") else "degraded",
                "running": runtime._task_running("change-detector"),
            },
            capabilities_hook=lambda: {"change_detector": True},
            default_status="disabled",
            depends_on=("pulse", "vision"),
            provides=("change_detection",),
            profiles=("voice", "edge_robot"),
        )

    if services.led_bridge is not None:
        components["signal_runtime"] = CallableComponent(
            name="signal_runtime",
            description="Status LED bridge driven from agent and safety state.",
            start_hook=lambda: runtime._start_task("led-bridge", services.led_bridge.run()),
            stop_hook=lambda: runtime._cancel_task("led-bridge"),
            health_hook=lambda: {
                "status": "ok" if runtime._task_running("led-bridge") else "degraded",
                "running": runtime._task_running("led-bridge"),
            },
            capabilities_hook=lambda: {"led_bridge": profile.led_bridge},
            default_status="disabled",
            depends_on=("pulse",),
            provides=("led_bridge",),
            profiles=("edge_robot",),
        )

    return components
