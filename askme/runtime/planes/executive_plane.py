"""Executive plane: operator IO, memory, skills, executor, supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from askme.runtime.components import CallableComponent, RuntimeComponent

if TYPE_CHECKING:
    from askme.runtime.assembly import RuntimeAssembly


def build_executive_plane(runtime: RuntimeAssembly) -> dict[str, RuntimeComponent]:
    """Build executive-plane components (operator IO, memory, skills, executor)."""
    services = runtime.services
    profile = runtime.profile

    def operator_io_health() -> dict[str, Any]:
        status = runtime.voice_status_snapshot()
        if not profile.has("operator_io"):
            status["status"] = "disabled"
            return status
        status["status"] = "ok" if status.get("pipeline_ok", False) else "degraded"
        return status

    def operator_io_capabilities() -> dict[str, Any]:
        return {
            "primary_loop": profile.primary_loop,
            "voice_enabled": profile.voice_io,
            "text_enabled": profile.text_io,
            "audio_router": services.audio_router is not None,
            "voice_bridge_enabled": services.voice_runtime_bridge.enabled,
            "address_detection": services.address_detector is not None,
        }

    def memory_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "warmup_running": runtime._task_running("memory-warmup"),
            "qp_memory_enabled": services.qp_memory is not None,
        }

    def memory_capabilities() -> dict[str, Any]:
        return {
            "session_memory": True,
            "episodic_memory": True,
            "vector_memory": True,
            "qp_memory": services.qp_memory is not None,
        }

    def executor_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_timeout_s": services.agent_shell._default_timeout,
        }

    def executor_capabilities() -> dict[str, Any]:
        return {
            "tool_count": len(services.tools),
            "background_task_cancel": True,
            "model": getattr(services.agent_shell, "_model", None),
        }

    def skills_health() -> dict[str, Any]:
        enabled_count = len(services.skill_manager.get_enabled())
        status = "ok" if enabled_count else "degraded"
        return {
            "status": status,
            "enabled_skill_count": enabled_count,
            "contract_count": len(services.skill_manager.get_contracts()),
        }

    def skills_capabilities() -> dict[str, Any]:
        return {
            "code_contracts": [
                contract.name
                for contract in services.skill_manager.get_contracts()
                if contract.source == "code"
            ],
            "openapi_generated": True,
            "mcp_catalog_ready": True,
        }

    components: dict[str, RuntimeComponent] = {
        "operator_io": CallableComponent(
            name="operator_io",
            description="Operator-facing audio input/output, wake word, ASR, TTS, and voice bridge.",
            health_hook=operator_io_health,
            capabilities_hook=operator_io_capabilities,
            default_status="disabled",
            depends_on=("memory",),
            provides=("voice", "tts", "asr"),
            profiles=("voice", "mcp", "edge_robot"),
        ),
        "memory": CallableComponent(
            name="memory",
            description="Conversation, episodic, vector, and qp_memory services.",
            start_hook=lambda: runtime._start_task("memory-warmup", services.memory.warmup()),
            stop_hook=lambda: runtime._cancel_task("memory-warmup"),
            health_hook=memory_health,
            capabilities_hook=memory_capabilities,
            depends_on=(),
            provides=("conversation", "episodic", "vector_memory", "qp_memory"),
            profiles=("*",),
        ),
        "executor": CallableComponent(
            name="executor",
            description="Task executor for multi-step tool use and long-running actions.",
            health_hook=executor_health,
            capabilities_hook=executor_capabilities,
            depends_on=("skills",),
            provides=("executor",),
            profiles=("*",),
        ),
        "skills": CallableComponent(
            name="skills",
            description="Skill discovery, dispatch, contracts, and OpenAPI generation.",
            health_hook=skills_health,
            capabilities_hook=skills_capabilities,
            depends_on=("memory",),
            provides=("skills", "openapi", "mcp_catalog"),
            profiles=("*",),
        ),
    }

    if services.proactive is not None:
        components["supervisor"] = CallableComponent(
            name="supervisor",
            description="Background anomaly handling and autonomous follow-up execution.",
            start_hook=lambda: runtime._start_task(
                "proactive-runtime",
                services.proactive.run(runtime.stop_event),
            ),
            stop_hook=lambda: runtime._cancel_task("proactive-runtime"),
            health_hook=lambda: {
                "status": "ok" if runtime._task_running("proactive-runtime") else "degraded",
                "running": runtime._task_running("proactive-runtime"),
            },
            capabilities_hook=lambda: {"enabled": profile.proactive},
            default_status="disabled",
            depends_on=("memory", "skills"),
            provides=("supervision",),
            profiles=("voice", "text", "edge_robot"),
        )

    return components


build_agent_plane = build_executive_plane
