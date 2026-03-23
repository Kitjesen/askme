"""Agent plane: LLM pipeline, dispatcher, skills, memory, loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from askme.runtime.components import CallableComponent, RuntimeComponent

if TYPE_CHECKING:
    from askme.runtime.assembly import RuntimeAssembly


def build_agent_plane(runtime: RuntimeAssembly) -> dict[str, RuntimeComponent]:
    """Build agent-plane components (memory, skills, voice/text IO, agent shell)."""
    services = runtime.services
    profile = runtime.profile

    def voice_health() -> dict[str, Any]:
        status = runtime.voice_status_snapshot()
        if not profile.voice_io:
            status["status"] = "disabled"
            return status
        status["status"] = "ok" if status.get("pipeline_ok", False) else "degraded"
        return status

    def voice_capabilities() -> dict[str, Any]:
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

    def agent_shell_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_timeout_s": services.agent_shell._default_timeout,
        }

    def agent_shell_capabilities() -> dict[str, Any]:
        return {
            "tool_count": len(services.tools),
            "background_task_cancel": True,
            "model": getattr(services.agent_shell, "_model", None),
        }

    def skill_runtime_health() -> dict[str, Any]:
        enabled_count = len(services.skill_manager.get_enabled())
        status = "ok" if enabled_count else "degraded"
        return {
            "status": status,
            "enabled_skill_count": enabled_count,
            "contract_count": len(services.skill_manager.get_contracts()),
        }

    def skill_runtime_capabilities() -> dict[str, Any]:
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
        "voice_io": CallableComponent(
            name="voice_io",
            description="Audio input/output, wake word, ASR, TTS, and voice bridge integration.",
            health_hook=voice_health,
            capabilities_hook=voice_capabilities,
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
        "agent_shell": CallableComponent(
            name="agent_shell",
            description="ThunderAgentShell multi-step tool execution runtime.",
            health_hook=agent_shell_health,
            capabilities_hook=agent_shell_capabilities,
            depends_on=("skill_runtime",),
            provides=("agent_shell",),
            profiles=("*",),
        ),
        "skill_runtime": CallableComponent(
            name="skill_runtime",
            description="Skill discovery, dispatch, contracts, and OpenAPI generation.",
            health_hook=skill_runtime_health,
            capabilities_hook=skill_runtime_capabilities,
            depends_on=("memory",),
            provides=("skills", "openapi", "mcp_catalog"),
            profiles=("*",),
        ),
    }

    if services.proactive is not None:
        components["proactive_runtime"] = CallableComponent(
            name="proactive_runtime",
            description="Background proactive anomaly handling and auto-solve orchestration.",
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
            depends_on=("memory", "skill_runtime"),
            provides=("proactive",),
            profiles=("voice", "text", "edge_robot"),
        )

    return components
