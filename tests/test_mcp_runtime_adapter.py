from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _runtime_app(**modules):
    return SimpleNamespace(modules=modules)


def _make_ctx(app_context):
    ctx = AsyncMock()
    ctx.request_context = MagicMock()
    ctx.request_context.lifespan_context = app_context
    ctx.info = AsyncMock()
    return ctx


def test_app_context_adapter_maps_runtime_app_stable_fields() -> None:
    from askme.mcp.runtime_adapter import app_context_from_runtime_app

    runtime = _runtime_app(
        llm=SimpleNamespace(client="llm-client"),
        memory=SimpleNamespace(
            conversation="conversation",
            session_memory="session-memory",
            memory_bridge="memory-bridge",
            episodic="episodic-memory",
        ),
        tools=SimpleNamespace(
            registry="tool-registry",
            navigation_client="navigation-client",
            temporal_memory_client="temporal-memory-client",
        ),
        skill=SimpleNamespace(
            skill_manager="skill-manager",
            skill_executor="skill-executor",
        ),
        perception=SimpleNamespace(
            vision_bridge="vision-bridge",
            scene_intelligence="scene-intelligence",
        ),
    )

    ctx = app_context_from_runtime_app(runtime, config={"mcp": {"enabled": True}})

    assert ctx.runtime_app is runtime
    assert ctx.config == {"mcp": {"enabled": True}}
    assert ctx.runtime_profile["name"] == "mcp"
    assert ctx.llm_client == "llm-client"
    assert ctx.conversation == "conversation"
    assert ctx.session_memory == "session-memory"
    assert ctx.memory_bridge == "memory-bridge"
    assert ctx.episodic_memory == "episodic-memory"
    assert ctx.tool_registry == "tool-registry"
    assert ctx.navigation_client == "navigation-client"
    assert ctx.temporal_memory_client == "temporal-memory-client"
    assert ctx.skill_manager == "skill-manager"
    assert ctx.skill_executor == "skill-executor"
    assert ctx.vision_bridge == "vision-bridge"
    assert ctx.scene_intelligence == "scene-intelligence"


def test_mcp_runtime_tool_surface_registers_skill_execution_tools() -> None:
    from askme.mcp.context import register_runtime_tool_surface
    from askme.tools.core.tool_registry import ToolRegistry

    class Navigation:
        def dispatch_navigation(self, *args, **kwargs):
            return {"session": {"mission_id": "nav-001", "state": "submitted"}}

        def status(self):
            return {"sessions": []}

    class TemporalMemory:
        def query_temporal_observations(self, params):
            return {"observations": []}

    class RobotControl:
        def dispatch_capability(self, capability, params=None):
            return {"status": "ok", "execution_id": "ctrl-001"}

        def is_configured(self):
            return True

    registry = ToolRegistry()
    navigation = Navigation()
    temporal_memory = TemporalMemory()
    robot_control = RobotControl()
    vision_bridge = object()

    register_runtime_tool_surface(
        registry,
        navigation_client=navigation,
        temporal_memory_client=temporal_memory,
        robot_control_client=robot_control,
        vision_bridge=vision_bridge,
    )

    for tool_name in ("nav_dispatch", "nav_status", "robot_api", "move_robot", "scan_around", "temporal_query"):
        assert registry.get(tool_name) is not None
    assert registry.get("move_robot")._navigation_client is navigation
    assert registry.get("scan_around")._robot_control_client is robot_control
    assert registry.get("scan_around")._vision is vision_bridge
    assert registry.get("temporal_query")._temporal_memory_client is temporal_memory


@pytest.mark.asyncio
async def test_robot_tools_work_with_runtime_app_adapted_context() -> None:
    from askme.mcp.runtime_adapter import app_context_from_runtime_app
    from askme.mcp.tools.robot_tools import robot_estop, robot_move, robot_state

    arm = MagicMock()
    arm.execute = MagicMock(return_value={"status": "ok", "action": "move"})
    arm.get_state = MagicMock(return_value={"connected": True})
    arm.emergency_stop = MagicMock()
    runtime = _runtime_app(control=SimpleNamespace(client=arm))
    app = app_context_from_runtime_app(runtime)
    ctx = _make_ctx(app)

    move = json.loads(await robot_move(1.0, 2.0, 3.0, ctx))
    state = json.loads(await robot_state(ctx))
    estop = json.loads(await robot_estop(ctx))

    assert app.robot_enabled is True
    assert move == {"status": "ok", "action": "move"}
    assert state == {"connected": True}
    assert estop["status"] == "emergency_stop_activated"
    arm.execute.assert_called_once_with("move", {"x": 1.0, "y": 2.0, "z": 3.0})
    arm.get_state.assert_called_once_with()
    arm.emergency_stop.assert_called_once_with()


@pytest.mark.asyncio
async def test_memory_search_uses_runtime_app_adapted_memory_bridge() -> None:
    from askme.mcp.runtime_adapter import app_context_from_runtime_app
    from askme.mcp.tools.memory_tools import memory_search

    bridge = AsyncMock()
    bridge.retrieve = AsyncMock(return_value="- route A\n- route B")
    runtime = _runtime_app(memory=SimpleNamespace(memory_bridge=bridge))
    ctx = _make_ctx(app_context_from_runtime_app(runtime))

    result = json.loads(await memory_search("route", layer="conversation", ctx=ctx))

    assert result == {
        "results": [
            {"text": "route A", "source": "L4_conversation"},
            {"text": "route B", "source": "L4_conversation"},
        ]
    }
    bridge.retrieve.assert_awaited_once_with("route")


def test_runtime_audio_frontend_is_exposed_as_mcp_voice_io() -> None:
    from askme.mcp.runtime_adapter import app_context_from_runtime_app

    class Audio:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []
            self.tts = "audio-tts"

        def listen_loop(self) -> str:
            self.calls.append(("listen_loop", None))
            return "hello"

        def speak(self, text: str) -> None:
            self.calls.append(("speak", text))

        def start_playback(self) -> None:
            self.calls.append(("start_playback", None))

        def wait_speaking_done(self) -> bool:
            self.calls.append(("wait_speaking_done", None))
            return True

        def stop_playback(self) -> None:
            self.calls.append(("stop_playback", None))

    audio = Audio()
    runtime = _runtime_app(
        voice=SimpleNamespace(audio=audio, asr_provider="asr-provider"),
    )
    app = app_context_from_runtime_app(runtime)

    assert app.voice_enabled is True
    assert app.voice_io.listen_once() == "hello"
    app.voice_io.speak_and_wait("hi")
    assert app.tts_engine == "audio-tts"
    assert app.asr_engine == "asr-provider"
    assert audio.calls == [
        ("listen_loop", None),
        ("speak", "hi"),
        ("start_playback", None),
        ("wait_speaking_done", None),
        ("stop_playback", None),
    ]
