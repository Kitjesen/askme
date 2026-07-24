"""Shared MCP application context and lifecycle wiring."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import FastMCP

from askme.config import get_config, get_section, validate_config
from askme.ports import (
    ArmControlPort,
    NavigationPort,
    RobotControlPort,
    SceneIntelligencePort,
    TemporalMemoryPort,
    VisionPort,
    VoiceIOPort,
    SpeechPlaybackPort,
)
from askme.runtime.core.profiles import MCP_PROFILE

# Logging MUST go to stderr; stdout is the JSON-RPC channel in stdio mode.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Shared application state accessible by all MCP tools and resources."""

    config: dict[str, Any] = field(default_factory=dict)
    runtime_app: Any = None
    arm_controller: ArmControlPort | None = None
    navigation_client: NavigationPort | None = None
    temporal_memory_client: TemporalMemoryPort | None = None
    robot_control_client: RobotControlPort | None = None
    voice_io: VoiceIOPort | None = None
    speech_playback: SpeechPlaybackPort | None = None
    tts_engine: Any = None
    asr_engine: Any = None
    vad_engine: Any = None
    skill_manager: Any = None
    skill_executor: Any = None
    tool_registry: Any = None
    llm_client: Any = None
    conversation: Any = None
    memory_bridge: Any = None
    session_memory: Any = None
    episodic_memory: Any = None
    vision_bridge: VisionPort | None = None
    scene_intelligence: SceneIntelligencePort | None = None
    robot_enabled: bool = False
    voice_enabled: bool = False
    runtime_profile: dict[str, Any] = field(default_factory=dict)


def register_runtime_tool_surface(
    tool_registry: Any,
    *,
    production_mode: bool = False,
    robot_control_client: RobotControlPort | None = None,
    navigation_client: NavigationPort | None = None,
    temporal_memory_client: TemporalMemoryPort | None = None,
    vision_bridge: VisionPort | None = None,
) -> None:
    """Register the runtime tool surface used by MCP skill execution."""

    from askme.tools.core.builtin_tools import register_builtin_tools
    from askme.tools.robot.move_tool import register_move_tools
    from askme.tools.robot.robot_api_tool import RobotApiTool
    from askme.tools.spatial.scan_tool import register_scan_tools
    from askme.tools.spatial.temporal_query_tool import register_temporal_tools

    register_builtin_tools(
        tool_registry,
        production_mode=production_mode,
        navigation_client=navigation_client,
        robot_control_client=robot_control_client,
    )
    tool_registry.register(RobotApiTool())
    register_move_tools(
        tool_registry,
        navigation_client=navigation_client,
        robot_control_client=robot_control_client,
    )
    register_scan_tools(
        tool_registry,
        vision=vision_bridge,
        robot_control_client=robot_control_client,
    )
    register_temporal_tools(
        tool_registry,
        temporal_memory_client=temporal_memory_client,
    )


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialise and tear down all askme subsystems."""
    ctx = AppContext()
    ctx.config = get_config()
    ctx.runtime_profile = MCP_PROFILE.snapshot()

    for warning in validate_config(ctx.config):
        logger.warning("Config: %s", warning)

    from askme.llm.core.client import LLMClient
    from askme.memory.core.conversation import ConversationManager
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.memory.core.session import SessionMemory
    from askme.memory.retrieval.bridge import MemoryBridge
    from askme.providers import (
        build_navigation,
        build_perception,
        build_robot_control,
        build_scene_intelligence,
        build_temporal_memory,
    )
    from askme.skills.core.skill_executor import SkillExecutor
    from askme.skills.core.skill_manager import SkillManager
    from askme.tools.core.tool_registry import ToolRegistry

    ctx.llm_client = LLMClient()
    ctx.session_memory = SessionMemory(llm=ctx.llm_client)
    ctx.conversation = ConversationManager(session_memory=ctx.session_memory)
    ctx.memory_bridge = MemoryBridge()
    ctx.episodic_memory = EpisodicMemory(llm=ctx.llm_client)
    perception_stack = build_perception(ctx.config)
    ctx.vision_bridge = perception_stack.vision
    ctx.scene_intelligence = build_scene_intelligence(
        episodic=ctx.episodic_memory,
        session=ctx.session_memory,
    )
    ctx.robot_control_client = build_robot_control(
        ctx.config.get("runtime", {}).get("dog_control", {}),
    )
    ctx.navigation_client = build_navigation(
        ctx.config.get("runtime", {}).get("navigation", {}),
    )
    ctx.temporal_memory_client = build_temporal_memory(
        ctx.config.get("runtime", {}).get("navigation", {}),
    )
    ctx.tool_registry = ToolRegistry()
    register_runtime_tool_surface(
        ctx.tool_registry,
        navigation_client=ctx.navigation_client,
        robot_control_client=ctx.robot_control_client,
        vision_bridge=ctx.vision_bridge,
        temporal_memory_client=ctx.temporal_memory_client,
    )

    ctx.skill_manager = SkillManager()
    ctx.skill_manager.load()
    ctx.skill_executor = SkillExecutor(
        ctx.llm_client,
        ctx.tool_registry,
        default_model=ctx.config.get("brain", {}).get("model", "MiniMax-M2.7-highspeed"),
    )

    robot_cfg = get_section("robot")
    if robot_cfg.get("enabled", False):
        try:
            from askme.providers import build_arm_control
            from askme.tools.robot.robot_tools import register_robot_tools

            ctx.arm_controller = build_arm_control(robot_cfg)
            register_robot_tools(ctx.tool_registry, ctx.arm_controller)
            ctx.robot_enabled = True
            logger.info("Robot arm initialised.")
        except Exception as exc:
            logger.warning("Robot init failed: %s", exc)

    voice_cfg = get_section("voice")
    if voice_cfg:
        try:
            from askme.providers import build_edge_voice_io, build_speech_playback

            ctx.voice_io = build_edge_voice_io(ctx.config)
            ctx.tts_engine = getattr(ctx.voice_io, "tts", None)
            ctx.asr_engine = getattr(ctx.voice_io, "asr", None)
            ctx.vad_engine = getattr(ctx.voice_io, "vad", None)
            ctx.speech_playback = build_speech_playback(ctx.config, audio=ctx.voice_io)
            await ctx.speech_playback.start()
            ctx.voice_enabled = True
            logger.info("Voice I/O initialised.")
        except Exception as exc:
            logger.warning("Voice I/O init failed: %s", exc)

    logger.info(
        "Askme MCP server ready (profile=%s, robot=%s, voice=%s, skills=%d)",
        ctx.runtime_profile.get("name", "mcp"),
        ctx.robot_enabled,
        ctx.voice_enabled,
        len(ctx.skill_manager.get_enabled()),
    )

    from askme.mcp.resource_surface import resource_surface_from_context, set_resource_surface

    previous_resource_surface = set_resource_surface(resource_surface_from_context(ctx))
    try:
        yield ctx
    finally:
        set_resource_surface(previous_resource_surface)
        logger.info("Shutting down askme MCP server...")
        if ctx.arm_controller:
            ctx.arm_controller.emergency_stop()
            ctx.arm_controller.close()
        if ctx.speech_playback:
            await ctx.speech_playback.shutdown()
        if ctx.tts_engine:
            ctx.tts_engine.shutdown()
        logger.info("Shutdown complete.")
