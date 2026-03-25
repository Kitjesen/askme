"""
Askme MCP Server — exposes robot control, voice I/O, and skills via MCP.

Usage::

    python -m askme.mcp.server           # stdio (for Claude Desktop/Code)
    askme-mcp                            # via pyproject.toml scripts
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from mcp.server.fastmcp import FastMCP

from askme.config import get_config, get_section, validate_config
from askme.runtime.profiles import MCP_PROFILE

# Logging MUST go to stderr — stdout is the JSON-RPC channel in stdio mode
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
    arm_controller: Any = None
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
    vision_bridge: Any = None
    scene_intelligence: Any = None
    robot_enabled: bool = False
    voice_enabled: bool = False
    runtime_profile: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialise and tear down all askme subsystems."""
    ctx = AppContext()
    ctx.config = get_config()
    ctx.runtime_profile = MCP_PROFILE.snapshot()

    # ── Validate configuration ────────────────────────────────
    for warning in validate_config(ctx.config):
        logger.warning("Config: %s", warning)

    # ── Always init: skills, tools, brain ──────────────────────
    from askme.llm.client import LLMClient
    from askme.llm.conversation import ConversationManager
    from askme.memory.bridge import MemoryBridge
    from askme.memory.session import SessionMemory
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.perception.vision_bridge import VisionBridge
    from askme.perception.scene_intelligence import SceneIntelligence
    from askme.skills.skill_manager import SkillManager
    from askme.skills.skill_executor import SkillExecutor
    from askme.tools.tool_registry import ToolRegistry
    from askme.tools.builtin_tools import register_builtin_tools

    ctx.llm_client = LLMClient()
    ctx.session_memory = SessionMemory(llm=ctx.llm_client)
    ctx.conversation = ConversationManager(session_memory=ctx.session_memory)
    ctx.memory_bridge = MemoryBridge()
    ctx.episodic_memory = EpisodicMemory(llm=ctx.llm_client)
    ctx.vision_bridge = VisionBridge()
    ctx.scene_intelligence = SceneIntelligence(
        episodic=ctx.episodic_memory,
        session=ctx.session_memory,
    )
    ctx.tool_registry = ToolRegistry()
    register_builtin_tools(ctx.tool_registry)

    ctx.skill_manager = SkillManager()
    ctx.skill_manager.load()
    ctx.skill_executor = SkillExecutor(
        ctx.llm_client,
        ctx.tool_registry,
        default_model=ctx.config.get("brain", {}).get("model", "MiniMax-M2.7-highspeed"),
    )

    # ── Robot (conditional) ────────────────────────────────────
    robot_cfg = get_section("robot")
    if robot_cfg.get("enabled", False):
        try:
            from askme.robot.arm_controller import ArmController
            from askme.tools.robot_tools import register_robot_tools

            ctx.arm_controller = ArmController(robot_cfg)
            register_robot_tools(ctx.tool_registry, ctx.arm_controller)
            ctx.robot_enabled = True
            logger.info("Robot arm initialised.")
        except Exception as exc:
            logger.warning("Robot init failed: %s", exc)

    # ── Voice engines (conditional) ────────────────────────────
    voice_cfg = get_section("voice")
    if voice_cfg:
        try:
            from askme.voice.tts import TTSEngine

            ctx.tts_engine = TTSEngine(voice_cfg.get("tts", {}))
            ctx.voice_enabled = True
            logger.info("TTS engine initialised.")
        except Exception as exc:
            logger.warning("TTS init failed: %s", exc)

        try:
            from askme.voice.asr import ASREngine
            from askme.voice.vad import VADEngine

            ctx.asr_engine = ASREngine(voice_cfg.get("asr", {}))
            ctx.vad_engine = VADEngine(voice_cfg.get("vad", {}))
        except Exception as exc:
            logger.warning("ASR/VAD init failed: %s", exc)

    logger.info(
        "Askme MCP server ready (profile=%s, robot=%s, voice=%s, skills=%d)",
        ctx.runtime_profile.get("name", "mcp"),
        ctx.robot_enabled,
        ctx.voice_enabled,
        len(ctx.skill_manager.get_enabled()),
    )

    try:
        yield ctx
    finally:
        logger.info("Shutting down askme MCP server...")
        if ctx.arm_controller:
            ctx.arm_controller.emergency_stop()
            ctx.arm_controller.close()
        if ctx.tts_engine:
            ctx.tts_engine.shutdown()
        logger.info("Shutdown complete.")


# ── FastMCP instance ──────────────────────────────────────────

mcp = FastMCP(
    "askme",
    instructions="Voice AI assistant with robot arm control and skills",
    lifespan=app_lifespan,
)

# Import tool/resource modules to trigger @mcp.tool()/@mcp.resource() registration.
# These MUST be imported after `mcp` is defined.
import askme.mcp.tools.robot_tools as _rt  # noqa: E402, F401
import askme.mcp.tools.voice_tools as _vt  # noqa: E402, F401
import askme.mcp.tools.skill_tools as _st  # noqa: E402, F401
import askme.mcp.resources.robot_resources as _rr  # noqa: E402, F401
import askme.mcp.resources.skill_resources as _sr  # noqa: E402, F401
import askme.mcp.resources.health_resources as _hr  # noqa: E402, F401
import askme.mcp.tools.vision_tools as _vit  # noqa: E402, F401
import askme.mcp.resources.perception_resources as _pr  # noqa: E402, F401
import askme.mcp.tools.memory_tools as _mt  # noqa: E402, F401


def main() -> None:
    """Entry point for ``askme-mcp`` command."""
    mcp.run()


if __name__ == "__main__":
    main()
