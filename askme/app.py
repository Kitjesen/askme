"""
Askme application — dependency assembly and lifecycle.

Usage::

    from askme.app import AskmeApp
    import asyncio

    app = AskmeApp(voice_mode=False)
    asyncio.run(app.run())
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys

import os
import re

from askme.config import get_config, get_section
from askme.brain.llm_client import LLMClient
from askme.brain.conversation import ConversationManager
from askme.brain.memory_bridge import MemoryBridge
from askme.brain.episodic_memory import EpisodicMemory
from askme.brain.session_memory import SessionMemory
from askme.brain.vision_bridge import VisionBridge
from askme.brain.intent_router import IntentRouter
from askme.skills.skill_manager import SkillManager
from askme.skills.skill_executor import SkillExecutor
from askme.tools.tool_registry import ToolRegistry
from askme.tools.builtin_tools import register_builtin_tools
from askme.voice.audio_agent import AudioAgent
from askme.voice.runtime_bridge import VoiceRuntimeBridge
from askme.voice.stream_splitter import StreamSplitter
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.commands import CommandHandler
from askme.pipeline.proactive_agent import ProactiveAgent
from askme.pipeline.text_loop import TextLoop
from askme.pipeline.voice_loop import VoiceLoop
from askme.ota_bridge import OTABridge, OTABridgeMetrics

logger = logging.getLogger(__name__)


class AskmeApp:
    """Top-level application: assembles modules and starts the loop."""

    def __init__(self, *, voice_mode: bool = False, robot_mode: bool = False) -> None:
        self.voice_mode = voice_mode
        self.robot_mode = robot_mode

        # Fix Windows console encoding
        if sys.platform == "win32":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

        self.cfg = get_config()
        self._setup_logging()

        # ── Create modules ──────────────────────────────────
        self.ota_metrics = OTABridgeMetrics()
        self.llm = LLMClient(metrics=self.ota_metrics)
        self.session_memory = SessionMemory(llm=self.llm)
        self.conversation = ConversationManager(
            session_memory=self.session_memory,
            metrics=self.ota_metrics,
        )
        self.memory = MemoryBridge()
        self.episodic = EpisodicMemory(llm=self.llm)
        self.vision = VisionBridge()
        self.tools = ToolRegistry()
        register_builtin_tools(self.tools)

        # Robot (optional)
        self.arm_controller = None
        if robot_mode or self.cfg.get("robot", {}).get("enabled", False):
            self._init_robot()

        # Skills
        self.skill_manager = SkillManager()
        self.skill_manager.load()
        self.skill_executor = SkillExecutor(
            self.llm,
            self.tools,
            default_model=self.cfg.get("brain", {}).get("model", "deepseek-chat"),
            metrics=self.ota_metrics,
        )

        # Intent router
        safety = self.arm_controller._safety if self.arm_controller else None  # noqa: SLF001
        self.router = IntentRouter(
            safety_checker=safety,
            voice_triggers=self.skill_manager.get_voice_triggers(),
        )

        # Audio / voice
        self.audio = AudioAgent(self.cfg, voice_mode=voice_mode, metrics=self.ota_metrics)
        self.voice_runtime_bridge = VoiceRuntimeBridge(
            self.cfg.get("runtime", {}).get("voice_bridge", {})
        )
        self.splitter = StreamSplitter()

        # ── Assemble pipeline ───────────────────────────────
        brain_cfg = get_section("brain")
        tools_cfg = get_section("tools")

        # Load SOUL.md character definition (overrides config prompt_seed)
        soul_seed = self._load_soul()
        prompt_seed = soul_seed if soul_seed else brain_cfg.get("prompt_seed", [])

        self._pipeline = BrainPipeline(
            llm=self.llm,
            conversation=self.conversation,
            memory=self.memory,
            tools=self.tools,
            skill_manager=self.skill_manager,
            skill_executor=self.skill_executor,
            audio=self.audio,
            splitter=self.splitter,
            arm_controller=self.arm_controller,
            vision=self.vision,
            session_memory=self.session_memory,
            episodic_memory=self.episodic,
            system_prompt=brain_cfg.get(
                "system_prompt",
                "你是一个有用的AI语音助手。用中文简洁口语化回答。",
            ),
            prompt_seed=prompt_seed,
            user_prefix=brain_cfg.get("user_prefix", ""),
            voice_model=brain_cfg.get("voice_model") if voice_mode else None,
            general_tool_max_safety_level=tools_cfg.get(
                "general_chat_max_safety_level",
                "normal",
            ),
        )

        self._commands = CommandHandler(
            conversation=self.conversation,
            skill_manager=self.skill_manager,
        )

        self._text_loop = TextLoop(
            router=self.router,
            pipeline=self._pipeline,
            commands=self._commands,
            conversation=self.conversation,
            skill_manager=self.skill_manager,
            audio=self.audio,
            voice_runtime_bridge=self.voice_runtime_bridge,
        )

        self._voice_loop = VoiceLoop(
            router=self.router,
            pipeline=self._pipeline,
            audio=self.audio,
            voice_runtime_bridge=self.voice_runtime_bridge,
        )

        self._proactive = ProactiveAgent(
            vision=self.vision,
            audio=self.audio,
            episodic=self.episodic,
            llm=self.llm,
            config=self.cfg,
        )
        self.ota_bridge = OTABridge(
            self.cfg.get("ota", {}),
            metrics=self.ota_metrics,
            voice_status_provider=self.audio.status_snapshot,
            app_name=self.cfg.get("app", {}).get("name", "askme"),
            app_version=self.cfg.get("app", {}).get("version", ""),
            voice_mode=voice_mode,
            robot_mode=self.robot_mode,
        )

        logger.info(
            "Askme initialised (voice=%s, robot=%s, skills=%d, tools=%d)",
            voice_mode, robot_mode,
            len(self.skill_manager.get_enabled()),
            len(self.tools),
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def run(self) -> None:
        """Start the appropriate main loop."""
        # Pre-warm memory in background (avoids cold-start on first query)
        warmup_task = asyncio.create_task(self.memory.warmup())
        stop_event = asyncio.Event()
        proactive_task = asyncio.create_task(self._proactive.run(stop_event))
        self.ota_bridge.start()
        try:
            if self.voice_mode:
                await self._voice_loop.run()
            else:
                await self._text_loop.run()
        finally:
            stop_event.set()
            warmup_task.cancel()
            proactive_task.cancel()
            await asyncio.gather(warmup_task, proactive_task, return_exceptions=True)
            await self.ota_bridge.stop()
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful cleanup."""
        self.audio.shutdown()
        if self.arm_controller:
            self.arm_controller.close()

    # ── Private ──────────────────────────────────────────────

    def _load_soul(self) -> list[dict[str, str]]:
        """Load SOUL.md and convert to prompt seed for character injection.

        Returns a list of fake user/assistant turns that establish the
        character defined in SOUL.md.  Falls back to empty list if the
        file is missing.
        """
        soul_file = self.cfg.get("brain", {}).get("soul_file", "SOUL.md")
        # Resolve relative to project root
        if not os.path.isabs(soul_file):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            soul_file = os.path.join(project_root, soul_file)

        if not os.path.isfile(soul_file):
            return []

        try:
            with open(soul_file, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return []

        # Strip markdown headers but keep content, compact whitespace
        brief = re.sub(r"^#+\s+.*$", "", raw, flags=re.MULTILINE)
        brief = re.sub(r"\n{3,}", "\n\n", brief).strip()

        if not brief:
            return []

        logger.info("Loaded SOUL.md (%d chars) as character definition.", len(brief))

        return [
            {"role": "user", "content": f"你的角色设定如下，严格遵守：\n{brief}"},
            {"role": "assistant", "content": "收到。我是Thunder，穹沛的巡检机器人。等待指令。"},
        ]

    def _init_robot(self) -> None:
        """Initialise the robot arm controller."""
        try:
            from askme.robot.arm_controller import ArmController
            from askme.tools.robot_tools import register_robot_tools

            robot_cfg = get_section("robot")
            self.arm_controller = ArmController(robot_cfg)
            register_robot_tools(self.tools, self.arm_controller)
            self.robot_mode = True
            logger.info("Robot arm controller initialised.")
        except Exception as exc:
            logger.warning("Failed to initialise robot: %s", exc)
            self.arm_controller = None
            self.robot_mode = False

    def _setup_logging(self) -> None:
        """Configure logging from config."""
        level_str = self.cfg.get("app", {}).get("log_level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
