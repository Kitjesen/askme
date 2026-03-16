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
import os
import re
import sys

from askme import __version__ as ASKME_VERSION
from askme.brain.conversation import ConversationManager
from askme.dog_control_client import DogControlClient
from askme.dog_safety_client import DogSafetyClient
from askme.brain.episodic_memory import EpisodicMemory
from askme.brain.intent_router import IntentRouter
from askme.brain.memory_system import MemorySystem
from askme.brain.llm_client import LLMClient
from askme.brain.memory_bridge import MemoryBridge
from askme.brain.session_memory import SessionMemory
from askme.brain.vision_bridge import VisionBridge
from askme.config import get_config, get_section, validate_config
from askme.health_server import (
    AskmeHealthServer,
    build_health_snapshot,
    merge_voice_pipeline_status,
)
from askme.ota_bridge import OTABridgeMetrics
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.commands import CommandHandler
from askme.pipeline.proactive_agent import ProactiveAgent
from askme.pipeline.planner_agent import PlannerAgent
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.pipeline.text_loop import TextLoop
from askme.pipeline.voice_loop import VoiceLoop
from askme.skills.skill_executor import SkillExecutor
from askme.skills.skill_manager import SkillManager
from askme.tools.builtin_tools import DispatchSkillTool, register_builtin_tools
from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
from askme.led_controller import HttpLedController, NullLedController
from askme.pipeline.state_led_bridge import StateLedBridge
from askme.tools.robot_api_tool import RobotApiTool
from askme.tools.builtin_tools import SpeakProgressTool
from askme.tools.skill_tools import register_skill_tools
from askme.tools.tool_registry import ToolRegistry
from askme.tools.voice_tools import register_voice_tools
from askme.voice.address_detector import AddressDetector
from askme.voice.audio_agent import AudioAgent
from askme.voice.runtime_bridge import VoiceRuntimeBridge
from askme.voice.stream_splitter import StreamSplitter

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
        self._app_name = self.cfg.get("app", {}).get("name", "askme")
        self._app_version = self.cfg.get("app", {}).get("version") or ASKME_VERSION
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
        self.memory_system = MemorySystem(
            llm=self.llm,
            conversation=self.conversation,
            session_memory=self.session_memory,
            episodic=self.episodic,
            vector_memory=self.memory,
        )
        self.vision = VisionBridge()
        self.tools = ToolRegistry()
        _production_mode = bool(self.cfg.get("tools", {}).get("production_mode", False))
        register_builtin_tools(self.tools, production_mode=_production_mode)
        # Register RobotApiTool (unified runtime API)
        self.tools.register(RobotApiTool())

        # Robot (optional)
        self.arm_controller = None
        if robot_mode or self.cfg.get("robot", {}).get("enabled", False):
            self._init_robot()

        # Skills
        self.skill_manager = SkillManager()
        self.skill_manager.load()
        # In voice mode, skills use the faster voice_model for low latency
        brain_cfg_skills = self.cfg.get("brain", {})
        skill_model = (
            brain_cfg_skills.get("voice_model")
            if voice_mode
            else brain_cfg_skills.get("model")
        ) or brain_cfg_skills.get("model", "claude-sonnet-4-5-20250929")
        self.skill_executor = SkillExecutor(
            self.llm,
            self.tools,
            default_model=skill_model,
            metrics=self.ota_metrics,
        )

        # Intent router
        safety = self.arm_controller._safety if self.arm_controller else None  # noqa: SLF001
        self.router = IntentRouter(
            safety_checker=safety,
            voice_triggers=self.skill_manager.get_voice_triggers(),
        )

        # Audio router — shared between AudioAgent (input) and TTSEngine (output).
        # Serialises mic open and aplay on half-duplex ALSA hardware (sunrise).
        # None in text-only mode — no audio device coordination needed.
        from askme.voice.audio_router import AudioRouter
        self.audio_router: AudioRouter | None = AudioRouter() if voice_mode else None

        # Audio / voice
        self.audio = AudioAgent(
            self.cfg,
            voice_mode=voice_mode,
            metrics=self.ota_metrics,
            audio_router=self.audio_router,
        )
        register_voice_tools(self.tools, self.audio)
        self.tools.register(SpeakProgressTool(self.audio))
        self.voice_runtime_bridge = VoiceRuntimeBridge(
            self.cfg.get("runtime", {}).get("voice_bridge", {})
        )
        self.splitter = StreamSplitter()

        # Dog safety client (optional — enabled when DOG_SAFETY_SERVICE_URL is set)
        self.dog_safety = DogSafetyClient(
            self.cfg.get("runtime", {}).get("dog_safety", {})
        )

        # Dog control client (optional — enabled when DOG_CONTROL_SERVICE_URL is set)
        self.dog_control = DogControlClient(
            self.cfg.get("runtime", {}).get("dog_control", {})
        )

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
            dog_safety_client=self.dog_safety,
            dog_control_client=self.dog_control,
            vision=self.vision,
            session_memory=self.session_memory,
            episodic_memory=self.episodic,
            memory_system=self.memory_system,
            system_prompt=brain_cfg.get(
                "system_prompt",
                "你是一个有用的AI语音助手。用中文简洁口语化回答。",
            ),
            prompt_seed=prompt_seed,
            user_prefix=brain_cfg.get("user_prefix", ""),
            voice_model=brain_cfg.get("voice_model"),
            general_tool_max_safety_level=tools_cfg.get(
                "general_chat_max_safety_level",
                "normal",
            ),
            max_response_chars=int(brain_cfg.get("max_response_chars", 0)),
        )

        # ── Skill dispatcher (unified orchestration) ────────────
        # plan_model: fast/cheap model for JSON-only planning task (e.g. haiku).
        # Falls back to the LLM client default when not configured.
        _plan_model = brain_cfg.get("plan_model")
        _planner = PlannerAgent(
            llm_client=self._pipeline._llm,
            skill_manager=self.skill_manager,
            model=_plan_model,
        )
        self.dispatcher = SkillDispatcher(
            pipeline=self._pipeline,
            skill_manager=self.skill_manager,
            audio=self.audio,
            planner=_planner,
        )

        # ── ThunderAgentShell (agentic task execution) ───────────────
        _agent_model = brain_cfg.get("agent_model")
        self.agent_shell = ThunderAgentShell(
            llm_client=self.llm,
            tool_registry=self.tools,
            audio=self.audio,
            model=_agent_model,
        )
        # Allow config override for agent task timeout (default 120s)
        self.agent_shell._default_timeout = float(brain_cfg.get("agent_timeout", 120.0))
        # Wire agent_shell into the pipeline
        self._pipeline._agent_shell = self.agent_shell

        # Register dispatch_skill meta-tool (LLM can invoke skills)
        dispatch_tool = DispatchSkillTool()
        dispatch_tool.set_dispatcher(self.dispatcher)
        self.tools.register(dispatch_tool)

        # Register create_skill tool (LLM can create new skills at runtime)
        register_skill_tools(self.tools, self.skill_manager, self.router)

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
            dispatcher=self.dispatcher,
        )

        self._voice_loop = VoiceLoop(
            router=self.router,
            pipeline=self._pipeline,
            audio=self.audio,
            voice_runtime_bridge=self.voice_runtime_bridge,
            dispatcher=self.dispatcher,
            audio_router=self.audio_router,
        )

        # Address detector: filter out bystander chat (not talking to robot)
        _ad_cfg = self.cfg.get("voice", {}).get("address_detection", {})
        self._address_detector = AddressDetector(_ad_cfg)
        self._voice_loop.set_address_detector(self._address_detector)

        self._proactive = ProactiveAgent(
            vision=self.vision,
            audio=self.audio,
            episodic=self.episodic,
            llm=self.llm,
            config=self.cfg,
        )
        # askme 只暴露本地 HTTP 健康/指标端点，不直接连接 OTA Server。
        # Terminal Agent (OTA Agent) 负责拉取此端点并统一上报给 OTA Server。
        self.health_server = AskmeHealthServer(
            self.cfg.get("health_server", {}),
            snapshot_provider=self.health_snapshot,
        )

        # Wire chat handler for /api/chat and /dashboard web UI.
        # Returns text immediately + fires TTS asynchronously (non-blocking).
        async def _chat_handler(text: str) -> str:
            import time as _t
            from askme.pipeline.brain_pipeline import strip_think_blocks

            self._pipeline._conversation.add_user_message(text)

            system_prompt = self._pipeline._build_system_prompt(
                None, user_text=text,
            )
            messages = self._pipeline._prepare_messages(
                self._pipeline._conversation.get_messages(system_prompt)
            )

            # Stream LLM — no tools, measure TTFT
            full = ""
            model = self._pipeline._voice_model
            t0 = _t.perf_counter()
            ttft_logged = False
            async for chunk in self._pipeline._llm.chat_stream(
                messages, model=model, tools=None, tool_choice=None,
            ):
                if not ttft_logged and chunk.choices[0].delta.content:
                    ttft_logged = True
                    logger.info("[WebChat] TTFT: %.2fs model=%s", _t.perf_counter() - t0, model)
                delta = chunk.choices[0].delta
                if delta.content:
                    full += delta.content

            full = strip_think_blocks(full).strip()
            total = _t.perf_counter() - t0
            logger.info("[WebChat] Total: %.2fs chars=%d", total, len(full))
            self._pipeline._conversation.add_assistant_message(full)
            self._pipeline._last_spoken_text = full

            # Fire-and-forget: speak on robot speaker, then stop playback
            if full:
                async def _speak_and_stop():
                    self.audio.speak(full)
                    self.audio.start_playback()
                    await asyncio.to_thread(self.audio.wait_speaking_done)
                    self.audio.stop_playback()
                asyncio.create_task(_speak_and_stop())
            return full

        self.health_server.set_chat_handler(_chat_handler)
        self.health_server.set_conversation_provider(
            lambda: list(self.conversation.history)
        )

        # ── LED state bridge ─────────────────────────────────────────────
        # Drives the status LED from AgentState + agent_task + ESTOP.
        # Configure dog-control-service URL to enable HTTP LED control:
        #   led:
        #     base_url: http://localhost:5080
        # Omit (or leave blank) to use NullLedController (silent no-op).
        led_cfg = self.cfg.get("led", {})
        led_base_url = led_cfg.get("base_url", "").strip()
        _led_ctrl = (
            HttpLedController(led_base_url)
            if led_base_url
            else NullLedController()
        )
        self._led_bridge = StateLedBridge(
            audio=self.audio,
            dispatcher=self.dispatcher,
            safety=getattr(self._pipeline, "_dog_safety", None),
            led=_led_ctrl,
        )
        logger.info(
            "[LED] controller=%s",
            "http(%s)" % led_base_url if led_base_url else "null",
        )

        # ── Config validation (warnings only — never crash) ─────────────
        cfg_warnings = validate_config(self.cfg)
        for w in cfg_warnings:
            logger.warning("[Config] %s", w)

        # ── Structured startup summary ───────────────────────────────────
        enabled_skills = self.skill_manager.get_enabled()
        skill_names = [s.name for s in enabled_skills]
        brain_cfg_log = self.cfg.get("brain", {})
        fallback_models = brain_cfg_log.get("fallback_models", [])
        dog_safety_url = (
            self.cfg.get("runtime", {}).get("dog_safety", {}).get("base_url", "")
        )
        logger.info(
            "Askme started | app=%s v%s | voice=%s robot=%s | "
            "llm=%s fallbacks=%s | skills(%d)=%s | tools=%d | "
            "voice_bridge=%s | dog_safety=%s",
            self._app_name,
            self._app_version,
            voice_mode,
            robot_mode,
            brain_cfg_log.get("model", "?"),
            fallback_models or [],
            len(skill_names),
            skill_names,
            len(self.tools),
            "enabled" if self.voice_runtime_bridge.enabled else "disabled",
            "configured" if dog_safety_url else "not configured",
        )
        from askme.runtime_health import log_startup_service_status
        log_startup_service_status()

    # ── Lifecycle ────────────────────────────────────────────

    async def run(self) -> None:
        """Start the appropriate main loop."""
        stop_event = asyncio.Event()
        warmup_task: asyncio.Task[None] | None = None
        proactive_task: asyncio.Task[None] | None = None
        led_task: asyncio.Task[None] | None = None
        try:
            await self.health_server.start()
            warmup_task = asyncio.create_task(self.memory.warmup())
            proactive_task = asyncio.create_task(self._proactive.run(stop_event))
            led_task = asyncio.create_task(self._led_bridge.run())
            if self.voice_mode:
                await self._voice_loop.run()
            else:
                await self._text_loop.run()
        finally:
            stop_event.set()
            pending_tasks = [
                task for task in (warmup_task, proactive_task, led_task)
                if task is not None
            ]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await self.health_server.stop()
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful cleanup — cancel in-flight tasks, drain audio, close hardware."""
        await self._pipeline.shutdown()
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
            with open(soul_file, encoding="utf-8") as f:
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
            {"role": "user", "content": "重要：你不是AI助手，不是Claude，不是开发工具。你就是Thunder巡检机器人。只用中文口语回答，绝不用英文，绝不用markdown。"},
            {"role": "assistant", "content": "明白。我是Thunder，穹沛的四足巡检机器人。中文口语回答，简洁汇报。"},
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
        fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        datefmt = "%H:%M:%S"
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

        # Also log to file so external tools can tail the output
        log_file = self.cfg.get("app", {}).get("log_file")
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logging.getLogger().addHandler(fh)

    def health_snapshot(self) -> dict[str, object]:
        """Return the compact HTTP health payload."""
        return build_health_snapshot(
            app_name=self._app_name,
            app_version=self._app_version,
            model_name=self.llm.model,
            metrics_snapshot=self.ota_metrics.snapshot(),
            active_skills=[skill.name for skill in self.skill_manager.get_enabled()],
            voice_status=self._voice_status_snapshot(),
            ota_status=None,  # OTA Agent pulls this endpoint and reports to OTA Server
            voice_bridge=self.voice_runtime_bridge.status_snapshot(),
        )

    def metrics_snapshot(self) -> dict[str, object]:
        """Return the latest runtime metrics snapshot."""
        return self.ota_metrics.snapshot()

    def _voice_status_snapshot(self) -> dict[str, object]:
        """Combine live voice readiness with recent OTA voice metrics."""
        live_status = self.audio.status_snapshot()
        return merge_voice_pipeline_status(
            live_status,
            self.ota_metrics.snapshot().get("voice_pipeline", {}),
        )
