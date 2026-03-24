"""Composable runtime assembly for askme."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
from askme.llm.conversation import ConversationManager
from askme.memory.episodic_memory import EpisodicMemory
from askme.llm.intent_router import IntentRouter
from askme.llm.client import LLMClient
from askme.memory.bridge import MemoryBridge
from askme.memory.system import MemorySystem
from askme.memory.session import SessionMemory
from askme.perception.vision_bridge import VisionBridge
from askme.config import validate_config
from askme.robot.control_client import DogControlClient
from askme.robot.pulse import Pulse
from askme.robot.pubsub import PubSubBase
from askme.robot.safety_client import DogSafetyClient
from askme.health_server import AskmeHealthServer, build_health_snapshot
from askme.robot.led_controller import HttpLedController, NullLedController
from askme.robot.ota_bridge import OTABridgeMetrics
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.commands import CommandHandler
from askme.pipeline.planner_agent import PlannerAgent
from askme.pipeline.proactive_agent import ProactiveAgent
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.robot.state_led_bridge import StateLedBridge
from askme.pipeline.text_loop import TextLoop
from askme.pipeline.voice_loop import VoiceLoop
from askme.runtime.components import RuntimeComponent, resolve_start_order
from askme.runtime.profiles import RuntimeProfile
from askme.robot.runtime_health import log_startup_service_status, merge_voice_pipeline_status
from askme.skills.skill_executor import SkillExecutor
from askme.skills.skill_manager import SkillManager
from askme.tools.builtin_tools import DispatchSkillTool, SpeakProgressTool, register_builtin_tools
from askme.tools.move_tool import register_move_tools
from askme.tools.robot_api_tool import RobotApiTool
from askme.tools.scan_tool import register_scan_tools
from askme.tools.skill_tools import register_skill_tools
from askme.tools.tool_registry import ToolRegistry
from askme.tools.vision_tool import register_vision_tools
from askme.tools.voice_tools import register_voice_tools
from askme.voice.address_detector import AddressDetector
from askme.voice.audio_agent import AudioAgent
from askme.voice.audio_router import AudioRouter
from askme.voice.runtime_bridge import VoiceRuntimeBridge
from askme.voice.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_soul_seed(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Load SOUL.md and convert it into a prompt seed."""
    soul_file = cfg.get("brain", {}).get("soul_file", "SOUL.md")
    if not os.path.isabs(soul_file):
        soul_file = str(_project_root() / soul_file)

    if not os.path.isfile(soul_file):
        return []

    try:
        with open(soul_file, encoding="utf-8") as file:
            raw = file.read()
    except OSError:
        return []

    brief = re.sub(r"^#+\s+.*$", "", raw, flags=re.MULTILINE)
    brief = re.sub(r"\n{3,}", "\n\n", brief).strip()
    if not brief:
        return []

    logger.info("Loaded SOUL.md (%d chars) as character definition.", len(brief))
    return [
        {"role": "user", "content": f"请读取这份角色定义，并在整个会话中保持一致。\n{brief}"},
        {"role": "assistant", "content": "已加载 Thunder 角色定义，将按该设定持续响应。"},
    ]


def _init_qp_memory(llm: LLMClient) -> Any:
    """Initialise qp_memory if the shared package is available."""
    qp_memory = None
    project_root = _project_root()
    try:
        for try_path in [
            project_root.parent.parent / "shared" / "python" / "src",
            project_root,
        ]:
            path_text = str(try_path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)

        from qp_memory import Memory as QpMemory

        qp_memory = QpMemory(
            data_dir=str(project_root / "data" / "qp_memory"),
            site_id="default",
            robot_id="thunder_01",
        )
        topo_dir = project_root / "maps" / "semantic"
        if topo_dir.is_dir():
            synced = qp_memory.sync_map(str(topo_dir))
            logger.info("qp_memory: synced %d locations from LingTu map", synced)

        try:
            from askme.memory.extraction_adapter import ExtractionAdapter

            extractor = ExtractionAdapter(llm, model="qwen-turbo")
            qp_memory.set_extraction_callback(extractor)
            logger.info("qp_memory: LLM auto-extraction enabled")
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.debug("qp_memory: auto-extraction not available: %s", exc)
    except ImportError:
        logger.info("qp_memory not available (shared package not installed)")
    except Exception as exc:  # pragma: no cover - environment-specific failures
        logger.warning("qp_memory init failed: %s", exc)
    return qp_memory


def _init_robot(
    cfg: dict[str, Any],
    *,
    enabled: bool,
    tools: ToolRegistry,
) -> tuple[Any | None, bool]:
    """Initialise the optional robot arm controller."""
    if not enabled:
        return (None, False)
    try:
        from askme.robot.arm_controller import ArmController
        from askme.tools.robot_tools import register_robot_tools

        robot_cfg = cfg.get("robot", {})
        arm_controller = ArmController(robot_cfg)
        register_robot_tools(tools, arm_controller)
        logger.info("Robot arm controller initialised.")
        return (arm_controller, True)
    except Exception as exc:
        logger.warning("Failed to initialise robot: %s", exc)
        return (None, False)


@dataclass
class RuntimeServices:
    """Concrete service instances created for a runtime mode."""

    ota_metrics: OTABridgeMetrics
    llm: LLMClient
    session_memory: SessionMemory
    conversation: ConversationManager
    memory: MemoryBridge
    episodic: EpisodicMemory
    memory_system: MemorySystem
    qp_memory: Any
    vision: VisionBridge
    tools: ToolRegistry
    arm_controller: Any | None
    skill_manager: SkillManager
    skill_executor: SkillExecutor
    router: IntentRouter
    audio_router: AudioRouter | None
    audio: AudioAgent
    voice_runtime_bridge: VoiceRuntimeBridge
    pulse: PubSubBase
    dog_safety: DogSafetyClient
    dog_control: DogControlClient
    pipeline: BrainPipeline
    dispatcher: SkillDispatcher
    agent_shell: ThunderAgentShell
    commands: CommandHandler
    text_loop: TextLoop
    voice_loop: VoiceLoop | None
    address_detector: AddressDetector | None
    proactive: ProactiveAgent | None
    health_server: AskmeHealthServer | None
    led_bridge: StateLedBridge | None
    change_detector: Any | None = None

    def bindings(self) -> dict[str, Any]:
        """Expose compatibility aliases for legacy runtime."""
        return {
            "ota_metrics": self.ota_metrics,
            "llm": self.llm,
            "session_memory": self.session_memory,
            "conversation": self.conversation,
            "memory": self.memory,
            "episodic": self.episodic,
            "memory_system": self.memory_system,
            "qp_memory": self.qp_memory,
            "vision": self.vision,
            "tools": self.tools,
            "arm_controller": self.arm_controller,
            "skill_manager": self.skill_manager,
            "skill_executor": self.skill_executor,
            "router": self.router,
            "audio_router": self.audio_router,
            "audio": self.audio,
            "voice_runtime_bridge": self.voice_runtime_bridge,
            "pulse": self.pulse,
            "telemetry": self.pulse,
            "dog_safety": self.dog_safety,
            "dog_control": self.dog_control,
            "dispatcher": self.dispatcher,
            "agent_shell": self.agent_shell,
            "executor": self.agent_shell,
            "health_server": self.health_server,
            "diagnostics_server": self.health_server,
            "_pipeline": self.pipeline,
            "_commands": self.commands,
            "_text_loop": self.text_loop,
            "_voice_loop": self.voice_loop,
            "_address_detector": self.address_detector,
            "_proactive": self.proactive,
            "_supervisor": self.proactive,
            "_led_bridge": self.led_bridge,
            "_indicators": self.led_bridge,
            "_change_detector": self.change_detector,
            "_change_monitor": self.change_detector,
        }


@dataclass
class RuntimeAssembly:
    """Assembled runtime with mode metadata and component introspection."""

    cfg: dict[str, Any]
    app_name: str
    app_version: str
    profile: RuntimeProfile
    services: RuntimeServices
    voice_mode: bool
    robot_mode: bool
    components: dict[str, RuntimeComponent] = field(default_factory=dict)
    _background_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict, init=False)
    _stop_event: asyncio.Event | None = field(default=None, init=False)
    _start_order: list[str] = field(default_factory=list, init=False)
    _started: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def stop_event(self) -> asyncio.Event:
        if self._stop_event is None:
            self._stop_event = asyncio.Event()
        return self._stop_event

    def _task_running(self, name: str) -> bool:
        task = self._background_tasks.get(name)
        return task is not None and not task.done()

    async def _cancel_task(self, name: str) -> None:
        task = self._background_tasks.pop(name, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _start_task(self, name: str, coro: Any) -> None:
        task = self._background_tasks.get(name)
        if task is not None and not task.done():
            return
        self._background_tasks[name] = asyncio.create_task(coro, name=f"askme-{name}")

    async def start(self) -> None:
        """Start all registered runtime components in dependency order."""
        if self._started:
            return
        self._closed = False
        self._stop_event = asyncio.Event()
        self._start_order = resolve_start_order(self.components)
        for name in self._start_order:
            await self.components[name].start()
        self._started = True

    async def stop(self) -> None:
        """Stop background work and release resources."""
        if self._closed:
            return
        self._closed = True

        if self._stop_event is not None:
            self._stop_event.set()

        for name in reversed(self._start_order):
            await self.components[name].stop()

        if self._background_tasks:
            tasks = list(self._background_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._background_tasks.clear()

        await self.services.pipeline.shutdown()
        self.services.audio.shutdown()

        if self.services.arm_controller is not None:
            self.services.arm_controller.close()

        if self.services.qp_memory is not None:
            try:
                self.services.qp_memory.save()
                logger.info("qp_memory saved on shutdown")
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("qp_memory save failed on shutdown: %s", exc)

        self._started = False

    def voice_status_snapshot(self) -> dict[str, Any]:
        """Combine live audio status with recorded OTA metrics."""
        live_status = self.services.audio.status_snapshot()
        return merge_voice_pipeline_status(
            live_status,
            self.services.ota_metrics.snapshot().get("voice_pipeline", {}),
        )

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return the latest runtime metrics snapshot."""
        return self.services.ota_metrics.snapshot()

    def health_snapshot(self) -> dict[str, object]:
        """Return the compact HTTP health payload."""
        return build_health_snapshot(
            app_name=self.app_name,
            app_version=self.app_version,
            model_name=self.services.llm.model,
            metrics_snapshot=self.services.ota_metrics.snapshot(),
            active_skills=[skill.name for skill in self.services.skill_manager.get_enabled()],
            voice_status=self.voice_status_snapshot(),
            ota_status=None,
            voice_bridge=self.services.voice_runtime_bridge.status_snapshot(),
        )

    def capabilities_snapshot(self) -> dict[str, Any]:
        """Return runtime/profile/component capabilities."""
        contracts = self.services.skill_manager.get_contracts()
        openapi_document = self.services.skill_manager.openapi_document()
        return {
            "app": {
                "name": self.app_name,
                "version": self.app_version,
                "voice_mode": self.voice_mode,
                "robot_mode": self.robot_mode,
            },
            "profile": self.profile.snapshot(),
            "components": {
                name: component.snapshot()
                for name, component in self.components.items()
            },
            "skills": {
                "count": len(self.services.skill_manager.get_all()),
                "enabled_count": len(self.services.skill_manager.get_enabled()),
                "contract_count": len(contracts),
                "code_contract_count": sum(
                    1 for contract in contracts if contract.source == "code"
                ),
                "legacy_contract_count": sum(
                    1 for contract in contracts if contract.source != "code"
                ),
                "catalog": self.services.skill_manager.get_contract_catalog(),
            },
            "openapi": {
                "title": openapi_document["info"]["title"],
                "version": openapi_document["info"]["version"],
                "path_count": len(openapi_document["paths"]),
            },
        }


def _build_components(runtime: RuntimeAssembly) -> dict[str, RuntimeComponent]:
    """Legacy stub — planes have been removed. Returns empty dict."""
    return {}


def build_runtime(
    *,
    cfg: dict[str, Any],
    app_name: str,
    app_version: str,
    profile: RuntimeProfile,
    robot_requested: bool,
) -> RuntimeAssembly:
    """Build the runtime services used by the interactive app."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    ota_metrics = OTABridgeMetrics()
    llm = LLMClient(metrics=ota_metrics)
    session_memory = SessionMemory(llm=llm)
    conversation = ConversationManager(
        session_memory=session_memory,
        metrics=ota_metrics,
    )
    memory = MemoryBridge()
    episodic = EpisodicMemory(llm=llm)
    memory_system = MemorySystem(
        llm=llm,
        conversation=conversation,
        session_memory=session_memory,
        episodic=episodic,
        vector_memory=memory,
    )
    qp_memory = _init_qp_memory(llm)

    vision = VisionBridge()
    tools = ToolRegistry()
    register_builtin_tools(
        tools,
        production_mode=bool(cfg.get("tools", {}).get("production_mode", False)),
    )
    tools.register(RobotApiTool())
    register_vision_tools(tools, vision)
    register_move_tools(tools)
    register_scan_tools(tools, vision)

    robot_enabled = robot_requested or bool(cfg.get("robot", {}).get("enabled", False))
    arm_controller, robot_mode = _init_robot(
        cfg,
        enabled=robot_enabled,
        tools=tools,
    )

    skill_manager = SkillManager()
    skill_manager.load()

    brain_cfg = cfg.get("brain", {})
    skill_model = (
        brain_cfg.get("voice_model")
        if profile.has("operator_io")
        else brain_cfg.get("model")
    ) or brain_cfg.get("model", "claude-sonnet-4-5-20250929")
    skill_executor = SkillExecutor(
        llm,
        tools,
        default_model=skill_model,
        metrics=ota_metrics,
    )

    safety = arm_controller._safety if arm_controller else None  # noqa: SLF001
    router = IntentRouter(
        safety_checker=safety,
        voice_triggers=skill_manager.get_voice_triggers(),
    )

    audio_router = AudioRouter() if profile.has("operator_io") else None
    audio = AudioAgent(
        cfg,
        voice_mode=profile.has("operator_io"),
        metrics=ota_metrics,
        audio_router=audio_router,
    )
    register_voice_tools(tools, audio)
    tools.register(SpeakProgressTool(audio))

    voice_runtime_bridge = VoiceRuntimeBridge(
        cfg.get("runtime", {}).get("voice_bridge", {})
    )
    splitter = StreamSplitter()
    pulse = Pulse(cfg.get("pulse", {}))
    dog_safety = DogSafetyClient(
        cfg.get("runtime", {}).get("dog_safety", {}),
        pulse=pulse,
    )
    dog_control = DogControlClient(cfg.get("runtime", {}).get("dog_control", {}))

    prompt_seed = _load_soul_seed(cfg) or brain_cfg.get("prompt_seed", [])
    pipeline = BrainPipeline(
        llm=llm,
        conversation=conversation,
        memory=memory,
        tools=tools,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        splitter=splitter,
        arm_controller=arm_controller,
        dog_safety_client=dog_safety,
        dog_control_client=dog_control,
        vision=vision,
        session_memory=session_memory,
        episodic_memory=episodic,
        memory_system=memory_system,
        system_prompt=brain_cfg.get("system_prompt", "You are Thunder, an industrial inspection AI."),
        prompt_seed=prompt_seed,
        user_prefix=brain_cfg.get("user_prefix", ""),
        voice_model=brain_cfg.get("voice_model"),
        general_tool_max_safety_level=cfg.get("tools", {}).get(
            "general_chat_max_safety_level",
            "normal",
        ),
        max_response_chars=int(brain_cfg.get("max_response_chars", 0)),
        qp_memory=qp_memory,
    )

    planner = PlannerAgent(
        llm_client=pipeline._llm,
        skill_manager=skill_manager,
        model=brain_cfg.get("plan_model"),
    )
    dispatcher = SkillDispatcher(
        pipeline=pipeline,
        skill_manager=skill_manager,
        audio=audio,
        planner=planner,
    )

    agent_shell = ThunderAgentShell(
        llm_client=llm,
        tool_registry=tools,
        audio=audio,
        model=brain_cfg.get("agent_model"),
    )
    agent_shell._default_timeout = float(brain_cfg.get("agent_timeout", 120.0))
    pipeline._agent_shell = agent_shell

    dispatch_tool = DispatchSkillTool()
    dispatch_tool.set_dispatcher(dispatcher)
    tools.register(dispatch_tool)
    register_skill_tools(tools, skill_manager, router)

    commands = CommandHandler(
        conversation=conversation,
        skill_manager=skill_manager,
    )
    text_loop = TextLoop(
        router=router,
        pipeline=pipeline,
        commands=commands,
        conversation=conversation,
        skill_manager=skill_manager,
        audio=audio,
        voice_runtime_bridge=voice_runtime_bridge,
        dispatcher=dispatcher,
    )

    voice_loop = None
    address_detector = None
    if profile.primary_loop == "voice":
        voice_loop = VoiceLoop(
            router=router,
            pipeline=pipeline,
            audio=audio,
            voice_runtime_bridge=voice_runtime_bridge,
            dispatcher=dispatcher,
            audio_router=audio_router,
        )
        address_detector = AddressDetector(cfg.get("voice", {}).get("address_detection", {}))
        voice_loop.set_address_detector(address_detector)

    proactive = None
    if profile.has("supervisor"):
        proactive = ProactiveAgent(
            vision=vision,
            audio=audio,
            episodic=episodic,
            llm=llm,
            config=cfg,
        )
        proactive.set_solve_callback(
            lambda anomaly_text: pipeline.execute_skill("solve_problem", anomaly_text)
        )

    led_bridge = None
    if profile.has("indicators"):
        led_cfg = cfg.get("led", {})
        led_base_url = led_cfg.get("base_url", "").strip()
        led_controller = (
            HttpLedController(led_base_url)
            if led_base_url
            else NullLedController()
        )
        led_bridge = StateLedBridge(
            audio=audio,
            dispatcher=dispatcher,
            safety=getattr(pipeline, "_dog_safety", None),
            led=led_controller,
        )
        logger.info(
            "[LED] controller=%s",
            f"http({led_base_url})" if led_base_url else "null",
        )

    change_detector = None
    if profile.has("change_monitor"):
        from askme.perception.change_detector import ChangeDetector

        change_detector = ChangeDetector(config=cfg, pulse=pulse)

    services = RuntimeServices(
        ota_metrics=ota_metrics,
        llm=llm,
        session_memory=session_memory,
        conversation=conversation,
        memory=memory,
        episodic=episodic,
        memory_system=memory_system,
        qp_memory=qp_memory,
        vision=vision,
        tools=tools,
        arm_controller=arm_controller,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        router=router,
        audio_router=audio_router,
        audio=audio,
        voice_runtime_bridge=voice_runtime_bridge,
        pulse=pulse,
        dog_safety=dog_safety,
        dog_control=dog_control,
        pipeline=pipeline,
        dispatcher=dispatcher,
        agent_shell=agent_shell,
        commands=commands,
        text_loop=text_loop,
        voice_loop=voice_loop,
        address_detector=address_detector,
        proactive=proactive,
        health_server=None,
        led_bridge=led_bridge,
        change_detector=change_detector,
    )

    runtime = RuntimeAssembly(
        cfg=cfg,
        app_name=app_name,
        app_version=app_version,
        profile=profile,
        services=services,
        voice_mode=profile.primary_loop == "voice",
        robot_mode=robot_mode,
    )

    if profile.has("diagnostics"):
        health_server = AskmeHealthServer(
            cfg.get("health_server", {}),
            snapshot_provider=runtime.health_snapshot,
            metrics_provider=runtime.metrics_snapshot,
        )

        async def chat_handler(text: str) -> str:
            full = await services.text_loop.process_turn(text)
            if full and services.qp_memory is not None:
                qp_memory_ref = services.qp_memory
                pipeline_ref = services.pipeline

                async def qp_background() -> None:
                    try:
                        await asyncio.to_thread(qp_memory_ref.record_observation, "webchat", text)
                        await asyncio.to_thread(qp_memory_ref.process_turn, text, full)
                        pipeline_ref._qp_turn_count += 1
                        if pipeline_ref._qp_turn_count % 10 == 0:
                            await asyncio.to_thread(qp_memory_ref.save)
                    except Exception:
                        return

                asyncio.create_task(qp_background(), name="askme-qp-webchat")
            return full

        health_server.set_chat_handler(chat_handler)
        health_server.set_conversation_provider(lambda: list(services.conversation.history))
        health_server.set_vision_bridge(services.vision)
        health_server.set_capabilities_provider(runtime.capabilities_snapshot)

        from askme.perception.image_archive import ImageArchive

        health_server.set_image_archive(ImageArchive())
        services.health_server = health_server

    runtime.components = _build_components(runtime)

    for warning in validate_config(cfg):
        logger.warning("[Config] %s", warning)

    enabled_skills = [skill.name for skill in skill_manager.get_enabled()]
    logger.info(
        "Askme started | app=%s v%s | profile=%s | voice=%s robot=%s | llm=%s | "
        "skills(%d)=%s | tools=%d | voice_bridge=%s | dog_safety=%s",
        app_name,
        app_version,
        profile.name,
        runtime.voice_mode,
        runtime.robot_mode,
        brain_cfg.get("model", "?"),
        len(enabled_skills),
        enabled_skills,
        len(tools),
        "enabled" if voice_runtime_bridge.enabled else "disabled",
        "configured" if dog_safety.is_configured() else "not configured",
    )
    log_startup_service_status()
    return runtime


build_legacy_runtime = build_runtime
