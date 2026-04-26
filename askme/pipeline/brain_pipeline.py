"""Brain pipeline — facade over StreamProcessor + SkillGate + TurnExecutor.

Decomposed from the original 1093-line monolith (GAP-CORE-1).
Public API is unchanged: process(), execute_skill(), shutdown().

Decoupling improvements:
  - Constructor accepts StreamProcessorProtocol / SkillGateProtocol /
    TurnExecutorProtocol — tests pass mocks directly, no private-attr access.
  - cancel_token (asyncio.Event) is injected externally; handle_estop() calls
    cancel_token.set() and each sub-component stops autonomously.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from askme.pipeline.hooks import PipelineHooks
from askme.pipeline.prompt_builder import PromptBuilder
from askme.pipeline.protocols import (
    SkillGateProtocol,
    StreamProcessorProtocol,
    TurnExecutorProtocol,
)
from askme.pipeline.skill_gate import SkillGate
from askme.pipeline.stream_processor import StreamProcessor
from askme.pipeline.tool_executor import ToolExecutor
from askme.pipeline.turn_executor import TurnExecutor
from askme.pipeline.utils import strip_think_blocks  # noqa: F401 — re-exported for compat

if TYPE_CHECKING:
    from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
    from askme.llm.client import LLMClient
    from askme.llm.conversation import ConversationManager
    from askme.memory.bridge import MemoryBridge
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.session import SessionMemory
    from askme.memory.system import MemorySystem
    from askme.perception.vision_bridge import VisionBridge
    from askme.robot.arm_controller import ArmController
    from askme.robot.control_client import DogControlClient
    from askme.robot.safety_client import DogSafetyClient
    from askme.skills.skill_executor import SkillExecutor
    from askme.skills.skill_manager import SkillManager
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


class BrainPipeline:
    """Orchestrates one turn of conversation.

    Delegates to three sub-components:
      - StreamProcessor: LLM streaming + think filter + TTS piping
      - SkillGate: skill execution + safety gates + AgentShell routing
      - TurnExecutor: full turn orchestration + memory + reflection
    """

    _DEFAULT_MAX_RESPONSE_CHARS = 500
    # Default system prompt used when none is provided via config.yaml.
    # Override by setting ``brain.system_prompt`` in config.yaml or passing
    # ``system_prompt=`` to the constructor. The prompt is intentionally kept
    # in the source as a self-documenting default, not buried in a data file.
    _DEFAULT_SYSTEM_PROMPT = (
        "你是穹沛科技的机器人语音助手，搭载在巡检机器人上。"
        "操作员是工程师和现场人员。"
        "说话简洁口语化，像对讲机里的值班员。"
        "短句为主，不超过80字。"
        "不用markdown、emoji、英文。"
        "不确定时说'不确定，需要确认'，不编造信息。"
        "绝不说自己是AI助手或语言模型。"
    )

    def __init__(
        self,
        *,
        llm: LLMClient,
        conversation: ConversationManager,
        memory: MemoryBridge,
        tools: ToolRegistry,
        skill_manager: SkillManager,
        skill_executor: SkillExecutor,
        audio: AudioAgent,
        splitter: StreamSplitter,
        arm_controller: ArmController | None = None,
        dog_safety_client: DogSafetyClient | None = None,
        dog_control_client: DogControlClient | None = None,
        vision: VisionBridge | None = None,
        session_memory: SessionMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        system_prompt: str = "",
        prompt_seed: list[dict[str, str]] | None = None,
        user_prefix: str = "",
        voice_model: str | None = None,
        general_tool_max_safety_level: str = "normal",
        max_response_chars: int = 0,
        agent_shell: ThunderAgentShell | None = None,
        memory_system: MemorySystem | None = None,
        qp_memory: Any = None,
        # ── Decoupled sub-component injection (Protocol types) ────────────
        # Pass pre-built instances for testing or custom implementations.
        # When None (default) the components are constructed from the raw args above.
        cancel_token: asyncio.Event | None = None,
        stream_processor: StreamProcessorProtocol | None = None,
        skill_gate: SkillGateProtocol | None = None,
        turn_executor: TurnExecutorProtocol | None = None,
        # ── Lifecycle hooks (Claude Code-style) ───────────────────────────
        # PipelineHooks provides pre/post callbacks for turns and tool calls.
        # If None, no hooks are fired. Build a PipelineHooks and register
        # callbacks via decorator syntax or direct list append.
        hooks: PipelineHooks | None = None,
    ) -> None:
        max_chars = (
            max_response_chars if max_response_chars > 0
            else self._DEFAULT_MAX_RESPONSE_CHARS
        )

        # Apply default system prompt when none provided.
        if not system_prompt:
            system_prompt = self._DEFAULT_SYSTEM_PROMPT

        # Shared state
        self._tools = tools
        self._audio_ref = audio  # use dunder to avoid shadowing property
        self._conversation = conversation
        self._arm = arm_controller
        self._dog_safety = dog_safety_client

        # cancel_token — shared across all sub-components.
        # handle_estop() calls cancel_token.set(); each component stops autonomously.
        self._cancel_token: asyncio.Event = cancel_token if cancel_token is not None else asyncio.Event()
        self._hooks = hooks

        if stream_processor is not None and skill_gate is not None and turn_executor is not None:
            # ── Injection path (tests / custom implementations) ───────────
            # All three protocol objects provided; skip internal construction.
            self._stream_processor: StreamProcessorProtocol = stream_processor
            self._skill_gate: SkillGateProtocol = skill_gate
            self._turn_executor: TurnExecutorProtocol = turn_executor
            # PromptBuilder not needed when sub-components are injected
            self._prompt_builder = None
            self._tool_executor = None
        else:
            # ── Default construction path ─────────────────────────────────
            # PromptBuilder (already extracted)
            self._prompt_builder = PromptBuilder(
                base_prompt=system_prompt,
                prompt_seed=prompt_seed or [],
                user_prefix=user_prefix,
                tools=tools,
                skill_manager=skill_manager,
                general_tool_max_safety_level=general_tool_max_safety_level,
                dog_safety=dog_safety_client,
                episodic=episodic_memory,
                session_memory=session_memory,
                vision=vision,
                qp_memory=qp_memory,
                memory_system=memory_system,
            )

            # StreamProcessor (LLM streaming + TTS)
            self._tool_executor = ToolExecutor(
                tools=tools,
                conversation=conversation,
                episodic=episodic_memory,
                general_tool_max_safety_level=general_tool_max_safety_level,
                prompt_builder=self._prompt_builder,
                stream_and_speak=None,  # patched below
                hooks=hooks,
            )
            self._stream_processor = (
                stream_processor
                if stream_processor is not None
                else StreamProcessor(
                    llm=llm,
                    audio=audio,
                    tools=tools,
                    tool_executor=self._tool_executor,
                    splitter=splitter,
                    general_tool_max_safety_level=general_tool_max_safety_level,
                    max_response_chars=max_chars,
                    voice_model=voice_model,
                    cancel_token=self._cancel_token,
                )
            )
            # Patch ToolExecutor callback to StreamProcessor
            self._tool_executor._stream_and_speak = self._stream_processor.stream_and_speak

            # SkillGate (skill execution + safety)
            self._skill_gate = (
                skill_gate
                if skill_gate is not None
                else SkillGate(
                    skill_manager=skill_manager,
                    skill_executor=skill_executor,
                    audio=audio,
                    conversation=conversation,
                    dog_safety=dog_safety_client,
                    dog_control=dog_control_client,
                    arm_controller=arm_controller,
                    episodic=episodic_memory,
                    memory_system=memory_system,
                    agent_shell=agent_shell,
                    prompt_seed=prompt_seed,
                    max_response_chars=max_chars,
                    cancel_token=self._cancel_token,
                    hooks=hooks,
                )
            )

            # TurnExecutor (full turn orchestration)
            self._turn_executor = (
                turn_executor
                if turn_executor is not None
                else TurnExecutor(
                    llm=llm,
                    conversation=conversation,
                    memory=memory,
                    audio=audio,
                    prompt_builder=self._prompt_builder,
                    stream_processor=self._stream_processor,
                    dog_safety=dog_safety_client,
                    vision=vision,
                    episodic=episodic_memory,
                    memory_system=memory_system,
                    qp_memory=qp_memory,
                    voice_model=voice_model,
                    cancel_token=self._cancel_token,
                    hooks=hooks,
                )
            )

        # Store for direct access (backward compat)
        self._skill_manager = skill_manager
        self._skill_executor = skill_executor
        self._agent_shell = agent_shell

    # ── Public API ───────────────────────────────────────────

    @property
    def last_spoken_text(self) -> str:
        return self._turn_executor.last_spoken_text or self._skill_gate.last_spoken_text

    async def process(
        self, user_text: str, *, memory_task: asyncio.Task[str] | None = None,
        source: str = "voice",
    ) -> str:
        """Run the full brain pipeline. Returns assistant reply."""
        return await self._turn_executor.process(
            user_text, memory_task=memory_task, source=source,
        )

    async def execute_skill(
        self, skill_name: str, user_text: str, extra_context: str = "",
        source: str = "voice",
    ) -> str:
        """Execute a named skill and speak the result."""
        return await self._skill_gate.execute_skill(
            skill_name, user_text, extra_context, source,
        )

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        return self._turn_executor.start_idle_reflection(idle_seconds)

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[str]:
        return self._turn_executor.start_memory_prefetch(user_text)

    async def shutdown(self) -> None:
        await self._turn_executor.shutdown()

    def handle_estop(self) -> None:
        """Trigger an emergency stop.

        Sets cancel_token so all sub-components (StreamProcessor, TurnExecutor,
        SkillGate) stop autonomously via their own cancel checks — no manual
        per-component coordination required.  Hardware stop and hooks follow.
        """
        logger.warning("E-STOP triggered!")
        # Signal all sub-components to stop — each checks cancel_token independently.
        self._cancel_token.set()
        if self._arm:
            self._arm.emergency_stop()
        if self._dog_safety and self._dog_safety.is_configured():
            self._dog_safety.notify_estop()
            logger.warning("E-STOP: notified dog-safety-service")
        # Fire E-STOP hooks synchronously (Claude Code: Stop hook).
        if self._hooks:
            self._hooks.fire_estop()
        logger.warning("E-STOP: cancel_token set, local motion halted.")

    def reset_estop(self) -> None:
        """Clear the emergency-stop signal so new turns can be processed.

        Call this after confirming the robot is safe and the operator has
        explicitly released the E-STOP.  Hardware interlock release must happen
        separately (e.g. dog_safety.release_estop()).

        Uses asyncio.Event.clear() so all sub-components that hold the *same*
        event reference (TurnExecutor, SkillGate, StreamProcessor) see the
        cleared state immediately — no reference-swap needed.
        """
        if not self._cancel_token.is_set():
            logger.info("reset_estop: cancel_token was not set — no-op")
            return
        self._cancel_token.clear()
        logger.warning("E-STOP cleared — new turns will be accepted.")

    def has_pending_tool_approval(self) -> bool:
        return self._tools.has_pending_approval()

    async def handle_pending_tool_response(self, user_text: str) -> str | None:
        return await self._tool_executor.handle_pending_tool_response(
            user_text, audio=self._audio_ref,
        )

    async def _respond_without_llm(
        self, user_text: str, assistant_text: str, *, source: str = "voice"
    ) -> str:
        return await self._tool_executor.respond_without_llm(
            user_text, assistant_text, audio=self._audio_ref, source=source,
        )

    # ── Late-binding setters ─────────────────────────────────

    def set_audio(self, audio: Any) -> None:
        self._audio_ref = audio
        if hasattr(self, "_stream_processor"):
            self._stream_processor.set_audio(audio)
        if hasattr(self, "_skill_gate"):
            self._skill_gate.set_audio(audio)
        if hasattr(self, "_turn_executor"):
            self._turn_executor.set_audio(audio)

    def set_skill_manager(self, manager: Any) -> None:
        self._skill_manager = manager
        self._skill_gate.set_skill_manager(manager)

    def set_skill_executor(self, executor: Any) -> None:
        self._skill_executor = executor
        self._skill_gate.set_skill_executor(executor)

    def set_agent_shell(self, shell: Any) -> None:
        self._agent_shell = shell
        self._skill_gate.set_agent_shell(shell)

    # ── Utilities (kept on facade for backward compat) ───────

    def extract_semantic_target(self, user_text: str) -> str:
        return self._skill_gate.extract_semantic_target(user_text)

    def _classify_error_message(self, exc: Exception) -> str:
        return self._turn_executor._classify_error_message(exc)

    def _classify_skill_error_message(self, exc: Exception, skill_name: str) -> str:
        return self._skill_gate._classify_skill_error_message(exc, skill_name)

    def _prepare_agent_result(self, result: str) -> tuple[str, str]:
        return self._skill_gate._prepare_agent_result(result)

    # ── Backward compat properties (delegate private attr access to sub-components) ───

    @property
    def _episodic(self):
        return self._turn_executor._episodic

    @_episodic.setter
    def _episodic(self, value):
        if hasattr(self, "_turn_executor"):
            self._turn_executor._episodic = value
        if hasattr(self, "_skill_gate"):
            self._skill_gate._episodic = value

    @property
    def _memory(self):
        return self._turn_executor._memory

    @_memory.setter
    def _memory(self, value):
        if hasattr(self, "_turn_executor"):
            self._turn_executor._memory = value

    @property
    def _mem(self):
        return self._turn_executor._mem

    @_mem.setter
    def _mem(self, value):
        if hasattr(self, "_turn_executor"):
            self._turn_executor._mem = value
        if hasattr(self, "_skill_gate"):
            self._skill_gate._mem = value

    @property
    def _llm(self):
        return self._turn_executor._llm

    @property
    def _splitter(self):
        return self._stream_processor._splitter

    @_splitter.setter
    def _splitter(self, value):
        if hasattr(self, "_stream_processor"):
            self._stream_processor._splitter = value

    @property
    def _think_filter(self):
        return self._stream_processor._think_filter

    @property
    def _pending_tasks(self):
        if hasattr(self, "_turn_executor"):
            return self._turn_executor._pending_tasks
        return getattr(self, "__pending_tasks_fallback", set())

    @_pending_tasks.setter
    def _pending_tasks(self, value):
        if hasattr(self, "_turn_executor"):
            self._turn_executor._pending_tasks = value
        else:
            self.__pending_tasks_fallback = value

    @property
    def _llm_semaphore(self) -> asyncio.Semaphore:
        return self._turn_executor._llm_semaphore

    @_llm_semaphore.setter
    def _llm_semaphore(self, value: asyncio.Semaphore) -> None:
        if hasattr(self, "_turn_executor"):
            self._turn_executor._llm_semaphore = value

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._prompt_builder.prepare_messages(messages)

    def _build_l0_runtime_block(self) -> str:
        return self._prompt_builder.build_l0_runtime_block()

    def _build_system_prompt(
        self, context_str: str | None, *, scene_desc: str = "", user_text: str = "",
    ) -> str:
        return self._prompt_builder.build_system_prompt(
            context_str, scene_desc=scene_desc, user_text=user_text,
        )
