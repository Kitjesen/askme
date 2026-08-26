"""Brain pipeline -facade over StreamProcessor + SkillGate + TurnExecutor.

Decomposed from the original 1093-line monolith (GAP-CORE-1).
Public API is unchanged: process(), execute_skill(), shutdown().

Decoupling improvements:
  - Constructor accepts StreamProcessorProtocol / SkillGateProtocol /
    TurnExecutorProtocol -tests pass mocks directly, no private-attr access.
  - cancel_token (asyncio.Event) is injected externally; handle_estop() calls
    cancel_token.set() and each sub-component stops autonomously.
"""

from __future__ import annotations

import asyncio
import logging
import re
from inspect import Parameter, getattr_static, signature
from typing import TYPE_CHECKING, Any

from askme.conversation import (
    ConversationLedgerError,
    InteractionInput,
    InteractionTurnContext,
    InteractionTurnManager,
    TurnOutcome,
    TurnStatus,
)
from askme.pipeline.core.hooks import PipelineHooks
from askme.pipeline.core.prompt_builder import PromptBuilder
from askme.pipeline.core.protocols import (
    CancellationToken,
    SkillExecutionDisposition,
    SkillGateProtocol,
    StreamProcessorProtocol,
    TurnExecutorProtocol,
)
from askme.pipeline.core.stream_processor import StreamProcessor
from askme.pipeline.core.tool_executor import ToolExecutor
from askme.pipeline.core.turn_control import TurnCancellationController
from askme.pipeline.core.turn_executor import TurnExecutor
from askme.pipeline.core.utils import (
    classify_skill_error,
    strip_think_blocks,  # noqa: F401 -re-exported for compat
)

if TYPE_CHECKING:
    from askme.agent_shell import AgentShell
    from askme.conversation import VoiceTurnLedger
    from askme.llm.core.client import LLMClient
    from askme.memory.core.conversation import ConversationManager
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.memory.core.session import SessionMemory
    from askme.memory.core.system import MemorySystem
    from askme.memory.retrieval.bridge import MemoryBridge
    from askme.ports import (
        ArmControlPort,
        AudioFrontendPort,
        RobotControlPort,
        SafetyPort,
        VisionPort,
    )
    from askme.skills.core.skill_executor import SkillExecutor
    from askme.skills.core.skill_manager import SkillManager
    from askme.tools.core.tool_registry import ToolRegistry
    from askme.voice.core.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


def _legacy_skill_execution_disposition(result: str) -> SkillExecutionDisposition:
    """Fail-closed settlement adapter for older injected skill gates."""

    text = str(result or "")
    stripped = text.lstrip()
    if not text:
        return SkillExecutionDisposition(
            status="failed",
            code="empty_skill_result",
        )
    if stripped.startswith(("[Timeout]", "[超时]")):
        return SkillExecutionDisposition(
            status="failed",
            code="execution_timeout",
        )
    if stripped.startswith(
        (
            "[Skill]",
            "[Skill Error]",
            "[AgentShell Error]",
            "[Error]",
            "[错误]",
            "[安全锁定]",
        )
    ):
        return SkillExecutionDisposition(
            status="failed",
            code="internal_error_result",
        )
    return SkillExecutionDisposition(status="succeeded", code="succeeded")


class _UnboundSkillGate:
    """Placeholder used until the runtime injects the concrete SkillGate."""

    def __init__(self, *, agent_shell: Any = None, max_response_chars: int = 500) -> None:
        self._agent_shell = agent_shell
        self._max_response_chars = max_response_chars
        self._last_spoken_text = ""
        self._skill_manager: Any = None
        self._skill_executor: Any = None
        self._audio: Any = None

    @property
    def last_spoken_text(self) -> str:
        return self._last_spoken_text

    def classify_execution_result(
        self,
        result: str,
        *,
        skill_name: str = "",
    ) -> SkillExecutionDisposition:
        del skill_name
        return _legacy_skill_execution_disposition(result)

    async def execute_skill(
        self,
        skill_name: str,
        user_text: str,
        extra_context: str = "",
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
    ) -> str:
        _ = (
            user_text,
            extra_context,
            source,
            conversation_session_id,
            voice_turn_id,
            turn_cancel_token,
        )
        logger.warning("Skill gate is not bound; cannot execute skill '%s'", skill_name)
        return f"[Skill] Skill gate is not configured: {skill_name}"

    def extract_semantic_target(self, user_text: str) -> str:
        text = str(user_text or "").strip()
        if not text:
            return text

        prefixes = ("导航到", "带我去", "前往", "走到", "去")
        filler_suffixes = ("好的", "可以", "那里", "这边", "一下", "嗯", "好", "吧", "啊")

        for prefix in prefixes:
            if not text.startswith(prefix):
                continue
            target = text[len(prefix) :].strip()
            target = re.split(r"[。！？?!，,；;]", target, maxsplit=1)[0].strip()
            changed = True
            while changed:
                changed = False
                for suffix in filler_suffixes:
                    if target.endswith(suffix) and len(target) > len(suffix):
                        target = target[: -len(suffix)].strip()
                        changed = True
                        break
            if target:
                return target
        return text

    def _classify_skill_error_message(self, exc: Exception, skill_name: str) -> str:
        return classify_skill_error(exc, skill_name)

    def _prepare_agent_result(self, result: str) -> tuple[str, str]:
        limit = self._max_response_chars or 200
        if len(result) <= limit:
            return result, result

        boundary = 0
        for ch in "。！？?":
            idx = result.rfind(ch, 0, limit)
            if idx > boundary:
                boundary = idx + 1
        if boundary == 0:
            boundary = limit

        spoken = result[:boundary].rstrip() + " 完整结果已保存到工作区。"
        workspace = getattr(self._agent_shell, "_workspace", None)
        if workspace:
            try:
                workspace.mkdir(parents=True, exist_ok=True)
                target = (workspace / "last_result.txt").resolve()
                if target.parent.resolve() == workspace.resolve():
                    target.write_text(result, encoding="utf-8")
            except OSError:
                logger.warning("Unable to persist unbound skill result", exc_info=True)
        return spoken, result

    def set_audio(self, audio: Any) -> None:
        self._audio = audio

    def set_skill_manager(self, manager: Any) -> None:
        self._skill_manager = manager

    def set_skill_executor(self, executor: Any) -> None:
        self._skill_executor = executor

    def set_agent_shell(self, shell: Any) -> None:
        self._agent_shell = shell


class BrainPipeline:
    """Orchestrates one turn of conversation.

    Delegates to three sub-components:
      - StreamProcessor: LLM streaming + think filter + TTS piping
      - SkillGate: skill execution + safety gates + legacy compat routing
      - TurnExecutor: full turn orchestration + memory + reflection
    """

    _DEFAULT_MAX_RESPONSE_CHARS = 500
    # Brand-neutral fallback used only when config did not provide persona or
    # system_prompt. Customer deployments should configure ``brain.persona``.
    _DEFAULT_SYSTEM_PROMPT = (
        "你是现场任务机器人语音助手，搭载在巡检机器人上。"
        "当前项目、客户名称和归属口径由部署配置决定，不要主动声明厂商归属。"
        "服务对象是现场运营人员、安保和交付工程师。"
        "说话简洁口语化，像对讲机里的值班员。"
        "短句为主，不超过80字。"
        "需要回复时，首句必须是10字以内的有效结论、动作状态或澄清问题，不含纯寒暄，"
        "并立即用句号、问号或叹号结束；第二句再展开；"
        "安全告警、拒绝和澄清不先寒暄。"
        "不用 markdown、emoji、英文。"
        "不确定时说不确定，需要确认，不编造信息。"
        "不要说自己是 AI 助手或语言模型。"
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
        audio: AudioFrontendPort,
        splitter: StreamSplitter,
        arm_controller: ArmControlPort | None = None,
        dog_safety_client: SafetyPort | None = None,
        dog_control_client: RobotControlPort | None = None,
        vision: VisionPort | None = None,
        session_memory: SessionMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        system_prompt: str = "",
        prompt_seed: list[dict[str, str]] | None = None,
        user_prefix: str = "",
        voice_model: str | None = None,
        voice_memory_retrieval_deadline_s: float | None = None,
        voice_llm_latency_budget_ms: int | None = None,
        general_tool_max_safety_level: str = "normal",
        max_response_chars: int = 0,
        voice_tts_coalesce: bool = False,
        agent_shell: AgentShell | None = None,
        memory_system: MemorySystem | None = None,
        qp_memory: Any = None,
        rag_policy_templates: dict[str, str] | None = None,
        relay_compat_mode: bool = False,
        # Decoupled sub-component injection (Protocol types)
        # Pass pre-built instances for testing or custom implementations.
        # When None (default) the components are constructed from the raw args above.
        cancel_token: asyncio.Event | None = None,
        stream_processor: StreamProcessorProtocol | None = None,
        skill_gate: SkillGateProtocol | None = None,
        turn_executor: TurnExecutorProtocol | None = None,
        # Durable product-level conversation lifecycle.  The legacy
        # ConversationManager remains the prompt-context projection while this
        # ledger owns Thread/Turn/Generation identity and settlement.
        turn_ledger: VoiceTurnLedger | None = None,
        # Temporary escape hatch for test/development deployments. When a
        # configured ledger fails, production defaults to fail-closed instead
        # of silently treating ConversationManager as canonical storage.
        conversation_core_legacy_fallback: bool = False,
        # Lifecycle hooks (Claude Code-style)
        # PipelineHooks provides pre/post callbacks for turns and tool calls.
        # If None, no hooks are fired. Build a PipelineHooks and register
        # callbacks via decorator syntax or direct list append.
        hooks: PipelineHooks | None = None,
    ) -> None:
        max_chars = (
            max_response_chars if max_response_chars > 0 else self._DEFAULT_MAX_RESPONSE_CHARS
        )

        # Apply default system prompt when none provided.
        if not system_prompt:
            system_prompt = self._DEFAULT_SYSTEM_PROMPT

        # Shared state
        self._tools = tools
        self._audio_ref = audio  # use dunder to avoid shadowing property
        self._conversation = conversation
        self._turn_ledger = turn_ledger
        self._interaction_turns = (
            InteractionTurnManager(turn_ledger) if turn_ledger is not None else None
        )
        self._conversation_core_legacy_fallback = bool(conversation_core_legacy_fallback)
        self._turn_ledger_failure_count = 0
        self._turn_ledger_last_error = ""
        self._thread_turn_locks: dict[str, asyncio.Lock] = {}
        self._thread_turn_lock_users: dict[str, int] = {}
        self._arm = arm_controller
        self._dog_safety = dog_safety_client

        # cancel_token -shared across all sub-components.
        # handle_estop() calls cancel_token.set(); each component stops autonomously.
        self._cancel_token: asyncio.Event = (
            cancel_token if cancel_token is not None else asyncio.Event()
        )
        self._turn_cancellations = TurnCancellationController()
        self._hooks = hooks

        if stream_processor is not None and skill_gate is not None and turn_executor is not None:
            # Injection path (tests / custom implementations)
            # All three protocol objects provided; skip internal construction.
            self._stream_processor: StreamProcessorProtocol = stream_processor
            self._skill_gate: SkillGateProtocol = skill_gate
            self._turn_executor: TurnExecutorProtocol = turn_executor
            # PromptBuilder not needed when sub-components are injected
            self._prompt_builder = None
            self._tool_executor = None
        else:
            # Default construction path
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
                rag_policy_templates=rag_policy_templates,
                relay_compat_mode=relay_compat_mode,
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
                    voice_tts_coalesce=voice_tts_coalesce,
                    voice_model=voice_model,
                    cancel_token=self._cancel_token,
                )
            )
            # Patch ToolExecutor callback to StreamProcessor
            self._tool_executor._stream_and_speak = self._stream_processor.stream_and_speak

            self._skill_gate = skill_gate or _UnboundSkillGate(
                agent_shell=agent_shell,
                max_response_chars=max_chars,
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
                    voice_memory_retrieval_deadline_s=voice_memory_retrieval_deadline_s,
                    voice_llm_latency_budget_ms=voice_llm_latency_budget_ms,
                    cancel_token=self._cancel_token,
                    hooks=hooks,
                )
            )

        # Store for direct access (backward compat)
        self._skill_manager = skill_manager
        self._skill_executor = skill_executor
        self._agent_shell = agent_shell
        self._skill_gate_context = {
            "audio": audio,
            "conversation": conversation,
            "dog_safety": dog_safety_client,
            "dog_control": dog_control_client,
            "arm_controller": arm_controller,
            "episodic": episodic_memory,
            "memory_system": memory_system,
            "agent_shell": agent_shell,
            "prompt_seed": prompt_seed,
            "max_response_chars": max_chars,
            "cancel_token": self._cancel_token,
            "hooks": hooks,
        }

    # Public API

    @property
    def last_spoken_text(self) -> str:
        return self._turn_executor.last_spoken_text or self._skill_gate.last_spoken_text

    @property
    def current_turn_rag(self) -> dict[str, Any] | None:
        return self._turn_executor.current_turn_rag

    @property
    def turn_ledger(self) -> VoiceTurnLedger | None:
        """Authoritative Conversation Core ledger, when runtime-wired."""

        return self._turn_ledger

    def conversation_core_health(self) -> dict[str, Any]:
        """Return non-sensitive audit-writer health for runtime diagnostics."""

        failures = int(self._turn_ledger_failure_count)
        return {
            "enabled": self._turn_ledger is not None,
            "status": "degraded" if failures else "ok",
            "write_failures": failures,
            "last_error_type": self._turn_ledger_last_error,
        }

    def _record_turn_ledger_failure(
        self,
        operation: str,
        exc: BaseException,
    ) -> None:
        self._turn_ledger_failure_count += 1
        self._turn_ledger_last_error = type(exc).__name__
        logger.exception("Conversation Core could not %s", operation)

    def _clear_legacy_projection_if_erased(
        self,
        turn_ledger: VoiceTurnLedger,
        ledger_turn: Any,
    ) -> None:
        """Prevent an in-flight legacy writer from reviving an erased thread."""

        try:
            thread = turn_ledger.get_thread(ledger_turn.thread_id)
        except Exception:
            return
        raw_status = getattr(thread, "status", None)
        status = str(getattr(raw_status, "value", raw_status) or "").lower()
        if status != "erased":
            return
        clear = getattr(self._conversation, "clear", None)
        if not callable(clear):
            logger.error(
                "Legacy conversation projection cannot clear erased thread %s",
                ledger_turn.thread_id,
            )
            return
        try:
            clear(
                conversation_session_id=ledger_turn.thread_id,
                durable=True,
            )
        except Exception:
            logger.exception(
                "Legacy conversation projection could not clear erased thread %s",
                ledger_turn.thread_id,
            )

    def _remove_legacy_turn_projection(
        self,
        *,
        conversation_session_id: str | None,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Remove one cancelled result from the legacy prompt cache, if present."""

        session_id = str(conversation_session_id or "").strip() or None
        remove_assistant = getattr(
            self._conversation,
            "remove_latest_assistant_message",
            None,
        )
        remove_user = getattr(self._conversation, "remove_latest_user_message", None)
        try:
            if assistant_text and callable(remove_assistant):
                remove_assistant(
                    assistant_text,
                    conversation_session_id=session_id,
                )
            if callable(remove_user):
                remove_user(
                    user_text,
                    conversation_session_id=session_id,
                )
        except Exception:
            logger.exception("Could not remove cancelled legacy turn projection")

    def clear_turn_context(self) -> None:
        self._turn_executor.clear_turn_context()

    def replace_llm(self, llm: Any, *, voice_model: str | None = None) -> None:
        """Route subsequent pipeline work to a replacement LLM gateway."""

        if hasattr(self._stream_processor, "_llm"):
            self._stream_processor._llm = llm
        if hasattr(self._turn_executor, "_llm"):
            self._turn_executor._llm = llm
        if voice_model is not None:
            if hasattr(self._stream_processor, "_voice_model"):
                self._stream_processor._voice_model = voice_model
            if hasattr(self._turn_executor, "_voice_model"):
                self._turn_executor._voice_model = voice_model

    def update_prompt(self, **settings: Any) -> dict[str, Any]:
        """Apply prompt settings to subsequent turns."""

        if self._prompt_builder is None:
            raise RuntimeError("prompt builder is not available")
        return self._prompt_builder.reconfigure(**settings)

    def prompt_settings(self) -> dict[str, Any]:
        if self._prompt_builder is None:
            return {}
        return self._prompt_builder.runtime_settings()

    async def process(
        self,
        user_text: str,
        *,
        memory_task: asyncio.Task[Any] | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        person_id: str | None = None,
        operator_id: str | None = None,
    ) -> str:
        """Serialize a full local turn per canonical Thread, while other Threads run."""

        thread_key = str(conversation_session_id or "").strip() or "__anonymous__"
        lock = self._thread_turn_locks.get(thread_key)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_turn_locks[thread_key] = lock
            self._thread_turn_lock_users[thread_key] = 0
        self._thread_turn_lock_users[thread_key] += 1
        try:
            async with lock:
                return await self._process_turn_unlocked(
                    user_text,
                    memory_task=memory_task,
                    source=source,
                    conversation_session_id=conversation_session_id,
                    voice_turn_id=voice_turn_id,
                    turn_cancel_token=turn_cancel_token,
                    person_id=person_id,
                    operator_id=operator_id,
                )
        finally:
            remaining = self._thread_turn_lock_users.get(thread_key, 1) - 1
            if remaining <= 0:
                self._thread_turn_lock_users.pop(thread_key, None)
                if self._thread_turn_locks.get(thread_key) is lock:
                    self._thread_turn_locks.pop(thread_key, None)
            else:
                self._thread_turn_lock_users[thread_key] = remaining

    async def _process_turn_unlocked(
        self,
        user_text: str,
        *,
        memory_task: asyncio.Task[Any] | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        person_id: str | None = None,
        operator_id: str | None = None,
    ) -> str:
        """Run the full brain pipeline. Returns assistant reply."""
        if source == "voice" and turn_cancel_token is not None and turn_cancel_token.is_set():
            logger.info("Voice turn was cancelled before pipeline ownership")
            return ""
        lease = (
            self._turn_cancellations.begin(
                voice_turn_id,
                cancel_event=turn_cancel_token,
            )
            if source == "voice"
            else None
        )
        ledger_turn: Any | None = None
        turn_ledger = self._turn_ledger
        if turn_ledger is not None:
            try:
                thread = turn_ledger.resolve_thread(
                    conversation_session_id=conversation_session_id,
                    # Thread channel is stable across text/cascade/realtime
                    # paths; the actual path belongs to Turn.source below.
                    channel="voice",
                    person_id=person_id,
                    operator_id=operator_id,
                )
                ledger_turn = turn_ledger.start_turn(
                    thread.thread_id,
                    turn_id=(lease.turn_id if lease is not None else voice_turn_id),
                    source=source,
                    user_text=user_text,
                )
                if ledger_turn.status in {
                    TurnStatus.COMMITTED,
                    TurnStatus.CANCELLED,
                    TurnStatus.FAILED,
                    TurnStatus.SUPPRESSED,
                }:
                    if lease is not None:
                        self._turn_cancellations.finish(lease)
                    return (
                        str(ledger_turn.assistant_text or "")
                        if ledger_turn.status is TurnStatus.COMMITTED
                        else ""
                    )
            except ConversationLedgerError:
                if lease is not None:
                    self._turn_cancellations.finish(lease)
                raise
            except Exception as exc:
                self._record_turn_ledger_failure("start the turn", exc)
                if not self._conversation_core_legacy_fallback:
                    if lease is not None:
                        self._turn_cancellations.finish(lease)
                    raise ConversationLedgerError(
                        "Conversation Core could not start the turn"
                    ) from exc
            if ledger_turn is not None:
                try:
                    start_generation = getattr(
                        turn_ledger,
                        "start_generation",
                        None,
                    )
                    if callable(start_generation):
                        start_generation(
                            ledger_turn.turn_id,
                            provider="askme_pipeline",
                            provider_generation_id=(
                                str(lease.epoch) if lease is not None else None
                            ),
                            metadata={"source": source},
                        )
                except ConversationLedgerError:
                    self._clear_legacy_projection_if_erased(turn_ledger, ledger_turn)
                    try:
                        turn_ledger.fail_turn(
                            ledger_turn.turn_id,
                            reason="generation_start_rejected",
                        )
                    except ConversationLedgerError:
                        pass
                    if lease is not None:
                        self._turn_cancellations.finish(lease)
                    raise
                except Exception as exc:
                    self._record_turn_ledger_failure(
                        "start the local generation",
                        exc,
                    )
                    if not self._conversation_core_legacy_fallback:
                        try:
                            turn_ledger.fail_turn(
                                ledger_turn.turn_id,
                                reason="generation_start_failed",
                            )
                        except ConversationLedgerError:
                            pass
                        except Exception as settlement_exc:
                            self._record_turn_ledger_failure(
                                "fail the turn after generation start",
                                settlement_exc,
                            )
                        if lease is not None:
                            self._turn_cancellations.finish(lease)
                        raise ConversationLedgerError(
                            "Conversation Core could not start the local generation"
                        ) from exc
        kwargs: dict[str, Any] = {
            "memory_task": memory_task,
            "source": source,
        }
        if conversation_session_id is not None:
            kwargs["conversation_session_id"] = conversation_session_id
        if ledger_turn is not None:
            kwargs["voice_turn_id"] = str(ledger_turn.turn_id)
        if lease is not None:
            kwargs.update(
                voice_turn_id=lease.turn_id,
                turn_epoch=lease.epoch,
                turn_cancel_token=lease,
            )
        try:
            result = await self._turn_executor.process(user_text, **kwargs)
        except asyncio.CancelledError:
            if ledger_turn is not None and turn_ledger is not None:
                try:
                    turn_ledger.cancel_turn(
                        ledger_turn.turn_id,
                        reason="task_cancelled",
                    )
                except ConversationLedgerError as exc:
                    self._clear_legacy_projection_if_erased(turn_ledger, ledger_turn)
                    logger.info("Conversation Core rejected cancellation: %s", exc)
                except Exception as exc:
                    self._record_turn_ledger_failure("cancel the turn", exc)
            self._remove_legacy_turn_projection(
                conversation_session_id=(
                    ledger_turn.thread_id if ledger_turn is not None else conversation_session_id
                ),
                user_text=user_text,
                assistant_text="",
            )
            raise
        except Exception as exc:
            if ledger_turn is not None and turn_ledger is not None:
                try:
                    turn_ledger.fail_turn(
                        ledger_turn.turn_id,
                        reason=type(exc).__name__,
                        metadata={"error": str(exc)},
                    )
                except ConversationLedgerError as ledger_exc:
                    self._clear_legacy_projection_if_erased(turn_ledger, ledger_turn)
                    logger.info("Conversation Core rejected failure settlement: %s", ledger_exc)
                except Exception as ledger_exc:
                    self._record_turn_ledger_failure("fail the turn", ledger_exc)
            self._remove_legacy_turn_projection(
                conversation_session_id=(
                    ledger_turn.thread_id if ledger_turn is not None else conversation_session_id
                ),
                user_text=user_text,
                assistant_text="",
            )
            raise
        else:
            # A non-empty TurnExecutor result means playback and its guarded
            # legacy settlement already completed.  In that case delivery
            # wins even if cancellation arrives just before this ledger write.
            cancellation_won = not result and lease is not None and lease.cancelled
            if ledger_turn is not None and turn_ledger is not None:
                try:
                    if result:
                        turn_ledger.commit_turn(
                            ledger_turn.turn_id,
                            user_text=user_text,
                            assistant_text=result,
                            heard_text=result,
                        )
                    elif cancellation_won:
                        turn_ledger.cancel_turn(
                            ledger_turn.turn_id,
                            reason=(self._turn_cancellations.last_cancel_reason or "cancelled"),
                        )
                    else:
                        turn_ledger.suppress_turn(
                            ledger_turn.turn_id,
                            reason="empty_response",
                        )
                except ConversationLedgerError:
                    self._clear_legacy_projection_if_erased(turn_ledger, ledger_turn)
                    self._remove_legacy_turn_projection(
                        conversation_session_id=ledger_turn.thread_id,
                        user_text=user_text,
                        assistant_text=str(result or ""),
                    )
                    raise
                except Exception as exc:
                    self._record_turn_ledger_failure("settle the turn", exc)
                    if not self._conversation_core_legacy_fallback:
                        self._remove_legacy_turn_projection(
                            conversation_session_id=ledger_turn.thread_id,
                            user_text=user_text,
                            assistant_text=str(result or ""),
                        )
                        raise ConversationLedgerError(
                            "Conversation Core could not settle the turn"
                        ) from exc
            if cancellation_won:
                self._remove_legacy_turn_projection(
                    conversation_session_id=(
                        ledger_turn.thread_id
                        if ledger_turn is not None
                        else conversation_session_id
                    ),
                    user_text=user_text,
                    assistant_text=str(result or ""),
                )
                return ""
            return result
        finally:
            if lease is not None:
                self._turn_cancellations.finish(lease)

    def cancel_active_turn(self, *, reason: str = "barge_in") -> bool:
        """Cancel the active answer without changing sticky safety state."""

        cancelled = self._turn_cancellations.cancel_active(reason=reason)
        if cancelled:
            logger.info("Active voice turn cancelled: %s", reason)
        return cancelled

    async def execute_skill(
        self,
        skill_name: str,
        user_text: str,
        extra_context: str = "",
        source: str = "voice",
        *,
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        record_turn: bool = True,
    ) -> str:
        """Execute one skill while Conversation Core owns the direct Turn."""

        metadata = {"skill_name": skill_name}
        interaction = (
            self._open_direct_interaction(
                user_text=user_text,
                source=source,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
                metadata=metadata,
            )
            if record_turn
            else None
        )
        if interaction is not None and self._is_cancelled(turn_cancel_token):
            self._settle_direct_interaction(
                interaction,
                TurnOutcome.cancel(reason="cancelled_before_skill_execution"),
            )
            return ""

        try:
            result = await self._call_skill_gate(
                skill_name,
                user_text,
                extra_context=extra_context,
                source=source,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
            )
        except asyncio.CancelledError:
            if interaction is not None:
                self._settle_direct_interaction(
                    interaction,
                    TurnOutcome.cancel(reason="skill_task_cancelled"),
                )
            raise
        except Exception as exc:
            if interaction is not None:
                self._settle_direct_interaction(
                    interaction,
                    TurnOutcome.fail(
                        reason=type(exc).__name__,
                        metadata=metadata,
                    ),
                )
            raise

        if interaction is None:
            return result
        if self._is_cancelled(turn_cancel_token):
            self._settle_direct_interaction(
                interaction,
                TurnOutcome.cancel(reason="skill_turn_cancelled"),
            )
            return ""

        disposition = self._classify_skill_execution_result(result, skill_name)
        if disposition.status == "succeeded" and result:
            self._settle_direct_interaction(
                interaction,
                TurnOutcome.commit(
                    assistant_text=result,
                    heard_text=result,
                    metadata=metadata,
                ),
            )
        elif disposition.status == "cancelled":
            self._settle_direct_interaction(
                interaction,
                TurnOutcome.cancel(reason=disposition.code),
            )
        else:
            self._settle_direct_interaction(
                interaction,
                TurnOutcome.fail(
                    reason=disposition.code if result else "empty_skill_result",
                    metadata=metadata,
                ),
            )
        return result

    async def record_direct_reply(
        self,
        user_text: str,
        assistant_text: str,
        *,
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        metadata: dict[str, Any] | None = None,
        person_id: str | None = None,
        operator_id: str | None = None,
    ) -> str:
        """Commit a deterministic direct reply without touching legacy history."""

        interaction = self._open_direct_interaction(
            user_text=user_text,
            source=source,
            conversation_session_id=conversation_session_id,
            voice_turn_id=voice_turn_id,
            turn_cancel_token=turn_cancel_token,
            metadata=metadata,
            person_id=person_id,
            operator_id=operator_id,
        )
        if interaction is not None:
            self._settle_direct_interaction(
                interaction,
                TurnOutcome.commit(
                    assistant_text=assistant_text,
                    heard_text=assistant_text,
                    metadata=metadata,
                ),
            )
        return assistant_text

    async def _call_skill_gate(
        self,
        skill_name: str,
        user_text: str,
        *,
        extra_context: str,
        source: str,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> str:
        callback = self._skill_gate.execute_skill
        context = {
            "conversation_session_id": conversation_session_id,
            "voice_turn_id": voice_turn_id,
            "turn_cancel_token": turn_cancel_token,
        }
        context = {name: value for name, value in context.items() if value is not None}
        try:
            parameters = signature(callback).parameters
            accepts_kwargs = any(
                parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            context = {}
        else:
            if not accepts_kwargs:
                context = {name: value for name, value in context.items() if name in parameters}
        return await callback(
            skill_name,
            user_text,
            extra_context,
            source,
            **context,
        )

    def _classify_skill_execution_result(
        self,
        result: str,
        skill_name: str,
    ) -> SkillExecutionDisposition:
        try:
            declared_classifier = getattr_static(
                self._skill_gate,
                "classify_execution_result",
            )
        except AttributeError:
            return _legacy_skill_execution_disposition(result)

        if not callable(declared_classifier):
            return _legacy_skill_execution_disposition(result)
        classifier = getattr(self._skill_gate, "classify_execution_result")
        try:
            disposition = classifier(result, skill_name=skill_name)
        except Exception:
            logger.exception(
                "Injected skill gate could not classify result for %s",
                skill_name,
            )
            return SkillExecutionDisposition(
                status="failed",
                code="skill_result_classification_failed",
            )

        if isinstance(disposition, SkillExecutionDisposition) and disposition.status in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return disposition

        logger.error(
            "Injected skill gate returned invalid disposition for %s: %s (%r)",
            skill_name,
            type(disposition).__name__,
            getattr(disposition, "status", None),
        )
        return SkillExecutionDisposition(
            status="failed",
            code="invalid_skill_execution_disposition",
        )

    def _open_direct_interaction(
        self,
        *,
        user_text: str,
        source: str,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
        metadata: dict[str, Any] | None,
        person_id: str | None = None,
        operator_id: str | None = None,
    ) -> InteractionTurnContext | None:
        manager = self._interaction_turns
        if manager is None:
            return None
        try:
            return manager.open(
                InteractionInput(
                    user_text=user_text,
                    source=source,
                    thread_id=conversation_session_id,
                    turn_id=voice_turn_id,
                    channel=source,
                    person_id=person_id,
                    operator_id=operator_id,
                    metadata=dict(metadata or {}),
                    cancel_token=turn_cancel_token,
                )
            )
        except ConversationLedgerError:
            raise
        except Exception as exc:
            self._record_turn_ledger_failure("start a direct interaction", exc)
            if self._conversation_core_legacy_fallback:
                return None
            raise ConversationLedgerError(
                "Conversation Core could not start the direct interaction"
            ) from exc

    def _settle_direct_interaction(
        self,
        interaction: InteractionTurnContext,
        outcome: TurnOutcome,
    ) -> None:
        manager = self._interaction_turns
        if manager is None:
            return
        try:
            manager.settle(interaction, outcome)
        except ConversationLedgerError:
            raise
        except Exception as exc:
            self._record_turn_ledger_failure("settle a direct interaction", exc)
            if not self._conversation_core_legacy_fallback:
                raise ConversationLedgerError(
                    "Conversation Core could not settle the direct interaction"
                ) from exc

    def _is_cancelled(self, token: CancellationToken | None) -> bool:
        return bool(self._cancel_token.is_set() or (token is not None and token.is_set()))

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        return self._turn_executor.start_idle_reflection(idle_seconds)

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[Any]:
        return self._turn_executor.start_memory_prefetch(user_text)

    async def shutdown(self) -> None:
        await self._turn_executor.shutdown()

    def handle_estop(self) -> None:
        """Trigger an emergency stop.

        Sets cancel_token so all sub-components (StreamProcessor, TurnExecutor,
        SkillGate) stop autonomously via their own cancel checks -no manual
        per-component coordination required.  Hardware stop and hooks follow.
        """
        logger.warning("E-STOP triggered!")
        # Signal all sub-components to stop -each checks cancel_token independently.
        self._cancel_token.set()
        self._turn_cancellations.cancel_active(reason="estop")
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
        cleared state immediately -no reference-swap needed.
        """
        if not self._cancel_token.is_set():
            logger.info("reset_estop: cancel_token was not set -no-op")
            return
        self._cancel_token.clear()
        logger.warning("E-STOP cleared -new turns will be accepted.")

    @staticmethod
    def _accepts_interaction_context(callback: Any) -> bool:
        try:
            parameters = signature(callback).parameters
        except (TypeError, ValueError):
            return False
        interaction_parameter = parameters.get("interaction_context")
        return (
            interaction_parameter is not None
            and interaction_parameter.kind
            in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        ) or any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())

    @staticmethod
    def _approval_identity_is_partial(
        *,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
    ) -> bool:
        """Return whether a caller supplied only half of canonical Turn identity."""

        has_thread = bool(str(conversation_session_id or "").strip())
        has_turn = bool(str(voice_turn_id or "").strip())
        return has_thread != has_turn

    @staticmethod
    def _approval_interaction_context(
        user_text: str,
        *,
        source: str,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
        operator_id: str | None,
    ) -> InteractionTurnContext | None:
        """Build a later-Turn policy context without manufacturing identity."""

        thread_id = str(conversation_session_id or "").strip()
        turn_id = str(voice_turn_id or "").strip()
        if not thread_id or not turn_id:
            return None
        channel = str(source or "voice").strip() or "voice"
        normalized_operator_id = str(operator_id or "").strip()
        return InteractionTurnContext(
            thread_id=thread_id,
            turn_id=turn_id,
            channel=channel,
            source=channel,
            user_text=user_text,
            operator_id=normalized_operator_id or None,
            cancel_token=turn_cancel_token,
        )

    def has_pending_tool_approval(
        self,
        *,
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        source: str = "voice",
        operator_id: str | None = None,
    ) -> bool:
        if self._approval_identity_is_partial(
            conversation_session_id=conversation_session_id,
            voice_turn_id=voice_turn_id,
        ):
            logger.warning(
                "Scoped tool approval probe rejected: canonical Turn identity is incomplete"
            )
            return False
        interaction_context = self._approval_interaction_context(
            "",
            source=source,
            conversation_session_id=conversation_session_id,
            voice_turn_id=voice_turn_id,
            turn_cancel_token=turn_cancel_token,
            operator_id=operator_id,
        )
        callback = self._tools.has_pending_approval
        if interaction_context is None:
            return bool(callback())
        if not self._accepts_interaction_context(callback):
            logger.warning(
                "Scoped tool approval probe rejected: registry does not accept interaction_context"
            )
            return False
        return bool(callback(interaction_context=interaction_context))

    async def handle_pending_tool_response(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        source: str = "voice",
        operator_id: str | None = None,
    ) -> str | None:
        """Resolve a scoped approval and record only its delivered response."""

        if self._approval_identity_is_partial(
            conversation_session_id=conversation_session_id,
            voice_turn_id=voice_turn_id,
        ):
            logger.warning(
                "Scoped tool approval response rejected: canonical Turn identity is incomplete"
            )
            return None
        tool_executor = self._tool_executor
        if tool_executor is None:
            raise RuntimeError("tool executor is not available")
        interaction_context = self._approval_interaction_context(
            user_text,
            source=source,
            conversation_session_id=conversation_session_id,
            voice_turn_id=voice_turn_id,
            turn_cancel_token=turn_cancel_token,
            operator_id=operator_id,
        )
        callback = tool_executor.handle_pending_tool_response
        kwargs: dict[str, Any] = {
            "audio": self._audio_ref,
            "source": source,
        }
        if interaction_context is not None:
            if not self._accepts_interaction_context(callback):
                logger.warning(
                    "Scoped tool approval response rejected: executor does not accept "
                    "interaction_context"
                )
                return None
            kwargs["interaction_context"] = interaction_context
        result = await callback(user_text, **kwargs)
        if result is None:
            return None
        if self._is_cancelled(turn_cancel_token):
            return ""
        if interaction_context is None or not result:
            return result

        metadata = {"interaction": "tool_approval_response"}
        canonical_interaction = self._open_direct_interaction(
            user_text=user_text,
            source=source,
            conversation_session_id=conversation_session_id,
            voice_turn_id=voice_turn_id,
            turn_cancel_token=turn_cancel_token,
            metadata=metadata,
            operator_id=operator_id,
        )
        if canonical_interaction is not None:
            self._settle_direct_interaction(
                canonical_interaction,
                TurnOutcome.commit(
                    assistant_text=result,
                    heard_text=result,
                    metadata=metadata,
                ),
            )
        return result

    async def _respond_without_llm(
        self, user_text: str, assistant_text: str, *, source: str = "voice"
    ) -> str:
        tool_executor = self._tool_executor
        if tool_executor is None:
            raise RuntimeError("tool executor is not available")
        return await tool_executor.respond_without_llm(
            user_text,
            assistant_text,
            audio=self._audio_ref,
            source=source,
        )

    # Late-binding setters

    def set_audio(self, audio: Any) -> None:
        self._audio_ref = audio
        self._skill_gate_context["audio"] = audio
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
        self._skill_gate_context["agent_shell"] = shell
        self._skill_gate.set_agent_shell(shell)

    def set_skill_gate(self, skill_gate: SkillGateProtocol) -> None:
        self._skill_gate = skill_gate
        if self._audio_ref is not None:
            self._skill_gate.set_audio(self._audio_ref)
        if self._skill_manager is not None:
            self._skill_gate.set_skill_manager(self._skill_manager)
        if self._skill_executor is not None:
            self._skill_gate.set_skill_executor(self._skill_executor)
        if self._agent_shell is not None:
            self._skill_gate.set_agent_shell(self._agent_shell)

    def skill_gate_context(self) -> dict[str, Any]:
        return dict(self._skill_gate_context)

    # Utilities (kept on facade for backward compat)

    def extract_semantic_target(self, user_text: str) -> str:
        return self._skill_gate.extract_semantic_target(user_text)

    def _classify_error_message(self, exc: Exception) -> str:
        return self._turn_executor._classify_error_message(exc)

    def _classify_skill_error_message(self, exc: Exception, skill_name: str) -> str:
        return self._skill_gate._classify_skill_error_message(exc, skill_name)

    def _prepare_agent_result(self, result: str) -> tuple[str, str]:
        return self._skill_gate._prepare_agent_result(result)

    # Backward compat properties (delegate private attr access to sub-components)

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

    def _build_l0_runtime_block(self) -> str:
        prompt_builder = self._prompt_builder
        if prompt_builder is None:
            raise RuntimeError("prompt builder is not available")
        return prompt_builder.build_l0_runtime_block()

    def _build_system_prompt(
        self,
        context_str: str | None,
        *,
        scene_desc: str = "",
        user_text: str = "",
    ) -> str:
        prompt_builder = self._prompt_builder
        if prompt_builder is None:
            raise RuntimeError("prompt builder is not available")
        return prompt_builder.build_system_prompt(
            context_str,
            scene_desc=scene_desc,
            user_text=user_text,
        )
