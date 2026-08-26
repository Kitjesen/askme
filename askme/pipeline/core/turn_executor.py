"""Turn executor; orchestrates one full conversation turn: memory, LLM, tools, TTS, save."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Coroutine
from contextvars import ContextVar
from copy import deepcopy
from inspect import isawaitable, iscoroutinefunction, signature
from typing import TYPE_CHECKING, Any

from askme.llm.core.contracts import LLMCallContext
from askme.pipeline.core.hooks import PipelineHooks
from askme.pipeline.core.protocols import CancellationToken, TurnContext
from askme.pipeline.core.trace import get_tracer
from askme.pipeline.core.utils import classify_llm_error, set_log_context

if TYPE_CHECKING:
    from askme.llm.core.client import LLMClient
    from askme.memory.core.conversation import ConversationManager
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.memory.core.system import MemorySystem
    from askme.memory.retrieval.bridge import MemoryBridge
    from askme.pipeline.core.prompt_builder import PromptBuilder
    from askme.pipeline.core.stream_processor import StreamProcessor
    from askme.ports import AudioFrontendPort, SafetyPort, VisionPort

logger = logging.getLogger(__name__)


class TurnExecutor:
    """Orchestrates one full conversation turn: memory, LLM, tools, TTS, save."""

    _SILENT_MARKER = "[SILENT]"
    _INTERNAL_PROTOCOL_MARKERS = ("DSML", "<TOOL_CALL", "TOOL_CALLS>")
    _INTERNAL_PROTOCOL_FALLBACK = "抱歉，刚才查询失败，请再说一遍。"
    _VISUAL_QUERY_MARKERS = (
        "看见",
        "看到",
        "看一下",
        "看看",
        "前面有什么",
        "周围有什么",
        "摄像头",
        "相机",
        "图像",
        "画面",
        "图片",
        "眼前有什么",
    )
    _REFLECT_DELAY_S = 5.0  # seconds to wait before post-turn reflection
    _BEHAVIOR_RETRIEVE_TIMEOUT_S = 0.45
    _VOICE_MEMORY_RETRIEVAL_DEADLINE_S = 0.25
    _KNOWLEDGE_DEPENDENT_MARKERS = (
        "在哪里",
        "在哪",
        "怎么走",
        "路线",
        "定位",
        "位置",
        "卫生间",
        "厕所",
        "设备",
        "SOP",
        "流程",
        "faq",
        "FAQ",
        "手册",
        "规程",
        "warehouse",
        "route",
        "location",
        "where",
    )

    def _track_task(
        self, coro: Coroutine[Any, Any, Any], *, name: str | None = None
    ) -> asyncio.Task[Any]:
        """Create a background task, track it in _pending_tasks, log any exception.

        Python 3.7+ asyncio.create_task() copies the *current* contextvars.Context
        into the new task at creation time (PEP 567).  Because set_log_context() is
        called before all _track_task() calls inside process(), background tasks
        automatically inherit the turn's trace_id and session_id (item 19).
        """
        # create_task() propagates the current context; no manual copy needed.
        t = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(t)

        def _done(task: asyncio.Task[Any]) -> None:
            self._pending_tasks.discard(task)
            if not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.warning("[TurnExecutor] Background task %r failed: %s", name or "?", exc)

        t.add_done_callback(_done)
        return t

    def __init__(
        self,
        *,
        llm: LLMClient,
        conversation: ConversationManager,
        memory: MemoryBridge,
        audio: AudioFrontendPort,
        prompt_builder: PromptBuilder,
        stream_processor: StreamProcessor,
        dog_safety: SafetyPort | None = None,
        vision: VisionPort | None = None,
        episodic: EpisodicMemory | None = None,
        memory_system: MemorySystem | None = None,
        qp_memory: Any = None,
        voice_model: str | None = None,
        voice_memory_retrieval_deadline_s: float | None = None,
        voice_llm_latency_budget_ms: int | None = None,
        cancel_token: CancellationToken | None = None,
        hooks: PipelineHooks | None = None,
    ) -> None:
        self._llm = llm
        self._conversation = conversation
        self._memory = memory
        self._audio = audio
        self._prompt_builder = prompt_builder
        self._stream_processor = stream_processor
        self._dog_safety = dog_safety
        self._vision = vision
        self._episodic = episodic
        self._mem = memory_system
        self._qp_memory = qp_memory
        self._voice_model = voice_model
        self._voice_memory_retrieval_deadline_s_config = voice_memory_retrieval_deadline_s
        self._voice_llm_latency_budget_ms = (
            int(voice_llm_latency_budget_ms)
            if voice_llm_latency_budget_ms is not None and int(voice_llm_latency_budget_ms) > 0
            else None
        )
        self._cancel_token = cancel_token
        self._hooks = hooks

        self._qp_turn_count = 0
        self._last_spoken_text: str = ""
        self._turn_rag_context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"askme_turn_rag_{id(self)}",
            default=None,
        )
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        # Semaphore initialized eagerly here so concurrent calls to process()
        # before the first turn completes don't each create their own semaphore.
        self._llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

    # Public API.

    @property
    def last_spoken_text(self) -> str:
        """The most recent text spoken via TTS. Used by repeat_last skill."""
        return self._last_spoken_text

    @property
    def current_turn_rag(self) -> dict[str, Any] | None:
        """Return RAG evidence for the current async turn, never another request."""

        payload = self._turn_rag_context.get()
        return deepcopy(payload) if isinstance(payload, dict) else None

    def clear_turn_context(self) -> None:
        """Clear task-local turn metadata before a non-LLM routing path."""
        self._turn_rag_context.set(None)

    def start_behavior_prefetch(self, user_text: str) -> asyncio.Task[str] | None:
        """Start optional personalization lookup outside customer RAG evidence."""

        if self._mem is None:
            return None
        retrieve = getattr(self._mem, "retrieve_behavior", None)
        if not iscoroutinefunction(retrieve):
            return None

        async def _bounded_retrieve() -> str:
            return await asyncio.wait_for(
                retrieve(user_text),
                timeout=self._BEHAVIOR_RETRIEVE_TIMEOUT_S,
            )

        return asyncio.create_task(_bounded_retrieve())

    @classmethod
    def _contains_internal_protocol(cls, text: str) -> bool:
        upper = str(text or "").upper()
        return any(marker in upper for marker in cls._INTERNAL_PROTOCOL_MARKERS)

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent (set by VoiceModule/TextModule after build)."""
        self._audio = audio

    def _start_playback_for_turn(self, voice_turn_id: str | None) -> Any:
        """Bind playback when the frontend exposes the optional owner fence."""

        start: Any = self._audio.start_playback
        try:
            parameters = signature(start).parameters.values()
        except (TypeError, ValueError):
            return start()
        supports_owner = any(
            parameter.name == "voice_turn_id"
            for parameter in parameters
        )
        if supports_owner and voice_turn_id:
            token = start(voice_turn_id=voice_turn_id)
            if token is None:
                raise RuntimeError("playback ownership admission failed")
            return token
        return start()

    def _stop_playback_for_owner(self, token: Any) -> None:
        """Release an explicit playback owner when the frontend supports it."""

        stop: Any = self._audio.stop_playback
        if token is None:
            stop()
            return
        try:
            supports_token = "token" in signature(stop).parameters
        except (TypeError, ValueError):
            supports_token = False
        if supports_token:
            stop(token)
        else:
            stop()

    # Core turn orchestration.

    def _log_episode(self, kind: str, text: str) -> None:
        if self._mem is not None:
            self._mem.log_event(kind, text)
        elif self._episodic:
            self._episodic.log(kind, text)

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        """Start an idle-time reflection background task (dream consolidation)."""
        _ep = self._mem.episodic if self._mem is not None else self._episodic
        if not _ep:
            return None

        async def _idle_reflect() -> None:
            await asyncio.sleep(idle_seconds)
            _should = (
                self._mem.should_reflect()
                if self._mem is not None
                else (_ep.should_reflect() if _ep else False)
            )
            if not _should:
                return
            if self._llm_semaphore.locked():
                logger.info("[Dream] Skipping reflection; LLM busy with user turn")
                return
            logger.info("[Dream] Idle-time reflection triggered")
            try:
                if self._mem is not None:
                    summary = await self._mem.reflect()
                else:
                    summary = await _ep.reflect()
                    _ep.cleanup_old_episodes()
                if summary:
                    logger.info("[Dream] Reflection result: %s", summary[:80])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[Dream] Reflection failed: %s", exc)

        return asyncio.create_task(_idle_reflect())

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[Any]:
        """Start memory retrieval as a background task. Call ASAP after ASR returns."""
        retrieve_with_context = getattr(self._memory, "retrieve_with_context", None)
        if iscoroutinefunction(retrieve_with_context):
            return asyncio.create_task(retrieve_with_context(user_text))
        return asyncio.create_task(self._memory.retrieve(user_text))

    async def process(
        self,
        user_text: str,
        *,
        memory_task: asyncio.Task[Any] | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_epoch: int | None = None,
        turn_cancel_token: CancellationToken | None = None,
    ) -> str:
        """Run the full brain pipeline for *user_text*. Returns assistant reply."""
        _turn_started_at = time.perf_counter()
        self._turn_rag_context.set(None)
        session_scope = str(conversation_session_id or "").strip() or None
        # Set structured log context for this turn so all log records carry
        # the trace ID and source without manual argument threading.
        _tracer = get_tracer()
        _trace = _tracer.current_trace
        _owns_trace = _trace is None
        if _owns_trace:
            _trace = _tracer.start_trace("voice_turn" if source == "voice" else "text_turn")
        log_session_id = f"{source}:{session_scope}" if session_scope else source
        set_log_context(trace_id=_trace.id, session_id=log_session_id)
        logger.info("Processing turn (chars=%d)", len(user_text))
        is_voice = source == "voice"

        def _turn_is_cancelled() -> bool:
            return bool(
                (self._cancel_token is not None and self._cancel_token.is_set())
                or (turn_cancel_token is not None and turn_cancel_token.is_set())
            )

        if _turn_is_cancelled():
            logger.warning("[TurnExecutor] cancel_token set; skipping turn")
            return ""

        # pre_turn hook (Claude Code: UserPromptSubmit).
        # Build a lightweight TurnContext snapshot for hooks; the token is
        # shared with sub-components so hooks can also trigger E-STOP.
        _token = (
            turn_cancel_token
            if turn_cancel_token is not None
            else self._cancel_token
            if self._cancel_token is not None
            else asyncio.Event()
        )
        _ctx = TurnContext(
            user_text=user_text,
            source=source,
            cancel_token=_token,
            voice_model=self._voice_model,
            conversation_session_id=session_scope,
            voice_turn_id=voice_turn_id,
            turn_epoch=turn_epoch,
        )

        user_message_staged = False
        assistant_settlement_started = False
        staged_episode_events: list[tuple[str, str]] = []

        def _stage_user_message() -> None:
            nonlocal user_message_staged
            self._add_user_message(
                user_text,
                conversation_session_id=session_scope,
            )
            user_message_staged = True

        def _rollback_staged_user_message() -> None:
            nonlocal user_message_staged
            if not user_message_staged:
                return
            self._remove_latest_user_message(
                user_text,
                conversation_session_id=session_scope,
            )
            user_message_staged = False

        def _stage_episode_event(kind: str, text: str) -> None:
            staged_episode_events.append((kind, text))

        def _commit_staged_episode_events() -> None:
            for kind, text in staged_episode_events:
                self._log_episode(kind, text)
            staged_episode_events.clear()

        def _settle_assistant(
            content: str,
            *,
            evidence: list[dict[str, Any]] | None = None,
            rag: dict[str, Any] | None = None,
        ) -> bool:
            """Commit assistant history at one cancellation linearization point."""

            if _turn_is_cancelled():
                return False

            committed = False

            def _commit() -> None:
                nonlocal assistant_settlement_started, committed
                assistant_settlement_started = True
                self._add_assistant_message(
                    content,
                    conversation_session_id=session_scope,
                    evidence=evidence,
                    rag=rag,
                )
                committed = True

            try_run = getattr(_token, "try_run", None)
            if callable(try_run):
                accepted, _ = try_run(_commit)
                if not accepted:
                    return False
            else:
                _commit()

            # Plain Event-like tokens cannot linearize set() against _commit().
            # Roll back synchronously if cancellation landed during that seam.
            if _turn_is_cancelled():
                if committed:
                    self._remove_latest_assistant_message(
                        content,
                        conversation_session_id=session_scope,
                    )
                return False
            return committed

        if self._hooks:
            skip = await self._hooks.fire_pre_turn(_ctx)
            if skip:
                logger.info("[TurnExecutor] pre_turn hook requested turn skip")
                return ""

        if is_voice:
            prepare_turn = getattr(self._audio, "prepare_turn", None)
            if callable(prepare_turn):
                prepare_turn()
            else:
                self._audio.drain_buffers()

        if self._dog_safety and self._dog_safety.is_configured():
            self._track_task(
                asyncio.to_thread(self._dog_safety.query_estop_state),
                name="estop_refresh",
            )

        if not memory_task:
            memory_task = self.start_memory_prefetch(user_text)
        behavior_task = self.start_behavior_prefetch(user_text)
        vision_task = self._start_vision_capture(user_text)

        try:
            with _tracer.span("memory_retrieve"):
                context_str, turn_rag = await self._resolve_memory_context(
                    memory_task,
                    user_text=user_text,
                    is_voice=is_voice,
                )
            if turn_rag is not None:
                self._turn_rag_context.set(turn_rag)
        except Exception as _me:
            logger.warning("[TurnExecutor] Memory retrieve failed: %s", _me)
            context_str = ""
            turn_rag = self._unavailable_memory_context(_me)
            self._turn_rag_context.set(turn_rag)

        if _turn_is_cancelled():
            if behavior_task is not None and not behavior_task.done():
                behavior_task.cancel()
            if vision_task is not None and not vision_task.done():
                vision_task.cancel()
            if is_voice:
                self._audio.drain_buffers()
            return ""

        behavior_context = ""
        if behavior_task is not None:
            try:
                behavior_context = str(await behavior_task or "").strip()
            except TimeoutError:
                logger.debug("[TurnExecutor] Behavior memory retrieval timed out")
            except Exception as exc:
                logger.debug("[TurnExecutor] Behavior memory unavailable: %s", exc)

        scene_desc = ""
        if vision_task:
            try:
                scene_desc = await vision_task
                if not scene_desc:
                    logger.debug("[TurnExecutor] Vision capture returned empty scene description")
            except Exception as _ve:
                logger.warning("[TurnExecutor] Vision capture failed: %s", _ve)
                scene_desc = ""

        if scene_desc:
            _stage_episode_event("perception", scene_desc)

        if _turn_is_cancelled():
            if is_voice:
                self._audio.drain_buffers()
            return ""

        # A visual question is already answered by the configured VLM.  Do not
        # send that evidence through the ordinary chat model again: it can
        # contradict the image result (for example, changing "桌子" to
        # "没有可识别物体") and adds another 10-20 seconds of latency.
        if self._is_visual_query(user_text):
            visual_reply = (
                scene_desc.strip()
                if scene_desc and scene_desc.strip()
                else "我暂时无法读取当前摄像头画面。"
            )
            _stage_user_message()
            if is_voice:
                visual_playback_token = self._start_playback_for_turn(voice_turn_id)
                self._audio.speak(visual_reply)
                await asyncio.to_thread(self._audio.wait_speaking_done)
                # This branch returns before the normal process finally block;
                # explicitly release TTS state so VAD is not gated forever.
                self._stop_playback_for_owner(visual_playback_token)
                self._audio.drain_buffers()
                if _turn_is_cancelled():
                    _rollback_staged_user_message()
                    return ""
            if not _settle_assistant(visual_reply):
                if not assistant_settlement_started:
                    _rollback_staged_user_message()
                if is_voice:
                    self._audio.drain_buffers()
                return ""
            self._last_spoken_text = visual_reply
            _commit_staged_episode_events()
            self._log_episode("action", f"视觉回复: {visual_reply[:100]}")
            return visual_reply

        rag_policy = self._turn_answer_policy(turn_rag)
        if rag_policy is None:
            rag_policy = await self._memory_answer_policy()
        system_prompt = self._prompt_builder.build_system_prompt(
            context_str,
            behavior_context=behavior_context,
            scene_desc=scene_desc,
            user_text=user_text,
            rag_policy=rag_policy,
        )

        _stage_user_message()
        forced_rag_reply = self._prompt_builder.build_forced_rag_reply(rag_policy)
        if forced_rag_reply:
            logger.info("[TurnExecutor] RAG policy forced deterministic reply")
            if is_voice:
                forced_playback_token = self._start_playback_for_turn(voice_turn_id)
                self._audio.speak(forced_rag_reply)
                await asyncio.to_thread(self._audio.wait_speaking_done)
                self._stop_playback_for_owner(forced_playback_token)
                if _turn_is_cancelled():
                    _rollback_staged_user_message()
                    self._audio.drain_buffers()
                    return ""
            if not _settle_assistant(
                forced_rag_reply,
                **self._assistant_rag_metadata(),
            ):
                if not assistant_settlement_started:
                    _rollback_staged_user_message()
                if is_voice:
                    self._audio.drain_buffers()
                return ""
            _commit_staged_episode_events()
            self._last_spoken_text = forced_rag_reply
            if self._hooks:
                await self._hooks.fire_post_turn(_ctx, forced_rag_reply)
            self._log_episode("action", f"回复: {forced_rag_reply[:100]}")
            return forced_rag_reply

        # Start compress AFTER add_user_message so the new user message is always
        # included in maybe_compress's recent[-KEEP_RECENT:] snapshot and never lost.
        async def _compress_bg() -> None:
            try:
                if session_scope is None:
                    await self._conversation.maybe_compress(self._llm)
                else:
                    await self._conversation.maybe_compress(
                        self._llm,
                        conversation_session_id=session_scope,
                    )
            except Exception as _e:
                logger.warning("Conversation compression failed (non-critical): %s", _e)

        self._track_task(_compress_bg(), name="conv_compress")
        messages = self._prompt_builder.prepare_messages(
            self._get_messages(system_prompt, conversation_session_id=session_scope),
            source=source,
        )

        _stage_episode_event("command", f"用户说: {user_text}")

        playback_token: Any = None
        if is_voice:
            playback_token = self._start_playback_for_turn(voice_turn_id)
        llm_call_context = LLMCallContext(
            trace_id=_trace.id,
            session_id=session_scope,
            turn_id=str(voice_turn_id or _trace.id),
            call_id=self._build_llm_call_id(_trace.id, str(voice_turn_id or _trace.id)),
            purpose="assistant_response",
            channel=source,
            request_class="voice_fast" if is_voice else "text",
            latency_budget_ms=(
                max(
                    1,
                    int(self._voice_llm_latency_budget_ms)
                    - int((time.perf_counter() - _turn_started_at) * 1000),
                )
                if is_voice and self._voice_llm_latency_budget_ms is not None
                else None
            ),
            privacy_class="conversation",
            allow_cache=False,
        )
        try:
            async with self._llm_semaphore:
                full_response = await self._stream_processor.stream_with_tools(
                    messages,
                    system_prompt,
                    model=self._voice_model,
                    source=source,
                    conversation_session_id=session_scope,
                    turn_cancel_token=_token,
                    llm_call_context=llm_call_context,
                )
            if _turn_is_cancelled():
                logger.info(
                    "[TurnExecutor] interrupted turn dropped before settlement: %s",
                    voice_turn_id or "legacy",
                )
                if is_voice:
                    self._audio.drain_buffers()
                _rollback_staged_user_message()
                return ""
            if self._contains_internal_protocol(full_response):
                logger.error("Blocked internal tool protocol from assistant response")
                full_response = self._INTERNAL_PROTOCOL_FALLBACK
                if is_voice:
                    self._audio.drain_buffers()
                    self._audio.speak(full_response)

            if full_response.lstrip().startswith(self._SILENT_MARKER):
                logger.info("[SILENT] Not addressed to robot, suppressing output")
                self._audio.drain_buffers()
                # Remove exactly the user message we added; match by content to
                # avoid popping the wrong message if compress ran concurrently.
                _rollback_staged_user_message()
                return ""

            if is_voice:
                await asyncio.to_thread(self._audio.wait_speaking_done)
                if _turn_is_cancelled():
                    logger.info(
                        "[TurnExecutor] interrupted playback omitted from history: %s",
                        voice_turn_id or "legacy",
                    )
                    self._audio.drain_buffers()
                    _rollback_staged_user_message()
                    return ""

            if not _settle_assistant(
                full_response,
                **self._assistant_rag_metadata(),
            ):
                if not assistant_settlement_started:
                    _rollback_staged_user_message()
                if is_voice:
                    self._audio.drain_buffers()
                return ""
            _commit_staged_episode_events()
            self._last_spoken_text = full_response

            # post_turn hook (Claude Code: Stop hook / notification).
            if self._hooks:
                await self._hooks.fire_post_turn(_ctx, full_response)

            self._log_episode("action", f"回复: {full_response[:100]}")

            _should = (
                self._mem.should_reflect()
                if self._mem is not None
                else (self._episodic.should_reflect() if self._episodic else False)
            )
            if _should:

                async def _delayed_reflect() -> None:
                    await asyncio.sleep(self._REFLECT_DELAY_S)
                    if self._mem is not None:
                        await self._mem.reflect()
                    elif self._episodic and self._episodic.should_reflect():
                        try:
                            await self._episodic.reflect()
                        except Exception as e:
                            logger.error("[Episodic] Reflection failed: %s", e)

                self._track_task(_delayed_reflect(), name="delayed_reflect")

            return full_response
        except Exception as exc:
            if _turn_is_cancelled():
                if is_voice:
                    self._audio.drain_buffers()
                _rollback_staged_user_message()
                return ""
            logger.error("LLM pipeline error: %s", exc)
            _stage_episode_event("error", f"LLM错误: {exc}")
            if is_voice:
                self._audio.speak(classify_llm_error(exc))
            error_msg = f"[系统错误] {type(exc).__name__}"
            if self._latest_history_role(conversation_session_id=session_scope) == "assistant":
                self._add_user_message(
                    "[系统错误恢复]",
                    conversation_session_id=session_scope,
                )
            if not _settle_assistant(error_msg):
                if not assistant_settlement_started:
                    _rollback_staged_user_message()
                if is_voice:
                    self._audio.drain_buffers()
                return ""
            _commit_staged_episode_events()
            return error_msg
        finally:
            if is_voice:
                self._stop_playback_for_owner(playback_token)
            if _owns_trace:
                _tracer.finish_trace()

    @staticmethod
    def _build_llm_call_id(trace_id: str, turn_id: str) -> str:
        base = f"{trace_id}:{turn_id}".encode()
        return f"sha256:{hashlib.sha256(base).hexdigest()[:24]}"

    async def shutdown(self) -> None:
        """Cancel all in-flight background tasks (delayed reflections, etc.)."""
        tasks = list(self._pending_tasks)
        if tasks:
            logger.info("BrainPipeline shutdown: cancelling %d pending tasks", len(tasks))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()

    # Internal helpers.

    @classmethod
    def _is_visual_query(cls, text: str) -> bool:
        normalized = str(text or "").strip()
        return any(marker in normalized for marker in cls._VISUAL_QUERY_MARKERS)

    def _start_vision_capture(self, user_text: str = "") -> asyncio.Task[str] | None:
        if not self._vision or not self._vision.available:
            return None
        auto_capture_enabled = getattr(self._vision, "auto_capture_enabled", None)
        auto_capture = bool(callable(auto_capture_enabled) and auto_capture_enabled())
        visual_query = self._is_visual_query(user_text)
        if not auto_capture and not visual_query:
            return None
        targeted = getattr(self._vision, "describe_scene_with_question", None)
        if visual_query and callable(targeted):
            return asyncio.create_task(targeted(user_text))
        return asyncio.create_task(self._vision.describe_scene())

    def _voice_memory_retrieval_deadline_s(self) -> float:
        value = self._voice_memory_retrieval_deadline_s_config
        if value is None:
            value = getattr(
                self._memory,
                "voice_retrieval_deadline_s",
                self._VOICE_MEMORY_RETRIEVAL_DEADLINE_S,
            )
        if isinstance(value, bool):
            return self._VOICE_MEMORY_RETRIEVAL_DEADLINE_S
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return self._VOICE_MEMORY_RETRIEVAL_DEADLINE_S
        if parsed != parsed or parsed in {float("inf"), float("-inf")}:
            return self._VOICE_MEMORY_RETRIEVAL_DEADLINE_S
        return max(0.0, parsed)

    @classmethod
    def _is_knowledge_dependent_query(cls, text: str) -> bool:
        normalized = str(text or "").strip()
        folded = normalized.lower()
        return any(marker.lower() in folded for marker in cls._KNOWLEDGE_DEPENDENT_MARKERS)

    async def _resolve_memory_context(
        self,
        memory_task: asyncio.Task[Any],
        *,
        user_text: str,
        is_voice: bool,
    ) -> tuple[str, dict[str, Any] | None]:
        if not is_voice:
            retrieval = await memory_task
            return self._coerce_memory_retrieval(retrieval)

        deadline_s = self._voice_memory_retrieval_deadline_s()
        done, pending = await asyncio.wait({memory_task}, timeout=deadline_s)
        if memory_task in done:
            retrieval = memory_task.result()
            return self._coerce_memory_retrieval(retrieval)

        for pending_task in pending:
            self._cancel_and_consume_memory_task(pending_task)
        turn_rag = self._latency_budget_memory_context(
            user_text,
            deadline_s=deadline_s,
        )
        return "", turn_rag

    @staticmethod
    def _cancel_and_consume_memory_task(memory_task: asyncio.Task[Any]) -> None:
        if not memory_task.done():
            memory_task.cancel()

        def _consume(task: asyncio.Task[Any]) -> None:
            if task.cancelled():
                return
            try:
                task.exception()
            except (asyncio.CancelledError, Exception):
                return

        memory_task.add_done_callback(_consume)

    def _latency_budget_memory_context(
        self,
        user_text: str,
        *,
        deadline_s: float,
    ) -> dict[str, Any]:
        knowledge_required = self._is_knowledge_dependent_query(user_text)
        if knowledge_required:
            answer_policy = {
                "state": "unavailable",
                "action": "refuse",
                "reason": "memory_retrieval_deadline_exceeded",
                "deadline_s": deadline_s,
            }
        else:
            answer_policy = {
                "state": "latency_budget_exhausted",
                "action": "answer_without_memory",
                "reason": "memory_retrieval_deadline_exceeded",
                "deadline_s": deadline_s,
            }
        return {
            "evidence": [],
            "rag": {
                "turn_scoped": True,
                "enabled": True,
                "fallback_reason": "latency_budget_exhausted",
                "dropped_evidence": [],
                "used_in_answer": False,
                "retrieval_deadline_s": deadline_s,
                "answer_policy": answer_policy,
            },
        }

    async def _memory_answer_policy(self) -> dict[str, Any] | None:
        health = getattr(self._memory, "health", None)
        if not callable(health):
            return None
        try:
            snapshot = health()
            if isawaitable(snapshot):
                snapshot = await snapshot
        except Exception as exc:
            logger.debug("[TurnExecutor] Memory health unavailable for RAG policy: %s", exc)
            return None
        if not isinstance(snapshot, dict):
            return None
        policy = snapshot.get("last_answer_policy")
        return policy if isinstance(policy, dict) else None

    @staticmethod
    def _coerce_memory_retrieval(
        retrieval: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        context = getattr(retrieval, "context", None)
        evidence = getattr(retrieval, "evidence", None)
        rag = getattr(retrieval, "rag", None)
        if isinstance(context, str) and isinstance(evidence, list) and isinstance(rag, dict):
            return context, {
                "evidence": [dict(item) for item in evidence if isinstance(item, dict)],
                "rag": deepcopy(rag),
            }
        return str(retrieval or ""), None

    @staticmethod
    def _unavailable_memory_context(exc: Exception) -> dict[str, Any]:
        reason = type(exc).__name__
        return {
            "evidence": [],
            "rag": {
                "turn_scoped": True,
                "enabled": True,
                "fallback_reason": reason,
                "dropped_evidence": [],
                "used_in_answer": False,
                "answer_policy": {
                    "state": "unavailable",
                    "action": "refuse",
                    "reason": reason,
                    "message": "Memory retrieval failed for this turn.",
                },
            },
        }

    @staticmethod
    def _turn_answer_policy(
        turn_rag: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        rag = turn_rag.get("rag") if isinstance(turn_rag, dict) else None
        policy = rag.get("answer_policy") if isinstance(rag, dict) else None
        return policy if isinstance(policy, dict) else None

    def _assistant_rag_metadata(self) -> dict[str, Any]:
        turn_rag = self.current_turn_rag
        if not isinstance(turn_rag, dict):
            return {}
        evidence = turn_rag.get("evidence")
        rag = turn_rag.get("rag")
        return {
            "evidence": evidence if isinstance(evidence, list) else [],
            "rag": rag if isinstance(rag, dict) else {},
        }

    def _history_for_session(
        self,
        conversation_session_id: str | None,
    ) -> list[dict[str, Any]]:
        if conversation_session_id is None:
            return self._conversation.history
        return list(
            self._conversation.get_messages(
                "",
                conversation_session_id=conversation_session_id,
            )[1:]
        )

    def _add_user_message(
        self,
        content: str,
        *,
        conversation_session_id: str | None,
    ) -> None:
        if conversation_session_id is None:
            self._conversation.add_user_message(content)
            return
        self._conversation.add_user_message(
            content,
            conversation_session_id=conversation_session_id,
        )

    def _add_assistant_message(
        self,
        content: str,
        *,
        conversation_session_id: str | None,
        evidence: list[dict[str, Any]] | None = None,
        rag: dict[str, Any] | None = None,
    ) -> None:
        metadata: dict[str, Any] = {}
        if evidence is not None:
            metadata["evidence"] = evidence
        if rag is not None:
            metadata["rag"] = rag
        if conversation_session_id is None:
            self._conversation.add_assistant_message(content, **metadata)
            return
        self._conversation.add_assistant_message(
            content,
            conversation_session_id=conversation_session_id,
            **metadata,
        )

    def _get_messages(
        self,
        system_prompt: str,
        *,
        conversation_session_id: str | None,
    ) -> list[dict[str, Any]]:
        if conversation_session_id is None:
            return self._conversation.get_messages(system_prompt)
        return self._conversation.get_messages(
            system_prompt,
            conversation_session_id=conversation_session_id,
        )

    def _remove_latest_user_message(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None,
    ) -> None:
        remove = getattr(self._conversation, "remove_latest_user_message", None)
        if callable(remove):
            if conversation_session_id is None:
                remove(user_text)
            else:
                remove(user_text, conversation_session_id=conversation_session_id)
            return
        history = self._history_for_session(conversation_session_id)
        for i in range(len(history) - 1, -1, -1):
            m = history[i]
            if m.get("role") == "user" and m.get("content") == user_text:
                history.pop(i)
                break

    def _remove_latest_assistant_message(
        self,
        content: str,
        *,
        conversation_session_id: str | None,
    ) -> None:
        remove = getattr(self._conversation, "remove_latest_assistant_message", None)
        if callable(remove):
            if conversation_session_id is None:
                remove(content)
            else:
                remove(content, conversation_session_id=conversation_session_id)
            return

        # Compatibility fallback for lightweight conversation adapters. The
        # production ConversationManager exposes the method above for sessions.
        history = getattr(self._conversation, "history", None)
        if not isinstance(history, list) or conversation_session_id is not None:
            return
        for i in range(len(history) - 1, -1, -1):
            message = history[i]
            if message.get("role") == "assistant" and message.get("content") == content:
                history.pop(i)
                return

    def _latest_history_role(
        self,
        *,
        conversation_session_id: str | None,
    ) -> str | None:
        history = self._history_for_session(conversation_session_id)
        if not history:
            return None
        return history[-1].get("role")

    def _classify_error_message(self, exc: Exception) -> str:
        """Return a user-facing voice message for an LLM pipeline error."""
        return classify_llm_error(exc)
