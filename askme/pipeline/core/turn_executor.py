"""Turn executor; orchestrates one full conversation turn: memory, LLM, tools, TTS, save."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from copy import deepcopy
from inspect import isawaitable, iscoroutinefunction
from typing import TYPE_CHECKING, Any

from askme.pipeline.core.hooks import PipelineHooks
from askme.pipeline.core.protocols import TurnContext
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
    _REFLECT_DELAY_S = 5.0       # seconds to wait before post-turn reflection

    def _track_task(
        self, coro: asyncio.Coroutine[Any, Any, Any], *, name: str | None = None
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
        cancel_token: asyncio.Event | None = None,
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

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent (set by VoiceModule/TextModule after build)."""
        self._audio = audio

    # Core turn orchestration.

    def _log_episode(self, kind: str, text: str) -> None:
        if self._mem is not None:
            self._mem.log_event(kind, text)
        elif self._episodic:
            self._episodic.log(kind, text)

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        """Start an idle-time reflection background task (dream consolidation)."""
        _ep = (self._mem.episodic if self._mem is not None else self._episodic)
        if not _ep:
            return None

        async def _idle_reflect() -> None:
            await asyncio.sleep(idle_seconds)
            _should = (
                self._mem.should_reflect() if self._mem is not None
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
        self, user_text: str, *, memory_task: asyncio.Task[Any] | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
    ) -> str:
        """Run the full brain pipeline for *user_text*. Returns assistant reply."""
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
        logger.info("Processing: %s", user_text[:60])
        is_voice = source == "voice"

        if self._cancel_token is not None and self._cancel_token.is_set():
            logger.warning("[TurnExecutor] cancel_token set; skipping turn")
            return ""

        # pre_turn hook (Claude Code: UserPromptSubmit).
        # Build a lightweight TurnContext snapshot for hooks; the token is
        # shared with sub-components so hooks can also trigger E-STOP.
        _token = self._cancel_token if self._cancel_token is not None else asyncio.Event()
        _ctx = TurnContext(
            user_text=user_text,
            source=source,
            cancel_token=_token,
            voice_model=self._voice_model,
            conversation_session_id=session_scope,
        )
        if self._hooks:
            skip = await self._hooks.fire_pre_turn(_ctx)
            if skip:
                logger.info("[TurnExecutor] pre_turn hook requested turn skip")
                return ""

        if is_voice:
            self._audio.drain_buffers()

        if self._dog_safety and self._dog_safety.is_configured():
            self._track_task(
                asyncio.to_thread(self._dog_safety.query_estop_state),
                name="estop_refresh",
            )

        if not memory_task:
            memory_task = self.start_memory_prefetch(user_text)
        vision_task = self._start_vision_capture()

        try:
            with _tracer.span("memory_retrieve"):
                retrieval = await memory_task
            context_str, turn_rag = self._coerce_memory_retrieval(retrieval)
            if turn_rag is not None:
                self._turn_rag_context.set(turn_rag)
        except Exception as _me:
            logger.warning("[TurnExecutor] Memory retrieve failed: %s", _me)
            context_str = ""
            turn_rag = self._unavailable_memory_context(_me)
            self._turn_rag_context.set(turn_rag)

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
            self._log_episode("perception", scene_desc)

        rag_policy = self._turn_answer_policy(turn_rag)
        if rag_policy is None:
            rag_policy = await self._memory_answer_policy()
        system_prompt = self._prompt_builder.build_system_prompt(
            context_str,
            scene_desc=scene_desc,
            user_text=user_text,
            rag_policy=rag_policy,
        )

        self._add_user_message(user_text, conversation_session_id=session_scope)
        forced_rag_reply = self._prompt_builder.build_forced_rag_reply(rag_policy)
        if forced_rag_reply:
            logger.info("[TurnExecutor] RAG policy forced deterministic reply")
            self._add_assistant_message(
                forced_rag_reply,
                conversation_session_id=session_scope,
                **self._assistant_rag_metadata(),
            )
            self._last_spoken_text = forced_rag_reply
            if self._hooks:
                await self._hooks.fire_post_turn(_ctx, forced_rag_reply)
            if self._mem is not None:
                self._track_task(
                    self._mem.save_to_vector(user_text, forced_rag_reply),
                    name="mem_save",
                )
            elif self._memory is not None:
                self._track_task(
                    self._memory.save(user_text, forced_rag_reply),
                    name="mem_save",
                )
            if is_voice:
                self._audio.speak(forced_rag_reply)
                await asyncio.to_thread(self._audio.wait_speaking_done)
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

        self._log_episode("command", f"用户说: {user_text}")

        if is_voice:
            self._audio.start_playback()
        try:
            async with self._llm_semaphore:
                full_response = await self._stream_processor.stream_with_tools(
                    messages, system_prompt, model=self._voice_model,
                    source=source,
                    conversation_session_id=session_scope,
                )
            if full_response.lstrip().startswith(self._SILENT_MARKER):
                logger.info("[SILENT] Not addressed to robot, suppressing output")
                self._audio.drain_buffers()
                # Remove exactly the user message we added; match by content to
                # avoid popping the wrong message if compress ran concurrently.
                self._remove_latest_user_message(
                    user_text,
                    conversation_session_id=session_scope,
                )
                return ""

            self._add_assistant_message(
                full_response,
                conversation_session_id=session_scope,
                **self._assistant_rag_metadata(),
            )
            self._last_spoken_text = full_response

            # post_turn hook (Claude Code: Stop hook / notification).
            if self._hooks:
                await self._hooks.fire_post_turn(_ctx, full_response)

            if self._mem is not None:
                self._track_task(
                    self._mem.save_to_vector(user_text, full_response), name="mem_save"
                )
            elif self._memory is not None:
                self._track_task(
                    self._memory.save(user_text, full_response), name="mem_save"
                )

            if is_voice:
                await asyncio.to_thread(self._audio.wait_speaking_done)

            self._log_episode("action", f"回复: {full_response[:100]}")

            if self._qp_memory is not None:
                _qp = self._qp_memory
                _resp = full_response

                async def _qp_voice_bg():
                    try:
                        await asyncio.to_thread(_qp.record_observation, "voice", user_text)
                        await asyncio.to_thread(_qp.process_turn, user_text, _resp)
                        if self._qp_turn_count % 10 == 0:
                            await asyncio.to_thread(_qp.save)
                    except Exception:
                        pass

                self._qp_turn_count += 1
                self._track_task(_qp_voice_bg(), name="qp_memory")

            _should = (
                self._mem.should_reflect() if self._mem is not None
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
            logger.error("LLM pipeline error: %s", exc)
            self._log_episode("error", f"LLM错误: {exc}")
            if is_voice:
                self._audio.speak(classify_llm_error(exc))
            error_msg = f"[系统错误] {type(exc).__name__}"
            if self._latest_history_role(conversation_session_id=session_scope) == "assistant":
                self._add_user_message(
                    "[系统错误恢复]",
                    conversation_session_id=session_scope,
                )
            self._add_assistant_message(error_msg, conversation_session_id=session_scope)
            return error_msg
        finally:
            if is_voice:
                self._audio.stop_playback()
            if _owns_trace:
                _tracer.finish_trace()

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

    def _start_vision_capture(self) -> asyncio.Task[str] | None:
        if not self._vision or not self._vision.available:
            return None
        auto_capture_enabled = getattr(self._vision, "auto_capture_enabled", None)
        if callable(auto_capture_enabled) and not auto_capture_enabled():
            return None
        return asyncio.create_task(self._vision.describe_scene())

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
