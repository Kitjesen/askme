"""Turn executor — orchestrates one full conversation turn: memory → LLM → tools → TTS → save."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from askme.pipeline.hooks import PipelineHooks
from askme.pipeline.protocols import TurnContext
from askme.pipeline.trace import get_tracer
from askme.pipeline.utils import classify_llm_error, set_log_context

if TYPE_CHECKING:
    from askme.llm.client import LLMClient
    from askme.llm.conversation import ConversationManager
    from askme.memory.bridge import MemoryBridge
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.system import MemorySystem
    from askme.perception.vision_bridge import VisionBridge
    from askme.pipeline.prompt_builder import PromptBuilder
    from askme.pipeline.stream_processor import StreamProcessor
    from askme.robot.safety_client import DogSafetyClient
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)


class TurnExecutor:
    """Orchestrates one full conversation turn: memory → LLM → tools → TTS → save."""

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
        # create_task() propagates the current context — no manual copy needed.
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
        audio: AudioAgent,
        prompt_builder: PromptBuilder,
        stream_processor: StreamProcessor,
        dog_safety: DogSafetyClient | None = None,
        vision: VisionBridge | None = None,
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
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        # Semaphore initialized eagerly here so concurrent calls to process()
        # before the first turn completes don't each create their own semaphore.
        self._llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

    # ── Public API ────────────────────────────────────────────

    @property
    def last_spoken_text(self) -> str:
        """The most recent text spoken via TTS. Used by repeat_last skill."""
        return self._last_spoken_text

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent (set by VoiceModule/TextModule after build)."""
        self._audio = audio

    # ── Core turn orchestration ───────────────────────────────

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
                logger.info("[Dream] Skipping reflection — LLM busy with user turn")
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

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[str]:
        """Start memory retrieval as a background task. Call ASAP after ASR returns."""
        return asyncio.create_task(self._memory.retrieve(user_text))

    async def process(
        self, user_text: str, *, memory_task: asyncio.Task[str] | None = None,
        source: str = "voice",
    ) -> str:
        """Run the full brain pipeline for *user_text*. Returns assistant reply."""
        # Set structured log context for this turn so all log records carry
        # the trace ID and source without manual argument threading.
        _tracer = get_tracer()
        _trace = _tracer.start_trace("voice_turn" if source == "voice" else "text_turn")
        set_log_context(trace_id=_trace.id, session_id=source)
        logger.info("Processing: %s", user_text[:60])

        if self._cancel_token is not None and self._cancel_token.is_set():
            logger.warning("[TurnExecutor] cancel_token set — skipping turn")
            return ""

        # ── pre_turn hook (Claude Code: UserPromptSubmit) ──────────────────
        # Build a lightweight TurnContext snapshot for hooks; the token is
        # shared with sub-components so hooks can also trigger E-STOP.
        _token = self._cancel_token if self._cancel_token is not None else asyncio.Event()
        _ctx = TurnContext(
            user_text=user_text,
            source=source,
            cancel_token=_token,
            voice_model=self._voice_model,
        )
        if self._hooks:
            skip = await self._hooks.fire_pre_turn(_ctx)
            if skip:
                logger.info("[TurnExecutor] pre_turn hook requested turn skip")
                return ""

        self._audio.drain_buffers()

        if self._dog_safety and self._dog_safety.is_configured():
            self._track_task(
                asyncio.to_thread(self._dog_safety.query_estop_state),
                name="estop_refresh",
            )

        if not memory_task:
            memory_task = asyncio.create_task(self._memory.retrieve(user_text))
        vision_task = self._start_vision_capture()

        try:
            with _tracer.span("memory_retrieve"):
                context_str = await memory_task
        except Exception as _me:
            logger.warning("[TurnExecutor] Memory retrieve failed: %s", _me)
            context_str = ""

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

        system_prompt = self._prompt_builder.build_system_prompt(
            context_str,
            scene_desc=scene_desc,
            user_text=user_text,
        )

        self._conversation.add_user_message(user_text)

        # Start compress AFTER add_user_message so the new user message is always
        # included in maybe_compress's recent[-KEEP_RECENT:] snapshot and never lost.
        async def _compress_bg() -> None:
            try:
                await self._conversation.maybe_compress(self._llm)
            except Exception as _e:
                logger.warning("Conversation compression failed (non-critical): %s", _e)

        self._track_task(_compress_bg(), name="conv_compress")
        messages = self._prompt_builder.prepare_messages(
            self._conversation.get_messages(system_prompt)
        )

        self._log_episode("command", f"用户说: {user_text}")

        self._audio.start_playback()
        try:
            async with self._llm_semaphore:
                full_response = await self._stream_processor.stream_with_tools(
                    messages, system_prompt, model=self._voice_model,
                    source=source,
                )
            if full_response.lstrip().startswith(self._SILENT_MARKER):
                logger.info("[SILENT] Not addressed to robot, suppressing output")
                self._audio.drain_buffers()
                # Remove exactly the user message we added — match by content to
                # avoid popping the wrong message if compress ran concurrently.
                for i in range(len(self._conversation.history) - 1, -1, -1):
                    m = self._conversation.history[i]
                    if m.get("role") == "user" and m.get("content") == user_text:
                        self._conversation.history.pop(i)
                        break
                return ""

            self._conversation.add_assistant_message(full_response)
            self._last_spoken_text = full_response

            # ── post_turn hook (Claude Code: Stop hook / notification) ─────
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

            if source == "voice":
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
            self._audio.speak(classify_llm_error(exc))
            error_msg = f"[系统错误] {type(exc).__name__}"
            last_role = (
                self._conversation.history[-1].get("role")
                if self._conversation.history
                else None
            )
            if last_role == "assistant":
                self._conversation.add_user_message("[系统恢复]")
            self._conversation.add_assistant_message(error_msg)
            return error_msg
        finally:
            self._audio.stop_playback()
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

    # ── Internal helpers ──────────────────────────────────────

    def _start_vision_capture(self) -> asyncio.Task[str] | None:
        if not self._vision or not self._vision.available:
            return None
        if not self._vision._vision_cfg.get("auto_capture", False):
            return None
        return asyncio.create_task(self._vision.describe_scene())

    def _classify_error_message(self, exc: Exception) -> str:
        """Return a user-facing voice message for an LLM pipeline error."""
        return classify_llm_error(exc)
