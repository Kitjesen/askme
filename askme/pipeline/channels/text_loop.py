"""Text-mode main loop -terminal input ->intent routing ->brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from askme.conversation import (
    InteractionInput,
    InteractionTurnManager,
    TurnOutcome,
)
from askme.pipeline.channels.external_turns import record_external_turn
from askme.pipeline.channels.runtime_bridge_calls import (
    try_handle_runtime_bridge_turn,
    try_runtime_bridge_turn,
)
from askme.pipeline.core.trace import get_tracer
from askme.robot_interaction import attach_intent_route_trace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from askme.memory.core.conversation import ConversationManager
    from askme.pipeline.channels.commands import CommandHandler
    from askme.pipeline.core.brain_pipeline import BrainPipeline
    from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
    from askme.ports import AudioFrontendPort, VoiceTurnBridgePort
    from askme.robot_interaction import IntentRouter
    from askme.skills.core.skill_manager import SkillManager


class _TextClarificationAudio:
    """Minimal audio adapter for text-mode proactive slot filling via stdin.

    Speaks by printing to console; listens by reading from stdin.
    This allows ClarificationPlannerAgent to fill slots interactively
    in text mode without requiring a microphone.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        logger.info(f"\n[Clarification]: {text}")

    def start_playback(self) -> None: ...
    def stop_playback(self) -> None: ...
    def wait_speaking_done(self) -> None: ...
    def drain_buffers(self) -> None: ...

    _LISTEN_TIMEOUT = 30.0  # seconds before clarification auto-cancels in text mode

    def listen_loop(self) -> str | None:
        import queue

        q: queue.Queue[str | None] = queue.Queue()

        def _read() -> None:
            try:
                q.put(input("[You]: ").strip() or None)
            except (EOFError, KeyboardInterrupt):
                q.put(None)

        import threading as _threading

        t = _threading.Thread(target=_read, daemon=True)
        t.start()
        try:
            return q.get(timeout=self._LISTEN_TIMEOUT)
        except queue.Empty:
            logger.info(
                "TextClarification: listen_loop timed out after %.0fs", self._LISTEN_TIMEOUT
            )
            return None


class TextLoop:
    """Interactive text-input loop.

    Reads from stdin, routes through :class:`IntentRouter`, delegates to
    :class:`BrainPipeline` or :class:`CommandHandler`.
    """

    MAX_CONSECUTIVE_ERRORS = 5  # text mode is more tolerant than voice (3)

    @property
    def current_turn_rag(self) -> dict[str, Any] | None:
        return self._pipeline.current_turn_rag

    @property
    def last_cognition_result(self) -> dict[str, Any] | None:
        return self._last_cognition_result.get()

    @last_cognition_result.setter
    def last_cognition_result(self, value: dict[str, Any] | None) -> None:
        self._last_cognition_result.set(value)

    def __init__(
        self,
        *,
        router: IntentRouter,
        pipeline: BrainPipeline,
        commands: CommandHandler,
        conversation: ConversationManager,
        skill_manager: SkillManager,
        audio: AudioFrontendPort,
        voice_runtime_bridge: VoiceTurnBridgePort | None = None,
        dispatcher: SkillDispatcher | None = None,
        cognition_handler: Any | None = None,
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._commands = commands
        self._conversation = conversation
        self._skill_manager = skill_manager
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge
        self._dispatcher = dispatcher
        self._cognition_handler = cognition_handler
        self._active_planning_session_id: str | None = None
        self._active_planning_session_ids: dict[str, str] = {}
        self._conversation_session_ids: dict[str, str] = {}
        self._thread_turn_locks: dict[str, asyncio.Lock] = {}
        self._thread_turn_lock_users: dict[str, int] = {}
        self._last_cognition_result: ContextVar[dict[str, Any] | None] = ContextVar(
            f"text_loop_cognition_{id(self)}",
            default=None,
        )

        from askme.pipeline.proactive import ProactiveOrchestrator

        self._proactive = ProactiveOrchestrator.default(pipeline=pipeline, dispatcher=dispatcher)
        self._text_audio = _TextClarificationAudio()

    async def run(self) -> None:
        """Block until the user types /quit or presses Ctrl+C."""
        from askme.robot_interaction import IntentType

        logger.info("Text mode. Commands: /clear /history /skills /quit")
        logger.info("Loaded %d previous messages.", len(self._conversation.history))
        logger.info("Skills: %s", self._skill_manager.get_skill_catalog())

        consecutive_errors = 0
        idle_task = self._pipeline.start_idle_reflection()
        _tracer = get_tracer()
        while True:
            memory_task: asyncio.Task[str] | None = None
            _trace = None
            self._text_audio.spoken.clear()  # prevent unbounded growth across turns
            try:
                user_text = await asyncio.to_thread(input, "[You]: ")
                user_text = user_text.strip()
                if not user_text:
                    continue

                conversation_session_id = self._conversation_session_for("text", channel="text")
                turn_correlation_id = f"text-turn-{uuid4().hex}"

                _trace = _tracer.start_trace("text_turn")
                _trace.metadata["user_text"] = user_text[:60]
                consecutive_errors = 0  # reset on successful input

                # Cancel idle reflection on user activity
                if idle_task and not idle_task.done():
                    idle_task.cancel()

                pending_handler = self._pipeline.handle_pending_tool_response
                pending_reply = await pending_handler(
                    user_text,
                    **self._supported_turn_context_kwargs(
                        pending_handler,
                        conversation_session_id=conversation_session_id,
                        turn_correlation_id=turn_correlation_id,
                        source="text",
                    ),
                )
                if pending_reply is not None:
                    logger.info("[Assistant]: %s", pending_reply)
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Start memory prefetch ASAP (overlaps with routing)
                memory_task = self._pipeline.start_memory_prefetch(user_text)

                with _tracer.span("intent_route") as _route_span:
                    intent = self._router.route(user_text)
                    _route_span.metadata.update(
                        attach_intent_route_trace(_trace, intent, source="text")
                    )

                if intent.type == IntentType.ESTOP:
                    self._pipeline.handle_estop()
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    reply = "急停已触发。"
                    record_external_turn(
                        self._pipeline,
                        user_text,
                        reply,
                        source="text",
                        channel="text",
                        conversation_session_id=conversation_session_id,
                        turn_id=turn_correlation_id,
                        metadata={"intent_type": "estop"},
                    )
                    logger.info("[Assistant]: %s", reply)
                    continue

                if intent.type == IntentType.COMMAND:
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    handled = self._commands.handle(intent.command or "")
                    self._suppress_direct_turn(
                        user_text,
                        source="text",
                        conversation_session_id=conversation_session_id,
                        turn_id=turn_correlation_id,
                        reason="text_command",
                        metadata={
                            "command": intent.command or "",
                            "handled": handled,
                        },
                    )
                    if handled:
                        break
                    continue
                cognition_reply = await self._maybe_handle_cognition_turn(
                    user_text,
                    source="text",
                    speak=False,
                    conversation_session_id=conversation_session_id,
                )
                if cognition_reply is not None:
                    logger.info("[Assistant]: %s", cognition_reply)
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                if intent.type == IntentType.VOICE_TRIGGER:
                    # Cancel memory prefetch -skill path never uses the result
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    # Try runtime bridge first -edge service may route to arbiter
                    bridge_handled = await self._maybe_handle_runtime_bridge(
                        user_text,
                        conversation_session_id=conversation_session_id,
                        turn_correlation_id=turn_correlation_id,
                    )
                    if bridge_handled:
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue
                    # Bridge not configured / failed -local skill dispatch
                    if self._dispatcher:
                        _result = await self._proactive.run(
                            intent.skill_name or "",
                            user_text,
                            self._text_audio,
                            source="text",
                        )
                        if _result.proceed:
                            await self._dispatcher.dispatch(
                                intent.skill_name or "",
                                _result.enriched_text,
                                source="text",
                                **self._supported_turn_context_kwargs(
                                    self._dispatcher.dispatch,
                                    conversation_session_id=conversation_session_id,
                                    turn_correlation_id=turn_correlation_id,
                                ),
                            )
                        elif _result.interrupt_payload:
                            # User bailed out and issued a new intent in the same breath
                            logger.info(
                                "TextLoop: rerouting interrupt_payload: %r",
                                _result.interrupt_payload,
                            )
                            _reroute_intent = self._router.route(_result.interrupt_payload)
                            attach_intent_route_trace(
                                _trace,
                                _reroute_intent,
                                source="text",
                                stage="interrupt_reroute",
                            )
                            if (
                                _reroute_intent.type == IntentType.VOICE_TRIGGER
                                and _reroute_intent.skill_name
                            ):
                                _rr = await self._proactive.run(
                                    _reroute_intent.skill_name,
                                    _result.interrupt_payload,
                                    self._text_audio,
                                    source="text",
                                )
                                if _rr.proceed:
                                    await self._dispatcher.dispatch(
                                        _reroute_intent.skill_name,
                                        _rr.enriched_text,
                                        source="text",
                                        **self._supported_turn_context_kwargs(
                                            self._dispatcher.dispatch,
                                            conversation_session_id=conversation_session_id,
                                            turn_correlation_id=turn_correlation_id,
                                        ),
                                    )
                            else:
                                reply = await self._dispatcher.handle_general(
                                    _result.interrupt_payload,
                                    source="text",
                                    **self._supported_turn_context_kwargs(
                                        self._dispatcher.handle_general,
                                        conversation_session_id=conversation_session_id,
                                        turn_correlation_id=turn_correlation_id,
                                    ),
                                )
                                logger.info("[Assistant]: %s", reply or "")
                    else:
                        await self._pipeline.execute_skill(
                            intent.skill_name or "",
                            user_text,
                            source="text",
                            **self._supported_turn_context_kwargs(
                                self._pipeline.execute_skill,
                                conversation_session_id=conversation_session_id,
                                turn_correlation_id=turn_correlation_id,
                            ),
                        )
                    continue

                if intent.type == IntentType.GENERAL:
                    bridge_handled = await self._maybe_handle_runtime_bridge(
                        user_text,
                        conversation_session_id=conversation_session_id,
                        turn_correlation_id=turn_correlation_id,
                    )
                    if bridge_handled:
                        # Cancel the memory prefetch we started earlier -the bridge
                        # handled the turn so the prefetched context is no longer needed.
                        if memory_task and not memory_task.done():
                            memory_task.cancel()
                        memory_task = None
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                # General ->LLM (pass pre-fetched memory)
                if self._dispatcher:
                    reply = await self._dispatcher.handle_general(
                        user_text,
                        source="text",
                        memory_task=memory_task,
                        **self._supported_turn_context_kwargs(
                            self._dispatcher.handle_general,
                            conversation_session_id=conversation_session_id,
                            turn_correlation_id=turn_correlation_id,
                        ),
                    )
                else:
                    reply = await self._pipeline.process(
                        user_text,
                        memory_task=memory_task,
                        source="text",
                        **self._supported_turn_context_kwargs(
                            self._pipeline.process,
                            conversation_session_id=conversation_session_id,
                            turn_correlation_id=turn_correlation_id,
                        ),
                    )
                memory_task = None  # pipeline took ownership
                logger.info("[Assistant]: %s", reply)
                try:
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                finally:
                    self._audio.stop_playback()

                # Restart idle reflection timer
                if idle_task and not idle_task.done():
                    idle_task.cancel()
                idle_task = self._pipeline.start_idle_reflection()

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as exc:
                consecutive_errors += 1
                logger.error("Text loop error: %s", exc)
                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "Text loop degraded: %d consecutive errors, pausing 3s",
                        consecutive_errors,
                    )
                    logger.info("多次错误，系统暂时异常，请稍后重试。")
                    await asyncio.sleep(3)
                    consecutive_errors = 0
            finally:
                if _trace is not None:
                    _tracer.finish_trace()
                if memory_task is not None and not memory_task.done():
                    memory_task.cancel()
                    try:
                        await memory_task
                    except (asyncio.CancelledError, Exception):
                        pass

        # Session-end summarization -save L2 summary if enough conversation happened
        _sm = getattr(self._pipeline, "_session_memory", None)
        if _sm and len(self._conversation.history) > 4:
            try:
                await asyncio.to_thread(_sm.summarize_and_save, self._conversation.history)
            except Exception as e:
                logger.warning("Session summary failed: %s", e)

        logger.info("Bye!")

    async def process_turn(
        self,
        user_text: str,
        *,
        speak: bool = False,
        runtime_policy: str = "disabled",
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
    ) -> str:
        """Serialize all routing paths for one Thread while other Threads run."""

        source = "voice" if speak else "text"
        resolved_conversation_session_id = str(conversation_session_id or "").strip() or None
        if resolved_conversation_session_id is None:
            resolved_conversation_session_id = self._conversation_session_for(
                source, channel=source
            )
        else:
            self._conversation_session_ids[self._conversation_cache_key(source, source)] = (
                resolved_conversation_session_id
            )
        turn_correlation_id = f"text-turn-{uuid4().hex}"
        thread_key = resolved_conversation_session_id or "__anonymous__"
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
                    speak=speak,
                    runtime_policy=runtime_policy,
                    conversation_session_id=resolved_conversation_session_id,
                    planning_session_id=planning_session_id,
                    turn_correlation_id=turn_correlation_id,
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
        speak: bool = False,
        runtime_policy: str = "disabled",
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        turn_correlation_id: str,
    ) -> str:
        """Execute a single turn through full intent routing + skill dispatch.

        Used by /api/chat for single-turn local chat routing. It does not use
        the runtime bridge unless ``runtime_policy="runtime_first"`` is passed.
        Returns the response string (empty string for commands/estop).
        """
        from askme.robot_interaction import IntentType

        clear_turn_context = getattr(self._pipeline, "clear_turn_context", None)
        if callable(clear_turn_context):
            clear_turn_context()
        self._text_audio.spoken.clear()
        self.last_cognition_result = None
        source = "voice" if speak else "text"
        if planning_session_id:
            cleaned_planning_session_id = str(planning_session_id).strip()
            self._set_active_planning_session(
                str(conversation_session_id or "").strip() or None,
                cleaned_planning_session_id or None,
            )
        runtime_policy = _normalize_runtime_policy(runtime_policy)
        pending_handler = getattr(self._pipeline, "handle_pending_tool_response", None)
        if callable(pending_handler):
            pending_reply = await pending_handler(
                user_text,
                **self._supported_turn_context_kwargs(
                    pending_handler,
                    conversation_session_id=conversation_session_id,
                    turn_correlation_id=turn_correlation_id,
                    source=source,
                ),
            )
            if pending_reply is not None:
                return str(pending_reply)
        memory_task = self._pipeline.start_memory_prefetch(user_text)
        _tracer = get_tracer()
        _trace = _tracer.start_trace("text_process_turn")
        _trace.metadata["user_text"] = user_text[:60]
        try:
            with _tracer.span("intent_route") as _route_span:
                intent = self._router.route(user_text)
                _route_span.metadata.update(
                    attach_intent_route_trace(_trace, intent, source=source)
                )

            if intent.type == IntentType.ESTOP:
                self._pipeline.handle_estop()
                memory_task.cancel()
                reply = "急停已触发。"
                record_external_turn(
                    self._pipeline,
                    user_text,
                    reply,
                    source=source,
                    channel=source,
                    conversation_session_id=conversation_session_id,
                    turn_id=turn_correlation_id,
                    metadata={"intent_type": "estop"},
                )
                return reply
            if intent.type == IntentType.QUICK_REPLY:
                memory_task.cancel()
                reply = intent.reply_text or intent.skill_name or ""
                if speak:
                    cache_key = str(intent.cached_audio_key or "").strip()
                    speak_cached = getattr(self._audio, "speak_cached_and_wait", None)
                    used_cache = bool(
                        cache_key
                        and callable(speak_cached)
                        and await speak_cached(reply, cache_key=cache_key)
                    )
                    if not used_cache:
                        self._audio.speak(reply)
                        self._audio.start_playback()
                        try:
                            await asyncio.to_thread(self._audio.wait_speaking_done)
                        finally:
                            self._audio.stop_playback()
                record_external_turn(
                    self._pipeline,
                    user_text,
                    reply,
                    source=source,
                    channel=source,
                    conversation_session_id=conversation_session_id,
                    turn_id=turn_correlation_id,
                )
                return reply

            if intent.type == IntentType.COMMAND:
                memory_task.cancel()
                self._suppress_direct_turn(
                    user_text,
                    source=source,
                    conversation_session_id=conversation_session_id,
                    turn_id=turn_correlation_id,
                    reason="text_command",
                    metadata={"command": intent.command or ""},
                )
                return ""
            if runtime_policy == "runtime_first":
                runtime_reply = await self._maybe_handle_runtime_bridge_reply(
                    user_text,
                    conversation_session_id=conversation_session_id,
                    turn_correlation_id=turn_correlation_id,
                    source=source,
                )
                if runtime_reply is not None:
                    memory_task.cancel()
                    return runtime_reply

            cognition_reply = await self._maybe_handle_cognition_turn(
                user_text,
                source=source,
                speak=speak,
                conversation_session_id=conversation_session_id,
            )
            if cognition_reply is not None:
                memory_task.cancel()
                return cognition_reply

            if intent.type == IntentType.VOICE_TRIGGER:
                memory_task.cancel()
                memory_task = None
                if self._dispatcher:
                    _result = await self._proactive.run(
                        intent.skill_name or "",
                        user_text,
                        self._text_audio,
                        source=source,
                    )
                    if _result.proceed:
                        return await self._dispatcher.dispatch(
                            intent.skill_name or "",
                            _result.enriched_text,
                            source=source,
                            **self._supported_turn_context_kwargs(
                                self._dispatcher.dispatch,
                                conversation_session_id=conversation_session_id,
                                turn_correlation_id=turn_correlation_id,
                            ),
                        )
                    if _result.interrupt_payload:
                        return await self._dispatcher.handle_general(
                            _result.interrupt_payload,
                            source=source,
                            **self._supported_turn_context_kwargs(
                                self._dispatcher.handle_general,
                                conversation_session_id=conversation_session_id,
                                turn_correlation_id=turn_correlation_id,
                            ),
                        )
                    return ""
                return await self._pipeline.execute_skill(
                    intent.skill_name or "",
                    user_text,
                    source=source,
                    **self._supported_turn_context_kwargs(
                        self._pipeline.execute_skill,
                        conversation_session_id=conversation_session_id,
                        turn_correlation_id=turn_correlation_id,
                    ),
                )

            # GENERAL
            if self._dispatcher:
                return await self._dispatcher.handle_general(
                    user_text,
                    source=source,
                    memory_task=memory_task,
                    **self._supported_turn_context_kwargs(
                        self._dispatcher.handle_general,
                        conversation_session_id=conversation_session_id,
                        turn_correlation_id=turn_correlation_id,
                    ),
                )
            reply = await self._pipeline.process(
                user_text,
                memory_task=memory_task,
                source=source,
                **self._supported_turn_context_kwargs(
                    self._pipeline.process,
                    conversation_session_id=conversation_session_id,
                    turn_correlation_id=turn_correlation_id,
                ),
            )
            memory_task = None
            return reply
        finally:
            _tracer.finish_trace()
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _maybe_handle_cognition_turn(
        self,
        user_text: str,
        *,
        source: str,
        speak: bool,
        conversation_session_id: str | None = None,
    ) -> str | None:
        """Route robot task planning turns into cognition when available."""
        cognition = self._cognition_handler
        if cognition is None:
            return None

        conversation_session_id = str(
            conversation_session_id or ""
        ).strip() or self._conversation_session_for(source)
        active_planning_session_id = self._active_planning_session_for(conversation_session_id)
        if not self._should_route_to_cognition(
            user_text,
            active_planning_session_id=active_planning_session_id,
        ):
            return None

        payload: dict[str, Any] = {
            "text": user_text,
            "channel": f"{source}-chat",
        }
        if conversation_session_id:
            payload["conversation_session_id"] = conversation_session_id
        if active_planning_session_id:
            payload["planning_session_id"] = active_planning_session_id

        if _is_confirmation_text(user_text):
            payload["operator_confirmation"] = True
        elif _is_cancel_text(user_text):
            payload["action"] = "cancel"

        plan_from_payload = getattr(cognition, "plan_from_payload", None)
        if not callable(plan_from_payload):
            return None

        try:
            result = await plan_from_payload(payload)
        except Exception as exc:
            logger.warning("TextLoop: cognition planning failed, falling back to chat: %s", exc)
            return None
        if not isinstance(result, dict):
            return None

        plan = result.get("plan")
        if not isinstance(plan, dict):
            return None

        session_id = str(plan.get("planning_session_id") or "").strip()
        stage = str(plan.get("interaction_state") or plan.get("stage") or "")
        if session_id and stage not in {"cancelled", "ready_for_arbiter", "answer_ready"}:
            self._set_active_planning_session(conversation_session_id, session_id)
        elif stage in {"cancelled", "ready_for_arbiter", "answer_ready"}:
            self._set_active_planning_session(conversation_session_id, None)

        reply = _cognition_reply_text(plan)
        if not reply:
            return None

        self.last_cognition_result = {
            "handled": True,
            "plan": plan,
            "sync": result.get("sync", {}),
        }
        record_external_turn(
            self._pipeline,
            user_text,
            reply,
            source="cognition",
            conversation_session_id=conversation_session_id,
            metadata={"planning_session_id": session_id} if session_id else None,
        )
        if speak:
            await self._speak_reply(reply)
        return reply

    def _conversation_session_for(self, source: str, *, channel: str | None = None) -> str | None:
        channel_name = channel or f"{source}-chat"
        cache_key = self._conversation_cache_key(source, channel_name)
        cached_session_id = self._conversation_session_ids.get(cache_key)
        if cached_session_id:
            if self._cached_session_is_active(cached_session_id):
                return cached_session_id
            self._conversation_session_ids.pop(cache_key, None)
        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        get_or_create = getattr(manager, "get_or_create", None)
        if not callable(get_or_create):
            session_id = f"{channel_name}-local-{uuid4().hex}"
            self._conversation_session_ids[cache_key] = session_id
            return session_id
        try:
            session = get_or_create(channel=channel_name)
        except Exception as exc:
            logger.debug("TextLoop: conversation session unavailable: %s", exc)
            session_id = f"{channel_name}-degraded-{uuid4().hex}"
            self._conversation_session_ids[cache_key] = session_id
            return session_id
        session_id = str(getattr(session, "session_id", "") or "").strip()
        if session_id:
            self._conversation_session_ids[cache_key] = session_id
        return session_id or None

    def _conversation_cache_key(self, source: str, channel: str) -> str:
        return f"{source}:{channel}"

    def _active_planning_session_for(
        self,
        conversation_session_id: str | None,
    ) -> str | None:
        if conversation_session_id:
            return self._active_planning_session_ids.get(conversation_session_id)
        return self._active_planning_session_id

    def _set_active_planning_session(
        self,
        conversation_session_id: str | None,
        planning_session_id: str | None,
    ) -> None:
        session = str(conversation_session_id or "").strip()
        planning = str(planning_session_id or "").strip()
        if session:
            if planning:
                self._active_planning_session_ids[session] = planning
            else:
                self._active_planning_session_ids.pop(session, None)
        self._active_planning_session_id = planning or None

    def _cached_session_is_active(self, session_id: str) -> bool:
        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if not callable(snapshot):
            return True
        current = snapshot(session_id)
        if current is None:
            return False
        return getattr(current, "status", "active") == "active"

    def _should_route_to_cognition(
        self,
        user_text: str,
        *,
        active_planning_session_id: str | None = None,
    ) -> bool:
        if active_planning_session_id:
            return True
        return _looks_like_cognition_request(user_text)

    @staticmethod
    def _supported_turn_context_kwargs(
        callback: Any,
        *,
        conversation_session_id: str | None,
        turn_correlation_id: str,
        source: str | None = None,
        turn_cancel_token: Any | None = None,
    ) -> dict[str, Any]:
        """Pass admitted Turn context without breaking legacy adapters."""

        signature_target = getattr(callback, "side_effect", None)
        if not callable(signature_target):
            signature_target = callback
        try:
            parameters = signature(signature_target).parameters
            accepts_kwargs = any(
                parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            return {}

        context: dict[str, Any] = {}
        if conversation_session_id is not None and (
            accepts_kwargs or "conversation_session_id" in parameters
        ):
            context["conversation_session_id"] = conversation_session_id
        if accepts_kwargs or "voice_turn_id" in parameters:
            context["voice_turn_id"] = turn_correlation_id
        elif "turn_id" in parameters:
            context["turn_id"] = turn_correlation_id
        if source is not None and (accepts_kwargs or "source" in parameters):
            context["source"] = source
        if turn_cancel_token is not None and (accepts_kwargs or "turn_cancel_token" in parameters):
            context["turn_cancel_token"] = turn_cancel_token
        return context

    def _suppress_direct_turn(
        self,
        user_text: str,
        *,
        source: str,
        conversation_session_id: str | None,
        turn_id: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an admitted text turn that product policy excludes from chat."""

        ledger = getattr(self._pipeline, "_turn_ledger", None)
        if ledger is None:
            return
        manager = InteractionTurnManager(ledger)
        context = manager.open(
            InteractionInput(
                user_text=user_text,
                source=source,
                thread_id=conversation_session_id,
                turn_id=turn_id,
                channel=source,
                metadata=metadata or {},
            )
        )
        manager.settle(
            context,
            TurnOutcome.suppress(
                reason=reason,
                metadata=metadata or {},
            ),
        )

    async def _speak_reply(self, reply: str) -> None:
        if not reply.strip():
            return
        self._audio.speak(reply.strip())
        self._audio.start_playback()
        try:
            await asyncio.to_thread(self._audio.wait_speaking_done)
        finally:
            self._audio.stop_playback()

    async def _maybe_handle_runtime_bridge_reply(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None = None,
        turn_correlation_id: str | None = None,
        source: str = "text",
    ) -> str | None:
        """Try the text runtime bridge for explicit single-turn chat policy."""
        if self._voice_runtime_bridge is None:
            return None

        resolved_session_id = str(conversation_session_id or "").strip() or None
        if resolved_session_id is None:
            resolved_session_id = self._conversation_session_for(source, channel=source)
        try:
            outcome = await try_runtime_bridge_turn(
                self._voice_runtime_bridge.handle_text_input,
                user_text,
                conversation_session_id=resolved_session_id,
                voice_turn_id=turn_correlation_id,
                pipeline=self._pipeline,
                dispatcher=self._dispatcher,
                on_spoken_reply=(
                    self._speak_reply
                    if source == "voice"
                    else lambda reply: logger.info("[Assistant]: %s", reply)
                ),
                label="Text",
            )
        except Exception as exc:
            logger.warning("TextLoop: runtime bridge failed, falling back locally: %s", exc)
            return None
        if not outcome.handled:
            return None
        return outcome.reply

    async def _maybe_handle_runtime_bridge(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None = None,
        turn_correlation_id: str | None = None,
    ) -> bool:
        """Try the runtime bridge first and fall back locally on bridge failures."""
        if self._voice_runtime_bridge is None:
            return False

        conversation_session_id = str(
            conversation_session_id or ""
        ).strip() or self._conversation_session_for("text", channel="text")
        turn_correlation_id = str(turn_correlation_id or "").strip() or f"text-turn-{uuid4().hex}"
        try:
            return await try_handle_runtime_bridge_turn(
                self._voice_runtime_bridge.handle_text_input,
                user_text,
                conversation_session_id=conversation_session_id,
                pipeline=self._pipeline,
                voice_turn_id=turn_correlation_id,
                dispatcher=self._dispatcher,
                on_spoken_reply=lambda reply: logger.info("[Assistant]: %s", reply),
                label="Text",
            )
        except Exception as exc:
            logger.warning("TextLoop: runtime bridge failed, falling back locally: %s", exc)
            return False


_COGNITION_KEYWORDS = (
    "巡检",
    "检查",
    "巡逻",
    "拍照",
    "截图",
    "抓拍",
    "取证",
    "状态",
    "电量",
    "导航",
    "去",
    "到",
    "拿",
    "抓",
    "夹取",
    "急停",
    "停止",
    "停下",
    "patrol",
    "inspect",
    "photo",
    "snapshot",
    "capture",
    "status",
    "battery",
    "navigate",
    "go to",
    "pick",
    "grab",
    "stop",
    "estop",
    "e-stop",
)
_CONFIRMATION_WORDS = {"确认", "可以", "是", "继续", "ok", "yes", "confirm", "confirmed"}
_CANCEL_WORDS = {"取消", "算了", "不要了", "停止规划", "cancel", "abort", "stop"}


def _looks_like_cognition_request(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return bool(lowered) and any(keyword in lowered for keyword in _COGNITION_KEYWORDS)


def _is_confirmation_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return lowered in _CONFIRMATION_WORDS


def _is_cancel_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return lowered in _CANCEL_WORDS


def _normalize_runtime_policy(value: str | None) -> str:
    policy = str(value or "disabled").strip().lower().replace("-", "_")
    if policy in {"runtime_first", "first", "bridge_first"}:
        return "runtime_first"
    if policy in {"control_only", "controls_only"}:
        return "control_only"
    return "disabled"


def _cognition_reply_text(plan: dict[str, Any]) -> str:
    for key in ("next_prompt", "clarification_question"):
        value = plan.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    stage = str(plan.get("interaction_state") or plan.get("stage") or "")
    if stage == "awaiting_confirmation":
        return "已生成任务草案，请确认后再交给运行时仲裁器。"
    if stage == "ready_for_arbiter":
        return "计划已确认，可以交给运行时仲裁器继续处理。"
    if stage == "cancelled":
        return "已取消当前规划。"
    if stage == "clarifying":
        return "请补充目标、位置或约束条件。"
    return ""
