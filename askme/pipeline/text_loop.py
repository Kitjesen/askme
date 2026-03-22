"""Text-mode main loop — terminal input → intent routing → brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from askme.pipeline.external_turns import record_external_turn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from askme.llm.conversation import ConversationManager
    from askme.llm.intent_router import IntentRouter
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.pipeline.commands import CommandHandler
    from askme.pipeline.skill_dispatcher import SkillDispatcher
    from askme.skills.skill_manager import SkillManager
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.runtime_bridge import VoiceRuntimeBridge


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
        print(f"\n[Clarification]: {text}")  # noqa: T201

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
            logger.info("TextClarification: listen_loop timed out after %.0fs", self._LISTEN_TIMEOUT)
            return None


class TextLoop:
    """Interactive text-input loop.

    Reads from stdin, routes through :class:`IntentRouter`, delegates to
    :class:`BrainPipeline` or :class:`CommandHandler`.
    """

    MAX_CONSECUTIVE_ERRORS = 5  # text mode is more tolerant than voice (3)

    def __init__(
        self,
        *,
        router: IntentRouter,
        pipeline: BrainPipeline,
        commands: CommandHandler,
        conversation: ConversationManager,
        skill_manager: SkillManager,
        audio: AudioAgent,
        voice_runtime_bridge: VoiceRuntimeBridge | None = None,
        dispatcher: SkillDispatcher | None = None,
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._commands = commands
        self._conversation = conversation
        self._skill_manager = skill_manager
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge
        self._dispatcher = dispatcher

        from askme.pipeline.proactive import ProactiveOrchestrator
        self._proactive = ProactiveOrchestrator.default(
            pipeline=pipeline, dispatcher=dispatcher
        )
        self._text_audio = _TextClarificationAudio()

    async def run(self) -> None:
        """Block until the user types /quit or presses Ctrl+C."""
        from askme.llm.intent_router import IntentType

        logger.info("Text mode. Commands: /clear /history /skills /quit")
        logger.info("Loaded %d previous messages.", len(self._conversation.history))
        logger.info("Skills: %s", self._skill_manager.get_skill_catalog())

        consecutive_errors = 0
        idle_task = self._pipeline.start_idle_reflection()
        while True:
            memory_task: asyncio.Task[str] | None = None
            self._text_audio.spoken.clear()  # prevent unbounded growth across turns
            try:
                user_text = await asyncio.to_thread(input, "[You]: ")
                user_text = user_text.strip()
                if not user_text:
                    continue

                consecutive_errors = 0  # reset on successful input

                # Cancel idle reflection on user activity
                if idle_task and not idle_task.done():
                    idle_task.cancel()

                pending_reply = await self._pipeline.handle_pending_tool_response(user_text)
                if pending_reply is not None:
                    logger.info("[Assistant]: %s", pending_reply)
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Start memory prefetch ASAP (overlaps with routing)
                memory_task = self._pipeline.start_memory_prefetch(user_text)

                intent = self._router.route(user_text)

                if intent.type == IntentType.ESTOP:
                    self._pipeline.handle_estop()
                    continue

                if intent.type == IntentType.COMMAND:
                    if self._commands.handle(intent.command or ""):
                        break
                    continue

                if intent.type == IntentType.VOICE_TRIGGER:
                    # Cancel memory prefetch — skill path never uses the result
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    # Try runtime bridge first — edge service may route to arbiter
                    bridge_handled = await self._maybe_handle_runtime_bridge(user_text)
                    if bridge_handled:
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue
                    # Bridge not configured / failed — local skill dispatch
                    if self._dispatcher:
                        _result = await self._proactive.run(
                            intent.skill_name or "", user_text, self._text_audio,
                            source="text",
                        )
                        if _result.proceed:
                            await self._dispatcher.dispatch(
                                intent.skill_name or "", _result.enriched_text,
                                source="text",
                            )
                        elif _result.interrupt_payload:
                            # User bailed out and issued a new intent in the same breath
                            logger.info(
                                "TextLoop: rerouting interrupt_payload: %r",
                                _result.interrupt_payload,
                            )
                            _reroute_intent = self._router.route(_result.interrupt_payload)
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
                                    )
                            else:
                                reply = await self._dispatcher.handle_general(
                                    _result.interrupt_payload, source="text",
                                )
                                logger.info("[Assistant]: %s", reply or "")
                    else:
                        await self._pipeline.execute_skill(
                            intent.skill_name or "", user_text,
                        )
                    continue

                if intent.type == IntentType.GENERAL:
                    bridge_handled = await self._maybe_handle_runtime_bridge(user_text)
                    if bridge_handled:
                        # Cancel the memory prefetch we started earlier — the bridge
                        # handled the turn so the prefetched context is no longer needed.
                        if memory_task and not memory_task.done():
                            memory_task.cancel()
                        memory_task = None
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                # General → LLM (pass pre-fetched memory)
                if self._dispatcher:
                    reply = await self._dispatcher.handle_general(
                        user_text, source="text", memory_task=memory_task,
                    )
                else:
                    reply = await self._pipeline.process(user_text, memory_task=memory_task)
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
                    print("⚠️ 多次错误，系统暂时异常，请稍候...")  # noqa: T201
                    await asyncio.sleep(3)
                    consecutive_errors = 0
            finally:
                if memory_task is not None and not memory_task.done():
                    memory_task.cancel()
                    try:
                        await memory_task
                    except (asyncio.CancelledError, Exception):
                        pass

        logger.info("Bye!")

    async def process_turn(self, user_text: str) -> str:
        """Execute a single turn through full intent routing + skill dispatch.

        Used by /api/chat so the HTTP endpoint gets the same routing as the
        terminal text loop (IntentRouter → ProactiveOrchestrator → SkillDispatcher).
        Returns the response string (empty string for commands/estop).
        """
        from askme.llm.intent_router import IntentType

        self._text_audio.spoken.clear()
        memory_task = self._pipeline.start_memory_prefetch(user_text)
        try:
            intent = self._router.route(user_text)

            if intent.type == IntentType.ESTOP:
                self._pipeline.handle_estop()
                memory_task.cancel()
                return "急停已触发。"

            if intent.type == IntentType.QUICK_REPLY:
                memory_task.cancel()
                reply = intent.skill_name or ""
                self._audio.speak(reply)
                self._audio.start_playback()

                async def _stop_after_play():
                    try:
                        await asyncio.to_thread(self._audio.wait_speaking_done)
                    finally:
                        self._audio.stop_playback()

                asyncio.create_task(_stop_after_play())
                return reply

            if intent.type == IntentType.COMMAND:
                memory_task.cancel()
                return ""

            if intent.type == IntentType.VOICE_TRIGGER:
                memory_task.cancel()
                memory_task = None
                if self._dispatcher:
                    _result = await self._proactive.run(
                        intent.skill_name or "", user_text, self._text_audio,
                        source="text",
                    )
                    if _result.proceed:
                        return await self._dispatcher.dispatch(
                            intent.skill_name or "", _result.enriched_text,
                            source="text",
                        )
                    elif _result.interrupt_payload:
                        return await self._dispatcher.handle_general(
                            _result.interrupt_payload, source="text",
                        )
                    return ""
                return await self._pipeline.execute_skill(
                    intent.skill_name or "", user_text,
                )

            # GENERAL
            if self._dispatcher:
                return await self._dispatcher.handle_general(
                    user_text, source="text", memory_task=memory_task,
                )
            reply = await self._pipeline.process(user_text, memory_task=memory_task)
            memory_task = None
            return reply
        finally:
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _maybe_handle_runtime_bridge(self, user_text: str) -> bool:
        """Try the runtime bridge first and fall back locally on bridge failures."""
        if self._voice_runtime_bridge is None:
            return False

        try:
            bridge_result = await asyncio.to_thread(
                self._voice_runtime_bridge.handle_text_input,
                user_text,
            )
        except Exception as exc:
            logger.warning(
                "Text runtime bridge failed, falling back to local pipeline: %s",
                exc,
            )
            return False

        if not isinstance(bridge_result, dict) or not bridge_result.get("handled"):
            return False

        turn = bridge_result.get("turn")
        if not isinstance(turn, dict):
            logger.warning(
                "Text runtime bridge returned an invalid handled payload; "
                "falling back to local pipeline.",
            )
            return False

        action_type = turn.get("action_type")
        skill_name = turn.get("skill_name")

        # Dispatch to local skill executor when the edge service resolved a skill.
        # Covers both action_type=="skill" (SKILL) and action_type=="general" with a
        # populated skill_name field (SKILL_SUGGESTED status from the edge planner).
        if isinstance(skill_name, str) and skill_name and (
            action_type == "skill" or action_type == "general"
        ):
            if self._dispatcher:
                await self._dispatcher.dispatch(
                    skill_name, user_text, source="runtime",
                )
            else:
                await self._pipeline.execute_skill(skill_name, user_text)
            return True

        spoken_reply = turn.get("spoken_reply")
        if isinstance(spoken_reply, str) and spoken_reply.strip():
            record_external_turn(
                self._pipeline,
                user_text,
                spoken_reply.strip(),
                source="runtime",
            )
            logger.info("[Assistant]: %s", spoken_reply.strip())
            return True

        logger.warning(
            "Text runtime bridge marked the turn handled but returned no usable "
            "reply (action_type=%r skill_name=%r); falling back to local pipeline.",
            action_type,
            skill_name,
        )
        return False
