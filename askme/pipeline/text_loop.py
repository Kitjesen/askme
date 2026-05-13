"""Text-mode main loop 鈥?terminal input 鈫?intent routing 鈫?brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from askme.pipeline.external_turns import record_external_turn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from askme.interaction.intent_router import IntentRouter
    from askme.memory.conversation import ConversationManager
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
        self.last_cognition_result: dict[str, Any] | None = None

        from askme.pipeline.proactive import ProactiveOrchestrator
        self._proactive = ProactiveOrchestrator.default(
            pipeline=pipeline, dispatcher=dispatcher
        )
        self._text_audio = _TextClarificationAudio()

    async def run(self) -> None:
        """Block until the user types /quit or presses Ctrl+C."""
        from askme.interaction.intent_router import IntentType

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

                cognition_reply = await self._maybe_handle_cognition_turn(
                    user_text,
                    source="text",
                    speak=False,
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
                    # Cancel memory prefetch 鈥?skill path never uses the result
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    # Try runtime bridge first 鈥?edge service may route to arbiter
                    bridge_handled = await self._maybe_handle_runtime_bridge(user_text)
                    if bridge_handled:
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue
                    # Bridge not configured / failed 鈥?local skill dispatch
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
                        # Cancel the memory prefetch we started earlier 鈥?the bridge
                        # handled the turn so the prefetched context is no longer needed.
                        if memory_task and not memory_task.done():
                            memory_task.cancel()
                        memory_task = None
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                # General 鈫?LLM (pass pre-fetched memory)
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
                    print("鈿狅笍 澶氭閿欒锛岀郴缁熸殏鏃跺紓甯革紝璇风◢鍊?..")  # noqa: T201
                    await asyncio.sleep(3)
                    consecutive_errors = 0
            finally:
                if memory_task is not None and not memory_task.done():
                    memory_task.cancel()
                    try:
                        await memory_task
                    except (asyncio.CancelledError, Exception):
                        pass

        # Session-end summarization 鈥?save L2 summary if enough conversation happened
        _sm = getattr(self._pipeline, "_session_memory", None)
        if _sm and len(self._conversation.history) > 4:
            try:
                await asyncio.to_thread(_sm.summarize_and_save, self._conversation.history)
            except Exception as e:
                logger.warning("Session summary failed: %s", e)

        logger.info("Bye!")

    async def process_turn(self, user_text: str, *, speak: bool = False) -> str:
        """Execute a single turn through full intent routing + skill dispatch.

        Used by /api/chat so the HTTP endpoint gets the same routing as the
        terminal text loop (IntentRouter 鈫?ProactiveOrchestrator 鈫?SkillDispatcher).
        Returns the response string (empty string for commands/estop).
        """
        from askme.interaction.intent_router import IntentType

        self._text_audio.spoken.clear()
        self.last_cognition_result = None
        memory_task = self._pipeline.start_memory_prefetch(user_text)
        source = "voice" if speak else "text"
        try:
            intent = self._router.route(user_text)

            if intent.type == IntentType.ESTOP:
                self._pipeline.handle_estop()
                memory_task.cancel()
                return "急停已触发。"

            if intent.type == IntentType.QUICK_REPLY:
                memory_task.cancel()
                reply = intent.skill_name or ""
                if not speak:
                    return reply
                self._audio.speak(reply)
                self._audio.start_playback()
                try:
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                finally:
                    self._audio.stop_playback()
                return reply

            if intent.type == IntentType.COMMAND:
                memory_task.cancel()
                return ""

            cognition_reply = await self._maybe_handle_cognition_turn(
                user_text,
                source=source,
                speak=speak,
            )
            if cognition_reply is not None:
                memory_task.cancel()
                return cognition_reply

            if intent.type == IntentType.VOICE_TRIGGER:
                memory_task.cancel()
                memory_task = None
                if self._dispatcher:
                    _result = await self._proactive.run(
                        intent.skill_name or "", user_text, self._text_audio,
                        source=source,
                    )
                    if _result.proceed:
                        return await self._dispatcher.dispatch(
                            intent.skill_name or "", _result.enriched_text,
                            source=source,
                        )
                    elif _result.interrupt_payload:
                        return await self._dispatcher.handle_general(
                            _result.interrupt_payload, source=source,
                        )
                    return ""
                return await self._pipeline.execute_skill(
                    intent.skill_name or "", user_text, source=source,
                )

            # GENERAL
            if self._dispatcher:
                return await self._dispatcher.handle_general(
                    user_text, source=source, memory_task=memory_task,
                )
            reply = await self._pipeline.process(
                user_text,
                memory_task=memory_task,
                source=source,
            )
            memory_task = None
            return reply
        finally:
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
    ) -> str | None:
        """Route robot task planning turns into cognition when available."""
        cognition = self._cognition_handler
        if cognition is None or not self._should_route_to_cognition(user_text):
            return None

        payload: dict[str, Any] = {
            "text": user_text,
            "channel": f"{source}-chat",
        }
        if self._active_planning_session_id:
            payload["planning_session_id"] = self._active_planning_session_id

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
        if session_id and stage not in {"cancelled", "ready_for_arbiter"}:
            self._active_planning_session_id = session_id
        elif stage in {"cancelled", "ready_for_arbiter"}:
            self._active_planning_session_id = None

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
        )
        if speak:
            await self._speak_reply(reply)
        return reply

    def _should_route_to_cognition(self, user_text: str) -> bool:
        if self._active_planning_session_id:
            return True
        return _looks_like_cognition_request(user_text)

    async def _speak_reply(self, reply: str) -> None:
        if not reply.strip():
            return
        self._audio.speak(reply.strip())
        self._audio.start_playback()
        try:
            await asyncio.to_thread(self._audio.wait_speaking_done)
        finally:
            self._audio.stop_playback()

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


_COGNITION_KEYWORDS = (
    "巡检", "检查", "巡逻", "拍照", "截图", "抓拍", "取证",
    "状态", "电量", "导航", "去", "到", "拿", "抓", "夹取",
    "急停", "停止", "停下",
    "patrol", "inspect", "photo", "snapshot", "capture", "status",
    "battery", "navigate", "go to", "pick", "grab", "stop", "estop", "e-stop",
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
