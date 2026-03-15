"""Voice-mode main loop — microphone → intent routing → brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from askme.pipeline.external_turns import record_external_turn
from askme.pipeline.trace import get_tracer
from askme.voice.address_detector import AddressDetector
from askme.voice.audio_router import AudioErrorKind, AudioRouter

if TYPE_CHECKING:
    from askme.brain.intent_router import IntentRouter
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.pipeline.skill_dispatcher import SkillDispatcher
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.runtime_bridge import VoiceRuntimeBridge

logger = logging.getLogger(__name__)

# Skills that can execute even while an agent_task is running in background.
# These are stateless/zero-cost and cannot conflict with the agent's work.
_AGENT_BYPASS_SKILLS: frozenset[str] = frozenset([
    "get_time", "volume_up", "volume_down", "volume_reset",
    "speed_up", "speed_down", "speed_reset",
    "repeat_last", "mute_mic", "unmute_mic",
])


class VoiceLoop:
    """Continuous voice-input loop.

    Listens via :class:`AudioAgent`, routes through :class:`IntentRouter`,
    delegates to :class:`BrainPipeline`.
    """

    MAX_CONSECUTIVE_ERRORS = 3

    def __init__(
        self,
        *,
        router: IntentRouter,
        pipeline: BrainPipeline,
        audio: AudioAgent,
        voice_runtime_bridge: VoiceRuntimeBridge | None = None,
        dispatcher: SkillDispatcher | None = None,
        audio_router: AudioRouter | None = None,
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge
        self._dispatcher = dispatcher
        self._audio_router = audio_router

        from askme.pipeline.proactive import ProactiveOrchestrator
        self._proactive = ProactiveOrchestrator.default(
            pipeline=pipeline, dispatcher=dispatcher
        )
        self._address_detector = AddressDetector()  # default disabled; app.py wires the real one

    def set_address_detector(self, detector: AddressDetector) -> None:
        """Wire the address detector after construction."""
        self._address_detector = detector

    async def run(self) -> None:
        """Block until Ctrl+C or too many consecutive errors."""
        from askme.brain.intent_router import IntentType

        logger.info("Voice mode active. Say something! (Ctrl+C to quit)")

        consecutive_errors = 0
        idle_task = self._pipeline.start_idle_reflection()
        _tracer = get_tracer()
        while True:
            memory_task: asyncio.Task[str] | None = None
            _trace = None
            try:
                # Tell the noise filter whether we're waiting for a
                # confirmation so that words like "好的"/"不" pass through.
                # Also detect when the last assistant message was a question
                # (ended with ？or ?) — the user's short reply is likely an answer.
                _last = self._pipeline.last_spoken_text or ""
                _ends_with_question = _last.rstrip().endswith(("？", "?"))
                self._audio.awaiting_confirmation = (
                    self._pipeline.has_pending_tool_approval()
                    or _ends_with_question
                )

                user_text = await asyncio.to_thread(self._audio.listen_loop)
                if not user_text:
                    continue

                consecutive_errors = 0

                # Start pipeline trace for this turn
                _trace = _tracer.start_trace("voice_turn")
                _trace.metadata["user_text"] = user_text[:60]

                # ── Muted state gate ──────────────────────────────────────
                # When muted, only the unmute_mic voice trigger and COMMAND
                # (quit/exit) pass through. Everything else is silently discarded.
                if self._audio.is_muted:
                    _muted_intent = self._router.route(user_text)
                    if (
                        _muted_intent.type == IntentType.VOICE_TRIGGER
                        and _muted_intent.skill_name == "unmute_mic"
                    ):
                        self._audio.unmute()
                        self._audio.acknowledge()
                        self._audio.speak("好的，重新开启。")
                        self._audio.start_playback()
                        await asyncio.to_thread(self._audio.wait_speaking_done)
                        self._audio.stop_playback()
                    elif _muted_intent.type == IntentType.COMMAND:
                        pass  # fall through to COMMAND handler below
                    else:
                        continue

                # Address detection: skip if not talking to the robot
                if not self._address_detector.is_addressed(user_text):
                    logger.info("Address filter: '%s' not for robot, ignoring", user_text[:30])
                    continue

                # Immediate audio feedback — user knows we heard them
                # Fires before LLM call to fill the latency gap
                self._audio.acknowledge()

                # Cancel idle reflection on user activity
                if idle_task and not idle_task.done():
                    idle_task.cancel()

                pending_reply = await self._pipeline.handle_pending_tool_response(user_text)
                if pending_reply is not None:
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Start memory prefetch ASAP (overlaps with routing)
                memory_task = self._pipeline.start_memory_prefetch(user_text)

                with _tracer.span("intent_route"):
                    intent = self._router.route(user_text)

                if intent.type == IntentType.ESTOP:
                    # Cancel any background agent task before hard stop
                    if self._dispatcher:
                        self._dispatcher.cancel_active_agent_task()
                    self._pipeline.handle_estop()
                    self._audio.drain_buffers()  # stop any ongoing TTS immediately
                    self._audio.speak("已紧急停止。")
                    self._audio.start_playback()
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                    self._audio.stop_playback()
                    continue

                # ── Stop speaking — also cancels any active agent task ────────
                if (
                    intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name == "stop_speaking"
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    if self._dispatcher and self._dispatcher.cancel_active_agent_task():
                        self._audio.drain_buffers()
                        self._audio.speak("已取消任务。")
                        self._audio.start_playback()
                        await asyncio.to_thread(self._audio.wait_speaking_done)
                        self._audio.stop_playback()
                    else:
                        self._audio.drain_buffers()
                    # acknowledge already fired — no extra chime needed
                    continue

                # ── Repeat last response — zero LLM, replay TTS ──────────
                if (
                    intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name == "repeat_last"
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    last = self._pipeline.last_spoken_text
                    self._audio.drain_buffers()
                    if last:
                        self._audio.speak(last)
                    else:
                        self._audio.speak("暂时没有内容可以重复。")
                    self._audio.start_playback()
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                    self._audio.stop_playback()
                    continue

                # ── Mute mic — zero latency, no LLM ──────────────────────
                if (
                    intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name == "mute_mic"
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    self._audio.drain_buffers()
                    self._audio.mute()
                    self._audio.speak('好的，已关闭麦克风。说"开麦"来重新打开。')
                    self._audio.start_playback()
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                    self._audio.stop_playback()
                    continue

                # ── Volume / speed — zero latency, no LLM ────────────────
                _vol_speed_skill = intent.skill_name if intent.type == IntentType.VOICE_TRIGGER else None
                if _vol_speed_skill in (
                    "volume_up", "volume_down", "volume_reset",
                    "speed_up", "speed_down", "speed_reset",
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    self._audio.drain_buffers()
                    if _vol_speed_skill == "volume_up":
                        v = self._audio.adjust_volume(+0.2)
                        msg = f"好的，音量已调大，当前{int(v * 100)}%。"
                    elif _vol_speed_skill == "volume_down":
                        v = self._audio.adjust_volume(-0.2)
                        msg = f"好的，音量已调小，当前{int(v * 100)}%。"
                    elif _vol_speed_skill == "volume_reset":
                        self._audio.set_volume(1.0)
                        msg = "好的，已恢复默认音量。"
                    elif _vol_speed_skill == "speed_up":
                        s = self._audio.adjust_speed(+0.3)
                        msg = f"好的，语速已加快，当前{s:.1f}倍。"
                    elif _vol_speed_skill == "speed_down":
                        s = self._audio.adjust_speed(-0.3)
                        msg = f"好的，语速已降低，当前{s:.1f}倍。"
                    else:  # speed_reset
                        self._audio.set_speed(1.0)
                        msg = "好的，已恢复默认语速。"
                    self._audio.speak(msg)
                    self._audio.start_playback()
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                    self._audio.stop_playback()
                    continue

                # ── Agent-busy gate ───────────────────────────────────────────
                # While a background agent_task is running, block new skill
                # dispatches and LLM turns to prevent audio conflicts.
                # ESTOP and stop_speaking are handled above and always pass through.
                # Lightweight skills (get_time, volume, etc.) bypass the gate.
                if (
                    self._dispatcher
                    and self._dispatcher.has_active_agent_task
                ):
                    _bypass = (
                        intent.type == IntentType.VOICE_TRIGGER
                        and intent.skill_name in _AGENT_BYPASS_SKILLS
                    )
                    if not _bypass:
                        if memory_task and not memory_task.done():
                            memory_task.cancel()
                            memory_task = None
                        self._audio.speak("正在处理中，说够了可取消。")
                        self._audio.start_playback()
                        await asyncio.to_thread(self._audio.wait_speaking_done)
                        self._audio.stop_playback()
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
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
                        result = await self._proactive.run(
                            intent.skill_name or "", user_text, self._audio,
                            source="voice",
                        )
                        if result.proceed:
                            await self._dispatcher.dispatch(
                                intent.skill_name or "", result.enriched_text,
                                source="voice",
                            )
                        elif result.interrupt_payload:
                            # User bailed out and issued a new intent in the same breath
                            # e.g. "算了，去仓库B" → reroute immediately without re-listening
                            logger.info(
                                "VoiceLoop: rerouting interrupt_payload: %r",
                                result.interrupt_payload,
                            )
                            _reroute_intent = self._router.route(result.interrupt_payload)
                            if (
                                _reroute_intent.type == IntentType.VOICE_TRIGGER
                                and _reroute_intent.skill_name
                            ):
                                _rr = await self._proactive.run(
                                    _reroute_intent.skill_name,
                                    result.interrupt_payload,
                                    self._audio,
                                    source="voice",
                                )
                                if _rr.proceed:
                                    await self._dispatcher.dispatch(
                                        _reroute_intent.skill_name,
                                        _rr.enriched_text,
                                        source="voice",
                                    )
                            else:
                                # Rerouted to a general intent — start fresh memory
                                # prefetch for the new payload so LLM gets context.
                                memory_task = self._pipeline.start_memory_prefetch(
                                    result.interrupt_payload
                                )
                                await self._dispatcher.handle_general(
                                    result.interrupt_payload,
                                    source="voice",
                                    memory_task=memory_task,
                                )
                                memory_task = None  # handle_general took ownership
                                if idle_task and not idle_task.done():
                                    idle_task.cancel()
                                idle_task = self._pipeline.start_idle_reflection()
                    else:
                        await self._pipeline.execute_skill(
                            intent.skill_name or "", user_text,
                        )
                    continue

                if intent.type == IntentType.COMMAND:
                    if intent.command in ("quit", "exit", "/quit", "/exit"):
                        logger.info("Exit command received in voice mode.")
                        break
                    # Other commands (/clear, /help, etc.) fall through to LLM
                    # so the assistant can respond naturally by voice

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
                with _tracer.span("llm_pipeline"):
                    if self._dispatcher:
                        await self._dispatcher.handle_general(
                            user_text, source="voice", memory_task=memory_task,
                        )
                    else:
                        await self._pipeline.process(user_text, memory_task=memory_task)
                memory_task = None  # pipeline took ownership

                # Don't block on wait_speaking_done — echo gate in listen_loop
                # suppresses speaker echo while allowing user barge-in.

                # Restart idle reflection timer
                if idle_task and not idle_task.done():
                    idle_task.cancel()
                idle_task = self._pipeline.start_idle_reflection()

            except KeyboardInterrupt:
                break
            except Exception as exc:
                kind = AudioRouter.classify_error(exc)

                # ── XRUN: silent retry, no user notification ──────────────
                # Buffer overrun after aplay finishes — expected on half-duplex
                # ALSA hardware.  The AudioRouter ownership model prevents most
                # XRUNs; the ones that slip through are recoverable silently.
                if kind == AudioErrorKind.XRUN:
                    logger.debug("Voice loop: XRUN (stream reset): %s", exc)
                    await asyncio.sleep(0.1)
                    continue

                # ── DEVICE_BUSY: short backoff, silent retry ───────────────
                if kind == AudioErrorKind.DEVICE_BUSY:
                    logger.warning("Voice loop: audio device busy — retrying in 2s: %s", exc)
                    await asyncio.sleep(2.0)
                    continue

                # ── TTS_FAIL: mic unaffected, retry quickly ───────────────
                if kind == AudioErrorKind.TTS_FAIL:
                    logger.error("Voice loop: TTS backend error: %s", exc)
                    await asyncio.sleep(0.5)
                    continue

                # ── DEVICE_LOST: notify user once, long backoff ───────────
                if kind == AudioErrorKind.DEVICE_LOST:
                    logger.error("Voice loop: audio device lost: %s", exc)
                    consecutive_errors += 1
                    if consecutive_errors == 1:
                        try:
                            self._audio.tts.speak("麦克风断开，正在重连。")
                        except Exception:
                            pass
                    await asyncio.sleep(5.0)
                    continue

                # ── UNKNOWN: standard consecutive-error escalation ────────
                consecutive_errors += 1
                logger.error("Voice loop error [%s]: %s", kind.value, exc)
                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "Voice loop degraded: %d consecutive errors, pausing 5s",
                        consecutive_errors,
                    )
                    try:
                        self._audio.tts.speak("系统暂时遇到问题，请稍候。")
                    except Exception:
                        pass
                    await asyncio.sleep(5)
                    consecutive_errors = 0
                await asyncio.sleep(1)
            finally:
                # Finish pipeline trace for this turn
                if _trace is not None:
                    _tracer.finish_trace()
                # Always clean up dangling memory task.
                # Await after cancel to suppress "Task exception was never
                # retrieved" GC warnings that mask real errors in log output.
                if memory_task is not None and not memory_task.done():
                    memory_task.cancel()
                    try:
                        await memory_task
                    except (asyncio.CancelledError, Exception):
                        pass

        logger.info("Bye!")

    def _slot_present(self, skill: "SkillDefinition", user_text: str) -> bool:  # type: ignore[name-defined]
        """Proxy to slot_utils.slot_present — kept for backward compatibility with tests."""
        from askme.pipeline.proactive.slot_utils import slot_present
        return slot_present(skill, user_text, self._pipeline)

    async def _maybe_handle_runtime_bridge(self, user_text: str) -> bool:
        """Try the runtime bridge first and fall back locally on bridge failures."""
        if self._voice_runtime_bridge is None:
            return False

        try:
            bridge_result = await asyncio.to_thread(
                self._voice_runtime_bridge.handle_voice_text,
                user_text,
            )
        except Exception as exc:
            logger.warning(
                "Voice runtime bridge failed, falling back to local pipeline: %s",
                exc,
            )
            return False

        if not isinstance(bridge_result, dict) or not bridge_result.get("handled"):
            return False

        turn = bridge_result.get("turn")
        if not isinstance(turn, dict):
            logger.warning(
                "Voice runtime bridge returned an invalid handled payload; "
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
            self._audio.speak(spoken_reply.strip())
            self._audio.start_playback()
            await asyncio.to_thread(self._audio.wait_speaking_done)
            self._audio.stop_playback()
            return True

        logger.warning(
            "Voice runtime bridge marked the turn handled but returned no usable "
            "reply (action_type=%r skill_name=%r); falling back to local pipeline.",
            action_type,
            skill_name,
        )
        return False
