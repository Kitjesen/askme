"""Voice-mode main loop — microphone → intent routing → brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from askme.pipeline.external_turns import record_external_turn

if TYPE_CHECKING:
    from askme.brain.intent_router import IntentRouter
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.pipeline.skill_dispatcher import SkillDispatcher
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.runtime_bridge import VoiceRuntimeBridge

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge
        self._dispatcher = dispatcher

    async def run(self) -> None:
        """Block until Ctrl+C or too many consecutive errors."""
        from askme.brain.intent_router import IntentType

        logger.info("Voice mode active. Say something! (Ctrl+C to quit)")

        consecutive_errors = 0
        idle_task = self._pipeline.start_idle_reflection()
        while True:
            memory_task: asyncio.Task[str] | None = None
            try:
                user_text = await asyncio.to_thread(self._audio.listen_loop)
                if not user_text:
                    continue

                consecutive_errors = 0

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
                        self._audio.wait_speaking_done()
                        self._audio.stop_playback()
                    elif _muted_intent.type == IntentType.COMMAND:
                        pass  # fall through to COMMAND handler below
                    else:
                        continue

                # Immediate audio feedback — user knows we heard them
                # Fires before LLM call to fill the latency gap
                self._audio.acknowledge()

                # Cancel idle reflection on user activity
                if idle_task and not idle_task.done():
                    idle_task.cancel()

                pending_reply = await self._pipeline.handle_pending_tool_response(user_text)
                if pending_reply is not None:
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Start memory prefetch ASAP (overlaps with routing)
                memory_task = self._pipeline.start_memory_prefetch(user_text)

                intent = self._router.route(user_text)

                if intent.type == IntentType.ESTOP:
                    self._pipeline.handle_estop()
                    self._audio.drain_buffers()  # stop any ongoing TTS immediately
                    self._audio.speak("已紧急停止。")
                    self._audio.start_playback()
                    self._audio.wait_speaking_done()
                    self._audio.stop_playback()
                    continue

                # ── Stop speaking — zero latency, no LLM ─────────────────
                if (
                    intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name == "stop_speaking"
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    self._audio.drain_buffers()
                    # acknowledge already fired — no extra chime needed
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
                    self._audio.wait_speaking_done()
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
                    self._audio.wait_speaking_done()
                    self._audio.stop_playback()
                    continue

                if intent.type == IntentType.VOICE_TRIGGER:
                    # Cancel memory prefetch — skill path never uses the result
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                    memory_task = None
                    # Try runtime bridge first — edge service may route to arbiter
                    bridge_handled = await self._maybe_handle_runtime_bridge(user_text)
                    if bridge_handled:
                        idle_task = self._pipeline.start_idle_reflection()
                        continue
                    # Bridge not configured / failed — local skill dispatch
                    if self._dispatcher:
                        await self._dispatcher.dispatch(
                            intent.skill_name or "", user_text, source="voice",
                        )
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
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                # General → LLM (pass pre-fetched memory)
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
                idle_task = self._pipeline.start_idle_reflection()

            except KeyboardInterrupt:
                break
            except Exception as exc:
                consecutive_errors += 1
                logger.error("Voice loop error: %s", exc)
                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "Voice loop degraded: %d consecutive errors, pausing 5s",
                        consecutive_errors,
                    )
                    try:
                        await self._audio.speak("系统暂时遇到问题，请稍候。")
                    except Exception:
                        pass
                    await asyncio.sleep(5)
                    consecutive_errors = 0
                await asyncio.sleep(1)
            finally:
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

        if action_type == "skill" and isinstance(skill_name, str) and skill_name:
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
            "reply; falling back to local pipeline.",
        )
        return False
