"""Voice-mode main loop — microphone → intent routing → brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from askme.pipeline.external_turns import record_external_turn

if TYPE_CHECKING:
    from askme.brain.intent_router import IntentRouter
    from askme.pipeline.brain_pipeline import BrainPipeline
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
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge

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
                    self._audio.speak("已紧急停止。")
                    self._audio.start_playback()
                    self._audio.wait_speaking_done()
                    self._audio.stop_playback()
                    continue

                if intent.type == IntentType.VOICE_TRIGGER:
                    await self._pipeline.execute_skill(
                        intent.skill_name or "", user_text
                    )
                    continue

                if intent.type == IntentType.COMMAND:
                    if intent.command in ("quit", "exit", "/quit", "/exit"):
                        logger.info("Exit command received in voice mode.")
                        break
                    # Other commands (/clear, /help, etc.) fall through to LLM
                    # so the assistant can respond naturally by voice

                if intent.type == IntentType.GENERAL and self._voice_runtime_bridge:
                    bridge_result = await asyncio.to_thread(
                        self._voice_runtime_bridge.handle_voice_text,
                        user_text,
                    )
                    if bridge_result and bridge_result.get("handled"):
                        turn = bridge_result.get("turn", {})
                        action_type = turn.get("action_type")
                        skill_name = turn.get("skill_name")

                        if action_type == "skill" and isinstance(skill_name, str) and skill_name:
                            await self._pipeline.execute_skill(skill_name, user_text)
                            continue

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
                            continue

                # General → LLM (pass pre-fetched memory)
                await self._pipeline.process(user_text, memory_task=memory_task)
                memory_task = None  # pipeline took ownership

                # Restart idle reflection timer
                idle_task = self._pipeline.start_idle_reflection()

            except KeyboardInterrupt:
                break
            except Exception as exc:
                consecutive_errors += 1
                logger.error("Voice loop error: %s", exc)
                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.critical("Too many consecutive errors, exiting.")
                    break
                await asyncio.sleep(1)
            finally:
                # Always clean up dangling memory task
                if memory_task is not None and not memory_task.done():
                    memory_task.cancel()

        logger.info("Bye!")
