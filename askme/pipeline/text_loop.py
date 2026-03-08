"""Text-mode main loop — terminal input → intent routing → brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from askme.pipeline.external_turns import record_external_turn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from askme.brain.conversation import ConversationManager
    from askme.brain.intent_router import IntentRouter
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.pipeline.commands import CommandHandler
    from askme.skills.skill_manager import SkillManager
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.runtime_bridge import VoiceRuntimeBridge


class TextLoop:
    """Interactive text-input loop.

    Reads from stdin, routes through :class:`IntentRouter`, delegates to
    :class:`BrainPipeline` or :class:`CommandHandler`.
    """

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
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._commands = commands
        self._conversation = conversation
        self._skill_manager = skill_manager
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge

    async def run(self) -> None:
        """Block until the user types /quit or presses Ctrl+C."""
        from askme.brain.intent_router import IntentType

        logger.info("Text mode. Commands: /clear /history /skills /quit")
        logger.info("Loaded %d previous messages.", len(self._conversation.history))
        logger.info("Skills: %s", self._skill_manager.get_skill_catalog())

        idle_task = self._pipeline.start_idle_reflection()
        while True:
            memory_task: asyncio.Task[str] | None = None
            try:
                user_text = await asyncio.to_thread(input, "[You]: ")
                user_text = user_text.strip()
                if not user_text:
                    continue

                # Cancel idle reflection on user activity
                if idle_task and not idle_task.done():
                    idle_task.cancel()

                pending_reply = await self._pipeline.handle_pending_tool_response(user_text)
                if pending_reply is not None:
                    logger.info("[Assistant]: %s", pending_reply)
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
                    await self._pipeline.execute_skill(
                        intent.skill_name or "", user_text
                    )
                    continue

                if intent.type == IntentType.GENERAL and self._voice_runtime_bridge:
                    bridge_result = await asyncio.to_thread(
                        self._voice_runtime_bridge.handle_text_input,
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
                            logger.info("[Assistant]: %s", spoken_reply.strip())
                            continue

                # General → LLM (pass pre-fetched memory)
                reply = await self._pipeline.process(user_text, memory_task=memory_task)
                memory_task = None  # pipeline took ownership
                logger.info("[Assistant]: %s", reply)
                await asyncio.to_thread(self._audio.wait_speaking_done)
                self._audio.stop_playback()

                # Restart idle reflection timer
                idle_task = self._pipeline.start_idle_reflection()

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as exc:
                logger.error("Error: %s", exc)
            finally:
                if memory_task is not None and not memory_task.done():
                    memory_task.cancel()

        logger.info("Bye!")
