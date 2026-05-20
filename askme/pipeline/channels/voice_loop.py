"""Voice-mode main loop -microphone ->intent routing ->brain pipeline."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from askme.contracts.adapters import (
    interaction_decision_to_action_decision,
    perception_snapshot_to_input,
)
from askme.pipeline.channels.runtime_bridge_calls import try_handle_runtime_bridge_turn
from askme.pipeline.core.trace import get_tracer
from askme.ports import AudioFrontendPort, AudioRouterPort, VoiceTurnBridgePort
from askme.robot_interaction import attach_intent_route_trace
from askme.robot_interaction.interaction_gate import (
    InteractionAction,
    InteractionDecision,
)
from askme.robot_interaction.perception_context import InteractionPerceptionSnapshot

if TYPE_CHECKING:
    from askme.pipeline.core.brain_pipeline import BrainPipeline
    from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
    from askme.robot_interaction import IntentRouter
    from askme.skills.core.skill_model import SkillDefinition

logger = logging.getLogger(__name__)


class AddressDetectorProtocol(Protocol):
    def is_addressed(self, text: str) -> bool:
        """Return whether recognized speech is addressed to the robot."""


class InteractionGateProtocol(Protocol):
    def evaluate(
        self,
        text: str,
        *,
        addressed: bool = True,
        perception: InteractionPerceptionSnapshot | dict[str, Any] | None = None,
    ) -> InteractionDecision:
        """Decide whether a voice turn should continue to the brain."""


class _DefaultAddressDetector:
    def is_addressed(self, text: str) -> bool:
        return True


class _DefaultInteractionGate:
    def evaluate(
        self,
        text: str,
        *,
        addressed: bool = True,
        perception: InteractionPerceptionSnapshot | dict[str, Any] | None = None,
    ) -> InteractionDecision:
        return InteractionDecision(InteractionAction.RESPOND, "gate_disabled", 1.0)


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
        audio: AudioFrontendPort,
        voice_runtime_bridge: VoiceTurnBridgePort | None = None,
        dispatcher: SkillDispatcher | None = None,
        audio_router: AudioRouterPort | None = None,
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
        self._address_detector: AddressDetectorProtocol = _DefaultAddressDetector()
        self._interaction_gate: InteractionGateProtocol = _DefaultInteractionGate()
        self._interaction_perception_provider: Callable[[], Any] | None = None
        self._last_interaction_decision: dict[str, Any] | None = None
        self._last_interaction_perception: dict[str, Any] | None = None
        self._last_input_contract: dict[str, Any] | None = None
        self._last_action_contract: dict[str, Any] | None = None
        self._conversation_session_id: str | None = None
        self._degraded_conversation_session_id: str | None = None

    def set_address_detector(self, detector: AddressDetectorProtocol) -> None:
        """Wire the address detector after construction."""
        self._address_detector = detector

    def set_interaction_gate(self, gate: InteractionGateProtocol) -> None:
        """Wire the real-world interaction gate after construction."""
        self._interaction_gate = gate

    def set_interaction_perception_provider(self, provider: Callable[[], Any] | None) -> None:
        """Wire a best-effort provider for vision/audio-source/pose context."""
        self._interaction_perception_provider = provider

    async def run(self) -> None:
        """Block until Ctrl+C or too many consecutive errors."""
        from askme.robot_interaction import IntentType

        logger.info("Voice mode active. Say something! (Ctrl+C to quit)")

        consecutive_errors = 0
        idle_task = self._pipeline.start_idle_reflection()
        _tracer = get_tracer()
        while True:
            memory_task: asyncio.Task[str] | None = None
            _trace = None
            try:
                # Tell the noise filter whether we're waiting for a
                # confirmation so short replies can pass through.
                # If the last assistant message was a question, the user's
                # short reply is likely an answer.
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

                # Muted state gate
                # When muted, only the unmute_mic voice trigger and COMMAND
                # (quit/exit) pass through. Everything else is silently discarded.
                if self._audio.is_muted:
                    _muted_intent = self._router.route(user_text)
                    attach_intent_route_trace(
                        _trace,
                        _muted_intent,
                        source="voice",
                        stage="muted_gate_route",
                    )
                    if (
                        _muted_intent.type == IntentType.VOICE_TRIGGER
                        and _muted_intent.skill_name == "unmute_mic"
                    ):
                        self._audio.unmute()
                        self._audio.acknowledge()
                        await self._audio.speak_and_wait("好的，已重新开启。")
                    elif _muted_intent.type == IntentType.COMMAND:
                        pass  # fall through to COMMAND handler below
                    else:
                        continue

                # Interaction gate: separate ambient speech from real user turns.
                addressed = self._address_detector.is_addressed(user_text)
                perception_snapshot = self._get_interaction_perception()
                gate_decision = self._interaction_gate.evaluate(
                    user_text,
                    addressed=addressed,
                    perception=perception_snapshot,
                )
                self._last_interaction_decision = _decision_to_dict(
                    gate_decision,
                    addressed=addressed,
                )
                self._last_interaction_perception = _snapshot_to_dict(perception_snapshot)
                input_contract = perception_snapshot_to_input(
                    perception_snapshot,
                    transcript=user_text,
                    addressed=addressed,
                )
                action_contract = interaction_decision_to_action_decision(
                    gate_decision,
                    user_text=user_text,
                    addressed=addressed,
                    perception=perception_snapshot,
                )
                self._last_input_contract = input_contract.to_dict()
                self._last_action_contract = action_contract.to_dict()
                _trace.metadata["interaction_gate"] = {
                    "action": gate_decision.action.value,
                    "reason": gate_decision.reason,
                    "confidence": gate_decision.confidence,
                    "addressed": addressed,
                    "perception": self._last_interaction_perception,
                }
                _trace.metadata["product_contract"] = {
                    "perception_input": self._last_input_contract,
                    "action_decision": self._last_action_contract,
                }
                if gate_decision.action in (
                    InteractionAction.IGNORE,
                    InteractionAction.RECORD_ONLY,
                ):
                    self._record_environment_speech(user_text, gate_decision)
                    continue
                if gate_decision.action in (
                    InteractionAction.CLARIFY,
                    InteractionAction.DEFER,
                    InteractionAction.REFUSE,
                ):
                    self._record_environment_speech(user_text, gate_decision)
                    if gate_decision.reply:
                        await self._audio.speak_and_wait(gate_decision.reply)
                    continue

                # Immediate audio feedback -user knows we heard them
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

                with _tracer.span("intent_route") as _route_span:
                    intent = self._router.route(user_text)
                    _route_span.metadata.update(
                        attach_intent_route_trace(_trace, intent, source="voice")
                    )

                if intent.type == IntentType.ESTOP:
                    # Cancel any background agent task before hard stop
                    if self._dispatcher:
                        self._dispatcher.cancel_active_agent_task()
                    self._pipeline.handle_estop()
                    self._audio.drain_buffers()  # stop any ongoing TTS immediately
                    await self._audio.speak_and_wait("已紧急停止。")
                    continue

                # Quick reply -zero LLM, instant response
                if intent.type == IntentType.QUICK_REPLY:
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    _quick_text = intent.reply_text or intent.skill_name or "好的。"
                    self._audio.drain_buffers()
                    await self._audio.speak_and_wait(_quick_text)
                    self._pipeline._turn_executor._last_spoken_text = _quick_text
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Stop speaking -also cancels any active agent task
                if (
                    intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name == "stop_speaking"
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    if self._dispatcher and self._dispatcher.cancel_active_agent_task():
                        self._audio.drain_buffers()
                        await self._audio.speak_and_wait("已取消任务。")
                    else:
                        self._audio.drain_buffers()
                    # acknowledge already fired -no extra chime needed
                    continue

                # Repeat last response -zero LLM, replay TTS
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
                        await self._audio.speak_and_wait(last)
                    else:
                        await self._audio.speak_and_wait("暂时没有内容可以重复。")
                    continue

                # Mute mic -zero latency, no LLM
                if (
                    intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name == "mute_mic"
                ):
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    self._audio.drain_buffers()
                    self._audio.mute()
                    await self._audio.speak_and_wait('好的，已关闭麦克风。说"开麦"来重新打开。')
                    continue

                # Volume / speed -zero latency, no LLM
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
                        msg = f"好的，音量已调大，当前 {int(v * 100)}%。"
                    elif _vol_speed_skill == "volume_down":
                        v = self._audio.adjust_volume(-0.2)
                        msg = f"好的，音量已调小，当前 {int(v * 100)}%。"
                    elif _vol_speed_skill == "volume_reset":
                        self._audio.set_volume(1.0)
                        msg = "好的，已恢复默认音量。"
                    elif _vol_speed_skill == "speed_up":
                        s = self._audio.adjust_speed(+0.3)
                        msg = f"好的，语速已加快，当前 {s:.1f} 倍。"
                    elif _vol_speed_skill == "speed_down":
                        s = self._audio.adjust_speed(-0.3)
                        msg = f"好的，语速已降低，当前 {s:.1f} 倍。"
                    else:  # speed_reset
                        self._audio.set_speed(1.0)
                        msg = "好的，已恢复默认语速。"
                    await self._audio.speak_and_wait(msg)
                    continue

                # Agent-busy gate
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
                        await self._audio.speak_and_wait("正在处理中，说够了可取消。")
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
                    bridge_handled = await self._maybe_handle_runtime_bridge(user_text)
                    if bridge_handled:
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue
                    # Bridge not configured / failed -local skill dispatch
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
                            # e.g. "算了，去仓库B" -> reroute immediately without re-listening
                            logger.info(
                                "VoiceLoop: rerouting interrupt_payload: %r",
                                result.interrupt_payload,
                            )
                            _reroute_intent = self._router.route(result.interrupt_payload)
                            attach_intent_route_trace(
                                _trace,
                                _reroute_intent,
                                source="voice",
                                stage="interrupt_reroute",
                            )
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
                                # Rerouted to a general intent -start fresh memory
                                # prefetch for the new payload so LLM gets context.
                                memory_task = self._pipeline.start_memory_prefetch(
                                    result.interrupt_payload
                                )
                                conversation_session_id = self._conversation_session_for()
                                await self._dispatcher.handle_general(
                                    result.interrupt_payload,
                                    source="voice",
                                    memory_task=memory_task,
                                    conversation_session_id=conversation_session_id,
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
                conversation_session_id = self._conversation_session_for()
                assistant_reply: Any = None
                with _tracer.span("llm_pipeline"):
                    if self._dispatcher:
                        assistant_reply = await self._dispatcher.handle_general(
                            user_text,
                            source="voice",
                            memory_task=memory_task,
                            conversation_session_id=conversation_session_id,
                        )
                    else:
                        assistant_reply = await self._pipeline.process(
                            user_text,
                            memory_task=memory_task,
                            conversation_session_id=conversation_session_id,
                        )
                self._record_local_gateway_turn(
                    conversation_session_id,
                    user_text,
                    assistant_reply,
                )
                memory_task = None  # pipeline took ownership

                # Don't block on wait_speaking_done -echo gate in listen_loop
                # suppresses speaker echo while allowing user barge-in.

                # Restart idle reflection timer
                if idle_task and not idle_task.done():
                    idle_task.cancel()
                idle_task = self._pipeline.start_idle_reflection()

            except KeyboardInterrupt:
                break
            except Exception as exc:
                kind = self._classify_audio_error(exc)
                kind_value = _audio_error_value(kind)

                # XRUN: silent retry, no user notification
                # Buffer overrun after aplay finishes -expected on half-duplex
                # ALSA hardware.  The AudioRouter ownership model prevents most
                # XRUNs; the ones that slip through are recoverable silently.
                if kind_value == "xrun":
                    logger.debug("Voice loop: XRUN (stream reset): %s", exc)
                    await asyncio.sleep(0.1)
                    continue

                # DEVICE_BUSY: short backoff, silent retry
                if kind_value == "device_busy":
                    logger.warning("Voice loop: audio device busy -retrying in 2s: %s", exc)
                    await asyncio.sleep(2.0)
                    continue

                # TTS_FAIL: mic unaffected, retry quickly
                if kind_value == "tts_fail":
                    logger.error("Voice loop: TTS backend error: %s", exc)
                    await asyncio.sleep(0.5)
                    continue

                # DEVICE_LOST: notify user once, long backoff
                if kind_value == "device_lost":
                    logger.error("Voice loop: audio device lost: %s", exc)
                    consecutive_errors += 1
                    if consecutive_errors == 1:
                        try:
                            self._audio.tts.speak("麦克风断开，正在重连。")
                        except Exception:
                            pass
                    await asyncio.sleep(5.0)
                    continue

                # UNKNOWN: standard consecutive-error escalation
                consecutive_errors += 1
                logger.error("Voice loop error [%s]: %s", kind_value, exc)
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

        # Session-end summarization -save L2 summary if enough conversation happened
        _sm = getattr(self._pipeline, "_session_memory", None)
        _conv = getattr(self._pipeline, "_conversation", None)
        if _sm and _conv and len(_conv.history) > 4:
            try:
                await asyncio.to_thread(_sm.summarize_and_save, _conv.history)
            except Exception as e:
                logger.warning("Session summary failed: %s", e)

        logger.info("Bye!")

    def _slot_present(self, skill: SkillDefinition, user_text: str) -> bool:
        """Proxy to slot_utils.slot_present -kept for backward compatibility with tests."""
        from askme.pipeline.proactive.slot_utils import slot_present
        return slot_present(skill, user_text, self._pipeline)

    def _record_environment_speech(
        self,
        user_text: str,
        decision: InteractionDecision,
    ) -> None:
        """Best-effort ambient speech record without creating a chat turn."""
        logger.info(
            "InteractionGate: %s (%s, %.2f): %r",
            decision.action.value,
            decision.reason,
            decision.confidence,
            user_text[:80],
        )
        if not decision.should_record_environment:
            return
        episodic = getattr(self._pipeline, "_episodic", None)
        log = getattr(episodic, "log", None)
        if callable(log):
            log(
                "perception",
                (
                    "ambient_speech "
                    f"action={decision.action.value} "
                    f"reason={decision.reason} "
                    f"text={user_text[:120]}"
                ),
            )

    def _get_interaction_perception(self) -> InteractionPerceptionSnapshot | dict[str, Any] | None:
        if self._interaction_perception_provider is None:
            return None
        try:
            return self._interaction_perception_provider()
        except Exception as exc:
            logger.debug("Interaction perception provider failed: %s", exc)
            return InteractionPerceptionSnapshot.unknown("provider_error")

    def interaction_status_snapshot(self) -> dict[str, Any]:
        return {
            "last_decision": dict(self._last_interaction_decision or {}),
            "last_perception": dict(self._last_interaction_perception or {}),
            "last_input_contract": dict(self._last_input_contract or {}),
            "last_action_contract": dict(self._last_action_contract or {}),
        }

    def _classify_audio_error(self, exc: BaseException) -> Any:
        if self._audio_router is None:
            return "unknown"
        classifier = getattr(self._audio_router, "classify_error", None)
        if not callable(classifier):
            return "unknown"
        return classifier(exc)

    async def _maybe_handle_runtime_bridge(self, user_text: str) -> bool:
        """Try the runtime bridge first and fall back locally on bridge failures."""
        if self._voice_runtime_bridge is None:
            return False

        conversation_session_id = self._conversation_session_for()
        try:
            return await try_handle_runtime_bridge_turn(
                self._voice_runtime_bridge.handle_voice_text,
                user_text,
                conversation_session_id=conversation_session_id,
                pipeline=self._pipeline,
                dispatcher=self._dispatcher,
                on_spoken_reply=self._audio.speak_and_wait,
                label="Voice",
            )
        except Exception as exc:
            logger.warning("VoiceLoop: runtime bridge failed, falling back locally: %s", exc)
            return False

    def _conversation_session_for(self) -> str | None:
        if self._conversation_session_id and self._cached_session_is_active(
            self._conversation_session_id
        ):
            return self._conversation_session_id
        self._conversation_session_id = None
        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        get_or_create = getattr(manager, "get_or_create", None)
        if not callable(get_or_create):
            return None
        try:
            session = get_or_create(channel="voice")
        except Exception as exc:
            logger.warning("VoiceLoop: conversation session unavailable: %s", exc)
            if self._degraded_conversation_session_id is None:
                self._degraded_conversation_session_id = (
                    f"voice-degraded-{uuid4().hex}"
                )
            self._conversation_session_id = self._degraded_conversation_session_id
            return self._conversation_session_id
        session_id = str(getattr(session, "session_id", "") or "").strip()
        if session_id:
            self._conversation_session_id = session_id
        return session_id or None

    def _record_local_gateway_turn(
        self,
        conversation_session_id: str | None,
        user_text: str,
        assistant_reply: Any,
    ) -> None:
        if not conversation_session_id or self._voice_runtime_bridge is None:
            return
        record_local_turn = getattr(
            self._voice_runtime_bridge,
            "record_local_turn",
            None,
        )
        if not callable(record_local_turn):
            return
        assistant_text = (
            assistant_reply
            if isinstance(assistant_reply, str)
            else getattr(self._pipeline, "last_spoken_text", "")
        )
        try:
            record_local_turn(
                conversation_session_id,
                user_text=user_text,
                assistant_text=str(assistant_text or ""),
                channel="voice",
            )
        except Exception as exc:
            logger.debug("VoiceLoop: local gateway turn record failed: %s", exc)

    def _cached_session_is_active(self, session_id: str) -> bool:
        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if not callable(snapshot):
            return True
        current = snapshot(session_id)
        if current is None:
            return False
        return getattr(current, "status", "active") == "active"


def _decision_to_dict(
    decision: InteractionDecision,
    *,
    addressed: bool,
) -> dict[str, Any]:
    return {
        "action": decision.action.value,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "addressed": addressed,
        "should_record_environment": decision.should_record_environment,
    }


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        return {}
    if hasattr(snapshot, "to_dict"):
        payload = snapshot.to_dict()
        return payload if isinstance(payload, dict) else {}
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def _audio_error_value(kind: Any) -> str:
    return str(getattr(kind, "value", kind) or "unknown")
