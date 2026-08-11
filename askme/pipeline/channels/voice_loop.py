"""Voice-mode main loop -microphone ->intent routing ->brain pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import Parameter, isawaitable, signature
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from askme.contracts.adapters import (
    interaction_decision_to_action_decision,
    perception_snapshot_to_input,
)
from askme.conversation import ConversationLedgerError, TurnOutcome
from askme.pipeline.channels.external_turns import (
    ExternalGenerationBeginError,
    begin_external_turn,
    cancel_external_turn,
    complete_external_turn,
    discard_external_generation,
)
from askme.pipeline.channels.runtime_bridge_calls import try_runtime_bridge_turn
from askme.pipeline.core.protocols import CancellationToken
from askme.pipeline.core.trace import get_tracer
from askme.pipeline.core.turn_control import AtomicCancellationToken
from askme.pipeline.skills.outcome import (
    GENERIC_SKILL_FAILURE_MESSAGE,
    NAV_LOCATION_UNAVAILABLE_MESSAGE,
    SkillOutcome,
    is_internal_skill_text,
)
from askme.ports import (
    AudioFrontendPort,
    AudioRouterPort,
    RealtimeApprovalPort,
    RealtimeVoiceFrontendPort,
    VoiceTurnBridgePort,
)
from askme.robot_interaction import attach_intent_route_trace
from askme.robot_interaction.interaction_gate import (
    InteractionAction,
    InteractionDecision,
    contains_emergency_intent,
    contains_robot_task_intent,
    contains_tool_route_intent,
)
from askme.robot_interaction.perception_context import InteractionPerceptionSnapshot
from askme.voice.diagnostics.status_privacy import sanitize_voice_status
from askme.voice.realtime.config import SUPPORTED_REALTIME_PROVIDERS
from askme.voice.realtime.policy import decide_realtime_route

if TYPE_CHECKING:
    from askme.pipeline.core.brain_pipeline import BrainPipeline
    from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
    from askme.robot_interaction import IntentRouter
    from askme.skills.core.skill_model import SkillDefinition

logger = logging.getLogger(__name__)
_SAFE_VOICE_TURN_ID = re.compile(r"[A-Za-z0-9._~:/@+=-]{1,256}")


def _validated_voice_turn_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if _SAFE_VOICE_TURN_ID.fullmatch(normalized) is not None else None


def _validated_asr_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


@dataclass(frozen=True, slots=True)
class _CapturedUtterance:
    """Transcript plus wake metadata captured before the next listen starts."""

    text: str | None
    wake_authorized: bool
    wake_source: str
    asr_confidence: float | None = None
    voice_turn_id: str | None = None
    realtime_generation: int = 0
    realtime_baseline_generation: int = 0


class AddressDetectorProtocol(Protocol):
    def is_addressed(self, text: str) -> bool:
        """Return whether recognized speech is addressed to the robot."""


class InteractionGateProtocol(Protocol):
    def evaluate(
        self,
        text: str,
        *,
        asr_confidence: float | None = None,
        addressed: bool = True,
        perception: InteractionPerceptionSnapshot | dict[str, Any] | None = None,
        mission_mode: str | None = None,
        actor_role: str | None = None,
        wake_authorized: bool = False,
        wake_source: str = "none",
        followup_active: bool = False,
        awaiting_confirmation: bool = False,
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
        asr_confidence: float | None = None,
        addressed: bool = True,
        perception: InteractionPerceptionSnapshot | dict[str, Any] | None = None,
        mission_mode: str | None = None,
        actor_role: str | None = None,
        wake_authorized: bool = False,
        wake_source: str = "none",
        followup_active: bool = False,
        awaiting_confirmation: bool = False,
    ) -> InteractionDecision:
        return InteractionDecision(InteractionAction.RESPOND, "gate_disabled", 1.0)


# Skills that can execute even while an agent_task is running in background.
# These are stateless/zero-cost and cannot conflict with the agent's work.
_AGENT_BYPASS_SKILLS: frozenset[str] = frozenset(
    [
        "get_time",
        "volume_up",
        "volume_down",
        "volume_reset",
        "speed_up",
        "speed_down",
        "speed_reset",
        "repeat_last",
        "mute_mic",
        "unmute_mic",
    ]
)

_KWS_UNAVAILABLE_SAFETY_SOURCE = "kws_unavailable_safety_only"
_KWS_UNAVAILABLE_LOCAL_SKILLS: frozenset[str] = frozenset(
    {"stop_speaking", "mute_mic", "unmute_mic"}
)


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
        anonymous_encounter_idle_seconds: float = 25.0,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge
        self._dispatcher = dispatcher
        self._audio_router = audio_router

        encounter_idle_seconds = float(anonymous_encounter_idle_seconds)
        if not math.isfinite(encounter_idle_seconds) or encounter_idle_seconds <= 0:
            raise ValueError("anonymous_encounter_idle_seconds must be a finite positive number")
        self._anonymous_encounter_idle_seconds = encounter_idle_seconds
        self._monotonic_clock = monotonic_clock or monotonic
        from askme.pipeline.proactive import ProactiveOrchestrator

        self._proactive = ProactiveOrchestrator.default(pipeline=pipeline, dispatcher=dispatcher)
        self._address_detector: AddressDetectorProtocol = _DefaultAddressDetector()
        self._interaction_gate: InteractionGateProtocol = _DefaultInteractionGate()
        self._interaction_perception_provider: Callable[[], Any] | None = None
        self._mission_context_provider: Callable[[], Any] | None = None
        self._last_interaction_decision: dict[str, Any] | None = None
        self._last_interaction_perception: dict[str, Any] | None = None
        self._last_mission_context: dict[str, Any] | None = None
        self._last_input_contract: dict[str, Any] | None = None
        self._last_action_contract: dict[str, Any] | None = None
        self._last_runtime_bridge_status: dict[str, Any] | None = None
        self._conversation_session_id: str | None = None
        self._degraded_conversation_session_id: str | None = None
        self._projected_direct_turn_ids: set[str] = set()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._conversation_session_last_used_monotonic: float | None = None
        self._conversation_session_closed = False
        self._listen_task: asyncio.Task[_CapturedUtterance] | None = None
        self._full_duplex_active = False
        self._closing = False
        self._active_interaction_cancel: AtomicCancellationToken | None = None
        self._solicited_response_active = False
        self._active_realtime_generation = 0
        self._active_realtime_baseline_generation = 0
        self._input_recovery_lock = threading.Lock()
        self._input_recovery_attempted = False
        self._lifecycle_generation = 0

    def set_address_detector(self, detector: AddressDetectorProtocol) -> None:
        """Wire the address detector after construction."""
        self._address_detector = detector

    def set_interaction_gate(self, gate: InteractionGateProtocol) -> None:
        """Wire the real-world interaction gate after construction."""
        self._interaction_gate = gate

    def set_interaction_perception_provider(self, provider: Callable[[], Any] | None) -> None:
        """Wire a best-effort provider for vision/audio-source/pose context."""
        self._interaction_perception_provider = provider

    def set_mission_context_provider(self, provider: Callable[[], Any] | None) -> None:
        """Wire the runtime-owned mission and actor state for voice admission."""
        self._mission_context_provider = provider

    async def _deliver_direct_reply(
        self,
        user_text: str,
        reply_text: str,
        *,
        conversation_session_id: str | None,
        interaction_turn_id: str | None,
        interaction_cancel: CancellationToken | None,
        interaction: str,
        speaker: Callable[[str], Awaitable[Any]] | None = None,
        before_playback: Callable[[], None] | None = None,
    ) -> bool:
        """Open, deliver, settle, then project one deterministic voice reply."""

        metadata = {"interaction": interaction}
        external_turn = begin_external_turn(
            self._pipeline,
            user_text,
            source="voice",
            channel="voice",
            conversation_session_id=conversation_session_id,
            turn_id=interaction_turn_id,
            metadata=metadata,
        )

        def _cancel(reason: str) -> None:
            cancel_external_turn(
                self._pipeline,
                external_turn,
                user_text=user_text,
                source="voice",
                reason=reason,
                conversation_session_id=conversation_session_id,
                metadata=metadata,
            )

        if interaction_cancel is not None and interaction_cancel.is_set():
            _cancel("cancelled_before_direct_reply")
            return False

        try:
            if before_playback is not None:
                before_playback()
            await (speaker or self._audio.speak_and_wait)(reply_text)
        except asyncio.CancelledError:
            try:
                _cancel("direct_reply_task_cancelled")
            except Exception as settlement_exc:
                logger.warning(
                    "VoiceLoop: direct reply cancellation settlement failed: %s",
                    settlement_exc,
                )
            raise
        except Exception as exc:
            try:
                self._fail_direct_reply_turn(
                    user_text=user_text,
                    conversation_session_id=conversation_session_id,
                    interaction_turn_id=interaction_turn_id,
                    interaction_cancel=interaction_cancel,
                    external_turn=external_turn,
                    reason=type(exc).__name__,
                    metadata=metadata,
                )
            except Exception as settlement_exc:
                logger.warning(
                    "VoiceLoop: direct reply failure settlement failed: %s",
                    settlement_exc,
                )
            raise

        if interaction_cancel is not None and interaction_cancel.is_set():
            _cancel("direct_reply_interrupted")
            return False

        recorder_contract_settled = False
        recorder = getattr(type(self._pipeline), "record_direct_reply", None)
        try:
            if callable(recorder):
                callback = getattr(self._pipeline, "record_direct_reply")
                optional_kwargs = {
                    "source": "voice",
                    "conversation_session_id": conversation_session_id,
                    "voice_turn_id": interaction_turn_id,
                    "turn_cancel_token": interaction_cancel,
                    "metadata": metadata,
                }
                try:
                    parameters = signature(callback).parameters
                    accepts_kwargs = any(
                        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
                    )
                except (TypeError, ValueError):
                    parameters = {}
                    accepts_kwargs = False
                supported_kwargs = (
                    optional_kwargs
                    if accepts_kwargs
                    else {
                        name: value for name, value in optional_kwargs.items() if name in parameters
                    }
                )
                recorder_contract_supported = accepts_kwargs or all(
                    name in parameters
                    for name in (
                        "conversation_session_id",
                        "voice_turn_id",
                        "turn_cancel_token",
                    )
                )
                result = callback(
                    user_text,
                    reply_text,
                    **supported_kwargs,
                )
                if isawaitable(result):
                    await result
                if external_turn is None:
                    recorder_contract_settled = recorder_contract_supported
                else:
                    canonical_status = self._direct_reply_turn_status(external_turn)
                    if canonical_status in {
                        "cancelled",
                        "failed",
                        "suppressed",
                    }:
                        return False
                    recorder_contract_settled = canonical_status == "committed"
            if not recorder_contract_settled:
                complete_external_turn(
                    self._pipeline,
                    external_turn,
                    user_text=user_text,
                    assistant_text=reply_text,
                    source="voice",
                    conversation_session_id=conversation_session_id,
                    metadata=metadata,
                )
        except asyncio.CancelledError:
            try:
                _cancel("direct_reply_settlement_cancelled")
            except Exception as settlement_exc:
                logger.warning(
                    "VoiceLoop: direct reply settlement cancellation failed: %s",
                    settlement_exc,
                )
            raise
        except Exception as exc:
            try:
                self._fail_direct_reply_turn(
                    user_text=user_text,
                    conversation_session_id=conversation_session_id,
                    interaction_turn_id=interaction_turn_id,
                    interaction_cancel=interaction_cancel,
                    external_turn=external_turn,
                    reason=f"settlement_{type(exc).__name__}",
                    metadata=metadata,
                )
            except Exception as settlement_exc:
                logger.warning(
                    "VoiceLoop: direct reply settlement failure could not settle: %s",
                    settlement_exc,
                )
            raise
        if external_turn is not None:
            canonical_status = self._direct_reply_turn_status(external_turn)
            if canonical_status is not None and canonical_status != "committed":
                if canonical_status == "started":
                    try:
                        self._fail_direct_reply_turn(
                            user_text=user_text,
                            conversation_session_id=conversation_session_id,
                            interaction_turn_id=interaction_turn_id,
                            interaction_cancel=interaction_cancel,
                            external_turn=external_turn,
                            reason="direct_reply_not_committed",
                            metadata=metadata,
                        )
                    except Exception as settlement_exc:
                        logger.warning(
                            "VoiceLoop: non-committed direct reply could not settle: %s",
                            settlement_exc,
                        )
                return False
        if interaction_cancel is not None and interaction_cancel.is_set():
            _cancel("direct_reply_interrupted_during_settlement")
            return False
        self._record_local_gateway_turn(
            conversation_session_id,
            user_text,
            reply_text,
            interaction_turn_id=interaction_turn_id,
        )
        return True

    def _direct_reply_turn_status(self, external_turn: Any) -> str | None:
        """Return an observable canonical status without activating mock attributes."""

        if external_turn is None:
            return None
        pipeline_state = getattr(self._pipeline, "__dict__", None)
        ledger = pipeline_state.get("_turn_ledger") if isinstance(pipeline_state, dict) else None
        if ledger is None and getattr(type(self._pipeline), "turn_ledger", None) is not None:
            ledger = getattr(self._pipeline, "turn_ledger", None)
        get_turn = getattr(ledger, "get_turn", None)
        if not callable(get_turn):
            return None
        try:
            turn = get_turn(external_turn.turn_id)
        except Exception:
            return None
        status = getattr(turn, "status", None)
        return str(getattr(status, "value", status) or "").strip().lower() or None

    def _fail_direct_reply_turn(
        self,
        *,
        user_text: str,
        conversation_session_id: str | None,
        interaction_turn_id: str | None,
        interaction_cancel: CancellationToken | None,
        external_turn: Any,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        """Fail through BrainPipeline when available; cancel legacy external turns."""

        opener = getattr(type(self._pipeline), "_open_direct_interaction", None)
        settler = getattr(type(self._pipeline), "_settle_direct_interaction", None)
        if callable(opener) and callable(settler):
            interaction = opener(
                self._pipeline,
                user_text=user_text,
                source="voice",
                conversation_session_id=conversation_session_id,
                voice_turn_id=interaction_turn_id,
                turn_cancel_token=interaction_cancel,
                metadata=metadata,
            )
            if interaction is not None:
                settler(
                    self._pipeline,
                    interaction,
                    TurnOutcome.fail(reason=reason, metadata=metadata),
                )
                return
        cancel_external_turn(
            self._pipeline,
            external_turn,
            user_text=user_text,
            source="voice",
            reason=f"direct_reply_failed:{reason}",
            conversation_session_id=conversation_session_id,
            metadata=metadata,
        )

    async def _speak_cached_or_fallback(
        self,
        text: str,
        *,
        cache_key: str | None,
        fallback_to_tts: bool,
    ) -> bool:
        """Prefer persisted PCM while keeping older audio frontends compatible."""

        cached_speaker = getattr(self._audio, "speak_cached_and_wait", None)
        if cache_key and callable(cached_speaker):
            try:
                if await cached_speaker(text, cache_key=cache_key):
                    return True
            except Exception as exc:
                logger.warning("Cached voice reply failed: %s", exc)
        if fallback_to_tts:
            await self._audio.speak_and_wait(text)
        return False

    async def _preflight_voice_skill(
        self,
        skill_name: str,
        user_text: str,
    ) -> SkillOutcome:
        """Check a local skill before any task-specific audible preface."""

        owner: Any = self._dispatcher
        if owner is None:
            pipeline_state = getattr(self._pipeline, "__dict__", {})
            if isinstance(pipeline_state, dict):
                owner = pipeline_state.get("_skill_gate")
        preflight = getattr(owner, "can_execute", None)
        if not callable(preflight):
            # A location preface promises that a real backend query is about
            # to happen. If there is no readiness probe, fail closed instead
            # of playing the preface and then returning an internal marker (or
            # silence) from an unbound skill path.
            if skill_name == "nav_query":
                return SkillOutcome.blocked(
                    code="preflight_unavailable",
                    result="[Skill] Preflight unavailable: nav_query",
                    user_message=NAV_LOCATION_UNAVAILABLE_MESSAGE,
                )
            return SkillOutcome.ready()
        try:
            outcome = await preflight(
                skill_name,
                user_text,
                source="voice",
            )
        except Exception as exc:
            logger.warning(
                "VoiceLoop: skill preflight failed closed for %s: %s",
                skill_name,
                exc,
            )
            return SkillOutcome.blocked(
                code="preflight_error",
                result=f"[Skill] Preflight failed: {skill_name}",
                user_message=(
                    NAV_LOCATION_UNAVAILABLE_MESSAGE
                    if skill_name == "nav_query"
                    else GENERIC_SKILL_FAILURE_MESSAGE
                ),
            )
        if isinstance(outcome, SkillOutcome):
            return outcome
        logger.error(
            "VoiceLoop: invalid skill preflight outcome for %s: %r",
            skill_name,
            outcome,
        )
        return SkillOutcome.blocked(
            code="invalid_preflight_outcome",
            result=f"[Skill] Invalid preflight: {skill_name}",
            user_message=(
                NAV_LOCATION_UNAVAILABLE_MESSAGE
                if skill_name == "nav_query"
                else GENERIC_SKILL_FAILURE_MESSAGE
            ),
        )

    async def _speak_skill_outcome(self, outcome: SkillOutcome) -> None:
        """Speak a blocked outcome once without exposing its internal result."""

        if not outcome.should_speak or not outcome.user_message:
            return
        self._audio.drain_buffers()
        await self._audio.speak_and_wait(outcome.user_message)
        self._remember_spoken_text(outcome.user_message)

    async def _handle_kws_unavailable_safety_turn(
        self,
        intent: Any,
        *,
        user_text: str,
        interaction_turn_id: str | None,
        interaction_cancel: CancellationToken | None,
    ) -> None:
        """Consume the fail-closed KWS lane without ACK, memory, LLM or skills."""

        skill_name = str(getattr(intent, "skill_name", "") or "")
        if skill_name not in _KWS_UNAVAILABLE_LOCAL_SKILLS:
            self._discard_realtime_turn(_KWS_UNAVAILABLE_SAFETY_SOURCE)
            logger.warning(
                "VoiceLoop: KWS unavailable; blocked non-safety voice turn (%s)",
                getattr(getattr(intent, "type", None), "value", "unknown"),
            )
            return

        self._discard_realtime_turn(_KWS_UNAVAILABLE_SAFETY_SOURCE)
        self._audio.drain_buffers()
        if skill_name == "stop_speaking":
            cancelled = bool(self._dispatcher and self._dispatcher.cancel_active_agent_task())
            if cancelled:
                await self._deliver_direct_reply(
                    user_text,
                    "已取消任务。",
                    conversation_session_id=self._conversation_session_for(),
                    interaction_turn_id=interaction_turn_id,
                    interaction_cancel=interaction_cancel,
                    interaction="kws_stop_speaking",
                )
            return
        if skill_name == "mute_mic":
            await self._deliver_direct_reply(
                user_text,
                '好的，已关闭麦克风。说"开麦"来重新打开。',
                conversation_session_id=self._conversation_session_for(),
                interaction_turn_id=interaction_turn_id,
                interaction_cancel=interaction_cancel,
                interaction="kws_mute_mic",
                before_playback=self._audio.mute,
            )
            return
        await self._deliver_direct_reply(
            user_text,
            "好的，已重新开启。",
            conversation_session_id=self._conversation_session_for(),
            interaction_turn_id=interaction_turn_id,
            interaction_cancel=interaction_cancel,
            interaction="kws_unmute_mic",
            before_playback=self._audio.unmute,
        )

    async def _safe_runtime_spoken_reply(self, text: str) -> None:
        reply = str(text or "").strip()
        if is_internal_skill_text(reply):
            logger.error("Suppressed runtime internal marker from TTS: %s", reply[:120])
            reply = GENERIC_SKILL_FAILURE_MESSAGE
        if reply:
            await self._audio.speak_and_wait(reply)

    def _remember_spoken_text(self, text: str) -> None:
        executor = getattr(self._pipeline, "_turn_executor", None)
        if executor is not None:
            executor._last_spoken_text = text
            return
        try:
            setattr(self._pipeline, "last_spoken_text", text)
        except (AttributeError, TypeError):
            pass

    async def _handle_estop_intent(
        self,
        *,
        user_text: str,
        interaction_turn_id: str | None,
        interaction_cancel: CancellationToken | None,
    ) -> None:
        """Execute one safety stop before any conversational admission gate."""

        self._discard_realtime_turn("estop")
        if self._dispatcher:
            self._dispatcher.cancel_active_agent_task()
        self._pipeline.handle_estop()
        self._audio.drain_buffers()
        await self._deliver_direct_reply(
            user_text,
            "\u5df2\u7ecf\u7d27\u6025\u505c\u6b62\u3002",
            conversation_session_id=self._conversation_session_for(),
            interaction_turn_id=interaction_turn_id,
            interaction_cancel=interaction_cancel,
            interaction="estop",
        )

    def _realtime_audio_port(self) -> RealtimeVoiceFrontendPort | None:
        audio = self._audio
        if isinstance(audio, RealtimeVoiceFrontendPort):
            return audio
        return None

    def _discard_realtime_turn(self, reason: str) -> None:
        realtime = self._realtime_audio_port()
        if realtime is None:
            return
        try:
            realtime.discard_realtime_turn(
                reason,
                expected_generation=self._active_realtime_generation,
                after_generation=self._active_realtime_baseline_generation,
            )
        except Exception as exc:
            logger.debug("Realtime turn discard failed: %s", exc)

    def _realtime_general_chat_ready(self) -> bool:
        realtime = self._realtime_audio_port()
        if realtime is None:
            return False
        try:
            return bool(realtime.realtime_general_chat_ready())
        except Exception as exc:
            logger.debug("Realtime voice readiness check failed: %s", exc)
            return False

    def _realtime_capture_active(self) -> bool:
        realtime = self._realtime_audio_port()
        if realtime is None:
            return False
        try:
            return bool(realtime.realtime_capture_active())
        except Exception as exc:
            logger.debug("Realtime capture status check failed: %s", exc)
            return False

    def _realtime_mode(self) -> str:
        """Return the provider mode without expanding the stable frontend port."""

        realtime = self._realtime_audio_port()
        if realtime is None:
            return "split"
        context_snapshot = getattr(realtime, "realtime_context_snapshot", None)
        if callable(context_snapshot):
            try:
                snapshot = context_snapshot()
                if isinstance(snapshot, dict):
                    mode = str(snapshot.get("mode") or "").strip().lower()
                    if mode in {"split", "shadow", "general_chat"}:
                        return mode
            except Exception as exc:
                logger.debug("Realtime mode snapshot unavailable: %s", exc)
        # Compatibility inference for older frontends/fakes.  The production
        # AudioAgent always exposes the explicit status snapshot above.
        if self._realtime_general_chat_ready():
            return "general_chat"
        if self._realtime_capture_active():
            return "shadow"
        return "split"

    def _realtime_provider(self) -> str:
        """Return the canonical provider selector without widening the audio port."""

        realtime = self._realtime_audio_port()
        if realtime is None:
            return ""
        context_snapshot = getattr(realtime, "realtime_context_snapshot", None)
        if callable(context_snapshot):
            try:
                snapshot = context_snapshot()
                if isinstance(snapshot, dict):
                    provider = str(snapshot.get("provider") or "").strip().lower()
                    if provider in SUPPORTED_REALTIME_PROVIDERS:
                        return provider
            except Exception as exc:
                logger.debug("Realtime provider snapshot unavailable: %s", exc)
        return ""

    def _abort_realtime_playback(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
    ) -> None:
        realtime = self._realtime_audio_port()
        if realtime is not None:
            abort: Any = realtime.abort_realtime_playback
            try:
                supports_generation = "expected_generation" in signature(abort).parameters
            except (TypeError, ValueError):
                supports_generation = False
            if expected_generation > 0 and supports_generation:
                abort(
                    reason,
                    expected_generation=expected_generation,
                )
            else:
                abort(reason)
            return
        self._discard_realtime_turn(reason)
        drain = getattr(self._audio, "drain_buffers", None)
        if callable(drain):
            drain()
        stop = getattr(self._audio, "stop_immediately", None)
        if callable(stop):
            stop()

    async def _try_handle_realtime_general_chat(
        self,
        user_text: str,
        *,
        expected_generation: int,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> bool:
        """Use approved provider audio without bypassing local intent/safety."""

        realtime = self._realtime_audio_port()
        if realtime is None:
            return False
        provider_selector = self._realtime_provider()
        if provider_selector not in SUPPORTED_REALTIME_PROVIDERS:
            self._discard_realtime_turn("unsupported_realtime_provider")
            return False
        if turn_cancel_token is not None and turn_cancel_token.is_set():
            self._discard_realtime_turn("cancelled_before_realtime_approval")
            return True
        prepare = getattr(realtime, "prepare_realtime_general_chat", None)
        release = getattr(realtime, "release_realtime_general_chat", None)
        two_phase = bool(callable(prepare) and callable(release))
        if not two_phase:
            logger.warning(
                "Realtime general-chat frontend lacks two-phase admission; using cascade"
            )
            self._discard_realtime_turn("two_phase_admission_unavailable")
            return False
        assert callable(prepare)

        def _admit_realtime() -> Any:
            return prepare(
                user_text,
                expected_generation=expected_generation,
            )

        try:
            approval = await asyncio.to_thread(_admit_realtime)
        except Exception as exc:
            logger.warning("Realtime voice approval failed; using cascade: %s", exc)
            return False
        if approval is None:
            return False
        if not isinstance(approval, RealtimeApprovalPort):
            logger.warning("Realtime provider returned an invalid approval handle")
            self._discard_realtime_turn("invalid_realtime_approval")
            return False

        initial_text = str(approval.initial_text or "")
        final_text = initial_text

        def _provider_context() -> dict[str, Any]:
            context_snapshot = getattr(realtime, "realtime_context_snapshot", None)
            if not callable(context_snapshot):
                return {}
            try:
                snapshot = context_snapshot()
                return dict(snapshot) if isinstance(snapshot, dict) else {}
            except Exception as exc:
                logger.debug("Realtime provider context unavailable: %s", exc)
                return {}

        provider_context = _provider_context()
        observed_provider = str(provider_context.get("provider") or "").strip().lower()
        if observed_provider != provider_selector:
            self._discard_realtime_turn("realtime_provider_identity_changed")
            return False
        ledger_provider = (
            "qwen" if provider_selector == "qwen3_5_omni" else "volcengine"
        )
        realtime_source = f"{ledger_provider}_realtime"
        provider_session_id = str(
            provider_context.get("provider_session_id")
            or ""
        ).strip() or None
        provider_dialog_id = str(
            provider_context.get("provider_dialog_id")
            or provider_context.get("dialog_id")
            or ""
        ).strip() or None
        turn_metadata: dict[str, Any] = {
            "realtime_generation": int(expected_generation or 0),
            "realtime_provider": provider_selector,
        }
        provider_model = str(provider_context.get("model") or "").strip()
        if provider_model:
            turn_metadata["provider_model"] = provider_model
        if provider_dialog_id:
            turn_metadata["provider_dialog_id"] = provider_dialog_id
        generation_id = (
            f"{voice_turn_id}:{ledger_provider}:{int(expected_generation or 0)}"
            if voice_turn_id
            else None
        )
        try:
            external_turn = begin_external_turn(
                self._pipeline,
                user_text,
                source=realtime_source,
                conversation_session_id=conversation_session_id,
                turn_id=voice_turn_id,
                provider=ledger_provider,
                provider_session_id=provider_session_id,
                provider_generation_id=str(expected_generation or "") or None,
                generation_id=generation_id,
                response_text=initial_text,
                metadata=turn_metadata,
            )
        except ExternalGenerationBeginError as exc:
            logger.info("Realtime generation rejected by Conversation Core: %s", exc)
            self._discard_realtime_turn("conversation_generation_begin_failed")
            return False
        except ConversationLedgerError as exc:
            # The provider candidate is still prepared/buffered on the
            # production two-phase path.  Fence it before the established local
            # cascade sees the canonical Turn conflict.
            logger.info("Realtime turn rejected by Conversation Core: %s", exc)
            self._discard_realtime_turn("conversation_turn_conflict")
            return False
        if external_turn is None:
            logger.error("Realtime release blocked because no durable Turn was begun")
            self._discard_realtime_turn("conversation_turn_begin_failed")
            return False
        if not external_turn.generation_id:
            logger.error(
                "Realtime release blocked because no durable provider Generation was begun"
            )
            self._discard_realtime_turn("conversation_generation_begin_failed")
            return False
        if turn_cancel_token is not None and turn_cancel_token.is_set():
            discard_external_generation(
                self._pipeline,
                external_turn,
                reason="cancelled_before_realtime_release",
                metadata=turn_metadata,
            )
            self._discard_realtime_turn("cancelled_before_realtime_release")
            return True
        assert callable(release)

        def _release_realtime() -> Any:
            kwargs: dict[str, Any] = {
                "expected_generation": expected_generation,
            }
            supports_owner = False
            try:
                parameters = signature(release).parameters.values()
            except (TypeError, ValueError):
                pass
            else:
                supports_owner = any(parameter.name == "voice_turn_id" for parameter in parameters)
            if voice_turn_id and supports_owner:
                kwargs["voice_turn_id"] = voice_turn_id
            return release(
                approval,
                **kwargs,
            )

        try:
            released = bool(await asyncio.to_thread(_release_realtime))
        except Exception as exc:
            logger.warning("Realtime PCM release failed; using cascade: %s", exc)
            released = False
        if not released:
            discard_external_generation(
                self._pipeline,
                external_turn,
                reason="realtime_release_rejected",
                metadata=turn_metadata,
            )
            self._discard_realtime_turn("realtime_release_rejected")
            return False

        def _cancel_realtime_turn(reason: str) -> None:
            context = _provider_context()
            played_ms = max(
                0,
                int(context.get("physical_played_ms") or context.get("committed_audio_ms") or 0),
            )
            # Capture the physical playhead before stop, then stop immediately;
            # the fsync-backed ledger write must never delay barge-in audio.
            self._abort_realtime_playback(
                reason,
                expected_generation=expected_generation,
            )
            stopped_context = _provider_context()
            played_ms = max(
                played_ms,
                int(
                    stopped_context.get("physical_played_ms")
                    or stopped_context.get("committed_audio_ms")
                    or 0
                ),
            )
            cancel_external_turn(
                self._pipeline,
                external_turn,
                user_text=user_text,
                source=realtime_source,
                reason=reason,
                played_ms=played_ms,
                heard_text="",
                conversation_session_id=conversation_session_id,
                metadata={
                    **turn_metadata,
                    "heard_text_alignment": "unavailable",
                },
            )

        try:
            completed_text = await asyncio.to_thread(approval.wait, 30.0)
            final_text = str(completed_text or initial_text)

            completed = bool(approval.completed)
            playback_started = bool(realtime.realtime_playback_started())
            if not completed or not playback_started:
                reason = (
                    "realtime_response_timeout"
                    if not completed
                    else "realtime_response_without_audio"
                )
                if playback_started:
                    _cancel_realtime_turn(reason)
                    return True
                self._abort_realtime_playback(
                    reason,
                    expected_generation=expected_generation,
                )
                discard_external_generation(
                    self._pipeline,
                    external_turn,
                    reason=reason,
                    metadata=turn_metadata,
                )
                return False

            playback_done = await asyncio.to_thread(self._audio.wait_speaking_done)
            if playback_done is False:
                _cancel_realtime_turn("realtime_playback_timeout")
                return True
            finish_realtime = getattr(
                realtime,
                "finish_realtime_playback",
                None,
            )
            if callable(finish_realtime):
                if not bool(
                    finish_realtime(
                        expected_generation=expected_generation,
                    )
                ):
                    raise RuntimeError("stale realtime playback owner")
            else:
                self._audio.stop_playback()
        except Exception as exc:
            # Once provider audio has been admitted, never emit a second local
            # answer for the same turn.
            logger.warning("Realtime voice playback ended early: %s", exc)
            _cancel_realtime_turn("realtime_playback_failure")
            return True

        if turn_cancel_token is not None and turn_cancel_token.is_set():
            _cancel_realtime_turn("cancelled_realtime_reply")
            return True

        if final_text:
            complete_external_turn(
                self._pipeline,
                external_turn,
                user_text=user_text,
                assistant_text=final_text,
                source=realtime_source,
                conversation_session_id=conversation_session_id,
                metadata=turn_metadata,
            )
            self._remember_spoken_text(final_text)
            self._record_local_gateway_turn(
                conversation_session_id,
                user_text,
                final_text,
            )
        else:
            # Provider audio was delivered but no trustworthy transcript was
            # returned.  Settle the canonical Turn as interrupted/unknown
            # instead of leaving it permanently STARTED or committing content
            # that cannot be audited.
            _cancel_realtime_turn("realtime_response_without_text")
        return True

    async def run(self) -> None:
        """Run the voice session and always release full-duplex callbacks/tasks."""

        if self._conversation_session_id is not None:
            self._rotate_anonymous_encounter("voice_loop_restart")
        self._event_loop = asyncio.get_running_loop()
        self._lifecycle_generation += 1
        self._closing = False
        self._full_duplex_active = getattr(self._audio, "full_duplex_enabled", False) is True
        if self._full_duplex_active:
            self._install_barge_in_callback()
        try:
            await self._run_session()
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the capture owner, then detach its callback and async task."""

        self._closing = True
        self._lifecycle_generation += 1
        self._event_loop = None
        setter = getattr(self._audio, "set_barge_in_callback", None)
        if callable(setter):
            try:
                setter(None)
            except Exception as exc:
                logger.debug("Failed to clear barge-in callback: %s", exc)

        listen_task = self._listen_task
        self._listen_task = None
        if listen_task is not None and not listen_task.done():
            # Cancel the asyncio wrapper first so an executor job that has not
            # started cannot become a late microphone consumer.  The audio
            # module below separately joins a job that already entered.
            listen_task.cancel()

        stop_listening = getattr(self._audio, "stop_listening", None)
        if callable(stop_listening):
            try:
                if not bool(await asyncio.to_thread(stop_listening)):
                    logger.warning("Microphone listener did not stop before timeout")
            except Exception as exc:
                logger.warning("Failed to stop microphone listener: %s", exc)

        if listen_task is not None:
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Pending listener stopped with an error: %s", exc)
        await asyncio.to_thread(self._finalize_input_recovery)
        self._full_duplex_active = False
        self._close_anonymous_encounter("voice_loop_stopped")

    def _install_barge_in_callback(self) -> None:
        setter = getattr(self._audio, "set_barge_in_callback", None)
        if not callable(setter):
            logger.warning("Full-duplex audio does not expose a confirmed barge-in callback")
            return
        setter(self._on_confirmed_barge_in)

    def _on_confirmed_barge_in(self) -> None:
        """Atomically mark the active turn cancelled from the capture thread."""

        if self._closing:
            return
        if self._solicited_response_active:
            # During clarification/confirmation the overlapping speech is the
            # answer we explicitly asked for. AudioAgent still stops the
            # question playback after this callback, but the parent skill turn
            # remains dispatchable and the captured answer is not discarded.
            logger.debug("Barge-in accepted as a solicited proactive response")
            return
        # Cancel the pipeline lease first while it still owns the shared
        # interaction token.  This records the controller epoch/reason; the
        # direct event set below also covers planner/proactive gaps where no
        # BrainPipeline lease exists yet.
        self._cancel_active_turn_for_barge_in()
        interaction_cancel = self._active_interaction_cancel
        if interaction_cancel is not None:
            interaction_cancel.set()

    def _cancel_active_turn_for_barge_in(self) -> None:
        cancel_active_turn = getattr(self._pipeline, "cancel_active_turn", None)
        if not callable(cancel_active_turn):
            return
        try:
            cancel_active_turn(reason="barge_in")
        except Exception as exc:
            logger.warning("Failed to cancel active voice turn on barge-in: %s", exc)

    async def _capture_once(self) -> _CapturedUtterance:
        text = await asyncio.to_thread(self._audio.listen_loop)
        realtime = self._realtime_audio_port()
        return _CapturedUtterance(
            asr_confidence=_validated_asr_confidence(
                getattr(self._audio, "last_turn_asr_confidence", None)
            ),
            text=text,
            wake_authorized=bool(getattr(self._audio, "last_turn_wake_authorized", False)),
            wake_source=str(getattr(self._audio, "last_turn_wake_source", "none") or "none")
            .strip()
            .lower(),
            voice_turn_id=_validated_voice_turn_id(
                getattr(self._audio, "last_accepted_voice_turn_id", None)
            ),
            realtime_generation=(
                int(realtime.last_turn_realtime_generation or 0) if realtime is not None else 0
            ),
            realtime_baseline_generation=(
                int(realtime.last_turn_realtime_baseline_generation or 0)
                if realtime is not None
                else 0
            ),
        )

    def _refresh_full_duplex_state(self) -> bool:
        if not self._full_duplex_active:
            return False
        if getattr(self._audio, "full_duplex_enabled", False) is True:
            return True

        self._full_duplex_active = False
        setter = getattr(self._audio, "set_barge_in_callback", None)
        if callable(setter):
            setter(None)
        logger.warning("VoiceLoop: audio frontend degraded to half-duplex")
        return False

    def _fail_closed_full_duplex_on_audio_error(
        self,
        *,
        reason: str,
        exc: BaseException,
    ) -> bool:
        """Downgrade overlap mode after a simultaneous-open device failure."""

        if not self._full_duplex_active:
            return False
        fail_closed = getattr(self._audio, "_full_duplex_fail_closed", None)
        if not callable(fail_closed):
            logger.error(
                "Full-duplex audio error has no fail-closed handler: %s",
                exc,
            )
            return False
        fail_closed(reason, exc)
        self._refresh_full_duplex_state()
        return True

    def _restart_audio_input(self, expected_generation: int | None = None) -> bool:
        """Release a failed capture handle and open a fresh input stream."""

        stop_input = getattr(self._audio, "stop_input", None)
        start_input = getattr(self._audio, "start_input", None)
        if not callable(stop_input) or not callable(start_input):
            logger.error("VoiceLoop: audio frontend cannot restart microphone input")
            return False

        with self._input_recovery_lock:
            if expected_generation is None:
                expected_generation = self._lifecycle_generation
            self._input_recovery_attempted = True
            if self._closing or expected_generation != self._lifecycle_generation:
                return False
            try:
                stop_input()
            except Exception as exc:
                # A robust input implementation clears its ownership pointer
                # before talking to a failed driver, so reopening may still work.
                logger.warning(
                    "VoiceLoop: microphone release failed before reconnect: %s",
                    exc,
                )
            if self._closing or expected_generation != self._lifecycle_generation:
                return False
            try:
                start_input()
            except Exception as exc:
                logger.warning("VoiceLoop: microphone reconnect failed: %s", exc)
                return False

            if self._closing or expected_generation != self._lifecycle_generation:
                try:
                    stop_input()
                except Exception as exc:
                    logger.warning(
                        "VoiceLoop: late microphone reconnect cleanup failed: %s",
                        exc,
                    )
                return False

            try:
                input_open = getattr(self._audio, "is_input_open", None)
                if callable(input_open):
                    input_open = input_open()
            except Exception as exc:
                logger.warning(
                    "VoiceLoop: microphone reconnect readiness probe failed: %s",
                    exc,
                )
                try:
                    stop_input()
                except Exception:
                    pass
                return False
            if input_open is not None and not bool(input_open):
                logger.warning("VoiceLoop: microphone reconnect returned without an open input")
                try:
                    stop_input()
                except Exception as exc:
                    logger.debug(
                        "VoiceLoop: failed to clean up false reconnect: %s",
                        exc,
                    )
                return False

        logger.info("VoiceLoop: microphone input reconnected")
        return True

    def _finalize_input_recovery(self) -> None:
        """Wait out any reconnect worker and leave its microphone input closed."""

        with self._input_recovery_lock:
            if not self._input_recovery_attempted:
                return
            stop_input = getattr(self._audio, "stop_input", None)
            if not callable(stop_input):
                self._input_recovery_attempted = False
                return
            try:
                stop_input()
            except Exception as exc:
                logger.warning(
                    "VoiceLoop: final microphone recovery cleanup failed: %s",
                    exc,
                )
            finally:
                self._input_recovery_attempted = False

    def _start_next_listen(self) -> None:
        if self._closing or not self._refresh_full_duplex_state():
            return
        if self._listen_task is not None and not self._listen_task.done():
            return
        self._listen_task = asyncio.create_task(
            self._capture_once(),
            name="voice-listen-next",
        )

    async def _next_utterance(self) -> _CapturedUtterance:
        if not self._refresh_full_duplex_state():
            return await self._capture_once()

        self._start_next_listen()
        listen_task = self._listen_task
        if listen_task is None:
            raise RuntimeError("full-duplex listener was not started")
        try:
            utterance = await listen_task
        finally:
            if self._listen_task is listen_task:
                self._listen_task = None
        self._start_next_listen()
        return utterance

    async def _next_utterance_text(self) -> str | None:
        """Share the single capture owner with proactive confirmation turns."""

        return (await self._next_utterance()).text

    async def _dispatcher_handle_general(
        self,
        user_text: str,
        *,
        source: str,
        memory_task: asyncio.Task[str] | None,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> Any:
        if self._dispatcher is None:
            raise RuntimeError("skill dispatcher is not configured")
        kwargs: dict[str, Any] = {
            "source": source,
            "memory_task": memory_task,
        }
        kwargs.update(
            self._supported_turn_context_kwargs(
                self._dispatcher.handle_general,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
            )
        )
        return await self._dispatcher.handle_general(user_text, **kwargs)

    async def _pipeline_process_general(
        self,
        user_text: str,
        *,
        memory_task: asyncio.Task[str] | None,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> Any:
        """Pass the capture-thread token when the pipeline supports it."""

        kwargs: dict[str, Any] = {
            "memory_task": memory_task,
        }
        kwargs.update(
            self._supported_turn_context_kwargs(
                self._pipeline.process,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
            )
        )
        return await self._pipeline.process(user_text, **kwargs)

    async def _pipeline_handle_pending_tool_response(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> Any:
        """Pass approval ownership context when the pipeline supports it."""

        return await self._pipeline.handle_pending_tool_response(
            user_text,
            **self._supported_turn_context_kwargs(
                self._pipeline.handle_pending_tool_response,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
            ),
        )

    @staticmethod
    def _supported_turn_context_kwargs(
        callback: Callable[..., Any],
        *,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> dict[str, Any]:
        """Return only turn-context keywords accepted by a legacy callable."""

        context = {
            "conversation_session_id": conversation_session_id,
            "voice_turn_id": voice_turn_id,
            "turn_cancel_token": turn_cancel_token,
        }
        try:
            parameters = signature(callback).parameters
            accepts_kwargs = any(
                parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            return {}
        if accepts_kwargs:
            return context
        return {name: value for name, value in context.items() if name in parameters}

    async def _run_proactive_interaction(
        self,
        skill_name: str,
        user_text: str,
    ) -> Any:
        """Mark the whole prompt/capture window as a solicited response."""

        self._solicited_response_active = True
        try:
            return await self._proactive.run(
                skill_name,
                user_text,
                self._audio,
                source="voice",
                listen_once=self._next_utterance_text,
            )
        finally:
            self._solicited_response_active = False

    def _arm_processing_feedback(
        self,
        cancel_token: AtomicCancellationToken | None,
    ) -> bool:
        try:
            return bool(self._audio.arm_processing_feedback(cancel_token))
        except Exception as exc:
            logger.debug(
                "VoiceLoop: processing feedback arm failed: %s",
                exc,
            )
            return False

    async def _run_session(self) -> None:
        """Block until Ctrl+C or too many consecutive errors."""
        from askme.robot_interaction import IntentType

        logger.info("Voice mode active. Say something! (Ctrl+C to quit)")

        consecutive_errors = 0
        idle_task = self._pipeline.start_idle_reflection()
        _tracer = get_tracer()
        while True:
            memory_task: asyncio.Task[str] | None = None
            interaction_cancel: AtomicCancellationToken | None = None
            processing_feedback_armed = False
            interaction_turn_id: str | None = None
            _trace = None
            try:
                # Tell the noise filter whether we're waiting for a
                # confirmation so short replies can pass through.
                # If the last assistant message was a question, the user's
                # short reply is likely an answer.
                _last = self._pipeline.last_spoken_text or ""
                _ends_with_question = _last.rstrip().endswith(("？", "?"))
                self._audio.awaiting_confirmation = (
                    self._pipeline.has_pending_tool_approval() or _ends_with_question
                )

                utterance = await self._next_utterance()
                user_text = utterance.text
                if not user_text:
                    continue
                self._active_realtime_generation = max(
                    0,
                    int(utterance.realtime_generation or 0),
                )
                self._active_realtime_baseline_generation = max(
                    0,
                    int(utterance.realtime_baseline_generation or 0),
                )
                interaction_cancel = AtomicCancellationToken()
                interaction_turn_id = utterance.voice_turn_id or uuid4().hex
                self._active_interaction_cancel = interaction_cancel

                normalize_text = getattr(self._address_detector, "normalize_text", None)
                if callable(normalize_text):
                    normalized_text = normalize_text(user_text)
                    if normalized_text != user_text:
                        logger.info(
                            "Normalized ASR robot-name alias (input_chars=%d, normalized_chars=%d)",
                            len(user_text),
                            len(normalized_text),
                        )
                        user_text = normalized_text

                consecutive_errors = 0

                # Start pipeline trace for this turn
                _trace = _tracer.start_trace("voice_turn")
                _trace.metadata["user_text_chars"] = len(user_text)

                # Route once before every conversational admission gate. A
                # deterministic E-STOP must remain available while muted, while
                # an approval is pending, and when ambient-speech policy would
                # otherwise reject the utterance.
                with _tracer.span("intent_route") as _route_span:
                    intent = self._router.route(user_text)
                    _route_span.metadata.update(
                        attach_intent_route_trace(
                            _trace,
                            intent,
                            source="voice",
                            include_content=False,
                        )
                    )
                _trace.metadata["voice_fast_path"] = bool(intent.fast_path)
                if intent.type == IntentType.ESTOP:
                    await self._handle_estop_intent(
                        user_text=user_text,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    continue
                safety_perception = self._get_interaction_perception()
                if safety_perception is not None:
                    safety_wake_source = utterance.wake_source
                    if utterance.wake_authorized and safety_wake_source == "none":
                        safety_wake_source = "keyword"
                    safety_decision = self._interaction_gate.evaluate(
                        user_text,
                        addressed=(
                            self._address_detector.is_addressed(user_text)
                            or utterance.wake_authorized
                        ),
                        perception=safety_perception,
                        wake_authorized=utterance.wake_authorized,
                        wake_source=safety_wake_source,
                        followup_active=safety_wake_source == "followup_window",
                        awaiting_confirmation=bool(self._audio.awaiting_confirmation),
                    )
                    if safety_decision.reason == "safety_stop_gesture":
                        await self._handle_estop_intent(
                            user_text=user_text,
                            interaction_turn_id=interaction_turn_id,
                            interaction_cancel=interaction_cancel,
                        )
                        continue

                # AudioAgent marks this source when product policy requires a
                # wake word but KWS is unavailable.  Only local, deterministic
                # safety controls may pass; everything else is consumed before
                # interaction admission, ACK, memory, LLM, bridge, or skill work.
                if utterance.wake_source == _KWS_UNAVAILABLE_SAFETY_SOURCE:
                    _trace.metadata["kws_unavailable_safety_only"] = True
                    await self._handle_kws_unavailable_safety_turn(
                        intent,
                        user_text=user_text,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    continue

                # Muted state gate
                # When muted, only the unmute_mic voice trigger and COMMAND
                # (quit/exit) pass through. Everything else is silently discarded.
                if self._audio.is_muted:
                    _muted_intent = intent
                    attach_intent_route_trace(
                        _trace,
                        _muted_intent,
                        source="voice",
                        stage="muted_gate_route",
                        include_content=False,
                    )
                    if (
                        _muted_intent.type == IntentType.VOICE_TRIGGER
                        and _muted_intent.skill_name == "unmute_mic"
                    ):

                        def _unmute_and_acknowledge() -> None:
                            self._audio.unmute()
                            self._audio.acknowledge()

                        await self._deliver_direct_reply(
                            user_text,
                            "好的，已重新开启。",
                            conversation_session_id=self._conversation_session_for(),
                            interaction_turn_id=interaction_turn_id,
                            interaction_cancel=interaction_cancel,
                            interaction="unmute_mic",
                            before_playback=_unmute_and_acknowledge,
                        )
                        continue
                    elif _muted_intent.type == IntentType.COMMAND:
                        pass  # fall through to COMMAND handler below
                    else:
                        self._discard_realtime_turn("muted")
                        continue

                # Interaction gate: separate ambient speech from real user turns.
                addressed_by_text = self._address_detector.is_addressed(user_text)
                wake_authorized = utterance.wake_authorized
                wake_source = utterance.wake_source
                if wake_authorized and wake_source == "none":
                    wake_source = "keyword"
                followup_active = wake_source == "followup_window"
                addressed = addressed_by_text or wake_authorized
                perception_snapshot = safety_perception
                mission_context = self._get_mission_context()
                gate_decision = self._interaction_gate.evaluate(
                    user_text,
                    asr_confidence=utterance.asr_confidence,
                    addressed=addressed,
                    perception=perception_snapshot,
                    mission_mode=_clean_optional_text(mission_context.get("mission_mode")),
                    actor_role=_clean_optional_text(mission_context.get("actor_role")),
                    wake_authorized=wake_authorized,
                    wake_source=wake_source,
                    followup_active=followup_active,
                    awaiting_confirmation=bool(self._audio.awaiting_confirmation),
                )
                self._last_interaction_decision = _decision_to_dict(
                    gate_decision,
                    addressed=addressed,
                    addressed_by_text=addressed_by_text,
                    wake_authorized=wake_authorized,
                    wake_source=wake_source,
                    followup_active=followup_active,
                    awaiting_confirmation=bool(self._audio.awaiting_confirmation),
                )
                self._last_interaction_perception = _snapshot_to_dict(perception_snapshot)
                self._last_mission_context = dict(mission_context)
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
                _trace.metadata["interaction_gate"] = sanitize_voice_status(
                    {
                        "action": gate_decision.action.value,
                        "reason": gate_decision.reason,
                        "asr_confidence": utterance.asr_confidence,
                        "confidence": gate_decision.confidence,
                        "addressed": addressed,
                        "addressed_by_text": addressed_by_text,
                        "wake_authorized": wake_authorized,
                        "wake_source": wake_source,
                        "followup_active": followup_active,
                        "awaiting_confirmation": bool(self._audio.awaiting_confirmation),
                        "perception": self._last_interaction_perception,
                        "mission_context": self._last_mission_context,
                    }
                )
                _trace.metadata["product_contract"] = sanitize_voice_status(
                    {
                        "perception_input": self._last_input_contract,
                        "action_decision": self._last_action_contract,
                    }
                )
                logger.info(
                    "InteractionGate: action=%s reason=%s confidence=%.2f "
                    "wake_source=%s addressed=%s",
                    gate_decision.action.value,
                    gate_decision.reason,
                    gate_decision.confidence,
                    wake_source,
                    addressed,
                )
                if gate_decision.reason == "safety_stop_gesture":
                    await self._handle_estop_intent(
                        user_text=user_text,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    continue
                if gate_decision.action in (
                    InteractionAction.IGNORE,
                    InteractionAction.RECORD_ONLY,
                ):
                    self._discard_realtime_turn(f"interaction_gate_{gate_decision.action.value}")
                    self._record_environment_speech(user_text, gate_decision)
                    self._rotate_anonymous_encounter("ambient_speech")
                    continue

                self._mark_interaction_turn()
                if gate_decision.action in (
                    InteractionAction.CLARIFY,
                    InteractionAction.DEFER,
                    InteractionAction.REFUSE,
                ):
                    self._discard_realtime_turn(f"interaction_gate_{gate_decision.action.value}")
                    self._record_environment_speech(user_text, gate_decision)
                    if gate_decision.reply:
                        await self._audio.speak_and_wait(gate_decision.reply)
                    continue

                pending_approval = self._pipeline.has_pending_tool_approval()
                realtime_capture_active = self._realtime_capture_active()
                realtime_general_ready = self._realtime_general_chat_ready()
                realtime_mode = self._realtime_mode()
                realtime_provider = self._realtime_provider()
                realtime_robot_task = bool(
                    contains_robot_task_intent(user_text)
                    or str(gate_decision.reason or "").startswith("robot_task")
                    or gate_decision.reason == "unaddressed_robot_task"
                )
                realtime_emergency = contains_emergency_intent(user_text)
                realtime_tool_route = bool(
                    intent.type != IntentType.GENERAL
                    or contains_tool_route_intent(user_text)
                    or getattr(intent, "skill_name", None)
                    or getattr(intent, "command", None)
                    or getattr(intent, "scenario_id", None)
                    or str(getattr(intent, "reason", "") or "") == "visual_query"
                )
                realtime_route = decide_realtime_route(
                    mode=realtime_mode,
                    interaction_admitted=(gate_decision.action is InteractionAction.RESPOND),
                    intent_type=intent.type.value,
                    provider_ready=realtime_capture_active,
                    provider=realtime_provider,
                    emergency=realtime_emergency,
                    pending_approval=pending_approval,
                    robot_task=realtime_robot_task,
                    tool_route=realtime_tool_route,
                )
                _trace.metadata["realtime_route"] = {
                    "mode": realtime_mode,
                    "provider": realtime_provider,
                    "route": realtime_route.route,
                    "allow_provider_audio": realtime_route.allow_provider_audio,
                    "interrupt_provider": realtime_route.interrupt_provider,
                    "reason": realtime_route.reason,
                    "robot_task": realtime_robot_task,
                    "tool_route": realtime_tool_route,
                    "emergency": realtime_emergency,
                }
                realtime_candidate = bool(
                    realtime_route.allow_provider_audio
                    and utterance.realtime_generation > 0
                    and realtime_general_ready
                )
                if not realtime_candidate and realtime_capture_active:
                    if pending_approval:
                        discard_reason = "pending_tool_approval"
                    elif intent.type != IntentType.GENERAL:
                        discard_reason = f"intent_{intent.type.value}"
                    elif realtime_route.route == "shadow":
                        discard_reason = "local_cascade"
                    elif not realtime_route.allow_provider_audio:
                        discard_reason = realtime_route.reason
                    else:
                        discard_reason = "realtime_generation_unavailable"
                    self._discard_realtime_turn(discard_reason)

                # Deterministic cached replies run before the ACK chime,
                # memory retrieval, LLM and online TTS. Pending tool approval
                # still takes precedence over ordinary quick replies.
                if not pending_approval and intent.type == IntentType.QUICK_REPLY:
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    _quick_text = intent.reply_text or intent.skill_name or "\u597d\u7684\u3002"
                    self._audio.drain_buffers()
                    conversation_session_id = self._conversation_session_for()
                    cache_hit = False

                    async def _speak_cached(reply_text: str) -> None:
                        nonlocal cache_hit
                        cache_hit = await self._speak_cached_or_fallback(
                            reply_text,
                            cache_key=intent.cached_audio_key,
                            fallback_to_tts=True,
                        )

                    delivered = await self._deliver_direct_reply(
                        user_text,
                        _quick_text,
                        conversation_session_id=conversation_session_id,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                        interaction="cached_quick_reply",
                        speaker=_speak_cached,
                    )
                    _trace.metadata["voice_phrase_cache_hit"] = cache_hit
                    if delivered:
                        self._remember_spoken_text(_quick_text)
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Immediate audio feedback -user knows we heard them
                # Fires before LLM call to fill the latency gap
                if not intent.fast_path and not realtime_candidate:
                    self._audio.acknowledge()

                # Cancel idle reflection on user activity
                if idle_task and not idle_task.done():
                    idle_task.cancel()

                conversation_session_id = self._conversation_session_for()
                pending_reply = await self._pipeline_handle_pending_tool_response(
                    user_text,
                    conversation_session_id=conversation_session_id,
                    voice_turn_id=interaction_turn_id,
                    turn_cancel_token=interaction_cancel,
                )
                if pending_reply is not None:
                    self._discard_realtime_turn("pending_tool_response")
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                if intent.type == IntentType.GENERAL and not realtime_candidate:
                    processing_feedback_armed = self._arm_processing_feedback(interaction_cancel)
                # Start memory prefetch only after deterministic paths have
                # exited.  This avoids needless retrieval work on fast replies.
                if not intent.fast_path:
                    memory_task = self._pipeline.start_memory_prefetch(user_text)

                # Quick reply -zero LLM, instant response
                if intent.type == IntentType.QUICK_REPLY:
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    _quick_text = intent.reply_text or intent.skill_name or "好的。"
                    self._audio.drain_buffers()
                    delivered = await self._deliver_direct_reply(
                        user_text,
                        _quick_text,
                        conversation_session_id=conversation_session_id,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                        interaction="quick_reply",
                    )
                    if delivered:
                        self._remember_spoken_text(_quick_text)
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Stop speaking -also cancels any active agent task
                if intent.type == IntentType.VOICE_TRIGGER and intent.skill_name == "stop_speaking":
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    if self._dispatcher and self._dispatcher.cancel_active_agent_task():
                        self._audio.drain_buffers()
                        await self._deliver_direct_reply(
                            user_text,
                            "\u5df2\u53d6\u6d88\u4efb\u52a1\u3002",
                            conversation_session_id=conversation_session_id,
                            interaction_turn_id=interaction_turn_id,
                            interaction_cancel=interaction_cancel,
                            interaction="stop_speaking",
                        )
                    else:
                        self._audio.drain_buffers()
                    # acknowledge already fired -no extra chime needed
                    continue

                # Repeat last response -zero LLM, replay TTS
                if intent.type == IntentType.VOICE_TRIGGER and intent.skill_name == "repeat_last":
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    last = self._pipeline.last_spoken_text
                    self._audio.drain_buffers()
                    if last:
                        reply = last
                    else:
                        reply = "暂时没有内容可以重复。"
                    await self._deliver_direct_reply(
                        user_text,
                        reply,
                        conversation_session_id=conversation_session_id,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                        interaction="repeat_last",
                    )
                    continue

                # Mute mic -zero latency, no LLM
                if intent.type == IntentType.VOICE_TRIGGER and intent.skill_name == "mute_mic":
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
                    self._audio.drain_buffers()
                    await self._deliver_direct_reply(
                        user_text,
                        '好的，已关闭麦克风。说"开麦"来重新打开。',
                        conversation_session_id=conversation_session_id,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                        interaction="mute_mic",
                        before_playback=self._audio.mute,
                    )
                    continue

                # Volume / speed -zero latency, no LLM
                _vol_speed_skill = (
                    intent.skill_name if intent.type == IntentType.VOICE_TRIGGER else None
                )
                if _vol_speed_skill in (
                    "volume_up",
                    "volume_down",
                    "volume_reset",
                    "speed_up",
                    "speed_down",
                    "speed_reset",
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
                    await self._deliver_direct_reply(
                        user_text,
                        msg,
                        conversation_session_id=conversation_session_id,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                        interaction=str(_vol_speed_skill),
                    )
                    continue

                # Agent-busy gate
                # While a background agent_task is running, block new skill
                # dispatches and LLM turns to prevent audio conflicts.
                # ESTOP and stop_speaking are handled above and always pass through.
                # Lightweight skills (get_time, volume, etc.) bypass the gate.
                if self._dispatcher and self._dispatcher.has_active_agent_task:
                    _bypass = (
                        intent.type == IntentType.VOICE_TRIGGER
                        and intent.skill_name in _AGENT_BYPASS_SKILLS
                    )
                    if not _bypass:
                        self._discard_realtime_turn("agent_task_busy")
                        if memory_task and not memory_task.done():
                            memory_task.cancel()
                            memory_task = None
                        await self._deliver_direct_reply(
                            user_text,
                            "正在处理中，说够了可取消。",
                            conversation_session_id=conversation_session_id,
                            interaction_turn_id=interaction_turn_id,
                            interaction_cancel=interaction_cancel,
                            interaction="agent_busy",
                        )
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
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    if bridge_handled:
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                    # Bridge not configured / failed: prove the local skill is
                    # runnable before saying a task-specific waiting sentence.
                    skill_name = intent.skill_name or ""
                    skill_outcome = await self._preflight_voice_skill(
                        skill_name,
                        user_text,
                    )
                    _trace.metadata["skill_preflight"] = {
                        "skill": skill_name,
                        "status": skill_outcome.status.value,
                        "code": skill_outcome.code,
                    }
                    if not skill_outcome.can_execute:
                        await self._speak_skill_outcome(skill_outcome)
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                    if intent.preface_text:
                        cache_hit = await self._speak_cached_or_fallback(
                            intent.preface_text,
                            cache_key=intent.preface_audio_key,
                            fallback_to_tts=False,
                        )
                        _trace.metadata["voice_preface_cache_hit"] = cache_hit
                        if not cache_hit:
                            self._audio.acknowledge()

                    # Ready local skill dispatch
                    if self._dispatcher:
                        result = await self._run_proactive_interaction(
                            skill_name,
                            user_text,
                        )
                        if result.proceed:
                            if not interaction_cancel.is_set():
                                conversation_session_id = self._conversation_session_for()
                                await self._dispatcher.dispatch(
                                    skill_name,
                                    result.enriched_text,
                                    source="voice",
                                    **self._supported_turn_context_kwargs(
                                        self._dispatcher.dispatch,
                                        conversation_session_id=conversation_session_id,
                                        voice_turn_id=interaction_turn_id,
                                        turn_cancel_token=interaction_cancel,
                                    ),
                                )
                        elif result.interrupt_payload:
                            # User bailed out and issued a new intent in the same breath
                            # e.g. "算了，去仓库B" -> reroute immediately without re-listening
                            logger.info(
                                "VoiceLoop: rerouting interrupt payload (chars=%d)",
                                len(result.interrupt_payload),
                            )
                            _reroute_intent = self._router.route(result.interrupt_payload)
                            attach_intent_route_trace(
                                _trace,
                                _reroute_intent,
                                source="voice",
                                stage="interrupt_reroute",
                                include_content=False,
                            )
                            if (
                                _reroute_intent.type == IntentType.VOICE_TRIGGER
                                and _reroute_intent.skill_name
                            ):
                                _rr = await self._run_proactive_interaction(
                                    _reroute_intent.skill_name,
                                    result.interrupt_payload,
                                )
                                if _rr.proceed:
                                    if not interaction_cancel.is_set():
                                        conversation_session_id = self._conversation_session_for()
                                        await self._dispatcher.dispatch(
                                            _reroute_intent.skill_name,
                                            _rr.enriched_text,
                                            source="voice",
                                            **self._supported_turn_context_kwargs(
                                                self._dispatcher.dispatch,
                                                conversation_session_id=conversation_session_id,
                                                voice_turn_id=interaction_turn_id,
                                                turn_cancel_token=interaction_cancel,
                                            ),
                                        )
                            else:
                                # Rerouted to a general intent -start fresh memory
                                # prefetch for the new payload so LLM gets context.
                                memory_task = self._pipeline.start_memory_prefetch(
                                    result.interrupt_payload
                                )
                                conversation_session_id = self._conversation_session_for()
                                await self._dispatcher_handle_general(
                                    result.interrupt_payload,
                                    source="voice",
                                    memory_task=memory_task,
                                    conversation_session_id=conversation_session_id,
                                    voice_turn_id=interaction_turn_id,
                                    turn_cancel_token=interaction_cancel,
                                )
                                memory_task = None  # handle_general took ownership
                                if idle_task and not idle_task.done():
                                    idle_task.cancel()
                                idle_task = self._pipeline.start_idle_reflection()
                    else:
                        if not interaction_cancel.is_set():
                            conversation_session_id = self._conversation_session_for()
                            await self._pipeline.execute_skill(
                                skill_name,
                                user_text,
                                **self._supported_turn_context_kwargs(
                                    self._pipeline.execute_skill,
                                    conversation_session_id=conversation_session_id,
                                    voice_turn_id=interaction_turn_id,
                                    turn_cancel_token=interaction_cancel,
                                ),
                            )
                    continue

                if intent.type == IntentType.COMMAND:
                    if intent.command in ("quit", "exit", "/quit", "/exit"):
                        logger.info("Exit command received in voice mode.")
                        break
                    # Other commands (/clear, /help, etc.) fall through to LLM
                    # so the assistant can respond naturally by voice

                if intent.type == IntentType.GENERAL:
                    bridge_handled = await self._maybe_handle_runtime_bridge(
                        user_text,
                        conversation_session_id=conversation_session_id,
                        interaction_turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    if bridge_handled:
                        self._discard_realtime_turn("runtime_bridge_handled")
                        # Cancel the memory prefetch we started earlier -the bridge
                        # handled the turn so the prefetched context is no longer needed.
                        if memory_task and not memory_task.done():
                            memory_task.cancel()
                        memory_task = None
                        if idle_task and not idle_task.done():
                            idle_task.cancel()
                        idle_task = self._pipeline.start_idle_reflection()
                        continue

                    conversation_session_id = self._conversation_session_for()
                    if realtime_candidate:
                        realtime_handled = await self._try_handle_realtime_general_chat(
                            user_text,
                            expected_generation=utterance.realtime_generation,
                            conversation_session_id=conversation_session_id,
                            voice_turn_id=interaction_turn_id,
                            turn_cancel_token=interaction_cancel,
                        )
                        if realtime_handled:
                            if memory_task and not memory_task.done():
                                memory_task.cancel()
                            memory_task = None
                            if idle_task and not idle_task.done():
                                idle_task.cancel()
                            idle_task = self._pipeline.start_idle_reflection()
                            continue
                        # Transcript mismatch, timeout, provider failure, or a
                        # stale turn token retains the established cascade.
                        self._discard_realtime_turn("realtime_general_fallback")
                        if not intent.fast_path:
                            self._audio.acknowledge()

                # Realtime fallback and non-exit command routes reach the same
                # LLM path without the early GENERAL fuse.
                if not processing_feedback_armed:
                    processing_feedback_armed = self._arm_processing_feedback(interaction_cancel)
                # General ->LLM (pass pre-fetched memory)
                conversation_session_id = self._conversation_session_for()
                assistant_reply: Any = None
                with _tracer.span("llm_pipeline"):
                    if self._dispatcher:
                        assistant_reply = await self._dispatcher_handle_general(
                            user_text,
                            source="voice",
                            memory_task=memory_task,
                            conversation_session_id=conversation_session_id,
                            voice_turn_id=interaction_turn_id,
                            turn_cancel_token=interaction_cancel,
                        )
                    else:
                        assistant_reply = await self._pipeline_process_general(
                            user_text,
                            memory_task=memory_task,
                            conversation_session_id=conversation_session_id,
                            voice_turn_id=interaction_turn_id,
                            turn_cancel_token=interaction_cancel,
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
                self._discard_realtime_turn("voice_loop_error")
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
                    self._fail_closed_full_duplex_on_audio_error(
                        reason="audio_device_runtime_failure",
                        exc=exc,
                    )
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
                            await self._audio.speak_and_wait("麦克风断开，正在重连。")
                        except Exception:
                            pass
                    recovery_generation = self._lifecycle_generation
                    recovered = await asyncio.to_thread(
                        self._restart_audio_input,
                        recovery_generation,
                    )
                    if recovered:
                        retry_delay = min(
                            5.0,
                            0.5 * (2 ** min(consecutive_errors - 1, 4)),
                        )
                    else:
                        retry_delay = 5.0
                    await asyncio.sleep(retry_delay)
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
                        await self._audio.speak_and_wait("系统暂时遇到问题，请稍候。")
                    except Exception:
                        pass
                    await asyncio.sleep(5)
                    consecutive_errors = 0
                await asyncio.sleep(1)
            finally:
                if processing_feedback_armed and self._audio.processing_feedback_armed:
                    try:
                        self._audio.cancel_processing_feedback()
                    except Exception as exc:
                        logger.debug(
                            "VoiceLoop: processing feedback cancel failed: %s",
                            exc,
                        )
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
                if self._active_interaction_cancel is interaction_cancel:
                    self._active_interaction_cancel = None
                self._active_realtime_generation = 0
                self._active_realtime_baseline_generation = 0

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
            "InteractionGate: %s (%s, %.2f, chars=%d)",
            decision.action.value,
            decision.reason,
            decision.confidence,
            len(user_text),
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

    def _mark_interaction_turn(self) -> None:
        marker = getattr(self._audio, "mark_interaction_turn", None)
        if callable(marker):
            marker()

    def _get_interaction_perception(self) -> InteractionPerceptionSnapshot | dict[str, Any] | None:
        if self._interaction_perception_provider is None:
            return None
        try:
            return self._interaction_perception_provider()
        except Exception as exc:
            logger.debug("Interaction perception provider failed: %s", exc)
            return InteractionPerceptionSnapshot.unknown("provider_error")

    def _get_mission_context(self) -> dict[str, Any]:
        if self._mission_context_provider is None:
            return {}
        try:
            payload = self._mission_context_provider()
        except Exception as exc:
            logger.warning("VoiceLoop: mission context provider failed: %s", exc)
            return {"source": "provider_error"}
        return dict(payload) if isinstance(payload, dict) else {}

    def interaction_status_snapshot(self) -> dict[str, Any]:
        return {
            "last_decision": dict(self._last_interaction_decision or {}),
            "last_perception": dict(self._last_interaction_perception or {}),
            "mission_context": dict(self._last_mission_context or {}),
            "last_input_contract": dict(self._last_input_contract or {}),
            "last_action_contract": dict(self._last_action_contract or {}),
            "runtime_bridge": dict(self._last_runtime_bridge_status or {}),
        }

    def _classify_audio_error(self, exc: BaseException) -> Any:
        if self._audio_router is None:
            return "unknown"
        classifier = getattr(self._audio_router, "classify_error", None)
        if not callable(classifier):
            return "unknown"
        return classifier(exc)

    async def _maybe_handle_runtime_bridge(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None,
        interaction_turn_id: str | None,
        interaction_cancel: CancellationToken | None,
    ) -> bool:
        """Try the runtime bridge first and fall back locally on bridge failures."""
        if self._voice_runtime_bridge is None:
            return False

        try:
            outcome = await try_runtime_bridge_turn(
                self._voice_runtime_bridge.handle_voice_text,
                user_text,
                conversation_session_id=conversation_session_id,
                voice_turn_id=interaction_turn_id,
                turn_cancel_token=interaction_cancel,
                pipeline=self._pipeline,
                dispatcher=self._dispatcher,
                on_spoken_reply=self._safe_runtime_spoken_reply,
                label="Voice",
            )
            status = self._runtime_bridge_snapshot()
            status.update(
                {
                    "handled": outcome.handled,
                    "local_fallback": not outcome.handled,
                }
            )
            self._last_runtime_bridge_status = status
            return outcome.handled
        except Exception as exc:
            logger.warning("VoiceLoop: runtime bridge failed, falling back locally: %s", exc)
            self._last_runtime_bridge_status = {
                **self._runtime_bridge_snapshot(),
                "handled": False,
                "local_fallback": True,
                "last_status": "exception",
                "last_error_type": type(exc).__name__,
            }
            return False

    def _runtime_bridge_snapshot(self) -> dict[str, Any]:
        status = getattr(self._voice_runtime_bridge, "status_snapshot", None)
        if not callable(status):
            return {}
        try:
            payload = status()
        except Exception as exc:
            return {
                "last_status": "status_unavailable",
                "last_error_type": type(exc).__name__,
            }
        return dict(payload) if isinstance(payload, dict) else {}

    def _conversation_session_for(self) -> str | None:
        """Return the privacy-bounded Thread for the current anonymous encounter."""

        now = float(self._monotonic_clock())
        current_session_id = self._conversation_session_id
        if current_session_id is not None:
            last_used = self._conversation_session_last_used_monotonic
            idle_expired = (
                last_used is not None and now >= last_used + self._anonymous_encounter_idle_seconds
            )
            if self._conversation_session_closed:
                self._rotate_anonymous_encounter("anonymous_encounter_closed")
            elif idle_expired:
                self._rotate_anonymous_encounter("anonymous_encounter_idle")
            elif self._cached_session_is_active(current_session_id):
                return self._activate_anonymous_encounter(current_session_id, now)
            else:
                self._forget_anonymous_encounter()

        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        get_or_create = getattr(manager, "get_or_create", None)
        if not callable(get_or_create):
            degraded_session_id = self._degraded_conversation_session_id
            if degraded_session_id is None:
                degraded_session_id = f"voice-local-{uuid4().hex}"
                self._degraded_conversation_session_id = degraded_session_id
            return self._activate_anonymous_encounter(degraded_session_id, now)
        try:
            session = get_or_create(channel="voice")
        except Exception as exc:
            logger.warning(
                "VoiceLoop: conversation session unavailable (%s)",
                type(exc).__name__,
            )
            degraded_session_id = self._degraded_conversation_session_id
            if degraded_session_id is None:
                degraded_session_id = f"voice-degraded-{uuid4().hex}"
                self._degraded_conversation_session_id = degraded_session_id
            return self._activate_anonymous_encounter(degraded_session_id, now)
        session_id = str(getattr(session, "session_id", "") or "").strip()
        if not session_id:
            return None
        return self._activate_anonymous_encounter(session_id, now)

    def _activate_anonymous_encounter(self, session_id: str, now: float) -> str:
        self._conversation_session_id = session_id
        self._conversation_session_last_used_monotonic = now
        self._conversation_session_closed = False
        if self._degraded_conversation_session_id != session_id:
            self._degraded_conversation_session_id = None
        return session_id

    def _forget_anonymous_encounter(self) -> None:
        self._conversation_session_id = None
        self._degraded_conversation_session_id = None
        self._conversation_session_last_used_monotonic = None
        self._conversation_session_closed = False

    def _rotate_anonymous_encounter(self, reason: str) -> None:
        self._close_anonymous_encounter(reason)
        self._forget_anonymous_encounter()

    def _close_anonymous_encounter(self, reason: str) -> None:
        session_id = self._conversation_session_id
        if session_id is None or self._conversation_session_closed:
            return
        self._conversation_session_closed = True

        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        close_session = getattr(manager, "close_session", None)
        if not callable(close_session):
            return
        kwargs: dict[str, Any] = {}
        try:
            parameters = signature(close_session).parameters
            accepts_kwargs = any(
                parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
            if accepts_kwargs or "reason" in parameters:
                kwargs["reason"] = reason
        except (TypeError, ValueError):
            pass
        try:
            close_session(session_id, **kwargs)
        except Exception as exc:
            logger.debug(
                "VoiceLoop: failed to close anonymous encounter (%s)",
                type(exc).__name__,
            )

    def _record_local_gateway_turn(
        self,
        conversation_session_id: str | None,
        user_text: str,
        assistant_reply: Any,
        *,
        interaction_turn_id: str | None = None,
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
        if interaction_turn_id:
            if interaction_turn_id in self._projected_direct_turn_ids:
                return
            self._projected_direct_turn_ids.add(interaction_turn_id)
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
                metadata={
                    "runtime_bridge": dict(self._last_runtime_bridge_status or {}),
                    **({"interaction_turn_id": interaction_turn_id} if interaction_turn_id else {}),
                },
            )
        except Exception as exc:
            logger.debug("VoiceLoop: local gateway turn record failed: %s", exc)

    def _cached_session_is_active(self, session_id: str) -> bool:
        manager = getattr(self._voice_runtime_bridge, "session_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if not callable(snapshot):
            return True
        try:
            current = snapshot(session_id)
        except Exception as exc:
            logger.debug(
                "VoiceLoop: conversation session status unavailable (%s)",
                type(exc).__name__,
            )
            return False
        if current is None:
            return False
        return getattr(current, "status", "active") == "active"


def _decision_to_dict(
    decision: InteractionDecision,
    *,
    addressed: bool,
    addressed_by_text: bool,
    wake_authorized: bool,
    wake_source: str,
    followup_active: bool,
    awaiting_confirmation: bool,
) -> dict[str, Any]:
    return {
        "action": decision.action.value,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "addressed": addressed,
        "addressed_by_text": addressed_by_text,
        "wake_authorized": wake_authorized,
        "wake_source": wake_source,
        "followup_active": followup_active,
        "awaiting_confirmation": awaiting_confirmation,
        "should_record_environment": decision.should_record_environment,
    }


def _clean_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        return {}
    if hasattr(snapshot, "to_dict"):
        payload = snapshot.to_dict()
        return payload if isinstance(payload, dict) else {}
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def _audio_error_value(kind: Any) -> str:
    return str(getattr(kind, "value", kind) or "unknown")
