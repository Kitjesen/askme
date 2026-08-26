"""Voice-mode main loop -microphone ->intent routing ->brain pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
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
from askme.pipeline.channels.runtime_bridge_calls import (
    RuntimeBridgeOutcome,
    try_runtime_bridge_turn,
)
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
from askme.runtime.task.voice_lifecycle import (
    VoiceTaskLifecycleService,
    VoiceTaskOperatorContext,
)
from askme.voice.diagnostics.status_privacy import sanitize_voice_status
from askme.voice.realtime.config import SUPPORTED_REALTIME_PROVIDERS
from askme.voice.realtime.policy import decide_realtime_route

if TYPE_CHECKING:
    from askme.pipeline.core.brain_pipeline import BrainPipeline
    from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
    from askme.robot_interaction import IntentRouter
    from askme.runtime.task.voice_lifecycle import DeliveryState, TaskLifecycleEvent
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
    if not isinstance(value, (str, int, float)):
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
_EXTERNAL_STATUS_REPORT_MARKERS: tuple[str, ...] = (
    "状态报告",
    "进度报告",
    "巡检报告",
    "情况报告",
    "生成报告",
    "整理报告",
    "输出报告",
    "汇报状态",
    "汇报进度",
)


def _task_prepare_message(task_type: str, *, confirmed: bool = False) -> str:
    prefix = "确认收到，" if confirmed else "好的，"
    labels = {
        "status_report": "状态报告任务",
        "inspection_patrol": "区域巡检任务",
        "navigate_to": "导航任务",
    }
    return f"{prefix}我准备提交{labels.get(task_type, '外部任务')}。"


def _is_image_artifact(artifact: Mapping[str, Any]) -> bool:
    media_type = str(
        artifact.get("mime_type")
        or artifact.get("content_type")
        or artifact.get("type")
        or artifact.get("kind")
        or ""
    ).lower()
    return media_type.startswith("image/") or media_type in {"image", "photo", "picture"}


def _task_observation_summary(observation: Mapping[str, Any]) -> str:
    value = (
        observation.get("summary")
        or observation.get("message")
        or observation.get("description")
        or observation.get("value")
        or ""
    )
    text = str(value).strip()
    return text if len(text) <= 120 else f"{text[:117]}..."


def _call_lifecycle_with_operator(
    method: Callable[..., Any],
    *args: Any,
    operator_context: VoiceTaskOperatorContext,
    **kwargs: Any,
) -> Any:
    """Pass per-turn identity while preserving narrow test/adapter protocols."""

    try:
        parameters = tuple(signature(method).parameters.values())
    except (TypeError, ValueError):
        parameters = ()
    supports_context = any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "operator_context"
        for parameter in parameters
    )
    if supports_context:
        kwargs["operator_context"] = operator_context
    return method(*args, **kwargs)


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
        voice_task_lifecycle: VoiceTaskLifecycleService | None = None,
        voice_task_operator_provider: (
            Callable[[str, str], VoiceTaskOperatorContext | Mapping[str, Any] | None] | None
        ) = None,
        dispatcher: SkillDispatcher | None = None,
        audio_router: AudioRouterPort | None = None,
        anonymous_encounter_idle_seconds: float = 25.0,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._router = router
        self._pipeline = pipeline
        self._audio = audio
        self._voice_runtime_bridge = voice_runtime_bridge
        self._voice_task_lifecycle = voice_task_lifecycle
        self._voice_task_operator_provider = voice_task_operator_provider
        self._voice_task_operator_by_session: dict[str, VoiceTaskOperatorContext] = {}
        self._voice_task_operator_by_turn: dict[str, VoiceTaskOperatorContext] = {}
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
        self._task_notification_active = False
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
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Open, deliver, settle, then project one deterministic voice reply."""

        metadata = {"interaction": interaction, **dict(metadata or {})}
        trusted_operator = self._trusted_operator_for_turn(
            conversation_session_id,
            interaction_turn_id,
        )
        external_turn = begin_external_turn(
            self._pipeline,
            user_text,
            source="voice",
            channel="voice",
            conversation_session_id=conversation_session_id,
            person_id=(trusted_operator.person_id or None) if trusted_operator else None,
            operator_id=(trusted_operator.operator_id or None) if trusted_operator else None,
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
                    "person_id": (
                        trusted_operator.person_id or None if trusted_operator else None
                    ),
                    "operator_id": (
                        trusted_operator.operator_id or None if trusted_operator else None
                    ),
                }
                parameters: Mapping[str, Parameter]
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
            metadata=metadata,
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
        ledger_provider = "qwen" if provider_selector == "qwen3_5_omni" else "volcengine"
        realtime_source = f"{ledger_provider}_realtime"
        provider_session_id = str(provider_context.get("provider_session_id") or "").strip() or None
        provider_dialog_id = (
            str(
                provider_context.get("provider_dialog_id")
                or provider_context.get("dialog_id")
                or ""
            ).strip()
            or None
        )
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
            trusted_operator = self._trusted_operator_for_turn(
                conversation_session_id,
                voice_turn_id,
            )
            external_turn = begin_external_turn(
                self._pipeline,
                user_text,
                source=realtime_source,
                conversation_session_id=conversation_session_id,
                person_id=(trusted_operator.person_id or None) if trusted_operator else None,
                operator_id=(trusted_operator.operator_id or None) if trusted_operator else None,
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
        if self._voice_task_lifecycle is not None:
            await self._voice_task_lifecycle.start()
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
        if self._voice_task_lifecycle is not None:
            await self._voice_task_lifecycle.close()

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
        if not self._task_notification_active:
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

    async def _next_voice_activity(
        self,
    ) -> tuple[_CapturedUtterance | None, bool]:
        """Wait for one capture or one task event without duplicating capture."""

        lifecycle = self._voice_task_lifecycle
        if lifecycle is None or not self._refresh_full_duplex_state():
            return await self._next_utterance(), False

        self._start_next_listen()
        listen_task = self._listen_task
        if listen_task is None:
            raise RuntimeError("full-duplex listener was not started")
        thread_id = self._conversation_session_for()
        if not thread_id:
            return await self._next_utterance(), False

        operator = self._voice_task_operator_for_session(thread_id)
        event_task = asyncio.create_task(
            _call_lifecycle_with_operator(
                lifecycle.wait_ready,
                thread_id,
                operator_context=operator,
            ),
            name="voice-task-event-ready",
        )
        try:
            done, _ = await asyncio.wait(
                {listen_task, event_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            # User input wins a same-loop tie.  The event remains pending in
            # the lifecycle inbox and will be claimed on the next wait.
            if listen_task in done:
                event_task.cancel()
                await asyncio.gather(event_task, return_exceptions=True)
                try:
                    utterance = await listen_task
                finally:
                    if self._listen_task is listen_task:
                        self._listen_task = None
                self._start_next_listen()
                return utterance, False
            if await event_task:
                # Never cancel or replace the single microphone capture owner.
                return None, True
            try:
                utterance = await listen_task
            finally:
                if self._listen_task is listen_task:
                    self._listen_task = None
            self._start_next_listen()
            return utterance, False
        finally:
            if not event_task.done():
                event_task.cancel()
                await asyncio.gather(event_task, return_exceptions=True)

    async def _deliver_next_task_event(self, thread_id: str) -> None:
        lifecycle = self._voice_task_lifecycle
        if lifecycle is None:
            return
        event = _call_lifecycle_with_operator(
            lifecycle.claim_next,
            thread_id,
            operator_context=self._voice_task_operator_for_session(thread_id),
        )
        if event is None:
            return
        if event.kind == "reserved":
            self._settle_task_delivery(event.event_id, "suppressed")
            return

        reply = self._task_event_reply(event)
        if not reply:
            self._settle_task_delivery(event.event_id, "suppressed")
            return
        turn_id = self._task_event_turn_id(event.event_id)
        cancel = AtomicCancellationToken()
        self._active_interaction_cancel = cancel
        self._task_notification_active = True
        try:
            delivered = await self._deliver_direct_reply(
                "",
                reply,
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=cancel,
                interaction="external_task_notification",
                metadata={
                    "task_event_id": event.event_id,
                    "task_event_kind": event.kind,
                    "task_state": event.state,
                    "task_reservation_id": event.reservation_id,
                    "task_run_correlation_id": str(
                        getattr(event, "correlation_id", "") or event.run_id
                    ),
                    "runtime_run_id": event.run_id,
                    "remote_task_id": str(getattr(event, "remote_task_id", "") or ""),
                    "task_turn_id": event.turn_id,
                    "task_originating_thread_id": str(
                        getattr(event, "originating_thread_id", "") or ""
                    ),
                },
            )
            self._settle_task_delivery(
                event.event_id,
                "delivered" if delivered else "interrupted",
            )
            if delivered:
                self._remember_spoken_text(reply)
        except asyncio.CancelledError:
            self._settle_task_delivery(event.event_id, "interrupted")
            raise
        except Exception as exc:
            logger.warning("VoiceLoop: task notification delivery failed: %s", exc)
            retry_delivery = getattr(lifecycle, "retry_delivery", None)
            retried = False
            if callable(retry_delivery):
                try:
                    retried = bool(
                        retry_delivery(
                            event.event_id,
                            error_code="voice_notification_delivery_failed",
                        )
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "VoiceLoop: task notification retry scheduling failed: %s",
                        retry_exc,
                    )
            if not retried:
                self._settle_task_delivery(event.event_id, "interrupted")
        finally:
            self._task_notification_active = False
            if self._active_interaction_cancel is cancel:
                self._active_interaction_cancel = None

    def _settle_task_delivery(self, event_id: str, state: DeliveryState) -> bool:
        """Best-effort bounded receipt settlement that cannot crash the voice loop."""

        lifecycle = self._voice_task_lifecycle
        if lifecycle is None:
            return False
        for attempt in range(2):
            try:
                lifecycle.settle_delivery(event_id, state)
                return True
            except Exception as exc:
                logger.warning(
                    "VoiceLoop: task delivery receipt settlement failed "
                    "(event=%s state=%s attempt=%d): %s",
                    event_id,
                    state,
                    attempt + 1,
                    exc,
                )
        return False

    @staticmethod
    def _task_event_turn_id(event_id: str) -> str:
        digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:24]
        return f"voice-task-event-{digest}"

    @staticmethod
    def _task_event_reply(event: TaskLifecycleEvent) -> str:
        if event.kind == "completed":
            return event.result_summary or event.message or "任务已完成。"
        if event.kind == "failed":
            return event.message or "任务执行失败。"
        if event.kind == "cancelled":
            return event.message or "任务已取消。"
        if event.kind in {"started", "progress"}:
            return event.message or "任务正在处理中。"
        return ""

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
        trusted_operator = self._trusted_operator_for_turn(
            conversation_session_id,
            voice_turn_id,
        )
        kwargs.update(
            self._supported_turn_context_kwargs(
                self._pipeline.process,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
                person_id=(trusted_operator.person_id or None) if trusted_operator else None,
                operator_id=(trusted_operator.operator_id or None) if trusted_operator else None,
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
        person_id: str | None = None,
        operator_id: str | None = None,
    ) -> dict[str, Any]:
        """Return only turn-context keywords accepted by a legacy callable."""

        context = {
            "conversation_session_id": conversation_session_id,
            "voice_turn_id": voice_turn_id,
            "turn_cancel_token": turn_cancel_token,
            "person_id": person_id,
            "operator_id": operator_id,
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

    def _voice_task_operator_for_turn(
        self,
        thread_id: str,
        turn_id: str,
    ) -> VoiceTaskOperatorContext:
        cached = self._voice_task_operator_by_turn.get(turn_id)
        if cached is not None:
            if cached.allows("runtime:read"):
                self._voice_task_operator_by_session[thread_id] = cached
            else:
                self._voice_task_operator_by_session.pop(thread_id, None)
            return cached
        provider = self._voice_task_operator_provider
        payload: VoiceTaskOperatorContext | Mapping[str, Any] | None = None
        if provider is not None:
            try:
                payload = provider(thread_id, turn_id)
            except Exception as exc:
                logger.warning(
                    "VoiceLoop: trusted turn identity provider failed: %s",
                    type(exc).__name__,
                )
        context = VoiceTaskOperatorContext.from_mapping(
            dict(payload) if isinstance(payload, Mapping) else payload
        )
        if context is not None and context.allows("runtime:read"):
            self._voice_task_operator_by_turn[turn_id] = context
            while len(self._voice_task_operator_by_turn) > 64:
                self._voice_task_operator_by_turn.pop(next(iter(self._voice_task_operator_by_turn)))
            self._voice_task_operator_by_session[thread_id] = context
            return context
        self._voice_task_operator_by_session.pop(thread_id, None)
        unverified = VoiceTaskOperatorContext(
            operator_id="",
            roles=(),
            authenticated=False,
            source="voice_turn_unverified",
            permissions=(),
        )
        self._voice_task_operator_by_turn[turn_id] = unverified
        while len(self._voice_task_operator_by_turn) > 64:
            self._voice_task_operator_by_turn.pop(next(iter(self._voice_task_operator_by_turn)))
        return unverified

    def _voice_task_operator_for_session(
        self,
        thread_id: str,
    ) -> VoiceTaskOperatorContext:
        context = self._voice_task_operator_by_session.get(thread_id)
        if context is not None:
            return context
        return VoiceTaskOperatorContext(
            operator_id="",
            roles=(),
            authenticated=False,
            source="voice_session_unverified",
            permissions=(),
        )

    def _trusted_operator_for_turn(
        self,
        thread_id: str | None,
        turn_id: str | None,
    ) -> VoiceTaskOperatorContext | None:
        """Return trusted identity for this turn without mutating gateway session state."""

        if not thread_id or not turn_id:
            return None
        operator = self._voice_task_operator_for_turn(thread_id, turn_id)
        return operator if operator.allows("runtime:read") else None

    def _can_continue_pending_runtime_task(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
    ) -> bool:
        lifecycle = self._voice_task_lifecycle
        checker = getattr(lifecycle, "can_continue_pending_task", None)
        if not callable(checker):
            return False
        operator = self._voice_task_operator_for_turn(thread_id, turn_id)
        try:
            return bool(
                _call_lifecycle_with_operator(
                    checker,
                    user_text,
                    thread_id,
                    operator_context=operator,
                )
            )
        except (LookupError, PermissionError, ValueError):
            return False

    def _can_revise_pending_runtime_task(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
    ) -> bool:
        lifecycle = self._voice_task_lifecycle
        checker = getattr(lifecycle, "can_revise_pending_task", None)
        if not callable(checker):
            return False
        operator = self._voice_task_operator_for_turn(thread_id, turn_id)
        try:
            return bool(
                _call_lifecycle_with_operator(
                    checker,
                    user_text,
                    thread_id,
                    operator_context=operator,
                )
            )
        except (LookupError, PermissionError, ValueError):
            return False

    async def _handle_task_control(
        self,
        skill_name: str,
        *,
        user_text: str,
        thread_id: str,
        turn_id: str,
        interaction_cancel: CancellationToken,
    ) -> None:
        lifecycle = self._voice_task_lifecycle
        if lifecycle is None:
            reply = "当前没有接入外部任务服务。"
            metadata: dict[str, Any] = {"task_state": "unavailable"}
        elif skill_name == "task_confirm":
            operator = self._voice_task_operator_for_turn(thread_id, turn_id)
            try:
                reservation = _call_lifecycle_with_operator(
                    lifecycle.confirm_pending,
                    thread_id,
                    turn_id,
                    operator_context=operator,
                )
            except PermissionError:
                reply = "当前语音操作者未通过任务确认授权。"
                metadata = {"task_state": "unauthorized"}
            except TimeoutError:
                reply = "任务确认已过期，请重新发起任务。"
                metadata = {"task_state": "expired"}
            except LookupError:
                reply = "当前没有等待确认的外部任务。"
                metadata = {"task_state": "idle"}
            except RuntimeError as exc:
                reply = "任务确认上下文已失效，请重新发起任务。"
                metadata = {"task_state": "invalid", "reason": str(exc)}
            else:
                await self._submit_reserved_task(
                    reservation,
                    user_text=user_text,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    interaction_cancel=interaction_cancel,
                    operator_context=operator,
                    acknowledgement=_task_prepare_message(
                        reservation.task_type,
                        confirmed=True,
                    ),
                )
                return
        elif skill_name == "task_evidence":
            operator = self._voice_task_operator_for_turn(thread_id, turn_id)
            try:
                report = _call_lifecycle_with_operator(
                    lifecycle.task_report,
                    thread_id,
                    operator_context=operator,
                )
            except PermissionError:
                reply = "当前语音操作者未通过任务证据查询授权。"
                metadata = {"task_state": "unauthorized"}
            else:
                artifacts = [
                    dict(item)
                    for item in report.get("artifacts", [])
                    if isinstance(item, dict)
                ]
                observations = [
                    dict(item)
                    for item in report.get("observations", [])
                    if isinstance(item, dict)
                ]
                status = str(report.get("status") or "idle")
                metadata = {
                    "task_state": status,
                    "runtime_run_id": str(report.get("run_id") or ""),
                    "task_artifact_count": len(artifacts),
                    "task_observation_count": len(observations),
                    "task_artifacts": artifacts[:20],
                    "task_observations": observations[:20],
                }
                if artifacts:
                    image_count = sum(1 for item in artifacts if _is_image_artifact(item))
                    detail = f"，其中{image_count}张图片" if image_count else ""
                    reply = f"已找到{len(artifacts)}个任务证据文件{detail}，已附在任务记录中。"
                elif observations:
                    summary = _task_observation_summary(observations[0])
                    reply = f"任务返回了{len(observations)}条结构化观察。{summary}".rstrip()
                elif status in {"queued", "executing", "cancelling", "submission_unknown"}:
                    reply = "任务还在进行，暂未返回照片或结构化证据。"
                elif report:
                    reply = "任务报告已生成，但执行器没有返回照片或结构化证据。"
                else:
                    reply = "当前没有可查询的任务报告。"
        elif skill_name == "task_cancel":
            operator = self._voice_task_operator_for_turn(thread_id, turn_id)
            clarification_cancelled = False
            cancel_clarification = getattr(lifecycle, "cancel_pending_clarification", None)
            if callable(cancel_clarification):
                try:
                    clarification_cancelled = bool(
                        _call_lifecycle_with_operator(
                            cancel_clarification,
                            thread_id,
                            operator_context=operator,
                        )
                    )
                except PermissionError:
                    clarification_cancelled = False
            if clarification_cancelled:
                reply = "已取消等待补充目标的任务，没有提交外部执行器。"
                metadata = {
                    "task_state": "cancelled",
                    "task_cancel_error_code": "pending_clarification_cancelled",
                    "remote_cancel_acknowledged": False,
                }
            else:
                result = await _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    operator_context=operator,
                )
                snapshot = result.snapshot
                metadata = {
                    "task_state": snapshot.state,
                    "task_reservation_id": snapshot.reservation_id,
                    "runtime_run_id": snapshot.run_id,
                    "remote_task_id": snapshot.remote_task_id,
                    "task_turn_id": snapshot.turn_id,
                    "remote_cancel_acknowledged": result.remote_acknowledged,
                    "task_cancel_error_code": result.error_code,
                }
                if result.error_code == "no_active_external_task":
                    reply = "当前没有正在执行的外部任务。"
                elif result.error_code == "operator_not_authorized":
                    reply = "当前语音操作者未通过任务取消授权。"
                elif result.error_code == "cancel_deferred_until_reconciled":
                    reply = "已记录取消请求；提交对账完成后会立即取消，不会重复提交。"
                elif result.error_code == "pending_task_cancelled":
                    reply = "已取消待确认任务，没有提交外部执行器。"
                elif result.remote_acknowledged:
                    reply = "已向外部任务发送取消请求，我会继续同步最终状态。"
                else:
                    reply = "取消请求暂未被外部执行器确认，我会继续同步任务状态。"
        else:
            operator = self._voice_task_operator_for_turn(thread_id, turn_id)
            try:
                snapshot = _call_lifecycle_with_operator(
                    lifecycle.status_snapshot,
                    thread_id,
                    operator_context=operator,
                )
            except PermissionError:
                reply = "当前语音操作者未通过任务状态查询授权。"
                metadata = {"task_state": "unauthorized"}
                self._audio.drain_buffers()
                await self._deliver_direct_reply(
                    user_text,
                    reply,
                    conversation_session_id=thread_id,
                    interaction_turn_id=turn_id,
                    interaction_cancel=interaction_cancel,
                    interaction=skill_name,
                    metadata=metadata,
                )
                return
            metadata = {
                "task_state": snapshot.state,
                "task_reservation_id": snapshot.reservation_id,
                "runtime_run_id": snapshot.run_id,
                "remote_task_id": snapshot.remote_task_id,
                "task_turn_id": snapshot.turn_id,
            }
            if not snapshot.reservation_id:
                pending = None
                pending_lookup = getattr(lifecycle, "pending_clarification", None)
                if callable(pending_lookup):
                    try:
                        pending = _call_lifecycle_with_operator(
                            pending_lookup,
                            thread_id,
                            operator_context=operator,
                        )
                    except PermissionError:
                        pending = None
                if pending is not None:
                    metadata.update(
                        {
                            "task_state": "collecting_parameters",
                            "task_clarification_id": pending.clarification_id,
                            "task_type": pending.task_type,
                            "missing_parameter": pending.missing_parameter,
                        }
                    )
                    reply = "任务正在等待目标区域，请告诉我要前往或巡检哪里。"
                else:
                    reply = "当前没有外部任务记录。"
            elif snapshot.active:
                reply = f"当前任务状态是{snapshot.state}。"
            elif snapshot.result_summary:
                reply = f"任务状态是{snapshot.state}。{snapshot.result_summary}"
            else:
                reply = f"最近任务状态是{snapshot.state}。"
        self._audio.drain_buffers()
        await self._deliver_direct_reply(
            user_text,
            reply,
            conversation_session_id=thread_id,
            interaction_turn_id=turn_id,
            interaction_cancel=interaction_cancel,
            interaction=skill_name,
            metadata=metadata,
        )

    @staticmethod
    def _is_bounded_external_status_report(user_text: str) -> bool:
        normalized = re.sub(r"\s+", "", user_text)
        return any(marker in normalized for marker in _EXTERNAL_STATUS_REPORT_MARKERS)

    async def _handle_pending_runtime_task_revision(
        self,
        *,
        user_text: str,
        thread_id: str,
        turn_id: str,
        interaction_cancel: CancellationToken,
    ) -> None:
        lifecycle = self._voice_task_lifecycle
        revise = getattr(lifecycle, "revise_pending_task", None)
        if lifecycle is None or not callable(revise):
            return
        operator = self._voice_task_operator_for_turn(thread_id, turn_id)
        try:
            reservation = _call_lifecycle_with_operator(
                revise,
                user_text,
                thread_id,
                turn_id,
                operator_context=operator,
            )
        except PermissionError:
            reply = "当前语音操作者未通过任务修改授权。"
            metadata: dict[str, Any] = {"task_state": "unauthorized", "submitted": False}
        except LookupError:
            reply = "当前没有等待确认、可以修改的任务。"
            metadata = {"task_state": "idle", "submitted": False}
        except (RuntimeError, ValueError) as exc:
            reply = "任务修改失败，请重新说完整任务。"
            metadata = {
                "task_state": "revision_failed",
                "submitted": False,
                "reason": str(exc),
            }
        else:
            reply = f"已修改。{reservation.confirmation_prompt}"
            metadata = {
                "task_state": reservation.state,
                "task_type": reservation.task_type,
                "task_target": reservation.target,
                "task_reservation_id": reservation.reservation_id,
                "runtime_run_id": reservation.run_id,
                "approval_id": reservation.approval_id,
                "task_revision": reservation.revision,
                "supersedes_reservation_id": reservation.supersedes_reservation_id,
                "submitted": False,
            }
        self._audio.drain_buffers()
        await self._deliver_direct_reply(
            user_text,
            reply,
            conversation_session_id=thread_id,
            interaction_turn_id=turn_id,
            interaction_cancel=interaction_cancel,
            interaction="external_task_revised",
            metadata=metadata,
        )

    async def _handle_local_external_task_start(
        self,
        *,
        user_text: str,
        thread_id: str,
        turn_id: str,
        interaction_cancel: CancellationToken,
        continue_pending: bool = False,
    ) -> None:
        lifecycle = self._voice_task_lifecycle
        if lifecycle is None:
            self._audio.drain_buffers()
            await self._deliver_direct_reply(
                user_text,
                "当前没有接入可跟踪的外部任务服务。",
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                interaction="external_task_rejected",
                metadata={"task_state": "unsupported", "submitted": False},
            )
            return

        operator = self._voice_task_operator_for_turn(thread_id, turn_id)
        try:
            reserve_task = (
                getattr(lifecycle, "continue_pending_task")
                if continue_pending
                else lifecycle.reserve_task
            )
            reservation = _call_lifecycle_with_operator(
                reserve_task,
                user_text,
                thread_id,
                turn_id,
                operator_context=operator,
            )
        except PermissionError as exc:
            reason = str(exc)
            reply = (
                "当前没有可信说话人身份，巡检和导航任务未提交。"
                if "physical_task_speaker_identity_required" in reason
                else "当前语音操作者未通过外部任务授权，任务未提交。"
            )
            self._audio.drain_buffers()
            await self._deliver_direct_reply(
                user_text,
                reply,
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                interaction="external_task_unauthorized",
                metadata={"task_state": "unauthorized", "submitted": False},
            )
            return
        except LookupError:
            self._audio.drain_buffers()
            await self._deliver_direct_reply(
                user_text,
                "上一次任务补充已经过期，请重新说完整任务。",
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                interaction="external_task_clarification_expired",
                metadata={"task_state": "expired", "submitted": False},
            )
            return
        except (RuntimeError, ValueError) as exc:
            reason = str(exc)
            if "task_target_required" in reason:
                reply = "要前往或巡检哪个目标区域，例如 A 区或北门？"
                task_state = "collecting_parameters"
            elif "voice_task_already_active" in reason:
                reply = "当前已有任务，请先查询任务状态、取消任务或等待任务完成。"
                task_state = "unsupported"
            elif "mission_service_unavailable" in reason:
                reply = "当前任务规划服务不可用，任务没有提交。"
                task_state = "unsupported"
            else:
                reply = "当前语音任务只支持状态报告、区域巡检和导航，任务没有提交。"
                task_state = "unsupported"
            self._audio.drain_buffers()
            await self._deliver_direct_reply(
                user_text,
                reply,
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                interaction="external_task_rejected",
                metadata={
                    "task_state": task_state,
                    "submitted": False,
                    "reason": reason,
                },
            )
            return
        if reservation.state == "waiting_user":
            self._audio.drain_buffers()
            try:
                delivered = await self._deliver_direct_reply(
                    user_text,
                    reservation.confirmation_prompt,
                    conversation_session_id=thread_id,
                    interaction_turn_id=turn_id,
                    interaction_cancel=interaction_cancel,
                    interaction="external_task_confirmation_required",
                    metadata={
                        "task_state": reservation.state,
                        "task_type": reservation.task_type,
                        "task_target": reservation.target,
                        "task_reservation_id": reservation.reservation_id,
                        "runtime_run_id": reservation.run_id,
                        "approval_id": reservation.approval_id,
                        "submitted": False,
                    },
                )
            except asyncio.CancelledError:
                cleanup = _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    reason="voice_confirmation_prompt_cancelled",
                    operator_context=operator,
                )
                try:
                    await asyncio.shield(cleanup)
                except Exception as exc:
                    logger.warning(
                        "VoiceLoop: cancelled confirmation cleanup failed: %s",
                        exc,
                    )
                raise
            except Exception:
                await _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    reason="voice_confirmation_prompt_failed",
                    operator_context=operator,
                )
                raise
            if not delivered:
                await _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    reason="voice_confirmation_prompt_interrupted",
                    operator_context=operator,
                )
            return
        await self._submit_reserved_task(
            reservation,
            user_text=user_text,
            thread_id=thread_id,
            turn_id=turn_id,
            interaction_cancel=interaction_cancel,
            operator_context=operator,
            acknowledgement=_task_prepare_message(reservation.task_type),
        )
        return

    async def _submit_reserved_task(
        self,
        reservation: Any,
        *,
        user_text: str,
        thread_id: str,
        turn_id: str,
        interaction_cancel: CancellationToken,
        operator_context: VoiceTaskOperatorContext,
        acknowledgement: str,
    ) -> None:
        lifecycle = self._voice_task_lifecycle
        if lifecycle is None:
            return
        ack_metadata = {
            "task_state": reservation.state,
            "task_reservation_id": reservation.reservation_id,
            "task_run_correlation_id": reservation.reservation_id,
            "runtime_run_id": reservation.run_id,
            "task_turn_id": reservation.turn_id,
        }
        self._audio.drain_buffers()
        try:
            delivered = await self._deliver_direct_reply(
                user_text,
                acknowledgement,
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                interaction="external_task_ack",
                metadata=ack_metadata,
            )
        except asyncio.CancelledError:
            if not lifecycle.abandon(reservation.reservation_id):
                cleanup = _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    reason="voice_submission_ack_cancelled",
                    operator_context=operator_context,
                )
                try:
                    await asyncio.shield(cleanup)
                except Exception as exc:
                    logger.warning(
                        "VoiceLoop: cancelled submission acknowledgement cleanup failed: %s",
                        exc,
                    )
            raise
        except Exception:
            if not lifecycle.abandon(reservation.reservation_id):
                await _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    reason="voice_submission_ack_failed",
                    operator_context=operator_context,
                )
            raise
        if not delivered:
            if not lifecycle.abandon(reservation.reservation_id):
                await _call_lifecycle_with_operator(
                    lifecycle.cancel_active,
                    thread_id,
                    reason="voice_submission_ack_interrupted",
                    operator_context=operator_context,
                )
            return
        try:
            handle = await _call_lifecycle_with_operator(
                lifecycle.commit_ack_and_submit,
                reservation.reservation_id,
                operator_context=operator_context,
            )
        except Exception as exc:
            logger.error("VoiceLoop: acknowledged external task submission failed: %s", exc)
            snapshot = None
            try:
                snapshot = _call_lifecycle_with_operator(
                    lifecycle.status_snapshot,
                    thread_id,
                    operator_context=operator_context,
                )
            except Exception:
                pass
            remote_may_be_running = bool(
                snapshot is not None
                and (
                    snapshot.remote_task_id
                    or snapshot.state
                    in {"queued", "submission_unknown", "cancel_requested", "executing"}
                )
            )
            failure_reply = (
                "任务可能已进入外部执行器，但本地提交记录暂时无法确认；"
                "我会使用同一任务标识继续对账，请不要重复提交。"
                if remote_may_be_running
                else "任务提交失败，没有进入外部执行器，请稍后重试。"
            )
            await self._deliver_direct_reply(
                "",
                failure_reply,
                conversation_session_id=thread_id,
                interaction_turn_id=self._task_event_turn_id(
                    f"{reservation.reservation_id}:submit-failed"
                ),
                interaction_cancel=interaction_cancel,
                interaction="external_task_submit_failed",
                metadata={
                    "task_state": (
                        snapshot.state if remote_may_be_running and snapshot is not None else "failed"
                    ),
                    "task_reservation_id": reservation.reservation_id,
                    "task_run_correlation_id": reservation.reservation_id,
                    "runtime_run_id": reservation.run_id,
                    "task_turn_id": reservation.turn_id,
                    "submitted": "unknown" if remote_may_be_running else False,
                    "remote_may_be_running": remote_may_be_running,
                },
            )
            return
        if handle.state == "submission_unknown":
            acceptance = "提交结果暂时无法确认，我正在使用同一任务标识对账，请不要重复提交。"
            interaction = "external_task_submission_unknown"
            submitted: bool | str = "unknown"
        elif not handle.accepted:
            acceptance = f"任务没有被外部执行器接受，当前状态是{handle.state}。"
            interaction = "external_task_not_accepted"
            submitted = False
        else:
            acceptance = (
                "任务已受理，完成后我会播报结果。"
                if self._refresh_full_duplex_state()
                else "任务已受理。当前是半双工模式，请稍后问我任务状态。"
            )
            interaction = "external_task_accepted"
            submitted = True
        await self._deliver_direct_reply(
            "",
            acceptance,
            conversation_session_id=thread_id,
            interaction_turn_id=self._task_event_turn_id(
                f"{reservation.reservation_id}:accepted"
            ),
            interaction_cancel=interaction_cancel,
            interaction=interaction,
            metadata={
                "task_state": handle.state,
                "task_reservation_id": reservation.reservation_id,
                "task_run_correlation_id": handle.correlation_id,
                "runtime_run_id": handle.run_id,
                "remote_task_id": handle.remote_task_id,
                "task_turn_id": reservation.turn_id,
                "submitted": submitted,
            },
        )

    async def _handle_runtime_task_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None,
        turn_id: str,
        interaction_cancel: CancellationToken,
        continue_pending: bool = False,
    ) -> RuntimeBridgeOutcome:
        """Choose exactly one execution authority for robot-runtime work."""

        if self._voice_task_lifecycle is not None:
            self._last_runtime_bridge_status = {
                **self._runtime_bridge_snapshot(),
                "handled": False,
                "local_fallback": False,
                "disposition": "bypassed_for_taskrun",
                "execution_authority": "local_taskrun",
            }
            if not thread_id:
                self._audio.drain_buffers()
                await self._deliver_direct_reply(
                    user_text,
                    "当前无法建立可恢复的任务会话，任务没有提交。",
                    conversation_session_id=None,
                    interaction_turn_id=turn_id,
                    interaction_cancel=interaction_cancel,
                    interaction="external_task_rejected",
                    metadata={"task_state": "session_unavailable", "submitted": False},
                )
                return RuntimeBridgeOutcome(handled=True, disposition="handled")
            await self._handle_local_external_task_start(
                user_text=user_text,
                thread_id=thread_id,
                turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                continue_pending=continue_pending,
            )
            return RuntimeBridgeOutcome(handled=True, disposition="handled")

        outcome = await self._runtime_bridge_outcome(
            user_text,
            conversation_session_id=thread_id,
            interaction_turn_id=turn_id,
            interaction_cancel=interaction_cancel,
            allow_agent_task_dispatch=False,
        )
        if outcome.explicitly_declined and thread_id:
            await self._handle_local_external_task_start(
                user_text=user_text,
                thread_id=thread_id,
                turn_id=turn_id,
                interaction_cancel=interaction_cancel,
            )
        elif outcome.ambiguous:
            self._audio.drain_buffers()
            await self._deliver_direct_reply(
                user_text,
                "远端执行状态暂时无法确认。为避免重复执行，我没有在本地再次提交，请稍后询问任务状态。",
                conversation_session_id=thread_id,
                interaction_turn_id=turn_id,
                interaction_cancel=interaction_cancel,
                interaction="external_task_reconciliation_required",
                metadata={
                    "task_state": "reconciliation_required",
                    "submitted_locally": False,
                    "runtime_bridge_disposition": outcome.disposition,
                },
            )
        return outcome

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

                utterance, task_event_ready = await self._next_voice_activity()
                if task_event_ready:
                    session_id = self._conversation_session_for()
                    if session_id:
                        await self._deliver_next_task_event(session_id)
                    continue
                if utterance is None:
                    continue
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
                identity_session_id = self._conversation_session_for()
                if identity_session_id:
                    self._voice_task_operator_for_turn(
                        identity_session_id,
                        interaction_turn_id,
                    )

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
                clarification_session_id = self._conversation_session_for()
                pending_task_clarification = bool(
                    not pending_approval
                    and intent.type == IntentType.GENERAL
                    and clarification_session_id
                    and self._can_continue_pending_runtime_task(
                        user_text,
                        clarification_session_id,
                        interaction_turn_id,
                    )
                )
                pending_task_revision = bool(
                    not pending_approval
                    and not pending_task_clarification
                    and intent.type == IntentType.GENERAL
                    and clarification_session_id
                    and self._can_revise_pending_runtime_task(
                        user_text,
                        clarification_session_id,
                        interaction_turn_id,
                    )
                )
                realtime_capture_active = self._realtime_capture_active()
                realtime_general_ready = self._realtime_general_chat_ready()
                realtime_mode = self._realtime_mode()
                realtime_provider = self._realtime_provider()
                realtime_robot_task = bool(
                    pending_task_clarification
                    or pending_task_revision
                    or contains_robot_task_intent(user_text)
                    or str(gate_decision.reason or "").startswith("robot_task")
                    or gate_decision.reason == "unaddressed_robot_task"
                )
                realtime_emergency = contains_emergency_intent(user_text)
                realtime_tool_route = bool(
                    pending_task_clarification
                    or pending_task_revision
                    or intent.type != IntentType.GENERAL
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

                task_control = (
                    intent.skill_name
                    if intent.type == IntentType.VOICE_TRIGGER
                    and intent.skill_name
                    in {"task_status", "task_cancel", "task_confirm", "task_evidence"}
                    else None
                )
                if task_control is not None:
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    session_id = self._conversation_session_for()
                    if session_id:
                        await self._handle_task_control(
                            task_control,
                            user_text=user_text,
                            thread_id=session_id,
                            turn_id=interaction_turn_id,
                            interaction_cancel=interaction_cancel,
                        )
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                if pending_task_clarification and clarification_session_id:
                    await self._handle_runtime_task_turn(
                        user_text=user_text,
                        thread_id=clarification_session_id,
                        turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                        continue_pending=True,
                    )
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                if pending_task_revision and clarification_session_id:
                    await self._handle_pending_runtime_task_revision(
                        user_text=user_text,
                        thread_id=clarification_session_id,
                        turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

                # Supported robot-runtime tasks use the persistent local TaskRun owner.
                # The handler consults the bridge only when that owner is absent,
                # so one turn cannot be submitted through two execution paths.
                if intent.type == IntentType.VOICE_TRIGGER and intent.skill_name == "runtime_task":
                    session_id = self._conversation_session_for()
                    await self._handle_runtime_task_turn(
                        user_text=user_text,
                        thread_id=session_id,
                        turn_id=interaction_turn_id,
                        interaction_cancel=interaction_cancel,
                    )
                    if idle_task and not idle_task.done():
                        idle_task.cancel()
                    idle_task = self._pipeline.start_idle_reflection()
                    continue

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

                # Stop speaking controls playback only.  Task cancellation is
                # a separate explicit lifecycle command.
                if intent.type == IntentType.VOICE_TRIGGER and intent.skill_name == "stop_speaking":
                    if memory_task and not memory_task.done():
                        memory_task.cancel()
                        memory_task = None
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
        """Consume handled/ambiguous bridge turns; fall back only on explicit decline."""
        outcome = await self._runtime_bridge_outcome(
            user_text,
            conversation_session_id=conversation_session_id,
            interaction_turn_id=interaction_turn_id,
            interaction_cancel=interaction_cancel,
        )
        if outcome.ambiguous:
            self._audio.drain_buffers()
            await self._deliver_direct_reply(
                user_text,
                "远端处理状态暂时无法确认。为避免重复处理，本次没有切换到本地执行。",
                conversation_session_id=conversation_session_id,
                interaction_turn_id=interaction_turn_id,
                interaction_cancel=interaction_cancel,
                interaction="runtime_bridge_reconciliation_required",
                metadata={
                    "runtime_bridge_disposition": outcome.disposition,
                    "local_fallback": False,
                },
            )
            return True
        return outcome.handled

    async def _runtime_bridge_outcome(
        self,
        user_text: str,
        *,
        conversation_session_id: str | None,
        interaction_turn_id: str | None,
        interaction_cancel: CancellationToken | None,
        allow_agent_task_dispatch: bool = True,
    ) -> RuntimeBridgeOutcome:
        """Return a three-state bridge result for execution-authority decisions."""

        if self._voice_runtime_bridge is None:
            return RuntimeBridgeOutcome()

        operator = (
            self._voice_task_operator_for_turn(
                conversation_session_id,
                interaction_turn_id,
            )
            if conversation_session_id and interaction_turn_id
            else None
        )
        trusted_context = (
            operator if operator is not None and operator.allows("runtime:read") else None
        )
        try:
            outcome = await try_runtime_bridge_turn(
                self._voice_runtime_bridge.handle_voice_text,
                user_text,
                conversation_session_id=conversation_session_id,
                voice_turn_id=interaction_turn_id,
                turn_cancel_token=interaction_cancel,
                person_id=(trusted_context.person_id or None) if trusted_context else None,
                operator_id=(trusted_context.operator_id or None) if trusted_context else None,
                metadata=(
                    {
                        "operator_authenticated": True,
                        "operator_source": trusted_context.source,
                    }
                    if trusted_context is not None
                    else None
                ),
                pipeline=self._pipeline,
                dispatcher=self._dispatcher,
                on_spoken_reply=self._safe_runtime_spoken_reply,
                label="Voice",
                allow_agent_task_dispatch=allow_agent_task_dispatch,
            )
            status = self._runtime_bridge_snapshot()
            status.update(
                {
                    "handled": outcome.handled,
                    "local_fallback": outcome.explicitly_declined,
                    "disposition": outcome.disposition,
                }
            )
            self._last_runtime_bridge_status = status
            return outcome
        except Exception as exc:
            logger.warning(
                "VoiceLoop: runtime bridge outcome is ambiguous: %s",
                exc,
            )
            self._last_runtime_bridge_status = {
                **self._runtime_bridge_snapshot(),
                "handled": False,
                "local_fallback": False,
                "disposition": "ambiguous",
                "last_status": "exception",
                "last_error_type": type(exc).__name__,
            }
            return RuntimeBridgeOutcome(disposition="ambiguous")

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
        metadata: dict[str, Any] | None = None,
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
                    **dict(metadata or {}),
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
