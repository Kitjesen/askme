"""AudioAgent - high-level voice I/O controller composing ASR, VAD, KWS, and TTS engines."""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast
from uuid import uuid4

import numpy as np

try:
    import sounddevice as sd
except ModuleNotFoundError:
    # Minimal stub so tests can patch sd.play / sd.InputStream without needing hardware
    class _CallbackAbort(Exception):
        pass

    class _CallbackStop(Exception):
        pass

    class _SoundDeviceStub:
        InputStream = None
        OutputStream = None
        CallbackAbort = _CallbackAbort
        CallbackStop = _CallbackStop

        @staticmethod
        def play(*args: object, **kwargs: object) -> None: ...
        @staticmethod
        def stop() -> None: ...
        @staticmethod
        def wait() -> None: ...
        @staticmethod
        def query_devices(device: object = None, kind: object = None) -> object:
            return {}

    sd = _SoundDeviceStub()  # type: ignore[assignment]

from askme.robot_interaction.routing_policy import (
    DEFAULT_QUICK_REPLIES,
    is_local_safety_utterance,
)
from askme.telemetry.ota_bridge import OTABridgeMetrics, get_ota_runtime_metrics
from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.punctuation import PunctuationRestorer
from askme.voice.core.turn_trace import VoiceTurnTraceRecorder
from askme.voice.input.asr_manager import (
    _CONFIRMATION_WORDS,  # noqa: F401 — re-exported for tests
    _MIN_VALID_TEXT_LEN,  # noqa: F401 — re-exported for tests
    _NOISE_UTTERANCES,  # noqa: F401 — re-exported for tests
    _SINGLE_CHAR_COMMANDS,  # noqa: F401 — re-exported for tests
    ASRManager,
)
from askme.voice.input.audio_processor import AudioProcessor
from askme.voice.input.fast_endpoint import FastEndpointAction, FastEndpointController
from askme.voice.input.kws import KWSEngine
from askme.voice.input.mic_input import MicInput
from askme.voice.input.vad_controller import (
    _BARGE_IN_HOLD_S,  # noqa: F401 — re-exported for tests
    _MAX_SPEECH_DURATION,  # noqa: F401 — re-exported for tests
    VADController,
    VADEvent,
)
from askme.voice.orchestration.interrupt_recovery import (
    InterruptionRecoveryCoordinator,
    InterruptionRecoveryState,
    PlaybackHoldPort,
)
from askme.voice.output.audio_router import AudioRouter
from askme.voice.output.phrase_prime import WAITING_FEEDBACK_CACHE_KEY
from askme.voice.output.tts import TTSEngine

logger = logging.getLogger(__name__)

# Default ASR timeout (overridden by config voice.asr.asr_timeout)
_DEFAULT_ASR_TIMEOUT = 10.0


class AgentState(Enum):
    """Observable lifecycle state of the audio agent.

    Transitions:
        IDLE → LISTENING (VAD triggers speech detection)
        LISTENING → PROCESSING (ASR endpoint detected, returning text)
        PROCESSING → SPEAKING (TTS playback starts)
        SPEAKING → IDLE (TTS finishes or barge-in)
        Any → MUTED (mute() called)
        MUTED → IDLE (unmute() called)
    """

    IDLE = "idle"  # Waiting for wake word / user input
    LISTENING = "listening"  # VAD active, collecting speech (speech_active=True)
    PROCESSING = "processing"  # ASR done, text returned, LLM/skill running
    SPEAKING = "speaking"  # TTS is playing back audio
    MUTED = "muted"  # Microphone muted by user


@dataclass(frozen=True, slots=True)
class OutputPlaybackTraceToken:
    """Immutable ownership fence for one physical playback interval."""

    voice_turn_id: str | None
    epoch: int
    provider_generation: int | None = None
    transport_generation: int | None = None


class AudioAgent:
    """Unified audio agent that manages microphone listening, wake word detection,
    VAD-gated speech recognition, and text-to-speech output.

    Config dict expected structure::

        voice:
          asr: { ... }
          vad: { ... }
          kws: { ... }
          tts: { ... }

    Parameters
    ----------
    config : dict
        Full configuration dict. The ``voice`` sub-dict is read automatically.
    voice_mode : bool
        If True (default), initialise ASR/VAD/KWS for microphone input.
        If False, only TTS is initialised (text-input mode).
    """

    def __init__(
        self,
        config: dict[str, Any],
        voice_mode: bool = True,
        *,
        metrics: OTABridgeMetrics | None = None,
        audio_router: AudioRouter | None = None,
        turn_traces: VoiceTurnTraceRecorder | None = None,
    ) -> None:
        voice_cfg = config.get("voice", {})
        readiness_cfg = voice_cfg.get("product_readiness", {})
        if not isinstance(readiness_cfg, dict):
            readiness_cfg = {}
        self.voice_mode = voice_mode
        self._require_wake_word = bool(readiness_cfg.get("require_wake_word", True))
        self._metrics = metrics or get_ota_runtime_metrics()
        self._audio_router = audio_router

        # Shared state
        self.audio_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.woken_up: bool = False
        self.last_turn_wake_authorized: bool = False
        self.last_turn_asr_confidence: float | None = None
        self.last_turn_wake_source: str = "none"
        self.last_accepted_voice_turn_id: str | None = None
        self._active_capture_voice_turn_id: str | None = None
        self.last_turn_operator_context: dict[str, Any] | None = None
        self._voice_turn_identity_lock = threading.RLock()
        self._init_output_ownership_state()
        self._muted: bool = False  # software mute — still listens, VoiceLoop filters results
        self._agent_state: AgentState = AgentState.IDLE
        # When True, confirmation words bypass the noise filter.
        self.awaiting_confirmation: bool = False
        self._chime_lock = threading.RLock()
        self._last_chime_at: float = 0.0
        self._last_thinking_chime_at: float = 0.0
        self._feedback_generation: int = 0
        self._feedback_active: bool = False
        self._feedback_event: str | None = None
        self._feedback_process: Any | None = None
        self._feedback_sounddevice_active: bool = False
        self._feedback_sounddevice_cancel_event: threading.Event | None = None
        feedback_cfg = voice_cfg.get("feedback", {}) or {}
        self._spoken_wait_prompt_enabled: bool = bool(
            feedback_cfg.get("spoken_wait_prompt_enabled", False)
        )
        self._spoken_wait_prompt_text: str = str(feedback_cfg.get("text", "") or "").strip()
        self._spoken_wait_prompt_cache_key: str = str(
            feedback_cfg.get("cache_key", "") or ""
        ).strip()
        if (
            self._spoken_wait_prompt_cache_key != WAITING_FEEDBACK_CACHE_KEY
            or not self._spoken_wait_prompt_text
        ):
            self._spoken_wait_prompt_enabled = False
            self._spoken_wait_prompt_text = ""
            self._spoken_wait_prompt_cache_key = ""
        self._processing_feedback_delay_s = self._finite_nonnegative_feedback_seconds(
            feedback_cfg.get("delay_s", 1.5),
            default=1.5,
        )
        self._spoken_wait_prompt_min_interval_s = self._finite_nonnegative_feedback_seconds(
            feedback_cfg.get("min_interval_s", 8.0),
            default=8.0,
        )
        self._last_spoken_wait_prompt_at: float = 0.0
        self._processing_feedback_generation: int = 0
        self._processing_feedback_armed: bool = False
        self._processing_feedback_timer: threading.Timer | None = None
        self._processing_feedback_armed_total: int = 0
        self._processing_feedback_triggered_total: int = 0
        self._processing_feedback_started_total: int = 0
        self._processing_feedback_cancelled_total: int = 0
        self._processing_feedback_suppressed_total: int = 0
        self._processing_feedback_overlap_prevented_total: int = 0
        self._processing_feedback_last_transition: str = "idle"
        self._runtime_switch_lock = threading.RLock()
        self._tts_activation_callback: Callable[[TTSEngine], None] | None = None
        self._runtime_switch_callback: Callable[[dict[str, Any]], None] | None = None
        self._listen_loop_active = False
        self._listen_cancel_event: threading.Event | None = None
        self._listen_loop_stopped = threading.Event()
        self._listen_loop_stopped.set()
        self._listen_thread_id: int | None = None
        self._input_requested = False
        self._asr_phase_active = False
        self._pending_asr_config: dict[str, Any] | None = None
        self._pending_tts_config: dict[str, Any] | None = None
        self._barge_in_callback: Callable[[], None] | None = None
        self._capture_processor: Callable[[np.ndarray, int, bool], np.ndarray] | None = None
        self._capture_processor_failure_callback: Callable[[BaseException], None] | None = None
        self._capture_processor_error_logged = False
        self._capture_processor_failed_current_frame = False
        # Optional cloud speech-to-speech stays behind the local microphone,
        # AEC, intent, and safety owners.  It never opens a second audio device.
        self._realtime_coordinator: Any | None = None
        self._realtime_mode = "split"
        self._realtime_output_lock = threading.Lock()
        self._realtime_output_tts_generation: int | None = None
        self._realtime_output_provider_generation = 0
        self._realtime_output_terminated_provider_generation = 0
        self._realtime_output_started = False
        self._realtime_output_voice_turn_id: str | None = None
        self._realtime_last_physical_played_ms = 0
        self._realtime_generation_at_listen_start = 0
        self.last_turn_realtime_generation = 0
        self.last_turn_realtime_baseline_generation = 0
        self._realtime_recovery_lock = threading.Lock()
        self._realtime_recovery_stop = threading.Event()
        self._realtime_recovery_thread: threading.Thread | None = None
        self._realtime_capture_armed = False
        self._realtime_faulted_coordinator: Any | None = None
        self._realtime_recovery_attempts = 0
        self._realtime_recovery_successes = 0
        self._realtime_recovery_failures = 0
        self._realtime_recovery_last_error = ""

        # Wake timeout: after wake word detection + successful interaction,
        # stay "awake" for this many seconds so the user can continue chatting
        # without repeating the wake word.  0 = require wake word every time.
        self._wake_timeout: float = float(voice_cfg.get("wake_timeout", 30.0))
        self._last_interaction_time: float = 0.0
        self._post_tts_input_cooldown_s: float = max(
            0.0,
            float(voice_cfg.get("post_tts_input_cooldown_s", 0.0)),
        )
        self._ready_chime_enabled: bool = bool(voice_cfg.get("ready_chime_enabled", False))
        self._ready_chime_min_interval_s: float = max(
            0.0,
            float(voice_cfg.get("ready_chime_min_interval_s", 8.0)),
        )
        self._last_ready_chime_at: float = 0.0
        self._ready_chime_generation: int = 0
        self._input_cooldown_until: float = 0.0
        self._input_cooldown_log_next: float = 0.0
        self._run_id: str = uuid4().hex[:16]
        self._input_state_lock = threading.Lock()
        self._input_level_window: deque[tuple[float, int, float]] = deque(maxlen=200)
        self._input_last_peak: int = 0
        self._input_last_rms: float = 0.0
        self._input_last_observed_at: float = 0.0
        self._input_vad_state: str = "idle"
        self._input_gate_state: str = "open"
        self._input_asr_timeouts: int = 0
        self._input_last_failure_reason: str | None = None
        self._turn_traces = turn_traces or VoiceTurnTraceRecorder()
        fast_path_cfg = voice_cfg.get("fast_path", {}) or {}
        self._fast_endpoint = FastEndpointController(
            quick_replies=DEFAULT_QUICK_REPLIES,
            enabled=bool(fast_path_cfg.get("enabled", False)),
            candidate_silence_ms=float(fast_path_cfg.get("candidate_silence_ms", 300.0)),
            estop_candidate_silence_ms=float(
                fast_path_cfg.get("estop_candidate_silence_ms", 150.0)
            ),
            stable_partial_ms=float(fast_path_cfg.get("stable_partial_ms", 160.0)),
        )

        # -- Input engines (only in voice mode) --
        self._asr_timeout: float = voice_cfg.get("asr", {}).get("asr_timeout", _DEFAULT_ASR_TIMEOUT)

        # Backward-compat attributes (tested by test_audio_agent.py)
        self._echo_gate_peak: int = int(voice_cfg.get("echo_gate_peak", 800))
        _raw_input = voice_cfg.get("input_device", None)
        if _raw_input is None:
            self._input_device: int | str | None = None
        elif isinstance(_raw_input, int):
            self._input_device = _raw_input
        else:
            try:
                self._input_device = int(_raw_input)
            except (ValueError, TypeError):
                self._input_device = str(_raw_input)
        _raw_gate = voice_cfg.get("noise_gate_peak", 0)
        self._noise_gate_peak: int = 0 if str(_raw_gate).lower() == "auto" else int(_raw_gate)
        # Fast endpointing needs its own low-energy threshold.  The main noise
        # gate may deliberately be disabled when software AEC is active, but a
        # disabled gate must not prevent safe fixed phrases from closing early.
        self._fast_quiet_peak_threshold: int = max(
            0,
            int(
                fast_path_cfg.get(
                    "quiet_peak_threshold",
                    self._noise_gate_peak or 500,
                )
            ),
        )
        self._fast_quiet_peak_threshold: int = max(
            0,
            int(
                fast_path_cfg.get(
                    "quiet_peak_threshold",
                    self._noise_gate_peak or 500,
                )
            ),
        )

        # -- New modular components --
        self._mic = MicInput.from_config(config, audio_router=audio_router)
        input_controller_setter = getattr(
            audio_router,
            "set_input_controller",
            None,
        )
        if callable(input_controller_setter):
            input_controller_setter(
                suspend=self._suspend_input_for_output,
                resume=self._resume_input_after_output,
            )
        self._media_transport = self._resolve_media_transport_label(voice_cfg)
        self._audio_proc = AudioProcessor(voice_cfg)
        self._barge_listener_stop = threading.Event()
        self._barge_listener_lock = threading.Lock()
        self._barge_listener_thread: threading.Thread | None = None
        self._barge_listener_results: queue.Queue[str] = queue.Queue()
        self._barge_in_active = threading.Event()
        self._barge_in_callback_lock = threading.Lock()
        self._barge_in_notified = threading.Event()
        self._barge_in_requested_mode = str(
            voice_cfg.get("barge_in_mode", "keyword")
        ).strip().lower()
        self._barge_in_mode = "keyword"
        self._barge_in_warning_emitted = False
        barge_pre_roll_s = max(0.5, float(voice_cfg.get("barge_in_pre_roll_s", 1.2)))
        self._barge_wake_preroll: deque[np.ndarray] = deque(
            maxlen=max(5, int(round(barge_pre_roll_s / 0.1)))
        )
        address_cfg = voice_cfg.get("address_detection", {}) or {}
        wake_terms = ["小算", *address_cfg.get("names", [])]
        aliases = address_cfg.get("aliases", {}) or {}
        if isinstance(aliases, dict):
            wake_terms.extend(aliases.keys())
        self._barge_wake_terms = tuple(
            dict.fromkeys(str(term).strip() for term in wake_terms if str(term).strip())
        )
        self._configure_barge_in_policy(voice_cfg)
        # Keep text-mode construction lightweight and safe on machines with
        # native audio/VAD libraries installed: sherpa-onnx VAD initialization
        # can abort the interpreter on some Windows setups. Only voice mode
        # needs a live VAD controller.
        self._vad_ctrl = VADController(voice_cfg) if voice_mode else None
        self._asr_mgr = ASRManager(voice_cfg) if voice_mode else None
        self.punct: PunctuationRestorer | None

        if voice_mode:
            assert self._vad_ctrl is not None
            assert self._asr_mgr is not None
            # Share engines from modules — avoid constructing duplicates
            self.asr = self._asr_mgr._asr
            self.vad = self._vad_ctrl._vad
            self.asr_stream = self._asr_mgr._stream
            self.punct = self._asr_mgr._punct
            self.kws = KWSEngine(voice_cfg.get("kws", {}))

            self.kws_stream = None
            if self.kws.available:
                try:
                    self.kws_stream = self.kws.create_stream()
                except Exception:
                    logger.exception("Failed to create KWS stream")
            if self.kws_stream is None:
                # Product mode fails closed. ASR remains available only so
                # narrow local emergency/mute commands can still be recognized.
                self.woken_up = not self._require_wake_word
        else:
            self.asr = None  # type: ignore[assignment]
            self.vad = None  # type: ignore[assignment]
            self.kws = None  # type: ignore[assignment]
            self.punct = None
            self.asr_stream = None
            self.kws_stream = None
            self.woken_up = True

        # -- Output engine --
        initial_tts_config = voice_cfg.get("tts", {})
        self.tts = TTSEngine(initial_tts_config, audio_router=audio_router)
        initial_asr = self._asr_config_selection(voice_cfg)
        initial_tts = self._tts_config_selection(initial_tts_config)
        self._runtime_switch_desired = {
            "asr": dict(initial_asr),
            "tts": dict(initial_tts),
        }
        self._runtime_switch_effective = {
            "asr": dict(initial_asr),
            "tts": dict(initial_tts),
        }
        self._runtime_switch_failed: dict[str, dict[str, str] | None] = {
            "asr": None,
            "tts": None,
        }
        interruption_cfg = voice_cfg.get("interruption_recovery", {}) or {}
        if not isinstance(interruption_cfg, dict):
            raise ValueError("voice.interruption_recovery must be a mapping")
        self._interruption_pause_timeout_s = float(interruption_cfg.get("pause_timeout_s", 0.05))
        self._interruption_hold_timeout_s = float(interruption_cfg.get("hold_timeout_s", 2.0))
        self._interruption_recovery = self._new_interruption_recovery(self.tts)
        logger.info("AudioAgent run_id=%s mode=%s", self._run_id, "voice" if voice_mode else "text")
        self._refresh_voice_metrics()

    def _init_output_ownership_state(self) -> None:
        """Initialise the isolated output-ownership state machine."""

        # Lock order is output owner -> realtime reservation -> TTS internals.
        # Code holding the realtime reservation lock must never enter this one.
        self._output_trace_lock = threading.RLock()
        self._output_trace_epoch = 0
        self._active_output_trace_token: OutputPlaybackTraceToken | None = None
        self._output_trace_context: ContextVar[OutputPlaybackTraceToken | None] = ContextVar(
            f"askme_output_playback_{id(self)}",
            default=None,
        )
        self._realtime_output_trace_token: OutputPlaybackTraceToken | None = None
        self._interruption_output_trace_token: OutputPlaybackTraceToken | None = None
        self._orphan_output_trace_event_count = 0
        self._playback_owner_conflict_count = 0
        self._stale_playback_stop_count = 0

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to TTS)
    # ------------------------------------------------------------------

    def _configure_barge_in_policy(self, voice_cfg: dict[str, Any]) -> None:
        """Resolve natural barge-in behind hardware and field acceptance gates."""

        tts_cfg = voice_cfg.get("tts", {}) or {}
        aec_cfg = voice_cfg.get("echo_cancellation", {}) or {}
        aec_enabled = bool(aec_cfg.get("enabled", False))
        requirements = {
            "full_duplex_verified": bool(
                tts_cfg.get("resident_output_full_duplex_verified", False)
            ),
            "aec_enabled": aec_enabled,
            "aec_ready": bool(
                aec_enabled and self._audio_proc.echo_reference is not None
            ),
            "field_acceptance_verified": bool(
                voice_cfg.get("barge_in_field_acceptance_verified", False)
            ),
        }
        missing_reasons = [
            reason
            for key, reason in (
                ("full_duplex_verified", "full_duplex_not_verified"),
                ("aec_enabled", "aec_not_enabled"),
                ("aec_ready", "aec_not_ready"),
                ("field_acceptance_verified", "field_acceptance_not_verified"),
            )
            if not requirements[key]
        ]
        speech_gate_ready = not missing_reasons
        requested_mode = self._barge_in_requested_mode
        if requested_mode == "speech" and speech_gate_ready:
            effective_mode = "speech"
            reason = "speech_gate_passed"
        elif requested_mode == "speech":
            effective_mode = "keyword"
            reason = "speech_gate_blocked:" + ",".join(missing_reasons)
            if not self._barge_in_warning_emitted:
                logger.warning(
                    "Speech barge-in disabled; falling back to keyword (%s)",
                    reason,
                )
                self._barge_in_warning_emitted = True
        elif requested_mode == "keyword":
            effective_mode = "keyword"
            reason = "keyword_configured"
        else:
            effective_mode = "keyword"
            reason = f"unsupported_mode:{requested_mode or 'empty'}"
        self._barge_in_mode = effective_mode
        self._barge_in_policy = {
            "requested_mode": requested_mode,
            "effective_mode": effective_mode,
            "speech_gate_ready": speech_gate_ready,
            "reason": reason,
            "requirements": requirements,
        }

    @property
    def is_busy(self) -> bool:
        """Whether TTS is actively playing or has queued text."""
        active_probe = getattr(self.tts, "is_active", None)
        active = (
            bool(active_probe())
            if callable(active_probe)
            else bool(getattr(self.tts, "_is_playing", False))
        )
        text_queue = getattr(self.tts, "tts_text_queue", None)
        queued = bool(text_queue is not None and not text_queue.empty())
        return active or queued

    @property
    def wake_word_ready(self) -> bool:
        """Whether both the KWS engine and its live stream are usable."""

        kws = getattr(self, "kws", None)
        return bool(
            kws
            and getattr(kws, "available", False)
            and getattr(self, "kws_stream", None) is not None
        )

    @property
    def kws_unavailable_safety_only(self) -> bool:
        """Whether microphone input is restricted to local safety commands."""

        return bool(
            getattr(self, "voice_mode", False)
            and getattr(self, "_require_wake_word", True)
            and not self.wake_word_ready
        )

    @property
    def is_input_open(self) -> bool:
        """Whether the physical microphone capture handle is currently open."""

        return bool(self._mic.is_open)

    def speak(self, text: str) -> None:
        """Queue text for TTS (strips emoji/markdown internally)."""
        # A delayed ACK/thinking chime is perceptual feedback, not part of the
        # answer.  Stop it before the first (or next) semantic clause so the
        # two streams cannot overlap on full-duplex output devices.  Keep the
        # feedback cancellation and semantic enqueue in one transaction so a
        # stale wait cue cannot slip between them and stop the answer audio.
        with self._output_trace_lock:
            active = self._active_output_trace_token
            if active is not None and self._output_trace_context.get() != active:
                self._playback_owner_conflict_count += 1
                raise RuntimeError("playback owner conflict")
            with self._realtime_output_lock:
                realtime_reserved = self._realtime_output_provider_generation > 0
            if active is None and realtime_reserved:
                self._playback_owner_conflict_count += 1
                raise RuntimeError("realtime playback owns the output reservation")
            with self._chime_lock:
                self.cancel_processing_feedback()
                self.tts.speak(text)
        self._refresh_voice_metrics()

    def start_playback(
        self,
        *,
        voice_turn_id: str | None = None,
    ) -> OutputPlaybackTraceToken | None:
        """Start one fenced playback interval and return its immutable owner.

        Repeated starts from the same task are idempotent.  A different task
        cannot replace an active owner; it must first observe that playback has
        ended.  This prevents a late callback from redirecting output evidence
        or acquiring authority over a newer turn.
        """

        normalized_voice_turn_id = self._normalize_playback_voice_turn_id(voice_turn_id)
        with self._output_trace_lock:
            token = self._start_playback_locked(
                normalized_voice_turn_id,
                provider_generation=None,
                transport_generation=None,
            )
        self._refresh_voice_metrics()
        return token

    @staticmethod
    def _normalize_playback_voice_turn_id(
        voice_turn_id: str | None,
    ) -> str | None:
        if voice_turn_id is None:
            return None
        if not isinstance(voice_turn_id, str):
            raise TypeError("voice_turn_id must be a string or None")
        normalized = voice_turn_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("voice_turn_id must contain 1..128 characters")
        return normalized

    def _start_playback_locked(
        self,
        voice_turn_id: str | None,
        *,
        provider_generation: int | None,
        transport_generation: int | None,
    ) -> OutputPlaybackTraceToken | None:
        """Start playback after the caller has acquired the output-owner lock."""

        active = self._active_output_trace_token
        caller_token = self._output_trace_context.get()
        if active is not None:
            if (
                caller_token == active
                and voice_turn_id in {None, active.voice_turn_id}
                and provider_generation == active.provider_generation
            ):
                return active
            self._playback_owner_conflict_count += 1
            return None

        with self._realtime_output_lock:
            reserved_provider_generation = self._realtime_output_provider_generation
            reserved_transport_generation = self._realtime_output_tts_generation
        if provider_generation is None:
            if reserved_provider_generation > 0:
                self._playback_owner_conflict_count += 1
                return None
        elif (
            provider_generation != reserved_provider_generation
            or transport_generation != reserved_transport_generation
        ):
            self._playback_owner_conflict_count += 1
            return None

        self.tts.start_playback()
        self._output_trace_epoch += 1
        token = OutputPlaybackTraceToken(
            voice_turn_id=voice_turn_id,
            epoch=self._output_trace_epoch,
            provider_generation=provider_generation,
            transport_generation=transport_generation,
        )
        self._active_output_trace_token = token
        self._output_trace_context.set(token)
        self._ready_chime_generation += 1
        self._agent_state = AgentState.SPEAKING
        self._record_output_voice_trace_for_token(
            token,
            "tts_playback_started",
        )
        return token

    def stop_playback(
        self,
        token: OutputPlaybackTraceToken | None = None,
    ) -> None:
        """Stop playback only when the caller still owns the active interval."""

        with self._output_trace_lock:
            requested = token or self._output_trace_context.get()
            active = self._active_output_trace_token
            if active is None or requested != active:
                if requested is not None:
                    self._stale_playback_stop_count += 1
                    if self._output_trace_context.get() == requested:
                        self._output_trace_context.set(None)
                return

            if self._interruption_output_trace_token in {None, active}:
                self._close_interruption_recovery()
                if self._interruption_output_trace_token == active:
                    self._interruption_output_trace_token = None
            frozen_played_ms = self._streaming_played_ms(active.transport_generation)
            self.tts.stop_playback()
            if active.provider_generation is not None:
                with self._realtime_output_lock:
                    if (
                        self._realtime_output_provider_generation == active.provider_generation
                        and self._realtime_output_tts_generation == active.transport_generation
                    ):
                        self._realtime_last_physical_played_ms = max(
                            int(
                                getattr(
                                    self,
                                    "_realtime_last_physical_played_ms",
                                    0,
                                )
                                or 0
                            ),
                            frozen_played_ms,
                        )
                        self._realtime_output_tts_generation = None
                        self._realtime_output_provider_generation = 0
                        self._realtime_output_terminated_provider_generation = 0
                        self._realtime_output_started = False
                        self._realtime_output_voice_turn_id = None
            self._begin_post_tts_input_cooldown()
            self._agent_state = AgentState.IDLE
            self._close_active_output_trace_locked("completed")
            self.mark_interaction_turn()
        self._refresh_voice_metrics()
        self._schedule_ready_chime()

    def finish_realtime_playback(self, *, expected_generation: int) -> bool:
        """Join provider-thread playback through its exact generation token."""

        with self._output_trace_lock:
            token = self._realtime_output_trace_token
            if token is None or token.provider_generation != int(expected_generation):
                self._stale_playback_stop_count += 1
                return False
            self.stop_playback(token)
            return True

    def wait_speaking_done(self, timeout: float = 30.0) -> bool:
        done = self.tts.wait_done(timeout=timeout)
        self._refresh_voice_metrics()
        return done

    async def speak_and_wait(self, text: str) -> None:
        """Speak text and wait for TTS to finish (voice mode convenience).

        Replaces the repeated 4-step pattern:
            self._audio.speak(text)
            self._audio.start_playback()
            await asyncio.to_thread(self._audio.wait_speaking_done)
            self._audio.stop_playback()
        """
        import asyncio

        token = self.start_playback()
        if token is None:
            raise RuntimeError("playback owner conflict")
        try:
            self.speak(text)
            await asyncio.to_thread(self.wait_speaking_done)
        finally:
            self.stop_playback(token)

    async def speak_cached_and_wait(self, text: str, *, cache_key: str) -> bool:
        """Play a persisted phrase without invoking a TTS provider."""

        token = self.start_playback()
        if token is None:
            raise RuntimeError("playback owner conflict")
        try:
            with self._chime_lock:
                self.cancel_processing_feedback()
                if not self.tts.queue_cached_phrase(text, cache_key=cache_key):
                    return False
            await asyncio.to_thread(self.wait_speaking_done)
            return True
        finally:
            self.stop_playback(token)

    def start_input(self) -> None:
        """Open the microphone input stream for long-lived voice sessions."""
        if not self.voice_mode:
            return
        self._input_requested = True
        try:
            self._mic.start()
        except Exception:
            self._input_requested = False
            raise
        coordinator = self._realtime_coordinator
        if coordinator is not None:
            try:
                if not coordinator.start():
                    logger.warning("Realtime dialogue unavailable; using cascade voice fallback")
            except Exception as exc:
                # Realtime is an optional latency path.  Mic + ASR + LLM + TTS
                # must remain usable if its preconnect fails.
                logger.warning(
                    "Realtime dialogue preconnect failed; using cascade fallback: %s",
                    exc,
                )
        self._refresh_voice_metrics()

    def stop_input(self) -> None:
        """Close the microphone input stream for long-lived voice sessions."""
        if not self.voice_mode:
            return
        self._input_requested = False
        self._mic.stop()
        self._refresh_voice_metrics()

    def _suspend_input_for_output(self) -> None:
        """Release a persistent mic before exclusive speaker ownership."""

        if self.voice_mode:
            self._mic.stop()

    def _resume_input_after_output(self) -> None:
        """Reopen the mic only when the runtime still requests voice input."""

        if self.voice_mode and self._input_requested and not self.stop_event.is_set():
            self._mic.start()

    def set_barge_in_callback(
        self,
        callback: Callable[[], None] | None,
    ) -> None:
        """Set the lightweight callback fired after a confirmed interruption."""

        self._barge_in_callback = callback

    def set_capture_processor(
        self,
        callback: Callable[[np.ndarray, int, bool], np.ndarray] | None,
        *,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Install an optional post-filter capture processor such as WebRTC AEC."""

        self._capture_processor = callback
        self._capture_processor_failure_callback = on_failure
        self._capture_processor_error_logged = False
        self._capture_processor_failed_current_frame = False

    def configure_realtime_dialogue(
        self,
        session: Any | None,
        realtime_config: Any | None,
    ) -> None:
        """Attach one provider session behind the existing local audio owner."""

        self._stop_realtime_recovery("reconfigured")
        previous = self._realtime_coordinator
        if previous is not None:
            try:
                previous.close("reconfigured")
            except Exception as exc:
                logger.debug("Realtime dialogue close during reconfigure failed: %s", exc)
        self._realtime_coordinator = None
        self._realtime_mode = "split"
        with self._realtime_recovery_lock:
            self._realtime_recovery_stop = threading.Event()
            self._realtime_recovery_thread = None
            self._realtime_capture_armed = False
            self._realtime_faulted_coordinator = None
            self._realtime_recovery_attempts = 0
            self._realtime_recovery_successes = 0
            self._realtime_recovery_failures = 0
            self._realtime_recovery_last_error = ""
        with self._realtime_output_lock:
            self._realtime_output_tts_generation = None
            self._realtime_output_provider_generation = 0
            self._realtime_output_terminated_provider_generation = 0
            self._realtime_output_started = False
            self._realtime_output_voice_turn_id = None
            self._realtime_last_physical_played_ms = 0
        if session is None or realtime_config is None or not self.voice_mode:
            return

        from askme.voice.core.realtime_contracts import RealtimeVoiceSessionContext
        from askme.voice.realtime.coordinator import RealtimeDialogueCoordinator

        mode = str(getattr(realtime_config, "mode", "split"))
        mode = str(getattr(getattr(realtime_config, "mode", None), "value", mode))
        self._realtime_mode = mode.strip().lower()
        context = RealtimeVoiceSessionContext(
            session_id=self._run_id,
            bot_name=str(getattr(realtime_config, "bot_name", "小算") or "小算"),
            system_role=str(getattr(realtime_config, "system_role", "") or ""),
            speaking_style=str(getattr(realtime_config, "speaking_style", "") or ""),
            input_mode=str(getattr(realtime_config, "input_mode", "audio") or "audio"),
            input_sample_rate=int(getattr(realtime_config, "input_sample_rate", 16_000)),
            output_sample_rate=int(getattr(realtime_config, "output_sample_rate", 24_000)),
            output_format=str(
                getattr(realtime_config, "output_format", "pcm_s16le") or "pcm_s16le"
            ),
            allow_tool_calls=False,
            allow_hardware_dispatch=False,
            metadata={"source": "local_post_aec_capture"},
        )
        self._realtime_coordinator = RealtimeDialogueCoordinator(
            session,
            context,
            mode=self._realtime_mode,
            audio_sink=(
                self._queue_realtime_audio if self._realtime_mode == "general_chat" else None
            ),
            pending_output_ms=int(getattr(realtime_config, "pending_output_ms", 2_000)),
        )

    def _prepare_realtime_turn_boundary(self) -> bool:
        """Arm an already healthy S2S lane or recover it for a later turn.

        Recovery is deliberately asynchronous.  A connection that comes back
        while the user is already speaking is not allowed to consume the tail
        of that utterance; the whole current turn stays on the local cascade.
        """

        coordinator = self._realtime_coordinator
        with self._realtime_recovery_lock:
            self._realtime_capture_armed = False
        if (
            coordinator is None
            or self._muted
            or self.stop_event.is_set()
            or self.kws_unavailable_safety_only
        ):
            return False

        try:
            status = coordinator.status_snapshot()
        except Exception as exc:
            status = {}
            with self._realtime_recovery_lock:
                self._realtime_recovery_last_error = type(exc).__name__
        if bool(status.get("active", False)):
            with self._realtime_recovery_lock:
                faulted = self._realtime_faulted_coordinator is coordinator
            ready = not bool(status.get("quarantined", False)) and not faulted
            with self._realtime_recovery_lock:
                if self._realtime_coordinator is coordinator:
                    self._realtime_capture_armed = ready
            return ready

        recover = getattr(coordinator, "recover_at_turn_boundary", None)
        if not callable(recover):
            return False
        with self._realtime_recovery_lock:
            current = self._realtime_recovery_thread
            if current is not None and current.is_alive():
                return False
            if self._realtime_coordinator is not coordinator or self.stop_event.is_set():
                return False
            cancel_event = threading.Event()
            self._realtime_recovery_stop = cancel_event
            self._realtime_recovery_attempts += 1
            worker = threading.Thread(
                target=self._recover_realtime_for_future_turn,
                args=(coordinator, cancel_event),
                name="realtime-turn-boundary-recovery",
                daemon=True,
            )
            self._realtime_recovery_thread = worker
            worker.start()
        return False

    def _recover_realtime_for_future_turn(
        self,
        coordinator: Any,
        cancel_event: threading.Event,
    ) -> None:
        success = False
        error = ""
        try:
            if not cancel_event.is_set() and not self.stop_event.is_set():
                success = bool(coordinator.recover_at_turn_boundary())
        except Exception as exc:
            error = type(exc).__name__
        cancelled = cancel_event.is_set() or self.stop_event.is_set()
        with self._realtime_recovery_lock:
            obsolete = self._realtime_coordinator is not coordinator
        if success and (cancelled or obsolete):
            try:
                coordinator.close("recovery_cancelled" if cancelled else "recovery_obsolete")
            except Exception as exc:
                error = error or type(exc).__name__
            success = False

        with self._realtime_recovery_lock:
            if self._realtime_coordinator is coordinator:
                if success:
                    self._realtime_recovery_successes += 1
                    self._realtime_recovery_last_error = ""
                    if self._realtime_faulted_coordinator is coordinator:
                        self._realtime_faulted_coordinator = None
                elif not cancelled:
                    self._realtime_recovery_failures += 1
                    self._realtime_recovery_last_error = error or "recovery_failed"
                # Never arm here.  Only the next local listen boundary can do
                # that, preventing mid-utterance attachment after reconnect.
                self._realtime_capture_armed = False
            if self._realtime_recovery_thread is threading.current_thread():
                self._realtime_recovery_thread = None

    def _stop_realtime_recovery(self, reason: str) -> None:
        """Cancel and join the optional recovery worker during lifecycle changes."""

        with self._realtime_recovery_lock:
            self._realtime_capture_armed = False
            cancel_event = self._realtime_recovery_stop
            worker = self._realtime_recovery_thread
            cancel_event.set()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=10.0)
            if worker.is_alive():
                logger.warning(
                    "Realtime recovery did not stop before %s timeout",
                    reason,
                )
        with self._realtime_recovery_lock:
            if self._realtime_recovery_thread is worker and (
                worker is None or not worker.is_alive()
            ):
                self._realtime_recovery_thread = None

    def _realtime_turn_capture_is_armed(self) -> bool:
        with self._realtime_recovery_lock:
            return bool(self._realtime_capture_armed)

    def _fence_failed_realtime_capture(
        self,
        coordinator: Any,
        reason: str,
        error: BaseException | None = None,
    ) -> None:
        """Keep a partial optional-cloud turn out of all later decisions."""

        with self._realtime_recovery_lock:
            if self._realtime_coordinator is coordinator:
                self._realtime_capture_armed = False
                self._realtime_faulted_coordinator = coordinator
                self._realtime_recovery_last_error = (
                    type(error).__name__ if error is not None else reason
                )

        def _close() -> None:
            try:
                coordinator.close(reason)
            except Exception as exc:
                with self._realtime_recovery_lock:
                    if self._realtime_coordinator is coordinator:
                        self._realtime_recovery_last_error = type(exc).__name__

        threading.Thread(
            target=_close,
            name="realtime-capture-fence",
            daemon=True,
        ).start()

    def realtime_general_chat_ready(self) -> bool:
        """Return whether the optional fast path is safe to attempt now."""

        coordinator = self._realtime_coordinator
        if coordinator is None or self._realtime_mode != "general_chat":
            return False
        try:
            status = coordinator.status_snapshot()
        except Exception:
            return False
        return bool(status.get("active", False) and not status.get("quarantined", False))

    def realtime_capture_active(self) -> bool:
        coordinator = self._realtime_coordinator
        if coordinator is None:
            return False
        try:
            status = coordinator.status_snapshot()
        except Exception:
            return False
        return bool(status.get("active", False) and not status.get("quarantined", False))

    def try_realtime_general_chat(
        self,
        local_text: str,
        *,
        expected_generation: int = 0,
    ) -> Any | None:
        """Reject the retired one-step admission surface.

        Provider PCM may only be released by ``VoiceLoop`` after central route
        policy and a durable Conversation Ledger begin.  Keeping this method as
        a fail-closed shim avoids an ``AttributeError`` for older integrations
        without preserving a path around those product invariants.
        """

        del local_text
        logger.warning(
            "Rejected legacy one-step realtime admission; use the VoiceLoop "
            "prepare/ledger/release path"
        )
        self.discard_realtime_turn(
            "legacy_one_step_admission_disabled",
            expected_generation=int(expected_generation or 0),
        )
        return None

    def prepare_realtime_general_chat(
        self,
        local_text: str,
        *,
        expected_generation: int = 0,
    ) -> Any | None:
        """Validate a provider candidate without opening a playback generation."""

        coordinator = self._realtime_coordinator
        prepare = getattr(coordinator, "prepare_general_chat", None)
        if coordinator is None or not self.realtime_general_chat_ready() or not callable(prepare):
            return None
        try:
            return prepare(
                local_text,
                expected_generation=int(expected_generation or 0),
            )
        except Exception as exc:
            logger.warning("Realtime general-chat prepare failed: %s", exc)
            return None

    def release_realtime_general_chat(
        self,
        approval: Any,
        *,
        expected_generation: int = 0,
        voice_turn_id: str | None = None,
    ) -> bool:
        """Bind playback and release PCM after the canonical Turn is durable."""

        coordinator = self._realtime_coordinator
        release = getattr(coordinator, "release_general_chat", None)
        if coordinator is None or not self.realtime_general_chat_ready() or not callable(release):
            return False
        approval_generation = int(getattr(approval, "generation", 0) or 0)
        if expected_generation > 0 and approval_generation != int(expected_generation):
            return False
        normalized_voice_turn_id = voice_turn_id.strip() if isinstance(voice_turn_id, str) else None
        if voice_turn_id is not None and (
            not normalized_voice_turn_id or len(normalized_voice_turn_id) > 128
        ):
            return False

        begin_streaming = getattr(self.tts, "begin_streaming_pcm", None)
        if not callable(begin_streaming):
            return False
        with self._output_trace_lock:
            with self._realtime_output_lock:
                reservation_active = self._realtime_output_provider_generation > 0
            if self._active_output_trace_token is not None or reservation_active:
                self._playback_owner_conflict_count += 1
                return False
            try:
                tts_generation = int(begin_streaming())
            except Exception as exc:
                logger.warning("Realtime PCM generation could not start: %s", exc)
                return False
            with self._realtime_output_lock:
                self._realtime_output_tts_generation = tts_generation
                self._realtime_output_provider_generation = approval_generation
                self._realtime_output_terminated_provider_generation = 0
                self._realtime_output_started = False
                self._realtime_output_voice_turn_id = normalized_voice_turn_id
                self._realtime_last_physical_played_ms = 0
        try:
            released = bool(release(approval))
        except Exception as exc:
            logger.warning("Realtime general-chat release failed: %s", exc)
            released = False
        if released:
            return True

        # Invalidate the just-created streaming generation.  The coordinator
        # still owns provider history fencing; the VoiceLoop supplies the
        # generation-specific discard reason before falling back to cascade.
        with self._output_trace_lock:
            realtime_token = self._realtime_output_trace_token
            active = self._active_output_trace_token
            try:
                if realtime_token is not None and active == realtime_token:
                    self.tts.stop_immediately()
                if active is None or active == realtime_token:
                    self.tts.drain_buffers()
                if realtime_token is not None and active == realtime_token:
                    self.tts.stop_playback()
                    self._agent_state = AgentState.IDLE
                    self._close_active_output_trace_locked("realtime_release_rejected")
            except Exception as exc:
                logger.debug("Realtime PCM release cleanup failed: %s", exc)
            with self._realtime_output_lock:
                if self._realtime_output_tts_generation == tts_generation:
                    self._realtime_output_tts_generation = None
                    self._realtime_output_provider_generation = 0
                    self._realtime_output_started = False
                    self._realtime_output_voice_turn_id = None
        return False

    def discard_realtime_turn(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
        after_generation: int = 0,
    ) -> None:
        """Fence speculative or interrupted provider audio before it can play."""

        with self._realtime_output_lock:
            if expected_generation <= 0:
                expected_generation = self._realtime_output_provider_generation
            generation_matches = bool(
                expected_generation > 0
                and expected_generation == self._realtime_output_provider_generation
            )
            if generation_matches:
                self._realtime_output_terminated_provider_generation = int(expected_generation)
                self._realtime_output_tts_generation = None
                self._realtime_output_provider_generation = 0
                self._realtime_output_started = False
                self._realtime_output_voice_turn_id = None
        coordinator = self._realtime_coordinator
        if coordinator is None:
            return
        try:
            coordinator.discard_current(
                str(reason or "discarded"),
                expected_generation=int(expected_generation or 0),
                after_generation=int(after_generation or 0),
            )
        except Exception as exc:
            logger.debug("Realtime turn discard failed: %s", exc)

    def _truncate_or_discard_realtime_playback(
        self,
        reason: str,
        *,
        asynchronous: bool = False,
        expected_generation: int = 0,
    ) -> None:
        """End one admitted provider response at its physical playhead.

        Volcengine can preserve the heard prefix in conversation history when
        a generation-bound physical playhead is available.  Any missing or
        failed capability falls back to the existing fail-closed history
        deletion path.
        """

        with self._realtime_output_lock:
            tts_generation = self._realtime_output_tts_generation
            provider_generation = self._realtime_output_provider_generation
            already_terminated = int(
                getattr(
                    self,
                    "_realtime_output_terminated_provider_generation",
                    0,
                )
                or 0
            )
            if expected_generation > 0 and provider_generation != int(expected_generation):
                stale_expected_generation = int(expected_generation)
            else:
                stale_expected_generation = 0
            if stale_expected_generation == 0 and provider_generation > 0:
                self._realtime_output_terminated_provider_generation = int(provider_generation)
            if stale_expected_generation == 0:
                self._realtime_output_tts_generation = None
                self._realtime_output_provider_generation = 0
                self._realtime_output_started = False
                self._realtime_output_voice_turn_id = None

        if stale_expected_generation > 0:
            if already_terminated != stale_expected_generation:
                self.discard_realtime_turn(
                    reason,
                    expected_generation=stale_expected_generation,
                )
            return

        if provider_generation <= 0 or tts_generation is None:
            if already_terminated > 0:
                return
            self.discard_realtime_turn(reason)
            return

        played_ms = self._streaming_played_ms(tts_generation)
        with self._realtime_output_lock:
            self._realtime_last_physical_played_ms = max(
                int(getattr(self, "_realtime_last_physical_played_ms", 0) or 0),
                played_ms,
            )

        def _resolve_history() -> None:
            coordinator = self._realtime_coordinator
            truncate = getattr(coordinator, "truncate_current", None)
            if played_ms > 0 and callable(truncate):
                try:
                    if bool(
                        truncate(
                            str(reason or "interrupted"),
                            audio_end_ms=played_ms,
                            expected_generation=int(provider_generation),
                        )
                    ):
                        return
                    status_snapshot = getattr(coordinator, "status_snapshot", None)
                    if callable(status_snapshot):
                        status = status_snapshot()
                        # The production coordinator performs its own fail-closed
                        # ConversationDelete before returning False.  Detect that
                        # contract so the caller never schedules a second delete;
                        # simpler duck-typed coordinators still fall through.
                        if bool(
                            status.get("quarantined", False)
                            or status.get("discarded", False)
                            or int(status.get("rollback_generation", 0) or 0)
                            == int(provider_generation)
                        ):
                            return
                except Exception as exc:
                    logger.debug("Realtime conversation truncate failed: %s", exc)

            self.discard_realtime_turn(
                reason,
                expected_generation=int(provider_generation),
            )

        if asynchronous:
            threading.Thread(
                target=_resolve_history,
                name="realtime-history-resolution",
                daemon=True,
            ).start()
            return
        _resolve_history()

    def realtime_playback_started(self) -> bool:
        with self._realtime_output_lock:
            return self._realtime_output_started

    def abort_realtime_playback(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
    ) -> bool:
        """Fence provider output and atomically clear already-admitted PCM."""

        with self._output_trace_lock:
            with self._realtime_output_lock:
                provider_generation = self._realtime_output_provider_generation
            target_generation = int(expected_generation or provider_generation)
            realtime_token = self._realtime_output_trace_token
            active = self._active_output_trace_token
            if target_generation > 0 and provider_generation != target_generation:
                self.discard_realtime_turn(
                    reason,
                    expected_generation=target_generation,
                )
                return False
            if active is not None and active != realtime_token:
                self.discard_realtime_turn(
                    reason,
                    expected_generation=target_generation,
                )
                return False
            self.tts.stop_immediately()
            self._truncate_or_discard_realtime_playback(
                reason,
                asynchronous=True,
                expected_generation=target_generation,
            )
            self.tts.drain_buffers()
            self.tts.stop_playback()
            self._agent_state = AgentState.IDLE
            if realtime_token is not None and active == realtime_token:
                self._close_active_output_trace_locked(reason)
        self._refresh_voice_metrics()
        return True

    def _offer_realtime_capture(
        self,
        samples_i16: np.ndarray,
        *,
        sample_rate: int,
    ) -> bool:
        """Offer the same post-AEC mono capture used by local ASR."""

        coordinator = self._realtime_coordinator
        if (
            coordinator is None
            or self._muted
            or self.kws_unavailable_safety_only
            or not self._realtime_turn_capture_is_armed()
        ):
            return False
        samples = np.asarray(samples_i16)
        if samples.ndim != 1 or samples.size == 0 or not np.issubdtype(samples.dtype, np.integer):
            return False
        pcm = np.ascontiguousarray(samples, dtype="<i2").tobytes()
        try:
            accepted = bool(
                coordinator.offer_audio(
                    VoiceMediaFrame(
                        pcm=pcm,
                        sample_rate=int(sample_rate),
                        channels=1,
                        metadata={"capture_stage": "post_aec"},
                    )
                )
            )
        except Exception as exc:
            logger.debug("Realtime capture offer failed: %s", exc)
            self._fence_failed_realtime_capture(
                coordinator,
                "audio_offer_failed",
                exc,
            )
            return False
        if not accepted:
            self._fence_failed_realtime_capture(
                coordinator,
                "audio_offer_failed",
            )
            return False
        return True

    def _queue_realtime_audio(self, frame: VoiceMediaFrame, final: bool) -> None:
        """Route approved provider PCM through the normal cancellable player."""

        if frame.channels != 1 or frame.sample_rate <= 0 or len(frame.pcm) % 2:
            raise ValueError("realtime output must be mono PCM16")
        samples = (
            np.frombuffer(frame.pcm, dtype="<i2").astype(np.float32) / 32768.0
            if frame.pcm
            else np.empty(0, dtype=np.float32)
        )
        queue_streaming = getattr(self.tts, "queue_streaming_pcm", None)
        if not callable(queue_streaming):
            raise RuntimeError("TTS engine does not support streaming PCM")
        with self._output_trace_lock:
            with self._realtime_output_lock:
                generation = self._realtime_output_tts_generation
                provider_generation = self._realtime_output_provider_generation
                already_started = self._realtime_output_started
                voice_turn_id = self._realtime_output_voice_turn_id
            if generation is None:
                return
            active = self._active_output_trace_token
            if active is not None and (
                not already_started or active != self._realtime_output_trace_token
            ):
                self._playback_owner_conflict_count += 1
                self.discard_realtime_turn(
                    "playback_owner_conflict",
                    expected_generation=provider_generation,
                )
                return
            queued = bool(
                queue_streaming(
                    samples,
                    frame.sample_rate,
                    final=bool(final),
                    generation=generation,
                )
            )
            if not queued:
                return
            if frame.pcm and not already_started:
                with self._realtime_output_lock:
                    if self._realtime_output_tts_generation != generation:
                        return
                    if self._realtime_output_started:
                        return
                    self._realtime_output_started = True
                token = self._start_playback_locked(
                    voice_turn_id,
                    provider_generation=provider_generation,
                    transport_generation=generation,
                )
                if token is None:
                    with self._realtime_output_lock:
                        if self._realtime_output_tts_generation == generation:
                            self._realtime_output_started = False
                    self.discard_realtime_turn(
                        "playback_owner_conflict",
                        expected_generation=provider_generation,
                    )
                    return
                self._realtime_output_trace_token = token
                self._refresh_voice_metrics()

    def drain_buffers(self) -> None:
        """Clear any leftover TTS text/audio from a previous turn."""
        self._close_interruption_recovery()
        with self._realtime_output_lock:
            provider_generation = self._realtime_output_provider_generation
        if provider_generation > 0:
            self.discard_realtime_turn(
                "local_audio_drain",
                expected_generation=provider_generation,
            )
        self.cancel_processing_feedback()
        self.tts.drain_buffers()
        self._refresh_voice_metrics()

    def prepare_turn(self) -> None:
        """Prepare a normal turn without discarding a warm cloud TTS session."""

        self._close_interruption_recovery()
        prepare = getattr(self.tts, "prepare_turn", None)
        if callable(prepare):
            prepare()
        else:
            self.tts.drain_buffers()
        self._refresh_voice_metrics()

    def stop_immediately(self) -> None:
        """Immediately stop TTS playback mid-chunk (barge-in support)."""
        with self._output_trace_lock:
            self._close_interruption_recovery()
            with self._realtime_output_lock:
                provider_generation = self._realtime_output_provider_generation
            if provider_generation > 0:
                self.discard_realtime_turn(
                    "local_audio_stop",
                    expected_generation=provider_generation,
                )
            self.cancel_processing_feedback()
            self.tts.stop_immediately()
            self.tts.drain_buffers()
            self.tts.stop_playback()
            self._agent_state = AgentState.IDLE
            self._close_active_output_trace_locked("local_audio_stop")
        self._refresh_voice_metrics()

    def _begin_post_tts_input_cooldown(self) -> None:
        """Briefly discard mic input after playback so TTS tail is not re-ASR'd."""
        if self._post_tts_input_cooldown_s <= 0:
            return
        self._input_cooldown_until = max(
            self._input_cooldown_until,
            time.monotonic() + self._post_tts_input_cooldown_s,
        )
        self._reset_listening_state()

    def _reset_listening_state(self) -> None:
        """Clear VAD/ASR state after local playback or known false input."""
        vad_controller = self._vad_ctrl
        if vad_controller is not None:
            try:
                vad_controller.reset()
            except Exception as exc:
                logger.debug("VAD reset failed during input cooldown (ignored): %s", exc)
        asr_manager = self._asr_mgr
        if asr_manager is not None:
            try:
                asr_manager.reset()
            except Exception as exc:
                logger.debug("ASR reset failed during input cooldown (ignored): %s", exc)

    def _in_post_tts_input_cooldown(self) -> float:
        """Return remaining post-TTS input cooldown seconds, or 0 when inactive."""
        remaining = self._input_cooldown_until - time.monotonic()
        return remaining if remaining > 0 else 0.0

    def _schedule_ready_chime(self) -> None:
        """Play a subtle cue when the mic is ready after TTS/cooldown."""
        if not self._ready_chime_enabled or not self.voice_mode or self._muted:
            return
        generation = self._ready_chime_generation
        delay = self._in_post_tts_input_cooldown()

        def _run() -> None:
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            if generation != self._ready_chime_generation:
                return
            if (
                self._muted
                or self.is_busy
                or self._agent_state == AgentState.SPEAKING
                or self._barge_in_active.is_set()
            ):
                return
            if self._in_post_tts_input_cooldown() > 0:
                return
            if now - self._last_ready_chime_at < self._ready_chime_min_interval_s:
                return
            self._last_ready_chime_at = now
            self._play_chime("ready")

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Volume / speed control
    # ------------------------------------------------------------------

    def set_volume(self, value: float) -> float:
        """Set TTS output volume (0.05–1.0). Returns new value."""
        v = self.tts.set_volume(value)
        self._refresh_voice_metrics()
        return v

    def adjust_volume(self, delta: float) -> float:
        """Adjust TTS volume by delta. Returns new value."""
        return self.set_volume(self.tts.volume + delta)

    def set_speed(self, value: float) -> float:
        """Set TTS speech speed (0.5–2.0). Returns new value."""
        v = self.tts.set_speed(value)
        self._refresh_voice_metrics()
        return v

    def set_pitch(self, semitones: float) -> float:
        """Set provider pitch in semitones within the TTS safety range."""
        value = self.tts.set_pitch(semitones)
        self._refresh_voice_metrics()
        return value

    def adjust_speed(self, delta: float) -> float:
        """Adjust TTS speed by delta. Returns new value."""
        return self.set_speed(self.tts.speed + delta)

    def reconfigure_asr(self, voice_config: dict[str, Any]) -> dict[str, Any]:
        """Replace ASR for the next listen cycle without restarting the runtime."""

        if not self.voice_mode:
            raise RuntimeError("ASR reconfiguration requires voice mode")
        clean = dict(voice_config or {})
        self._record_runtime_switch_attempt("asr", clean)
        with self._runtime_switch_lock:
            if self._asr_phase_active or bool(getattr(self._asr_mgr, "_recognition_active", False)):
                self._pending_asr_config = clean
                return {
                    "updated": True,
                    "component": "asr",
                    "state": "pending",
                    "effective": "next_listen_cycle",
                }
        try:
            return self._apply_asr_config(clean)
        except Exception as exc:
            self._record_runtime_switch_failure("asr", clean, exc)
            raise

    def set_tts_activation_callback(
        self,
        callback: Callable[[TTSEngine], None] | None,
    ) -> None:
        """Register the runtime owner notified when a new TTS engine is active."""

        with self._runtime_switch_lock:
            self._tts_activation_callback = callback

    def set_runtime_switch_callback(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        """Register the owner notified when a deferred ASR/TTS switch resolves."""

        with self._runtime_switch_lock:
            self._runtime_switch_callback = callback

    def reconfigure_tts(self, tts_config: dict[str, Any]) -> dict[str, Any]:
        """Replace TTS immediately when idle, or queue it after playback."""

        clean = dict(tts_config or {})
        self._record_runtime_switch_attempt("tts", clean)
        with self._runtime_switch_lock:
            if self.stop_event.is_set():
                raise RuntimeError("TTS reconfiguration rejected after AudioAgent shutdown")
            if self.is_busy or self.tts.is_active():
                self._pending_tts_config = clean
                return {
                    "updated": True,
                    "component": "tts",
                    "state": "pending",
                    "effective": "after_playback",
                }
        try:
            return self._apply_tts_config(clean)
        except Exception as exc:
            self._record_runtime_switch_failure("tts", clean, exc)
            raise

    def _apply_asr_config(self, voice_config: dict[str, Any]) -> dict[str, Any]:
        next_manager = ASRManager(voice_config)
        with self._runtime_switch_lock:
            previous = self._asr_mgr
            self._asr_mgr = next_manager
            self.asr = next_manager._asr
            self.asr_stream = next_manager._stream
            self.punct = next_manager._punct
            self._pending_asr_config = None
        if previous is not None:
            previous.reset()
        self._refresh_voice_metrics()
        self._record_runtime_switch_active("asr", voice_config)
        return {
            "updated": True,
            "component": "asr",
            "state": "active",
            "runtime": next_manager.status_snapshot(),
        }

    def _apply_tts_config(self, tts_config: dict[str, Any]) -> dict[str, Any]:
        next_tts = TTSEngine(tts_config, audio_router=self._audio_router)
        next_interruption_recovery = self._new_interruption_recovery(next_tts)
        previous_interruption_recovery: InterruptionRecoveryCoordinator | None = None
        with self._runtime_switch_lock:
            if self.stop_event.is_set():
                self._pending_tts_config = None
                rejected = True
                previous = None
            else:
                rejected = False
                previous = self.tts
                previous_interruption_recovery = self._interruption_recovery
                self.tts = next_tts
                self._interruption_recovery = next_interruption_recovery
                self._pending_tts_config = None
        if rejected:
            try:
                next_tts.shutdown()
            except Exception as exc:
                logger.warning("Rejected TTS engine shutdown failed: %s", exc)
            raise RuntimeError("TTS reconfiguration rejected after AudioAgent shutdown")
        if previous is not None and previous_interruption_recovery is not None:
            previous_interruption_recovery.close()
            previous.shutdown()
        self._refresh_voice_metrics()
        self._record_runtime_switch_active("tts", tts_config)
        runtime = next_tts.status_snapshot()
        self._notify_tts_activated(next_tts)
        return {
            "updated": True,
            "component": "tts",
            "state": "active",
            "runtime": runtime,
        }

    def _new_interruption_recovery(
        self,
        tts: TTSEngine,
    ) -> InterruptionRecoveryCoordinator:
        return InterruptionRecoveryCoordinator(
            cast(PlaybackHoldPort, tts),
            pause_timeout_s=self._interruption_pause_timeout_s,
            hold_timeout_s=self._interruption_hold_timeout_s,
        )

    def _notify_tts_activated(self, tts: TTSEngine) -> None:
        with self._runtime_switch_lock:
            callback = self._tts_activation_callback
        if callback is None:
            return
        try:
            callback(tts)
        except Exception as exc:
            # Warming is a latency optimization. A notification consumer must
            # never roll back an already committed, otherwise healthy engine.
            logger.warning("TTS activation callback failed: %s", exc)

    def _ensure_runtime_switch_tracking(self) -> None:
        lock = getattr(self, "_runtime_switch_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._runtime_switch_lock = lock
        with lock:
            pending_asr = getattr(self, "_pending_asr_config", None)
            pending_tts = getattr(self, "_pending_tts_config", None)
            default_asr = (
                self._asr_config_selection(pending_asr)
                if isinstance(pending_asr, dict)
                else {"provider": "local", "model": "local"}
            )
            if isinstance(pending_tts, dict):
                default_tts = self._tts_config_selection(pending_tts)
            else:
                default_tts = {
                    "backend": str(getattr(getattr(self, "tts", None), "backend", "")),
                    "model": "",
                    "voice_id": "",
                }
            desired = getattr(self, "_runtime_switch_desired", None)
            if not isinstance(desired, dict):
                desired = {}
                self._runtime_switch_desired = desired
            effective = getattr(self, "_runtime_switch_effective", None)
            if not isinstance(effective, dict):
                effective = {}
                self._runtime_switch_effective = effective
            failed = getattr(self, "_runtime_switch_failed", None)
            if not isinstance(failed, dict):
                failed = {}
                self._runtime_switch_failed = failed
            desired.setdefault("asr", dict(default_asr))
            desired.setdefault("tts", dict(default_tts))
            effective.setdefault("asr", dict(default_asr))
            effective.setdefault("tts", dict(default_tts))
            failed.setdefault("asr", None)
            failed.setdefault("tts", None)
            if not hasattr(self, "_runtime_switch_callback"):
                self._runtime_switch_callback = None

    @staticmethod
    def _asr_config_selection(voice_config: dict[str, Any]) -> dict[str, str]:
        cloud_cfg = voice_config.get("cloud_asr", {})
        if not isinstance(cloud_cfg, dict) or not cloud_cfg.get("enabled"):
            return {"provider": "local", "model": "local"}
        return {
            "provider": str(cloud_cfg.get("provider") or ""),
            "model": str(cloud_cfg.get("model") or ""),
        }

    @staticmethod
    def _tts_config_selection(tts_config: dict[str, Any]) -> dict[str, str]:
        backend = str(tts_config.get("backend") or "")
        if backend == "volcengine":
            model = str(tts_config.get("volcengine_tts_resource_id") or "")
            voice_id = str(tts_config.get("volcengine_tts_speaker") or "")
        elif backend == "edge":
            model = str(tts_config.get("voice") or "")
            voice_id = model
        elif backend == "local":
            raw_model = str(tts_config.get("model_dir") or "").replace("\\", "/").rstrip("/")
            model = raw_model.rsplit("/", 1)[-1] if raw_model else ""
            voice_id = str(tts_config.get("sid") or "")
        else:
            model = str(tts_config.get("minimax_tts_model") or "")
            voice_id = str(tts_config.get("minimax_voice_id") or "")
        return {
            "backend": backend,
            "model": model,
            "voice_id": voice_id,
        }

    def _runtime_switch_selection(
        self,
        component: str,
        config: dict[str, Any],
    ) -> dict[str, str]:
        if component == "asr":
            return self._asr_config_selection(config)
        return self._tts_config_selection(config)

    def _record_runtime_switch_attempt(
        self,
        component: str,
        config: dict[str, Any],
    ) -> None:
        selection = self._runtime_switch_selection(component, config)
        with self._runtime_switch_lock:
            self._runtime_switch_desired[component] = selection
            self._runtime_switch_failed[component] = None

    def _record_runtime_switch_active(
        self,
        component: str,
        config: dict[str, Any],
    ) -> None:
        selection = self._runtime_switch_selection(component, config)
        with self._runtime_switch_lock:
            self._runtime_switch_desired[component] = dict(selection)
            self._runtime_switch_effective[component] = dict(selection)
            self._runtime_switch_failed[component] = None

    def _record_runtime_switch_failure(
        self,
        component: str,
        config: dict[str, Any],
        error: BaseException,
    ) -> None:
        selection = self._runtime_switch_selection(component, config)
        with self._runtime_switch_lock:
            self._runtime_switch_desired[component] = selection
            self._runtime_switch_failed[component] = {
                "reason": str(error) or type(error).__name__,
                "type": type(error).__name__,
            }

    def _runtime_switch_status_snapshot(self) -> dict[str, dict[str, Any]]:
        self._ensure_runtime_switch_tracking()
        with self._runtime_switch_lock:
            pending_configs = {
                "asr": self._pending_asr_config,
                "tts": self._pending_tts_config,
            }
            snapshot: dict[str, dict[str, Any]] = {}
            for component, pending_config in pending_configs.items():
                failed = self._runtime_switch_failed.get(component)
                pending = (
                    self._runtime_switch_selection(component, pending_config)
                    if pending_config is not None
                    else None
                )
                snapshot[component] = {
                    "state": "pending"
                    if pending is not None
                    else ("failed" if failed else "active"),
                    "desired": dict(self._runtime_switch_desired[component]),
                    "effective": dict(self._runtime_switch_effective[component]),
                    "pending": dict(pending) if pending is not None else None,
                    "failed": dict(failed) if failed is not None else None,
                }
            return snapshot

    def _notify_runtime_switch_outcome(
        self,
        *,
        component: str,
        state: str,
        config: dict[str, Any],
        runtime: dict[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        with self._runtime_switch_lock:
            callback = self._runtime_switch_callback
        if callback is None:
            return
        event: dict[str, Any] = {
            "component": component,
            "state": state,
            "config": config,
        }
        if runtime is not None:
            event["runtime"] = runtime
        if reason:
            event["reason"] = reason
        try:
            callback(event)
        except Exception as exc:
            logger.warning("Runtime switch callback failed: %s", exc)

    def _clear_pending_runtime_switch(
        self,
        component: str,
        config: dict[str, Any],
    ) -> bool:
        attribute = f"_pending_{component}_config"
        with self._runtime_switch_lock:
            if getattr(self, attribute, None) is not config:
                return False
            setattr(self, attribute, None)
            return True

    def _apply_pending_runtime_updates(self) -> None:
        with self._runtime_switch_lock:
            asr_config = self._pending_asr_config
            tts_config = self._pending_tts_config if not self.is_busy else None
        for component, config, apply in (
            ("asr", asr_config, self._apply_asr_config),
            ("tts", tts_config, self._apply_tts_config),
        ):
            if config is None:
                continue
            try:
                result = apply(config)
            except Exception as exc:
                if self._clear_pending_runtime_switch(component, config):
                    self._record_runtime_switch_failure(component, config, exc)
                    self._notify_runtime_switch_outcome(
                        component=component,
                        state="failed",
                        config=config,
                        reason=str(exc) or type(exc).__name__,
                    )
                logger.warning("Pending %s switch failed: %s", component.upper(), exc)
            else:
                self._notify_runtime_switch_outcome(
                    component=component,
                    state="active",
                    config=config,
                    runtime=result.get("runtime"),
                )

    # ------------------------------------------------------------------
    # Microphone mute control
    # ------------------------------------------------------------------

    def mute(self) -> None:
        """Software-mute: still listens, but VoiceLoop discards non-unmute input.

        Use cases: "闭麦" to stop the assistant from responding while a
        meeting/demo is happening; unmute with "开麦".
        """
        self._muted = True
        self._agent_state = AgentState.MUTED
        self._refresh_voice_metrics()

    def unmute(self) -> None:
        """Resume normal voice processing after a mute()."""
        self._muted = False
        self._agent_state = AgentState.IDLE
        self._refresh_voice_metrics()

    @property
    def is_muted(self) -> bool:
        """Whether the voice assistant is software-muted."""
        return self._muted

    @property
    def state(self) -> AgentState:
        """Current observable state of the audio agent."""
        return self._agent_state

    @staticmethod
    def _finite_nonnegative_feedback_seconds(value: Any, *, default: float) -> float:
        """Parse optional feedback timing without letting bad config break startup."""

        if isinstance(value, bool):
            return float(default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return max(0.0, parsed)

    def acknowledge(self) -> None:
        """Play a brief confirmation tone: 'heard you, thinking'.

        Non-blocking. Fires immediately after ASR so the user has audio
        feedback during the LLM latency gap instead of dead silence.
        """
        self._play_chime("acknowledge")

    @property
    def processing_feedback_delay_s(self) -> float:
        """Delay before non-semantic processing feedback may be played."""

        return float(getattr(self, "_processing_feedback_delay_s", 1.5))

    @property
    def processing_feedback_armed(self) -> bool:
        """Whether this accepted turn already owns the processing-feedback fuse."""

        lock = getattr(self, "_chime_lock", None)
        if lock is None:
            return bool(getattr(self, "_processing_feedback_armed", False))
        with lock:
            return bool(getattr(self, "_processing_feedback_armed", False))

    def processing_feedback_status_snapshot(self) -> dict[str, Any]:
        """Return bounded observability for the non-semantic wait fuse.

        These counters let hardware tests prove that slow turns receive
        feedback, fast turns never arm it, and semantic audio wins every race.
        No transcript or customer content is included.
        """

        lock = getattr(self, "_chime_lock", None)

        def _snapshot() -> dict[str, Any]:
            event = str(getattr(self, "_feedback_event", "") or "")
            processing_active = bool(
                getattr(self, "_feedback_active", False) and event in {"thinking", "waiting_prompt"}
            )
            return {
                "armed": bool(getattr(self, "_processing_feedback_armed", False)),
                "active": processing_active,
                "active_event": event if processing_active else "",
                "delay_ms": int(round(self.processing_feedback_delay_s * 1000.0)),
                "generation": int(getattr(self, "_processing_feedback_generation", 0)),
                "last_transition": str(
                    getattr(
                        self,
                        "_processing_feedback_last_transition",
                        "idle",
                    )
                ),
                "armed_total": int(getattr(self, "_processing_feedback_armed_total", 0)),
                "triggered_total": int(getattr(self, "_processing_feedback_triggered_total", 0)),
                "started_total": int(getattr(self, "_processing_feedback_started_total", 0)),
                "cancelled_total": int(getattr(self, "_processing_feedback_cancelled_total", 0)),
                "suppressed_total": int(getattr(self, "_processing_feedback_suppressed_total", 0)),
                "overlap_prevented_total": int(
                    getattr(
                        self,
                        "_processing_feedback_overlap_prevented_total",
                        0,
                    )
                ),
            }

        if lock is None:
            return _snapshot()
        with lock:
            return _snapshot()

    def arm_processing_feedback(self, cancel_token: Any | None = None) -> bool:
        """Start the non-semantic wait fuse at turn admission, without blocking."""

        lock = getattr(self, "_chime_lock", None)
        if lock is None:
            return False
        delay = max(0.0, self.processing_feedback_delay_s)
        with lock:
            previous = getattr(self, "_processing_feedback_timer", None)
            self._processing_feedback_generation = (
                int(getattr(self, "_processing_feedback_generation", 0)) + 1
            )
            generation = self._processing_feedback_generation
            self._processing_feedback_armed = True
            self._processing_feedback_timer = None
        if previous is not None:
            previous.cancel()

        def _fire() -> None:
            def _play_if_current() -> None:
                with lock:
                    if (
                        self._processing_feedback_generation != generation
                        or not self._processing_feedback_armed
                    ):
                        return
                    self._processing_feedback_timer = None
                self._play_processing_feedback(generation)
                with lock:
                    stale = (
                        self._processing_feedback_generation != generation
                        or not self._processing_feedback_armed
                    )
                if stale:
                    self.cancel_feedback()

            atomic_runner = getattr(cancel_token, "try_run", None)
            if callable(atomic_runner):
                ran, _ = atomic_runner(_play_if_current)
                if not ran:
                    with lock:
                        if self._processing_feedback_generation == generation:
                            self._processing_feedback_armed = False
                            self._processing_feedback_timer = None
                            self._processing_feedback_cancelled_total = (
                                int(
                                    getattr(
                                        self,
                                        "_processing_feedback_cancelled_total",
                                        0,
                                    )
                                )
                                + 1
                            )
                            self._processing_feedback_last_transition = "turn_cancelled"
                return

            cancelled = bool(
                cancel_token is not None
                and callable(getattr(cancel_token, "is_set", None))
                and cancel_token.is_set()
            )
            if cancelled:
                with lock:
                    if self._processing_feedback_generation == generation:
                        self._processing_feedback_armed = False
                        self._processing_feedback_timer = None
                        self._processing_feedback_cancelled_total = (
                            int(
                                getattr(
                                    self,
                                    "_processing_feedback_cancelled_total",
                                    0,
                                )
                            )
                            + 1
                        )
                        self._processing_feedback_last_transition = "turn_cancelled"
                return
            _play_if_current()

        timer = threading.Timer(delay, _fire)
        timer.daemon = True
        with lock:
            if (
                self._processing_feedback_generation != generation
                or not self._processing_feedback_armed
            ):
                return False
            self._processing_feedback_timer = timer
            self._processing_feedback_armed_total = (
                int(getattr(self, "_processing_feedback_armed_total", 0)) + 1
            )
            self._processing_feedback_last_transition = "armed"
        try:
            timer.start()
        except Exception as exc:
            with lock:
                if self._processing_feedback_generation == generation:
                    self._processing_feedback_armed = False
                    self._processing_feedback_timer = None
                    self._processing_feedback_armed_total = max(
                        0,
                        int(
                            getattr(
                                self,
                                "_processing_feedback_armed_total",
                                0,
                            )
                        )
                        - 1,
                    )
                    self._processing_feedback_last_transition = "arm_failed"
            logger.debug("processing feedback timer start failed: %s", exc)
            return False
        return True

    def cancel_processing_feedback(self) -> None:
        """Cancel the pending wait fuse and any feedback already being rendered."""

        lock = getattr(self, "_chime_lock", None)
        if lock is None:
            self.cancel_feedback()
            return
        with lock:
            timer = getattr(self, "_processing_feedback_timer", None)
            event = str(getattr(self, "_feedback_event", "") or "")
            was_processing = bool(
                getattr(self, "_processing_feedback_armed", False)
                or (
                    getattr(self, "_feedback_active", False)
                    and event in {"thinking", "waiting_prompt"}
                )
            )
            self._processing_feedback_generation = (
                int(getattr(self, "_processing_feedback_generation", 0)) + 1
            )
            self._processing_feedback_armed = False
            self._processing_feedback_timer = None
            if was_processing:
                self._processing_feedback_cancelled_total = (
                    int(getattr(self, "_processing_feedback_cancelled_total", 0)) + 1
                )
                self._processing_feedback_last_transition = "cancelled"
        if timer is not None:
            timer.cancel()
        self.cancel_feedback()

    def _semantic_tts_busy_for_feedback(self) -> bool:
        """Return True once semantic TTS has started or is queued."""

        tts = getattr(self, "tts", None)
        active_probe = getattr(tts, "is_active", None)
        if callable(active_probe):
            try:
                if bool(active_probe()):
                    return True
            except Exception as exc:
                logger.debug("semantic TTS activity probe failed: %s", exc)
        if bool(getattr(tts, "_is_playing", False)):
            return True
        text_queue = getattr(tts, "tts_text_queue", None)
        if text_queue is not None:
            try:
                if not text_queue.empty():
                    return True
            except Exception as exc:
                logger.debug("semantic TTS queue probe failed: %s", exc)
        return False

    def play_thinking(self) -> None:
        """Play non-semantic processing feedback while a reply is pending."""

        self._play_processing_feedback()

    def _play_processing_feedback(self, generation: int | None = None) -> None:
        with self._chime_lock:
            self._processing_feedback_triggered_total = (
                int(getattr(self, "_processing_feedback_triggered_total", 0)) + 1
            )
            self._processing_feedback_last_transition = "triggered"
        cached = self._waiting_prompt_pcm_if_available()
        if cached is not None:
            audio, sample_rate = cached
            self._play_chime(
                "waiting_prompt",
                audio=audio,
                sample_rate=sample_rate,
                expected_processing_generation=generation,
            )
            return
        self._play_chime("thinking", expected_processing_generation=generation)

    def _waiting_prompt_pcm_if_available(self) -> tuple[np.ndarray, int] | None:
        if not bool(getattr(self, "_spoken_wait_prompt_enabled", False)):
            return None
        text = str(getattr(self, "_spoken_wait_prompt_text", "") or "").strip()
        cache_key = str(getattr(self, "_spoken_wait_prompt_cache_key", "") or "").strip()
        if not text or cache_key != WAITING_FEEDBACK_CACHE_KEY:
            return None
        now = time.monotonic()
        last_at = float(getattr(self, "_last_spoken_wait_prompt_at", 0.0) or 0.0)
        min_interval = self._finite_nonnegative_feedback_seconds(
            getattr(self, "_spoken_wait_prompt_min_interval_s", 8.0),
            default=8.0,
        )
        if last_at > 0.0 and now - last_at < min_interval:
            return None
        lookup = getattr(getattr(self, "tts", None), "cached_phrase_pcm", None)
        if not callable(lookup):
            return None
        cached = lookup(text, cache_key=cache_key, target_sample_rate=self._SR)
        if cached is None:
            return None
        audio, sample_rate = cached
        self._last_spoken_wait_prompt_at = now
        return audio, int(sample_rate)

    def cancel_feedback(self) -> None:
        """Invalidate and stop an in-flight non-semantic feedback chime.

        This is deliberately separate from ``stop_immediately``: cancelling a
        350 ms thinking cue must not advance or abort the semantic TTS turn.
        """

        lock = getattr(self, "_chime_lock", None)
        if lock is None:
            return
        with lock:
            was_active = bool(getattr(self, "_feedback_active", False))
            self._feedback_generation = int(getattr(self, "_feedback_generation", 0)) + 1
            process = getattr(self, "_feedback_process", None)
            sounddevice_active = bool(getattr(self, "_feedback_sounddevice_active", False))
            sounddevice_cancel_event = getattr(
                self,
                "_feedback_sounddevice_cancel_event",
                None,
            )
            self._feedback_active = False
            self._feedback_event = None
            self._feedback_process = None
            self._feedback_sounddevice_active = False
            self._feedback_sounddevice_cancel_event = None

        if not was_active:
            return
        cancel_provider_feedback = getattr(
            getattr(self, "tts", None),
            "cancel_feedback_audio",
            None,
        )
        if callable(cancel_provider_feedback):
            try:
                cancel_provider_feedback()
            except Exception as exc:
                logger.debug("provider feedback cancellation failed: %s", exc)
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
            except Exception as exc:
                logger.debug("aplay feedback cancellation failed: %s", exc)
        if sounddevice_active and sounddevice_cancel_event is not None:
            sounddevice_cancel_event.set()

    def speak_error(self) -> None:
        """Speak a short error notification to the user."""
        self._metrics.mark_voice_error("voice interaction error")
        self._play_chime("error")
        self.tts.speak("抱歉，出现了问题，请重试。")
        self._refresh_voice_metrics()

    def _start_voice_turn_trace(self) -> str:
        self._realtime_generation_at_listen_start = self._current_realtime_generation()
        self.last_turn_realtime_baseline_generation = self._realtime_generation_at_listen_start
        self.last_turn_realtime_generation = 0
        trace = self._turn_traces.start(
            source="microphone",
            media_transport=self._media_transport,
            metadata={
                "input_transport": getattr(self._mic, "_input_transport", None),
                "sample_rate": getattr(self._mic, "sample_rate", None),
                "native_rate": getattr(self._mic, "_native_rate", None),
                "channels": getattr(self._mic, "_native_channels", None),
                "channel_select": getattr(self._mic, "_channel_select", None),
                "asr_provider": self._asr_provider_label(),
            },
        )
        self._active_capture_voice_turn_id = trace.voice_turn_id
        self.last_accepted_voice_turn_id = None
        with self._voice_turn_identity_lock:
            self.last_turn_operator_context = None
        return trace.voice_turn_id

    def bind_voice_turn_operator_context(
        self,
        voice_turn_id: str,
        context: dict[str, Any],
    ) -> bool:
        """Bind trusted verifier output to exactly one captured microphone turn."""

        normalized_turn_id = str(voice_turn_id or "").strip()
        if not normalized_turn_id or not isinstance(context, dict):
            return False
        with self._voice_turn_identity_lock:
            if normalized_turn_id not in {
                self._active_capture_voice_turn_id,
                self.last_accepted_voice_turn_id,
            }:
                return False
            self.last_turn_operator_context = {
                **dict(context),
                "voice_turn_id": normalized_turn_id,
            }
        return True

    def voice_task_operator_context_for_turn(
        self,
        _session_id: str,
        voice_turn_id: str,
    ) -> dict[str, Any] | None:
        """Return verifier output only when it is bound to the requested turn."""

        normalized_turn_id = str(voice_turn_id or "").strip()
        with self._voice_turn_identity_lock:
            context = self.last_turn_operator_context
            if not isinstance(context, dict):
                return None
            if str(context.get("voice_turn_id") or "").strip() != normalized_turn_id:
                return None
            self.last_turn_operator_context = None
            return dict(context)

    def _record_output_voice_trace_for_token(
        self,
        token: OutputPlaybackTraceToken,
        name: str,
        **metadata: Any,
    ) -> bool:
        """Record one output fact against an already-fenced playback owner."""

        if token.voice_turn_id:
            try:
                return self._turn_traces.mark_for(
                    token.voice_turn_id,
                    name,
                    **metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Output trace projection failed for %s: %s",
                    name,
                    type(exc).__name__,
                )
                return False
        with self._output_trace_lock:
            self._orphan_output_trace_event_count += 1
        return False

    def _mark_output_voice_trace(self, name: str, **metadata: Any) -> bool:
        """Route asynchronous output facts to the active playback owner."""

        with self._output_trace_lock:
            token = self._active_output_trace_token
        if token is not None:
            return self._record_output_voice_trace_for_token(
                token,
                name,
                **metadata,
            )
        with self._output_trace_lock:
            self._orphan_output_trace_event_count += 1
        return False

    def _mark_output_barge_in(self, **metadata: Any) -> bool:
        with self._output_trace_lock:
            token = self._active_output_trace_token
        if token is not None and token.voice_turn_id:
            try:
                return self._turn_traces.mark_barge_in_for(
                    token.voice_turn_id,
                    **metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Output barge-in trace projection failed: %s",
                    type(exc).__name__,
                )
                return False
        with self._output_trace_lock:
            self._orphan_output_trace_event_count += 1
        return False

    def _begin_output_interruption(self, **metadata: Any) -> bool:
        """Freeze the playback owner before opening a validation hold."""

        with self._output_trace_lock:
            token = self._active_output_trace_token
            if token is None:
                self._orphan_output_trace_event_count += 1
                self._interruption_output_trace_token = None
                return False
            self._record_output_voice_trace_for_token(
                token,
                "barge_in_detected",
                **metadata,
            )
            held = self._interruption_recovery.begin_detection()
            self._interruption_output_trace_token = token
            return held

    def _confirm_output_interruption(self, **metadata: Any) -> bool:
        """Confirm only the playback interval frozen at detection time."""

        with self._output_trace_lock:
            token = self._interruption_output_trace_token
            if token is None:
                token = self._active_output_trace_token
                if token is None:
                    self._orphan_output_trace_event_count += 1
                    return False
                # Some VAD implementations collapse START and CONFIRMED into
                # one event.  Freeze the owner before confirm() opens its hold.
                self._interruption_output_trace_token = token
            if self._active_output_trace_token != token:
                self._stale_playback_stop_count += 1
                self._close_interruption_recovery()
                return False
            if token.voice_turn_id:
                try:
                    marked = self._turn_traces.mark_barge_in_for(
                        token.voice_turn_id,
                        **metadata,
                    )
                except Exception as exc:
                    logger.warning(
                        "Output barge-in trace projection failed: %s",
                        type(exc).__name__,
                    )
                    marked = False
            else:
                self._orphan_output_trace_event_count += 1
                marked = False
            self._interruption_recovery.confirm()
            return marked

    def _expire_output_interruption_hold(self) -> bool:
        """Resume a timed-out hold and attribute it to its frozen owner."""

        with self._output_trace_lock:
            token = self._interruption_output_trace_token
            if not self._interruption_recovery.expire_hold():
                return False
            if token is not None:
                self._record_output_voice_trace_for_token(
                    token,
                    "barge_in_hold_timeout",
                    reason_code="validation_hold_timeout",
                )
            else:
                self._orphan_output_trace_event_count += 1
            return True

    def _close_active_output_trace_locked(self, reason_code: str) -> bool:
        """Close and clear the active output owner while holding its lock."""

        token = self._active_output_trace_token
        if token is None:
            return False
        self._record_output_voice_trace_for_token(
            token,
            "playback_done",
            reason_code=reason_code,
        )
        self._active_output_trace_token = None
        if self._realtime_output_trace_token == token:
            self._realtime_output_trace_token = None
        if self._output_trace_context.get() == token:
            self._output_trace_context.set(None)
        return True

    # ------------------------------------------------------------------
    # Microphone listen loop
    # ------------------------------------------------------------------

    def _process_capture_frame(
        self,
        samples: np.ndarray,
        *,
        sample_rate_hz: int,
        tts_active: bool,
    ) -> np.ndarray | None:
        self._capture_processor_failed_current_frame = False
        processor = self._capture_processor
        if processor is None:
            return None
        try:
            processed = np.asarray(
                processor(samples, sample_rate_hz, tts_active),
                dtype=np.float32,
            )
            if processed.shape != samples.shape:
                raise ValueError("capture processor must preserve the input sample shape")
            if not np.all(np.isfinite(processed)):
                raise ValueError("capture processor returned non-finite samples")
            return np.ascontiguousarray(processed, dtype=np.float32)
        except Exception as exc:
            self._capture_processor_failed_current_frame = True
            failure_callback = self._capture_processor_failure_callback
            # A failed AEC implementation is never used again in this
            # session.  The configured failure callback owns fail-closed
            # routing and restoration of the half-duplex gates.
            self._capture_processor = None
            self._capture_processor_failure_callback = None
            if not self._capture_processor_error_logged:
                logger.warning(
                    "Capture processor failed; using unprocessed microphone frame: %s",
                    exc,
                )
                self._capture_processor_error_logged = True
            if failure_callback is not None:
                try:
                    failure_callback(exc)
                except Exception as failure_exc:
                    logger.error(
                        "Capture processor fail-closed callback failed: %s",
                        failure_exc,
                    )
            return None

    def _notify_confirmed_barge_in(self) -> bool:
        """Commit a validated interruption, then stop its fenced output."""

        recovery = getattr(self, "_interruption_recovery", None)
        if recovery is None or recovery.state is InterruptionRecoveryState.IDLE:
            return False
        with self._output_trace_lock:
            token = self._interruption_output_trace_token
            if token is None or self._active_output_trace_token != token:
                self._stale_playback_stop_count += 1
                self._close_interruption_recovery()
                return False

            # Mark the pipeline turn cancelled first; the callback is
            # intentionally lightweight and must become visible before any
            # device/network work.
            callback = self._barge_in_callback
            if callback is not None:
                try:
                    callback()
                except Exception as exc:
                    logger.warning("Confirmed barge-in callback failed: %s", exc)

            # The ASR result has now passed local admission.  Abort only the
            # exact playback generation held for this candidate.
            recovery.commit("accepted_transcript")
            self._interruption_output_trace_token = None

            # Stop the physical sink before any cloud truncate/delete wait.
            self.tts.stop_immediately()
            if token.provider_generation is not None:
                self._truncate_or_discard_realtime_playback(
                    "barge_in",
                    asynchronous=True,
                    expected_generation=token.provider_generation,
                )
            self.tts.drain_buffers()
            self.tts.stop_playback()
            self._agent_state = AgentState.IDLE
            self._close_active_output_trace_locked("barge_in")
        return True

    def _recover_interrupted_playback(
        self,
        reason: str,
        *,
        expected_token: OutputPlaybackTraceToken | None = None,
    ) -> bool:
        """Recover only the output interval frozen by the calling event."""

        recovery = getattr(self, "_interruption_recovery", None)
        if recovery is None:
            return False
        with self._output_trace_lock:
            token = self._interruption_output_trace_token
            if token is None:
                return False
            if (
                expected_token is not None and token != expected_token
            ) or self._active_output_trace_token != token:
                self._stale_playback_stop_count += 1
                return False
            if not recovery.recover(reason):
                return False
            if self._interruption_output_trace_token == token:
                self._interruption_output_trace_token = None
            self._record_output_voice_trace_for_token(
                token,
                "barge_in_recovered",
                reason_code=reason,
            )
            return True

    def _close_interruption_recovery(self) -> bool:
        recovery = getattr(self, "_interruption_recovery", None)
        if recovery is None:
            return False
        with self._output_trace_lock:
            closed = bool(recovery.close())
            if closed or recovery.state is InterruptionRecoveryState.IDLE:
                self._interruption_output_trace_token = None
            return closed

    def _interruption_recovery_status_snapshot(self) -> dict[str, object]:
        recovery = getattr(self, "_interruption_recovery", None)
        if recovery is None:
            return {
                "state": "unavailable",
                "hold_active": False,
                "hold_supported": None,
            }
        return recovery.status_snapshot()

    def stop_listening(self, *, timeout: float = 2.5) -> bool:
        """Cooperatively stop and join the current microphone listen cycle.

        The process-wide ``stop_event`` remains reserved for final shutdown;
        this per-cycle cancellation lets ``VoiceLoop`` stop and restart without
        leaving the worker created by ``asyncio.to_thread`` behind.
        """

        with self._runtime_switch_lock:
            cancel_event = self._listen_cancel_event
            active = self._listen_loop_active
            listen_thread_id = self._listen_thread_id
            listen_stopped = self._listen_loop_stopped
        if not active or cancel_event is None:
            return True

        cancel_event.set()
        self._wake_microphone_reader()
        if listen_thread_id == threading.get_ident():
            return False
        with self._runtime_switch_lock:
            asr_phase_active = self._asr_phase_active
            asr_manager = self._asr_mgr
        if asr_phase_active:
            abort_session = getattr(asr_manager, "abort_session", None)
            if callable(abort_session):
                try:
                    abort_session()
                except Exception as exc:
                    logger.warning("Failed to abort active ASR session: %s", exc)
        if listen_stopped.wait(timeout=max(0.0, timeout)):
            return True
        if not asr_phase_active:
            return False

        # Provider implementations have bounded connect/final-result waits.
        # If socket cancellation could not release one immediately, wait once
        # through that configured bound before allowing a caller to restart.
        provider_timeout = float(getattr(asr_manager, "_cloud_finish_timeout", 5.0) or 5.0)
        provider_grace = min(10.0, max(0.5, provider_timeout + 0.5))
        return listen_stopped.wait(timeout=provider_grace)

    def _wake_microphone_reader(self) -> None:
        """Wake either capture adapter so it can observe listen cancellation."""

        mic = self._mic
        if getattr(mic, "_usb_audio_proc", None) is not None:
            try:
                mic.stop()
            except Exception as exc:
                logger.debug("Failed to stop direct USB capture: %s", exc)
            return

        audio_queue = getattr(mic, "_audio_queue", None)
        chunk_samples = int(getattr(mic, "_chunk_samples", 0) or 0)
        if audio_queue is None or chunk_samples <= 0:
            return
        try:
            audio_queue.put_nowait(np.zeros(chunk_samples, dtype=np.float32))
        except queue.Full:
            # A queued frame already guarantees that read_chunk will wake.
            pass

    def listen_loop(self) -> str | None:
        """Listen with VAD-gated ASR using modular pipeline.

        Flow: MicInput -> AudioProcessor -> VADController -> ASRManager -> text

        Returns recognized text, or None on timeout/stop.
        """
        vad_controller = self._vad_ctrl
        asr_manager = self._asr_mgr
        if self.asr is None or self.vad is None or vad_controller is None or asr_manager is None:
            raise RuntimeError("listen_loop requires voice_mode=True")

        mic = self._mic
        proc = self._audio_proc
        vad = vad_controller

        with self._runtime_switch_lock:
            if self._listen_loop_active:
                raise RuntimeError("listen_loop already has an active microphone consumer")
            listen_cancel = threading.Event()
            listen_stopped = threading.Event()
            self._listen_cancel_event = listen_cancel
            self._listen_loop_stopped = listen_stopped
            self._listen_loop_active = True
            self._listen_thread_id = threading.get_ident()
            self._asr_phase_active = False
        interruption_token: OutputPlaybackTraceToken | None = None

        def recover_frozen_interruption(reason: str) -> bool:
            nonlocal interruption_token
            token = interruption_token
            if token is None:
                return False
            recovered = self._recover_interrupted_playback(
                reason,
                expected_token=token,
            )
            if recovered:
                interruption_token = None
            return recovered

        try:
            with self._output_trace_lock:
                prior_interruption_token = self._interruption_output_trace_token
            self._recover_interrupted_playback(
                "new_listen_cycle",
                expected_token=prior_interruption_token,
            )
            self._prepare_realtime_turn_boundary()
            self._start_voice_turn_trace()
            self.last_turn_wake_authorized = False
            self.last_turn_asr_confidence = None
            self.last_turn_wake_source = "none"
            barge_keyword_authorized = False
            self._metrics.mark_voice_listen_started()
            self._refresh_voice_metrics()

            # Mic is persistently open (started by VoiceModule).
            # Flush stale audio that accumulated during LLM+TTS processing.
            mic._flush_queue()

            with mic.open() as mic_ctx:
                # Phase 1: Wake word detection (if KWS available)
                if self.wake_word_ready:
                    _within_wake_window = self._followup_window_active()
                    if _within_wake_window:
                        self.last_turn_wake_source = "followup_window"
                        logger.info(
                            "Wake timeout active (%.0fs left), skipping KWS",
                            self._wake_timeout - (time.monotonic() - self._last_interaction_time),
                        )
                    else:
                        self.woken_up = False
                        self._refresh_voice_metrics()
                        if not self._wait_for_wake_word_mic(
                            mic_ctx,
                            cancel_event=listen_cancel,
                        ):
                            self._turn_traces.finish("wake_word_not_detected")
                            return None
                        self.last_turn_wake_authorized = True
                        self.last_turn_wake_source = "keyword"
                        self._play_chime("wake")
                elif self.kws_unavailable_safety_only:
                    self.last_turn_wake_source = "kws_unavailable_safety_only"
                    logger.error(
                        "KWS is unavailable while wake word is required; "
                        "accepting local safety commands only"
                    )
                elif self._followup_window_active():
                    self.last_turn_wake_source = "followup_window"
                    logger.info(
                        "Follow-up window active (%.0fs left), accepting speech without wake word",
                        self._wake_timeout - (time.monotonic() - self._last_interaction_time),
                    )
                else:
                    self.last_turn_wake_source = "always_awake"

                # Phase 2: VAD-gated ASR. The wake chime above already routes
                # through the configured cross-platform output path.
                with self._runtime_switch_lock:
                    self._asr_phase_active = True
                    asr = self._asr_mgr
                if asr is None:
                    raise RuntimeError("ASR manager became unavailable during listen")
                if listen_cancel.is_set() or self.stop_event.is_set():
                    self._turn_traces.finish("cancelled")
                    return None
                logger.info("Listening for speech...")
                asr.preconnect_cloud()  # warm up WebSocket (fast, ~100ms)
                if listen_cancel.is_set() or self.stop_event.is_set():
                    abort_session = getattr(asr, "abort_session", None)
                    if callable(abort_session):
                        abort_session()
                    self._turn_traces.finish("cancelled")
                    return None
                deadline = time.monotonic() + self._asr_timeout
                vad.reset()
                self._fast_endpoint.reset()
                _vol_log_interval = 0.5
                _vol_log_next = time.monotonic() + _vol_log_interval

                while not self.stop_event.is_set() and not listen_cancel.is_set():
                    if self._agent_state not in (AgentState.MUTED, AgentState.LISTENING):
                        self._agent_state = AgentState.IDLE

                    self._expire_output_interruption_hold()

                    if time.monotonic() > deadline:
                        logger.info(
                            "ASR timeout: no speech detected within %.0fs.", self._asr_timeout
                        )
                        asr.reset()
                        self._mark_input_failure("asr_timeout")
                        recover_frozen_interruption("asr_timeout")
                        self._turn_traces.finish("timeout")
                        self._refresh_voice_metrics()
                        return None

                    raw = mic_ctx.read_chunk()
                    if listen_cancel.is_set():
                        self._turn_traces.finish("cancelled")
                        return None
                    self._turn_traces.mark(
                        "first_audio_frame",
                        chunk_samples=len(raw),
                        sample_rate=mic_ctx.sample_rate,
                    )
                    cooldown_remaining = self._in_post_tts_input_cooldown()
                    if cooldown_remaining > 0:
                        raw_i16 = (raw * 32768).clip(-32768, 32767).astype(np.int16)
                        self._record_input_observation(
                            peak=int(np.max(np.abs(raw_i16))) if len(raw_i16) else 0,
                            rms=self._rms_int16(raw_i16),
                            vad_state="cooldown",
                            gate_state="cooldown",
                        )
                        now = time.monotonic()
                        if now >= self._input_cooldown_log_next:
                            logger.info(
                                "MIC input suppressed for %.1fs after TTS playback",
                                cooldown_remaining,
                            )
                            self._input_cooldown_log_next = now + 0.5
                        self._reset_listening_state()
                        mic_ctx.pre_roll.clear()
                        continue

                    tts_active = self.tts.is_active()
                    result = proc.process(
                        raw, tts_active=tts_active, speech_active=vad.speech_active
                    )
                    samples_f32, samples_i16, peak, echo_gated = result
                    processed_capture = self._process_capture_frame(
                        samples_f32,
                        sample_rate_hz=mic_ctx.sample_rate,
                        tts_active=tts_active,
                    )
                    if processed_capture is not None:
                        samples_f32 = processed_capture
                        samples_i16 = (samples_f32 * 32767.0).clip(-32768, 32767).astype(np.int16)
                        peak = (
                            int(np.max(np.abs(samples_i16.astype(np.int32))))
                            if samples_i16.size
                            else 0
                        )
                        # A successful real capture processor supersedes the
                        # legacy amplitude echo gate for this frame.
                        echo_gated = False
                    elif self._capture_processor_failed_current_frame and tts_active:
                        # The frame that exposed an AEC failure was processed
                        # while the legacy gate was disabled.  Never admit that
                        # raw speaker-echo frame into VAD during fail-closed.
                        echo_gated = True
                    rms = self._rms_int16(samples_i16)

                    # Periodic volume logging
                    now = time.monotonic()
                    if now >= _vol_log_next:
                        if echo_gated:
                            logger.info("MIC peak=%5d VAD=gated (TTS playing)", peak)
                        else:
                            vad_label = "SPEECH" if vad.speech_active else "silent"
                            bar = "#" * min(peak // 500, 30)
                            logger.info("MIC peak=%5d VAD=%s %s", peak, vad_label, bar)
                        _vol_log_next = now + _vol_log_interval

                    if echo_gated:
                        self._record_input_observation(
                            peak=peak,
                            rms=rms,
                            vad_state="gated",
                            gate_state="echo",
                        )
                        mic_ctx.buffer_pre_roll(raw)
                        continue

                    # Feed the optional S2S path only after the same capture
                    # processing/AEC and echo fail-closed checks used locally.
                    # This is a non-blocking bounded offer; local ASR remains
                    # authoritative if the cloud path is slow or unavailable.
                    self._offer_realtime_capture(
                        samples_i16,
                        sample_rate=mic_ctx.sample_rate,
                    )

                    # Noise gate: skip VAD when peak below threshold AND not in speech.
                    # HKMIC noise floor ~15, quiet speech moments ~74-164, gate at 50.
                    # Only gates silence→speech transition; during speech all audio passes
                    # to keep Cloud ASR stream continuous.
                    if proc.is_noise_gated(peak) and not vad.speech_active:
                        self._record_input_observation(
                            peak=peak,
                            rms=rms,
                            vad_state="silent",
                            gate_state="noise",
                        )
                        mic_ctx.buffer_pre_roll(raw)
                        continue

                    # Once the wake keyword has authorized a barge-in, VAD must
                    # treat the following words as an ordinary utterance even
                    # while the old playback thread is still unwinding.
                    vad_tts_active = tts_active and not barge_keyword_authorized
                    event = vad.feed(samples_i16, peak, tts_active=vad_tts_active)
                    self._record_input_observation(
                        peak=peak,
                        rms=rms,
                        vad_state="speech" if vad.speech_active else "silent",
                        gate_state="open",
                    )

                    if event == VADEvent.SILENCE:
                        mic_ctx.buffer_pre_roll(raw)
                        # Keep cloud ASR connected; actual silence feeding is configurable.
                        if asr._cloud_active:
                            asr.feed_cloud_only(samples_i16)

                    elif event == VADEvent.SPEECH_START:
                        deadline = time.monotonic() + self._asr_timeout
                        self._fast_endpoint.reset()
                        self._turn_traces.mark(
                            "vad_start",
                            peak=peak,
                            rms=rms,
                        )
                        self._agent_state = AgentState.LISTENING
                        self._refresh_voice_metrics()
                        self._fast_endpoint.reset()
                        asr.start_session()
                        if listen_cancel.is_set() or self.stop_event.is_set():
                            abort_session = getattr(asr, "abort_session", None)
                            if callable(abort_session):
                                abort_session()
                            self._turn_traces.finish("cancelled")
                            return None
                        for buf in mic_ctx.flush_pre_roll():
                            asr.feed_audio(buf, MicInput.to_int16(buf), mic_ctx.sample_rate)
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.SPEECH_CONTINUE:
                        deadline = time.monotonic() + self._asr_timeout
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.BARGE_IN_START:
                        self._begin_output_interruption(
                            peak=peak,
                            rms=rms,
                        )
                        with self._output_trace_lock:
                            interruption_token = self._interruption_output_trace_token

                    elif event == VADEvent.BARGE_IN_CONFIRMED:
                        self._confirm_output_interruption(peak=peak, rms=rms)
                        if interruption_token is None:
                            with self._output_trace_lock:
                                interruption_token = self._interruption_output_trace_token
                        self._agent_state = AgentState.LISTENING
                        self._refresh_voice_metrics()
                        # Confirmation opens ASR validation only.  Destructive
                        # cancellation waits until _accept_result admits text.
                        asr.start_session()
                        if listen_cancel.is_set() or self.stop_event.is_set():
                            abort_session = getattr(asr, "abort_session", None)
                            if callable(abort_session):
                                abort_session()
                            self._turn_traces.finish("cancelled")
                            return None
                        for buf in vad.barge_in_buffer:
                            asr.feed_audio(buf, MicInput.to_int16(buf), mic_ctx.sample_rate)
                        vad.barge_in_buffer.clear()
                        mic_ctx.pre_roll.clear()
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.BARGE_IN_DISMISSED:
                        recover_frozen_interruption("vad_dismissed")
                        mic_ctx.buffer_pre_roll(raw)

                    elif event == VADEvent.SPEECH_END:
                        logger.info("VAD: speech end")
                        self._turn_traces.mark("vad_end", peak=peak, rms=rms)
                        cloud_result = asr.finish_and_get_result(self.awaiting_confirmation)
                        if listen_cancel.is_set() or self.stop_event.is_set():
                            self._turn_traces.finish("cancelled")
                            return None
                        if cloud_result and not cloud_result.is_noise:
                            return self._accept_result(
                                cloud_result.text,
                                asr_source=cloud_result.source,
                                asr_latency_ms=cloud_result.latency_ms,
                                asr_confidence=getattr(cloud_result, "confidence", None),
                                interruption_token=interruption_token,
                            )
                        if cloud_result and cloud_result.is_noise:
                            logger.info(
                                "ASR noise filtered (chars=%d)",
                                len(cloud_result.text),
                            )
                            self._mark_input_failure("asr_noise_filtered")
                            recover_frozen_interruption("asr_noise_filtered")
                            self._turn_traces.finish(
                                "noise_filtered",
                                asr_source=cloud_result.source,
                            )
                            asr.reset()
                            deadline = time.monotonic() + self._asr_timeout
                            vad.reset()
                            self._start_voice_turn_trace()
                            continue

                    elif event == VADEvent.MAX_DURATION_EXCEEDED:
                        logger.warning("VAD: max speech duration exceeded, forcing endpoint")
                        forced = asr.force_endpoint()
                        if forced and not forced.is_noise:
                            return self._accept_result(
                                forced.text,
                                asr_source=forced.source,
                                asr_latency_ms=forced.latency_ms,
                                asr_confidence=getattr(forced, "confidence", None),
                                forced_endpoint=True,
                                interruption_token=interruption_token,
                            )
                        self._mark_input_failure("asr_forced_empty")
                        recover_frozen_interruption("asr_forced_empty")
                        deadline = time.monotonic() + self._asr_timeout
                        self._turn_traces.finish("forced_empty")
                        self._start_voice_turn_trace()
                        continue

                    # Safe fixed intents may commit from a stable streaming
                    # transcript during low-energy trailing audio.  This avoids
                    # waiting for the full VAD tail and cloud finalization, but
                    # never admits physical actions.
                    if event == VADEvent.SPEECH_CONTINUE and self._fast_endpoint.enabled:
                        partial = asr.partial_result()
                        decision = self._fast_endpoint.observe(
                            partial_text=partial.text if partial is not None else "",
                            quiet=(
                                self._fast_quiet_peak_threshold > 0
                                and peak < self._fast_quiet_peak_threshold
                            ),
                            now=now,
                        )
                        if decision.action is FastEndpointAction.COMMIT and partial is not None:
                            fast_result = asr.commit_partial(
                                partial,
                                self.awaiting_confirmation,
                            )
                            if fast_result and not fast_result.is_noise:
                                self._turn_traces.mark(
                                    "fast_endpoint",
                                    intent_id=(
                                        decision.intent.intent_id
                                        if decision.intent is not None
                                        else ""
                                    ),
                                    silence_ms=round(decision.silence_ms, 2),
                                    stable_text_ms=round(decision.stable_text_ms, 2),
                                )
                                return self._accept_result(
                                    fast_result.text,
                                    asr_source=fast_result.source,
                                    asr_latency_ms=fast_result.latency_ms,
                                    asr_confidence=getattr(fast_result, "confidence", None),
                                    interruption_token=interruption_token,
                                )

                    # Check local ASR endpoint (runs every iteration during speech)
                    ep_result = asr.check_endpoint()
                    if ep_result:
                        # Finish cloud session and prefer cloud result over local
                        cloud_result = asr.finish_and_get_result(self.awaiting_confirmation)
                        if cloud_result and not cloud_result.is_noise:
                            return self._accept_result(
                                cloud_result.text,
                                asr_source=cloud_result.source,
                                asr_latency_ms=cloud_result.latency_ms,
                                asr_confidence=getattr(cloud_result, "confidence", None),
                                interruption_token=interruption_token,
                            )
                        # Fall back to local ASR result
                        is_noise = asr.is_noise(ep_result.text, self.awaiting_confirmation)
                        if not is_noise:
                            return self._accept_result(
                                ep_result.text,
                                asr_source=ep_result.source,
                                asr_latency_ms=ep_result.latency_ms,
                                asr_confidence=getattr(ep_result, "confidence", None),
                                interruption_token=interruption_token,
                            )
                        else:
                            logger.info(
                                "ASR noise filtered (chars=%d)",
                                len(ep_result.text),
                            )
                            self._mark_input_failure("asr_noise_filtered")
                            recover_frozen_interruption("asr_noise_filtered")
                            self._turn_traces.finish(
                                "noise_filtered",
                                asr_source=ep_result.source,
                            )
                            asr.reset()
                            vad.reset()
                            deadline = time.monotonic() + self._asr_timeout
                            self._start_voice_turn_trace()

        except Exception as exc:
            if self.stop_event.is_set() or listen_cancel.is_set():
                self._turn_traces.finish("cancelled")
                return None
            recover_frozen_interruption("asr_error")
            error_type = type(exc).__name__
            self._metrics.mark_voice_error(error_type)
            self._mark_input_failure("asr_error")
            self._turn_traces.finish("error", error_type=error_type)
            self._refresh_voice_metrics(pipeline_ok=False)
            raise
        finally:
            recover_frozen_interruption("listen_loop_exit")
            if (listen_cancel.is_set() or self.stop_event.is_set()) and int(
                getattr(self, "last_turn_realtime_generation", 0) or 0
            ) <= 0:
                self._discard_realtime_capture_if_started("listen_cancelled")
            with self._runtime_switch_lock:
                self._listen_loop_active = False
                self._asr_phase_active = False
                if self._listen_cancel_event is listen_cancel:
                    self._listen_cancel_event = None
                self._listen_thread_id = None
                listen_stopped.set()
            with self._realtime_recovery_lock:
                self._realtime_capture_armed = False
            self._apply_pending_runtime_updates()

        return None

    def _accept_result(
        self,
        text: str,
        *,
        asr_source: str = "",
        asr_latency_ms: float | None = None,
        asr_confidence: float | None = None,
        forced_endpoint: bool = False,
        interruption_token: OutputPlaybackTraceToken | None = None,
    ) -> str | None:
        """Accept a recognized text result: log, queue, update state."""
        if asr_confidence is not None:
            try:
                asr_confidence = float(asr_confidence)
            except (TypeError, ValueError):
                asr_confidence = None
        if asr_confidence is not None and (
            not math.isfinite(asr_confidence) or not 0.0 <= asr_confidence <= 1.0
        ):
            asr_confidence = None
        capture_voice_turn_id = getattr(self, "_active_capture_voice_turn_id", None)
        asr_metadata = {
            "asr_source": asr_source,
            "asr_latency_ms": asr_latency_ms,
            "forced_endpoint": forced_endpoint,
            "asr_confidence": asr_confidence,
            "text_chars": len(text),
        }
        if capture_voice_turn_id:
            self._turn_traces.mark_for(
                capture_voice_turn_id,
                "asr_final",
                **asr_metadata,
            )
        else:
            self._turn_traces.mark("asr_final", **asr_metadata)
        if self.kws_unavailable_safety_only and not is_local_safety_utterance(text):
            reason = "kws_unavailable_safety_only_filtered"
            if capture_voice_turn_id:
                self._turn_traces.finish_for(
                    capture_voice_turn_id,
                    reason,
                    asr_source=asr_source,
                    text_chars=len(text),
                )
            else:
                self._turn_traces.finish(
                    reason,
                    asr_source=asr_source,
                    text_chars=len(text),
                )
            self._active_capture_voice_turn_id = None
            self.last_accepted_voice_turn_id = None
            logger.warning(
                "Discarded non-safety ASR result because required KWS is unavailable (chars=%d)",
                len(text),
            )
            self._recover_interrupted_playback(
                reason,
                expected_token=interruption_token,
            )
            self._discard_realtime_capture_if_started(reason)
            self.last_turn_realtime_generation = 0
            self._agent_state = (
                AgentState.MUTED if getattr(self, "_muted", False) else AgentState.IDLE
            )
            asr_manager = getattr(self, "_asr_mgr", None)
            if asr_manager is not None:
                asr_manager.reset()
            self._refresh_voice_metrics()
            return None

        self._notify_confirmed_barge_in()
        if capture_voice_turn_id:
            self._turn_traces.finish_for(
                capture_voice_turn_id,
                "accepted",
                asr_source=asr_source,
                text_chars=len(text),
            )
        else:
            self._turn_traces.finish(
                "accepted",
                asr_source=asr_source,
                text_chars=len(text),
            )
        self._active_capture_voice_turn_id = None
        self.last_accepted_voice_turn_id = capture_voice_turn_id
        logger.info("ASR result accepted (chars=%d)", len(text))
        self.last_turn_asr_confidence = asr_confidence
        coordinator = getattr(self, "_realtime_coordinator", None)
        finish_input = getattr(coordinator, "finish_input", None)
        realtime_capture_armed = bool(
            coordinator is not None and self._realtime_turn_capture_is_armed()
        )
        if realtime_capture_armed and callable(finish_input):
            try:
                finish_result = finish_input()
            except Exception as exc:
                self._fence_failed_realtime_capture(
                    coordinator,
                    "finish_input_failed",
                    exc,
                )
                realtime_capture_armed = False
            else:
                if finish_result is False:
                    self._fence_failed_realtime_capture(
                        coordinator,
                        "finish_input_failed",
                    )
                    realtime_capture_armed = False
        if realtime_capture_armed:
            current_realtime_generation = self._current_realtime_generation()
            if current_realtime_generation > int(
                getattr(self, "_realtime_generation_at_listen_start", 0) or 0
            ):
                self.last_turn_realtime_generation = current_realtime_generation
            else:
                self.last_turn_realtime_generation = 0
        else:
            self.last_turn_realtime_generation = 0
        self.audio_queue.put(text)
        self._metrics.mark_voice_input(text)
        self._clear_input_failure()
        self._agent_state = AgentState.PROCESSING
        self._refresh_voice_metrics()
        asr_manager = self._asr_mgr
        if asr_manager is not None:
            asr_manager.reset()
        return text

    def _current_realtime_generation(self) -> int:
        coordinator = getattr(self, "_realtime_coordinator", None)
        if coordinator is None:
            return 0
        try:
            status = coordinator.status_snapshot()
        except Exception:
            return 0
        try:
            return max(0, int(status.get("generation", 0)))
        except (TypeError, ValueError):
            return 0

    def _discard_realtime_capture_if_started(self, reason: str) -> None:
        """Delete a provider turn that local capture did not admit."""

        if not self._realtime_turn_capture_is_armed():
            return
        baseline = int(getattr(self, "_realtime_generation_at_listen_start", 0) or 0)
        generation = self._current_realtime_generation()
        try:
            if generation > baseline:
                self.discard_realtime_turn(
                    reason,
                    expected_generation=generation,
                    after_generation=baseline,
                )
        finally:
            with self._realtime_recovery_lock:
                self._realtime_capture_armed = False

    def mark_interaction_turn(self) -> None:
        """Renew the weak follow-up window only after a turn is admitted."""
        self._last_interaction_time = time.monotonic()
        self._refresh_voice_metrics()

    def _followup_window_active(self) -> bool:
        """Return whether speech may continue without repeating the wake word."""
        return bool(
            self._wake_timeout > 0
            and self._last_interaction_time > 0
            and (time.monotonic() - self._last_interaction_time) < self._wake_timeout
        )

    @staticmethod
    def _rms_int16(samples_int16: np.ndarray) -> float:
        if len(samples_int16) == 0:
            return 0.0
        values: np.ndarray = samples_int16.astype(np.float64)
        return round(float(np.sqrt(np.mean(values * values))), 2)

    def _record_input_observation(
        self,
        *,
        peak: int,
        rms: float,
        vad_state: str,
        gate_state: str,
    ) -> None:
        now = time.monotonic()
        with self._input_state_lock:
            self._input_last_peak = int(max(peak, 0))
            self._input_last_rms = float(max(rms, 0.0))
            self._input_last_observed_at = now
            self._input_vad_state = vad_state
            self._input_gate_state = gate_state
            self._input_level_window.append((now, self._input_last_peak, self._input_last_rms))

    def _mark_input_failure(self, reason: str) -> None:
        with self._input_state_lock:
            if reason == "asr_timeout":
                self._input_asr_timeouts += 1
            self._input_last_failure_reason = str(reason)
            self._input_vad_state = "timeout" if reason == "asr_timeout" else self._input_vad_state
        self._discard_realtime_capture_if_started(str(reason or "input_failure"))

    def _clear_input_failure(self) -> None:
        with self._input_state_lock:
            self._input_last_failure_reason = None

    def _input_status_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        cutoff = now - 10.0
        with self._input_state_lock:
            while self._input_level_window and self._input_level_window[0][0] < cutoff:
                self._input_level_window.popleft()
            peaks = [peak for _observed_at, peak, _rms in self._input_level_window]
            rms_values = [rms for _observed_at, _peak, rms in self._input_level_window]
            peak_max_10s = max(
                peaks,
                default=self._input_last_peak,
            )
            peak_p50_10s = self._percentile(peaks, 50)
            peak_p95_10s = self._percentile(peaks, 95)
            rms_p50_10s = self._percentile(rms_values, 50)
            rms_p95_10s = self._percentile(rms_values, 95)
            sample_count_10s = len(peaks)
            last_observed_age_s = (
                round(now - self._input_last_observed_at, 2)
                if self._input_last_observed_at > 0
                else None
            )
            last_peak = self._input_last_peak
            last_rms = self._input_last_rms
            vad_state = self._input_vad_state
            gate_state = self._input_gate_state
            asr_timeouts = self._input_asr_timeouts
            last_failure_reason = self._input_last_failure_reason

        noise_gate_peak = int(getattr(self._audio_proc, "noise_gate_peak", self._noise_gate_peak))
        gate_recommendation = None
        if sample_count_10s == 0:
            gate_recommendation = "no_microphone_frames_observed:listen_loop_not_sampling_input"
        elif peak_max_10s <= 1:
            gate_recommendation = (
                "microphone_captured_silence:check_input_device_permission_or_physical_mute"
            )
        elif noise_gate_peak > 0 and peak_max_10s < noise_gate_peak:
            gate_recommendation = f"observed_peak_below_noise_gate:{peak_max_10s}<{noise_gate_peak}"

        tts_is_active = getattr(self.tts, "is_active", None)
        if callable(tts_is_active):
            try:
                tts_active = bool(tts_is_active())
            except Exception:
                logger.exception("[Audio] tts_is_active() check failed, using fallback")
                tts_active = self.is_busy
        else:
            tts_active = self.is_busy

        return {
            "run_id": self._run_id,
            "device": self._input_device,
            "transport": getattr(self._mic, "_input_transport", "auto"),
            "sample_rate": self._mic.sample_rate,
            "native_rate": getattr(self._mic, "_native_rate", None),
            "channels": getattr(self._mic, "_native_channels", None),
            "channel_select": getattr(self._mic, "_channel_select", None),
            "chunk_ms": getattr(self._mic, "_chunk_ms", None),
            "chunk_samples": self._mic.chunk_samples,
            "mic_open": self._mic.is_open,
            "noise_gate_peak": noise_gate_peak,
            "echo_gate_peak": self._echo_gate_peak,
            "last_peak": last_peak,
            "peak_max_10s": peak_max_10s,
            "peak_p50_10s": peak_p50_10s,
            "peak_p95_10s": peak_p95_10s,
            "last_rms": last_rms,
            "rms_p50_10s": rms_p50_10s,
            "rms_p95_10s": rms_p95_10s,
            "sample_count_10s": sample_count_10s,
            "last_observed_age_s": last_observed_age_s,
            "vad_state": vad_state,
            "gate_state": gate_state,
            "tts_active": tts_active,
            "cooldown_remaining_s": round(self._in_post_tts_input_cooldown(), 2),
            "asr_timeouts": asr_timeouts,
            "last_failure_reason": last_failure_reason,
            "gate_recommendation": gate_recommendation,
        }

    @staticmethod
    def _percentile(values: list[int] | list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return round(ordered[0], 2)
        index = (max(0.0, min(percentile, 100.0)) / 100.0) * (len(ordered) - 1)
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        frac = index - lo
        return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 2)

    # ------------------------------------------------------------------
    # Wake word detection
    # ------------------------------------------------------------------

    def _wait_for_wake_word_mic(
        self,
        mic_ctx: MicInput,
        *,
        barge_only: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Block until wake word is detected via KWS (MicInput API).

        In ``barge_only`` mode playback audio is first removed by AEC, recent
        near-end speech is required, and the wait ends with normal playback.

        Returns True when wake word is detected, False if listening stops.
        """
        logger.info(
            "Waiting for wake word%s...",
            " during playback" if barge_only else "",
        )
        sample_rate = mic_ctx.sample_rate
        detector = "kws"
        if barge_only and not (self.kws and self.kws.available):
            detector = "asr"
            detector_stream = self.asr.create_stream() if self.asr is not None else None
            logger.info("Barge-in keyword detector: local streaming ASR fallback")
        else:
            detector_stream = self.kws.create_stream() if barge_only else self.kws_stream
        if detector_stream is None:
            logger.warning("Wake-word stream unavailable")
            return False
        if detector == "kws" and (
            self.kws is None or self.kws.spotter is None
        ):
            logger.warning("Wake-word spotter unavailable")
            return False

        pending_keyword = ""
        pending_keyword_at = 0.0
        last_near_end_speech_at = 0.0
        if barge_only:
            mic_ctx.pre_roll.clear()
            self._barge_wake_preroll.clear()
            self._vad_ctrl.reset()

        while not self.stop_event.is_set() and not (
            cancel_event is not None and cancel_event.is_set()
        ):
            if barge_only and (
                self._barge_listener_stop.is_set() or not self.tts.is_active()
            ):
                return False
            samples = mic_ctx.read_chunk()
            if cancel_event is not None and cancel_event.is_set():
                return False
            samples_i16 = MicInput.to_int16(samples)
            peak = MicInput.get_peak(samples_i16) if len(samples_i16) else 0
            gate_state = "open"

            if barge_only:
                samples, samples_i16, peak, echo_gated = self._audio_proc.process(
                    samples,
                    tts_active=True,
                    speech_active=False,
                )
                if echo_gated:
                    gate_state = "echo"
                elif self._audio_proc.is_noise_gated(peak):
                    gate_state = "noise"

            self._record_input_observation(
                peak=peak,
                rms=self._rms_int16(samples_i16),
                vad_state="barge_wake_word" if barge_only else "wake_word",
                gate_state=gate_state,
            )

            if barge_only and gate_state != "open":
                continue

            if barge_only:
                vad_event = self._vad_ctrl.feed(samples_i16, peak, tts_active=False)
                if self._vad_ctrl.speech_active or vad_event in (
                    VADEvent.SPEECH_START,
                    VADEvent.SPEECH_CONTINUE,
                    VADEvent.SPEECH_END,
                ):
                    last_near_end_speech_at = time.monotonic()
                self._barge_wake_preroll.append(samples.copy())

            try:
                detector_stream.accept_waveform(sample_rate, samples)
                if detector == "kws":
                    if self.kws is None or self.kws.spotter is None:
                        return False
                    while self.kws.spotter.is_ready(detector_stream):
                        self.kws.spotter.decode_stream(detector_stream)
                    result = self.kws.spotter.get_result(detector_stream)
                else:
                    if self.asr is None:
                        return False
                    while self.asr.is_ready(detector_stream):
                        self.asr.decode_stream(detector_stream)
                    result = self.asr.get_result(detector_stream)
            except Exception as e:
                logger.error("Wake-word detector error: %s", e)
                if detector == "kws" and not barge_only:
                    self.kws_stream = None
                    self._refresh_voice_metrics()
                return False

            if result:
                candidate = result.strip()
                compact_candidate = "".join(candidate.split())
                matched_term = next(
                    (
                        term
                        for term in self._barge_wake_terms
                        if term in compact_candidate
                    ),
                    "",
                )
                if not barge_only or matched_term:
                    pending_keyword = matched_term or candidate
                    pending_keyword_at = time.monotonic()

            now = time.monotonic()
            near_end_confirmed = (
                not barge_only
                or (
                    last_near_end_speech_at > 0
                    and now - last_near_end_speech_at <= 0.6
                )
            )
            if pending_keyword and near_end_confirmed:
                logger.info("Wake word detected: %s", pending_keyword)
                self.woken_up = True
                if not barge_only and self.kws is not None:
                    try:
                        self.kws_stream = self.kws.create_stream()
                    except Exception:
                        self.kws_stream = None
                        logger.exception(
                            "Failed to renew KWS stream; next turn will be "
                            "restricted to local safety commands"
                        )
                self._refresh_voice_metrics()
                return True

            if (
                barge_only
                and pending_keyword
                and now - pending_keyword_at > 0.6
            ):
                logger.info(
                    "Ignoring wake-word candidate without near-end speech: %s",
                    pending_keyword,
                )
                pending_keyword = ""
                pending_keyword_at = 0.0
                detector_stream = (
                    self.kws.create_stream()
                    if detector == "kws"
                    else self.asr.create_stream()
                )
                if detector_stream is None:
                    return False

        return False

    # ------------------------------------------------------------------
    # Audio feedback — chime synthesis
    # ------------------------------------------------------------------

    _SR = 44100

    def _play_chime(
        self,
        event: str,
        *,
        audio: np.ndarray | None = None,
        sample_rate: int | None = None,
        expected_processing_generation: int | None = None,
    ) -> None:
        """Synthesize and play a short chime for the given event.

        Supported events: ``acknowledge``, ``wake``, ``error``.

        On Linux with aplay available, chimes are piped to aplay in a
        background thread.  This avoids ALSA half-duplex conflicts that
        occur when sd.play() is called while sd.InputStream is open (wake
        word + acknowledge chimes both fire inside listen_loop).

        Reliability: normalizes volume to -12 dBFS, retries once on aplay
        failure, and logs success/failure for diagnostics.
        """
        feedback_generation: int | None = None
        expected_processing = expected_processing_generation

        try:
            now = time.monotonic()
            with self._chime_lock:
                if expected_processing is not None and (
                    self._processing_feedback_generation != expected_processing
                    or not self._processing_feedback_armed
                ):
                    return
                if (
                    event in {"thinking", "waiting_prompt"}
                    and self._semantic_tts_busy_for_feedback()
                ):
                    self._processing_feedback_suppressed_total = (
                        int(getattr(self, "_processing_feedback_suppressed_total", 0)) + 1
                    )
                    self._processing_feedback_overlap_prevented_total = (
                        int(
                            getattr(
                                self,
                                "_processing_feedback_overlap_prevented_total",
                                0,
                            )
                        )
                        + 1
                    )
                    self._processing_feedback_last_transition = "semantic_overlap_prevented"
                    return
                if event == "thinking" and now - self._last_thinking_chime_at < 2.0:
                    self._processing_feedback_suppressed_total = (
                        int(getattr(self, "_processing_feedback_suppressed_total", 0)) + 1
                    )
                    self._processing_feedback_last_transition = "rate_limited"
                    logger.debug("chime '%s' skipped due to recent feedback", event)
                    return
                # A wake/ACK cue must not remain alive when a later cue takes over.
                self.cancel_feedback()
                if expected_processing is not None and (
                    self._processing_feedback_generation != expected_processing
                    or not self._processing_feedback_armed
                ):
                    return
                if (
                    event in {"thinking", "waiting_prompt"}
                    and self._semantic_tts_busy_for_feedback()
                ):
                    self._processing_feedback_suppressed_total = (
                        int(getattr(self, "_processing_feedback_suppressed_total", 0)) + 1
                    )
                    self._processing_feedback_overlap_prevented_total = (
                        int(
                            getattr(
                                self,
                                "_processing_feedback_overlap_prevented_total",
                                0,
                            )
                        )
                        + 1
                    )
                    self._processing_feedback_last_transition = "semantic_overlap_prevented"
                    return
                self._last_chime_at = now
                if event == "thinking":
                    self._last_thinking_chime_at = now
                self._feedback_generation += 1
                feedback_generation = self._feedback_generation
                self._feedback_active = True
                self._feedback_event = event
                self._feedback_process = None
                self._feedback_sounddevice_active = False
                self._feedback_sounddevice_cancel_event = None
                if event in {"thinking", "waiting_prompt"}:
                    self._processing_feedback_started_total = (
                        int(getattr(self, "_processing_feedback_started_total", 0)) + 1
                    )
                    self._processing_feedback_last_transition = "started"

            feedback_sample_rate = int(sample_rate or self._SR)
            if audio is None:
                generators = {
                    "acknowledge": self._chime_acknowledge,
                    "wake": self._chime_wake,
                    "error": self._chime_error,
                    "thinking": self._chime_thinking,
                    "ready": self._chime_ready,
                }
                gen = generators.get(event, self._chime_acknowledge)
                audio = gen()
            # Normalize to -12 dBFS for consistent audibility
            peak_val = float(np.max(np.abs(audio)))
            if peak_val > 0:
                target_peak = 0.25  # ~-12 dBFS
                audio = audio * (target_peak / peak_val)

            aplay_bin = getattr(self.tts, "_aplay_bin", None)
            output_device = getattr(self.tts, "_output_device", None)

            def _feedback_is_current() -> bool:
                with self._chime_lock:
                    return bool(
                        self._feedback_active and self._feedback_generation == feedback_generation
                    )

            def _run() -> None:
                def _report_transport_failure(
                    reason: str,
                    exc: BaseException | None = None,
                ) -> None:
                    reporter = getattr(
                        self.tts,
                        "report_render_transport_failure",
                        None,
                    )
                    if callable(reporter):
                        reporter(reason, exc)

                def _publish_render_reference() -> None:
                    publisher = getattr(
                        self.tts,
                        "publish_feedback_render_reference",
                        None,
                    )
                    if callable(publisher):
                        publisher(audio, feedback_sample_rate, render_at=time.monotonic())

                try:
                    with self._chime_lock:
                        if not _feedback_is_current() or self._semantic_tts_busy_for_feedback():
                            return
                        if self.tts.play_feedback_audio(audio, feedback_sample_rate):
                            logger.debug("chime '%s' played via TTS feedback path", event)
                            return
                except Exception as exc:
                    logger.debug("chime '%s' feedback path failed: %s", event, exc)

                route = (
                    self._audio_router.output_session()
                    if self._audio_router is not None
                    else nullcontext()
                )
                with route:
                    if aplay_bin:
                        pcm = (audio * 32767).clip(-32768, 32767).astype("int16")
                        pcm_bytes = pcm.tobytes()
                        chime_cmd = [
                            aplay_bin,
                            "-r",
                            str(feedback_sample_rate),
                            "-f",
                            "S16_LE",
                            "-c",
                            "1",
                            "-q",
                        ]
                        if output_device is not None:
                            chime_cmd += ["-D", str(output_device)]

                        import subprocess

                        for attempt in range(2):
                            if not _feedback_is_current():
                                return
                            try:
                                proc = subprocess.Popen(
                                    chime_cmd,
                                    stdin=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                )
                                if proc.stdin is None:
                                    raise BrokenPipeError("chime aplay stdin unavailable")
                                first_bytes = min(
                                    len(pcm_bytes),
                                    max(2, feedback_sample_rate // 100 * 2),
                                )
                                with self._chime_lock:
                                    if (
                                        self._feedback_generation != feedback_generation
                                        or not self._feedback_active
                                        or self._semantic_tts_busy_for_feedback()
                                    ):
                                        proc.terminate()
                                        return
                                    self._feedback_process = proc
                                    proc.stdin.write(pcm_bytes[:first_bytes])
                                    proc.stdin.flush()
                                _publish_render_reference()
                                _, stderr = proc.communicate(
                                    input=pcm_bytes[first_bytes:],
                                    timeout=3,
                                )
                                if proc.returncode == 0:
                                    logger.debug("chime '%s' played OK", event)
                                    return
                                logger.warning(
                                    "chime '%s' aplay exit %d (attempt %d): %s",
                                    event,
                                    proc.returncode,
                                    attempt + 1,
                                    stderr.decode(errors="replace").strip()[:100],
                                )
                                _report_transport_failure(f"feedback_aplay_exit_{proc.returncode}")
                            except subprocess.TimeoutExpired:
                                logger.warning(
                                    "chime '%s' timed out (attempt %d)",
                                    event,
                                    attempt + 1,
                                )
                                _report_transport_failure("feedback_aplay_timeout")
                                try:
                                    proc.kill()
                                except Exception:
                                    logger.exception("[Audio] Chime proc kill failed")
                            except Exception as _e:
                                if not _feedback_is_current():
                                    return
                                logger.warning(
                                    "chime '%s' failed (attempt %d): %s",
                                    event,
                                    attempt + 1,
                                    _e,
                                )
                                _report_transport_failure("feedback_aplay", _e)
                            if attempt == 0:
                                time.sleep(0.05)  # brief pause before retry
                        return

                    stream = None
                    sounddevice_cancel_event = threading.Event()
                    sounddevice_finished = threading.Event()
                    playback_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    playback_offset = 0

                    def _sounddevice_callback(
                        outdata: np.ndarray,
                        frames: int,
                        _time_info: Any,
                        _status: Any,
                    ) -> None:
                        nonlocal playback_offset
                        outdata.fill(0)
                        if sounddevice_cancel_event.is_set():
                            raise sd.CallbackAbort
                        end = min(playback_offset + frames, len(playback_audio))
                        copied = end - playback_offset
                        if copied > 0:
                            outdata[:copied, 0] = playback_audio[playback_offset:end]
                            playback_offset = end
                        if playback_offset >= len(playback_audio):
                            raise sd.CallbackStop

                    try:
                        stream = sd.OutputStream(
                            samplerate=feedback_sample_rate,
                            channels=1,
                            dtype="float32",
                            callback=_sounddevice_callback,
                            finished_callback=sounddevice_finished.set,
                        )
                        with self._chime_lock:
                            if (
                                self._feedback_generation != feedback_generation
                                or not self._feedback_active
                                or self._semantic_tts_busy_for_feedback()
                            ):
                                sounddevice_cancel_event.set()
                                return
                            self._feedback_sounddevice_active = True
                            self._feedback_sounddevice_cancel_event = sounddevice_cancel_event
                            stream.start()
                        _publish_render_reference()
                        max_playback_s = max(
                            0.25,
                            len(playback_audio) / max(1, feedback_sample_rate) + 0.75,
                        )
                        if not sounddevice_finished.wait(timeout=max_playback_s):
                            sounddevice_cancel_event.set()
                            logger.warning(
                                "chime '%s' sounddevice stream timed out",
                                event,
                            )
                            stream.abort(ignore_errors=True)
                        logger.debug("chime '%s' played via sounddevice", event)
                    except Exception as exc:
                        _report_transport_failure("feedback_sounddevice", exc)
                        raise
                    finally:
                        sounddevice_cancel_event.set()
                        if stream is not None:
                            stream.close(ignore_errors=True)

            def _run_and_clear() -> None:
                try:
                    _run()
                except Exception as exc:
                    # Background feedback must never surface as an unhandled
                    # daemon-thread exception during shutdown or device loss.
                    logger.warning("chime '%s' playback failed: %s", event, exc)
                finally:
                    with self._chime_lock:
                        if self._feedback_generation == feedback_generation:
                            self._feedback_active = False
                            self._feedback_event = None
                            self._feedback_process = None
                            self._feedback_sounddevice_active = False
                            self._feedback_sounddevice_cancel_event = None
                            if event in {"thinking", "waiting_prompt"}:
                                self._processing_feedback_last_transition = "completed"

            threading.Thread(target=_run_and_clear, daemon=True).start()
        except Exception as _e:
            if feedback_generation is not None:
                with self._chime_lock:
                    if self._feedback_generation == feedback_generation:
                        self._feedback_active = False
                        self._feedback_event = None
                        self._feedback_process = None
                        self._feedback_sounddevice_active = False
                        self._feedback_sounddevice_cancel_event = None
                        if event in {"thinking", "waiting_prompt"}:
                            self._processing_feedback_last_transition = "failed"
            logger.warning("chime '%s' synthesis failed: %s", event, _e)

    # -- Individual chime generators --

    def _chime_acknowledge(self) -> np.ndarray:
        """Two-note ascending major third — quick, warm, like iOS 'received'."""
        sr = self._SR
        notes = [880, 1108.73]  # A5 -> C#6 (major third)
        note_dur = 0.06
        gap = 0.015
        total = len(notes) * note_dur + (len(notes) - 1) * gap
        audio: np.ndarray = np.zeros(int(sr * total), dtype=np.float32)

        offset = 0
        for freq in notes:
            n = int(sr * note_dur)
            t = np.linspace(0, note_dur, n, endpoint=False, dtype=np.float32)
            # Bell-like: fundamental + inharmonic partials (2.76x, 5.40x)
            tone = (
                0.30 * np.sin(2 * np.pi * freq * t)
                + 0.12 * np.sin(2 * np.pi * freq * 2.76 * t)
                + 0.05 * np.sin(2 * np.pi * freq * 5.40 * t)
            )
            tone *= np.exp(-t * 25)  # fast decay
            audio[offset : offset + n] += tone
            offset += n + int(sr * gap)

        return audio

    def _chime_ready(self) -> np.ndarray:
        """Single soft tone used after playback when the mic is ready again."""
        sr = self._SR
        note_dur = 0.09
        n = int(sr * note_dur)
        t = np.linspace(0, note_dur, n, endpoint=False, dtype=np.float32)
        tone = 0.22 * np.sin(2 * np.pi * 659.25 * t)
        tone *= np.exp(-t * 18)
        return tone.astype(np.float32)

    def _chime_wake(self) -> np.ndarray:
        """Three-note ascending pentatonic arpeggio — bright, alert."""
        sr = self._SR
        # C6 -> E6 -> G6 (major triad, bright register)
        notes = [1046.50, 1318.51, 1567.98]
        note_dur = 0.055
        gap = 0.01
        total = len(notes) * note_dur + (len(notes) - 1) * gap + 0.15
        audio: np.ndarray = np.zeros(int(sr * total), dtype=np.float32)

        offset = 0
        for i, freq in enumerate(notes):
            n = int(sr * note_dur)
            t = np.linspace(0, note_dur, n, endpoint=False, dtype=np.float32)
            # Metallic bell partials (tubular bell ratios ~ 2:3:4.2)
            tone = (
                0.28 * np.sin(2 * np.pi * freq * t)
                + 0.14 * np.sin(2 * np.pi * freq * 1.5 * t)
                + 0.07 * np.sin(2 * np.pi * freq * 2.1 * t)
            )
            # Each note slightly louder for rising energy
            tone *= (0.7 + 0.15 * i) * np.exp(-t * 20)
            audio[offset : offset + n] += tone
            offset += n + int(sr * gap)

        return audio

    def _chime_thinking(self) -> np.ndarray:
        """Mid-frequency thinking tone — 'processing your request'.

        500ms tone at ~900Hz with gentle vibrato. Optimized for factory
        audibility: 900Hz sits above typical industrial noise (50-500Hz)
        and well within robot speaker range (120Hz-8kHz).
        Designed to play instantly via chime path (no TTS network).
        """
        sr = self._SR
        dur = 0.50  # 500ms — long enough to perceive in noise
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
        # 900Hz base with gentle vibrato for organic feel
        freq = 900.0 + 30.0 * np.sin(2 * np.pi * 3.0 * t)
        phase = np.cumsum(2 * np.pi * freq / sr).astype(np.float32)
        # Fundamental + harmonics for better noise penetration
        tone = 0.30 * np.sin(phase)
        tone += 0.12 * np.sin(phase * 2.0)  # 1800Hz harmonic
        tone += 0.05 * np.sin(phase * 3.0)  # 2700Hz harmonic
        # Smooth fade-in (30ms) / fade-out (80ms)
        fade_in = int(sr * 0.03)
        fade_out = int(sr * 0.08)
        tone[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        tone[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
        return tone

    def _chime_error(self) -> np.ndarray:
        """Descending minor second — gentle 'something went wrong'."""
        sr = self._SR
        notes = [523.25, 493.88]  # C5 -> B4 (descending semitone)
        note_dur = 0.08
        gap = 0.02
        total = len(notes) * note_dur + gap + 0.1
        audio: np.ndarray = np.zeros(int(sr * total), dtype=np.float32)

        offset = 0
        for freq in notes:
            n = int(sr * note_dur)
            t = np.linspace(0, note_dur, n, endpoint=False, dtype=np.float32)
            tone = 0.25 * np.sin(2 * np.pi * freq * t)
            tone *= np.exp(-t * 12)
            audio[offset : offset + n] += tone
            offset += n + int(sr * gap)

        return audio

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal all background threads to stop."""
        self._barge_in_callback = None
        interruption_recovery = getattr(self, "_interruption_recovery", None)
        if interruption_recovery is not None:
            interruption_recovery.close()
        self._capture_processor = None
        self._capture_processor_failure_callback = None
        # AudioAgent owns fallback aplay/sounddevice chimes, so TTS shutdown
        # alone cannot stop them.  Invalidate feedback before tearing down the
        # provider/output engine.
        self.cancel_processing_feedback()
        self.stop_event.set()
        self._stop_realtime_recovery("shutdown")
        self._input_requested = False
        if not self.stop_listening():
            logger.warning("Microphone listener did not stop before shutdown timeout")
        stop_input = getattr(self._mic, "stop", None)
        if callable(stop_input):
            stop_input()
        coordinator = self._realtime_coordinator
        self._realtime_coordinator = None
        if coordinator is not None:
            try:
                coordinator.close("shutdown")
            except Exception as exc:
                logger.debug("Realtime dialogue shutdown failed: %s", exc)
        with self._runtime_switch_lock:
            tts = self.tts
            self._pending_asr_config = None
            self._pending_tts_config = None
            self._tts_activation_callback = None
            self._runtime_switch_callback = None
        tts.shutdown()
        self._refresh_voice_metrics(
            input_ready=False,
            output_ready=False,
            pipeline_ok=False,
            tts_busy=False,
        )

    def status_snapshot(self) -> dict[str, Any]:
        """Return a compact voice-pipeline health snapshot for telemetry."""
        return self._refresh_voice_metrics()

    def _refresh_voice_metrics(self, **overrides: Any) -> dict[str, Any]:
        tts_snapshot = self._component_status_snapshot(self.tts)
        output_ready = self._tts_output_ready(tts_snapshot)
        barge_in_status = dict(self._barge_in_policy)
        barge_in_status["requirements"] = dict(
            self._barge_in_policy["requirements"]
        )
        kws_available = bool(self.kws and getattr(self.kws, "available", False))
        wake_word_ready = self.wake_word_ready
        safety_only = self.kws_unavailable_safety_only
        if not self.voice_mode:
            input_policy = "text"
        elif safety_only:
            input_policy = "kws_unavailable_safety_only"
        elif wake_word_ready:
            input_policy = "wake_word"
        else:
            input_policy = "always_awake_lab"
        voice_turn_snapshot = self._turn_traces.snapshot()
        voice_turn_counters = voice_turn_snapshot.get("counters")
        if isinstance(voice_turn_counters, dict):
            with self._output_trace_lock:
                voice_turn_counters.update(
                    {
                        "orphan_output_event_count": (self._orphan_output_trace_event_count),
                        "playback_owner_conflict_count": (self._playback_owner_conflict_count),
                        "stale_playback_stop_count": (self._stale_playback_stop_count),
                        "playback_owner_active": (self._active_output_trace_token is not None),
                    }
                )
        snapshot = {
            "run_id": self._run_id,
            "mode": "voice" if self.voice_mode else "text",
            "enabled": self.voice_mode,
            "input_ready": bool(self.voice_mode and self.asr is not None and self.vad is not None),
            "output_ready": output_ready,
            "pipeline_ok": output_ready
            and (not self.voice_mode or (self.asr is not None and self.vad is not None)),
            "asr_available": self.asr is not None,
            "vad_available": self.vad is not None,
            "kws_available": kws_available,
            "kws_stream_ready": wake_word_ready,
            "kws_required": self._require_wake_word,
            "kws_unavailable_safety_only": safety_only,
            "input_policy": input_policy,
            "wake_word_enabled": bool(self.voice_mode and wake_word_ready),
            "woken_up": self.woken_up,
            "last_turn_wake_authorized": self.last_turn_wake_authorized,
            "last_turn_wake_source": self.last_turn_wake_source,
            "wake_timeout_s": self._wake_timeout,
            "wake_timeout_remaining_s": round(
                max(
                    0.0,
                    self._wake_timeout - (time.monotonic() - self._last_interaction_time),
                )
                if self._last_interaction_time > 0
                else 0.0,
                2,
            ),
            "muted": self._muted,
            "tts_backend": self.tts.backend,
            "tts_busy": self.is_busy,
            "agent_state": self._agent_state.value,
            "barge_in": barge_in_status,
            "interaction": self._interaction_status_snapshot(output_ready=output_ready),
            "media": self._media_status_snapshot(),
            "voice_turn": voice_turn_snapshot,
            "asr": self._component_status_snapshot(self._asr_mgr),
            "tts": tts_snapshot,
            "processing_feedback": self.processing_feedback_status_snapshot(),
            "interruption_recovery": self._interruption_recovery_status_snapshot(),
            "realtime": self._realtime_status_snapshot(),
            "pending_runtime_updates": {
                "asr": self._pending_asr_config is not None,
                "tts": self._pending_tts_config is not None,
            },
            "runtime_switches": self._runtime_switch_status_snapshot(),
            "input": self._input_status_snapshot(),
        }
        snapshot.update(overrides)
        self._metrics.update_voice_state(**snapshot)
        return snapshot

    def _realtime_status_snapshot(self) -> dict[str, Any]:
        coordinator = self._realtime_coordinator
        recovery = self._realtime_recovery_status_snapshot()
        if coordinator is None:
            return {
                "mode": self._realtime_mode,
                "active": False,
                "fallback": "cascade",
                "turn_boundary_recovery": recovery,
            }
        try:
            status = coordinator.status_snapshot()
        except Exception as exc:
            return {
                "mode": self._realtime_mode,
                "active": False,
                "fallback": "cascade",
                "last_error": type(exc).__name__,
                "turn_boundary_recovery": recovery,
            }
        snapshot = (
            dict(status)
            if isinstance(status, dict)
            else {
                "mode": self._realtime_mode,
                "active": False,
                "fallback": "cascade",
            }
        )
        snapshot["turn_boundary_recovery"] = recovery
        return snapshot

    def realtime_context_snapshot(self) -> dict[str, Any]:
        """Return provider/session correlation data for Conversation Core.

        This optional duck-typed surface deliberately stays outside the audio
        admission protocol: older frontends continue to work, while the active
        realtime adapter can correlate replaceable provider sessions with a
        stable product conversation thread.
        """

        snapshot = self._realtime_status_snapshot()
        with self._realtime_output_lock:
            tts_generation = self._realtime_output_tts_generation
            frozen_played_ms = int(getattr(self, "_realtime_last_physical_played_ms", 0) or 0)
        played_ms = max(
            frozen_played_ms,
            self._streaming_played_ms(tts_generation),
        )
        snapshot["physical_played_ms"] = played_ms
        return snapshot

    def _streaming_played_ms(self, generation: int | None) -> int:
        if generation is None:
            return 0
        played_query = getattr(self.tts, "streaming_pcm_played_ms", None)
        if not callable(played_query):
            return 0
        try:
            return max(0, int(played_query(generation)))
        except Exception as exc:
            logger.debug("Realtime physical playhead query failed: %s", exc)
            return 0

    def _realtime_recovery_status_snapshot(self) -> dict[str, Any]:
        with self._realtime_recovery_lock:
            worker = self._realtime_recovery_thread
            return {
                "running": bool(worker is not None and worker.is_alive()),
                "capture_armed": self._realtime_capture_armed,
                "faulted": self._realtime_faulted_coordinator is not None,
                "attempts": self._realtime_recovery_attempts,
                "successes": self._realtime_recovery_successes,
                "failures": self._realtime_recovery_failures,
                "last_error": self._realtime_recovery_last_error,
            }

    def _component_status_snapshot(self, component: Any) -> dict[str, Any]:
        status_snapshot = getattr(component, "status_snapshot", None)
        if callable(status_snapshot):
            result = status_snapshot()
            if isinstance(result, dict):
                return result
        return {"available": component is not None}

    def _tts_output_ready(
        self,
        snapshot: dict[str, Any] | None = None,
    ) -> bool:
        """Use the TTS physical sink contract instead of object existence."""

        if self.tts is None:
            return False
        current = snapshot
        if current is None:
            current = self._component_status_snapshot(self.tts)
        if "output_ready" in current:
            return bool(current["output_ready"])
        return True

    def _interaction_status_snapshot(
        self,
        *,
        output_ready: bool | None = None,
    ) -> dict[str, Any]:
        cooldown_remaining = round(self._in_post_tts_input_cooldown(), 2)
        input_ready = bool(self.voice_mode and self.asr is not None and self.vad is not None)
        output_ready = self.tts is not None
        if self._muted:
            state = "muted"
            can_talk = False
            hint = "mic_muted"
        elif not self.voice_mode:
            state = "text_mode"
            can_talk = False
            hint = "use_text_input"
        elif not input_ready or not output_ready:
            state = "not_ready"
            can_talk = False
            hint = "voice_pipeline_not_ready"
        elif self._agent_state == AgentState.SPEAKING or self.is_busy:
            state = "speaking"
            can_talk = getattr(self, "full_duplex_enabled", False) is True
            hint = "barge_in_allowed" if can_talk else "wait_for_playback"
        elif cooldown_remaining > 0:
            state = "cooldown"
            can_talk = False
            hint = "wait_for_ready_cue"
        elif self._agent_state == AgentState.LISTENING:
            state = "listening"
            can_talk = True
            hint = "keep_speaking"
        elif self._agent_state == AgentState.PROCESSING:
            state = "processing"
            can_talk = False
            hint = "thinking"
        else:
            state = "ready_to_talk"
            can_talk = True
            hint = "speak_now"
        return {
            "state": state,
            "can_talk": can_talk,
            "hint": hint,
            "ready_chime_enabled": self._ready_chime_enabled,
            "cooldown_remaining_s": cooldown_remaining,
        }

    def _media_status_snapshot(self) -> dict[str, Any]:
        full_duplex = getattr(self, "full_duplex_status", None)
        if not isinstance(full_duplex, dict):
            full_duplex = {
                "enabled": getattr(self, "full_duplex_enabled", False) is True,
                "reason": "not_configured",
                "echo_control": "none",
                "aec_backend": "unavailable",
            }
        return {
            "media_transport": self._media_transport,
            "room_id": "",
            "participant_count": 1 if self.voice_mode else 0,
            "packet_loss": None,
            "jitter_ms": None,
            "input_transport": getattr(self._mic, "_input_transport", "auto"),
            "output_transport": getattr(self.tts, "_output_transport", "auto"),
            "session_id": self._run_id,
            "full_duplex": dict(full_duplex),
        }

    def _asr_provider_label(self) -> str:
        cloud = getattr(self._asr_mgr, "_cloud", None)
        if getattr(cloud, "available", False) is True:
            return "cloud+local"
        return "local"

    @staticmethod
    def _resolve_media_transport_label(voice_cfg: dict[str, Any]) -> str:
        transport = str(voice_cfg.get("input_transport", "sounddevice")).lower()
        if transport in {"auto", "sounddevice"}:
            return "local_sounddevice"
        return f"local_{transport}"
