"""AudioAgent - high-level voice I/O controller composing ASR, VAD, KWS, and TTS engines."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import numpy as np

try:
    import sounddevice as sd
except ModuleNotFoundError:
    # Minimal stub so tests can patch sd.play / sd.InputStream without needing hardware
    class _SoundDeviceStub:
        InputStream = None
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

from askme.robot_interaction.routing_policy import DEFAULT_QUICK_REPLIES
from askme.telemetry.ota_bridge import OTABridgeMetrics, get_ota_runtime_metrics
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
from askme.voice.output.audio_router import AudioRouter
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
    IDLE = "idle"              # Waiting for wake word / user input
    LISTENING = "listening"    # VAD active, collecting speech (speech_active=True)
    PROCESSING = "processing"  # ASR done, text returned, LLM/skill running
    SPEAKING = "speaking"      # TTS is playing back audio
    MUTED = "muted"            # Microphone muted by user


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
    ) -> None:
        voice_cfg = config.get("voice", {})
        self.voice_mode = voice_mode
        self._metrics = metrics or get_ota_runtime_metrics()
        self._audio_router = audio_router

        # Shared state
        self.audio_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.woken_up: bool = False
        self.last_turn_wake_authorized: bool = False
        self.last_turn_wake_source: str = "none"
        self._muted: bool = False  # software mute — still listens, VoiceLoop filters results
        self._agent_state: AgentState = AgentState.IDLE
        # When True, confirmation words bypass the noise filter.
        self.awaiting_confirmation: bool = False
        self._chime_lock = threading.Lock()
        self._last_chime_at: float = 0.0
        self._runtime_switch_lock = threading.RLock()
        self._listen_loop_active = False
        self._asr_phase_active = False
        self._pending_asr_config: dict[str, Any] | None = None
        self._pending_tts_config: dict[str, Any] | None = None

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
        self._turn_traces = VoiceTurnTraceRecorder()
        fast_path_cfg = voice_cfg.get("fast_path", {}) or {}
        self._fast_endpoint = FastEndpointController(
            quick_replies=DEFAULT_QUICK_REPLIES,
            enabled=bool(fast_path_cfg.get("enabled", False)),
            candidate_silence_ms=float(
                fast_path_cfg.get("candidate_silence_ms", 300.0)
            ),
            stable_partial_ms=float(fast_path_cfg.get("stable_partial_ms", 160.0)),
        )

        # A second capture loop is used only while TTS/LLM output is active.
        # Its configured policy authorizes when it may take ownership of the turn.
        self._barge_listener_stop = threading.Event()
        self._barge_listener_lock = threading.Lock()
        self._barge_listener_thread: threading.Thread | None = None
        self._barge_listener_results: queue.Queue[str] = queue.Queue()
        self._barge_in_active = threading.Event()
        self._barge_in_callback: Callable[[], None] | None = None
        self._barge_in_callback_lock = threading.Lock()
        self._barge_in_notified = threading.Event()
        self._barge_in_requested_mode = (
            str(voice_cfg.get("barge_in_mode", "keyword")).strip().lower()
        )
        self._barge_in_mode = "keyword"
        self._barge_in_warning_emitted = False
        barge_pre_roll_s = max(
            0.5,
            float(voice_cfg.get("barge_in_pre_roll_s", 1.2)),
        )
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

        # -- Input engines (only in voice mode) --
        self._asr_timeout: float = voice_cfg.get("asr", {}).get(
            "asr_timeout", _DEFAULT_ASR_TIMEOUT
        )

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
        self._noise_gate_peak: int = (
            0 if str(_raw_gate).lower() == "auto" else int(_raw_gate)
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
        self._media_transport = self._resolve_media_transport_label(voice_cfg)
        self._audio_proc = AudioProcessor(voice_cfg)
        self._configure_barge_in_policy(voice_cfg)
        # Keep text-mode construction lightweight and safe on machines with
        # native audio/VAD libraries installed: sherpa-onnx VAD initialization
        # can abort the interpreter on some Windows setups. Only voice mode
        # needs a live VAD controller.
        self._vad_ctrl = VADController(voice_cfg) if voice_mode else None
        self._asr_mgr = ASRManager(voice_cfg) if voice_mode else None

        if voice_mode:
            assert self._vad_ctrl is not None
            assert self._asr_mgr is not None
            # Share engines from modules — avoid constructing duplicates
            self.asr = self._asr_mgr._asr
            self.vad = self._vad_ctrl._vad
            self.asr_stream = self._asr_mgr._stream
            self.punct = self._asr_mgr._punct
            self.kws = KWSEngine(voice_cfg.get("kws", {}))

            if self.kws.available:
                self.kws_stream = self.kws.create_stream()
            else:
                self.kws_stream = None
                self.woken_up = True  # fallback: always awake when no KWS
        else:
            self.asr = None  # type: ignore[assignment]
            self.vad = None  # type: ignore[assignment]
            self.kws = None  # type: ignore[assignment]
            self.punct = None
            self.asr_stream = None
            self.kws_stream = None
            self.woken_up = True

        # -- Output engine --
        # Feed the exact TTS PCM into the microphone-side AEC. Injecting the
        # reference after construction preserves compatibility with TTS test
        # doubles and alternate output backends.
        self.tts = TTSEngine(voice_cfg.get("tts", {}), audio_router=audio_router)
        if hasattr(self.tts, "_echo_reference"):
            self.tts._echo_reference = self._audio_proc.echo_reference
        logger.info("AudioAgent run_id=%s mode=%s", self._run_id, "voice" if voice_mode else "text")
        self._refresh_voice_metrics()

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to TTS)
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Whether TTS is actively playing or has queued text."""
        return self.tts.is_active() or not self.tts.tts_text_queue.empty()

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
                (
                    "field_acceptance_verified",
                    "field_acceptance_not_verified",
                ),
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
            if not self._barge_in_warning_emitted:
                logger.warning("Unsupported barge-in mode; using keyword (%s)", reason)
                self._barge_in_warning_emitted = True

        self._barge_in_mode = effective_mode
        self._barge_in_policy = {
            "requested_mode": requested_mode,
            "effective_mode": effective_mode,
            "speech_gate_ready": speech_gate_ready,
            "reason": reason,
            "requirements": requirements,
        }

    def speak(self, text: str) -> None:
        """Queue text for TTS (strips emoji/markdown internally)."""
        self.tts.speak(text)
        self._refresh_voice_metrics()

    def start_playback(self) -> None:
        with self._barge_in_callback_lock:
            self._barge_in_notified.clear()
        self._ready_chime_generation += 1
        self._agent_state = AgentState.SPEAKING
        self._turn_traces.mark("tts_playback_started")
        self.tts.start_playback()
        self._start_barge_listener()
        self._refresh_voice_metrics()

    def set_barge_in_callback(
        self,
        callback: Callable[[], None] | None,
    ) -> None:
        """Set the fast, synchronous callback invoked on confirmed barge-in."""
        with self._barge_in_callback_lock:
            self._barge_in_callback = callback

    def _confirm_barge_in(self) -> bool:
        """Cancel generation, advance audio buffers, then stop physical output once."""
        with self._barge_in_callback_lock:
            if self._barge_in_notified.is_set():
                return False
            self._barge_in_notified.set()
            callback = self._barge_in_callback

        if callback is not None:
            try:
                callback()
            except Exception:
                logger.exception("Barge-in cancellation callback failed")

        try:
            self.tts.drain_buffers()
        finally:
            self.tts.stop_immediately()
        return True

    def stop_playback(self) -> None:
        self.tts.stop_playback()
        barge_capture_active = self._barge_in_active.is_set()
        self._stop_barge_listener()
        self._turn_traces.mark("playback_done")

        # A confirmed barge-in owns VAD/ASR until its utterance ends.
        # Cooldown resets those engines, so applying it here would discard the
        # first words after "小算" and let the main loop race for the mic.
        if barge_capture_active:
            logger.info("Playback stopped for confirmed barge-in; ASR capture continues")
            self._input_cooldown_until = 0.0
            self._agent_state = AgentState.LISTENING
            self._refresh_voice_metrics()
            return

        self._begin_post_tts_input_cooldown()
        self._agent_state = AgentState.IDLE
        self.mark_interaction_turn()
        self._refresh_voice_metrics()
        self._schedule_ready_chime()

    def _start_barge_listener(self) -> None:
        """Listen concurrently while TTS is active so speech can interrupt it."""
        if not self.voice_mode or self._muted:
            return
        with self._barge_listener_lock:
            thread = self._barge_listener_thread
            if thread is not None and thread.is_alive():
                return
            self._barge_listener_stop.clear()
            self._barge_in_active.clear()
            self._barge_listener_thread = threading.Thread(
                target=self._barge_listener_worker,
                name="askme-barge-listener",
                daemon=True,
            )
            self._barge_listener_thread.start()

    def _stop_barge_listener(self) -> None:
        """Stop the background listener after normal playback completion."""
        # Once VAD has confirmed barge-in, its ASR session owns the mic until
        # the interrupted utterance reaches an endpoint. Do not cut it off.
        if self._barge_in_active.is_set():
            return
        self._barge_listener_stop.set()
        with self._barge_listener_lock:
            thread = self._barge_listener_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        with self._barge_listener_lock:
            if thread is self._barge_listener_thread and (thread is None or not thread.is_alive()):
                self._barge_listener_thread = None

    def _barge_listener_worker(self) -> None:
        """Run configured barge-in policy until playback ends or a new turn wins."""
        try:
            while not self._barge_listener_stop.is_set() and self.tts.is_active():
                result = self.listen_loop(_barge_mode=True)
                if result:
                    self._barge_listener_results.put(result)
                    return
                if not self.tts.is_active():
                    return
        except Exception as exc:
            logger.debug("Barge-in listener stopped: %s", exc)
        finally:
            with self._barge_listener_lock:
                if self._barge_listener_thread is threading.current_thread():
                    self._barge_listener_thread = None
            self._barge_in_active.clear()

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
        self.speak(text)
        self.start_playback()
        try:
            completed = await asyncio.to_thread(self.wait_speaking_done)
            if not completed:
                raise TimeoutError("Speech playback did not complete within 30 seconds.")
        finally:
            self.stop_playback()

    async def speak_cached_and_wait(self, text: str, *, cache_key: str) -> bool:
        """Play a persisted phrase without invoking a TTS provider."""

        if not self.tts.queue_cached_phrase(text, cache_key=cache_key):
            return False
        self.start_playback()
        try:
            completed = await asyncio.to_thread(self.wait_speaking_done)
            if not completed:
                raise TimeoutError(
                    "Cached speech playback did not complete within 30 seconds."
                )
            return True
        finally:
            self.stop_playback()

    def start_input(self) -> None:
        """Open the microphone input stream for long-lived voice sessions."""
        if not self.voice_mode:
            return
        self._mic.start()
        self._refresh_voice_metrics()

    def stop_input(self) -> None:
        """Close the microphone input stream for long-lived voice sessions."""
        if not self.voice_mode:
            return
        self._mic.stop()
        self._refresh_voice_metrics()

    def drain_buffers(self) -> None:
        """Clear any leftover TTS text/audio from a previous turn."""
        self.tts.drain_buffers()
        self._refresh_voice_metrics()

    def stop_immediately(self) -> None:
        """Immediately stop TTS playback mid-chunk (barge-in support)."""
        self.tts.stop_immediately()
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
        try:
            self._vad_ctrl.reset()
        except Exception as exc:
            logger.debug("VAD reset failed during input cooldown (ignored): %s", exc)
        try:
            self._asr_mgr.reset()
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
        with self._runtime_switch_lock:
            if self._asr_phase_active or bool(
                getattr(self._asr_mgr, "_recognition_active", False)
            ):
                self._pending_asr_config = clean
                return {
                    "updated": True,
                    "component": "asr",
                    "state": "pending",
                    "effective": "next_listen_cycle",
                }
        return self._apply_asr_config(clean)

    def reconfigure_tts(self, tts_config: dict[str, Any]) -> dict[str, Any]:
        """Replace TTS immediately when idle, or queue it after playback."""

        clean = dict(tts_config or {})
        with self._runtime_switch_lock:
            if self.is_busy or self.tts.is_active():
                self._pending_tts_config = clean
                return {
                    "updated": True,
                    "component": "tts",
                    "state": "pending",
                    "effective": "after_playback",
                }
        return self._apply_tts_config(clean)

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
        return {
            "updated": True,
            "component": "asr",
            "state": "active",
            "runtime": next_manager.status_snapshot(),
        }

    def _apply_tts_config(self, tts_config: dict[str, Any]) -> dict[str, Any]:
        next_tts = TTSEngine(tts_config, audio_router=self._audio_router)
        with self._runtime_switch_lock:
            previous = self.tts
            self.tts = next_tts
            self._pending_tts_config = None
        if previous is not None:
            previous.shutdown()
        self._refresh_voice_metrics()
        return {
            "updated": True,
            "component": "tts",
            "state": "active",
            "runtime": next_tts.status_snapshot(),
        }

    def _apply_pending_runtime_updates(self) -> None:
        with self._runtime_switch_lock:
            asr_config = self._pending_asr_config
            tts_config = self._pending_tts_config if not self.is_busy else None
        if asr_config is not None:
            try:
                self._apply_asr_config(asr_config)
            except Exception as exc:
                logger.warning("Pending ASR switch failed: %s", exc)
        if tts_config is not None:
            try:
                self._apply_tts_config(tts_config)
            except Exception as exc:
                logger.warning("Pending TTS switch failed: %s", exc)

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

    def acknowledge(self) -> None:
        """Play a brief confirmation tone: 'heard you, thinking'.

        Non-blocking. Fires immediately after ASR so the user has audio
        feedback during the LLM latency gap instead of dead silence.
        """
        self._play_chime("acknowledge")

    def play_thinking(self) -> None:
        """Play a brief thinking/processing tone — 'I'm working on it'.

        A gentle humming tone (350ms) that plays immediately via the chime
        path (direct PCM → aplay), bypassing the TTS network entirely.
        This eliminates the 700ms+ MiniMax TTS round-trip that made the
        old ``speak("嗯...")`` thinking indicator arrive slower than the
        actual LLM response.
        """
        self._play_chime("thinking")

    def speak_error(self) -> None:
        """Speak a short error notification to the user."""
        self._metrics.mark_voice_error("voice interaction error")
        self._play_chime("error")
        self.tts.speak("抱歉，出现了问题，请重试。")
        self._refresh_voice_metrics()

    def _start_voice_turn_trace(self) -> None:
        self._turn_traces.start(
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

    # ------------------------------------------------------------------
    # Microphone listen loop
    # ------------------------------------------------------------------

    def listen_loop(self, *, _barge_mode: bool = False) -> str | None:
        """Listen with VAD-gated ASR using modular pipeline.

        Flow: MicInput -> AudioProcessor -> VADController -> ASRManager -> text

        Returns recognized text, or None on timeout/stop.
        """
        if self.asr is None or self.vad is None:
            raise RuntimeError("listen_loop requires voice_mode=True")

        # A completed background barge-in is handed to the normal voice loop
        # on its next iteration, after the interrupted response has unwound.
        if not _barge_mode:
            try:
                return self._barge_listener_results.get_nowait()
            except queue.Empty:
                pass

            # The keyword was heard and the old reply has stopped, but the
            # background listener may still be transcribing the new command.
            # Wait for that handoff instead of opening a second mic reader.
            while self._barge_in_active.is_set() and not self.stop_event.is_set():
                try:
                    return self._barge_listener_results.get(timeout=0.1)
                except queue.Empty:
                    continue
            try:
                return self._barge_listener_results.get_nowait()
            except queue.Empty:
                pass

        self._start_voice_turn_trace()
        self.last_turn_wake_authorized = False
        self.last_turn_wake_source = "none"
        barge_keyword_authorized = False
        self._metrics.mark_voice_listen_started()
        self._refresh_voice_metrics()

        mic = self._mic
        proc = self._audio_proc
        vad = self._vad_ctrl

        with self._runtime_switch_lock:
            self._listen_loop_active = True
            self._asr_phase_active = False
        try:
            # Mic is persistently open (started by VoiceModule).
            # Flush stale audio that accumulated during LLM+TTS processing.
            mic._flush_queue()

            with mic.open() as mic_ctx:
                # Phase 1: Wake word detection (if KWS available)
                if _barge_mode and self._barge_in_mode == "keyword":
                    logger.info("Barge-in listener active; waiting for wake word")
                    self.woken_up = False
                    if not self._wait_for_wake_word_mic(mic_ctx, barge_only=True):
                        self._turn_traces.finish("barge_in_keyword_not_detected")
                        return None

                    # The keyword, not VAD alone, authorizes interruption. Keep
                    # this listener alive after TTS stops so it can finish ASR
                    # for the words that follow "小算".
                    barge_keyword_authorized = True
                    self._barge_in_active.set()
                    self.last_turn_wake_authorized = True
                    self.last_turn_wake_source = "barge_in_keyword"
                    self._turn_traces.mark_barge_in(keyword="小算")
                    self._agent_state = AgentState.LISTENING
                    self._refresh_voice_metrics()
                    self._confirm_barge_in()
                    self._play_chime("wake")
                elif _barge_mode:
                    logger.info("Barge-in listener active; waiting for confirmed speech")
                    self.woken_up = False
                elif self.kws and self.kws.available and self.kws_stream:
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
                        if not self._wait_for_wake_word_mic(mic_ctx):
                            self._turn_traces.finish("wake_word_not_detected")
                            return None
                        self.last_turn_wake_authorized = True
                        self.last_turn_wake_source = "keyword"
                        self._play_chime("wake")
                elif self._followup_window_active():
                    self.last_turn_wake_authorized = True
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
                logger.info("Listening for speech...")
                asr.preconnect_cloud()  # warm up WebSocket (fast, ~100ms)
                deadline = time.monotonic() + self._asr_timeout
                vad.reset()
                self._fast_endpoint.reset()
                asr_session_started = False

                if barge_keyword_authorized:
                    # Seed VAD with a longer AEC-cleaned history so continuous
                    # "小算带我去..." remains one utterance. Feed only the tail
                    # to ASR to preserve a tightly attached command without
                    # making the wake word itself dominate the transcript.
                    seed_buffers = list(self._barge_wake_preroll)
                    self._barge_wake_preroll.clear()
                    for buf in seed_buffers:
                        buf_i16 = MicInput.to_int16(buf)
                        buf_peak = MicInput.get_peak(buf_i16) if len(buf_i16) else 0
                        vad.feed(buf_i16, buf_peak, tts_active=False)
                    asr.start_session()
                    asr_session_started = True
                    for buf in seed_buffers[-2:]:
                        asr.feed_audio(
                            buf,
                            MicInput.to_int16(buf),
                            mic_ctx.sample_rate,
                        )

                _vol_log_interval = 0.5
                _vol_log_next = time.monotonic() + _vol_log_interval

                while (
                    not self.stop_event.is_set()
                    and not (_barge_mode and self._barge_listener_stop.is_set())
                ):
                    if self._agent_state not in (AgentState.MUTED, AgentState.LISTENING):
                        self._agent_state = AgentState.IDLE

                    if time.monotonic() > deadline:
                        logger.info("ASR timeout: no speech detected within %.0fs.", self._asr_timeout)
                        asr.reset()
                        self._mark_input_failure("asr_timeout")
                        self._turn_traces.finish("timeout")
                        self._refresh_voice_metrics()
                        return None

                    raw = mic_ctx.read_chunk()
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
                    result = proc.process(raw, tts_active=tts_active, speech_active=vad.speech_active)
                    samples_f32, samples_i16, peak, echo_gated = result
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
                        if not asr_session_started:
                            asr.start_session()
                            asr_session_started = True
                            for buf in mic_ctx.flush_pre_roll():
                                asr.feed_audio(
                                    buf,
                                    MicInput.to_int16(buf),
                                    mic_ctx.sample_rate,
                                )
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.SPEECH_CONTINUE:
                        deadline = time.monotonic() + self._asr_timeout
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.BARGE_IN_CONFIRMED:
                        if _barge_mode and self._barge_in_mode == "speech":
                            barge_keyword_authorized = True
                            self._barge_in_active.set()
                            self.last_turn_wake_authorized = True
                            self.last_turn_wake_source = "barge_in_speech"
                            self._turn_traces.mark_barge_in(mode="speech")
                            self._turn_traces.mark(
                                "vad_start", peak=peak, rms=rms, mode="barge_in_speech"
                            )
                            self._agent_state = AgentState.LISTENING
                            self._refresh_voice_metrics()
                            self._confirm_barge_in()

                            deadline = time.monotonic() + self._asr_timeout
                            if not asr_session_started:
                                asr.start_session()
                                asr_session_started = True
                            buffered_audio = list(vad.barge_in_buffer)
                            if not buffered_audio:
                                buffered_audio = [samples_i16.copy()]
                            for buf_i16 in buffered_audio:
                                asr.feed_audio(
                                    buf_i16.astype(np.float32) / 32768.0,
                                    buf_i16,
                                    mic_ctx.sample_rate,
                                )
                            vad.barge_in_buffer.clear()
                        else:
                            # VAD-only speech is ambient conversation until KWS has
                            # heard "小算". Never stop playback on this event.
                            logger.info(
                                "Ignoring VAD-only barge-in; wake word '小算' is required"
                            )
                            self._turn_traces.mark(
                                "barge_in_rejected", reason="wake_word_required"
                            )
                            asr.reset()
                            asr_session_started = False
                            vad.reset()
                            mic_ctx.pre_roll.clear()
                            deadline = time.monotonic() + self._asr_timeout

                    elif event == VADEvent.BARGE_IN_DISMISSED:
                        mic_ctx.buffer_pre_roll(raw)

                    elif event == VADEvent.SPEECH_END:
                        logger.info("VAD: speech end")
                        self._turn_traces.mark("vad_end", peak=peak, rms=rms)
                        cloud_result = asr.finish_and_get_result(self.awaiting_confirmation)
                        if cloud_result and not cloud_result.is_noise:
                            return self._accept_result(
                                cloud_result.text,
                                asr_source=cloud_result.source,
                                asr_latency_ms=cloud_result.latency_ms,
                            )
                        if cloud_result and cloud_result.is_noise:
                            logger.info("ASR noise filtered: '%s'", cloud_result.text)
                            self._mark_input_failure("asr_noise_filtered")
                            self._turn_traces.finish(
                                "noise_filtered",
                                asr_source=cloud_result.source,
                            )
                            asr.reset()
                            asr_session_started = False
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
                                forced_endpoint=True,
                            )
                        self._mark_input_failure("asr_forced_empty")
                        deadline = time.monotonic() + self._asr_timeout
                        self._turn_traces.finish("forced_empty")
                        self._start_voice_turn_trace()
                        continue

                    # Stable exact-match partials for safe fixed phrases may
                    # close during low-energy trailing audio. Physical actions
                    # never enter this path.
                    if (
                        event == VADEvent.SPEECH_CONTINUE
                        and self._fast_endpoint.enabled
                    ):
                        partial = asr.partial_result()
                        decision = self._fast_endpoint.observe(
                            partial_text=partial.text if partial is not None else "",
                            quiet=(
                                self._fast_quiet_peak_threshold > 0
                                and peak < self._fast_quiet_peak_threshold
                            ),
                            now=now,
                        )
                        if (
                            decision.action is FastEndpointAction.COMMIT
                            and partial is not None
                        ):
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
                            )
                        # Fall back to local ASR result
                        is_noise = self._asr_mgr.is_noise(
                            ep_result.text, self.awaiting_confirmation
                        )
                        if not is_noise:
                            return self._accept_result(
                                ep_result.text,
                                asr_source=ep_result.source,
                                asr_latency_ms=ep_result.latency_ms,
                            )
                        else:
                            logger.info("ASR noise filtered: '%s'", ep_result.text)
                            self._mark_input_failure("asr_noise_filtered")
                            self._turn_traces.finish(
                                "noise_filtered",
                                asr_source=ep_result.source,
                            )
                            asr.reset()
                            asr_session_started = False
                            vad.reset()
                            deadline = time.monotonic() + self._asr_timeout
                            self._start_voice_turn_trace()

        except Exception as exc:
            self._metrics.mark_voice_error(str(exc))
            self._mark_input_failure(str(exc))
            self._turn_traces.finish("error", error=str(exc))
            self._refresh_voice_metrics(pipeline_ok=False)
            raise
        finally:
            with self._runtime_switch_lock:
                self._listen_loop_active = False
                self._asr_phase_active = False
            self._apply_pending_runtime_updates()

        return None

    def _accept_result(
        self,
        text: str,
        *,
        asr_source: str = "",
        asr_latency_ms: float | None = None,
        forced_endpoint: bool = False,
    ) -> str:
        """Accept a recognized text result: log, queue, update state."""
        self._turn_traces.mark(
            "asr_final",
            asr_source=asr_source,
            asr_latency_ms=asr_latency_ms,
            forced_endpoint=forced_endpoint,
            text_chars=len(text),
        )
        self._turn_traces.finish(
            "accepted",
            asr_source=asr_source,
            text_chars=len(text),
        )
        logger.info("Recognized: %s", text)
        self.audio_queue.put(text)
        self._metrics.mark_voice_input(text)
        self._clear_input_failure()
        self._agent_state = AgentState.PROCESSING
        self._refresh_voice_metrics()
        self._asr_mgr.reset()
        return text

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
        values = samples_int16.astype(np.float64)
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
            self._input_level_window.append(
                (now, self._input_last_peak, self._input_last_rms)
            )

    def _mark_input_failure(self, reason: str) -> None:
        with self._input_state_lock:
            if reason == "asr_timeout":
                self._input_asr_timeouts += 1
            self._input_last_failure_reason = str(reason)
            self._input_vad_state = "timeout" if reason == "asr_timeout" else self._input_vad_state

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
            gate_recommendation = (
                f"observed_peak_below_noise_gate:{peak_max_10s}<{noise_gate_peak}"
            )

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
            detector_stream = (
                self.kws.create_stream() if barge_only else self.kws_stream
            )
        if detector_stream is None:
            logger.warning("Wake-word stream unavailable")
            return False

        pending_keyword = ""
        pending_keyword_at = 0.0
        last_near_end_speech_at = 0.0
        if barge_only:
            mic_ctx.pre_roll.clear()
            self._barge_wake_preroll.clear()
            self._vad_ctrl.reset()

        while not self.stop_event.is_set():
            if barge_only and (
                self._barge_listener_stop.is_set() or not self.tts.is_active()
            ):
                return False

            samples = mic_ctx.read_chunk()
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
                vad_event = self._vad_ctrl.feed(
                    samples_i16,
                    peak,
                    tts_active=False,
                )
                if self._vad_ctrl.speech_active or vad_event in (
                    VADEvent.SPEECH_START,
                    VADEvent.SPEECH_CONTINUE,
                    VADEvent.SPEECH_END,
                ):
                    last_near_end_speech_at = time.monotonic()
                # Preserve enough AEC-cleaned history for a command spoken
                # without a pause ("小算带我去...").
                self._barge_wake_preroll.append(samples.copy())

            try:
                detector_stream.accept_waveform(sample_rate, samples)
                if detector == "kws":
                    while self.kws.spotter.is_ready(detector_stream):
                        self.kws.spotter.decode_stream(detector_stream)
                    result = self.kws.spotter.get_result(detector_stream)
                else:
                    while self.asr.is_ready(detector_stream):
                        self.asr.decode_stream(detector_stream)
                    result = self.asr.get_result(detector_stream)
            except Exception as e:
                logger.error("Wake-word detector error: %s", e)
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
                if not barge_only:
                    self.kws_stream = self.kws.create_stream()
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

    def _play_chime(self, event: str) -> None:
        """Synthesize and play a short chime for the given event.

        Supported events: ``acknowledge``, ``wake``, ``error``.

        On Linux with aplay available, chimes are piped to aplay in a
        background thread.  This avoids ALSA half-duplex conflicts that
        occur when sd.play() is called while sd.InputStream is open (wake
        word + acknowledge chimes both fire inside listen_loop).

        Reliability: normalizes volume to -12 dBFS, retries once on aplay
        failure, and logs success/failure for diagnostics.
        """
        try:
            now = time.monotonic()
            with self._chime_lock:
                if event == "thinking" and now - self._last_chime_at < 2.0:
                    logger.debug("chime '%s' skipped due to recent feedback", event)
                    return
                self._last_chime_at = now

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

            def _run() -> None:
                try:
                    if self.tts.play_feedback_audio(audio, self._SR):
                        logger.debug("chime '%s' played via TTS feedback path", event)
                        return
                except Exception as exc:
                    logger.debug("chime '%s' feedback path failed: %s", event, exc)

                if aplay_bin:
                    pcm = (audio * 32767).clip(-32768, 32767).astype("int16")
                    pcm_bytes = pcm.tobytes()
                    chime_cmd = [
                        aplay_bin,
                        "-r",
                        str(self._SR),
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
                        try:
                            proc = subprocess.Popen(
                                chime_cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                            _, stderr = proc.communicate(input=pcm_bytes, timeout=3)
                            if proc.returncode == 0:
                                logger.debug("chime '%s' played OK", event)
                                return
                            logger.warning(
                                "chime '%s' aplay exit %d (attempt %d): %s",
                                event, proc.returncode, attempt + 1,
                                stderr.decode(errors="replace").strip()[:100],
                            )
                        except subprocess.TimeoutExpired:
                            logger.warning("chime '%s' timed out (attempt %d)", event, attempt + 1)
                            try:
                                proc.kill()
                            except Exception:
                                logger.exception("[Audio] Chime proc kill failed")
                        except Exception as _e:
                            logger.warning("chime '%s' failed (attempt %d): %s", event, attempt + 1, _e)
                        if attempt == 0:
                            time.sleep(0.05)  # brief pause before retry
                    return

                sd.play(audio, self._SR, blocking=False)
                logger.debug("chime '%s' queued via sounddevice", event)

            threading.Thread(target=_run, daemon=True).start()
        except Exception as _e:
            logger.warning("chime '%s' synthesis failed: %s", event, _e)

    # -- Individual chime generators --

    def _chime_acknowledge(self) -> np.ndarray:
        """Two-note ascending major third — quick, warm, like iOS 'received'."""
        sr = self._SR
        notes = [880, 1108.73]  # A5 -> C#6 (major third)
        note_dur = 0.06
        gap = 0.015
        total = len(notes) * note_dur + (len(notes) - 1) * gap
        audio = np.zeros(int(sr * total), dtype=np.float32)

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
            audio[offset:offset + n] += tone
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
        audio = np.zeros(int(sr * total), dtype=np.float32)

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
            audio[offset:offset + n] += tone
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
        tone += 0.12 * np.sin(phase * 2.0)   # 1800Hz harmonic
        tone += 0.05 * np.sin(phase * 3.0)   # 2700Hz harmonic
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
        audio = np.zeros(int(sr * total), dtype=np.float32)

        offset = 0
        for freq in notes:
            n = int(sr * note_dur)
            t = np.linspace(0, note_dur, n, endpoint=False, dtype=np.float32)
            tone = 0.25 * np.sin(2 * np.pi * freq * t)
            tone *= np.exp(-t * 12)
            audio[offset:offset + n] += tone
            offset += n + int(sr * gap)

        return audio

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal all background threads to stop."""
        self.stop_event.set()
        self.tts.shutdown()
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
        input_ready = bool(
            self.voice_mode
            and self.asr is not None
            and self.vad is not None
            and self._mic.is_open
        )
        snapshot = {
            "run_id": self._run_id,
            "mode": "voice" if self.voice_mode else "text",
            "enabled": self.voice_mode,
            "input_ready": input_ready,
            "output_ready": output_ready,
            "pipeline_ok": output_ready and (not self.voice_mode or input_ready),
            "asr_available": self.asr is not None,
            "vad_available": self.vad is not None,
            "kws_available": bool(self.kws and getattr(self.kws, "available", False)),
            "wake_word_enabled": bool(
                self.voice_mode and self.kws and getattr(self.kws, "available", False)
            ),
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
            "voice_turn": self._turn_traces.snapshot(),
            "asr": self._component_status_snapshot(self._asr_mgr),
            "tts": tts_snapshot,
            "pending_runtime_updates": {
                "asr": self._pending_asr_config is not None,
                "tts": self._pending_tts_config is not None,
            },
            "input": self._input_status_snapshot(),
        }
        snapshot.update(overrides)
        self._metrics.update_voice_state(**snapshot)
        return snapshot

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
        input_ready = bool(
            self.voice_mode
            and self.asr is not None
            and self.vad is not None
            and self._mic.is_open
        )
        if output_ready is None:
            output_ready = self._tts_output_ready()
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
            can_talk = True
            hint = "barge_in_allowed"
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
        return {
            "media_transport": self._media_transport,
            "room_id": "",
            "participant_count": 1 if self.voice_mode else 0,
            "packet_loss": None,
            "jitter_ms": None,
            "input_transport": getattr(self._mic, "_input_transport", "auto"),
            "output_transport": getattr(self.tts, "_output_transport", "auto"),
            "session_id": self._run_id,
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
