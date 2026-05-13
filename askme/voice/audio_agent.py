"""AudioAgent - high-level voice I/O controller composing ASR, VAD, KWS, and TTS engines."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from enum import Enum
from typing import Any
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

from askme.robot.ota_bridge import OTABridgeMetrics, get_ota_runtime_metrics

from .asr_manager import (
    _CONFIRMATION_WORDS,  # noqa: F401 — re-exported for tests
    _MIN_VALID_TEXT_LEN,  # noqa: F401 — re-exported for tests
    _NOISE_UTTERANCES,  # noqa: F401 — re-exported for tests
    _SINGLE_CHAR_COMMANDS,  # noqa: F401 — re-exported for tests
    ASRManager,
)
from .audio_processor import AudioProcessor
from .audio_router import AudioRouter
from .kws import KWSEngine
from .mic_input import MicInput
from .tts import TTSEngine
from .turn_trace import VoiceTurnTraceRecorder
from .vad_controller import (
    _BARGE_IN_HOLD_S,  # noqa: F401 — re-exported for tests
    _MAX_SPEECH_DURATION,  # noqa: F401 — re-exported for tests
    VADController,
    VADEvent,
)

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
        self._muted: bool = False  # software mute — still listens, VoiceLoop filters results
        self._agent_state: AgentState = AgentState.IDLE
        # When True, confirmation words bypass the noise filter.
        self.awaiting_confirmation: bool = False
        self._chime_lock = threading.Lock()
        self._last_chime_at: float = 0.0

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

        # -- New modular components --
        self._mic = MicInput.from_config(config, audio_router=audio_router)
        self._media_transport = self._resolve_media_transport_label(voice_cfg)
        self._audio_proc = AudioProcessor(voice_cfg)
        self._vad_ctrl = VADController(voice_cfg)
        self._asr_mgr = ASRManager(voice_cfg)

        if voice_mode:
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
            self.punct = self._asr_mgr._punct
            self.asr_stream = None
            self.kws_stream = None
            self.woken_up = True

        # -- Output engine --
        self.tts = TTSEngine(voice_cfg.get("tts", {}), audio_router=audio_router)
        logger.info("AudioAgent run_id=%s mode=%s", self._run_id, "voice" if voice_mode else "text")
        self._refresh_voice_metrics()

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to TTS)
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Whether TTS is actively playing or has queued text."""
        return self.tts._is_playing or not self.tts.tts_text_queue.empty()

    def speak(self, text: str) -> None:
        """Queue text for TTS (strips emoji/markdown internally)."""
        self.tts.speak(text)
        self._refresh_voice_metrics()

    def start_playback(self) -> None:
        self._ready_chime_generation += 1
        self._agent_state = AgentState.SPEAKING
        self._turn_traces.mark("tts_playback_started")
        self.tts.start_playback()
        self._refresh_voice_metrics()

    def stop_playback(self) -> None:
        self.tts.stop_playback()
        self._begin_post_tts_input_cooldown()
        self._agent_state = AgentState.IDLE
        self._turn_traces.mark("playback_done")
        self._refresh_voice_metrics()
        self._schedule_ready_chime()

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
        await asyncio.to_thread(self.wait_speaking_done)
        self.stop_playback()

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
            if self._muted or self.is_busy or self._agent_state == AgentState.SPEAKING:
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

    def adjust_speed(self, delta: float) -> float:
        """Adjust TTS speed by delta. Returns new value."""
        return self.set_speed(self.tts.speed + delta)

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

    def listen_loop(self) -> str | None:
        """Listen with VAD-gated ASR using modular pipeline.

        Flow: MicInput -> AudioProcessor -> VADController -> ASRManager -> text

        Returns recognized text, or None on timeout/stop.
        """
        if self.asr is None or self.vad is None:
            raise RuntimeError("listen_loop requires voice_mode=True")

        self._start_voice_turn_trace()
        self._metrics.mark_voice_listen_started()
        self._refresh_voice_metrics()

        mic = self._mic
        proc = self._audio_proc
        vad = self._vad_ctrl
        asr = self._asr_mgr

        try:
            # Mic is persistently open (started by VoiceModule).
            # Flush stale audio that accumulated during LLM+TTS processing.
            mic._flush_queue()

            with mic.open() as mic_ctx:
                # Phase 1: Wake word detection (if KWS available)
                if self.kws and self.kws.available and self.kws_stream:
                    _within_wake_window = (
                        self._wake_timeout > 0
                        and self._last_interaction_time > 0
                        and (time.monotonic() - self._last_interaction_time) < self._wake_timeout
                    )
                    if _within_wake_window:
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
                        self._play_chime("wake")

                # Phase 2: VAD-gated ASR
                # Play beep in background (non-blocking)
                try:
                    import subprocess as _sp
                    import threading as _th
                    _out_dev = getattr(self.tts, "_output_device", None)
                    _beep_cmd = ["aplay", "-r", "44100", "-f", "S16_LE", "-c", "1", "-q"]
                    if _out_dev:
                        _beep_cmd += ["-D", str(_out_dev)]
                    _beep_sr = 44100
                    _beep_t = np.linspace(0, 0.15, int(_beep_sr * 0.15))
                    _beep_pcm = (np.sin(2 * np.pi * 880 * _beep_t) * 20000).astype(np.int16).tobytes()
                    def _play_beep():
                        try:
                            _sp.run(_beep_cmd, input=_beep_pcm, capture_output=True, timeout=2)
                        except Exception:
                            pass
                    _th.Thread(target=_play_beep, daemon=True).start()
                except Exception:
                    pass
                logger.info("Listening for speech...")
                asr.preconnect_cloud()  # warm up WebSocket (fast, ~100ms)
                deadline = time.monotonic() + self._asr_timeout
                vad.reset()
                _vol_log_interval = 0.5
                _vol_log_next = time.monotonic() + _vol_log_interval

                while not self.stop_event.is_set():
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

                    event = vad.feed(samples_i16, peak, tts_active=tts_active)
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
                        self._turn_traces.mark(
                            "vad_start",
                            peak=peak,
                            rms=rms,
                        )
                        self._agent_state = AgentState.LISTENING
                        self._refresh_voice_metrics()
                        asr.start_session()
                        for buf in mic_ctx.flush_pre_roll():
                            asr.feed_audio(buf, MicInput.to_int16(buf), mic_ctx.sample_rate)
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.SPEECH_CONTINUE:
                        deadline = time.monotonic() + self._asr_timeout
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

                    elif event == VADEvent.BARGE_IN_CONFIRMED:
                        self._turn_traces.mark_barge_in(peak=peak, rms=rms)
                        self._agent_state = AgentState.LISTENING
                        self._refresh_voice_metrics()
                        self.tts.drain_buffers()
                        self.tts.stop_immediately()
                        asr.start_session()
                        for buf in vad.barge_in_buffer:
                            asr.feed_audio(buf, MicInput.to_int16(buf), mic_ctx.sample_rate)
                        vad.barge_in_buffer.clear()
                        mic_ctx.pre_roll.clear()
                        asr.feed_audio(samples_f32, samples_i16, mic_ctx.sample_rate)

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
                            vad.reset()
                            deadline = time.monotonic() + self._asr_timeout
                            self._start_voice_turn_trace()

        except Exception as exc:
            self._metrics.mark_voice_error(str(exc))
            self._mark_input_failure(str(exc))
            self._turn_traces.finish("error", error=str(exc))
            self._refresh_voice_metrics(pipeline_ok=False)
            raise

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
        self._last_interaction_time = time.monotonic()
        self._refresh_voice_metrics()
        self._asr_mgr.reset()
        return text

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

    def _wait_for_wake_word_mic(self, mic_ctx: MicInput) -> bool:
        """Block until wake word is detected via KWS (MicInput API).

        Returns True when wake word is detected, False if stop_event is set.
        """
        logger.info("Waiting for wake word...")
        sample_rate = mic_ctx.sample_rate
        while not self.stop_event.is_set():
            samples = mic_ctx.read_chunk()

            try:
                self.kws_stream.accept_waveform(sample_rate, samples)

                while self.kws.spotter.is_ready(self.kws_stream):
                    self.kws.spotter.decode_stream(self.kws_stream)

                result = self.kws.spotter.get_result(self.kws_stream)
            except Exception as e:
                logger.error("KWS error: %s", e)
                return False

            if result:
                logger.info("Wake word detected: %s", result.strip())
                self.woken_up = True
                self.kws_stream = self.kws.create_stream()
                self._refresh_voice_metrics()
                return True

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
                                pass
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
        snapshot = {
            "run_id": self._run_id,
            "mode": "voice" if self.voice_mode else "text",
            "enabled": self.voice_mode,
            "input_ready": bool(
                self.voice_mode and self.asr is not None and self.vad is not None
            ),
            "output_ready": self.tts is not None,
            "pipeline_ok": bool(self.tts) and (
                not self.voice_mode or (self.asr is not None and self.vad is not None)
            ),
            "asr_available": self.asr is not None,
            "vad_available": self.vad is not None,
            "kws_available": bool(self.kws and getattr(self.kws, "available", False)),
            "wake_word_enabled": bool(
                self.voice_mode and self.kws and getattr(self.kws, "available", False)
            ),
            "woken_up": self.woken_up,
            "muted": self._muted,
            "tts_backend": self.tts.backend,
            "tts_busy": self.is_busy,
            "agent_state": self._agent_state.value,
            "interaction": self._interaction_status_snapshot(),
            "media": self._media_status_snapshot(),
            "voice_turn": self._turn_traces.snapshot(),
            "asr": self._asr_mgr.status_snapshot(),
            "tts": self.tts.status_snapshot(),
            "input": self._input_status_snapshot(),
        }
        snapshot.update(overrides)
        self._metrics.update_voice_state(**snapshot)
        return snapshot

    def _interaction_status_snapshot(self) -> dict[str, Any]:
        cooldown_remaining = round(self._in_post_tts_input_cooldown(), 2)
        input_ready = bool(
            self.voice_mode and self.asr is not None and self.vad is not None
        )
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
