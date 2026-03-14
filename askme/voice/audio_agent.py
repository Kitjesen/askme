"""AudioAgent - high-level voice I/O controller composing ASR, VAD, KWS, and TTS engines."""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from enum import Enum
from typing import Any

import numpy as np
import sounddevice as sd

from askme.ota_bridge import OTABridgeMetrics, get_ota_runtime_metrics

from .asr import ASREngine
from .audio_router import AudioRouter
from .kws import KWSEngine
from .punctuation import PunctuationRestorer
from .tts import TTSEngine
from .vad import VADEngine

logger = logging.getLogger(__name__)

# Default ASR timeout (overridden by config voice.asr.asr_timeout)
_DEFAULT_ASR_TIMEOUT = 10.0

# Number of audio chunks to buffer before VAD triggers (pre-roll).
# Each chunk is ~100ms, so 5 chunks = ~500ms lookback.
_PRE_ROLL_CHUNKS = 5

# Maximum continuous speech duration before forcing an ASR endpoint.
# Prevents VAD deadlock from persistent background noise.
_MAX_SPEECH_DURATION = 30.0

# Minimum continuous speech duration (seconds) required to confirm a barge-in
# during TTS playback.  Shorter bursts (coughs, noise) are ignored and TTS
# continues.  Only applies when TTS is actively playing; when idle the first
# VAD frame triggers immediately as before.
_BARGE_IN_HOLD_S = 0.15

# ASR results that are clearly noise or feedback sounds, not commands.
# When matched, we speak a clarification prompt and re-listen.
_NOISE_UTTERANCES: frozenset[str] = frozenset([
    "嗯", "哦", "啊", "呢", "哈", "噢", "哇", "呀",
    "嗯嗯", "哦哦", "啊啊", "嗯？", "哦？", "啊？",
    "的", "了", "吧", "嘛",
])


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

        # -- Input engines (only in voice mode) --
        self._asr_timeout: float = voice_cfg.get("asr", {}).get(
            "asr_timeout", _DEFAULT_ASR_TIMEOUT
        )

        # Echo gate: suppress mic input during TTS playback when peak is low.
        # Speaker echo is typically peak 50-500; direct speech is peak 800+.
        # Set 0 to disable (fall back to wait-until-done behaviour).
        self._echo_gate_peak: int = int(voice_cfg.get("echo_gate_peak", 800))

        # Microphone input device index (None = system default).
        # On sunrise: device 0 = MCP01 USB Audio (hw:1,0).
        _raw_input = voice_cfg.get("input_device", None)
        self._input_device: int | None = int(_raw_input) if _raw_input is not None else None

        # Noise gate: skip VAD entirely when peak is below this threshold.
        # Prevents USB mic noise floor from continuously triggering Silero VAD.
        # Set 0 to disable. Typical values: 300-600 for noisy USB mics.
        self._noise_gate_peak: int = int(voice_cfg.get("noise_gate_peak", 0))

        if voice_mode:
            self.asr = ASREngine(voice_cfg.get("asr", {}))
            self.vad = VADEngine(voice_cfg.get("vad", {}))
            self.kws = KWSEngine(voice_cfg.get("kws", {}))
            self.punct = PunctuationRestorer(voice_cfg.get("punctuation", {}))
            self.asr_stream = self.asr.create_stream()

            if self.kws.available:
                self.kws_stream = self.kws.create_stream()
            else:
                self.kws_stream = None
                self.woken_up = True  # fallback: always awake when no KWS
        else:
            self.asr = None  # type: ignore[assignment]
            self.vad = None  # type: ignore[assignment]
            self.kws = None  # type: ignore[assignment]
            self.punct = PunctuationRestorer({})
            self.asr_stream = None
            self.kws_stream = None
            self.woken_up = True

        # -- Output engine --
        self.tts = TTSEngine(voice_cfg.get("tts", {}), audio_router=audio_router)
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
        self._agent_state = AgentState.SPEAKING
        self.tts.start_playback()
        self._refresh_voice_metrics()

    def stop_playback(self) -> None:
        self.tts.stop_playback()
        self._agent_state = AgentState.IDLE
        self._refresh_voice_metrics()

    def wait_speaking_done(self) -> None:
        self.tts.wait_done()
        self._refresh_voice_metrics()

    def drain_buffers(self) -> None:
        """Clear any leftover TTS text/audio from a previous turn."""
        self.tts.drain_buffers()
        self._refresh_voice_metrics()

    def stop_immediately(self) -> None:
        """Immediately stop TTS playback mid-chunk (barge-in support)."""
        self.tts.stop_immediately()
        self._refresh_voice_metrics()

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

    def speak_error(self) -> None:
        """Speak a short error notification to the user."""
        self._metrics.mark_voice_error("voice interaction error")
        self._play_chime("error")
        self.tts.speak("抱歉，出现了问题，请重试。")
        self._refresh_voice_metrics()

    # ------------------------------------------------------------------
    # Microphone listen loop
    # ------------------------------------------------------------------

    def listen_loop(self) -> str | None:
        """Listen from microphone with optional wake word detection and VAD-gated ASR.

        Flow:
            1. If KWS is available, wait for wake word first
            2. Play acknowledgment tone
            3. Listen for speech with VAD-gated ASR (timeout: self._asr_timeout)
            4. Return recognized text, or None on timeout/stop

        Returns None if ``stop_event`` is set or ASR times out.
        """
        if self.asr is None or self.vad is None:
            raise RuntimeError("listen_loop requires voice_mode=True")

        sample_rate: int = self.asr.sample_rate
        samples_per_read: int = int(0.1 * sample_rate)  # 100ms chunks

        self._metrics.mark_voice_listen_started()
        self._refresh_voice_metrics()

        # Wait for TTS output to release the device before opening the mic.
        # On half-duplex ALSA hardware (sunrise) aplay and sd.InputStream share
        # the same physical device; opening the mic while aplay is running
        # causes an XRUN cascade on the next listen iteration.
        if self._audio_router is not None:
            self._audio_router.wait_for_input_ready(timeout=10.0)

        try:
            with sd.InputStream(
                device=self._input_device,
                channels=1,
                dtype="float32",
                samplerate=sample_rate,
            ) as mic:
                # Phase 1: Wait for wake word (if KWS available)
                if self.kws and self.kws.available and self.kws_stream:
                    self.woken_up = False
                    self._refresh_voice_metrics()
                    if not self._wait_for_wake_word(mic, sample_rate, samples_per_read):
                        return None  # stop_event was set
                    self._play_chime("wake")  # bright chime = "I'm listening"

                # Phase 2: VAD-gated ASR with timeout
                logger.info("Listening for speech...")
                speech_active = False
                speech_start_time: float = 0.0  # monotonic time when speech started
                deadline = time.monotonic() + self._asr_timeout
                _vol_log_interval = 0.5  # log volume every 0.5s
                _vol_log_next = time.monotonic() + _vol_log_interval

                # Pre-roll buffer: keep recent chunks so VAD latency
                # doesn't lose the beginning of speech.
                pre_roll: collections.deque[np.ndarray] = collections.deque(
                    maxlen=_PRE_ROLL_CHUNKS
                )

                # Barge-in hold state: tracks a candidate barge-in while TTS
                # is playing.  Reset when confirmed or dismissed.
                barge_in_pending: bool = False
                barge_in_start: float = 0.0
                barge_in_buffer: list[np.ndarray] = []

                while not self.stop_event.is_set():
                    # Mark IDLE at the top of each iteration (waiting for input)
                    if self._agent_state not in (AgentState.MUTED, AgentState.LISTENING):
                        self._agent_state = AgentState.IDLE

                    # Timeout check
                    if time.monotonic() > deadline:
                        logger.info(
                            "ASR timeout: no speech detected within %.0fs.",
                            self._asr_timeout,
                        )
                        self.asr.reset(self.asr_stream)
                        self.asr_stream = self.asr.create_stream()
                        self._refresh_voice_metrics()
                        if speech_active:
                            self.tts.speak("没听清楚，请再说一遍。")
                        else:
                            self.tts.speak("还在听呢，有什么需要帮忙的？")
                        self.start_playback()
                        self.wait_speaking_done()
                        self.stop_playback()
                        return None

                    samples, _ = mic.read(samples_per_read)
                    samples = samples.reshape(-1)

                    # Feed VAD with int16 samples
                    samples_int16 = (samples * 32768).astype(np.int16)
                    peak = int(np.max(np.abs(samples_int16)))

                    # Echo gate: during TTS playback, suppress low-energy mic
                    # input to prevent speaker echo from triggering VAD.
                    # High-energy input (user barge-in) still passes through.
                    # Do NOT gate once speech_active=True — the user is already
                    # speaking (barge-in), so we must keep feeding VAD+ASR even
                    # if volume drops, or the utterance will be silently truncated.
                    tts_active = self.tts.is_active()
                    if tts_active and self._echo_gate_peak > 0 and peak < self._echo_gate_peak and not speech_active:
                        pre_roll.append(samples.copy())
                        # Still log periodically
                        now = time.monotonic()
                        if now >= _vol_log_next:
                            logger.info(
                                "MIC peak=%5d VAD=gated (TTS playing)", peak,
                            )
                            _vol_log_next = now + _vol_log_interval
                        continue

                    try:
                        self.vad.accept_waveform(samples_int16)
                    except Exception as e:
                        logger.error("VAD accept_waveform error: %s", e)
                        break

                    # Periodically log audio level for diagnostics
                    now = time.monotonic()
                    if now >= _vol_log_next:
                        try:
                            vad_on = self.vad.is_speech_detected()
                        except Exception as e:
                            logger.error("VAD is_speech_detected error: %s", e)
                            break
                        bar_len = min(peak // 500, 30)
                        bar = "#" * bar_len
                        logger.info(
                            "MIC peak=%5d VAD=%s %s",
                            peak, "SPEECH" if vad_on else "silent", bar,
                        )
                        _vol_log_next = now + _vol_log_interval

                    # Only feed ASR when VAD detects speech
                    try:
                        vad_speech = self.vad.is_speech_detected()
                    except Exception as e:
                        logger.error("VAD is_speech_detected error: %s", e)
                        break

                    if vad_speech:
                        # Reset deadline on every frame with detected speech,
                        # so the timeout is always relative to the last voice
                        # activity, not the start of listening.
                        deadline = time.monotonic() + self._asr_timeout

                        if not speech_active:
                            # Noise gate: require amplitude above threshold to
                            # start speech capture. This prevents USB mic noise
                            # floor (peak ~100-200) from triggering ASR even
                            # when Silero VAD classifies it as speech.
                            # Always feed VAD (above) for correct state tracking;
                            # only gate the speech_active transition.
                            if self._noise_gate_peak > 0 and peak < self._noise_gate_peak:
                                pre_roll.append(samples.copy())
                            elif self.tts.is_active() and not barge_in_pending:
                                # TTS is playing: start barge-in hold.
                                # Don't interrupt TTS yet — wait _BARGE_IN_HOLD_S
                                # to confirm this is real speech, not a cough or
                                # noise burst.
                                barge_in_pending = True
                                barge_in_start = time.monotonic()
                                barge_in_buffer = list(pre_roll) + [samples.copy()]
                                logger.info(
                                    "VAD: barge-in candidate (peak=%d), "
                                    "holding %.0fms",
                                    peak, _BARGE_IN_HOLD_S * 1000,
                                )
                            elif barge_in_pending:
                                # Still within hold window — accumulate
                                barge_in_buffer.append(samples.copy())
                                if (time.monotonic() - barge_in_start) >= _BARGE_IN_HOLD_S:
                                    # Confirmed real barge-in
                                    speech_active = True
                                    speech_start_time = barge_in_start
                                    barge_in_pending = False
                                    self._agent_state = AgentState.LISTENING
                                    logger.info(
                                        "VAD: barge-in confirmed (peak=%d)", peak
                                    )
                                    self.tts.drain_buffers()
                                    self.tts.stop_immediately()
                                    self._refresh_voice_metrics()
                                    for buffered in barge_in_buffer:
                                        try:
                                            self.asr_stream.accept_waveform(
                                                sample_rate, buffered
                                            )
                                        except Exception as e:
                                            logger.error(
                                                "ASR accept_waveform (barge-in) error: %s", e
                                            )
                                            break
                                    barge_in_buffer.clear()
                                    pre_roll.clear()
                            else:
                                # No TTS playing — immediate activation (original behaviour)
                                speech_active = True
                                speech_start_time = time.monotonic()
                                self._agent_state = AgentState.LISTENING
                                logger.info("VAD: speech start (peak=%d)", peak)
                                self.tts.drain_buffers()
                                self.tts.stop_immediately()
                                self._refresh_voice_metrics()
                                # Flush pre-roll buffer -> catch the speech onset
                                for buffered in pre_roll:
                                    try:
                                        self.asr_stream.accept_waveform(
                                            sample_rate, buffered
                                        )
                                    except Exception as e:
                                        logger.error("ASR accept_waveform (pre-roll) error: %s", e)
                                        break
                                pre_roll.clear()
                        if speech_active:
                            try:
                                self.asr_stream.accept_waveform(sample_rate, samples)
                            except Exception as e:
                                logger.error("ASR accept_waveform error: %s", e)
                                break

                            try:
                                while self.asr.is_ready(self.asr_stream):
                                    self.asr.decode_stream(self.asr_stream)
                            except Exception as e:
                                logger.error("ASR decode error: %s", e)
                                break

                            # Max speech duration guard: continuous speech > 30s
                            # means VAD is stuck (background noise). Force an
                            # ASR endpoint to avoid infinite listen_loop.
                            if (time.monotonic() - speech_start_time) > _MAX_SPEECH_DURATION:
                                logger.warning(
                                    "VAD: max speech duration (%.0fs) exceeded, forcing endpoint",
                                    _MAX_SPEECH_DURATION,
                                )
                                try:
                                    while self.asr.is_ready(self.asr_stream):
                                        self.asr.decode_stream(self.asr_stream)
                                    text = self.asr.get_result(self.asr_stream)
                                except Exception as e:
                                    logger.error("ASR forced-endpoint error: %s", e)
                                    text = ""
                                self.asr.reset(self.asr_stream)
                                self.asr_stream = self.asr.create_stream()
                                speech_active = False
                                speech_start_time = 0.0
                                # Result is likely noise -- only return if non-trivial
                                if text and len(text.strip()) > 1:
                                    text = text.strip()
                                    if self.punct.available:
                                        text = self.punct.restore(text)
                                    logger.info("Recognized (forced): %s", text)
                                    self.audio_queue.put(text)
                                    self._metrics.mark_voice_input(text)
                                    self._agent_state = AgentState.PROCESSING
                                    self._refresh_voice_metrics()
                                    return text
                                else:
                                    logger.info("VAD: forced endpoint yielded noise, discarding")
                                    deadline = time.monotonic() + self._asr_timeout
                                    continue
                    else:
                        # Buffer recent silence chunks for pre-roll
                        pre_roll.append(samples.copy())
                        # False barge-in: VAD dropped during hold window.
                        # TTS continues undisturbed.
                        if barge_in_pending:
                            elapsed = time.monotonic() - barge_in_start
                            logger.info(
                                "VAD: false barge-in dismissed (only %.0fms)", elapsed * 1000
                            )
                            barge_in_pending = False
                            barge_in_buffer.clear()
                        if speech_active:
                            # Speech just ended -- feed remaining and check
                            logger.info("VAD: speech end")
                            speech_active = False
                            speech_start_time = 0.0
                            try:
                                self.asr_stream.accept_waveform(sample_rate, samples)
                                while self.asr.is_ready(self.asr_stream):
                                    self.asr.decode_stream(self.asr_stream)
                            except Exception as e:
                                logger.error("ASR error at speech end: %s", e)
                                break

                    # Check for endpoint
                    try:
                        is_endpoint = self.asr.is_endpoint(self.asr_stream)
                        text = self.asr.get_result(self.asr_stream)
                    except Exception as e:
                        logger.error("ASR endpoint/result error: %s", e)
                        break

                    if is_endpoint and text:
                        text = text.strip()
                        if len(text) > 0:
                            # Noise / feedback sound filter: single syllables
                            # or known feedback sounds ("嗯", "哦", etc.) are
                            # ignored and we re-listen without exiting.
                            if text in _NOISE_UTTERANCES or (len(text) == 1 and text not in ("停", "走", "站")):
                                logger.info(
                                    "ASR noise utterance filtered: '%s' — re-listening", text
                                )
                                self.asr.reset(self.asr_stream)
                                self.asr_stream = self.asr.create_stream()
                                speech_active = False
                                speech_start_time = 0.0
                                self.tts.speak("没有收到有效指令，请说完整的话。")
                                self.start_playback()
                                self.wait_speaking_done()
                                self.stop_playback()
                                deadline = time.monotonic() + self._asr_timeout
                                continue
                            # Add punctuation to raw ASR output
                            if self.punct.available:
                                text = self.punct.restore(text)
                            logger.info("Recognized: %s", text)
                            self.audio_queue.put(text)
                            self._metrics.mark_voice_input(text)
                            self._agent_state = AgentState.PROCESSING
                            self._refresh_voice_metrics()
                            self.asr.reset(self.asr_stream)
                            self.asr_stream = self.asr.create_stream()
                            return text
        except Exception as exc:
            self._metrics.mark_voice_error(str(exc))
            self._refresh_voice_metrics(pipeline_ok=False)
            raise

        return None

    # ------------------------------------------------------------------
    # Wake word detection
    # ------------------------------------------------------------------

    def _wait_for_wake_word(self, mic: Any, sample_rate: int, samples_per_read: int) -> bool:
        """Block until wake word is detected via KWS.

        Returns True when wake word is detected, False if stop_event is set.
        """
        logger.info("Waiting for wake word...")
        while not self.stop_event.is_set():
            samples, _ = mic.read(samples_per_read)
            samples = samples.reshape(-1)

            try:
                self.kws_stream.accept_waveform(sample_rate, samples)

                while self.kws.spotter.is_ready(self.kws_stream):
                    self.kws.spotter.decode(self.kws_stream)

                result = self.kws.spotter.get_result(self.kws_stream)
            except Exception as e:
                logger.error("KWS error: %s", e)
                return False

            if result:
                logger.info("Wake word detected: %s", result.strip())
                self.woken_up = True
                # Reset stream for next detection cycle
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
        """
        try:
            generators = {
                "acknowledge": self._chime_acknowledge,
                "wake": self._chime_wake,
                "error": self._chime_error,
            }
            gen = generators.get(event, self._chime_acknowledge)
            audio = gen()

            aplay_bin = getattr(self.tts, "_aplay_bin", None)
            if aplay_bin:
                # aplay path: non-blocking, runs in daemon thread
                pcm = (audio * 32767).clip(-32768, 32767).astype("int16")
                pcm_bytes = pcm.tobytes()

                output_device = getattr(self.tts, "_output_device", None)
                chime_cmd = [aplay_bin, "-r", str(self._SR), "-f", "S16_LE", "-c", "1", "-q"]
                if output_device is not None:
                    chime_cmd += ["-D", str(output_device)]

                def _run() -> None:
                    try:
                        import subprocess
                        proc = subprocess.Popen(chime_cmd, stdin=subprocess.PIPE)
                        proc.communicate(input=pcm_bytes)
                    except Exception as _e:
                        logger.debug("chime subprocess failed: %s", _e)

                threading.Thread(target=_run, daemon=True).start()
            else:
                sd.play(audio, self._SR, blocking=False)
        except Exception as _e:
            logger.debug("chime failed: %s", _e)

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
        }
        snapshot.update(overrides)
        self._metrics.update_voice_state(**snapshot)
        return snapshot
