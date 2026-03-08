"""AudioAgent - high-level voice I/O controller composing ASR, VAD, KWS, and TTS engines."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import sounddevice as sd

from .asr import ASREngine
from .kws import KWSEngine
from .tts import TTSEngine
from .vad import VADEngine

# Default ASR timeout (overridden by config voice.asr.asr_timeout)
_DEFAULT_ASR_TIMEOUT = 10.0


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
        Full configuration dict.  The ``voice`` sub-dict is read automatically.
    voice_mode : bool
        If True (default), initialise ASR/VAD/KWS for microphone input.
        If False, only TTS is initialised (text-input mode).
    """

    def __init__(self, config: dict[str, Any], voice_mode: bool = True) -> None:
        voice_cfg = config.get("voice", {})

        # Shared state
        self.audio_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.woken_up: bool = False

        # -- Input engines (only in voice mode) --
        self._asr_timeout: float = voice_cfg.get("asr", {}).get(
            "asr_timeout", _DEFAULT_ASR_TIMEOUT
        )

        if voice_mode:
            self.asr = ASREngine(voice_cfg.get("asr", {}))
            self.vad = VADEngine(voice_cfg.get("vad", {}))
            self.kws = KWSEngine(voice_cfg.get("kws", {}))
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
            self.asr_stream = None
            self.kws_stream = None
            self.woken_up = True

        # -- Output engine --
        self.tts = TTSEngine(voice_cfg.get("tts", {}))

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

    def start_playback(self) -> None:
        self.tts.start_playback()

    def stop_playback(self) -> None:
        self.tts.stop_playback()

    def wait_speaking_done(self) -> None:
        self.tts.wait_done()

    def drain_buffers(self) -> None:
        """Clear any leftover TTS text/audio from a previous turn."""
        self.tts.drain_buffers()

    def acknowledge(self) -> None:
        """Play a brief confirmation tone: 'heard you, thinking'.

        Non-blocking. Fires immediately after ASR so the user has audio
        feedback during the LLM latency gap instead of dead silence.
        """
        self._play_tone(660, 0.07)

    def speak_error(self) -> None:
        """Speak a short error notification to the user."""
        self._play_tone(440, 0.1)
        self.tts.speak("抱歉，出了点问题，请再试一次。")

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

        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as mic:
            # Phase 1: Wait for wake word (if KWS available)
            if self.kws and self.kws.available and self.kws_stream:
                if not self._wait_for_wake_word(mic, sample_rate, samples_per_read):
                    return None  # stop_event was set
                self._play_tone(880, 0.12)  # short high beep = "I'm listening"

            # Phase 2: VAD-gated ASR with timeout
            logger.info("Listening for speech...")
            speech_active = False
            deadline = time.monotonic() + self._asr_timeout

            while not self.stop_event.is_set():
                # Timeout check
                if time.monotonic() > deadline:
                    logger.info("ASR timeout — no speech detected within %.0fs.", self._asr_timeout)
                    self.asr.reset(self.asr_stream)
                    self.asr_stream = self.asr.create_stream()
                    return None

                samples, _ = mic.read(samples_per_read)
                samples = samples.reshape(-1)

                # Feed VAD with int16 samples
                samples_int16 = (samples * 32768).astype(np.int16)
                self.vad.accept_waveform(samples_int16)

                # Only feed ASR when VAD detects speech
                if self.vad.is_speech_detected():
                    if not speech_active:
                        speech_active = True
                        logger.debug("VAD: speech detected — barge-in")
                        self.tts.drain_buffers()
                    # Extend deadline while user is actively speaking
                    deadline = time.monotonic() + self._asr_timeout

                    self.asr_stream.accept_waveform(sample_rate, samples)

                    while self.asr.is_ready(self.asr_stream):
                        self.asr.decode_stream(self.asr_stream)
                else:
                    if speech_active:
                        # Speech just ended -- feed remaining and check
                        speech_active = False
                        self.asr_stream.accept_waveform(sample_rate, samples)
                        while self.asr.is_ready(self.asr_stream):
                            self.asr.decode_stream(self.asr_stream)

                # Check for endpoint
                is_endpoint = self.asr.is_endpoint(self.asr_stream)
                text = self.asr.get_result(self.asr_stream)

                if is_endpoint and text:
                    text = text.strip()
                    if len(text) > 0:
                        logger.info("Recognized: %s", text)
                        self.audio_queue.put(text)
                        self.asr.reset(self.asr_stream)
                        self.asr_stream = self.asr.create_stream()
                        return text

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

            self.kws_stream.accept_waveform(sample_rate, samples)

            while self.kws.spotter.is_ready(self.kws_stream):
                self.kws.spotter.decode(self.kws_stream)

            result = self.kws.spotter.get_result(self.kws_stream)
            if result:
                logger.info("Wake word detected: %s", result.strip())
                # Reset stream for next detection cycle
                self.kws_stream = self.kws.create_stream()
                return True

        return False

    # ------------------------------------------------------------------
    # Audio feedback
    # ------------------------------------------------------------------

    def _play_tone(self, frequency: float, duration: float) -> None:
        """Play a short tone for audio feedback (non-blocking)."""
        try:
            sr = self.asr.sample_rate if self.asr else 16000
            n_samples = int(sr * duration)
            t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)
            tone = 0.3 * np.sin(2 * np.pi * frequency * t)
            # Fade in/out to avoid clicks (10ms)
            fade_len = min(int(0.01 * sr), n_samples // 4)
            if fade_len > 0:
                tone[:fade_len] *= np.linspace(0, 1, fade_len)
                tone[-fade_len:] *= np.linspace(1, 0, fade_len)
            sd.play(tone, sr, blocking=False)
        except Exception:
            pass  # non-critical audio feedback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal all background threads to stop."""
        self.stop_event.set()
        self.tts.shutdown()
