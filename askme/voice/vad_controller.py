"""VAD Controller - state machine with barge-in detection.

Wraps VADEngine with state tracking, barge-in hold logic,
and max speech duration guard.  Extracted from audio_agent.py
for independent testing.

Pure state machine -- no audio device access, no ASR, no TTS.
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np

from .vad import VADEngine


class VADEvent(Enum):
    """Events emitted by :class:`VADController.feed`."""

    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH_CONTINUE = "speech"
    SPEECH_END = "speech_end"
    BARGE_IN_START = "barge_in_start"
    BARGE_IN_CONFIRMED = "barge_in_confirmed"
    BARGE_IN_DISMISSED = "barge_in_dismissed"
    MAX_DURATION_EXCEEDED = "max_duration_exceeded"


class VADController:
    """VAD state machine with barge-in detection.

    Wraps Silero VAD with state tracking, barge-in hold logic,
    and max speech duration guard.  Extracted from audio_agent.py
    for independent testing.

    Config keys::

        vad: {model_path, threshold, min_silence_duration,
              min_speech_duration, sample_rate}
        noise_gate_peak: int
        barge_in_hold_s: float  (default 0.15)
        max_speech_duration: float  (default 30.0)
    """

    def __init__(self, config: dict) -> None:
        self._vad = VADEngine(config.get("vad", {}))
        self._noise_gate_peak = int(config.get("noise_gate_peak", 0))
        self._barge_in_hold = float(config.get("barge_in_hold_s", 0.15))
        self._max_speech_duration = float(config.get("max_speech_duration", 30.0))

        # State
        self._speech_active = False
        self._speech_start_time: float = 0.0
        self._barge_in_pending = False
        self._barge_in_start: float = 0.0
        self._barge_in_buffer: list[np.ndarray] = []

    # -- public API -----------------------------------------------------------

    def feed(
        self,
        samples_int16: np.ndarray,
        peak: int,
        tts_active: bool = False,
        *,
        _now: float | None = None,
    ) -> VADEvent:
        """Feed one audio chunk and return the state-transition event.

        Logic extracted from ``audio_agent.py`` listen_loop lines 448-660:

        1. Feed VAD.
        2. If VAD says *speech*:
           a. ``not speech_active`` + noise gate blocks -> SILENCE
           b. ``not speech_active`` + TTS playing      -> start barge-in hold
           c. barge-in pending + hold exceeded          -> BARGE_IN_CONFIRMED
           d. ``not speech_active`` + no TTS            -> SPEECH_START
           e. ``speech_active``                         -> SPEECH_CONTINUE
           f. max duration exceeded                     -> MAX_DURATION_EXCEEDED
        3. If VAD says *silence*:
           a. barge-in pending -> BARGE_IN_DISMISSED
           b. ``speech_active`` -> SPEECH_END
           c. else             -> SILENCE

        Parameters
        ----------
        samples_int16:
            One chunk of int16 PCM audio.
        peak:
            ``int(np.max(np.abs(samples_int16)))`` -- caller-computed.
        tts_active:
            Whether TTS is currently playing audio.
        _now:
            Monotonic timestamp override (testing only).
        """
        now = _now if _now is not None else time.monotonic()

        # Always feed the underlying VAD engine.
        self._vad.accept_waveform(samples_int16)
        vad_speech = self._vad.is_speech_detected()

        if vad_speech:
            if not self._speech_active:
                # (a) Noise gate blocks transition
                if self._noise_gate_peak > 0 and peak < self._noise_gate_peak:
                    return VADEvent.SILENCE

                # (b) TTS playing -- start barge-in hold
                if tts_active and not self._barge_in_pending:
                    self._barge_in_pending = True
                    self._barge_in_start = now
                    self._barge_in_buffer = [samples_int16.copy()]
                    return VADEvent.BARGE_IN_START

                # (c) Barge-in pending -- accumulate and check hold
                if self._barge_in_pending:
                    self._barge_in_buffer.append(samples_int16.copy())
                    if (now - self._barge_in_start) >= self._barge_in_hold:
                        self._speech_active = True
                        self._speech_start_time = self._barge_in_start
                        self._barge_in_pending = False
                        return VADEvent.BARGE_IN_CONFIRMED
                    # Still within hold window -- keep accumulating.
                    # Return SILENCE so caller knows no transition yet.
                    return VADEvent.SILENCE

                # (d) No TTS -- immediate activation
                self._speech_active = True
                self._speech_start_time = now
                return VADEvent.SPEECH_START

            # (e) Already speaking
            # (f) Max duration guard
            if (now - self._speech_start_time) > self._max_speech_duration:
                self._speech_active = False
                self._speech_start_time = 0.0
                return VADEvent.MAX_DURATION_EXCEEDED

            return VADEvent.SPEECH_CONTINUE

        else:
            # VAD says silence
            # (a) False barge-in
            if self._barge_in_pending:
                self._barge_in_pending = False
                self._barge_in_buffer.clear()
                return VADEvent.BARGE_IN_DISMISSED

            # (b) Speech just ended
            if self._speech_active:
                self._speech_active = False
                self._speech_start_time = 0.0
                return VADEvent.SPEECH_END

            # (c) Plain silence
            return VADEvent.SILENCE

    # -- properties -----------------------------------------------------------

    @property
    def speech_active(self) -> bool:
        """Whether the controller is currently in the speech-active state."""
        return self._speech_active

    @property
    def barge_in_buffer(self) -> list[np.ndarray]:
        """Audio chunks buffered during the barge-in hold period."""
        return self._barge_in_buffer

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new listening session."""
        self._speech_active = False
        self._speech_start_time = 0.0
        self._barge_in_pending = False
        self._barge_in_start = 0.0
        self._barge_in_buffer.clear()
