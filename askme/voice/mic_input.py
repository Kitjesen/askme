"""Microphone input module — device management + audio capture.

Encapsulates sounddevice InputStream, chunk reading, peak calculation,
and pre-roll buffer. Extracted from audio_agent.py for independent testing.

Usage::

    mic = MicInput(device=0, sample_rate=16000)
    with mic.open():
        chunk = mic.read_chunk()         # float32 array
        peak = mic.get_peak(chunk)       # int peak from int16
        int16 = mic.to_int16(chunk)      # int16 conversion
"""

from __future__ import annotations

import collections
import logging
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Default chunk duration: 100ms per read
_DEFAULT_CHUNK_MS = 100

# Pre-roll buffer: keep recent chunks so VAD latency doesn't lose speech onset
_DEFAULT_PRE_ROLL_CHUNKS = 5


class MicInput:
    """Microphone input device wrapper.

    Manages the sounddevice InputStream lifecycle and provides
    clean chunk-based audio reading with pre-roll buffering.

    Config keys (under ``voice``)::

        input_device: int|str|null  - Device index or ALSA name (null=default)
        asr.sample_rate: int        - Sample rate (default 16000)
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = _DEFAULT_CHUNK_MS,
        pre_roll_chunks: int = _DEFAULT_PRE_ROLL_CHUNKS,
        audio_router: Any | None = None,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._chunk_samples = int(chunk_ms / 1000 * sample_rate)
        self._audio_router = audio_router
        self._stream: sd.InputStream | None = None

        # Pre-roll buffer: recent chunks for catching speech onset
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=pre_roll_chunks
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @contextmanager
    def open(self) -> Generator["MicInput", None, None]:
        """Open the microphone as a context manager.

        Coordinates with AudioRouter on half-duplex ALSA hardware
        (waits for TTS output to release the device first).
        """
        if self._audio_router is not None:
            self._audio_router.wait_for_input_ready(timeout=10.0)

        try:
            with sd.InputStream(
                device=self._device,
                channels=1,
                dtype="float32",
                samplerate=self._sample_rate,
            ) as stream:
                self._stream = stream
                self.pre_roll.clear()
                yield self
        finally:
            self._stream = None

    def read_chunk(self) -> np.ndarray:
        """Read one chunk of audio from the microphone.

        Returns float32 array of shape ``(chunk_samples,)``.
        Raises RuntimeError if mic is not open.
        """
        if self._stream is None:
            raise RuntimeError("MicInput not open — use 'with mic.open():'")
        samples, _ = self._stream.read(self._chunk_samples)
        return samples.reshape(-1)

    def buffer_pre_roll(self, samples: np.ndarray) -> None:
        """Add a chunk to the pre-roll buffer (recent silence for speech onset)."""
        self.pre_roll.append(samples.copy())

    def flush_pre_roll(self) -> list[np.ndarray]:
        """Return and clear the pre-roll buffer contents."""
        chunks = list(self.pre_roll)
        self.pre_roll.clear()
        return chunks

    @staticmethod
    def to_int16(samples: np.ndarray) -> np.ndarray:
        """Convert float32 samples to int16."""
        return (samples * 32768).astype(np.int16)

    @staticmethod
    def get_peak(samples_int16: np.ndarray) -> int:
        """Get the peak amplitude from int16 samples."""
        return int(np.max(np.abs(samples_int16)))

    @classmethod
    def from_config(cls, config: dict[str, Any], audio_router: Any = None) -> "MicInput":
        """Create MicInput from askme voice config dict."""
        voice_cfg = config.get("voice", {})

        raw_input = voice_cfg.get("input_device", None)
        if raw_input is None:
            device = None
        elif isinstance(raw_input, int):
            device = raw_input
        else:
            try:
                device = int(raw_input)
            except (ValueError, TypeError):
                device = str(raw_input)

        sample_rate = int(voice_cfg.get("asr", {}).get("sample_rate", 16000))

        return cls(
            device=device,
            sample_rate=sample_rate,
            audio_router=audio_router,
        )
