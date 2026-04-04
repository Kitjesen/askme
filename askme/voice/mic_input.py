"""Microphone input module — device management + audio capture.

Encapsulates sounddevice InputStream, chunk reading, peak calculation,
and pre-roll buffer. Extracted from audio_agent.py for independent testing.

Supports mics with non-standard native rates (e.g. 48 kHz only USB devices)
via automatic resampling + signal conditioning (high-pass filter, AGC).

Uses callback-based InputStream (not blocking reads) because some USB audio
devices on ALSA/aarch64 produce silence with blocking sd.InputStream.read().

Usage::

    mic = MicInput(device=0, sample_rate=16000)
    with mic.open():
        chunk = mic.read_chunk()         # float32 array at sample_rate
        peak = mic.get_peak(chunk)       # int peak from int16
        int16 = mic.to_int16(chunk)      # int16 conversion
"""

from __future__ import annotations

import collections
import logging
import queue
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
try:
    import sounddevice as sd
except ModuleNotFoundError:
    class _SoundDeviceStub:
        InputStream = None
        @staticmethod
        def query_devices(device: object = None, kind: object = None) -> object:
            return {}
    sd = _SoundDeviceStub()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default chunk duration: 100ms per read
_DEFAULT_CHUNK_MS = 100

# Pre-roll buffer: keep recent chunks so VAD latency doesn't lose speech onset
_DEFAULT_PRE_ROLL_CHUNKS = 5


class MicInput:
    """Microphone input device wrapper.

    Manages the sounddevice InputStream lifecycle and provides
    clean chunk-based audio reading with pre-roll buffering.

    When ``mic_native_rate`` differs from ``sample_rate``, the mic is opened
    at the native rate and each chunk is processed through:
    high-pass filter → AGC → polyphase resample → output at ``sample_rate``.

    Config keys (under ``voice``)::

        input_device: int|str|null      - Device index or ALSA name (null=default)
        asr.sample_rate: int            - Target sample rate for ASR (default 16000)
        mic_native_rate: int|null       - Mic hardware sample rate (null=same as asr)
        mic_channels: int               - Mic hardware channels (default 1)
        mic_channel_select: int         - Which channel to use (default 0)
        mic_highpass_hz: int            - High-pass filter cutoff (default 80)
        mic_agc_target_rms: float       - AGC target RMS (default 0.15, 0=off)
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = _DEFAULT_CHUNK_MS,
        pre_roll_chunks: int = _DEFAULT_PRE_ROLL_CHUNKS,
        audio_router: Any | None = None,
        *,
        mic_native_rate: int | None = None,
        mic_channels: int = 1,
        mic_channel_select: int = 0,
        mic_highpass_hz: int = 80,
        mic_agc_target_rms: float = 0.15,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._chunk_samples = int(chunk_ms / 1000 * sample_rate)
        self._audio_router = audio_router
        self._stream: sd.InputStream | None = None

        # Resampling pipeline config
        self._native_rate = mic_native_rate or sample_rate
        self._native_channels = mic_channels
        self._channel_select = mic_channel_select
        self._needs_resample = self._native_rate != self._sample_rate
        self._highpass_hz = mic_highpass_hz
        self._agc_target = mic_agc_target_rms

        # Native chunk size (at hardware rate)
        self._native_chunk = int(chunk_ms / 1000 * self._native_rate)

        # Streaming filter state (initialized on open)
        self._filter_state: np.ndarray | None = None
        self._filter_sos: np.ndarray | None = None
        self._agc_gain: float = 1.0

        # Callback-based audio queue (replaces blocking stream.read)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()

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

    def _init_pipeline(self) -> None:
        """Initialize the signal processing pipeline for resampling mics."""
        if not self._needs_resample:
            return

        # High-pass IIR filter to remove DC offset
        if self._highpass_hz > 0:
            from scipy.signal import butter, sosfilt_zi
            self._filter_sos = butter(
                2, self._highpass_hz, btype="high",
                fs=self._native_rate, output="sos",
            )
            self._filter_state = sosfilt_zi(self._filter_sos)
            # Expand zi for our mono signal
            self._filter_state = self._filter_state * 0.0

        # AGC gain
        self._agc_gain = 1.0

        logger.info(
            "MicInput pipeline: %dHz %dch → HPF@%dHz → AGC(%.2f) → %dHz",
            self._native_rate, self._native_channels,
            self._highpass_hz, self._agc_target, self._sample_rate,
        )

    def _process_chunk(self, raw: np.ndarray) -> np.ndarray:
        """Process a native-rate chunk through the signal pipeline.

        Args:
            raw: float32 array from InputStream (may be multi-channel).

        Returns:
            float32 mono array at ``self._sample_rate``.
        """
        # Channel select
        if raw.ndim == 2:
            audio = raw[:, self._channel_select]
        else:
            audio = raw.reshape(-1)

        # High-pass filter (streaming, preserves state across chunks)
        if self._filter_sos is not None:
            from scipy.signal import sosfilt
            audio, self._filter_state = sosfilt(
                self._filter_sos, audio, zi=self._filter_state,
            )
            audio = audio.astype(np.float32)

        # AGC: smooth gain adjustment targeting self._agc_target RMS
        if self._agc_target > 0:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms > 1e-5:
                desired_gain = self._agc_target / rms
                # Smooth gain changes (attack=fast, release=slow)
                if desired_gain < self._agc_gain:
                    self._agc_gain = 0.3 * desired_gain + 0.7 * self._agc_gain
                else:
                    self._agc_gain = 0.05 * desired_gain + 0.95 * self._agc_gain
                # Clamp gain to prevent saturation
                self._agc_gain = min(self._agc_gain, 5.0)
            audio = np.clip(audio * self._agc_gain, -1.0, 1.0)

        # Resample with scipy polyphase (deterministic, fixed output length)
        from scipy.signal import resample_poly
        from math import gcd
        up = self._sample_rate
        down = self._native_rate
        g = gcd(up, down)
        audio = resample_poly(audio, up // g, down // g).astype(np.float32)

        # Clip after resample: sinc interpolation can overshoot [-1, 1]
        return np.clip(audio, -1.0, 1.0)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """InputStream callback — pushes raw audio chunks to the queue."""
        if status:
            logger.debug("MicInput callback status: %s", status)
        self._audio_queue.put(indata.copy())

    def start(self) -> None:
        """Open mic stream persistently. Pair with stop().

        The mic stays open across listen/speak cycles so VAD can detect
        barge-in during TTS playback and LLM processing.
        """
        if self._stream is not None:
            return  # already open

        self._init_pipeline()

        open_rate = self._native_rate if self._needs_resample else self._sample_rate
        open_channels = self._native_channels if self._needs_resample else 1
        blocksize = self._native_chunk if self._needs_resample else self._chunk_samples

        self._flush_queue()

        stream = sd.InputStream(
            device=self._device,
            channels=open_channels,
            dtype="float32",
            samplerate=open_rate,
            blocksize=blocksize,
            callback=self._audio_callback,
        )
        stream.start()
        self._stream = stream
        self.pre_roll.clear()
        logger.info("MicInput: started (persistent)")

    def stop(self) -> None:
        """Close mic stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("MicInput: stopped")

    def _flush_queue(self) -> None:
        """Discard stale audio chunks from the callback queue."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @contextmanager
    def open(self) -> Generator["MicInput", None, None]:
        """Open the microphone as a context manager.

        If the mic is already persistently open (via start()), yields
        without reopening. Otherwise opens/closes for backward compat.
        """
        already_open = self.is_open
        if not already_open:
            self.start()
        try:
            yield self
        finally:
            if not already_open:
                self.stop()

    def read_chunk(self) -> np.ndarray:
        """Read one chunk of audio from the microphone.

        Returns float32 array of shape ``(chunk_samples,)`` at ``sample_rate``.
        Raises RuntimeError if mic is not open.
        """
        if self._stream is None:
            raise RuntimeError("MicInput not open — use 'with mic.open():'")

        # Get raw audio from callback queue (blocks until available)
        try:
            raw = self._audio_queue.get(timeout=2.0)
        except queue.Empty:
            # Return silence if no data (shouldn't happen in normal operation)
            logger.warning("MicInput: no audio data received (timeout)")
            if self._needs_resample:
                return np.zeros(self._chunk_samples, dtype=np.float32)
            return np.zeros(self._chunk_samples, dtype=np.float32)

        if self._needs_resample:
            return self._process_chunk(raw)
        else:
            return raw.reshape(-1)

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
        return (samples * 32768).clip(-32768, 32767).astype(np.int16)

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

        # Resampling pipeline config
        native_rate_raw = voice_cfg.get("mic_native_rate", None)
        native_rate = int(native_rate_raw) if native_rate_raw is not None else None
        mic_channels = int(voice_cfg.get("mic_channels", 1))
        channel_select = int(voice_cfg.get("mic_channel_select", 0))
        highpass_hz = int(voice_cfg.get("mic_highpass_hz", 80))
        agc_target = float(voice_cfg.get("mic_agc_target_rms", 0.15))

        return cls(
            device=device,
            sample_rate=sample_rate,
            audio_router=audio_router,
            mic_native_rate=native_rate,
            mic_channels=mic_channels,
            mic_channel_select=channel_select,
            mic_highpass_hz=highpass_hz,
            mic_agc_target_rms=agc_target,
        )
