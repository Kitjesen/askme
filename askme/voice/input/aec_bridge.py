"""Adapt arbitrary mono float32 chunks to WebRTC APM's 10 ms PCM16 API."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from askme.voice.input.aec_processor import AecProcessor

Float32Audio: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AecBridgeStats:
    render_frames: int
    capture_frames: int
    render_tail_samples: int


class AecPcmBridge:
    """Feed final render PCM and clean capture PCM through one AEC instance."""

    def __init__(
        self,
        processor: AecProcessor,
        *,
        sample_rate_hz: int,
        delay_ms: int = 40,
    ) -> None:
        if sample_rate_hz <= 0 or sample_rate_hz % 100:
            raise ValueError("AEC sample rate must provide an integral 10 ms frame")
        if delay_ms < 0:
            raise ValueError("AEC render delay must be non-negative")
        self._processor = processor
        self._sample_rate_hz = sample_rate_hz
        self._frame_samples = sample_rate_hz // 100
        self._delay_ms = delay_ms
        self._render_tail: NDArray[np.int16] = np.empty(0, dtype=np.int16)
        self._lock = threading.RLock()
        self._render_frames = 0
        self._capture_frames = 0

    def feed_render(self, samples: Float32Audio, *, sample_rate_hz: int) -> None:
        """Feed the exact mono PCM about to be written to the speaker."""

        target = self._prepare_pcm16(samples, sample_rate_hz=sample_rate_hz)
        with self._lock:
            if self._render_tail.size:
                target = np.concatenate((self._render_tail, target))
            frame_count = target.size // self._frame_samples
            used = frame_count * self._frame_samples
            for offset in range(0, used, self._frame_samples):
                frame = np.ascontiguousarray(
                    target[offset : offset + self._frame_samples],
                    dtype=np.int16,
                )
                self._processor.process_render(frame)
                self._render_frames += 1
            self._render_tail = target[used:].copy()

    def process_capture(
        self,
        samples: Float32Audio,
        *,
        sample_rate_hz: int,
    ) -> Float32Audio:
        """Cancel echo while preserving the caller's sample count and rate."""

        original = self._as_float32_mono(samples)
        target = self._prepare_pcm16(original, sample_rate_hz=sample_rate_hz)
        cleaned = target.copy()
        with self._lock:
            used = (target.size // self._frame_samples) * self._frame_samples
            for offset in range(0, used, self._frame_samples):
                frame = np.ascontiguousarray(
                    target[offset : offset + self._frame_samples],
                    dtype=np.int16,
                )
                processed = self._processor.process_capture(
                    frame,
                    delay_ms=self._delay_ms,
                )
                cleaned[offset : offset + self._frame_samples] = processed
                self._capture_frames += 1

        cleaned_float = cleaned.astype(np.float32) / 32768.0
        if sample_rate_hz != self._sample_rate_hz:
            cleaned_float = self._resample(
                cleaned_float,
                self._sample_rate_hz,
                sample_rate_hz,
                output_samples=original.size,
            )
        elif cleaned_float.size != original.size:
            cleaned_float = self._resample(
                cleaned_float,
                self._sample_rate_hz,
                sample_rate_hz,
                output_samples=original.size,
            )
        return np.ascontiguousarray(cleaned_float, dtype=np.float32)

    def stats(self) -> AecBridgeStats:
        with self._lock:
            return AecBridgeStats(
                render_frames=self._render_frames,
                capture_frames=self._capture_frames,
                render_tail_samples=int(self._render_tail.size),
            )

    def reset(self) -> None:
        with self._lock:
            self._processor.reset()
            self._render_tail = np.empty(0, dtype=np.int16)
            self._render_frames = 0
            self._capture_frames = 0

    def _prepare_pcm16(
        self,
        samples: Float32Audio,
        *,
        sample_rate_hz: int,
    ) -> NDArray[np.int16]:
        mono = self._as_float32_mono(samples)
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if sample_rate_hz != self._sample_rate_hz:
            mono = self._resample(mono, sample_rate_hz, self._sample_rate_hz)
        return (mono * 32767.0).clip(-32768, 32767).astype(np.int16)

    @staticmethod
    def _as_float32_mono(samples: Float32Audio) -> Float32Audio:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("AEC bridge requires one-dimensional mono audio")
        return np.ascontiguousarray(array)

    @staticmethod
    def _resample(
        samples: Float32Audio,
        source_rate: int,
        target_rate: int,
        *,
        output_samples: int | None = None,
    ) -> Float32Audio:
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        target_count = output_samples
        if target_count is None:
            target_count = max(1, round(samples.size * target_rate / source_rate))
        if target_count == samples.size and source_rate == target_rate:
            return samples.copy()
        source_axis = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target_axis = np.linspace(0.0, 1.0, target_count, endpoint=False)
        return np.interp(target_axis, source_axis, samples).astype(np.float32)
