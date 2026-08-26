"""Software acoustic echo cancellation for the local microphone path.

The board has no hardware AEC. This module uses the small SpeexDSP C API
already shipped by Ubuntu and keeps a time-based reference of the PCM sent to
the speaker. The reference is deliberately separated from the capture thread:
TTS and microphone callbacks run on different threads.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_I16_PTR = ctypes.POINTER(ctypes.c_int16)


class PlaybackReference:
    """Timestamped mono speaker reference consumed by the AEC."""

    def __init__(self, sample_rate: int, *, history_seconds: float = 4.0) -> None:
        self.sample_rate = int(sample_rate)
        self._history_seconds = max(1.0, float(history_seconds))
        self._segments: deque[tuple[float, np.ndarray]] = deque()
        self._timeline_end = 0.0
        self._lock = threading.Lock()

    def push_playback(self, samples: np.ndarray, sample_rate: int) -> None:
        """Publish PCM that has just been handed to the speaker transport."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(data) == 0:
            return
        source_rate = int(sample_rate)
        if source_rate != self.sample_rate:
            target_len = max(1, int(round(len(data) * self.sample_rate / source_rate)))
            positions = np.linspace(0.0, len(data) - 1, target_len, dtype=np.float64)
            data = np.interp(positions, np.arange(len(data)), data).astype(np.float32)
        else:
            data = data.copy()

        now = time.monotonic()
        duration = len(data) / float(self.sample_rate)
        with self._lock:
            start = max(now, self._timeline_end)
            self._segments.append((start, data))
            self._timeline_end = start + duration
            cutoff = now - self._history_seconds
            while self._segments:
                seg_start, seg = self._segments[0]
                if seg_start + len(seg) / self.sample_rate >= cutoff:
                    break
                self._segments.popleft()

    def read(self, start_time: float, count: int) -> np.ndarray:
        """Read a reference interval; gaps are silence."""
        out = np.zeros(max(0, int(count)), dtype=np.float32)
        if len(out) == 0:
            return out
        end_time = start_time + len(out) / float(self.sample_rate)
        with self._lock:
            for seg_start, seg in self._segments:
                seg_end = seg_start + len(seg) / float(self.sample_rate)
                overlap_start = max(start_time, seg_start)
                overlap_end = min(end_time, seg_end)
                if overlap_end <= overlap_start:
                    continue
                out_start = int(round((overlap_start - start_time) * self.sample_rate))
                out_end = min(len(out), int(round((overlap_end - start_time) * self.sample_rate)))
                seg_start_i = int(round((overlap_start - seg_start) * self.sample_rate))
                seg_end_i = seg_start_i + (out_end - out_start)
                if out_end > out_start and seg_end_i > seg_start_i:
                    out[out_start:out_end] = seg[seg_start_i:seg_end_i]
        return out


class SpeexEchoCanceller:
    """ctypes wrapper around SpeexDSP's frame-based acoustic echo canceller."""

    def __init__(self, config: dict[str, Any] | None, sample_rate: int) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.sample_rate = int(sample_rate)
        self.reference: PlaybackReference | None = None
        self._lib: Any | None = None
        self._state: Any | None = None
        self._frame_size = max(80, int(round(self.sample_rate * float(cfg.get("frame_ms", 10)) / 1000.0)))
        filter_ms = max(100.0, float(cfg.get("filter_length_ms", 500.0)))
        self._filter_length = max(self._frame_size, int(round(self.sample_rate * filter_ms / 1000.0)))
        self._delay_seconds = max(0.0, float(cfg.get("playback_delay_ms", 250.0)) / 1000.0)
        if not self.enabled:
            return

        library_name = ctypes.util.find_library("speexdsp") or "libspeexdsp.so.1"
        try:
            lib = ctypes.CDLL(library_name)
            lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.speex_echo_state_init.restype = ctypes.c_void_p
            lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
            lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            lib.speex_echo_ctl.restype = ctypes.c_int
            lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p, _I16_PTR, _I16_PTR, _I16_PTR]
            lib.speex_echo_cancellation.restype = None
            state = lib.speex_echo_state_init(self._frame_size, self._filter_length)
            if not state:
                raise RuntimeError("speex_echo_state_init returned null")
            rate = ctypes.c_int(self.sample_rate)
            if lib.speex_echo_ctl(state, 24, ctypes.byref(rate)) != 0:
                raise RuntimeError("speex_echo_ctl(SPEEX_ECHO_SET_SAMPLING_RATE) failed")
            self._lib = lib
            self._state = state
            self.reference = PlaybackReference(self.sample_rate)
            logger.info(
                "Software AEC enabled: SpeexDSP frame=%d samples filter=%d samples delay=%.0fms",
                self._frame_size, self._filter_length, self._delay_seconds * 1000.0,
            )
        except Exception as exc:
            self.enabled = False
            self._lib = None
            self._state = None
            logger.warning("Software AEC unavailable; continuing without it: %s", exc)

    def process(self, captured: np.ndarray, capture_time: float | None = None) -> np.ndarray:
        """Remove the speaker reference from one arbitrary-sized mic chunk."""
        data = np.asarray(captured, dtype=np.int16).reshape(-1)
        if not self.enabled or self._lib is None or self._state is None or self.reference is None:
            return data
        if len(data) == 0:
            return data
        when = time.monotonic() if capture_time is None else float(capture_time)
        reference = self.reference.read(when - self._delay_seconds, len(data))
        reference_i16 = np.clip(reference * 32767.0, -32768, 32767).astype(np.int16)
        out = np.empty_like(data)
        for start in range(0, len(data), self._frame_size):
            end = min(len(data), start + self._frame_size)
            length = end - start
            if length < self._frame_size:
                rec_frame = np.zeros(self._frame_size, dtype=np.int16)
                play_frame = np.zeros(self._frame_size, dtype=np.int16)
                rec_frame[:length] = data[start:end]
                play_frame[:length] = reference_i16[start:end]
                out_frame = np.empty(self._frame_size, dtype=np.int16)
                self._cancel_frame(rec_frame, play_frame, out_frame)
                out[start:end] = out_frame[:length]
            else:
                self._cancel_frame(
                    np.ascontiguousarray(data[start:end]),
                    np.ascontiguousarray(reference_i16[start:end]),
                    out[start:end],
                )
        return out

    def _cancel_frame(self, rec: np.ndarray, play: np.ndarray, out: np.ndarray) -> None:
        self._lib.speex_echo_cancellation(
            self._state, rec.ctypes.data_as(_I16_PTR), play.ctypes.data_as(_I16_PTR), out.ctypes.data_as(_I16_PTR)
        )

    def close(self) -> None:
        if self._lib is not None and self._state is not None:
            self._lib.speex_echo_state_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
