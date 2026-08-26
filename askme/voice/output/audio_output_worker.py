"""Resident, native-format audio output for low-latency voice turns.

The worker owns one output adapter for its whole lifetime.  Callers submit
floating-point PCM and do not need to know about resampling, channel layout,
period sizing, or the output stream lifecycle.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class AudioOutputAdapter(Protocol):
    """Adapter used by :class:`AudioOutputWorker` to reach the audio sink."""

    def open(self) -> None:
        """Open and configure the native output stream."""

    def write(self, pcm: bytes) -> None:
        """Write interleaved signed 16-bit PCM to the stream."""

    def drop(self) -> None:
        """Discard buffered output immediately."""

    def close(self) -> None:
        """Close the output stream."""


@dataclass(frozen=True, slots=True)
class AudioOutputConfig:
    """Native stream and speech-boundary settings."""

    native_sample_rate: int = 48_000
    channels: int = 2
    period_ms: int = 10
    idle_keepalive: bool = False
    warm_hold_seconds: float = 45.0
    idle_dither_lsb: int = 0
    buffer_ms: int = 80
    cold_preroll_ms: int = 0
    warm_leadin_ms: int = 0

    def __post_init__(self) -> None:
        if self.native_sample_rate <= 0:
            raise ValueError("native_sample_rate must be positive")
        if self.channels not in {1, 2}:
            raise ValueError("channels must be 1 or 2")
        if self.period_ms <= 0:
            raise ValueError("period_ms must be positive")
        if self.buffer_ms < self.period_ms:
            raise ValueError("buffer_ms must be greater than or equal to period_ms")
        if self.warm_hold_seconds < 0:
            raise ValueError("warm_hold_seconds must not be negative")
        if not 0 <= self.idle_dither_lsb <= 16:
            raise ValueError("idle_dither_lsb must be between 0 and 16")
        if self.cold_preroll_ms < 0:
            raise ValueError("cold_preroll_ms must not be negative")
        if self.warm_leadin_ms < 0:
            raise ValueError("warm_leadin_ms must not be negative")


class AplayOutputAdapter:
    """Low-buffer native ALSA adapter backed by one resident ``aplay`` process."""

    def __init__(
        self,
        *,
        device: str | None,
        config: AudioOutputConfig,
        executable: str = "aplay",
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._device = device
        self._config = config
        self._executable = executable
        self._process_factory = process_factory or subprocess.Popen
        self._process: Any | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            command = [
                self._executable,
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(self._config.native_sample_rate),
                "-c",
                str(self._config.channels),
                f"--period-time={self._config.period_ms * 1000}",
                f"--buffer-time={self._config.buffer_ms * 1000}",
            ]
            if self._device:
                command.extend(["-D", self._device])
            process = self._process_factory(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            if process.stdin is None:
                process.terminate()
                raise RuntimeError("aplay stdin is unavailable")
            self._process = process

    def write(self, pcm: bytes) -> None:
        with self._lock:
            process = self._process
            if process is None:
                raise BrokenPipeError("aplay output is not open")
            status = process.poll()
            if status is not None:
                self._process = None
                raise BrokenPipeError(f"aplay exited with status {status}")
            stream = process.stdin
        if stream is None:
            raise BrokenPipeError("aplay stdin is unavailable")
        stream.write(pcm)
        stream.flush()

    def drop(self) -> None:
        process = self._detach_process()
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.25)

    def close(self) -> None:
        process = self._detach_process()
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.25)

    def _detach_process(self) -> Any | None:
        with self._lock:
            process = self._process
            self._process = None
            return process


class PlaybackTicket:
    """Completion handle returned for one submitted utterance."""

    def __init__(self, generation: int) -> None:
        self.generation = int(generation)
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._cancelled = False

    def wait(self, timeout: float | None = None) -> bool:
        """Return whether the utterance completed before *timeout*."""

        return self._done.wait(timeout)

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _finish(
        self, error: BaseException | None = None, *, cancelled: bool = False
    ) -> None:
        if self._done.is_set():
            return
        self._error = error
        self._cancelled = cancelled
        self._done.set()


@dataclass(slots=True)
class _Submission:
    samples: np.ndarray
    sample_rate: int
    ticket: PlaybackTicket


@dataclass(frozen=True, slots=True)
class _WarmRequest:
    seconds: float


_STOP = object()


class AudioOutputWorker:
    """Serialize utterances through one long-lived native output stream."""

    def __init__(
        self,
        adapter: AudioOutputAdapter,
        *,
        config: AudioOutputConfig | None = None,
        render_reference: Callable[[np.ndarray, int], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._config = config or AudioOutputConfig()
        self._render_reference = render_reference
        self._queue: queue.Queue[_Submission | _WarmRequest | object] = queue.Queue()
        self._closed = threading.Event()
        self._state_lock = threading.Lock()
        self._cancelled_generations: set[int] = set()
        self._cancel_errors: dict[int, BaseException] = {}
        self._pending_generations: dict[int, int] = {}
        self._active_generation: int | None = None
        self._active_ticket: PlaybackTicket | None = None
        self._stream_open = False
        self._warm_until = 0.0
        self._last_error: str | None = None
        self._dither_rng = np.random.default_rng(0)
        self._thread = threading.Thread(
            target=self._run,
            name="askme-audio-output",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        generation: int,
    ) -> PlaybackTicket:
        """Queue one utterance and return its completion handle."""

        if self._closed.is_set():
            raise RuntimeError("audio output worker is shut down")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim not in {1, 2}:
            raise ValueError("samples must be mono or channel-interleaved PCM")
        if audio.ndim == 2 and audio.shape[1] not in {1, 2}:
            raise ValueError("samples must have one or two channels")

        ticket = PlaybackTicket(generation)
        with self._state_lock:
            self._pending_generations[ticket.generation] = (
                self._pending_generations.get(ticket.generation, 0) + 1
            )
        self._queue.put(
            _Submission(
                samples=audio.copy(),
                sample_rate=int(sample_rate),
                ticket=ticket,
            )
        )
        return ticket

    def warm_for(self, seconds: float | None = None) -> None:
        """Open the stream early and keep it warm for a bounded interval."""

        if self._closed.is_set():
            raise RuntimeError("audio output worker is shut down")
        duration = (
            self._config.warm_hold_seconds if seconds is None else float(seconds)
        )
        if duration < 0:
            raise ValueError("warm duration must not be negative")
        self._queue.put(_WarmRequest(duration))

    def cancel(self, *, generation: int) -> bool:
        """Cancel pending work and drop the stream when that work is active."""

        target = int(generation)
        with self._state_lock:
            pending = self._pending_generations.get(target, 0) > 0
            active = self._active_generation == target
            if not pending and not active:
                return False
            self._cancelled_generations.add(target)
            self._warm_until = 0.0

        if active:
            self._drop_stream()
        return True

    def _is_cancelled(self, generation: int) -> bool:
        with self._state_lock:
            return generation in self._cancelled_generations

    def _finish_cancelled(self, ticket: PlaybackTicket) -> None:
        with self._state_lock:
            error = self._cancel_errors.get(ticket.generation)
        ticket._finish(error, cancelled=True)

    def status_snapshot(self) -> dict[str, Any]:
        """Return non-secret stream state for health and latency telemetry."""

        now = time.monotonic()
        with self._state_lock:
            return {
                "stream_open": self._stream_open,
                "active_generation": self._active_generation,
                "queued": sum(self._pending_generations.values()),
                "idle_keepalive": self._config.idle_keepalive,
                "native_sample_rate": self._config.native_sample_rate,
                "channels": self._config.channels,
                "period_ms": self._config.period_ms,
                "buffer_ms": self._config.buffer_ms,
                "warm_remaining_ms": max(
                    0, round((self._warm_until - now) * 1000)
                ),
                "last_error": self._last_error,
                "thread_alive": self._thread.is_alive(),
            }

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the worker and force-drop a blocked device after *timeout*."""

        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._drop_stream()
            self._thread.join(timeout=0.25)

    def _run(self) -> None:
        try:
            while True:
                timeout = self._queue_timeout()
                try:
                    if timeout is None:
                        item = self._queue.get()
                    else:
                        item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    self._service_idle_window()
                    continue
                try:
                    if item is _STOP:
                        return
                    if isinstance(item, _WarmRequest):
                        self._process_warm_request(item)
                    else:
                        assert isinstance(item, _Submission)
                        self._process_submission(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._record_error(exc)
            self._fail_pending(exc)
        finally:
            self._close_stream()

    def _queue_timeout(self) -> float | None:
        now = time.monotonic()
        with self._state_lock:
            if not self._stream_open:
                return None
            remaining = self._warm_until - now
        if remaining <= 0:
            return 0.0
        if self._config.idle_keepalive:
            return min(remaining, self._config.period_ms / 1000.0)
        return remaining

    def _service_idle_window(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            should_close = self._stream_open and now >= self._warm_until
            should_feed = (
                self._stream_open
                and self._config.idle_keepalive
                and not should_close
            )
        if should_close:
            self._close_stream()
            return
        if should_feed:
            try:
                self._write_idle_period()
            except BaseException as exc:
                self._record_error(exc)
                self._drop_stream()

    def _process_warm_request(self, item: _WarmRequest) -> None:
        try:
            opened = self._open_stream()
            if opened:
                self._write_padding(self._config.cold_preroll_ms)
            self._extend_warm_window(item.seconds)
            self._clear_error()
        except BaseException as exc:
            self._record_error(exc)
            self._drop_stream()

    def _process_submission(self, item: _Submission) -> None:
        generation = item.ticket.generation
        if self._is_cancelled(generation):
            self._finish_cancelled(item.ticket)
            self._complete_generation(generation)
            return

        with self._state_lock:
            self._active_generation = generation
            self._active_ticket = item.ticket
        try:
            opened = self._open_stream()
            if self._is_cancelled(generation):
                self._drop_stream()
                self._finish_cancelled(item.ticket)
                return
            if opened:
                self._write_padding(
                    self._config.cold_preroll_ms,
                    generation=generation,
                )
            self._write_padding(
                self._config.warm_leadin_ms,
                generation=generation,
            )
            self._write_submission(item)
            if not item.ticket.cancelled and item.ticket.error is None:
                self._extend_warm_window(self._config.warm_hold_seconds)
                self._clear_error()
        except BaseException as exc:
            self._record_error(exc)
            self._drop_stream()
            if self._is_cancelled(generation):
                self._finish_cancelled(item.ticket)
            else:
                item.ticket._finish(exc)
        finally:
            with self._state_lock:
                self._active_generation = None
                if self._active_ticket is item.ticket:
                    self._active_ticket = None
            self._complete_generation(generation)

    def _complete_generation(self, generation: int) -> None:
        with self._state_lock:
            remaining = self._pending_generations.get(generation, 0) - 1
            if remaining > 0:
                self._pending_generations[generation] = remaining
            else:
                self._pending_generations.pop(generation, None)
                self._cancelled_generations.discard(generation)
                self._cancel_errors.pop(generation, None)

    def _extend_warm_window(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        with self._state_lock:
            self._warm_until = max(self._warm_until, deadline)

    def _open_stream(self) -> bool:
        with self._state_lock:
            if self._stream_open:
                return False
        self._adapter.open()
        with self._state_lock:
            self._stream_open = True
        return True

    def _drop_stream(self) -> None:
        with self._state_lock:
            self._stream_open = False
            self._warm_until = 0.0
            active_generation = self._active_generation
            active_ticket = self._active_ticket
        try:
            self._adapter.drop()
        except Exception as exc:
            self._record_error(exc)
            if active_generation is not None:
                with self._state_lock:
                    self._cancelled_generations.add(active_generation)
                    self._cancel_errors[active_generation] = exc
                if active_ticket is not None:
                    active_ticket._finish(exc, cancelled=True)

    def _close_stream(self) -> None:
        with self._state_lock:
            stream_open = self._stream_open
            self._stream_open = False
            self._warm_until = 0.0
        if not stream_open:
            return
        try:
            self._adapter.close()
        except Exception as exc:
            self._record_error(exc)

    def _write_idle_period(self) -> None:
        self._write_pcm(self._dither_pcm(self._config.period_ms))

    def _write_padding(
        self,
        milliseconds: int,
        *,
        generation: int | None = None,
    ) -> None:
        remaining = int(milliseconds)
        while remaining > 0:
            if generation is not None and self._is_cancelled(generation):
                raise InterruptedError("audio generation cancelled")
            duration = min(remaining, self._config.period_ms)
            self._write_pcm(self._dither_pcm(duration))
            remaining -= duration

    def _dither_pcm(self, milliseconds: int) -> bytes:
        frames = max(
            1,
            round(self._config.native_sample_rate * milliseconds / 1000),
        )
        sample_count = frames * self._config.channels
        amplitude = self._config.idle_dither_lsb
        if amplitude <= 0:
            return bytes(sample_count * np.dtype(np.int16).itemsize)
        samples = self._dither_rng.integers(
            -amplitude,
            amplitude + 1,
            size=sample_count,
            dtype=np.int16,
        )
        return samples.astype("<i2", copy=False).tobytes()

    def _write_pcm(self, pcm: bytes) -> None:
        if self._render_reference is not None:
            interleaved = np.frombuffer(pcm, dtype="<i2").reshape(
                -1, self._config.channels
            )
            mono = interleaved.astype(np.float32).mean(axis=1) / 32767.0
            try:
                self._render_reference(
                    mono,
                    self._config.native_sample_rate,
                )
            except Exception:
                # AEC telemetry must never make physical playback fail.
                pass
        self._adapter.write(pcm)

    def _write_submission(self, item: _Submission) -> None:
        pcm = self._native_pcm(item.samples, item.sample_rate)
        period_frames = max(
            1,
            round(
                self._config.native_sample_rate
                * self._config.period_ms
                / 1000
            ),
        )
        frame_bytes = self._config.channels * np.dtype(np.int16).itemsize
        period_bytes = period_frames * frame_bytes
        for offset in range(0, len(pcm), period_bytes):
            if self._is_cancelled(item.ticket.generation):
                self._finish_cancelled(item.ticket)
                return
            self._write_pcm(pcm[offset : offset + period_bytes])
        item.ticket._finish()

    def _native_pcm(self, samples: np.ndarray, sample_rate: int) -> bytes:
        channels = self._normalise_channels(samples)
        if sample_rate != self._config.native_sample_rate:
            channels = self._resample(channels, sample_rate)

        clipped = np.clip(channels, -1.0, 1.0)
        pcm = np.rint(clipped * 32767.0).astype("<i2", copy=False)
        return pcm.tobytes(order="C")

    def _normalise_channels(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            mono = samples[:, np.newaxis]
        else:
            mono = samples

        if mono.shape[1] == self._config.channels:
            return mono
        if self._config.channels == 2 and mono.shape[1] == 1:
            return np.repeat(mono, 2, axis=1)
        return mono.mean(axis=1, keepdims=True)

    def _resample(self, samples: np.ndarray, source_rate: int) -> np.ndarray:
        source_frames = len(samples)
        if source_frames == 0:
            return samples.copy()
        target_frames = max(
            1,
            round(source_frames * self._config.native_sample_rate / source_rate),
        )
        source_positions = np.arange(source_frames, dtype=np.float64)
        target_positions = np.linspace(
            0.0,
            float(source_frames - 1),
            target_frames,
            dtype=np.float64,
        )
        result = np.empty((target_frames, samples.shape[1]), dtype=np.float32)
        for channel in range(samples.shape[1]):
            result[:, channel] = np.interp(
                target_positions,
                source_positions,
                samples[:, channel],
                right=float(samples[-1, channel]),
            )
        return result

    def _record_error(self, error: BaseException) -> None:
        with self._state_lock:
            self._last_error = f"{type(error).__name__}: {error}"

    def _clear_error(self) -> None:
        with self._state_lock:
            self._last_error = None

    def _fail_pending(self, error: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if isinstance(item, _Submission):
                    item.ticket._finish(error)
                    self._complete_generation(item.ticket.generation)
            finally:
                self._queue.task_done()
