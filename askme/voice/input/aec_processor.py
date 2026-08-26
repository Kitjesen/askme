"""Echo-cancellation seam for the local full-duplex voice runtime.

Only the optional ``_askme_webrtc_apm`` extension is treated as acoustic echo
cancellation.  The fallback adapter is intentionally marked degraded and only
passes valid PCM through, so callers can fail closed instead of mistaking an
echo gate for AEC.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

Pcm16Frame: TypeAlias = NDArray[np.int16]

_NATIVE_MODULE = "_askme_webrtc_apm"
_NATIVE_SAMPLE_RATES = frozenset({8_000, 16_000, 32_000, 48_000})
_SUPPORTED_CHANNELS = frozenset({1, 2})


class AecUnavailableError(RuntimeError):
    """Raised when the runtime requires AEC but no working backend exists."""


class AecFrameError(ValueError):
    """Raised when PCM does not satisfy the processor's 10 ms contract."""


def _validate_configuration(*, sample_rate_hz: int, channels: int) -> None:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz not in _NATIVE_SAMPLE_RATES
    ):
        supported = ", ".join(str(rate) for rate in sorted(_NATIVE_SAMPLE_RATES))
        raise ValueError(f"AEC sample_rate_hz must be one of {supported}")
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels not in _SUPPORTED_CHANNELS
    ):
        raise ValueError("AEC channels must be 1 (mono) or 2 (stereo)")


def _validate_frame(frame: Pcm16Frame, *, expected_samples: int) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.dtype(np.int16):
        raise AecFrameError("AEC frames must be NumPy arrays of signed 16-bit PCM")
    if frame.ndim != 1:
        raise AecFrameError("AEC frames must be one-dimensional interleaved PCM")
    if not frame.flags.c_contiguous:
        raise AecFrameError("AEC frames must be C-contiguous")
    if frame.size != expected_samples:
        raise AecFrameError(
            f"AEC requires exactly one 10 ms frame ({expected_samples} interleaved samples); "
            f"received {frame.size}"
        )


def _validate_delay_ms(delay_ms: int) -> None:
    if isinstance(delay_ms, bool) or not isinstance(delay_ms, int):
        raise ValueError("AEC render delay must be a non-negative integer")
    if delay_ms < 0:
        raise ValueError("AEC render delay must be non-negative")


@dataclass(frozen=True, slots=True)
class AecStats:
    """Observable AEC health and WebRTC metrics."""

    available: bool
    active: bool
    degraded: bool
    backend: str
    reason: str | None = None
    render_frames: int = 0
    capture_frames: int = 0
    delay_ms: int = 0
    echo_return_loss_db: float | None = None
    echo_return_loss_enhancement_db: float | None = None
    residual_echo_likelihood: float | None = None


class AecProcessor(Protocol):
    """Interface shared by native and explicitly degraded AEC adapters.

    Both methods accept exactly one interleaved signed-16-bit PCM frame whose
    duration is 10 ms at the configured sample rate and channel count.
    """

    def process_render(self, frame: Pcm16Frame) -> None:
        """Feed the final speaker PCM frame to the echo-reference path."""

    def process_capture(self, frame: Pcm16Frame, *, delay_ms: int) -> Pcm16Frame:
        """Cancel render echo from one microphone frame."""

    def stats(self) -> AecStats:
        """Return current adapter health and processing counters."""

    def reset(self) -> None:
        """Reset signal history and counters."""


class _NativeBackend(Protocol):
    def process_render(self, frame: Pcm16Frame) -> None: ...

    def process_capture(self, frame: Pcm16Frame, delay_ms: int) -> Pcm16Frame: ...

    def stats(self) -> Mapping[str, object]: ...

    def reset(self) -> None: ...


class NativeAecProcessor:
    """Validated Python adapter around the pinned WebRTC APM extension."""

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        channels: int,
        backend: _NativeBackend,
    ) -> None:
        _validate_configuration(sample_rate_hz=sample_rate_hz, channels=channels)
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._frame_samples = sample_rate_hz // 100 * channels
        self._backend = backend
        self._render_frames = 0
        self._capture_frames = 0
        self._delay_ms = 0

    def process_render(self, frame: Pcm16Frame) -> None:
        _validate_frame(frame, expected_samples=self._frame_samples)
        self._backend.process_render(frame)
        self._render_frames += 1

    def process_capture(self, frame: Pcm16Frame, *, delay_ms: int) -> Pcm16Frame:
        _validate_frame(frame, expected_samples=self._frame_samples)
        _validate_delay_ms(delay_ms)
        output = self._backend.process_capture(frame, delay_ms)
        if not isinstance(output, np.ndarray):
            raise AecFrameError("native AEC returned a non-NumPy capture frame")
        _validate_frame(output, expected_samples=self._frame_samples)
        self._capture_frames += 1
        self._delay_ms = delay_ms
        return output

    def stats(self) -> AecStats:
        raw = self._backend.stats()
        return AecStats(
            available=True,
            active=True,
            degraded=False,
            backend="webrtc-apm-v2.1",
            render_frames=self._render_frames,
            capture_frames=self._capture_frames,
            delay_ms=self._delay_ms,
            echo_return_loss_db=_optional_float(raw, "echo_return_loss_db"),
            echo_return_loss_enhancement_db=_optional_float(raw, "echo_return_loss_enhancement_db"),
            residual_echo_likelihood=_optional_float(raw, "residual_echo_likelihood"),
        )

    def reset(self) -> None:
        self._backend.reset()
        self._render_frames = 0
        self._capture_frames = 0
        self._delay_ms = 0


class UnavailableAecProcessor:
    """Explicit degraded adapter used only when AEC is optional."""

    def __init__(self, *, sample_rate_hz: int, channels: int, reason: str) -> None:
        _validate_configuration(sample_rate_hz=sample_rate_hz, channels=channels)
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._reason = reason
        self._frame_samples = sample_rate_hz // 100 * channels
        self._render_frames = 0
        self._capture_frames = 0
        self._delay_ms = 0

    def process_render(self, frame: Pcm16Frame) -> None:
        _validate_frame(frame, expected_samples=self._frame_samples)
        self._render_frames += 1

    def process_capture(self, frame: Pcm16Frame, *, delay_ms: int) -> Pcm16Frame:
        _validate_frame(frame, expected_samples=self._frame_samples)
        _validate_delay_ms(delay_ms)
        self._capture_frames += 1
        self._delay_ms = delay_ms
        return frame

    def stats(self) -> AecStats:
        return AecStats(
            available=False,
            active=False,
            degraded=True,
            backend="unavailable",
            reason=self._reason,
            render_frames=self._render_frames,
            capture_frames=self._capture_frames,
            delay_ms=self._delay_ms,
        )

    def reset(self) -> None:
        self._render_frames = 0
        self._capture_frames = 0
        self._delay_ms = 0


def _optional_float(values: Mapping[str, object], key: str) -> float | None:
    value = values.get(key)
    return None if value is None else float(cast(Any, value))


def _load_native_extension() -> ModuleType:
    return importlib.import_module(_NATIVE_MODULE)


def _unavailable(*, sample_rate_hz: int, channels: int, reason: str) -> AecProcessor:
    return UnavailableAecProcessor(
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        reason=reason,
    )


def create_aec_processor(
    *,
    sample_rate_hz: int = 48_000,
    channels: int = 1,
    required: bool = False,
) -> AecProcessor:
    """Create native WebRTC APM or an explicitly degraded pass-through.

    ``required=True`` is the fail-closed production mode: import, ABI, or
    initialization failures raise :class:`AecUnavailableError`.  Optional mode
    returns a degraded adapter carrying the exact reason and never reports AEC
    as active.
    """
    _validate_configuration(sample_rate_hz=sample_rate_hz, channels=channels)
    try:
        extension = _load_native_extension()
        backend_type = getattr(extension, "AudioProcessing")
        backend = backend_type(sample_rate_hz, channels)
        return NativeAecProcessor(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            backend=cast(_NativeBackend, backend),
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        reason = f"askme-webrtc-apm native extension ({_NATIVE_MODULE}) is unavailable: {exc}"
        if required:
            raise AecUnavailableError(reason) from exc
        return _unavailable(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            reason=reason,
        )
