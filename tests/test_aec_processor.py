from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from askme.voice.input import aec_processor
from askme.voice.input.aec_processor import (
    AecFrameError,
    AecUnavailableError,
    create_aec_processor,
)


@pytest.fixture(autouse=True)
def native_extension_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_extension() -> None:
        raise ModuleNotFoundError("test extension is not installed")

    monkeypatch.setattr(aec_processor, "_load_native_extension", missing_extension)


def test_optional_aec_is_explicitly_degraded_without_native_extension() -> None:
    processor = create_aec_processor(sample_rate_hz=16_000, channels=1)

    status = processor.stats()

    assert status.available is False
    assert status.active is False
    assert status.degraded is True
    assert status.backend == "unavailable"
    assert "test extension is not installed" in (status.reason or "")


def test_required_aec_fails_closed_without_native_extension() -> None:
    with pytest.raises(AecUnavailableError, match="test extension is not installed"):
        create_aec_processor(sample_rate_hz=16_000, channels=1, required=True)


def test_degraded_adapter_preserves_10ms_capture_and_reports_processing() -> None:
    processor = create_aec_processor(sample_rate_hz=16_000, channels=1)
    render = np.arange(160, dtype=np.int16)
    capture = np.arange(160, dtype=np.int16) * -1

    processor.process_render(render)
    output = processor.process_capture(capture, delay_ms=37)

    np.testing.assert_array_equal(output, capture)
    status = processor.stats()
    assert status.render_frames == 1
    assert status.capture_frames == 1
    assert status.delay_ms == 37


def test_processing_rejects_frames_that_are_not_exactly_10ms() -> None:
    processor = create_aec_processor(sample_rate_hz=16_000, channels=1)
    short_frame = np.zeros(159, dtype=np.int16)

    with pytest.raises(AecFrameError, match="10 ms"):
        processor.process_render(short_frame)
    with pytest.raises(AecFrameError, match="10 ms"):
        processor.process_capture(short_frame, delay_ms=0)


def test_processing_rejects_audio_that_is_not_interleaved_pcm16() -> None:
    processor = create_aec_processor(sample_rate_hz=16_000, channels=1)

    with pytest.raises(AecFrameError, match="signed 16-bit"):
        processor.process_render(np.zeros(160, dtype=np.float32))
    with pytest.raises(AecFrameError, match="one-dimensional"):
        processor.process_capture(np.zeros((16, 10), dtype=np.int16), delay_ms=0)
    with pytest.raises(AecFrameError, match="C-contiguous"):
        processor.process_render(np.zeros(320, dtype=np.int16)[::2])


@pytest.mark.parametrize("delay_ms", [-1, 1.25, True])
def test_capture_rejects_invalid_render_delay(delay_ms: object) -> None:
    processor = create_aec_processor(sample_rate_hz=16_000, channels=1)

    with pytest.raises(ValueError, match="non-negative"):
        processor.process_capture(
            np.zeros(160, dtype=np.int16),
            delay_ms=delay_ms,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("sample_rate_hz", [0, 44_100, 96_000, True])
def test_factory_rejects_sample_rates_not_supported_by_int16_apm(
    sample_rate_hz: int,
) -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        create_aec_processor(sample_rate_hz=sample_rate_hz, channels=1)


@pytest.mark.parametrize("channels", [0, 3, True])
def test_factory_rejects_unsupported_channel_counts(channels: int) -> None:
    with pytest.raises(ValueError, match="channels"):
        create_aec_processor(sample_rate_hz=16_000, channels=channels)


class _FakeNativeBackend:
    def __init__(self, sample_rate_hz: int, channels: int) -> None:
        self.configuration = (sample_rate_hz, channels)
        self.render: np.ndarray | None = None
        self.capture_delay_ms: int | None = None
        self.reset_count = 0

    def process_render(self, frame: np.ndarray) -> None:
        self.render = frame.copy()

    def process_capture(self, frame: np.ndarray, delay_ms: int) -> np.ndarray:
        self.capture_delay_ms = delay_ms
        return np.negative(frame, dtype=np.int16)

    def stats(self) -> dict[str, float]:
        return {
            "echo_return_loss_db": 12.5,
            "echo_return_loss_enhancement_db": 24.0,
            "residual_echo_likelihood": 0.125,
        }

    def reset(self) -> None:
        self.reset_count += 1


def test_native_extension_is_wrapped_and_reports_real_aec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = SimpleNamespace(AudioProcessing=_FakeNativeBackend)
    monkeypatch.setattr(aec_processor, "_load_native_extension", lambda: extension)

    processor = create_aec_processor(sample_rate_hz=16_000, channels=1, required=True)
    render = np.arange(160, dtype=np.int16)
    capture = np.arange(160, dtype=np.int16)

    processor.process_render(render)
    output = processor.process_capture(capture, delay_ms=28)

    np.testing.assert_array_equal(output, -capture)
    status = processor.stats()
    assert status.available is True
    assert status.active is True
    assert status.degraded is False
    assert status.backend == "webrtc-apm-v2.1"
    assert status.render_frames == 1
    assert status.capture_frames == 1
    assert status.delay_ms == 28
    assert status.echo_return_loss_db == 12.5
    assert status.echo_return_loss_enhancement_db == 24.0
    assert status.residual_echo_likelihood == 0.125

    processor.reset()
    reset_status = processor.stats()
    assert reset_status.render_frames == 0
    assert reset_status.capture_frames == 0


def test_native_initialization_failure_degrades_or_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenBackend:
        def __init__(self, sample_rate_hz: int, channels: int) -> None:
            raise RuntimeError(f"APM init failed for {sample_rate_hz}/{channels}")

    extension = SimpleNamespace(AudioProcessing=BrokenBackend)
    monkeypatch.setattr(aec_processor, "_load_native_extension", lambda: extension)

    optional = create_aec_processor(sample_rate_hz=48_000, channels=2)
    assert optional.stats().degraded is True
    assert "APM init failed" in (optional.stats().reason or "")

    with pytest.raises(AecUnavailableError, match="APM init failed"):
        create_aec_processor(sample_rate_hz=48_000, channels=2, required=True)


def test_native_adapter_rejects_invalid_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenOutputBackend(_FakeNativeBackend):
        def process_capture(self, frame: np.ndarray, delay_ms: int) -> np.ndarray:
            return np.zeros(frame.size - 1, dtype=np.int16)

    extension = SimpleNamespace(AudioProcessing=BrokenOutputBackend)
    monkeypatch.setattr(aec_processor, "_load_native_extension", lambda: extension)
    processor = create_aec_processor(sample_rate_hz=16_000, channels=1)

    with pytest.raises(AecFrameError, match="10 ms"):
        processor.process_capture(np.zeros(160, dtype=np.int16), delay_ms=0)
