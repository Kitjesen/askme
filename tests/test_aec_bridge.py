from __future__ import annotations

import numpy as np
import pytest

from askme.voice.input.aec_bridge import AecPcmBridge
from askme.voice.input.aec_processor import AecStats


class _FakeAec:
    def __init__(self) -> None:
        self.render_frames: list[np.ndarray] = []
        self.capture_frames: list[np.ndarray] = []

    def process_render(self, frame: np.ndarray) -> None:
        self.render_frames.append(frame.copy())

    def process_capture(self, frame: np.ndarray, *, delay_ms: int) -> np.ndarray:
        assert delay_ms == 40
        self.capture_frames.append(frame.copy())
        return (frame // 2).astype(np.int16)

    def stats(self) -> AecStats:
        return AecStats(
            available=True,
            active=True,
            degraded=False,
            backend="fake",
        )

    def reset(self) -> None:
        self.render_frames.clear()
        self.capture_frames.clear()


def test_bridge_splits_final_render_pcm_into_ten_ms_frames() -> None:
    processor = _FakeAec()
    bridge = AecPcmBridge(processor, sample_rate_hz=16_000)
    samples = np.linspace(-0.5, 0.5, 480, dtype=np.float32)

    bridge.feed_render(samples, sample_rate_hz=16_000)

    assert [frame.size for frame in processor.render_frames] == [160, 160, 160]
    assert bridge.stats().render_frames == 3


def test_bridge_processes_capture_and_preserves_input_shape() -> None:
    processor = _FakeAec()
    bridge = AecPcmBridge(processor, sample_rate_hz=16_000, delay_ms=40)
    capture = np.full(320, 0.4, dtype=np.float32)

    output = bridge.process_capture(capture, sample_rate_hz=16_000)

    assert output.shape == capture.shape
    assert output.dtype == np.float32
    assert np.max(output) == pytest.approx(0.2, abs=0.01)
    assert len(processor.capture_frames) == 2


def test_bridge_resamples_render_reference_to_aec_rate() -> None:
    processor = _FakeAec()
    bridge = AecPcmBridge(processor, sample_rate_hz=16_000)
    render_24k = np.ones(240, dtype=np.float32) * 0.25

    bridge.feed_render(render_24k, sample_rate_hz=24_000)

    assert len(processor.render_frames) == 1
    assert processor.render_frames[0].size == 160
