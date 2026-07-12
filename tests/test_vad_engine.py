"""Tests for the native sherpa-onnx VAD boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from askme.voice.input.vad import VADEngine


def test_accept_waveform_normalizes_pcm16_for_sherpa() -> None:
    captured: dict[str, np.ndarray] = {}

    detector = SimpleNamespace(
        accept_waveform=lambda waveform: captured.setdefault("waveform", waveform)
    )
    engine = object.__new__(VADEngine)
    engine.detector = detector

    pcm16 = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    engine.accept_waveform(pcm16)

    waveform = captured["waveform"]
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(
        waveform,
        np.array([-1.0, -0.5, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
    )
