"""Unit tests for the Sunrise audio sentinel helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_sentinel():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "test_sunrise_audio_sentinel.py"
    spec = importlib.util.spec_from_file_location("sunrise_audio_sentinel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_transcript_maps_digit_variants():
    sentinel = _load_sentinel()

    assert sentinel.normalize_transcript(" 1，2，幺。") == "一二一"
    assert sentinel.transcript_has_prefix("幺二三四五", "一")
    assert sentinel.transcript_has_prefix("1, 2, 3", "一")


def test_detect_onset_ms_uses_baseline_and_threshold():
    sentinel = _load_sentinel()
    sample_rate = 1000
    samples = np.zeros(2000, dtype=np.float32)
    samples[:500] = 2 / 32768.0
    samples[900:950] = 500 / 32768.0

    onset_ms, threshold = sentinel.detect_onset_ms(
        samples,
        sample_rate,
        playback_start_offset_s=0.5,
        min_peak=300,
        window_ms=50,
    )

    assert threshold == 300
    assert onset_ms == 400.0


def test_summarize_trials_requires_majority_prefix_passes():
    sentinel = _load_sentinel()

    assert sentinel.summarize_trials(
        [{"prefix_ok": True}, {"prefix_ok": False}, {"prefix_ok": True}]
    ) == {"trials": 3, "passes": 2, "required_passes": 2, "passed": True}
    assert sentinel.summarize_trials(
        [{"prefix_ok": True}, {"prefix_ok": False}, {"prefix_ok": False}]
    )["passed"] is False


def test_summarize_trials_requires_signal_when_present():
    sentinel = _load_sentinel()

    assert sentinel.summarize_trials(
        [
            {"prefix_ok": True, "signal_ok": True},
            {"prefix_ok": True, "signal_ok": False},
            {"prefix_ok": True, "signal_ok": False},
        ]
    ) == {"trials": 3, "passes": 1, "required_passes": 2, "passed": False}


def test_detect_onset_ms_returns_none_when_signal_too_low():
    sentinel = _load_sentinel()
    sample_rate = 1000
    samples = np.zeros(2000, dtype=np.float32)
    samples[900:950] = 100 / 32768.0

    onset_ms, threshold = sentinel.detect_onset_ms(
        samples,
        sample_rate,
        playback_start_offset_s=0.5,
        min_peak=300,
        window_ms=50,
    )

    assert threshold == 300
    assert onset_ms is None
