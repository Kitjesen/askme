"""Backpressure behavior for callback-based microphone capture."""

from unittest.mock import patch

import numpy as np
import pytest

from askme.voice.input.mic_input import MicInput


@patch("askme.voice.input.mic_input.sd.InputStream")
def test_callback_queue_drops_oldest_audio_and_reports_backlog(mock_stream_cls):
    mic = MicInput(
        sample_rate=16_000,
        chunk_ms=100,
        queue_max_chunks=3,
        input_transport="sounddevice",
    )

    with mic.open():
        callback = mock_stream_cls.call_args.kwargs["callback"]
        for value in range(4):
            callback(
                np.full((1_600, 1), value, dtype=np.float32),
                1_600,
                None,
                None,
            )

        assert mic.status_snapshot() == {
            "dropped_frames": 1,
            "depth": 3,
            "max_depth": 3,
            "queued_audio_ms": 300,
        }
        assert [mic.read_chunk()[0] for _ in range(3)] == [1.0, 2.0, 3.0]
        assert mic.status_snapshot()["queued_audio_ms"] == 0


def test_queue_depth_is_configurable_from_voice_config():
    mic = MicInput.from_config({"voice": {"mic_queue_max_chunks": 4}})

    assert mic.status_snapshot()["max_depth"] == 4


def test_default_queue_bound_is_one_second_of_audio():
    mic = MicInput(chunk_ms=100)

    assert mic.status_snapshot() == {
        "dropped_frames": 0,
        "depth": 0,
        "max_depth": 10,
        "queued_audio_ms": 0,
    }


def test_queue_bound_must_be_positive():
    with pytest.raises(ValueError, match="queue_max_chunks"):
        MicInput(queue_max_chunks=0)
