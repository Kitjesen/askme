from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from askme.voice.diagnostics.hardware_audio_capture import (
    HardwareAudioCaptureError,
    StreamingOnsetDetector,
    build_instrumented_trial_evidence,
    build_manual_trial_evidence,
    calibrate_noise_floor,
    frame_rms,
    list_audio_devices,
)


class _FakeDefault:
    device = (0, 1)


class _FakeSoundDevice:
    default = _FakeDefault()

    @staticmethod
    def query_devices() -> list[dict[str, Any]]:
        return [
            {
                "name": "Speakerphone Mic",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            },
            {
                "name": "Speakerphone Speaker",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
            },
            {
                "name": "WASAPI Loopback",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            },
        ]

    @staticmethod
    def query_hostapis() -> list[dict[str, Any]]:
        return [
            {
                "name": "Windows WASAPI",
                "device_count": 3,
                "default_input_device": 0,
                "default_output_device": 1,
            },
            {
                "name": "Manual",
                "device_count": 1,
                "default_input_device": -1,
                "default_output_device": -1,
            },
        ]


def test_list_audio_devices_uses_lazy_fake_sounddevice_without_opening_streams() -> None:
    payload = list_audio_devices(_FakeSoundDevice())

    assert payload["status"] == "ok"
    assert payload["default_input_device"] == 0
    assert payload["default_output_device"] == 1
    assert payload["hostapis"][0]["name"] == "Windows WASAPI"
    devices = payload["devices"]
    assert devices[0]["is_input"] is True
    assert devices[0]["is_default_input"] is True
    assert devices[1]["is_output"] is True
    assert devices[1]["is_default_output"] is True
    assert devices[2]["hostapi_name"] == "Manual"


def test_list_audio_devices_reports_controlled_query_errors() -> None:
    class BrokenSoundDevice:
        default = _FakeDefault()

        @staticmethod
        def query_devices() -> list[dict[str, Any]]:
            raise RuntimeError("portaudio exploded")

        @staticmethod
        def query_hostapis() -> list[dict[str, Any]]:
            return []

    with pytest.raises(HardwareAudioCaptureError) as exc:
        list_audio_devices(BrokenSoundDevice())

    assert "sounddevice device query failed: RuntimeError" in str(exc.value)
    assert "portaudio exploded" not in str(exc.value)


def test_calibrate_noise_floor_uses_finite_rms_percentiles_and_margin() -> None:
    frames: list[np.ndarray] = [
        np.full(160, 0.001, dtype=np.float32),
        np.full(160, 0.002, dtype=np.float32),
        np.array([np.nan, np.nan], dtype=np.float32),
        np.full(160, 0.003, dtype=np.float32),
    ]

    calibration = calibrate_noise_floor(
        frames,
        sample_rate_hz=16_000,
        source_label="microphone",
        percentile=95,
        margin_db=6,
    )

    assert calibration.frame_count == 4
    assert calibration.valid_frame_count == 3
    assert calibration.rms_p50 == pytest.approx(0.002, abs=1e-7)
    assert calibration.rms_p95 > calibration.rms_p50
    assert calibration.threshold > calibration.rms_p95
    assert asdict(calibration)["source_label"] == "microphone"
    assert calibration.source_evidence_kind == "physical_acoustic"


def test_calibrate_noise_floor_rejects_empty_or_nan_only_frames() -> None:
    with pytest.raises(HardwareAudioCaptureError):
        calibrate_noise_floor(
            [np.array([], dtype=np.float32), np.array([np.nan], dtype=np.float32)],
            sample_rate_hz=16_000,
        )


def test_streaming_detector_finds_onset_and_offset_with_hangover() -> None:
    detector = StreamingOnsetDetector(
        threshold=0.1,
        source_label="wasapi_loopback",
        consecutive_on_frames=2,
        consecutive_off_frames=2,
        hangover_frames=1,
    )

    frames: list[tuple[float, np.ndarray]] = [
        (0.00, np.zeros(80, dtype=np.float32)),
        (0.01, np.full(80, 0.2, dtype=np.float32)),
        (0.02, np.full(80, 0.2, dtype=np.float32)),
        (0.03, np.full(80, 0.2, dtype=np.float32)),
        (0.04, np.zeros(80, dtype=np.float32)),
        (0.05, np.zeros(80, dtype=np.float32)),
        (0.06, np.zeros(80, dtype=np.float32)),
    ]

    result = None
    for timestamp, pcm in frames:
        result = detector.process_frame(timestamp=timestamp, pcm=pcm)

    assert result is not None
    assert result.detected is True
    assert result.stopped is True
    assert result.source_label == "wasapi_loopback"
    assert result.source_evidence_kind == "render_chain"
    assert result.onset_timestamp == pytest.approx(0.01)
    assert result.onset_confirm_timestamp == pytest.approx(0.02)
    assert result.offset_timestamp == pytest.approx(0.06)
    assert result.onset_frame_index == 1
    assert result.onset_confirm_frame_index == 2
    assert result.offset_frame_index == 6
    result_dict = result.to_dict()
    assert result_dict["source_evidence_kind"] == "render_chain"
    assert result_dict["detected"] is True


def test_streaming_detector_ignores_single_frame_transient_noise() -> None:
    detector = StreamingOnsetDetector(
        threshold=0.1,
        consecutive_on_frames=2,
        consecutive_off_frames=2,
    )

    detector.process_frame(timestamp=0.0, pcm=np.zeros(80, dtype=np.float32))
    result = detector.process_frame(
        timestamp=0.01,
        pcm=np.full(80, 0.5, dtype=np.float32),
    )
    result = detector.process_frame(timestamp=0.02, pcm=np.zeros(80, dtype=np.float32))

    assert result.detected is False
    assert result.active is False


def test_streaming_detector_does_not_count_nan_or_empty_frames_as_audio() -> None:
    detector = StreamingOnsetDetector(threshold=0.1, source_label="manual")

    detector.process_frame(timestamp=0.0, pcm=np.array([np.nan], dtype=np.float32))
    result = detector.process_frame(timestamp=0.01, pcm=np.array([], dtype=np.float32))

    assert frame_rms(np.array([np.nan], dtype=np.float32)) is None
    assert frame_rms(np.array([], dtype=np.float32)) is None
    assert result.detected is False
    assert result.processed_frames == 2
    assert result.source_evidence_kind == "manual"


def test_streaming_detector_rejects_non_finite_timestamps_without_counting_frame() -> None:
    detector = StreamingOnsetDetector(threshold=0.1)

    with pytest.raises(ValueError, match="timestamp must be finite"):
        detector.process_frame(timestamp=float("nan"), pcm=np.zeros(80, dtype=np.float32))

    assert detector.result().processed_frames == 0

    with pytest.raises(ValueError, match="timestamp must be finite"):
        detector.process_frame(timestamp=float("inf"), pcm=np.zeros(80, dtype=np.float32))

    assert detector.result().processed_frames == 0


def test_streaming_detector_rejects_decreasing_timestamps_without_counting_frame() -> None:
    detector = StreamingOnsetDetector(threshold=0.1)

    detector.process_frame(timestamp=1.0, pcm=np.zeros(80, dtype=np.float32))
    with pytest.raises(ValueError, match="timestamp must be non-decreasing"):
        detector.process_frame(timestamp=0.99, pcm=np.zeros(80, dtype=np.float32))

    assert detector.result().processed_frames == 1


def test_invalid_source_label_fails_fast() -> None:
    with pytest.raises(ValueError):
        StreamingOnsetDetector(threshold=0.1, source_label="ros2")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        calibrate_noise_floor(
            [np.zeros(80, dtype=np.float32)],
            sample_rate_hz=16_000,
            source_label="ros2",  # type: ignore[arg-type]
        )


def test_instrumented_trial_builder_rejects_manual_method_labels() -> None:
    with pytest.raises(ValueError, match="automatic capture method"):
        build_instrumented_trial_evidence(
            evidence_kind="physical_acoustic",
            method="entry",
            capture={},
            reference={},
            reference_timestamp_s=1.0,
            event_timestamp_s=1.1,
            calibration={},
            dropped_frames=0,
            clock_id="clock-1",
        )


def test_manual_trial_builder_emits_explicit_non_instrumented_v2_fields() -> None:
    evidence = build_manual_trial_evidence(
        method="manual_stopwatch",
        reference_event="speech_end",
        reference_timestamp_s=1.0,
        event_timestamp_s=1.8,
    )

    assert evidence["evidence_kind"] == "manual"
    assert evidence["capture"]["instrumented"] is False
    assert evidence["reference"]["instrumented"] is False
    assert evidence["calibration"] == {"performed": False}
    assert evidence["dropped_frames"] is None
