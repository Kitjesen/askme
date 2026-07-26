from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from askme.voice.diagnostics.full_duplex_hardware import (
    HARDWARE_REPORT_SCHEMA_VERSION,
    evaluate_hardware_run,
    preflight_hardware_run,
    runtime_readiness,
)
from askme.voice.diagnostics.hardware_audio_capture import (
    build_instrumented_trial_evidence,
    build_manual_trial_evidence,
)


def _timestamp(offset: timedelta = timedelta()) -> str:
    return (datetime.now(UTC) + offset).isoformat()


def _healthy_hardware_status() -> dict[str, object]:
    return {
        "status": "ok",
        "snapshot_at": _timestamp(),
        "voice_pipeline_status": {
            "pipeline_ok": True,
            "recorded_at": _timestamp(),
            "media": {
                "full_duplex": {
                    "enabled": True,
                    "reason": "verified_echo_control",
                    "echo_control": "hardware",
                    "aec_backend": "hardware",
                }
            },
        },
    }


def _metadata() -> dict[str, object]:
    return {
        "operating_system": "Windows-11",
        "python_version": "3.11.9",
        "room": "target-room",
        "audio_device": "target-speakerphone",
        "audio_driver": "WASAPI",
        "input_device_id": "1",
        "output_device_id": "3",
        "input_sample_rate_hz": 16_000,
        "output_sample_rate_hz": 16_000,
        "aec_backend": "hardware",
    }


def _speaker_trials(status: dict[str, object], *, count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "false_barge_in": False,
            "runtime_status": status,
            **build_manual_trial_evidence(
                method="manual_observation",
                reference_event="speaker_only_false_barge_in",
                observed_timestamp_s=10.0 + index,
            ),
        }
        for index in range(count)
    ]


def _physical_latency_evidence(*, scenario: str, latency_ms: float) -> dict[str, object]:
    reference_s = 100.0
    capture_role = (
        "isolated_speaker_monitor" if scenario == "human_overlap" else "room_acoustic_monitor"
    )
    reference_event = "human_speech_onset" if scenario == "human_overlap" else "speech_end"
    return build_instrumented_trial_evidence(
        evidence_kind="physical_acoustic",
        method="rms_threshold_v2",
        capture={
            "source_label": "microphone",
            "source_evidence_kind": "physical_acoustic",
            "instrumented": True,
            "device_id": "speaker-probe",
            "stream_id": "speaker-stream",
            "channel": 0,
            "clock_id": "perf-counter-1",
            "role": capture_role,
            "isolated_from_reference": scenario == "human_overlap",
        },
        reference={
            "event": reference_event,
            "instrumented": True,
            "device_id": "speech-reference",
            "stream_id": "speech-stream",
            "channel": 0,
            "clock_id": "perf-counter-1",
        },
        reference_timestamp_s=reference_s,
        event_timestamp_s=reference_s + latency_ms / 1000.0,
        clock_id="perf-counter-1",
        calibration={
            "performed": True,
            "source_label": "microphone",
            "source_evidence_kind": "physical_acoustic",
            "sample_rate_hz": 48_000,
            "valid_frame_count": 200,
            "threshold": 0.02,
        },
        dropped_frames=0,
    )


def _overlap_trials(
    status: dict[str, object],
    *,
    count: int = 20,
    latency_ms: float = 180.0,
) -> list[dict[str, object]]:
    return [
        {
            "detected": True,
            "speaker_stop_latency_ms": latency_ms,
            "runtime_status": status,
            **_physical_latency_evidence(
                scenario="human_overlap",
                latency_ms=latency_ms,
            ),
        }
        for _ in range(count)
    ]


def _response_trials(
    status: dict[str, object],
    *,
    count: int = 20,
    latency_ms: float = 850.0,
) -> list[dict[str, object]]:
    return [
        {
            "heard": True,
            "speech_end_to_first_sound_ms": latency_ms,
            "runtime_status": status,
            **_physical_latency_evidence(
                scenario="assistant_response",
                latency_ms=latency_ms,
            ),
        }
        for _ in range(count)
    ]


def _render_latency_evidence(*, scenario: str, latency_ms: float) -> dict[str, object]:
    return build_instrumented_trial_evidence(
        evidence_kind="render_chain",
        method="render_loopback_threshold_v2",
        capture={
            "source_label": "wasapi_loopback",
            "source_evidence_kind": "render_chain",
            "instrumented": True,
            "device_id": "render-device",
            "stream_id": "render-stream",
            "channel": 0,
            "clock_id": "perf-counter-1",
            "role": "render_loopback",
        },
        reference={
            "event": "human_speech_onset" if scenario == "human_overlap" else "speech_end",
            "instrumented": True,
            "device_id": "speech-reference",
            "stream_id": "speech-stream",
            "channel": 0,
            "clock_id": "perf-counter-1",
        },
        reference_timestamp_s=100.0,
        event_timestamp_s=100.0 + latency_ms / 1000.0,
        clock_id="perf-counter-1",
        calibration={
            "performed": True,
            "source_label": "wasapi_loopback",
            "source_evidence_kind": "render_chain",
            "sample_rate_hz": 48_000,
            "valid_frame_count": 200,
            "threshold": 0.02,
        },
        dropped_frames=0,
    )


def test_report_passes_only_with_complete_healthy_hardware_trials() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=_overlap_trials(status),
        response_trials=_response_trials(status),
    )

    assert report["status"] == "passed"
    assert report["schema_version"] == HARDWARE_REPORT_SCHEMA_VERSION
    assert report["failed_checks"] == []
    assert report["summary"]["speaker_only"]["pass_rate"] == 1.0
    assert report["summary"]["human_overlap"]["detection_rate"] == 1.0
    assert report["summary"]["human_overlap"]["speaker_stop_latency_ms"]["p95"] == 180.0


def test_report_accepts_measured_physical_first_sound_trials() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=_overlap_trials(status),
        response_trials=_response_trials(status),
    )

    assert report["status"] == "passed"
    assert report["summary"]["assistant_response"]["count"] == 20
    assert (
        report["summary"]["assistant_response"]["speech_end_to_physical_first_sound_ms"]["p95"]
        == 850.0
    )


def test_shared_room_microphone_cannot_prove_speaker_stop_during_overlap() -> None:
    status = _healthy_hardware_status()
    overlap = _overlap_trials(status)
    capture = overlap[0]["capture"]
    assert isinstance(capture, dict)
    capture["role"] = "room_acoustic_monitor"
    capture["isolated_from_reference"] = False

    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=overlap,
    )

    assert report["status"] == "failed"
    assert report["checks"]["physical_speaker_stop_sample_count"] is False
    assert any(
        failure["reason"] == "speaker_stop_requires_isolated_monitor"
        for failure in report["evidence_failures"]
    )


def test_render_chain_stop_latency_is_separate_and_cannot_satisfy_physical_gate() -> None:
    status = _healthy_hardware_status()
    render_trials = [
        {
            "detected": True,
            "speaker_stop_latency_ms": 75.0,
            "runtime_status": status,
            **_render_latency_evidence(
                scenario="human_overlap",
                latency_ms=75.0,
            ),
        }
        for _ in range(20)
    ]

    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=render_trials,
    )

    summary = report["summary"]["human_overlap"]
    assert report["status"] == "failed"
    assert report["checks"]["physical_speaker_stop_sample_count"] is False
    assert summary["physical_speaker_stop_latency_ms"]["count"] == 0
    assert summary["render_chain_speaker_stop_latency_ms"]["p95"] == 75.0


def test_physical_gate_rejects_capture_with_dropped_frames() -> None:
    status = _healthy_hardware_status()
    overlap = _overlap_trials(status)
    overlap[0]["dropped_frames"] = 1

    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=overlap,
    )

    assert report["status"] == "failed"
    assert any(
        failure["reason"] == "dropped_frames_nonzero" for failure in report["evidence_failures"]
    )


def test_report_rejects_too_few_physical_first_sound_trials() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=[
            {"false_barge_in": False, "runtime_status": status} for _ in range(20)
        ],
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": status,
            }
            for _ in range(20)
        ],
        response_trials=[
            {
                "heard": True,
                "speech_end_to_first_sound_ms": 850.0,
                "runtime_status": status,
            }
            for _ in range(19)
        ],
    )

    assert report["status"] == "failed"
    assert "assistant_response_sample_count" in report["failed_checks"]


def test_v2_first_sound_gate_cannot_be_disabled_by_legacy_opt_out() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=_overlap_trials(status),
        response_trials=[],
        require_response_trials=False,
    )

    assert report["status"] == "failed"
    assert "assistant_response_sample_count" in report["failed_checks"]
    assert "physical_first_sound_sample_count" in report["failed_checks"]


def test_unverified_hardware_echo_control_fails_even_when_trials_look_good() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": False,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=[
            {"false_barge_in": False, "runtime_status": status} for _ in range(20)
        ],
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": status,
            }
            for _ in range(20)
        ],
    )

    assert report["status"] == "failed"
    assert report["echo_control_evidence"]["proven"] is False
    assert "echo_control_proven" in report["failed_checks"]


def test_runtime_degradation_during_any_trial_fails_the_run() -> None:
    healthy = _healthy_hardware_status()
    degraded = {
        "status": "degraded",
        "voice_pipeline_status": {
            "pipeline_ok": True,
            "media": {
                "full_duplex": {
                    "enabled": False,
                    "reason": "render_transport_runtime_failure",
                    "echo_control": "none",
                    "aec_backend": "hardware",
                }
            },
        },
    }
    speaker_trials = [{"false_barge_in": False, "runtime_status": healthy} for _ in range(20)]
    speaker_trials[7]["runtime_status"] = degraded

    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=speaker_trials,
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": healthy,
            }
            for _ in range(20)
        ],
    )

    assert report["status"] == "failed"
    assert "runtime_remained_full_duplex" in report["failed_checks"]
    assert report["runtime_failures"] == [{"snapshot": 8, "reason": "runtime_status_degraded"}]


def test_report_rejects_fewer_than_twenty_trials_per_scenario() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=[
            {"false_barge_in": False, "runtime_status": status} for _ in range(19)
        ],
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": status,
            }
            for _ in range(19)
        ],
    )

    assert report["status"] == "failed"
    assert "speaker_only_sample_count" in report["failed_checks"]
    assert "human_overlap_sample_count" in report["failed_checks"]


def test_active_native_aec_is_runtime_proof_without_hardware_flag() -> None:
    status = {
        "status": "ok",
        "snapshot_at": _timestamp(),
        "voice_pipeline_status": {
            "pipeline_ok": True,
            "recorded_at": _timestamp(),
            "media": {
                "full_duplex": {
                    "enabled": True,
                    "reason": "native_aec_ready",
                    "echo_control": "native",
                    "aec_backend": "webrtc-apm-v2.1",
                }
            },
        },
    }
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "auto",
                    "echo_control_verified": False,
                }
            }
        },
        metadata={**_metadata(), "aec_backend": "webrtc-apm-v2.1"},
        speaker_only_trials=_speaker_trials(status),
        overlap_trials=_overlap_trials(status),
        response_trials=_response_trials(status),
    )

    assert report["status"] == "passed"
    assert report["echo_control_evidence"]["proof"] == "active_native_aec"


def test_missing_hardware_metadata_fails_traceability_check() -> None:
    status = _healthy_hardware_status()
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata={"audio_device": "target-speakerphone"},
        speaker_only_trials=[
            {"false_barge_in": False, "runtime_status": status} for _ in range(20)
        ],
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": status,
            }
            for _ in range(20)
        ],
    )

    assert report["status"] == "failed"
    assert "hardware_metadata_complete" in report["failed_checks"]
    assert "audio_driver" in report["metadata_missing"]


def test_preflight_refuses_unverified_hardware_before_operator_trials() -> None:
    result = preflight_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": False,
                }
            }
        },
        runtime_status=_healthy_hardware_status(),
    )

    assert result["status"] == "failed"
    assert result["runtime_ready"] is True
    assert result["echo_control_proven"] is False
    assert "echo_control_unproven" in result["errors"]


def test_missing_trial_outcome_cannot_be_hidden_by_extra_samples() -> None:
    status = _healthy_hardware_status()
    speaker_trials = [{"false_barge_in": False, "runtime_status": status} for _ in range(20)]
    speaker_trials.append({"runtime_status": status})

    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=speaker_trials,
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": status,
            }
            for _ in range(20)
        ],
    )

    assert report["status"] == "failed"
    assert "speaker_only_results_complete" in report["failed_checks"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda status: status.pop("status"), "runtime_status_missing"),
        (
            lambda status: status["voice_pipeline_status"].pop("pipeline_ok"),
            "voice_pipeline_health_missing",
        ),
        (lambda status: status.pop("snapshot_at"), "snapshot_at_missing"),
        (
            lambda status: status["voice_pipeline_status"].pop("recorded_at"),
            "voice_recorded_at_missing",
        ),
    ],
)
def test_runtime_readiness_rejects_missing_health_proof(
    mutation: object,
    expected_reason: str,
) -> None:
    status = _healthy_hardware_status()
    mutation(status)  # type: ignore[operator]

    assert runtime_readiness(status) == (False, expected_reason)


@pytest.mark.parametrize(
    ("field", "offset", "expected_reason"),
    [
        ("snapshot_at", timedelta(seconds=-30), "snapshot_at_stale"),
        ("snapshot_at", timedelta(seconds=30), "snapshot_at_future"),
        ("recorded_at", timedelta(seconds=-30), "voice_recorded_at_stale"),
        ("recorded_at", timedelta(seconds=30), "voice_recorded_at_future"),
    ],
)
def test_runtime_readiness_rejects_stale_or_future_health_proof(
    field: str,
    offset: timedelta,
    expected_reason: str,
) -> None:
    status = _healthy_hardware_status()
    if field == "snapshot_at":
        status[field] = _timestamp(offset)
    else:
        status["voice_pipeline_status"][field] = _timestamp(offset)  # type: ignore[index]

    assert runtime_readiness(status) == (False, expected_reason)


@pytest.mark.parametrize("field", ["snapshot_at", "recorded_at"])
def test_runtime_readiness_rejects_unparseable_health_timestamps(field: str) -> None:
    status = _healthy_hardware_status()
    if field == "snapshot_at":
        status[field] = "not-a-timestamp"
        expected_reason = "snapshot_at_invalid"
    else:
        status["voice_pipeline_status"][field] = "not-a-timestamp"  # type: ignore[index]
        expected_reason = "voice_recorded_at_invalid"

    assert runtime_readiness(status) == (False, expected_reason)


@pytest.mark.parametrize("runtime_mode", ["hardware", "system"])
@pytest.mark.parametrize("runtime_backend", ["unknown", "native", ""])
def test_verified_external_echo_control_requires_matching_runtime_backend(
    runtime_mode: str,
    runtime_backend: str,
) -> None:
    status = _healthy_hardware_status()
    full_duplex = status["voice_pipeline_status"]["media"]["full_duplex"]  # type: ignore[index]
    full_duplex["echo_control"] = runtime_mode
    full_duplex["aec_backend"] = runtime_backend

    result = preflight_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": runtime_mode,
                    "echo_control_verified": True,
                }
            }
        },
        runtime_status=status,
    )

    assert result["status"] == "failed"
    assert result["echo_control_proven"] is False
    assert "echo_control_unproven" in result["errors"]


@pytest.mark.parametrize("backend", ["unknown", "unavailable", "none", "fake-native-aec"])
def test_native_echo_control_rejects_unproven_backend(backend: str) -> None:
    status = _healthy_hardware_status()
    full_duplex = status["voice_pipeline_status"]["media"]["full_duplex"]  # type: ignore[index]
    full_duplex.update(
        {
            "reason": "native_aec_ready",
            "echo_control": "native",
            "aec_backend": backend,
        }
    )

    result = preflight_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "auto",
                    "echo_control_verified": False,
                }
            }
        },
        runtime_status=status,
    )

    assert result["status"] == "failed"
    assert result["echo_control_proven"] is False


def test_twenty_plus_twenty_trials_cannot_pass_with_incomplete_runtime_status() -> None:
    incomplete = _healthy_hardware_status()
    incomplete.pop("snapshot_at")
    report = evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata=_metadata(),
        speaker_only_trials=[
            {"false_barge_in": False, "runtime_status": incomplete} for _ in range(20)
        ],
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": 180.0,
                "runtime_status": incomplete,
            }
            for _ in range(20)
        ],
    )

    assert report["status"] == "failed"
    assert "runtime_remained_full_duplex" in report["failed_checks"]
    assert {failure["reason"] for failure in report["runtime_failures"]} == {"snapshot_at_missing"}
