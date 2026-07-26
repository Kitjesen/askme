from __future__ import annotations

from copy import deepcopy

import pytest

from askme.voice.lab.service import (
    VoiceLabConflict,
    VoiceLabService,
    VoiceLabStateError,
    VoiceLabValidationError,
)


class FakeAudioBackend:
    def __init__(self, *, device_status: str = "ok") -> None:
        self.device_status = device_status
        self.device_checks = 0
        self.calibrations = 0

    def inventory(self) -> dict:
        return {
            "status": "ok",
            "platform": "Windows-test",
            "devices": [
                {
                    "index": 1,
                    "name": "Test microphone",
                    "hostapi": 0,
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "default_samplerate": 48_000,
                    "is_input": True,
                    "is_output": False,
                },
                {
                    "index": 2,
                    "name": "Test speaker",
                    "hostapi": 0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48_000,
                    "is_input": False,
                    "is_output": True,
                },
            ],
            "hostapis": [{"index": 0, "name": "Windows WASAPI"}],
            "recommendation": {"input_device": 1, "output_device": 2},
        }

    def capabilities(self) -> dict:
        return {
            "automatic_device_check": True,
            "automatic_microphone_calibration": True,
            "physical_first_sound_collector": False,
            "physical_overlap_stop_collector": False,
            "render_loopback_collector": False,
        }

    def run_device_check(self, *, run_dir, device_binding: dict) -> dict:
        self.device_checks += 1
        return {
            "status": self.device_status,
            "tone_detected": self.device_status == "ok",
            "signal_ok": self.device_status == "ok",
            "input_device": device_binding["input_device_id"],
            "output_device": device_binding["output_device_id"],
            "sample_rate": device_binding["input_sample_rate_hz"],
            "wav_out": str(run_dir / "device-check.wav"),
        }

    def calibrate_microphone(self, *, device_binding: dict, duration_s: float) -> dict:
        self.calibrations += 1
        return {
            "status": "ok",
            "duration_s": duration_s,
            "calibration": {
                "performed": True,
                "source_label": "microphone",
                "source_evidence_kind": "physical_acoustic",
                "sample_rate_hz": device_binding["input_sample_rate_hz"],
                "frame_count": 100,
                "valid_frame_count": 100,
                "rms_p50": 0.001,
                "rms_p95": 0.002,
                "rms_p99": 0.003,
                "margin_db": 12.0,
                "threshold": 0.008,
            },
        }


def run_body() -> dict:
    return {
        "operator_id": "dashboard.operator",
        "room": "meeting-room-a",
        "device_binding": {
            "input_device_id": 1,
            "output_device_id": 2,
            "audio_device": "ordinary speakerphone",
            "audio_driver": "Windows WASAPI",
            "input_sample_rate_hz": 48_000,
            "output_sample_rate_hz": 48_000,
            "aec_backend": "none",
        },
    }


def ready_run(service: VoiceLabService) -> dict:
    run = service.create_run(run_body(), idempotency_key="create-run")
    run = service.check_devices(
        run["run_id"], expected_version=run["version"], idempotency_key="device-check"
    )
    return service.calibrate(
        run["run_id"],
        {"duration_s": 1.0},
        expected_version=run["version"],
        idempotency_key="calibrate",
    )


def trial_body(scenario: str, ordinal: int) -> dict:
    body = {
        "scenario": scenario,
        "ordinal": ordinal,
        "quality": "clear",
        "notes": "operator heard the trial",
    }
    if scenario == "speaker_only":
        body["false_barge_in"] = False
    elif scenario == "human_overlap":
        body["detected"] = True
    else:
        body["heard"] = True
    return body


def complete_trial(
    service: VoiceLabService,
    run: dict,
    scenario: str,
    ordinal: int,
    *,
    idempotency_key: str,
) -> dict:
    started = service.begin_trial(
        run["run_id"],
        expected_version=run["version"],
        idempotency_key=f"begin-{idempotency_key}",
    )
    body = trial_body(scenario, ordinal)
    body["attempt_id"] = started["active_trial"]["attempt_id"]
    return service.submit_trial(
        run["run_id"],
        body,
        expected_version=started["version"],
        idempotency_key=idempotency_key,
    )


def test_create_run_is_idempotent_and_conflicting_body_is_rejected(tmp_path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    first = service.create_run(run_body(), idempotency_key="same-key")
    replay = service.create_run(deepcopy(run_body()), idempotency_key="same-key")

    assert replay["run_id"] == first["run_id"]
    assert replay["version"] == first["version"]

    changed = run_body()
    changed["room"] = "another-room"
    with pytest.raises(VoiceLabConflict, match="idempotency"):
        service.create_run(changed, idempotency_key="same-key")


def test_run_persists_progress_and_recovers_next_trial(tmp_path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    run = ready_run(service)
    run = complete_trial(
        service,
        run,
        "speaker_only",
        1,
        idempotency_key="speaker-1",
    )

    recovered = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend()).get_run(run["run_id"])

    assert recovered["progress"]["speaker_only"] == 1
    assert recovered["next_action"] == {
        "action": "trial",
        "scenario": "speaker_only",
        "ordinal": 2,
    }
    assert recovered["trials"][0]["evidence_kind"] == "manual"
    assert recovered["trials"][0]["product_gate_usable"] is False


def test_expected_version_and_sequential_trial_are_enforced(tmp_path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    run = ready_run(service)
    started = service.begin_trial(
        run["run_id"],
        expected_version=run["version"],
        idempotency_key="begin-speaker-1",
    )
    speaker = trial_body("speaker_only", 1)
    speaker["attempt_id"] = started["active_trial"]["attempt_id"]

    with pytest.raises(VoiceLabConflict, match="version"):
        service.submit_trial(
            run["run_id"],
            speaker,
            expected_version=run["version"],
            idempotency_key="stale",
        )

    out_of_order = trial_body("human_overlap", 1)
    out_of_order["attempt_id"] = started["active_trial"]["attempt_id"]
    with pytest.raises(VoiceLabStateError, match="active trial attempt"):
        service.submit_trial(
            run["run_id"],
            out_of_order,
            expected_version=started["version"],
            idempotency_key="out-of-order",
        )


def test_device_failure_blocks_calibration(tmp_path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend(device_status="degraded"))
    run = service.create_run(run_body(), idempotency_key="create")
    run = service.check_devices(
        run["run_id"], expected_version=run["version"], idempotency_key="check"
    )

    assert run["status"] == "blocked"
    assert run["next_action"]["action"] == "device_check"
    with pytest.raises(VoiceLabStateError, match="device check"):
        service.calibrate(
            run["run_id"],
            {"duration_s": 1.0},
            expected_version=run["version"],
            idempotency_key="calibrate",
        )


def test_pause_resume_keeps_trials_but_requires_device_revalidation(tmp_path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    run = ready_run(service)
    run = complete_trial(
        service,
        run,
        "speaker_only",
        1,
        idempotency_key="trial-1",
    )
    run = service.pause(
        run["run_id"], expected_version=run["version"], idempotency_key="pause"
    )
    run = service.resume(
        run["run_id"], expected_version=run["version"], idempotency_key="resume"
    )

    assert run["progress"]["speaker_only"] == 1
    assert run["status"] == "needs_device_check"
    assert run["device_check"]["status"] == "stale"
    assert run["calibration"]["status"] == "stale"


def test_sixty_manual_trials_complete_diagnostic_but_fail_physical_gate(tmp_path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    run = ready_run(service)
    for scenario in ("speaker_only", "human_overlap", "assistant_response"):
        for ordinal in range(1, 21):
            run = complete_trial(
                service,
                run,
                scenario,
                ordinal,
                idempotency_key=f"{scenario}-{ordinal}",
            )

    assert run["status"] == "ready_for_report"
    run = service.generate_report(
        run["run_id"], expected_version=run["version"], idempotency_key="report"
    )

    assert run["status"] == "completed"
    assert run["manual_diagnostic_complete"] is True
    assert run["product_gate"]["status"] == "failed"
    assert run["product_gate"]["report"]["checks"]["physical_speaker_stop_sample_count"] is False
    assert run["product_gate"]["report"]["checks"]["physical_first_sound_sample_count"] is False
    assert run["product_gate"]["report"]["summary"]["speaker_only"]["count"] == 20


@pytest.mark.parametrize("run_id", ["../escape", "bad/slash", "", "a" * 100])
def test_invalid_run_ids_are_rejected(tmp_path, run_id: str) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    with pytest.raises(VoiceLabValidationError):
        service.get_run(run_id)
