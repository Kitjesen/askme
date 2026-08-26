from __future__ import annotations

from pathlib import Path

import pytest

from askme.voice.lab.service import VoiceLabConflict, VoiceLabService, VoiceLabStateError
from tests.test_voice_lab_service import FakeAudioBackend, ready_run, run_body


def test_trial_requires_server_started_attempt(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    run = ready_run(service)

    with pytest.raises(VoiceLabStateError, match="started trial attempt"):
        service.submit_trial(
            run["run_id"],
            {
                "scenario": "speaker_only",
                "ordinal": 1,
                "attempt_id": "vat_missing",
                "quality": "clear",
                "notes": "",
                "false_barge_in": False,
            },
            expected_version=run["version"],
            idempotency_key="trial-without-begin",
        )


def test_begin_trial_is_exclusive_recoverable_and_consumed(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)

    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-speaker-1",
    )
    attempt = started["active_trial"]
    assert attempt["scenario"] == "speaker_only"
    assert attempt["ordinal"] == 1
    assert started["next_action"]["action"] == "trial_active"
    assert service.get_run(ready["run_id"])["active_trial"] == attempt

    with pytest.raises(VoiceLabConflict, match="version conflict"):
        service.begin_trial(
            ready["run_id"],
            expected_version=ready["version"],
            idempotency_key="second-tab",
        )

    completed = service.submit_trial(
        ready["run_id"],
        {
            "scenario": "speaker_only",
            "ordinal": 1,
            "attempt_id": attempt["attempt_id"],
            "quality": "clear",
            "notes": "stable",
            "false_barge_in": False,
        },
        expected_version=started["version"],
        idempotency_key="finish-speaker-1",
    )
    assert completed["active_trial"] is None
    assert completed["progress"]["speaker_only"] == 1
    assert completed["next_action"] == {
        "action": "trial",
        "scenario": "speaker_only",
        "ordinal": 2,
    }


def test_submit_rejects_attempt_from_another_trial(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-speaker-1",
    )

    with pytest.raises(VoiceLabStateError, match="active trial attempt"):
        service.submit_trial(
            ready["run_id"],
            {
                "scenario": "speaker_only",
                "ordinal": 1,
                "attempt_id": "vat_wrong",
                "quality": "clear",
                "notes": "",
                "false_barge_in": False,
            },
            expected_version=started["version"],
            idempotency_key="wrong-attempt",
        )


def test_pause_invalidates_active_attempt_but_keeps_completed_trials(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-speaker-1",
    )

    paused = service.pause(
        ready["run_id"],
        expected_version=started["version"],
        idempotency_key="pause-active",
    )
    assert paused["active_trial"] is None
    assert paused["invalidated_trials"][0]["attempt_id"] == started["active_trial"]["attempt_id"]
    assert paused["invalidated_trials"][0]["reason"] == "run_paused"


def test_begin_trial_replay_is_idempotent(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)

    first = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-speaker-1",
    )
    replay = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-speaker-1",
    )
    assert replay == first


def test_create_initializes_without_an_active_attempt(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    created = service.create_run(run_body(), idempotency_key="create-attempt-test")
    assert created["active_trial"] is None


def test_report_rejects_incomplete_plan_and_active_attempt(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)

    with pytest.raises(VoiceLabStateError, match="completed trial plan"):
        service.generate_report(
            ready["run_id"],
            expected_version=ready["version"],
            idempotency_key="premature-report",
        )

    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-before-report",
    )
    with pytest.raises(VoiceLabStateError, match="completed trial plan"):
        service.generate_report(
            ready["run_id"],
            expected_version=started["version"],
            idempotency_key="active-report",
        )
