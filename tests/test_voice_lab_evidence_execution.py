from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from askme.api.routes.voice_lab import create_voice_lab_router
from askme.voice.lab.service import (
    VoiceLabConflict,
    VoiceLabEvidenceUnavailable,
    VoiceLabService,
    VoiceLabStateError,
    VoiceLabValidationError,
    _immutable_context,
)
from tests.test_voice_lab_routes import (
    cors_options_response,
    mission_json,
    optional_json_body,
)
from tests.test_voice_lab_service import FakeAudioBackend, ready_run, trial_body


def runtime_evidence(correlation_id: str) -> dict[str, Any]:
    return {
        "correlation_id": correlation_id,
        "source": "server_runtime",
        "captured_at": "1900-01-01T00:00:00+00:00",
        "timeline": [
            {
                "event": "playback_started",
                "stage": "speaker_render_started",
                "offset_ms": 0.0,
                "sequence": 1,
            },
            {
                "event": "turn_finished",
                "stage": "turn_finished",
                "offset_ms": 125.5,
                "sequence": 2,
            },
        ],
        "fallback": {
            "used": False,
            "from": "realtime",
            "to": "cascade",
            "reason": "",
        },
        "interrupt": {
            "detected": False,
            "confirmed": False,
            "dismissed": False,
            "playback_resumed": False,
        },
        "configured_full_duplex": True,
        "runtime_full_duplex": True,
        "echo_control_evidence": {
            "backend": "webrtc-apm-v2.1",
            "active": True,
            "degraded": False,
        },
        "aec_stats": {
            "backend": "webrtc-apm-v2.1",
            "active": True,
            "degraded": False,
            "erl_db": 12.5,
            "erle_db": 19.25,
            "residual_echo_likelihood": 0.08,
            "evidence_kind": "algorithm_telemetry",
        },
    }


def test_provider_context_snapshot_is_deeply_immutable_readable_and_detached() -> None:
    source: dict[str, Any] = {
        "run": {"plan": {"speaker_only": 20}},
        "scenario": {"timeline_fields": ["event", "stage"]},
        "device": {"routes": [{"input": 1, "output": 2}]},
    }

    context = _immutable_context(source)

    assert context["run"]["plan"]["speaker_only"] == 20
    assert context["scenario"]["timeline_fields"] == ("event", "stage")
    assert context["device"]["routes"][0]["output"] == 2
    with pytest.raises(TypeError):
        context["run"]["plan"]["speaker_only"] = 99
    with pytest.raises(AttributeError):
        context["scenario"]["timeline_fields"].append("offset_ms")
    with pytest.raises(TypeError):
        context["device"]["routes"][0]["output"] = 9

    source["run"]["plan"]["speaker_only"] = 1
    source["scenario"]["timeline_fields"].append("sequence")
    source["device"]["routes"][0]["output"] = 8
    assert context["run"]["plan"]["speaker_only"] == 20
    assert context["scenario"]["timeline_fields"] == ("event", "stage")
    assert context["device"]["routes"][0]["output"] == 2


def test_execute_persists_server_owned_evidence_across_restart(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def provider(**context: Any) -> dict[str, Any]:
        calls.append(context)
        with pytest.raises(TypeError):
            context["run_context"]["room"] = "tampered"
        with pytest.raises(TypeError):
            context["scenario_context"]["scenario"] = "tampered"
        with pytest.raises(TypeError):
            context["device_context"]["aec_backend"] = "tampered"
        return runtime_evidence(context["correlation_id"])

    captured_at = datetime(2026, 7, 26, 10, 30, tzinfo=UTC)
    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
        now=lambda: captured_at,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-evidence",
    )
    attempt_id = started["active_trial"]["attempt_id"]

    executed = service.execute_trial(
        ready["run_id"],
        attempt_id,
        expected_version=started["version"],
        idempotency_key="execute-evidence",
    )

    assert len(calls) == 1
    assert calls[0]["correlation_id"] == attempt_id
    assert calls[0]["run_context"]["run_id"] == ready["run_id"]
    assert calls[0]["scenario_context"] == {
        "attempt_id": attempt_id,
        "scenario": "speaker_only",
        "ordinal": 1,
        "started_at": started["active_trial"]["started_at"],
    }
    assert calls[0]["device_context"] == ready["device_binding"]
    evidence = executed["active_trial"]["turn_evidence"]
    assert evidence["correlation_id"] == attempt_id
    assert evidence["source"] == "server_runtime"
    assert evidence["captured_at"] == captured_at.isoformat()
    assert executed["version"] == started["version"] + 1

    recovered = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend()).get_run(
        ready["run_id"]
    )
    assert recovered["active_trial"]["turn_evidence"] == evidence


def test_execute_replay_calls_provider_once_and_key_cannot_move_attempts(tmp_path: Path) -> None:
    correlations: list[str] = []

    def provider(**context: Any) -> dict[str, Any]:
        correlations.append(context["correlation_id"])
        return runtime_evidence(context["correlation_id"])

    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-replay",
    )
    attempt_id = started["active_trial"]["attempt_id"]
    first = service.execute_trial(
        ready["run_id"],
        attempt_id,
        expected_version=started["version"],
        idempotency_key="execute-once",
    )
    replay = service.execute_trial(
        ready["run_id"],
        attempt_id,
        expected_version=started["version"],
        idempotency_key="execute-once",
    )

    assert replay == first
    assert correlations == [attempt_id]
    with pytest.raises(VoiceLabConflict, match="idempotency"):
        service.execute_trial(
            ready["run_id"],
            "vat_another_attempt",
            expected_version=first["version"],
            idempotency_key="execute-once",
        )
    assert correlations == [attempt_id]


def test_execute_rejects_stale_wrong_and_paused_attempts_before_provider(tmp_path: Path) -> None:
    calls = 0

    def provider(**context: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return runtime_evidence(context["correlation_id"])

    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-invalid",
    )
    attempt_id = started["active_trial"]["attempt_id"]

    with pytest.raises(VoiceLabConflict, match="version conflict"):
        service.execute_trial(
            ready["run_id"],
            attempt_id,
            expected_version=ready["version"],
            idempotency_key="stale-execute",
        )
    with pytest.raises(VoiceLabStateError, match="active trial attempt"):
        service.execute_trial(
            ready["run_id"],
            "vat_wrong_attempt",
            expected_version=started["version"],
            idempotency_key="wrong-execute",
        )

    paused = service.pause(
        ready["run_id"],
        expected_version=started["version"],
        idempotency_key="pause-before-execute",
    )
    with pytest.raises(VoiceLabStateError, match="running"):
        service.execute_trial(
            ready["run_id"],
            attempt_id,
            expected_version=paused["version"],
            idempotency_key="paused-execute",
        )
    assert calls == 0


def test_execute_without_provider_fails_closed_without_committing(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-no-provider",
    )
    attempt_id = started["active_trial"]["attempt_id"]

    with pytest.raises(VoiceLabEvidenceUnavailable, match="unavailable"):
        service.execute_trial(
            ready["run_id"],
            attempt_id,
            expected_version=started["version"],
            idempotency_key="execute-no-provider",
        )

    assert service.get_run(ready["run_id"]) == started


def test_provider_exception_leaves_version_and_attempt_uncommitted(tmp_path: Path) -> None:
    def provider(**_context: Any) -> dict[str, Any]:
        raise RuntimeError("runtime transport failed")

    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-provider-error",
    )

    with pytest.raises(VoiceLabEvidenceUnavailable, match="RuntimeError"):
        service.execute_trial(
            ready["run_id"],
            started["active_trial"]["attempt_id"],
            expected_version=started["version"],
            idempotency_key="execute-provider-error",
        )

    assert service.get_run(ready["run_id"]) == started


def test_execute_route_enforces_auth_headers_and_declares_response_model(tmp_path: Path) -> None:
    calls: list[str] = []
    authorizations: list[tuple[dict[str, Any], str]] = []

    def provider(**context: Any) -> dict[str, Any]:
        calls.append(context["correlation_id"])
        return runtime_evidence(context["correlation_id"])

    def authorize(
        request: Request, body: dict[str, Any], permission: str
    ) -> JSONResponse | None:
        authorizations.append((body, permission))
        if request.headers.get("X-Voice-Access") != "allowed":
            return mission_json({"error": "forbidden"}, status_code=403)
        return None

    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-route",
    )
    attempt_id = started["active_trial"]["attempt_id"]
    path = f"/api/voice/lab/runs/{ready['run_id']}/trials/{attempt_id}/execute"
    app = FastAPI()
    app.include_router(
        create_voice_lab_router(
            service=service,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            authorize=authorize,
        )
    )

    with TestClient(app) as http:
        denied = http.post(path)
        missing_key = http.post(
            path,
            headers={
                "X-Voice-Access": "allowed",
                "If-Match": str(started["version"]),
            },
        )
        missing_version = http.post(
            path,
            headers={
                "X-Voice-Access": "allowed",
                "Idempotency-Key": "route-execute",
            },
        )
        response = http.post(
            path,
            headers={
                "X-Voice-Access": "allowed",
                "Idempotency-Key": "route-execute",
                "If-Match": str(started["version"]),
            },
        )
        openapi = app.openapi()
        operation = openapi["paths"][
            "/api/voice/lab/runs/{run_id}/trials/{attempt_id}/execute"
        ]["post"]
        schemas = openapi["components"]["schemas"]

    assert denied.status_code == 403
    assert missing_key.status_code == 428
    assert missing_version.status_code == 428
    assert response.status_code == 200
    assert response.json()["active_trial"]["turn_evidence"]["correlation_id"] == attempt_id
    assert calls == [attempt_id]
    assert authorizations[-1] == ({}, "voice:system:update")
    assert {parameter["name"] for parameter in operation["parameters"]} >= {
        "Idempotency-Key",
        "If-Match",
    }
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/VoiceLabRunResponse")
    assert set(schemas["VoiceLabTurnEvidenceResponse"]["properties"]) >= {
        "correlation_id",
        "source",
        "captured_at",
        "timeline",
        "fallback",
        "interrupt",
        "configured_full_duplex",
        "runtime_full_duplex",
        "echo_control_evidence",
        "aec_stats",
        "residual_audio",
    }


def test_execute_route_reports_unavailable_provider_as_503(tmp_path: Path) -> None:
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-route-unavailable",
    )
    attempt_id = started["active_trial"]["attempt_id"]
    app = FastAPI()
    app.include_router(
        create_voice_lab_router(
            service=service,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            authorize=lambda _request, _body, _permission: None,
        )
    )

    with TestClient(app) as http:
        response = http.post(
            f"/api/voice/lab/runs/{ready['run_id']}/trials/{attempt_id}/execute",
            headers={
                "Idempotency-Key": "execute-route-unavailable",
                "If-Match": str(started["version"]),
            },
        )

    assert response.status_code == 503
    assert response.json() == {"error": "trial evidence provider is unavailable"}
    assert service.get_run(ready["run_id"]) == started


def test_submit_copies_persisted_evidence_and_cannot_promote_it_to_a_product_gate(
    tmp_path: Path,
) -> None:
    def provider(**context: Any) -> dict[str, Any]:
        evidence = runtime_evidence(context["correlation_id"])
        evidence["echo_control_evidence"]["proven"] = True
        evidence["aec_stats"]["residual_echo_likelihood"] = 0.0
        evidence["residual_audio"] = {
            "evidence_kind": "physical",
            "measurement_source": "isolated_room_microphone",
            "clock_domain": "voice-lab-monotonic",
            "dropped_frames": 0,
            "tail_ms": 21.5,
        }
        return evidence

    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key="begin-submit-copy",
    )
    attempt_id = started["active_trial"]["attempt_id"]
    executed = service.execute_trial(
        ready["run_id"],
        attempt_id,
        expected_version=started["version"],
        idempotency_key="execute-submit-copy",
    )
    persisted = executed["active_trial"]["turn_evidence"]
    submission = trial_body("speaker_only", 1)
    submission.update(
        {
            "attempt_id": attempt_id,
            "turn_evidence": {"correlation_id": "client-forged"},
            "timeline": [{"text": "client transcript"}],
            "aec_stats": {"evidence_kind": "physical"},
            "echo_control_verified": True,
            "product_gate_usable": True,
        }
    )

    completed = service.submit_trial(
        ready["run_id"],
        submission,
        expected_version=executed["version"],
        idempotency_key="submit-copy",
    )

    trial = completed["trials"][0]
    assert trial["turn_evidence"] == persisted
    assert trial["turn_evidence"]["correlation_id"] == attempt_id
    assert trial["turn_evidence"]["aec_stats"]["evidence_kind"] == "algorithm_telemetry"
    assert trial["turn_evidence"]["residual_audio"]["evidence_kind"] == "physical"
    assert trial["evidence_kind"] == "manual"
    assert trial["product_gate_usable"] is False
    assert trial["product_gate_reason"] == (
        "runtime execution evidence alone does not prove physical product gates"
    )
    assert completed["product_gate"]["status"] == "not_evaluated"


@pytest.mark.parametrize(
    ("invalid_case", "error"),
    [
        ("correlation", "correlation_id"),
        ("transcript", "forbidden field"),
        ("pcm", "forbidden field"),
        ("credentials", "forbidden field"),
        ("product_gate", "forbidden field"),
        ("nonfinite", "finite"),
        ("deep", "nesting"),
        ("fake_physical_aec", "algorithm_telemetry"),
    ],
)
def test_execute_rejects_untrusted_or_unbounded_provider_payloads_without_commit(
    tmp_path: Path,
    invalid_case: str,
    error: str,
) -> None:
    def provider(**context: Any) -> dict[str, Any]:
        evidence = runtime_evidence(context["correlation_id"])
        if invalid_case == "correlation":
            evidence["correlation_id"] = "vat_not_the_active_attempt"
        elif invalid_case == "transcript":
            evidence["timeline"][0]["text"] = "raw words must not persist"
        elif invalid_case == "pcm":
            evidence["pcm"] = [0, 1, 2]
        elif invalid_case == "credentials":
            evidence["echo_control_evidence"]["api_key"] = "do-not-store"
        elif invalid_case == "product_gate":
            evidence["product_gate_usable"] = True
        elif invalid_case == "nonfinite":
            evidence["aec_stats"]["erle_db"] = float("nan")
        elif invalid_case == "deep":
            evidence["echo_control_evidence"]["detail"] = {
                "a": {"b": {"c": {"d": "too deep"}}}
            }
        else:
            evidence["aec_stats"]["evidence_kind"] = "physical"
        return evidence

    service = VoiceLabService(
        tmp_path,
        audio_backend=FakeAudioBackend(),
        trial_evidence_provider=provider,
    )
    ready = ready_run(service)
    started = service.begin_trial(
        ready["run_id"],
        expected_version=ready["version"],
        idempotency_key=f"begin-invalid-{invalid_case}",
    )

    with pytest.raises(VoiceLabValidationError, match=error):
        service.execute_trial(
            ready["run_id"],
            started["active_trial"]["attempt_id"],
            expected_version=started["version"],
            idempotency_key=f"execute-invalid-{invalid_case}",
        )
    assert service.get_run(ready["run_id"]) == started
