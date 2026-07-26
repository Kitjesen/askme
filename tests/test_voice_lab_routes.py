from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from askme.api.routes.voice_lab import create_voice_lab_router
from askme.voice.lab.service import VoiceLabService
from tests.test_voice_lab_service import FakeAudioBackend, run_body


async def optional_json_body(request: Request) -> dict:
    return await request.json()


def mission_json(payload: dict, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


def cors_options_response(methods: str) -> Response:
    return Response(headers={"Access-Control-Allow-Methods": methods})


def allow(_request: Request, _body: dict, _permission: str):
    return None


def client(tmp_path) -> TestClient:
    app = FastAPI()
    service = VoiceLabService(tmp_path, audio_backend=FakeAudioBackend())
    app.include_router(
        create_voice_lab_router(
            service=service,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            authorize=allow,
        )
    )
    return TestClient(app)


def test_devices_and_run_http_contract(tmp_path) -> None:
    with client(tmp_path) as http:
        devices = http.get("/api/voice/lab/devices")
        assert devices.status_code == 200
        assert devices.json()["capabilities"]["physical_overlap_stop_collector"] is False

        created = http.post(
            "/api/voice/lab/runs",
            headers={"Idempotency-Key": "browser-run-1"},
            json=run_body(),
        )
        assert created.status_code == 201
        run = created.json()
        assert run["next_action"]["action"] == "device_check"

        missing_version = http.post(
            f"/api/voice/lab/runs/{run['run_id']}/device-check",
            headers={"Idempotency-Key": "check-1"},
            json={},
        )
        assert missing_version.status_code == 428

        checked = http.post(
            f"/api/voice/lab/runs/{run['run_id']}/device-check",
            headers={
                "Idempotency-Key": "check-1",
                "If-Match": str(run["version"]),
            },
            json={},
        )
        assert checked.status_code == 200
        assert checked.json()["status"] == "needs_calibration"

        calibrated = http.post(
            f"/api/voice/lab/runs/{run['run_id']}/calibration",
            headers={
                "Idempotency-Key": "calibration-1",
                "If-Match": str(checked.json()["version"]),
            },
            json={"duration_s": 1.0},
        )
        assert calibrated.status_code == 200

        started = http.post(
            f"/api/voice/lab/runs/{run['run_id']}/trials/begin",
            headers={
                "Idempotency-Key": "begin-1",
                "If-Match": str(calibrated.json()["version"]),
            },
            json={},
        )
        assert started.status_code == 200
        assert started.json()["next_action"]["action"] == "trial_active"
        assert started.json()["active_trial"]["attempt_id"].startswith("vat_")


def test_voice_lab_openapi_declares_devices_and_run_response_contracts(tmp_path) -> None:
    with client(tmp_path) as http:
        paths = http.app.openapi()["paths"]

    devices_schema = paths["/api/voice/lab/devices"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    create_schema = paths["/api/voice/lab/runs"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["schema"]
    mutation_schema = paths["/api/voice/lab/runs/{run_id}/calibration"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]

    assert devices_schema["$ref"].endswith("/VoiceLabDevicesResponse")
    assert create_schema["$ref"].endswith("/VoiceLabRunResponse")
    assert mutation_schema["$ref"].endswith("/VoiceLabRunResponse")


def test_voice_lab_rejects_non_object_json_before_service_dispatch(tmp_path) -> None:
    with client(tmp_path) as http:
        response = http.post(
            "/api/voice/lab/runs",
            headers={"Idempotency-Key": "invalid-body"},
            json=["not", "an", "object"],
        )

    assert response.status_code == 400
    assert response.json() == {"error": "JSON object body required"}


def test_http_rejects_idempotency_conflict(tmp_path) -> None:
    with client(tmp_path) as http:
        first = http.post(
            "/api/voice/lab/runs",
            headers={"Idempotency-Key": "same"},
            json=run_body(),
        )
        changed = run_body()
        changed["room"] = "other-room"
        second = http.post(
            "/api/voice/lab/runs",
            headers={"Idempotency-Key": "same"},
            json=changed,
        )

        assert first.status_code == 201
        assert second.status_code == 409
        assert "idempotency" in second.json()["error"]
