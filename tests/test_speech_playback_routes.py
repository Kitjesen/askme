from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from askme.api.routes.voice import create_voice_router


def _job(playback_id: str, state: str = "queued") -> dict[str, Any]:
    return {
        "playback_id": playback_id,
        "state": state,
        "target": {
            "robot_id": "robot-1",
            "device_id": "speaker-1",
            "site_id": "site-a",
            "channel": "speaker",
        },
        "delivery": "playback",
        "priority": "normal",
        "text_chars": 11,
        "idempotency_key": "request-1",
        "timestamps": {"queued_at": "2026-07-23T00:00:00Z"},
        "cache_hit": False,
        "customer_message": "queued",
    }


def _client(*, deny: set[str] | None = None, artifact_path=None):
    calls: list[tuple[str, tuple[Any, ...]]] = []
    permissions: list[str] = []

    async def dispatch(method: str, *args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append((method, args))
        if method == "speak_payload":
            return _job("spk_1")
        if method == "speech_playback_status_payload":
            return _job(str(args[0]), "playing")
        if method == "cancel_speech_playback_payload":
            return _job(str(args[0]), "cancelled")
        if method == "speech_playback_audio_payload":
            return {
                "path": str(artifact_path),
                "filename": "preview.wav",
                "media_type": "audio/wav",
                "sha256": "abc",
            }
        if method == "synthesize_speech_payload":
            result = _job("spk_wav", "queued")
            result["delivery"] = "synthesize_only"
            return result
        raise AssertionError(method)

    def authorize(_request: Request, body: dict[str, Any], permission: str):
        permissions.append(permission)
        if permission in (deny or set()):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        body["operator_id"] = "operator-1"
        body["operator_auth"] = {
            "operator": {"operator_id": "operator-1", "roles": ["operator"]}
        }
        return None

    def mission_json(payload: dict[str, Any], *, status_code: int = 200):
        return JSONResponse(payload, status_code=status_code)

    async def optional_json_body(request: Request) -> dict[str, Any]:
        return await request.json()

    app = FastAPI()
    app.include_router(
        create_voice_router(
            dispatch_voice=dispatch,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=lambda methods: Response(
                headers={"Access-Control-Allow-Methods": methods}
            ),
            authorize=authorize,
        )
    )
    return TestClient(app), calls, permissions


def test_post_voice_speak_is_verbatim_async_and_idempotent() -> None:
    client, calls, permissions = _client()

    response = client.post(
        "/api/voice/speak",
        headers={"Idempotency-Key": "request-1"},
        json={
            "text": "Hello robot",
            "robot_id": "robot-1",
            "device_id": "speaker-1",
            "site_id": "site-a",
        },
    )

    assert response.status_code == 202
    assert response.headers["location"] == "/api/voice/playbacks/spk_1"
    assert response.json()["state"] == "queued"
    assert permissions == ["voice:playback:create"]
    method, args = calls[0]
    assert method == "speak_payload"
    assert args[0]["semantics"] == "verbatim"
    assert args[0]["idempotency_key"] == "request-1"


def test_voice_speak_rejects_conversational_semantics() -> None:
    client, calls, _permissions = _client()

    response = client.post(
        "/api/voice/speak",
        json={
            "text": "What is here?",
            "semantics": "conversational",
            "robot_id": "robot-1",
            "device_id": "speaker-1",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == "semantic_mode_mismatch"
    assert response.json()["conversation_endpoint"] == "/api/chat"
    assert calls == []


def test_playback_status_and_cancel_have_independent_permissions() -> None:
    client, calls, permissions = _client()

    status = client.get("/api/voice/playbacks/spk_1")
    cancelled = client.post(
        "/api/voice/playbacks/spk_1/cancel",
        json={"reason": "operator_cancelled"},
    )

    assert status.status_code == 200
    assert status.json()["state"] == "playing"
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert permissions == ["voice:playback:read", "voice:playback:cancel"]
    assert [item[0] for item in calls] == [
        "speech_playback_status_payload",
        "cancel_speech_playback_payload",
    ]


def test_authorization_denial_happens_before_playback_dispatch() -> None:
    client, calls, _permissions = _client(deny={"voice:playback:create"})

    response = client.post(
        "/api/voice/speak",
        json={
            "text": "Do not dispatch",
            "robot_id": "robot-1",
            "device_id": "speaker-1",
        },
    )

    assert response.status_code == 403
    assert calls == []


def test_synthesize_only_is_a_separate_endpoint() -> None:
    client, calls, permissions = _client()

    response = client.post(
        "/api/voice/synthesize",
        headers={"Idempotency-Key": "wav-1"},
        json={
            "text": "Preview only",
            "robot_id": "robot-1",
            "device_id": "speaker-1",
        },
    )

    assert response.status_code == 202
    assert response.json()["delivery"] == "synthesize_only"
    assert calls[0][0] == "synthesize_speech_payload"
    assert permissions == ["voice:synthesis:create"]


def test_synthesized_audio_has_authorized_download_endpoint(tmp_path) -> None:
    wav = tmp_path / "preview.wav"
    wav.write_bytes(b"RIFFtest")
    client, calls, permissions = _client(artifact_path=wav)

    response = client.get("/api/voice/playbacks/spk_wav/audio")

    assert response.status_code == 200
    assert response.content == b"RIFFtest"
    assert response.headers["content-type"] == "audio/wav"
    assert calls[0][0] == "speech_playback_audio_payload"
    assert permissions == ["voice:playback:read"]
