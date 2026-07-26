"""Route-level RBAC and API-surface regression tests for voice controls."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from askme.api.composition import api_surface_for_route_module
from askme.api.routes.conversation import create_conversation_router
from askme.api.routes.voice import create_voice_router
from askme.api.routes.voice_lab import create_voice_lab_router


def _mission_json(payload: dict, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


def _cors_options_response(methods: str) -> Response:
    return Response(headers={"Access-Control-Allow-Methods": methods})


async def _optional_json_body(request: Request) -> dict:
    return await request.json()


def _forbidden(permission: str) -> JSONResponse:
    return JSONResponse(
        {"error": "forbidden", "permission": permission},
        status_code=403,
    )


class _FailIfCalledVoiceLab:
    def __getattr__(self, name: str):
        def fail(*_args: Any, **_kwargs: Any) -> dict:
            raise AssertionError(f"Voice Lab operation should not run: {name}")

        return fail


def test_voice_lab_is_admin_surface_and_all_methods_require_supervisor_permission() -> None:
    calls: list[str] = []

    def deny(_request: Request, _body: dict, permission: str) -> JSONResponse:
        calls.append(permission)
        return _forbidden(permission)

    app = FastAPI()
    app.include_router(
        create_voice_lab_router(
            service=_FailIfCalledVoiceLab(),  # type: ignore[arg-type]
            mission_json=_mission_json,
            optional_json_body=_optional_json_body,
            cors_options_response=_cors_options_response,
            authorize=deny,
        )
    )

    with TestClient(app) as client:
        assert client.get("/api/voice/lab/devices").status_code == 403
        assert client.get("/api/voice/lab/runs/run-1").status_code == 403
        assert (
            client.post(
                "/api/voice/lab/runs",
                headers={"Idempotency-Key": "denied-run"},
                json={},
            ).status_code
            == 403
        )

    assert api_surface_for_route_module("askme.api.routes.voice_lab") == "admin"
    assert calls == ["voice:system:update"] * 3


def test_voice_system_customer_view_redacts_prompt_memory_and_persistence() -> None:
    payload = {
        "status": "ready",
        "runtime": {
            "llm": {"provider": "deepseek"},
            "diagnostics": {
                "system_prompt": "nested prompt",
                "memory": {"records": [{"secret": "nested memory"}]},
                "persistence": {"path": "C:/nested/state.json"},
            },
        },
        "catalog": {},
        "prompt": {"system_prompt": "internal prompt", "persona": {"secret": "persona"}},
        "memory": {"status": "ready", "records": [{"secret": "memory"}]},
        "issues": [],
        "persistence": {"path": "C:/internal/control-state.json"},
    }

    async def dispatch(_method: str) -> dict:
        return payload

    def authorize(request: Request, _body: dict, permission: str) -> JSONResponse | None:
        role = request.headers.get("X-Test-Role", "customer")
        if permission == "voice:system:update" and role == "supervisor":
            return None
        if permission == "voice:profile:read" and role in {"customer", "supervisor"}:
            return None
        return _forbidden(permission)

    app = FastAPI()
    app.include_router(
        create_voice_router(
            dispatch_voice=dispatch,
            mission_json=_mission_json,
            optional_json_body=_optional_json_body,
            cors_options_response=_cors_options_response,
            authorize=authorize,
        )
    )

    with TestClient(app) as client:
        customer = client.get("/api/voice/system", headers={"X-Test-Role": "customer"})
        supervisor = client.get(
            "/api/voice/system",
            headers={"X-Test-Role": "supervisor"},
        )
        unknown = client.get("/api/voice/system", headers={"X-Test-Role": "unknown"})

    assert customer.status_code == 200
    customer_payload = customer.json()
    assert customer_payload["prompt"] == {"redacted": True, "configured": True}
    assert customer_payload["memory"] == {
        "redacted": True,
        "available": True,
        "status": "ready",
    }
    assert "persistence" not in customer_payload
    assert "internal prompt" not in customer.text
    assert "nested prompt" not in customer.text
    assert "nested memory" not in customer.text
    assert "C:/nested/state.json" not in customer.text

    assert supervisor.status_code == 200
    assert supervisor.json()["prompt"]["system_prompt"] == "internal prompt"
    assert supervisor.json()["memory"]["records"][0]["secret"] == "memory"
    assert supervisor.json()["persistence"]["path"].endswith("control-state.json")
    assert unknown.status_code == 403


def test_runtime_voice_turn_checks_runtime_submit_before_dispatch() -> None:
    permissions: list[str] = []

    def deny(_request: Request, _body: dict, permission: str) -> JSONResponse:
        permissions.append(permission)
        return _forbidden(permission)

    async def dispatch(*_args: Any, **_kwargs: Any) -> dict:
        raise AssertionError("runtime dispatch must not run for a forbidden request")

    app = FastAPI()
    app.include_router(
        create_conversation_router(
            conversation_service=object(),  # type: ignore[arg-type]
            runtime_available=True,
            dispatch_runtime=dispatch,
            cors_options_response=_cors_options_response,
            logger=logging.getLogger("tests.voice_route_authorization"),
            authorize=deny,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/runtime/voice-turn",
            json={"text": "暂停", "speak": True},
        )

    assert response.status_code == 403
    assert permissions == ["runtime:submit"]
