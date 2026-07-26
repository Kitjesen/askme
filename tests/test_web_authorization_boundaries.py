"""Authorization regression tests for dashboard and operator-only voice routes."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from askme.health_server import create_health_app


def _runtime_snapshot() -> dict[str, object]:
    return {
        "status": "ok",
        "voice_pipeline_status": {"pipeline_ok": True},
    }


def _mission_json(payload: dict, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


def _cors_options_response(methods: str) -> Response:
    return Response(headers={"Access-Control-Allow-Methods": methods})


async def _optional_json_body(request: Request) -> dict:
    return await request.json()


def test_dashboard_shell_and_nested_routes_share_control_api_auth() -> None:
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            control_api_key="test-control-key",
        )
    )

    for path in ("/dashboard", "/dashboard/conversation", "/dashboard/app.js"):
        assert client.get(path).status_code == 401
        assert (
            client.get(path, headers={"Authorization": "Bearer test-control-key"}).status_code
            == 200
        )
