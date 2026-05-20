"""HTTP tests for mission draft and submission routes."""

import logging

from fastapi.testclient import TestClient

from askme.api.schemas.mission import (
    MissionDetailResponse,
    MissionDraftResponse,
    MissionListResponse,
    MissionReportResponse,
    MissionSubmitResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


def test_mission_openapi_response_schemas():
    app = create_health_app(lambda: _runtime_snapshot())
    paths = app.openapi()["paths"]
    expected_refs = {
        ("/api/missions/draft", "post"): "MissionDraftResponse",
        ("/api/missions", "get"): "MissionListResponse",
        ("/api/missions", "post"): "MissionSubmitResponse",
        ("/api/missions/{mission_id}", "get"): "MissionDetailResponse",
        ("/api/missions/{mission_id}/report", "get"): "MissionReportResponse",
    }

    for (path, method), schema_name in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        assert schema["$ref"].endswith(f"/{schema_name}")


def test_mission_endpoints_delegate_to_handler():
    class DummyMissionHandler:
        def __init__(self):
            self.mission = {
                "mission_id": "mission-1",
                "goal": "inspect area-a",
                "status": "draft",
            }

        def draft_from_payload(self, payload):
            self.mission["goal"] = payload["text"]
            return {"mission": self.mission, "drafted": True}

        def submit_from_payload(self, payload):
            self.mission["status"] = "dry_run" if payload.get("dry_run", True) else "submitted"
            return {
                "mission": self.mission,
                "submission": {"submitted": False, "dry_run": True},
            }

        def list_payload(self):
            return {"missions": [self.mission], "count": 1}

        def get_payload(self, mission_id):
            if mission_id != self.mission["mission_id"]:
                return {"error": "mission not found", "mission_id": mission_id}
            return {"mission": self.mission}

        def report_payload(self, mission_id):
            if mission_id != self.mission["mission_id"]:
                return {"error": "mission not found", "mission_id": mission_id}
            return {"report": {"mission_id": mission_id, "status": self.mission["status"]}}

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            mission_handler=DummyMissionHandler(),
        )
    )

    draft = client.post("/api/missions/draft", json={"text": "inspect area-a"})
    assert draft.status_code == 200
    MissionDraftResponse.model_validate(draft.json())
    assert draft.json()["drafted"] is True

    submit = client.post("/api/missions", json={"text": "inspect area-a", "dry_run": True})
    assert submit.status_code == 200
    MissionSubmitResponse.model_validate(submit.json())
    assert submit.json()["submission"]["dry_run"] is True

    mission_list = client.get("/api/missions")
    assert mission_list.status_code == 200
    MissionListResponse.model_validate(mission_list.json())
    assert mission_list.json()["count"] == 1

    mission_get = client.get("/api/missions/mission-1")
    assert mission_get.status_code == 200
    MissionDetailResponse.model_validate(mission_get.json())
    assert mission_get.json()["mission"]["mission_id"] == "mission-1"

    report = client.get("/api/missions/mission-1/report")
    assert report.status_code == 200
    MissionReportResponse.model_validate(report.json())
    assert report.json()["report"]["status"] == "dry_run"

    missing = client.get("/api/missions/missing")
    assert missing.status_code == 404
    MissionDetailResponse.model_validate(missing.json())


def test_mission_endpoint_returns_unconfigured_status():
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post("/api/missions/draft", json={"text": "inspect area-a"})

    assert response.status_code == 503
    assert response.json()["error"] == "mission handler not configured"


def test_mission_routes_have_unique_method_contracts():
    app = create_health_app(lambda: _runtime_snapshot())
    seen: dict[tuple[str, str], str] = {}
    duplicates: list[tuple[str, str]] = []

    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/missions"):
            continue
        for method in getattr(route, "methods", set()) or set():
            key = (path, method)
            if key in seen:
                duplicates.append(key)
            seen[key] = getattr(route, "name", "")

    assert duplicates == []
    assert ("/api/missions/draft", "POST") in seen
    assert ("/api/missions", "POST") in seen
    assert ("/api/missions", "GET") in seen
    assert ("/api/missions/{mission_id}", "GET") in seen
    assert ("/api/missions/{mission_id}/report", "GET") in seen
    assert ("/api/missions", "OPTIONS") in seen
    assert ("/api/missions/draft", "OPTIONS") in seen


def test_mission_cors_preflight_stays_public_under_control_auth():
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            control_api_key="secret",
        )
    )

    collection = client.options("/api/missions")
    draft = client.options("/api/missions/draft")
    detail = client.options("/api/missions/mission-1")
    report = client.options("/api/missions/mission-1/report")

    assert collection.status_code == 204
    assert collection.headers["access-control-allow-methods"] == "GET, POST, OPTIONS"
    assert draft.status_code == 204
    assert draft.headers["access-control-allow-methods"] == "GET, POST, OPTIONS"
    assert detail.status_code == 204
    assert detail.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert report.status_code == 204
    assert report.headers["access-control-allow-methods"] == "GET, OPTIONS"


def test_mission_submit_requires_control_auth_and_marks_trusted_confirmation():
    class TrustedMissionHandler:
        def __init__(self):
            self.trusted_confirmations: list[bool] = []

        def submit_from_payload(self, payload, trusted_confirmation: bool = False):
            self.trusted_confirmations.append(trusted_confirmation)
            return {
                "mission": {"mission_id": "mission-1", "goal": payload.get("text")},
                "submission": {"trusted_confirmation": trusted_confirmation},
            }

    handler = TrustedMissionHandler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            mission_handler=handler,
            control_api_key="secret",
        )
    )

    unauth_get = client.get("/api/missions")
    unauth_submit = client.post("/api/missions", json={"text": "inspect area-a"})
    authed_submit = client.post(
        "/api/missions",
        json={"text": "inspect area-a"},
        headers={"X-Askme-Api-Key": "secret"},
    )

    assert unauth_get.status_code == 401
    assert unauth_submit.status_code == 401
    assert authed_submit.status_code == 200
    assert authed_submit.json()["submission"]["trusted_confirmation"] is True
    assert handler.trusted_confirmations == [True]


def test_mission_rejects_non_object_json_body():
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post("/api/missions/draft", json=["inspect area-a"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_mission_submit_rejects_non_object_json_body_after_control_auth():
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            control_api_key="secret",
        )
    )

    response = client.post(
        "/api/missions",
        json=["inspect area-a"],
        headers={"X-Askme-Api-Key": "secret"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_mission_generic_exception_is_logged_and_sanitized(caplog):
    class FailingMissionHandler:
        def draft_from_payload(self, payload):
            _ = payload
            raise Exception("internal mission planner secret")

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            mission_handler=FailingMissionHandler(),
        )
    )

    with caplog.at_level(logging.ERROR, logger="askme.health_server"):
        response = client.post("/api/missions/draft", json={"text": "inspect area-a"})

    assert response.status_code == 500
    assert response.json()["error"] == "mission request failed"
    assert "Mission draft failed" in caplog.text
    assert "internal mission planner secret" in caplog.text
