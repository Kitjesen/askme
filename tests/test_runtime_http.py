"""HTTP tests for runtime control and handoff routes."""

from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.api.schemas.runtime import (
    RuntimeContextResponse,
    RuntimeHandoffSubmitResponse,
    RuntimeProfilesResponse,
    RuntimeRunActionResponse,
    RuntimeRunDetailResponse,
    RuntimeRunListResponse,
    RuntimeRunReportResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


def test_runtime_openapi_response_schemas():
    app = create_health_app(lambda: _runtime_snapshot())
    paths = app.openapi()["paths"]
    expected_refs = {
        ("/api/runtime/context", "get"): "RuntimeContextResponse",
        ("/api/runtime/profiles", "get"): "RuntimeProfilesResponse",
        ("/api/runtime/runs", "get"): "RuntimeRunListResponse",
        ("/api/runtime/handoff", "post"): "RuntimeHandoffSubmitResponse",
        ("/api/runtime/runs/{run_id}", "get"): "RuntimeRunDetailResponse",
        ("/api/runtime/runs/{run_id}/report", "get"): "RuntimeRunReportResponse",
        ("/api/runtime/runs/{run_id}/pause", "post"): "RuntimeRunActionResponse",
        ("/api/runtime/runs/{run_id}/resume", "post"): "RuntimeRunActionResponse",
        ("/api/runtime/runs/{run_id}/cancel", "post"): "RuntimeRunActionResponse",
        ("/api/runtime/runs/{run_id}/advance", "post"): "RuntimeRunActionResponse",
    }

    for (path, method), schema_name in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith(f"/{schema_name}")


def test_runtime_endpoints_delegate_to_handler():
    class DummyRuntimeHandler:
        def context_payload(self):
            return {
                "profile": "sim",
                "active_run": {"run_id": "run-1", "current_state": "queued"},
            }

        def profiles_payload(self):
            return {
                "current_profile": "sim",
                "profiles": [
                    {"name": "fake"},
                    {"name": "shadow"},
                    {"name": "sim"},
                ],
            }

        def list_payload(self):
            return {"runs": [{"run_id": "run-1"}], "count": 1}

        def submit_plan_payload(self, plan):
            return {
                "accepted": True,
                "handoff": {"task_type": plan["mission"]["mission"]["mission_type"]},
                "run": {"run_id": "run-submitted", "current_state": "completed"},
            }

        def events_payload(self, *, after=None, limit=20):
            return {
                "profile": "sim",
                "hardware_dispatch": False,
                "cursor": 123.0,
                "events": [
                    {
                        "event_id": "evt-1",
                        "run_id": "run-1",
                        "event_type": "task_queued",
                        "state": "queued",
                        "message": "queued",
                        "created_at": 123.0,
                    }
                ],
                "event_count": 1,
                "active_run": {"run_id": "run-1", "current_state": "queued"},
            }

        def get_payload(self, run_id):
            return {"run": {"run_id": run_id, "current_state": "queued"}}

        def report_payload(self, run_id):
            return {"report": {"run_id": run_id, "status": "queued"}}

        def pause_payload(self, run_id):
            return {
                "handled": True,
                "run": {"run_id": run_id, "current_state": "paused"},
            }

        def resume_payload(self, run_id):
            return {
                "handled": True,
                "run": {"run_id": run_id, "current_state": "executing"},
            }

        def cancel_payload(self, run_id):
            return {
                "handled": True,
                "run": {"run_id": run_id, "current_state": "cancelled"},
            }

        def advance_payload(self, run_id):
            return {
                "handled": True,
                "run": {"run_id": run_id, "current_state": "executing"},
            }

        def voice_turn_payload(self, text, **kwargs):
            return {
                "handled": True,
                "reply": "TaskRun paused.",
                "runtime": {"run": {"run_id": "run-1", "current_state": "paused"}},
                "voice_turn": {
                    "recognized_text": text,
                    "runtime_control_intent": "pause",
                    "safety_bypass_allowed": False,
                    "transcript_id": kwargs.get("transcript_id", ""),
                },
            }

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=DummyRuntimeHandler(),
        )
    )

    context = client.get("/api/runtime/context")
    assert context.status_code == 200
    RuntimeContextResponse.model_validate(context.json())
    assert context.json()["active_run"]["run_id"] == "run-1"
    assert context.json()["profile"] == "sim"

    profiles = client.get("/api/runtime/profiles")
    assert profiles.status_code == 200
    RuntimeProfilesResponse.model_validate(profiles.json())
    assert profiles.json()["current_profile"] == "sim"

    runs = client.get("/api/runtime/runs")
    assert runs.status_code == 200
    RuntimeRunListResponse.model_validate(runs.json())
    assert runs.json()["count"] == 1

    submitted = client.post(
        "/api/runtime/handoff",
        json={
            "operator_id": "dashboard.operator",
            "runtime_handoff_plan": {
                "plan_id": "space-escort-test",
                "planning_session_id": "space-session-test",
                "intent": "visitor_escort",
                "handoff_ready": True,
                "operator_id": "dashboard.operator",
                "mission": {"mission": {"mission_type": "visitor_escort"}},
            },
        },
    )
    assert submitted.status_code == 200
    RuntimeHandoffSubmitResponse.model_validate(submitted.json())
    assert submitted.json()["run"]["run_id"] == "run-submitted"
    assert submitted.json()["handoff"]["task_type"] == "visitor_escort"

    events = client.get("/api/runtime/events?once=1")
    assert events.status_code == 200
    assert "text/event-stream" in events.headers["content-type"]
    assert "event: runtime.events" in events.text
    assert '"event_type":"task_queued"' in events.text

    run = client.get("/api/runtime/runs/run-1")
    assert run.status_code == 200
    RuntimeRunDetailResponse.model_validate(run.json())
    assert run.json()["run"]["current_state"] == "queued"

    report = client.get("/api/runtime/runs/run-1/report")
    assert report.status_code == 200
    RuntimeRunReportResponse.model_validate(report.json())
    assert report.json()["report"]["status"] == "queued"

    missing_operator = client.post("/api/runtime/runs/run-1/pause")
    assert missing_operator.status_code == 403
    assert missing_operator.json()["reason"] == "runtime_operator_context_required"

    paused = client.post(
        "/api/runtime/runs/run-1/pause", headers={"X-Askme-Operator-Id": "dashboard.operator"}
    )
    assert paused.status_code == 200
    RuntimeRunActionResponse.model_validate(paused.json())
    assert paused.json()["run"]["current_state"] == "paused"

    resumed = client.post(
        "/api/runtime/runs/run-1/resume", headers={"X-Askme-Operator-Id": "dashboard.operator"}
    )
    assert resumed.status_code == 200
    RuntimeRunActionResponse.model_validate(resumed.json())
    assert resumed.json()["run"]["current_state"] == "executing"

    cancelled_forbidden = client.post("/api/runtime/runs/run-1/cancel")
    assert cancelled_forbidden.status_code == 403

    cancelled = client.post(
        "/api/runtime/runs/run-1/cancel",
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert cancelled.status_code == 200
    RuntimeRunActionResponse.model_validate(cancelled.json())
    assert cancelled.json()["run"]["current_state"] == "cancelled"

    advanced = client.post(
        "/api/runtime/runs/run-1/advance",
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert advanced.status_code == 200
    RuntimeRunActionResponse.model_validate(advanced.json())
    assert advanced.json()["run"]["current_state"] == "executing"

    voice = client.post(
        "/api/runtime/voice-turn",
        json={
            "operator_id": "dashboard.operator",
            "text": "pause current task",
            "transcript_id": "voice-1",
            "confidence": 0.9,
        },
    )
    assert voice.status_code == 200
    assert voice.json()["runtime"]["run"]["current_state"] == "paused"
    assert voice.json()["voice_turn"]["recognized_text"] == "pause current task"
    assert voice.json()["voice_turn"]["safety_bypass_allowed"] is False


def test_runtime_handoff_rejects_non_object_json_body_before_dispatch():
    class DummyRuntimeHandler:
        def submit_plan_payload(self, plan):
            raise AssertionError(f"runtime handoff handler should not be called: {plan}")

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=DummyRuntimeHandler(),
        )
    )

    response = client.post("/api/runtime/handoff", json=["plan"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_runtime_voice_turn_rejects_non_object_json_body_before_dispatch():
    class DummyRuntimeHandler:
        def voice_turn_payload(self, text, **kwargs):
            raise AssertionError("runtime handler should not be called")

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=DummyRuntimeHandler(),
        )
    )

    response = client.post("/api/runtime/voice-turn", json=["pause current task"])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_runtime_voice_turn_endpoint_reports_timeout_from_config(monkeypatch):
    monkeypatch.setattr(
        health_server,
        "get_config",
        lambda: {
            "conversation": {"runtime_voice_turn_timeout_s": 0.001},
            "field_operations": {"operators": {"dashboard.operator": {"roles": ["operator"]}}},
        },
    )

    class SlowRuntimeHandler:
        async def voice_turn_payload(self, text, **kwargs):
            import asyncio

            await asyncio.sleep(0.05)
            return {"handled": True, "reply": text}

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=SlowRuntimeHandler(),
        )
    )

    response = client.post(
        "/api/runtime/voice-turn",
        json={"operator_id": "dashboard.operator", "text": "pause current task"},
    )

    assert response.status_code == 504
    assert response.json()["error"] == "runtime voice-turn timed out"


def test_runtime_control_endpoint_forwards_sanitized_operator_context(monkeypatch):
    monkeypatch.setattr(
        health_server,
        "get_config",
        lambda: {
            "field_operations": {
                "operators": {
                    "guard-1": {"roles": ["operator"]},
                    "attacker-1": {"roles": ["admin"]},
                }
            }
        },
    )

    class DummyRuntimeHandler:
        def __init__(self):
            self.seen = {}

        def pause_payload(
            self,
            run_id,
            *,
            operator_id="askme.operator",
            reason="",
            risk_acknowledgement=False,
            operator_context=None,
        ):
            self.seen = {
                "run_id": run_id,
                "operator_id": operator_id,
                "reason": reason,
                "risk_acknowledgement": risk_acknowledgement,
                "operator_context": operator_context,
            }
            return {
                "handled": True,
                "run": {"run_id": run_id, "current_state": "paused"},
                "operator": self.seen,
            }

    runtime = DummyRuntimeHandler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    response = client.post(
        "/api/runtime/runs/run-7/pause",
        json={
            "operator_id": "guard-1",
            "reason": "visitor entered path",
            "conversation_session_id": "conv-http-1",
            "risk_acknowledgement": True,
            "operator_auth": {
                "allowed": True,
                "permission": "runtime:pause",
                "operator": {
                    "operator_id": "attacker-1",
                    "roles": ["admin"],
                    "known": True,
                    "authenticated": True,
                    "source": "forged-body",
                },
            },
        },
    )

    assert response.status_code == 200
    assert runtime.seen == {
        "run_id": "run-7",
        "operator_id": "guard-1",
        "reason": "visitor entered path",
        "risk_acknowledgement": True,
        "operator_context": {
            "operator_id": "guard-1",
            "roles": ["operator"],
            "authenticated": False,
            "source": "local_config",
            "permission": "runtime:pause",
            "conversation_session_id": "conv-http-1",
        },
    }
