"""Tests for the safe mission adapter."""

from __future__ import annotations

from typing import Any

from askme.runtime.mission import MissionService


def test_draft_inspection_patrol_requires_confirmation() -> None:
    service = MissionService()

    plan = service.draft(
        "inspect area-a",
        operator_id="operator-1",
        robot_id="dog-1",
        site_id="factory-1",
    )

    assert plan.mission_type == "inspection_patrol"
    assert plan.risk_tier == "high"
    assert plan.requires_confirmation is True
    assert plan.approval_required is True
    assert "safety" in plan.required_services
    assert "control" in plan.required_services
    assert plan.robot_id == "dog-1"
    assert plan.site_id == "factory-1"
    assert all("cmd_vel" not in step.capability for step in plan.steps)


def test_dry_run_submit_does_not_call_runtime(monkeypatch) -> None:
    service = MissionService({
        "runtime": {
            "mission": {
                "submit_enabled": True,
                "base_url": "http://arbiter.test",
            }
        }
    })
    plan = service.draft("inspect area-a")

    def _fail_post(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry-run must not call requests.post")

    monkeypatch.setattr("askme.runtime.mission.requests.post", _fail_post)

    result = service.submit(plan, dry_run=True, confirmed=True)

    assert result["submission"]["submitted"] is False
    assert result["submission"]["dry_run"] is True
    assert result["mission"]["status"] == "dry_run"


def test_live_submit_requires_confirmation_before_http(monkeypatch) -> None:
    service = MissionService({
        "runtime": {
            "mission": {
                "submit_enabled": True,
                "base_url": "http://arbiter.test",
            }
        }
    })
    plan = service.draft("inspect area-a")

    def _fail_post(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unconfirmed mission must not call requests.post")

    monkeypatch.setattr("askme.runtime.mission.requests.post", _fail_post)

    result = service.submit(plan, dry_run=False, confirmed=False)

    assert result["submission"]["submitted"] is False
    assert result["submission"]["reason"] == "confirmation_required"
    assert result["mission"]["status"] == "pending_confirmation"


def test_payload_confirm_flag_is_not_trusted_by_default(monkeypatch) -> None:
    service = MissionService({
        "runtime": {
            "mission": {
                "submit_enabled": True,
                "base_url": "http://arbiter.test",
            }
        }
    })

    def _fail_post(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("untrusted payload confirmation must not call requests.post")

    monkeypatch.setattr("askme.runtime.mission.requests.post", _fail_post)

    result = service.submit_from_payload({
        "text": "inspect area-a",
        "dry_run": False,
        "confirmed": True,
    })

    assert result["submission"]["submitted"] is False
    assert result["submission"]["reason"] == "confirmation_required"
    assert result["mission"]["status"] == "pending_confirmation"


def test_live_submit_uses_runtime_create_mission_shape(monkeypatch) -> None:
    service = MissionService({
        "runtime": {
            "mission": {
                "submit_enabled": True,
                "base_url": "http://arbiter.test",
                "operator_id": "operator-1",
            }
        }
    })
    plan = service.draft(
        "status report",
        metadata={"ticket": "A-1", "_adapter": {"local": True}},
    )
    seen: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"runtime_mission_id": "runtime-1"}

    def _fake_post(url: str, **kwargs: Any) -> FakeResponse:
        seen["url"] = url
        seen.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("askme.runtime.mission.requests.post", _fake_post)

    result = service.submit(plan, dry_run=False, confirmed=True)

    assert result["submission"]["submitted"] is True
    assert seen["url"] == "http://arbiter.test/api/v1/missions"
    payload = seen["json"]
    assert payload["mission_type"] == "status_report"
    assert payload["requested_capability"] == "status_report"
    assert payload["requested_by"] == "operator-1"
    assert payload["channel"] == "text"
    assert payload["parameters"]["ticket"] == "A-1"
    assert "_adapter" not in payload["parameters"]
    assert seen["headers"]["X-Operator-Id"] == "operator-1"
    assert seen["headers"]["Idempotency-Key"].startswith("mission-")


def test_critical_action_is_blocked() -> None:
    service = MissionService({
        "runtime": {
            "mission": {
                "submit_enabled": True,
                "base_url": "http://arbiter.test",
            }
        }
    })

    result = service.submit_from_payload({
        "text": "emergency stop and disable safety",
        "dry_run": False,
        "confirmed": True,
    })

    assert result["mission"]["risk_tier"] == "critical"
    assert result["mission"]["status"] == "blocked"
    assert result["submission"]["submitted"] is False
    assert result["submission"]["reason"] == "critical_action_requires_safety_service"


def test_report_payload_uses_evidence_and_status() -> None:
    service = MissionService()
    plan = service.draft("capture photo at area-a")
    plan.status = "dry_run"
    plan.evidence.append({"kind": "image", "id": "capture-1"})

    payload = service.report_payload(plan.mission_id)

    assert payload["report"]["mission_id"] == plan.mission_id
    assert payload["report"]["status"] == "dry_run"
    assert payload["report"]["findings"] == [{"kind": "image", "id": "capture-1"}]
    assert payload["report"]["media"] == [{"kind": "image", "id": "capture-1"}]
