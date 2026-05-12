# ruff: noqa: I001
"""Simulate a realistic askme robot task interaction over local HTTP routes.

This is a product demo harness, not a hardware runner. It exercises the same
planning/runtime handoff services that back the Dashboard routes while keeping
runtime profile `sim` by default and `hardware_dispatch` false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService  # noqa: E402
from askme.health_server import create_health_app  # noqa: E402
from askme.runtime.audit import RuntimeAuditConfig  # noqa: E402
from askme.runtime.handoff import RuntimeHandoffService  # noqa: E402
from askme.runtime.mission import MissionService  # noqa: E402


DEFAULT_AUDIT_PATH = Path("artifacts/runtime_handoff/simulation-audit.jsonl")
DEFAULT_TRANSCRIPT_PATH = Path("artifacts/runtime_handoff/simulation-transcript.json")


class InteractionHarness:
    """Small HTTP-backed harness for operator command -> TaskRun lifecycle."""

    def __init__(
        self,
        *,
        profile: str,
        audit_path: Path | None,
        estop_active: bool = False,
    ) -> None:
        self.world = _seed_world(estop_active=estop_active)
        self.planner = CognitivePlanner(
            world_state=self.world,
            working_memory=WorkingMemory(),
            mission_service=MissionService(),
        )
        self.runtime = RuntimeHandoffService(
            world_state=self.world,
            profile=profile,
            auto_complete=False if profile == "sim" else True,
            audit_config=RuntimeAuditConfig(
                enabled=audit_path is not None,
                path=audit_path,
            ),
        )
        self._active_session_id = ""
        self._last_plan_id = ""

    def client(self) -> TestClient:
        app = create_health_app(
            lambda: {
                "status": "ok",
                "service": "askme-simulation",
                "runtime_profile": self.runtime.profile,
                "hardware_dispatch": False,
            },
            chat_handler=self.chat,
            runtime_handler=self.runtime,
        )
        return TestClient(app)

    async def chat(self, text: str, *, speak: bool = False) -> dict[str, Any]:
        runtime_control = self.runtime.handle_chat_control(text)
        if runtime_control is not None:
            return {
                "reply": str(runtime_control.get("reply", "")),
                "runtime": runtime_control.get("runtime", runtime_control),
                "spoken": False if speak else None,
            }

        confirmation = _is_confirmation(text)
        plan = self.planner.plan_from_text(
            "" if confirmation and self._active_session_id else text,
            planning_session_id=self._active_session_id if confirmation else None,
            operator_confirmation=True if confirmation else None,
            operator_id="operator-demo",
            robot_id="dog-1",
            site_id="factory-demo",
            channel="http-demo",
        )
        self._active_session_id = plan.planning_session_id
        self._last_plan_id = plan.plan_id

        cognition = {"handled": True, "plan": plan.to_dict()}
        payload: dict[str, Any] = {
            "reply": plan.next_prompt or _state_reply(plan.interaction_state),
            "cognition": cognition,
        }
        if plan.handoff_ready:
            runtime_result = self.runtime.submit_plan_payload(plan.to_dict())
            cognition["runtime"] = runtime_result
            payload["runtime"] = runtime_result
        if speak:
            payload["spoken"] = False
        return payload


def run_simulation(
    *,
    profile: str = "sim",
    audit_path: Path | None = DEFAULT_AUDIT_PATH,
    transcript_path: Path | None = DEFAULT_TRANSCRIPT_PATH,
    include_blocked_demo: bool = True,
) -> dict[str, Any]:
    if audit_path is not None and audit_path.exists():
        audit_path.unlink()

    harness = InteractionHarness(profile=profile, audit_path=audit_path)
    transcript: list[dict[str, Any]] = []
    with harness.client() as client:
        draft = _chat(client, "巡检 A 区")
        transcript.append(_entry("operator", "巡检 A 区"))
        transcript.append(_chat_entry(draft))

        confirmed = _chat(client, "确认")
        transcript.append(_entry("operator", "确认"))
        transcript.append(_chat_entry(confirmed))

        run = confirmed.get("runtime", {}).get("run", {})
        run_id = str(run.get("run_id", ""))
        if not run_id:
            raise RuntimeError("simulation did not produce a TaskRun")

        status = _voice_turn(client, "现在执行到哪了", transcript_id="voice-status-1")
        transcript.append(_entry("operator", "现在执行到哪了"))
        transcript.append(_chat_entry(status))

        first_advance = _post(client, f"/api/runtime/runs/{run_id}/advance")
        transcript.append(_runtime_entry("advance", first_advance))

        pause = _voice_turn(client, "先停一下", transcript_id="voice-pause-1")
        transcript.append(_entry("operator", "先停一下"))
        transcript.append(_chat_entry(pause))

        paused_advance = _post(client, f"/api/runtime/runs/{run_id}/advance")
        transcript.append(_runtime_entry("advance while paused", paused_advance))

        resume = _voice_turn(client, "继续", transcript_id="voice-resume-1")
        transcript.append(_entry("operator", "继续"))
        transcript.append(_chat_entry(resume))

        current = resume.get("runtime", {}).get("run", {})
        while current.get("terminal") is not True:
            advanced = _post(client, f"/api/runtime/runs/{run_id}/advance")
            transcript.append(_runtime_entry("advance", advanced))
            current = advanced.get("run", {})

        report = _get(client, f"/api/runtime/runs/{run_id}/report")
        transcript.append(_runtime_entry("report", report))

        event_stream = _sse_snapshot(client, "/api/runtime/events?once=1")
        transcript.append(_runtime_entry("event stream snapshot", event_stream))

        context = _get(client, "/api/runtime/context")

    blocked_payload = None
    perception_payload = None
    if include_blocked_demo:
        blocked_payload = _run_blocked_preflight_demo(profile=profile)
        perception_payload = _run_perception_request_demo(profile=profile)

    result = {
        "profile": profile,
        "hardware_dispatch": False,
        "transcript": transcript,
        "final_context": context,
        "report": report,
        "event_stream": event_stream,
        "audit_path": str(audit_path) if audit_path is not None else "",
        "audit_records": _audit_record_count(audit_path),
        "blocked_preflight_demo": blocked_payload,
        "perception_request_demo": perception_payload,
    }
    if transcript_path is not None:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def _seed_world(*, estop_active: bool) -> WorldStateService:
    world = WorldStateService()
    world.update_robot_state(
        {
            "robot_id": "dog-1",
            "online": True,
            "battery_percent": 86,
            "estop_active": estop_active,
            "localized": True,
            "current_area": "dock",
        },
        source="simulation",
        stale_after_s=60.0,
    )
    world.update_area_catalog(
        [
            {"area_id": "area-a", "name": "A 区", "allowed": True},
            {"area_id": "dock", "name": "充电区", "allowed": True},
        ],
        source="simulation",
        map_id="map-main",
        map_version="v1",
        stale_after_s=120.0,
    )
    world.update_device_catalog(
        [
            {
                "device_id": "panel-3",
                "name": "3 号设备面板",
                "area_id": "area-a",
                "device_type": "status_panel",
                "status": "normal",
            }
        ],
        source="simulation",
        stale_after_s=120.0,
    )
    world.update_map_state(
        map_id="map-main",
        map_version="v1",
        localized=True,
        localization_quality=0.92,
        source="simulation",
        stale_after_s=60.0,
    )
    world.update_scene(
        summary="A 区入口可见，通道无遮挡，3 号设备面板可见。",
        objects=[
            {
                "label": "equipment-panel-3",
                "class_id": "status_panel",
                "confidence": 0.92,
                "distance_m": 2.4,
                "track_id": "panel-3",
            }
        ],
        source="simulation",
        stale_after_s=60.0,
    )
    world.record_event(
        "perception.area_clear",
        {"area_id": "area-a", "confidence": 0.88},
        source="simulation",
    )
    return world


def _run_blocked_preflight_demo(*, profile: str) -> dict[str, Any]:
    harness = InteractionHarness(profile=profile, audit_path=None, estop_active=True)
    with harness.client() as client:
        _chat(client, "巡检 A 区")
        confirmed = _chat(client, "确认")
    runtime = confirmed.get("runtime", {})
    return {
        "accepted": runtime.get("accepted"),
        "state": runtime.get("run", {}).get("current_state"),
        "failed_checks": runtime.get("preflight", {}).get("failed_checks", []),
        "perception_requests": runtime.get("preflight", {}).get("perception_requests", []),
        "replan_proposal": runtime.get("replan_proposal", {}),
        "recommended_fix": runtime.get("preflight", {}).get("recommended_fix", ""),
    }


def _run_perception_request_demo(*, profile: str) -> dict[str, Any]:
    harness = InteractionHarness(profile=profile, audit_path=None, estop_active=False)
    harness.world.update_map_state(
        map_id="map-main",
        map_version="v1",
        localized=False,
        localization_quality=0.1,
        source="simulation",
        stale_after_s=60.0,
    )
    with harness.client() as client:
        _chat(client, "巡检 A 区")
        confirmed = _chat(client, "确认")
    runtime = confirmed.get("runtime", {})
    return {
        "accepted": runtime.get("accepted"),
        "state": runtime.get("run", {}).get("current_state"),
        "failed_checks": runtime.get("preflight", {}).get("failed_checks", []),
        "perception_requests": runtime.get("preflight", {}).get("perception_requests", []),
        "replan_proposal": runtime.get("replan_proposal", {}),
        "recommended_fix": runtime.get("preflight", {}).get("recommended_fix", ""),
    }


def _chat(client: TestClient, text: str) -> dict[str, Any]:
    return _post(client, "/api/chat", {"text": text, "speak": False})


def _voice_turn(client: TestClient, text: str, *, transcript_id: str) -> dict[str, Any]:
    return _post(
        client,
        "/api/runtime/voice-turn",
        {
            "text": text,
            "transcript_id": transcript_id,
            "confidence": 0.92,
            "is_final": True,
        },
    )


def _post(client: TestClient, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.post(path, json=payload or {})
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"{path} returned non-object payload")
    return body


def _get(client: TestClient, path: str) -> dict[str, Any]:
    response = client.get(path)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"{path} returned non-object payload")
    return body


def _sse_snapshot(client: TestClient, path: str) -> dict[str, Any]:
    response = client.get(path)
    response.raise_for_status()
    return {
        "handled": True,
        "run": {},
        "reply": "SSE snapshot received.",
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "contains_runtime_events": "event: runtime.events" in response.text,
    }


def _entry(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "text": text}


def _chat_entry(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("cognition", {}).get("plan", {})
    runtime = payload.get("runtime", {})
    return {
        "role": "askme",
        "reply": payload.get("reply", ""),
        "planning_state": plan.get("interaction_state"),
        "handoff_ready": plan.get("handoff_ready"),
        "runtime_state": runtime.get("run", {}).get("current_state"),
        "runtime_accepted": runtime.get("accepted"),
        "run_id": runtime.get("run", {}).get("run_id"),
        "voice_turn": payload.get("voice_turn"),
    }


def _runtime_entry(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("run", {})
    report = payload.get("report", {})
    return {
        "role": "runtime",
        "action": action,
        "handled": payload.get("handled"),
        "state": run.get("current_state") or report.get("status"),
        "current_step_index": run.get("current_step_index"),
        "terminal": run.get("terminal"),
        "reply": payload.get("reply") or report.get("summary") or "",
        "failed_reason": payload.get("reason", ""),
    }


def _is_confirmation(text: str) -> bool:
    return str(text or "").strip().lower() in {"确认", "可以", "ok", "confirm", "yes"}


def _state_reply(state: str) -> str:
    if state == "awaiting_confirmation":
        return "已生成任务草案，请确认。"
    if state == "ready_for_arbiter":
        return "计划已确认，正在提交运行时。"
    return f"当前规划状态：{state}"


def _audit_record_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _print_summary(result: dict[str, Any]) -> None:
    print("=== askme robot interaction simulation ===")
    print(f"profile: {result['profile']}")
    print(f"hardware_dispatch: {result['hardware_dispatch']}")
    print("")
    for item in result["transcript"]:
        if item["role"] == "operator":
            print(f"operator> {item['text']}")
        elif item["role"] == "askme":
            print(
                "askme> "
                f"{item['reply']} "
                f"[planning={item['planning_state']}, "
                f"runtime={item['runtime_state']}]"
            )
        else:
            detail = item["reply"] or item["failed_reason"]
            print(
                "runtime> "
                f"{item['action']} -> {item['state']} "
                f"step={item['current_step_index']} terminal={item['terminal']} {detail}"
            )
    print("")
    report = result.get("report", {}).get("report", {})
    print(f"report: {report.get('summary', '')}")
    print(f"audit_records: {result.get('audit_records', 0)}")
    if result.get("audit_path"):
        print(f"audit_path: {result['audit_path']}")
    blocked = result.get("blocked_preflight_demo") or {}
    if blocked:
        print("")
        print("blocked_preflight_demo:")
        print(json.dumps(blocked, ensure_ascii=False, indent=2))
    perception = result.get("perception_request_demo") or {}
    if perception:
        print("")
        print("perception_request_demo:")
        print(json.dumps(perception, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["sim", "fake", "shadow"], default="sim")
    parser.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--transcript-path", default=str(DEFAULT_TRANSCRIPT_PATH))
    parser.add_argument("--no-blocked-demo", action="store_true")
    args = parser.parse_args(argv)

    result = run_simulation(
        profile=args.profile,
        audit_path=Path(args.audit_path) if args.audit_path else None,
        transcript_path=Path(args.transcript_path) if args.transcript_path else None,
        include_blocked_demo=not args.no_blocked_demo,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
