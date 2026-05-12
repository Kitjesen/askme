"""Runtime-side helpers for posting auditable field-event callbacks."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from typing import Any
from urllib import request

FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG = "hmac-sha256"
FIELD_RUNTIME_CALLBACK_SIGNATURE_FIELDS = frozenset(
    {
        "runtime_signature",
        "signature",
        "x_signature",
        "runtime_signature_alg",
        "signature_alg",
    }
)
FIELD_RUNTIME_DELIVERY_STATUSES = frozenset(
    {
        "policy_ready",
        "submitted",
        "created",
        "validating",
        "preflight",
        "queued",
        "executing",
        "paused",
        "resuming",
        "blocked",
        "completed",
        "shadowed",
        "failed",
        "cancelling",
        "cancelled",
        "rejected",
        "skipped",
        "submission_failed",
    }
)


def unsigned_field_runtime_callback_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical callback body used for signing and id derivation."""

    return {
        key: value
        for key, value in body.items()
        if key not in FIELD_RUNTIME_CALLBACK_SIGNATURE_FIELDS
    }


def sign_field_runtime_callback_payload(body: dict[str, Any], *, secret: str) -> str:
    """Sign a runtime callback body using the askme runtime-delivery contract."""

    encoded = json.dumps(
        unsigned_field_runtime_callback_payload(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(str(secret).encode("utf-8"), encoded, hashlib.sha256).hexdigest()


def derive_field_runtime_callback_id(body: dict[str, Any]) -> str:
    """Create a stable id when a runtime producer did not provide one."""

    encoded = json.dumps(
        unsigned_field_runtime_callback_payload(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_field_runtime_callback_payload(
    *,
    status: str,
    secret: str | None = None,
    runtime_callback_id: str | None = None,
    run_id: str = "",
    handoff_id: str = "",
    dispatch_mode: str = "task_handoff",
    robot_motion_policy: str = "",
    hardware_dispatch: bool = False,
    reason: str = "",
    timestamp: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload a lab/shadow/robot runtime should POST back to askme."""

    clean_status = str(status or "").strip()
    if clean_status not in FIELD_RUNTIME_DELIVERY_STATUSES:
        raise ValueError(f"unsupported field runtime callback status: {clean_status}")

    body: dict[str, Any] = {
        "runtime_callback_id": runtime_callback_id or f"rtc-{uuid.uuid4().hex}",
        "status": clean_status,
        "dispatch_mode": str(dispatch_mode or "task_handoff"),
        "robot_motion_policy": str(robot_motion_policy or ""),
        "hardware_dispatch": bool(hardware_dispatch),
        "run_id": str(run_id or ""),
        "handoff_id": str(handoff_id or ""),
        "reason": str(reason or ""),
        "runtime_signature_timestamp": round(time.time() if timestamp is None else timestamp, 3),
    }
    if extra:
        for key, value in extra.items():
            if key in FIELD_RUNTIME_CALLBACK_SIGNATURE_FIELDS:
                continue
            body[str(key)] = value

    if not body.get("runtime_callback_id"):
        body["runtime_callback_id"] = derive_field_runtime_callback_id(body)
    if secret:
        body["runtime_signature_alg"] = FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG
        body["runtime_signature"] = sign_field_runtime_callback_payload(body, secret=secret)
    return body


def field_event_id_from_runtime_result(result: dict[str, Any]) -> str:
    """Extract the FieldIncident id from a RuntimeHandoffService submission result."""

    for candidate in _runtime_result_handoff_candidates(result):
        source_plan = candidate.get("source_plan") if isinstance(candidate, dict) else {}
        if not isinstance(source_plan, dict):
            continue
        event_id = _field_event_id_from_plan(source_plan)
        if event_id:
            return event_id
    return ""


def build_field_runtime_callback_sequence(
    result: dict[str, Any],
    *,
    secret: str | None = None,
    event_id: str | None = None,
    reason: str = "",
) -> list[dict[str, Any]]:
    """Build signed callback payloads from a runtime submission result."""

    clean_event_id = str(event_id or field_event_id_from_runtime_result(result) or "").strip()
    if not clean_event_id:
        raise ValueError("field event id is required")
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    handoff = result.get("handoff") if isinstance(result.get("handoff"), dict) else {}
    statuses = _runtime_result_status_sequence(result)
    payloads: list[dict[str, Any]] = []
    for index, status in enumerate(statuses, start=1):
        payloads.append(
            build_field_runtime_callback_payload(
                status=status,
                secret=secret,
                runtime_callback_id=f"{clean_event_id}:{run.get('run_id') or 'run'}:{index}:{status}",
                run_id=str(run.get("run_id") or ""),
                handoff_id=str(handoff.get("handoff_id") or run.get("handoff_id") or ""),
                dispatch_mode=str(result.get("dispatch_mode") or "task_handoff"),
                robot_motion_policy=_robot_motion_policy_from_runtime_result(result),
                hardware_dispatch=bool(result.get("hardware_dispatch", False)),
                reason=reason or str(result.get("reason") or ""),
                extra={
                    "profile": str(run.get("profile") or result.get("profile") or ""),
                    "event_id": clean_event_id,
                    "sequence": index,
                },
            )
        )
    return payloads


def post_field_runtime_callback_sequence(
    *,
    base_url: str,
    event_id: str,
    payloads: list[dict[str, Any]],
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    """POST a prepared status sequence and return one response per callback."""

    return [
        post_field_runtime_callback(
            base_url=base_url,
            event_id=event_id,
            payload=payload,
            timeout_s=timeout_s,
        )
        for payload in payloads
    ]


def post_field_runtime_callback(
    *,
    base_url: str,
    event_id: str,
    payload: dict[str, Any],
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """POST a prepared runtime callback payload and return the JSON response."""

    root = str(base_url or "").rstrip("/")
    if not root:
        raise ValueError("base_url is required")
    clean_event_id = str(event_id or "").strip()
    if not clean_event_id:
        raise ValueError("event_id is required")

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{root}/api/field/events/{clean_event_id}/runtime-delivery",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310 - operator-configured local/runtime URL.
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _runtime_result_handoff_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in (result.get("handoff"), result.get("run", {}).get("handoff")):
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def _field_event_id_from_plan(plan: dict[str, Any]) -> str:
    reference = plan.get("reference") if isinstance(plan.get("reference"), dict) else {}
    resolved = reference.get("resolved") if isinstance(reference.get("resolved"), dict) else {}
    for value in (resolved.get("field_event_id"), resolved.get("event_id")):
        text = str(value or "").strip()
        if text:
            return text
    mission_wrapper = plan.get("mission") if isinstance(plan.get("mission"), dict) else {}
    mission = mission_wrapper.get("mission") if isinstance(mission_wrapper.get("mission"), dict) else {}
    field_event = mission.get("field_event") if isinstance(mission.get("field_event"), dict) else {}
    return str(field_event.get("event_id") or "").strip()


def _runtime_result_status_sequence(result: dict[str, Any]) -> list[str]:
    if result.get("accepted") is False:
        return ["rejected"]
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    statuses: list[str] = []
    for event in run.get("runtime_events") or []:
        if not isinstance(event, dict):
            continue
        status = str(event.get("state") or "").strip()
        if status in FIELD_RUNTIME_DELIVERY_STATUSES and status not in statuses:
            statuses.append(status)
    current_state = str(run.get("current_state") or result.get("status") or "").strip()
    if current_state in FIELD_RUNTIME_DELIVERY_STATUSES and current_state not in statuses:
        statuses.append(current_state)
    return statuses or ["submitted"]


def _robot_motion_policy_from_runtime_result(result: dict[str, Any]) -> str:
    for candidate in _runtime_result_handoff_candidates(result):
        source_plan = candidate.get("source_plan") if isinstance(candidate, dict) else {}
        if not isinstance(source_plan, dict):
            continue
        mission_wrapper = (
            source_plan.get("mission") if isinstance(source_plan.get("mission"), dict) else {}
        )
        mission = (
            mission_wrapper.get("mission")
            if isinstance(mission_wrapper.get("mission"), dict)
            else {}
        )
        field_event = mission.get("field_event") if isinstance(mission.get("field_event"), dict) else {}
        policy = str(field_event.get("robot_motion_policy") or "").strip()
        if policy:
            return policy
    return ""
