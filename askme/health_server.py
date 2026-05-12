"""Embedded HTTP health endpoints for the askme runtime."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import time
from collections.abc import Callable
from datetime import datetime, timezone
from inspect import Parameter, isawaitable, signature
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from askme.api.routes.governance import register_governance_routes
from askme.api.routes.cognition import register_cognition_routes
from askme.api.routes.memory import register_memory_routes
from askme.api.routes.runtime import register_runtime_routes
from askme.api.routes.space import register_space_routes
from askme.api.routes.system import register_system_routes
from askme.api.routes.voice import register_voice_routes
from askme.config import get_config
from askme.governance import OperatorDirectory
from askme.pipeline.rag_policy import forced_rag_reply
from askme.robot.runtime_health import RuntimeHealthSnapshot, merge_voice_pipeline_status
from askme.runtime.field_callbacks import (
    derive_field_runtime_callback_id,
    unsigned_field_runtime_callback_payload,
)
from askme.runtime.field_callbacks import (
    sign_field_runtime_callback_payload as _sign_runtime_callback_payload,
)

logger = logging.getLogger(__name__)

_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_DEGRADED_OTA_STATES = {"auth_error", "degraded"}
_UTC = timezone.utc  # noqa: UP017 - Sunrise runs Python 3.10, where datetime.UTC is unavailable.
_PUBLIC_HTTP_PATHS = frozenset(("/health", "/healthz", "/metrics", "/metrics/prometheus"))
_PROTECTED_HTTP_PREFIXES = ("/api/",)
_PROTECTED_HTTP_PATHS = frozenset(("/dashboard", "/trace"))
_REMOTE_BIND_HOSTS = frozenset(("", "0.0.0.0", "::", "[::]"))
_FIELD_EVIDENCE_ROOT_NAMES = ("artifacts", "output", "data")
_FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG = "hmac-sha256"
_FIELD_RUNTIME_CALLBACK_SIGNATURE_FIELDS = {
    "runtime_signature",
    "signature",
    "x_signature",
    "runtime_signature_alg",
    "signature_alg",
}
_FIELD_RUNTIME_CALLBACK_TIMESTAMP_FIELDS = (
    "runtime_signature_timestamp",
    "signature_timestamp",
)
_FIELD_RUNTIME_CALLBACK_ID_FIELDS = (
    "runtime_callback_id",
    "callback_id",
    "delivery_id",
    "message_id",
)

HealthProvider = Callable[[], dict[str, Any]]
MetricsProvider = Callable[[], dict[str, Any]]
CapabilitiesProvider = Callable[[], dict[str, Any]]
MissionHandler = Any
CognitionHandler = Any
RuntimeHandler = Any
MemoryHandler = Any
FieldOperationsHandler = Any
VoiceHandler = Any
SpaceHandler = Any


class _MutableHandlerProxy:
    """Small mutable handler wrapper used by routes created before runtime wiring."""

    def __init__(self, handler: Any | None = None, *, label: str = "handler") -> None:
        self._handler = handler
        self._label = label

    def set(self, handler: Any) -> None:
        self._handler = handler

    def __getattr__(self, method_name: str) -> Any:
        if self._handler is None:
            raise RuntimeError(f"{self._label} not configured")
        return getattr(self._handler, method_name)


def _resolve_field_evidence_path(raw_path: str) -> Path | None:
    raw = str(raw_path or "").strip().replace("\\", "/")
    if not raw or "\x00" in raw or raw.startswith(("http://", "https://", "data:")):
        return None
    candidate = Path(raw)
    cwd = Path.cwd().resolve()
    allowed_roots = [(cwd / name).resolve() for name in _FIELD_EVIDENCE_ROOT_NAMES]
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if not candidate.parts or candidate.parts[0] not in _FIELD_EVIDENCE_ROOT_NAMES:
            return None
        resolved = (cwd / candidate).resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _field_runtime_plan_from_event(
    event: dict[str, Any],
    *,
    operator_id: str,
) -> dict[str, Any]:
    """Build a runtime-handoff plan from an accepted field incident."""

    event_id = str(event.get("event_id") or "")
    scenario_id = str(event.get("scenario_id") or "field_event")
    playbook = event.get("playbook") if isinstance(event.get("playbook"), dict) else {}
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    policy = str(playbook.get("robot_motion_policy") or "observe_then_continue")
    location = str(
        event.get("location")
        or payload.get("location")
        or payload.get("target_location")
        or "-"
    )
    area_id = _field_runtime_area_id(event, payload)
    task_type = _field_runtime_task_type(scenario_id, policy)
    risk_tier = _field_runtime_risk_tier(event)
    goal = (
        f"Handle field event {scenario_id} at {location}. "
        f"Apply robot policy {policy} and keep operator in control."
    )
    return {
        "plan_id": f"field-{event_id or scenario_id}",
        "planning_session_id": f"field-session-{event_id or scenario_id}",
        "intent": task_type,
        "goal": goal,
        "handoff_ready": True,
        "operator_id": operator_id,
        "operator_roles": ["operator"],
        "safety_constraints": [
            "Do not bypass field safety policy.",
            "Do not execute low-level motor commands from LLM output.",
            "Keep hardware dispatch disabled unless the runtime profile explicitly enables it.",
        ],
        "missing_inputs": [],
        "reference": {
            "resolved": {
                "area_id": area_id,
                "label": location,
                "field_event_id": event_id,
                "scenario_id": scenario_id,
            }
        },
        "mission": {
            "mission": {
                "mission_type": task_type,
                "goal": goal,
                "risk_tier": risk_tier,
                "operator_id": operator_id,
                "operator_roles": ["operator"],
                "steps": [{"target": area_id, "policy": policy}],
                "safety_notes": [
                    f"field_event_id={event_id}",
                    f"robot_motion_policy={policy}",
                    f"priority={event.get('priority') or ''}",
                    "field event runtime handoff is high-level only",
                ],
                "field_event": {
                    "event_id": event_id,
                    "scenario_id": scenario_id,
                    "priority": event.get("priority"),
                    "severity": event.get("severity"),
                    "location": location,
                    "notification_group": event.get("notification_group"),
                    "robot_motion_policy": policy,
                },
            }
        },
    }


def _field_runtime_area_id(event: dict[str, Any], payload: dict[str, Any]) -> str:
    for value in (
        payload.get("zone_id"),
        payload.get("map_zone_id"),
        payload.get("help_point_id"),
        event.get("location"),
        payload.get("location"),
        payload.get("target_location"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith(("area-", "zone-", "checkpoint-", "route-")):
            return lowered
        slug = re.sub(r"[^a-z0-9_-]+", "-", lowered).strip("-")
        if slug:
            return f"zone-{slug[:48]}"
    return "zone-field-event"


def _field_runtime_task_type(scenario_id: str, policy: str) -> str:
    normalized = f"{scenario_id} {policy}".strip()
    if normalized:
        return "field_incident_response"
    return "status_report"


def _field_runtime_risk_tier(event: dict[str, Any]) -> str:
    priority = str(event.get("priority") or "").upper()
    severity = str(event.get("severity") or "").lower()
    if priority == "P0" or severity == "error":
        return "high"
    if priority in {"P1", "P2"}:
        return "medium"
    return "low"


def _field_runtime_delivery_status(
    runtime_result: dict[str, Any],
    run: dict[str, Any],
) -> str:
    if runtime_result.get("accepted") is False:
        return "rejected"
    state = str(run.get("current_state") or runtime_result.get("state") or "").strip()
    if state:
        return state
    return "submitted"


def sign_field_runtime_callback_payload(body: dict[str, Any], *, secret: str) -> str:
    """Return the HMAC signature expected on field runtime-delivery callbacks."""

    return _sign_runtime_callback_payload(body, secret=secret)


def _unsigned_field_runtime_callback_payload(body: dict[str, Any]) -> dict[str, Any]:
    return unsigned_field_runtime_callback_payload(body)


def _field_runtime_callback_signature_value(body: dict[str, Any]) -> str:
    for key in ("runtime_signature", "signature", "x_signature"):
        value = body.get(key)
        if value:
            return str(value).strip()
    return ""


def _field_runtime_callback_timestamp(body: dict[str, Any]) -> float | None:
    for key in _FIELD_RUNTIME_CALLBACK_TIMESTAMP_FIELDS:
        parsed = _parse_field_runtime_timestamp(body.get(key))
        if parsed is not None:
            return parsed
    return None


def _field_runtime_callback_id(body: dict[str, Any]) -> str:
    for key in _FIELD_RUNTIME_CALLBACK_ID_FIELDS:
        value = body.get(key)
        if value:
            return str(value).strip()
    return derive_field_runtime_callback_id(body)


def _parse_field_runtime_timestamp(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    return parsed.timestamp()


def _field_runtime_callback_trust(
    body: dict[str, Any],
    *,
    secret: str,
    max_age_s: float,
    now: float | None = None,
) -> dict[str, Any]:
    base = {
        "signature_alg": _FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG,
        "secret_configured": bool(secret),
        "signature_verified": False,
        "timestamp_verified": False,
    }
    if not secret:
        return {
            **base,
            "trusted": True,
            "status": "unsigned",
            "reason": "runtime_callback_secret_not_configured",
        }
    signature_alg = str(
        body.get("runtime_signature_alg")
        or body.get("signature_alg")
        or _FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG
    )
    if signature_alg != _FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG:
        return {
            **base,
            "trusted": False,
            "status": "blocked",
            "reason": "unsupported_runtime_signature_alg",
        }
    actual_signature = _field_runtime_callback_signature_value(body)
    if not actual_signature:
        return {
            **base,
            "trusted": False,
            "status": "blocked",
            "reason": "missing_runtime_signature",
        }
    expected_signature = sign_field_runtime_callback_payload(body, secret=secret)
    if not hmac.compare_digest(actual_signature, expected_signature):
        return {
            **base,
            "trusted": False,
            "status": "blocked",
            "reason": "runtime_signature_mismatch",
        }
    timestamp = _field_runtime_callback_timestamp(body)
    if timestamp is None:
        return {
            **base,
            "signature_verified": True,
            "trusted": False,
            "status": "blocked",
            "reason": "missing_runtime_signature_timestamp",
        }
    current = time.time() if now is None else now
    age_s = current - timestamp
    if age_s < -5.0:
        return {
            **base,
            "signature_verified": True,
            "trusted": False,
            "status": "blocked",
            "reason": "runtime_signature_from_future",
            "signature_age_s": round(age_s, 3),
        }
    if age_s > max_age_s:
        return {
            **base,
            "signature_verified": True,
            "trusted": False,
            "status": "blocked",
            "reason": "runtime_signature_expired",
            "signature_age_s": round(age_s, 3),
        }
    return {
        **base,
        "trusted": True,
        "status": "trusted",
        "reason": "signature_verified",
        "signature_verified": True,
        "timestamp_verified": True,
        "signature_age_s": round(age_s, 3),
    }


def _field_runtime_callback_delivery_body(
    body: dict[str, Any],
    *,
    trust: dict[str, Any],
) -> dict[str, Any]:
    delivery = _unsigned_field_runtime_callback_payload(body)
    delivery.setdefault("runtime_callback_id", _field_runtime_callback_id(body))
    delivery["runtime_callback_trust"] = trust
    return delivery


class HealthSnapshotProvider(RuntimeHealthSnapshot):
    """Compatibility adapter for tests and lightweight standalone usage."""

    def __init__(
        self,
        *,
        metrics: Any,
        skill_manager: Any,
        voice_status_provider: Callable[[], dict[str, Any]],
        default_model: str,
        app_name: str,
        app_version: str,
        voice_mode: bool,
        robot_mode: bool,
        ota_status_provider: Callable[[], dict[str, Any]] | None = None,
        voice_model: str | None = None,
    ) -> None:
        if callable(metrics):
            metrics_provider = metrics
        elif hasattr(metrics, "snapshot"):
            metrics_provider = metrics.snapshot
        else:
            raise TypeError("metrics must be callable or expose snapshot()")

        super().__init__(
            app_name=app_name,
            app_version=app_version,
            brain_config={
                "model": default_model,
                "voice_model": voice_model,
            },
            voice_mode=voice_mode,
            robot_mode=robot_mode,
            metrics_provider=metrics_provider,
            active_skill_names_provider=lambda: [
                skill.name for skill in skill_manager.get_enabled()
            ],
            voice_status_provider=voice_status_provider,
            ota_status_provider=ota_status_provider or _disabled_ota_status,
        )

    def __call__(self) -> dict[str, Any]:
        return self.health_snapshot()


def build_health_snapshot(
    *,
    app_name: str,
    app_version: str,
    model_name: str,
    metrics_snapshot: dict[str, Any],
    active_skills: list[str],
    voice_status: dict[str, Any],
    ota_status: dict[str, Any] | None = None,
    voice_bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured runtime payload returned by `/health`."""
    llm_snapshot = metrics_snapshot.get("llm", {})
    resolved_model_name = llm_snapshot.get("last_model") or model_name or "unknown"
    enabled_skills = sorted(
        skill_name.strip()
        for skill_name in active_skills
        if isinstance(skill_name, str) and skill_name.strip()
    )
    merged_voice_status = merge_voice_pipeline_status(
        voice_status,
        metrics_snapshot.get("voice_pipeline", {}),
    )

    # Inject recorded_at into voice_pipeline_status so consumers can detect stale data.
    # Prefer a timestamp already in the metrics snapshot; otherwise stamp now.
    voice_pipeline_metrics = metrics_snapshot.get("voice_pipeline", {})
    recorded_at_raw = voice_pipeline_metrics.get("recorded_at") or voice_status.get("recorded_at")
    if recorded_at_raw:
        merged_voice_status["recorded_at"] = str(recorded_at_raw)
    else:
        _now_rec = datetime.now(_UTC)
        merged_voice_status["recorded_at"] = (
            _now_rec.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{_now_rec.microsecond // 1000:03d}Z"
        )

    degraded_reasons: list[str] = []
    if not merged_voice_status.get("pipeline_ok", True):
        degraded_reasons.append("voice_pipeline")
    if ota_status and ota_status.get("enabled") and ota_status.get("state") in _DEGRADED_OTA_STATES:
        degraded_reasons.append("ota_bridge")

    # ISO 8601 UTC timestamp for this snapshot — lets OTA Agent detect stale payloads.
    now_utc = datetime.now(_UTC)
    snapshot_at = (
        now_utc.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{now_utc.microsecond // 1000:03d}Z"
    )

    snapshot: dict[str, Any] = {
        "status": "degraded" if degraded_reasons else "ok",
        "service": app_name or "askme",
        "version": app_version or "unknown",
        "snapshot_at": snapshot_at,
        "schema_version": "2",
        "uptime_seconds": metrics_snapshot.get("uptime_seconds", 0.0),
        "model_name": resolved_model_name,
        "last_llm_latency_ms": llm_snapshot.get("last_latency_ms"),
        "total_conversations": metrics_snapshot.get("conversation_count", 0),
        "active_skills": enabled_skills,
        "active_skill_count": len(enabled_skills),
        "voice_pipeline_status": merged_voice_status,
        "degraded_reasons": degraded_reasons,
    }
    if ota_status is not None:
        snapshot["ota_bridge_status"] = ota_status
    if voice_bridge is not None:
        snapshot["voice_bridge"] = voice_bridge

    # Runtime service connectivity (nav-gateway, dog-control, dog-safety)
    try:
        from askme.robot.runtime_health import get_service_summary
        snapshot["services"] = get_service_summary()
    except Exception:
        pass

    return snapshot


ChatHandler = Callable[..., Any]  # async def handler(text: str, *, speak: bool = False)


VisionSnapshotHandler = Callable[[], Any]   # async () -> dict | None
VisionAnalyzeHandler = Callable[[str], Any]  # async (image_b64: str) -> str

# async (image_bytes, label, description, width, height) -> dict
ArchiveSnapshotHandler = Callable[[bytes, str, str, int, int], Any]
ArchiveListHandler = Callable[[], Any]           # async () -> list[dict]
ArchiveGetHandler = Callable[[str], Any]         # async (capture_id) -> dict | None
ArchiveDeleteHandler = Callable[[str], Any]      # async (capture_id) -> bool


def create_health_app(
    provider: HealthProvider | None = None,
    *,
    health_provider: HealthProvider | None = None,
    metrics_provider: MetricsProvider | None = None,
    capabilities_provider: CapabilitiesProvider | None = None,
    chat_handler: ChatHandler | None = None,
    conversation_provider: Callable[[], list[dict[str, Any]]] | None = None,
    vision_snapshot_handler: VisionSnapshotHandler | None = None,
    vision_analyze_handler: VisionAnalyzeHandler | None = None,
    archive_snapshot_handler: ArchiveSnapshotHandler | None = None,
    archive_list_handler: ArchiveListHandler | None = None,
    archive_get_handler: ArchiveGetHandler | None = None,
    archive_delete_handler: ArchiveDeleteHandler | None = None,
    mission_handler: MissionHandler | None = None,
    cognition_handler: CognitionHandler | None = None,
    runtime_handler: RuntimeHandler | None = None,
    memory_handler: MemoryHandler | None = None,
    field_operations_handler: FieldOperationsHandler | None = None,
    voice_handler: VoiceHandler | None = None,
    space_handler: SpaceHandler | None = None,
    control_api_key: str | None = None,
    field_runtime_callback_secret: str | None = None,
    field_runtime_callback_max_age_s: float = 300.0,
) -> FastAPI:
    """Create the HTTP app used for readiness and telemetry probes."""
    resolved_health_provider = health_provider or provider
    if resolved_health_provider is None:
        raise ValueError("health_provider is required")
    resolved_metrics_provider = metrics_provider or resolved_health_provider
    resolved_control_api_key = _clean_secret(control_api_key)
    resolved_runtime_callback_secret = _clean_secret(
        field_runtime_callback_secret
        or os.getenv("ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET")
    )
    resolved_runtime_callback_max_age_s = max(1.0, float(field_runtime_callback_max_age_s))
    if field_operations_handler is None:
        from askme.pipeline.field_operations import FieldOperationsService

        field_operations_handler = FieldOperationsService.from_env()
    if space_handler is None:
        from askme.space import ParkSpaceService

        space_handler = ParkSpaceService.from_config(get_config())

    app = FastAPI(
        title="askme-health",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    _CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}
    _MISSION_JSON_HEADERS = {"Cache-Control": "no-store", **_CORS_HEADERS}
    _CORS_ALLOW_HEADERS = (
        "Content-Type, Authorization, X-Askme-Api-Key, "
        "X-Askme-Operator-Id, X-Operator-Id"
    )
    _operator_directory = OperatorDirectory(get_config())

    def _json_error(message: str, *, status_code: int) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status_code, headers=_CORS_HEADERS)

    def _mission_json(payload: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
        return JSONResponse(payload, status_code=status_code, headers=_MISSION_JSON_HEADERS)

    def _cors_options_response(methods: str) -> Response:
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": methods,
                "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
            },
        )

    def _request_has_control_auth(request: Request) -> bool:
        if not resolved_control_api_key:
            return False
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            supplied = bearer[7:].strip()
            if secrets.compare_digest(supplied, resolved_control_api_key):
                return True
        supplied_key = request.headers.get("x-askme-api-key", "").strip()
        return bool(
            supplied_key
            and secrets.compare_digest(supplied_key, resolved_control_api_key)
        )

    def _request_requires_control_auth(request: Request) -> bool:
        if not resolved_control_api_key:
            return False
        path = request.url.path
        if request.method.upper() == "OPTIONS" or path in _PUBLIC_HTTP_PATHS:
            return False
        return path in _PROTECTED_HTTP_PATHS or path.startswith(_PROTECTED_HTTP_PREFIXES)

    @app.middleware("http")
    async def _control_api_auth(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if _request_requires_control_auth(request) and not _request_has_control_auth(request):
            return JSONResponse(
                {"error": "control API authentication required"},
                status_code=401,
                headers={
                    "Cache-Control": "no-store",
                    "WWW-Authenticate": 'Bearer realm="askme-control"',
                    **_CORS_HEADERS,
                },
            )
        return await call_next(request)

    async def _dispatch_mission(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if mission_handler is None:
            raise RuntimeError("mission handler not configured")
        method = getattr(mission_handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"mission handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("mission handler returned non-object payload")
        return payload

    async def _dispatch_cognition(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if cognition_handler is None:
            raise RuntimeError("cognition handler not configured")
        method = getattr(cognition_handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"cognition handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("cognition handler returned non-object payload")
        return payload

    async def _dispatch_runtime(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if runtime_handler is None:
            raise RuntimeError("runtime handler not configured")
        method = getattr(runtime_handler, method_name, None)
        if not callable(method) and method_name.startswith("runtime_"):
            method = getattr(runtime_handler, method_name.removeprefix("runtime_"), None)
        if not callable(method):
            raise RuntimeError(f"runtime handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("runtime handler returned non-object payload")
        return payload

    async def _optional_json_body(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if not raw:
            return {}
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("JSON object body required")
        return body

    def _operator_action_kwargs(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "operator_id": str(body.get("operator_id") or "askme.operator"),
            "reason": str(body.get("reason") or ""),
            "risk_acknowledgement": bool(
                body.get("risk_acknowledgement")
                or body.get("risk_ack")
                or body.get("acknowledged")
            ),
        }

    def _field_manual_trigger_body(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        payload.setdefault("operator_id", _operator_id_from_request(request, body))
        payload.setdefault("trigger_source", "operator_manual")
        payload.setdefault("admission_path", "field_events_manual")
        return payload

    def _operator_id_from_request(request: Request, body: dict[str, Any]) -> str:
        return str(
            body.get("operator_id")
            or request.headers.get("x-askme-operator-id")
            or request.headers.get("x-operator-id")
            or "dashboard.operator"
        ).strip()

    def _require_permission(
        request: Request,
        body: dict[str, Any],
        permission: str,
    ) -> JSONResponse | None:
        operator_id = _operator_id_from_request(request, body)
        decision = _operator_directory.authorize(operator_id, permission)
        if decision.get("allowed"):
            body.setdefault("operator_id", operator_id)
            body.setdefault("operator_auth", decision)
            return None
        return _mission_json(
            {
                "error": "operator not authorized",
                "reason": decision.get("reason") or "operator_missing_permission",
                "operator_auth": decision,
            },
            status_code=403,
        )

    def _operator_directory_payload() -> dict[str, Any]:
        return _operator_directory.payload()
        try:
            cfg = get_config()
        except Exception as exc:
            logger.warning("Operator directory config unavailable: %s", exc)
            cfg = {}
        field_cfg = cfg.get("field_operations") if isinstance(cfg.get("field_operations"), dict) else {}
        directory_cfg = (
            field_cfg.get("operator_directory")
            if isinstance(field_cfg.get("operator_directory"), dict)
            else {}
        )
        operators_cfg = field_cfg.get("operators") if isinstance(field_cfg.get("operators"), dict) else {}
        operators = []
        for operator_id, operator in sorted(operators_cfg.items()):
            operator_payload = operator if isinstance(operator, dict) else {}
            roles = operator_payload.get("roles") if isinstance(operator_payload.get("roles"), list) else []
            operators.append({
                "operator_id": str(operator_id),
                "display_name": str(operator_payload.get("display_name") or operator_id),
                "roles": [str(role) for role in roles],
                "source": "config.yaml",
            })
        return {
            "mode": str(directory_cfg.get("mode") or "demo_config"),
            "identity_provider": str(directory_cfg.get("identity_provider") or "local_config"),
            "production_binding_required": bool(
                directory_cfg.get("production_binding_required", True)
            ),
            "session_operator_header": "x-askme-operator-id",
            "operators": operators,
            "limitations": [
                "当前是 demo operator directory，不等于企业账号体系。",
                "生产环境应接入企业 SSO/IAM，并把审批、关闭、运行控制写入统一审计。",
            ],
        }

    def _looks_like_device_ingest_without_scenario(body: dict[str, Any]) -> bool:
        if body.get("scenario_id"):
            return False
        source = str(body.get("source") or "").strip().lower()
        if source in {"camera", "sensor", "robot", "mqtt", "ros", "hikvision"}:
            return True
        return any(key in body for key in ("device_id", "camera_id", "sensor", "robot", "detections", "predictions"))

    async def _dispatch_memory(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if memory_handler is None:
            raise RuntimeError("memory handler not configured")
        method = getattr(memory_handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"memory handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("memory handler returned non-object payload")
        return payload

    async def _dispatch_field_operations(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if field_operations_handler is None:
            raise RuntimeError("field operations handler not configured")
        method = getattr(field_operations_handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"field operations handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("field operations handler returned non-object payload")
        return payload

    async def _dispatch_voice(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if voice_handler is None:
            raise RuntimeError("voice handler not configured")
        method = getattr(voice_handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"voice handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("voice handler returned non-object payload")
        return payload

    async def _dispatch_space(
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if space_handler is None:
            raise RuntimeError("space handler not configured")
        method = getattr(space_handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"space handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("space handler returned non-object payload")
        return payload

    async def _record_field_voice_delivery(
        result: dict[str, Any],
        delivery: dict[str, Any],
    ) -> dict[str, Any]:
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            return result
        try:
            recorded = await _dispatch_field_operations(
                "record_voice_delivery_payload",
                event_id,
                delivery,
            )
        except Exception as exc:
            result["voice_delivery_record"] = {
                "recorded": False,
                "reason": str(exc),
            }
            return result
        result["voice_delivery_record"] = {
            "recorded": bool(recorded.get("recorded")),
            "reason": recorded.get("reason", ""),
        }
        if isinstance(recorded.get("event"), dict):
            result["event"] = recorded["event"]
        return result

    async def _record_field_runtime_delivery(
        result: dict[str, Any],
        delivery: dict[str, Any],
    ) -> dict[str, Any]:
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            return result
        try:
            recorded = await _dispatch_field_operations(
                "record_runtime_delivery_payload",
                event_id,
                delivery,
            )
        except Exception as exc:
            result["runtime_delivery_record"] = {
                "recorded": False,
                "reason": str(exc),
            }
            return result
        result["runtime_delivery_record"] = {
            "recorded": bool(recorded.get("recorded")),
            "reason": recorded.get("reason", ""),
        }
        if isinstance(recorded.get("event"), dict):
            result["event"] = recorded["event"]
        return result

    async def _dispatch_field_voice_directive(result: dict[str, Any]) -> dict[str, Any]:
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        directive = event.get("voice_directive") if isinstance(event, dict) else {}
        if not isinstance(directive, dict) or not directive.get("text"):
            return result
        if not result.get("accepted") or event.get("status") != "triggered":
            result["voice_delivery"] = {
                "status": "skipped",
                "reason": "event_not_triggered",
            }
            return await _record_field_voice_delivery(result, result["voice_delivery"])
        if voice_handler is None:
            result["voice_delivery"] = {
                "status": "skipped",
                "reason": "voice_handler_not_configured",
            }
            return await _record_field_voice_delivery(result, result["voice_delivery"])

        delivery: dict[str, Any] = {
            "status": "queued",
            "profile": None,
            "text_chars": len(str(directive.get("text") or "")),
        }
        profile_id = str(
            directive.get("resolved_profile")
            or directive.get("requested_profile")
            or ""
        ).strip()
        if profile_id:
            try:
                delivery["profile"] = await _dispatch_voice(
                    "set_voice_profile_payload",
                    {"profile_id": profile_id},
                )
            except Exception as exc:
                delivery["status"] = "profile_failed"
                delivery["reason"] = str(exc)
                result["voice_delivery"] = delivery
                return await _record_field_voice_delivery(result, delivery)

        try:
            speak = getattr(voice_handler, "speak", None)
            if not callable(speak):
                raise RuntimeError("voice handler missing speak")
            await _maybe_await(speak(str(directive.get("text") or "")))
            start_playback = getattr(voice_handler, "start_playback", None)
            if callable(start_playback):
                await _maybe_await(start_playback())
        except Exception as exc:
            delivery["status"] = "playback_failed"
            delivery["reason"] = str(exc)
        result["voice_delivery"] = delivery
        return await _record_field_voice_delivery(result, delivery)

    async def _dispatch_field_runtime_policy(
        result: dict[str, Any],
        *,
        operator_id: str,
    ) -> dict[str, Any]:
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        playbook = event.get("playbook") if isinstance(event.get("playbook"), dict) else {}
        policy = str(playbook.get("robot_motion_policy") or "").strip()
        if not policy:
            return result
        delivery: dict[str, Any] = {
            "status": "policy_ready",
            "target": "runtime-arbiter",
            "dispatch_mode": "policy_only",
            "hardware_dispatch": False,
            "robot_motion_policy": policy,
            "safety_boundary": "no_hardware_dispatch_from_field_event",
        }
        if not result.get("accepted") or event.get("status") != "triggered":
            delivery["status"] = "skipped"
            delivery["reason"] = "event_not_triggered"
            result["runtime_delivery"] = delivery
            return await _record_field_runtime_delivery(result, delivery)
        if runtime_handler is None:
            delivery["reason"] = "runtime_handler_not_configured"
            result["runtime_delivery"] = delivery
            return await _record_field_runtime_delivery(result, delivery)

        plan = _field_runtime_plan_from_event(event, operator_id=operator_id)
        delivery["dispatch_mode"] = "task_handoff"
        delivery["plan_id"] = plan["plan_id"]
        delivery["task_type"] = plan["mission"]["mission"]["mission_type"]
        try:
            runtime_result = await _dispatch_runtime("submit_plan_payload", plan)
        except Exception as exc:
            delivery["status"] = "submission_failed"
            delivery["reason"] = str(exc)
            result["runtime_delivery"] = delivery
            return await _record_field_runtime_delivery(result, delivery)

        run = runtime_result.get("run") if isinstance(runtime_result.get("run"), dict) else {}
        handoff = run.get("handoff") if isinstance(run.get("handoff"), dict) else {}
        result["runtime_handoff_result"] = runtime_result
        delivery.update(
            {
                "status": _field_runtime_delivery_status(runtime_result, run),
                "accepted": bool(runtime_result.get("accepted", True)),
                "profile": runtime_result.get("profile") or run.get("profile") or "",
                "run_id": str(run.get("run_id") or runtime_result.get("run_id") or ""),
                "handoff_id": str(
                    handoff.get("handoff_id")
                    or runtime_result.get("handoff_id")
                    or ""
                ),
                "current_state": str(run.get("current_state") or runtime_result.get("state") or ""),
                "reason": str(runtime_result.get("reason") or ""),
            }
        )
        result["runtime_delivery"] = delivery
        return await _record_field_runtime_delivery(result, delivery)

    def _handler_accepts_speak(handler: ChatHandler) -> bool:
        try:
            params = signature(handler).parameters
        except (TypeError, ValueError):
            return True
        return (
            "speak" in params
            or any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
        )

    async def _dispatch_chat_handler(text: str, *, speak: bool) -> Any:
        if chat_handler is None:
            raise RuntimeError("chat not available")
        if _handler_accepts_speak(chat_handler):
            return await _maybe_await(chat_handler(text, speak=speak))
        return await _maybe_await(chat_handler(text))

    def _voice_turn_payload_from_body(
        body: dict[str, Any],
        *,
        text: str,
        channel: str = "voice",
    ) -> dict[str, Any] | None:
        is_voice = bool(
            body.get("voice")
            or body.get("transcript_id")
            or body.get("asr_confidence") is not None
        )
        if not is_voice:
            return None
        payload: dict[str, Any] = {
            "transcript_id": str(body.get("transcript_id") or f"voice-turn-{secrets.token_hex(6)}"),
            "recognized_text": text,
            "is_final": bool(body.get("is_final", True)),
            "channel": str(body.get("channel") or channel or "voice"),
            "safety_bypass_allowed": False,
            "created_at": time.time(),
        }
        confidence = body.get("asr_confidence", body.get("confidence"))
        if confidence is not None:
            try:
                payload["confidence"] = min(max(float(confidence), 0.0), 1.0)
            except (TypeError, ValueError):
                payload["confidence"] = 0.0
        return payload

    def _chat_response_payload(
        result: Any,
        *,
        text: str,
        speak: bool,
        voice_turn: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("text", text)
            payload.setdefault("reply", "")
        else:
            payload = {"reply": result, "text": text}
        payload.setdefault("evidence", [])
        if speak:
            payload.setdefault("spoken", False)
        if voice_turn is not None:
            payload.setdefault("voice_turn", voice_turn)
        return payload

    async def _attach_memory_chat_context(payload: dict[str, Any]) -> dict[str, Any]:
        """Attach latest RAG evidence/policy when chat handler returned plain text."""
        if memory_handler is None:
            return payload
        health_method = getattr(memory_handler, "health", None)
        if not callable(health_method):
            return payload
        try:
            health = await _maybe_await(health_method())
        except Exception as exc:
            logger.debug("memory health unavailable for chat evidence: %s", exc)
            return payload
        if not isinstance(health, dict):
            return payload

        evidence = health.get("last_evidence")
        dropped = health.get("last_dropped_evidence")
        answer_policy = health.get("last_answer_policy")

        if not payload.get("evidence") and isinstance(evidence, list):
            payload["evidence"] = evidence

        rag_payload = payload.get("rag")
        if not isinstance(rag_payload, dict):
            rag_payload = {}
            payload["rag"] = rag_payload
        rag_payload.setdefault("enabled", health.get("enabled", False))
        rag_payload.setdefault("backend", health.get("backend", ""))
        rag_payload.setdefault("available", health.get("available", False))
        rag_payload.setdefault("last_backend", health.get("last_backend", ""))
        rag_payload.setdefault("last_retrieve_ms", health.get("last_retrieve_ms"))
        rag_payload.setdefault("last_retrieved_items", health.get("last_retrieved_items", 0))
        if isinstance(dropped, list):
            rag_payload.setdefault("dropped_evidence", dropped)
        if isinstance(answer_policy, dict):
            rag_payload.setdefault("answer_policy", answer_policy)
            forced_reply = forced_rag_reply(answer_policy)
            if forced_reply and not payload.get("evidence"):
                payload["reply"] = forced_reply
                payload["rag_blocked"] = True
                rag_payload["answer_blocked"] = True
                rag_payload["forced_reply"] = True
                rag_payload["block_reason"] = answer_policy.get("reason", "")
        return payload

    register_system_routes(
        app,
        health_provider=resolved_health_provider,
        metrics_provider=resolved_metrics_provider,
        render_prometheus_metrics=render_prometheus_metrics,
        json_snapshot_response=_json_snapshot_response,
        snapshot_payload=_snapshot_payload,
        prometheus_content_type=_PROMETHEUS_CONTENT_TYPE,
    )
    register_governance_routes(
        app,
        governance_payload=_operator_directory_payload,
        mission_json=_mission_json,
        cors_options_response=_cors_options_response,
    )
    register_memory_routes(
        app,
        dispatch_memory=lambda method_name, body: _dispatch_memory(method_name, body),
        mission_json=_mission_json,
        cors_options_response=_cors_options_response,
        logger=logger,
        authorize=_require_permission,
    )
    register_cognition_routes(
        app,
        dispatch_cognition=_dispatch_cognition,
        json_error=_json_error,
        cors_options_response=_cors_options_response,
        cors_headers=_CORS_HEADERS,
    )
    register_runtime_routes(
        app,
        dispatch_runtime=_dispatch_runtime,
        json_error=_json_error,
        cors_options_response=_cors_options_response,
        optional_json_body=_optional_json_body,
        operator_action_kwargs=_operator_action_kwargs,
        authorize=_require_permission,
        cors_headers=_CORS_HEADERS,
    )
    register_voice_routes(
        app,
        dispatch_voice=_dispatch_voice,
        mission_json=_mission_json,
        optional_json_body=_optional_json_body,
        cors_options_response=_cors_options_response,
        authorize=_require_permission,
    )
    register_space_routes(
        app,
        dispatch_space=lambda method_name, body: _dispatch_space(method_name, body),
        mission_json=_mission_json,
        optional_json_body=_optional_json_body,
        cors_options_response=_cors_options_response,
        logger=logger,
        authorize=_require_permission,
    )

    @app.post("/api/chat", tags=["Monitor"])
    async def chat(request: Request) -> JSONResponse:
        """Send text to the brain pipeline and return the response."""
        if chat_handler is None:
            return JSONResponse(
                {"error": "chat not available"},
                status_code=503,
                headers={"Access-Control-Allow-Origin": "*"},
            )
        try:
            body = await request.json()
            raw_text = body.get("text") or body.get("message") or body.get("prompt") or ""
            text = str(raw_text).strip()
            if not text:
                return JSONResponse(
                    {"error": "empty text"},
                    status_code=400,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            speak = bool(
                body.get("speak")
                or body.get("voice")
                or body.get("play_audio")
            )
            voice_turn = _voice_turn_payload_from_body(body, text=text)
            result = await _dispatch_chat_handler(text, speak=speak)
            payload = _chat_response_payload(
                result, text=text, speak=speak, voice_turn=voice_turn
            )
            payload = await _attach_memory_chat_context(payload)
            return JSONResponse(
                payload,
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        except Exception as exc:
            logger.error("Chat endpoint failed: %s", exc)
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    @app.post("/api/runtime/voice-turn", tags=["Runtime"])
    async def runtime_voice_turn(request: Request) -> JSONResponse:
        """Route a final voice transcript to runtime controls only."""
        if runtime_handler is None:
            return JSONResponse(
                {"error": "runtime handler not configured"},
                status_code=503,
                headers={"Access-Control-Allow-Origin": "*"},
            )
        try:
            body = await request.json()
            raw_text = body.get("text") or body.get("message") or body.get("transcript") or ""
            text = str(raw_text).strip()
            if not text:
                return JSONResponse(
                    {
                        "handled": False,
                        "reason": "empty_transcript",
                        "voice_turn": _voice_turn_payload_from_body(body, text=text),
                    },
                    status_code=400,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            payload = await _dispatch_runtime(
                "voice_turn_payload",
                text,
                speak=bool(body.get("speak") or body.get("play_audio")),
                transcript_id=str(body.get("transcript_id") or ""),
                confidence=body.get("asr_confidence", body.get("confidence")),
                is_final=bool(body.get("is_final", True)),
                channel=str(body.get("channel") or "voice"),
            )
            return JSONResponse(
                payload,
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        except Exception as exc:
            logger.error("Runtime voice-turn endpoint failed: %s", exc)
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    @app.get("/api/field/scenarios", tags=["Field Operations"])
    async def field_scenarios() -> JSONResponse:
        """Return customer-visible field operation scenarios."""
        try:
            result = await _dispatch_field_operations("scenarios_payload")
            return _mission_json(result)
        except Exception as exc:
            logger.error("Field scenarios endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/events", tags=["Field Operations"])
    async def field_events(
        limit: int = 50,
        status: str = "",
        notification_group: str = "",
        needs_attention: bool = False,
    ) -> JSONResponse:
        """Return recent field operation events."""
        try:
            result = await _dispatch_field_operations(
                "list_payload",
                limit=limit,
                status=status or None,
                notification_group=notification_group or None,
                needs_attention=needs_attention,
            )
            return _mission_json(result)
        except Exception as exc:
            logger.error("Field events endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/evidence", tags=["Field Operations"], response_model=None)
    async def field_evidence(path: str) -> Response:
        """Serve a local field evidence artifact from approved evidence roots."""
        resolved = _resolve_field_evidence_path(path)
        if resolved is None:
            return _mission_json({"error": "field evidence not found"}, status_code=404)
        media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        return FileResponse(
            resolved,
            media_type=media_type,
            filename=resolved.name,
            headers={
                "Cache-Control": "private, max-age=60",
                **_CORS_HEADERS,
            },
        )

    @app.post("/api/field/events", tags=["Field Operations"])
    async def field_event_trigger(request: Request) -> JSONResponse:
        """Evaluate a field event and dispatch alerts when rules pass."""
        try:
            body = await _optional_json_body(request)
            if _looks_like_device_ingest_without_scenario(body):
                return _mission_json(
                    {
                        "accepted": False,
                        "status": "rejected",
                        "reason": "device_payload_must_use_field_ingest",
                        "message": "Device camera, sensor, and robot payloads must be submitted to /api/field/ingest.",
                    },
                    status_code=422,
                )
            failure = _require_permission(request, body, "field:event:create")
            if failure is not None:
                return failure
            body = _field_manual_trigger_body(request, body)
            result = await _dispatch_field_operations("trigger_payload", body)
            result = await _dispatch_field_voice_directive(result)
            result = await _dispatch_field_runtime_policy(
                result,
                operator_id=str(body.get("operator_id") or "dashboard.operator"),
            )
            result.setdefault(
                "trigger_contract",
                {
                    "admission_path": "field_events_manual",
                    "trigger_source": body.get("trigger_source"),
                    "operator_id": body.get("operator_id"),
                    "device_payload_endpoint": "/api/field/ingest",
                },
            )
            status_code = 200 if result.get("accepted", True) else 422
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event trigger endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/close", tags=["Field Operations"])
    async def field_event_close(event_id: str, request: Request) -> JSONResponse:
        """Close a field operation event with an operator note."""
        try:
            body = await _optional_json_body(request)
            failure = _require_permission(request, body, "field:event:close")
            if failure is not None:
                return failure
            result = await _dispatch_field_operations("close_payload", event_id, body)
            status_code = 200 if result.get("closed") else 404
            if result.get("reason") in {
                "close_requires_supervisor_approval",
                "event_already_closed",
                "event_not_closable",
            }:
                status_code = 409
            if result.get("reason") in {"operator_not_authorized", "supervisor_not_authorized"}:
                status_code = 403
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event close endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/request-close", tags=["Field Operations"])
    async def field_event_request_close(event_id: str, request: Request) -> JSONResponse:
        """Request supervisor approval before closing a high-risk field event."""
        try:
            body = await _optional_json_body(request)
            failure = _require_permission(request, body, "field:event:request_close")
            if failure is not None:
                return failure
            result = await _dispatch_field_operations("request_close_payload", event_id, body)
            status_code = 200 if result.get("requested") else 404
            if result.get("reason") in {"event_already_closed", "event_not_closable"}:
                status_code = 409
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event close request endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/acknowledge", tags=["Field Operations"])
    async def field_event_acknowledge(event_id: str, request: Request) -> JSONResponse:
        """Acknowledge a field operation event without closing it."""
        try:
            body = await _optional_json_body(request)
            failure = _require_permission(request, body, "field:event:acknowledge")
            if failure is not None:
                return failure
            result = await _dispatch_field_operations("acknowledge_payload", event_id, body)
            status_code = 200 if result.get("acknowledged") else 409
            if result.get("reason") == "event_not_found":
                status_code = 404
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event acknowledge endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/resend-notification", tags=["Field Operations"])
    async def field_event_resend_notification(event_id: str, request: Request) -> JSONResponse:
        """Retry notification delivery for an open field operation event."""
        try:
            body = await _optional_json_body(request)
            failure = _require_permission(request, body, "field:event:acknowledge")
            if failure is not None:
                return failure
            result = await _dispatch_field_operations(
                "resend_notification_payload",
                event_id,
                body,
            )
            status_code = 200 if result.get("resent") else 409
            if result.get("reason") == "event_not_found":
                status_code = 404
            if result.get("reason") == "operator_not_authorized":
                status_code = 403
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event notification resend endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/events/{event_id}/report", tags=["Field Operations"])
    async def field_event_report(event_id: str) -> JSONResponse:
        """Return an auditable customer-facing field event report."""
        try:
            result = await _dispatch_field_operations("event_report_payload", event_id)
            status_code = 200 if result.get("found") else 404
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field event report endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/events/{event_id}/runtime-delivery", tags=["Field Operations"])
    async def field_event_runtime_delivery(event_id: str, request: Request) -> JSONResponse:
        """Record a runtime-arbiter or robot callback for a field event."""
        try:
            body = await _optional_json_body(request)
            trust = _field_runtime_callback_trust(
                body,
                secret=resolved_runtime_callback_secret,
                max_age_s=resolved_runtime_callback_max_age_s,
            )
            if not trust.get("trusted"):
                return _mission_json(
                    {
                        "recorded": False,
                        "reason": trust.get("reason") or "runtime_callback_not_trusted",
                        "runtime_callback_trust": trust,
                    },
                    status_code=403,
                )
            delivery = _field_runtime_callback_delivery_body(body, trust=trust)
            result = await _dispatch_field_operations(
                "record_runtime_delivery_payload",
                event_id,
                delivery,
            )
            status_code = 200 if result.get("recorded") else 422
            if result.get("reason") == "event_not_found":
                status_code = 404
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field runtime-delivery endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/notification-test", tags=["Field Operations"])
    async def field_notification_test(request: Request) -> JSONResponse:
        """Send a low-risk notification smoke test to a responder group."""
        try:
            body = await _optional_json_body(request)
            failure = _require_permission(request, body, "field:notification:test")
            if failure is not None:
                return failure
            result = await _dispatch_field_operations("test_notification_payload", body)
            status_code = 200 if result.get("status") != "invalid_group" else 422
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field notification test endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/notification-preflight", tags=["Field Operations"])
    async def field_notification_preflight() -> JSONResponse:
        """Check whether real DingTalk responder notification credentials are configured."""
        try:
            result = await _dispatch_field_operations("notification_preflight_payload")
            status_code = 200 if result.get("ready") else 409
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field notification preflight endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/devices", tags=["Field Operations"])
    async def field_devices() -> JSONResponse:
        """Return registered and observed field-device trust/online status."""
        try:
            result = await _dispatch_field_operations("device_status_payload")
            return _mission_json(result)
        except Exception as exc:
            logger.error("Field devices endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.options("/api/field/scenarios", include_in_schema=False)
    async def field_scenarios_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/field/events", include_in_schema=False)
    async def field_events_cors() -> Response:
        return _cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/evidence", include_in_schema=False)
    async def field_evidence_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/field/events/{event_id}/acknowledge", include_in_schema=False)
    async def field_event_acknowledge_cors(event_id: str) -> Response:
        _ = event_id
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/close", include_in_schema=False)
    async def field_event_close_cors(event_id: str) -> Response:
        _ = event_id
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/request-close", include_in_schema=False)
    async def field_event_request_close_cors(event_id: str) -> Response:
        _ = event_id
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/resend-notification", include_in_schema=False)
    async def field_event_resend_notification_cors(event_id: str) -> Response:
        _ = event_id
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/field/events/{event_id}/report", include_in_schema=False)
    async def field_event_report_cors(event_id: str) -> Response:
        _ = event_id
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/field/events/{event_id}/runtime-delivery", include_in_schema=False)
    async def field_event_runtime_delivery_cors(event_id: str) -> Response:
        _ = event_id
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/field/notification-test", include_in_schema=False)
    async def field_notification_test_cors() -> Response:
        return _cors_options_response("POST, OPTIONS")

    @app.options("/api/field/notification-preflight", include_in_schema=False)
    async def field_notification_preflight_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/field/devices", include_in_schema=False)
    async def field_devices_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.options("/api/field/ingest", include_in_schema=False)
    async def field_ingest_cors() -> Response:
        return _cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/field/audit/integrity", include_in_schema=False)
    async def field_action_audit_integrity_cors() -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.get("/api/field/ingest", tags=["Field Operations"])
    async def field_ingest_help() -> JSONResponse:
        """Return examples for raw camera/sensor/robot event ingestion."""
        try:
            result = await _dispatch_field_operations("ingest_help_payload")
            return _mission_json(result)
        except Exception as exc:
            logger.error("Field ingest help endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/readiness", tags=["Field Operations"])
    async def field_readiness() -> JSONResponse:
        """Return deployment readiness gates for field operations."""
        try:
            result = await _dispatch_field_operations("readiness_payload")
            return _mission_json(result)
        except Exception as exc:
            logger.error("Field readiness endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/field/audit/integrity", tags=["Field Operations"])
    async def field_action_audit_integrity() -> JSONResponse:
        """Verify the append-only field action audit hash chain."""
        try:
            result = await _dispatch_field_operations("action_audit_integrity_payload")
            status_code = 200
            if result.get("enabled") is not False and not result.get("valid"):
                status_code = 409
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field action audit integrity endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.post("/api/field/ingest", tags=["Field Operations"])
    async def field_ingest(request: Request) -> JSONResponse:
        """Normalize raw camera/sensor/robot/map payloads into field events."""
        try:
            body = await _optional_json_body(request)
            result = await _dispatch_field_operations("ingest_payload", body)
            result = await _dispatch_field_voice_directive(result)
            result = await _dispatch_field_runtime_policy(
                result,
                operator_id=str(body.get("operator_id") or "askme.operator"),
            )
            status_code = 200 if result.get("accepted", True) else 422
            return _mission_json(result, status_code=status_code)
        except Exception as exc:
            logger.error("Field ingest endpoint failed: %s", exc)
            return _mission_json({"error": str(exc)}, status_code=500)

    @app.get("/api/status", tags=["Monitor"])
    async def system_status() -> JSONResponse:
        """Unified system status — all key metrics in one endpoint."""
        import time as _time

        status: dict[str, Any] = {"timestamp": _time.time()}

        # Perception
        perception: dict[str, Any] = {}
        try:
            with open("/tmp/askme_frame_daemon.heartbeat") as f:
                hb = float(f.read().strip())
            perception["frame_daemon"] = {
                "alive": _time.time() - hb < 3.0,
                "age_s": round(_time.time() - hb, 1),
            }
        except (FileNotFoundError, ValueError):
            perception["frame_daemon"] = {"alive": False}

        try:
            with open("/tmp/askme_frame_detections.json") as f:
                det = json.load(f)
            perception["detections"] = {
                "count": len(det.get("detections", [])),
                "infer_ms": det.get("infer_ms", 0),
                "objects": [d["class_id"] for d in det.get("detections", [])],
            }
        except (FileNotFoundError, json.JSONDecodeError):
            perception["detections"] = {"count": 0}

        try:
            import os
            event_path = "/tmp/askme_events.jsonl"
            if os.path.exists(event_path):
                with open(event_path) as f:
                    lines = f.readlines()
                perception["change_events"] = {"total": len(lines)}
                if lines:
                    last = json.loads(lines[-1].strip())
                    perception["change_events"]["last"] = last
            else:
                perception["change_events"] = {"total": 0}
        except Exception:
            perception["change_events"] = {"total": 0}

        status["perception"] = perception

        # Services
        try:
            import subprocess
            orbbec = subprocess.run(
                ["systemctl", "is-active", "orbbec-camera"],
                capture_output=True, timeout=3,
            )
            status["orbbec_camera"] = orbbec.stdout.decode().strip() == "active"
        except Exception:
            status["orbbec_camera"] = False

        # Memory
        try:
            import os
            knowledge_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "qp_memory", "knowledge",
            )
            if os.path.isdir(knowledge_dir):
                files = [f for f in os.listdir(knowledge_dir) if f.endswith(".md")]
                status["memory"] = {"knowledge_files": len(files)}
            else:
                status["memory"] = {"knowledge_files": 0}
        except Exception:
            status["memory"] = {"knowledge_files": 0}

        return JSONResponse(
            status,
            headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
        )

    @app.get("/api/capabilities", tags=["System"])
    async def capabilities() -> JSONResponse:
        """Return the runtime profile, components, and generated contracts."""
        if capabilities_provider is None:
            return JSONResponse(
                {"error": "capabilities not available"},
                status_code=503,
                headers={"Access-Control-Allow-Origin": "*"},
            )
        try:
            payload = capabilities_provider()
            return JSONResponse(
                payload,
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        except Exception as exc:
            logger.error("Capabilities endpoint failed: %s", exc)
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    @app.post("/api/missions/draft", tags=["Mission"])
    async def mission_draft(request: Request) -> JSONResponse:
        """Draft a high-level mission without dispatching hardware."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return _json_error("JSON object body required", status_code=400)
            payload = await _dispatch_mission("draft_from_payload", body)
            return _mission_json(payload)
        except ValueError as exc:
            return _json_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return _json_error(str(exc), status_code=503)
        except Exception as exc:
            logger.error("Mission draft failed: %s", exc)
            return _json_error(str(exc), status_code=500)

    @app.post("/api/missions", tags=["Mission"])
    async def mission_submit(request: Request) -> JSONResponse:
        """Dry-run or submit a mission through the configured runtime arbiter."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return _json_error("JSON object body required", status_code=400)
            payload = await _dispatch_mission(
                "submit_from_payload",
                body,
                trusted_confirmation=_request_has_control_auth(request),
            )
            return _mission_json(payload)
        except ValueError as exc:
            return _json_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return _json_error(str(exc), status_code=503)
        except Exception as exc:
            logger.error("Mission submit failed: %s", exc)
            return _json_error(str(exc), status_code=500)

    @app.get("/api/missions", tags=["Mission"])
    async def mission_list() -> JSONResponse:
        """Return locally drafted/submitted mission records."""
        try:
            payload = await _dispatch_mission("list_payload")
            return _mission_json(payload)
        except RuntimeError as exc:
            return _json_error(str(exc), status_code=503)
        except Exception as exc:
            logger.error("Mission list failed: %s", exc)
            return _json_error(str(exc), status_code=500)

    @app.get("/api/missions/{mission_id}", tags=["Mission"])
    async def mission_get(mission_id: str) -> JSONResponse:
        """Return a single mission plan and its latest submission state."""
        try:
            payload = await _dispatch_mission("get_payload", mission_id)
            status_code = 404 if payload.get("error") else 200
            return _mission_json(payload, status_code=status_code)
        except RuntimeError as exc:
            return _json_error(str(exc), status_code=503)
        except Exception as exc:
            logger.error("Mission get failed: %s", exc)
            return _json_error(str(exc), status_code=500)

    @app.get("/api/missions/{mission_id}/report", tags=["Mission"])
    async def mission_report(mission_id: str) -> JSONResponse:
        """Build an inspection report shell from mission evidence."""
        try:
            payload = await _dispatch_mission("report_payload", mission_id)
            status_code = 404 if payload.get("error") else 200
            return _mission_json(payload, status_code=status_code)
        except RuntimeError as exc:
            return _json_error(str(exc), status_code=503)
        except Exception as exc:
            logger.error("Mission report failed: %s", exc)
            return _json_error(str(exc), status_code=500)

    @app.options("/api/missions", include_in_schema=False)
    @app.options("/api/missions/draft", include_in_schema=False)
    async def mission_collection_cors() -> Response:
        return _cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/missions/{mission_id}", include_in_schema=False)
    @app.options("/api/missions/{mission_id}/report", include_in_schema=False)
    async def mission_item_cors(mission_id: str) -> Response:
        return _cors_options_response("GET, OPTIONS")

    @app.get("/dashboard", tags=["Monitor"])
    async def dashboard() -> Response:
        """Serve the product dashboard shell."""
        return Response(content=_DASHBOARD_HTML, media_type="text/html")

    @app.get("/dashboard/{asset_path:path}", tags=["Monitor"])
    async def dashboard_asset(asset_path: str) -> Response:
        """Serve dashboard pages and assets without mixing them into one HTML file."""
        clean_path = asset_path.strip("/")
        if not clean_path or clean_path in _DASHBOARD_PAGES:
            return Response(content=_DASHBOARD_HTML, media_type="text/html")
        resolved = (_DASHBOARD_ASSET_DIR / clean_path).resolve()
        try:
            resolved.relative_to(_DASHBOARD_ASSET_DIR.resolve())
        except ValueError:
            return _json_error("dashboard asset path is outside static directory", status_code=404)
        if not resolved.is_file():
            return _json_error("dashboard asset not found", status_code=404)
        return FileResponse(
            resolved,
            media_type=mimetypes.guess_type(str(resolved))[0] or "application/octet-stream",
            headers={"Cache-Control": "private, max-age=30"},
        )

    @app.options("/api/chat", include_in_schema=False)
    async def chat_cors() -> Response:
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
            },
        )

    @app.get("/api/live", tags=["Monitor"])
    async def live() -> JSONResponse:
        """Return in-memory conversation history (voice + web chat combined)."""
        if conversation_provider is None:
            return JSONResponse(
                {"messages": [], "count": 0},
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        try:
            messages = conversation_provider()
            return JSONResponse(
                {"messages": messages, "count": len(messages)},
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        except Exception as exc:
            return JSONResponse(
                {"messages": [], "count": 0, "error": str(exc)},
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    @app.get("/api/conversations", tags=["Monitor"])
    async def conversations() -> JSONResponse:
        """Return conversation history for the monitor UI."""
        try:
            from askme.config import get_config, project_root
            cfg = get_config().get("conversation", {})
            raw_path = cfg.get("history_file", "data/conversation_history.json")
            history_file = Path(raw_path)
            if not history_file.is_absolute():
                history_file = project_root() / history_file
            if history_file.exists():
                with open(history_file, encoding="utf-8") as fh:
                    history = json.load(fh)
            else:
                history = []
            return JSONResponse(
                {"messages": history, "count": len(history)},
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        except Exception as exc:
            logger.error("Conversations endpoint failed: %s", exc)
            return JSONResponse(
                {"messages": [], "count": 0, "error": str(exc)},
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    # ---- Vision endpoints ----

    @app.get("/api/vision/snapshot", tags=["Vision"])
    async def vision_snapshot() -> JSONResponse:
        """Capture a frame from the robot camera and return it as base64 JPEG."""
        if vision_snapshot_handler is None:
            return JSONResponse({"error": "vision not configured"}, status_code=503,
                                headers=_CORS_HEADERS)
        try:
            result = await vision_snapshot_handler()
            if result is None:
                return JSONResponse({"error": "camera not available"}, status_code=503,
                                    headers=_CORS_HEADERS)
            # Auto-archive if handler available
            if archive_snapshot_handler is not None:
                try:
                    import base64 as _b64
                    image_bytes = _b64.b64decode(result.get("image_base64", ""))
                    if image_bytes:
                        meta = await archive_snapshot_handler(
                            image_bytes,
                            "manual",
                            "",
                            result.get("width", 0),
                            result.get("height", 0),
                        )
                        result = dict(result)
                        result["capture_id"] = meta.get("id")
                except Exception as _arc_exc:
                    logger.warning("[Vision] Auto-archive failed: %s", _arc_exc)
            return JSONResponse(result, headers=_CORS_HEADERS)
        except Exception as exc:
            logger.error("Vision snapshot failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/vision/snapshot", include_in_schema=False)
    async def vision_snapshot_cors() -> Response:
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
        })

    @app.post("/api/vision/analyze", tags=["Vision"])
    async def vision_analyze(request: Request) -> JSONResponse:
        """Analyze an image (base64 JPEG) with the VLM and return a description."""
        if vision_analyze_handler is None:
            return JSONResponse({"error": "vision not configured"}, status_code=503,
                                headers=_CORS_HEADERS)
        try:
            body = await request.json()
            image_b64: str = body.get("image_base64", "")
            if not image_b64:
                return JSONResponse({"error": "image_base64 required"}, status_code=400,
                                    headers=_CORS_HEADERS)
            description = await vision_analyze_handler(image_b64)
            return JSONResponse({"description": description}, headers=_CORS_HEADERS)
        except Exception as exc:
            logger.error("Vision analyze failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/vision/analyze", include_in_schema=False)
    async def vision_analyze_cors() -> Response:
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
        })

    # ---- Image archive endpoints ----

    @app.get("/api/vision/captures", tags=["Vision"])
    async def vision_captures_list(limit: int = 50, label: str | None = None) -> JSONResponse:
        """List archived captures (metadata only, no image_base64)."""
        if archive_list_handler is None:
            return JSONResponse({"error": "image archive not configured"}, status_code=503,
                                headers=_CORS_HEADERS)
        try:
            captures = await archive_list_handler()
            # Apply optional label filter and limit in handler or here
            if label is not None:
                captures = [c for c in captures if c.get("label") == label]
            captures = captures[:limit]
            return JSONResponse({"captures": captures, "count": len(captures)},
                                headers={"Cache-Control": "no-store", **_CORS_HEADERS})
        except Exception as exc:
            logger.error("Captures list failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/vision/captures", include_in_schema=False)
    async def vision_captures_list_cors() -> Response:
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
        })

    @app.get("/api/vision/captures/{capture_id}", tags=["Vision"])
    async def vision_captures_get(capture_id: str) -> JSONResponse:
        """Return full metadata + image_base64 for a capture."""
        if archive_get_handler is None:
            return JSONResponse({"error": "image archive not configured"}, status_code=503,
                                headers=_CORS_HEADERS)
        try:
            data = await archive_get_handler(capture_id)
            if data is None:
                return JSONResponse({"error": "capture not found"}, status_code=404,
                                    headers=_CORS_HEADERS)
            return JSONResponse(data, headers={"Cache-Control": "no-store", **_CORS_HEADERS})
        except Exception as exc:
            logger.error("Captures get failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/vision/captures/{capture_id}", include_in_schema=False)
    async def vision_captures_item_cors(capture_id: str) -> Response:
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
        })

    @app.delete("/api/vision/captures/{capture_id}", tags=["Vision"])
    async def vision_captures_delete(capture_id: str) -> JSONResponse:
        """Delete a capture (JPEG + JSON sidecar)."""
        if archive_delete_handler is None:
            return JSONResponse({"error": "image archive not configured"}, status_code=503,
                                headers=_CORS_HEADERS)
        try:
            deleted = await archive_delete_handler(capture_id)
            if not deleted:
                return JSONResponse({"error": "capture not found"}, status_code=404,
                                    headers=_CORS_HEADERS)
            return JSONResponse({"deleted": True, "capture_id": capture_id},
                                headers=_CORS_HEADERS)
        except Exception as exc:
            logger.error("Captures delete failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    return app


class AskmeHealthServer:
    """Run the embedded FastAPI health server inside the current event loop."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        health_provider: HealthProvider | None = None,
        metrics_provider: MetricsProvider | None = None,
        snapshot_provider: HealthProvider | None = None,
        provider: HealthProvider | None = None,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.host = str(cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        self._access_log = bool(cfg.get("access_log", False))
        self._log_level = str(cfg.get("log_level", "warning")).strip().lower() or "warning"
        self._control_api_key = _clean_secret(
            cfg.get("control_api_key")
            or cfg.get("api_key")
            or os.environ.get("ASKME_CONTROL_API_KEY")
            or os.environ.get("ASKME_HEALTH_API_KEY")
        )
        self._allow_unsafe_remote = bool(
            cfg.get("allow_unsafe_remote", False)
            or cfg.get("allow_unsafe_control_api", False)
        )
        if (
            self.enabled
            and _is_remote_bind_host(self.host)
            and not self._control_api_key
            and not self._allow_unsafe_remote
        ):
            raise ValueError(
                "health_server.host binds outside loopback; set "
                "health_server.control_api_key or ASKME_CONTROL_API_KEY, "
                "or explicitly set health_server.allow_unsafe_remote=true"
            )

        raw_port = cfg.get("port", 8765)
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = 8765
        self.port = min(max(port, 1024), 65535)
        self._startup_timeout_s = max(0.1, float(cfg.get("startup_timeout", 5.0)))
        self._shutdown_timeout_s = max(0.1, float(cfg.get("shutdown_timeout", 5.0)))

        self._chat_handler: ChatHandler | None = None
        self._vision_bridge: Any | None = None
        self._image_archive: Any | None = None
        self._capabilities_provider: CapabilitiesProvider | None = None
        self._mission_handler: MissionHandler | None = None
        self._cognition_handler: CognitionHandler | None = None
        self._runtime_handler: RuntimeHandler | None = None
        self._memory_handler: MemoryHandler | None = None
        self._voice_handler: VoiceHandler | None = None
        from askme.pipeline.field_operations import FieldOperationsService

        self._field_operations_proxy = _MutableHandlerProxy(
            FieldOperationsService.from_env(),
            label="field operations handler",
        )

        resolved_health_provider = health_provider or snapshot_provider or provider
        if resolved_health_provider is None:
            raise ValueError("health_provider is required")
        resolved_metrics_provider = metrics_provider or resolved_health_provider

        self._conversation_provider: Callable[[], list[dict[str, Any]]] | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._started_event: asyncio.Event | None = None  # lazy-init in async context
        self._bound_port: int | None = None

        self._app = create_health_app(
            health_provider=resolved_health_provider,
            metrics_provider=resolved_metrics_provider,
            capabilities_provider=self._get_capabilities,
            chat_handler=self._dispatch_chat,
            conversation_provider=self._get_conversation,
            vision_snapshot_handler=self._dispatch_snapshot,
            vision_analyze_handler=self._dispatch_analyze,
            archive_snapshot_handler=self._dispatch_archive,
            archive_list_handler=self._dispatch_archive_list,
            archive_get_handler=self._dispatch_archive_get,
            archive_delete_handler=self._dispatch_archive_delete,
            mission_handler=self,
            cognition_handler=self,
            runtime_handler=self,
            memory_handler=self,
            voice_handler=self,
            field_operations_handler=self._field_operations_proxy,
            control_api_key=self._control_api_key,
        )

    def _get_conversation(self) -> list[dict[str, Any]]:
        if self._conversation_provider is None:
            return []
        return self._conversation_provider()

    async def _dispatch_chat(self, text: str, *, speak: bool = False) -> Any:
        if self._chat_handler is None:
            return "[chat handler not configured]"
        try:
            params = signature(self._chat_handler).parameters
            accepts_speak = (
                "speak" in params
                or any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
            )
        except (TypeError, ValueError):
            accepts_speak = True
        if accepts_speak:
            return await _maybe_await(self._chat_handler(text, speak=speak))
        return await _maybe_await(self._chat_handler(text))

    def set_chat_handler(self, handler: ChatHandler) -> None:
        """Wire the chat handler after construction (avoids circular deps)."""
        self._chat_handler = handler

    def set_capabilities_provider(self, provider: CapabilitiesProvider) -> None:
        """Wire the capabilities provider after construction."""
        self._capabilities_provider = provider

    def set_conversation_provider(self, provider: Callable[[], list[dict[str, Any]]]) -> None:
        """Wire conversation history provider for /api/live endpoint."""
        self._conversation_provider = provider

    def set_mission_handler(self, handler: MissionHandler) -> None:
        """Wire the mission adapter after construction."""
        self._mission_handler = handler

    def set_cognition_handler(self, handler: CognitionHandler) -> None:
        """Wire the cognition adapter after construction."""
        self._cognition_handler = handler

    def set_runtime_handler(self, handler: RuntimeHandler) -> None:
        """Wire the runtime handoff adapter after construction."""
        self._runtime_handler = handler

    def set_memory_handler(self, handler: MemoryHandler) -> None:
        """Wire the memory/RAG adapter after construction."""
        self._memory_handler = handler

    def set_voice_handler(self, handler: VoiceHandler) -> None:
        """Wire the voice/TTS adapter after construction."""
        self._voice_handler = handler

    def set_field_operations_handler(self, handler: FieldOperationsHandler) -> None:
        """Wire the field event service after construction."""
        self._field_operations_proxy.set(handler)

    def _get_capabilities(self) -> dict[str, Any]:
        if self._capabilities_provider is None:
            return {}
        return self._capabilities_provider()

    async def draft_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_mission_handler("draft_from_payload", payload)

    async def submit_from_payload(
        self,
        payload: dict[str, Any],
        *,
        trusted_confirmation: bool = False,
    ) -> dict[str, Any]:
        return await self._dispatch_mission_handler(
            "submit_from_payload",
            payload,
            trusted_confirmation=trusted_confirmation,
        )

    async def list_payload(self) -> dict[str, Any]:
        return await self._dispatch_mission_handler("list_payload")

    async def get_payload(self, mission_id: str) -> dict[str, Any]:
        return await self._dispatch_mission_handler("get_payload", mission_id)

    async def report_payload(self, mission_id: str) -> dict[str, Any]:
        return await self._dispatch_mission_handler("report_payload", mission_id)

    async def context_payload(self, *, refresh_perception: bool = False) -> dict[str, Any]:
        return await self._dispatch_cognition_handler(
            "context_payload",
            refresh_perception=refresh_perception,
        )

    async def plan_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_cognition_handler("plan_from_payload", payload)

    async def search_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_memory_handler("search_payload", payload)

    async def preview_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_memory_handler("preview_payload", payload)

    async def import_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_memory_handler("import_payload", payload)

    async def list_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_memory_handler("list_knowledge_payload", payload)

    async def update_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_memory_handler("update_knowledge_payload", payload)

    async def voice_profiles_payload(self) -> dict[str, Any]:
        return await self._dispatch_voice_handler("voice_profiles_payload")

    async def set_voice_profile_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_voice_handler("set_voice_profile_payload", payload)

    async def runtime_context_payload(self) -> dict[str, Any]:
        return await self._dispatch_runtime_handler("context_payload")

    async def runtime_events_payload(
        self,
        *,
        after: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "events_payload",
            after=after,
            limit=limit,
        )

    async def runtime_profiles_payload(self) -> dict[str, Any]:
        return await self._dispatch_runtime_handler("profiles_payload")

    async def runtime_list_payload(self) -> dict[str, Any]:
        return await self._dispatch_runtime_handler("list_payload")

    async def runtime_get_payload(self, run_id: str) -> dict[str, Any]:
        return await self._dispatch_runtime_handler("get_payload", run_id)

    async def runtime_report_payload(self, run_id: str) -> dict[str, Any]:
        return await self._dispatch_runtime_handler("report_payload", run_id)

    async def runtime_pause_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "pause_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    async def runtime_resume_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "resume_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    async def runtime_cancel_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "cancel_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    async def runtime_advance_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "advance_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    async def _dispatch_mission_handler(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        handler = self._mission_handler
        if handler is None:
            raise RuntimeError("mission handler not configured")
        method = getattr(handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"mission handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("mission handler returned non-object payload")
        return payload

    async def _dispatch_cognition_handler(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        handler = self._cognition_handler
        if handler is None:
            raise RuntimeError("cognition handler not configured")
        method = getattr(handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"cognition handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("cognition handler returned non-object payload")
        return payload

    async def _dispatch_runtime_handler(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        handler = self._runtime_handler
        if handler is None:
            raise RuntimeError("runtime handler not configured")
        method = getattr(handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"runtime handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("runtime handler returned non-object payload")
        return payload

    async def _dispatch_memory_handler(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        handler = self._memory_handler
        if handler is None:
            raise RuntimeError("memory handler not configured")
        method = getattr(handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"memory handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("memory handler returned non-object payload")
        return payload

    async def _dispatch_voice_handler(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        handler = self._voice_handler
        if handler is None:
            raise RuntimeError("voice handler not configured")
        method = getattr(handler, method_name, None)
        if not callable(method):
            raise RuntimeError(f"voice handler missing {method_name}")
        kwargs = _accepted_keyword_args(method, kwargs)
        payload = await _maybe_await(method(*args, **kwargs))
        if not isinstance(payload, dict):
            raise RuntimeError("voice handler returned non-object payload")
        return payload

    def set_vision_bridge(self, bridge: Any) -> None:
        """Wire the VisionBridge after construction."""
        self._vision_bridge = bridge

    def set_image_archive(self, archive: Any) -> None:
        """Wire the ImageArchive after construction."""
        self._image_archive = archive

    async def _dispatch_archive(
        self,
        image_bytes: bytes,
        label: str,
        description: str,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """Save image_bytes to the archive. Runs blocking IO in a thread."""
        archive = self._image_archive
        if archive is None:
            return {}
        return await asyncio.to_thread(
            archive.save, image_bytes, label, description, width, height
        )

    async def _dispatch_archive_list(self) -> list[dict[str, Any]]:
        """Return all captures metadata list. Runs blocking IO in a thread."""
        archive = self._image_archive
        if archive is None:
            return []
        return await asyncio.to_thread(archive.list_captures)

    async def _dispatch_archive_get(self, capture_id: str) -> dict[str, Any] | None:
        """Return metadata + image_base64 for capture_id. Runs blocking IO in a thread."""
        archive = self._image_archive
        if archive is None:
            return None
        return await asyncio.to_thread(archive.get_capture, capture_id)

    async def _dispatch_archive_delete(self, capture_id: str) -> bool:
        """Delete a capture. Runs blocking IO in a thread."""
        archive = self._image_archive
        if archive is None:
            return False
        return await asyncio.to_thread(archive.delete_capture, capture_id)

    async def _dispatch_snapshot(self) -> dict[str, Any] | None:
        """Capture a camera frame and return base64 JPEG payload."""
        vb = self._vision_bridge
        if vb is None:
            return None
        import asyncio
        import base64
        frame = await asyncio.to_thread(vb._capture_frame)
        if frame is None:
            return None
        try:
            import cv2  # type: ignore[import-untyped]
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode()
            h, w = frame.shape[:2]
            return {
                "image_base64": b64,
                "width": w,
                "height": h,
                "timestamp": datetime.now(_UTC).isoformat(),
            }
        except Exception as exc:
            logger.warning("[Vision] Encode error: %s", exc)
            return None

    async def _dispatch_analyze(self, image_b64: str) -> str:
        """Run VLM on a base64 image and return a Chinese description."""
        vb = self._vision_bridge
        if vb is None:
            return "视觉模块未配置"
        try:
            import base64

            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
            img_bytes = base64.b64decode(image_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            description = await vb._describe_scene_vlm(frame)
            return description
        except Exception as exc:
            logger.warning("[Vision] Analyze error: %s", exc)
            return f"分析失败: {exc}"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.bound_port}"

    @property
    def bound_port(self) -> int:
        """Return the actual bound port once the server has started."""
        return self._bound_port or self.port

    async def start(self) -> asyncio.Task[None] | None:
        """Start the background health server if enabled."""
        if not self.enabled:
            return None
        if self._task is not None and not self._task.done():
            return self._task

        self._started_event = asyncio.Event()
        self._task = asyncio.create_task(self.serve(), name="askme-health-server")
        await self.wait_started(self._task, timeout_s=self._startup_timeout_s)
        logger.info("Askme health server listening on %s", self.url)
        return self._task

    async def serve(self) -> None:
        """Run the HTTP server until ``stop()`` is called."""
        if not self.enabled:
            return

        current_task = asyncio.current_task()
        if self._task is None and current_task is not None:
            self._task = current_task

        self._started_event = asyncio.Event()
        self._bound_port = None
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            access_log=self._access_log,
            log_level=self._log_level,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

        try:
            await self._server.serve()
        except SystemExit:
            logger.warning(
                "Health server failed to bind on %s:%d (port in use?). "
                "Continuing without health endpoint.",
                self.host,
                self.port,
            )
        finally:
            self._started_event.set()
            self._bound_port = None
            self._server = None
            if current_task is self._task:
                self._task = None

    async def stop(self) -> None:
        """Stop the background health server."""
        server = self._server
        if server is None:
            return

        server.should_exit = True
        task = self._task
        if task is None or task.done():
            return

        try:
            await asyncio.wait_for(task, timeout=self._shutdown_timeout_s)
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def wait_started(
        self,
        task: asyncio.Task[None],
        *,
        timeout_s: float = 5.0,
    ) -> None:
        """Wait until the background task has either started or failed."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._server is not None and self._server.started:
                self._bound_port = self._resolve_bound_port()
                self._started_event.set()
                return
            if self._started_event.is_set():
                return
            if task.done():
                exc = task.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError(
                    f"Askme health server exited before binding {self.url}"
                )
            await asyncio.sleep(0.05)

        raise RuntimeError(f"Askme health server did not start within {timeout_s:.1f}s")

    def _resolve_bound_port(self) -> int:
        if self._server is None:
            return self.port

        servers = getattr(self._server, "servers", None) or []
        for running_server in servers:
            sockets = getattr(running_server, "sockets", None) or []
            if sockets:
                return int(sockets[0].getsockname()[1])

        return self.port


def render_prometheus_metrics(snapshot: dict[str, Any]) -> str:
    """Render the runtime snapshot as Prometheus text exposition."""
    voice_status = snapshot.get("voice_pipeline_status", {})
    voice_input = voice_status.get("input", {}) if isinstance(voice_status, dict) else {}
    active_skills = snapshot.get("active_skills", [])
    ota_status = snapshot.get("ota_bridge_status") or snapshot.get("ota_bridge") or {}

    lines: list[str] = []
    _append_metric(lines, "askme_up", "Whether the askme process is running", "gauge", 1)
    _append_metric(
        lines,
        "askme_service_info",
        "Static askme service metadata",
        "gauge",
        1,
        labels={
            "service": snapshot.get("service") or snapshot.get("service_name", "askme"),
            "version": snapshot.get("version") or snapshot.get("service_version", "unknown"),
        },
    )
    _append_metric(
        lines,
        "askme_model_info",
        "Configured primary LLM model",
        "gauge",
        1,
        labels={"model_name": snapshot.get("model_name", "unknown")},
    )
    _append_metric(
        lines,
        "askme_health_status",
        "Overall askme health status (1=ok, 0=degraded)",
        "gauge",
        snapshot.get("status") == "ok",
    )
    _append_metric(
        lines,
        "askme_uptime_seconds",
        "Process uptime in seconds",
        "gauge",
        snapshot.get("uptime_seconds"),
    )
    _append_metric(
        lines,
        "askme_conversations_total",
        "Total conversation turns recorded",
        "counter",
        snapshot.get("total_conversations"),
    )
    _append_metric(
        lines,
        "askme_last_llm_latency_ms",
        "Latency of the most recent LLM call in milliseconds",
        "gauge",
        snapshot.get("last_llm_latency_ms"),
    )
    llm_snap = snapshot.get("llm", {})
    _append_metric(
        lines,
        "askme_llm_latency_p50_ms",
        "LLM call latency p50 over last 100 calls (ms)",
        "gauge",
        llm_snap.get("p50_latency_ms"),
    )
    _append_metric(
        lines,
        "askme_llm_latency_p95_ms",
        "LLM call latency p95 over last 100 calls (ms)",
        "gauge",
        llm_snap.get("p95_latency_ms"),
    )
    _append_metric(
        lines,
        "askme_llm_latency_p99_ms",
        "LLM call latency p99 over last 100 calls (ms)",
        "gauge",
        llm_snap.get("p99_latency_ms"),
    )
    _append_metric(
        lines,
        "askme_active_skills",
        "Number of currently enabled skills",
        "gauge",
        snapshot.get("active_skill_count", len(active_skills) if isinstance(active_skills, list) else 0),
    )

    for skill_name in active_skills if isinstance(active_skills, list) else []:
        _append_metric(
            lines,
            "askme_active_skill_info",
            "Enabled skill metadata",
            "gauge",
            1,
            labels={"skill": skill_name},
        )

    _append_metric(
        lines,
        "askme_voice_pipeline_ok",
        "Whether the voice pipeline is currently healthy",
        "gauge",
        voice_status.get("pipeline_ok"),
    )
    _append_metric(
        lines,
        "askme_voice_mode_enabled",
        "Whether askme is running in voice mode",
        "gauge",
        voice_status.get("mode") == "voice",
    )
    _append_metric(
        lines,
        "askme_voice_input_ready",
        "Whether ASR and VAD are available for voice input",
        "gauge",
        voice_status.get("input_ready"),
    )
    _append_metric(
        lines,
        "askme_voice_output_ready",
        "Whether TTS output is available",
        "gauge",
        voice_status.get("output_ready"),
    )
    _append_metric(
        lines,
        "askme_voice_asr_available",
        "Whether the ASR engine is available",
        "gauge",
        voice_status.get("asr_available"),
    )
    _append_metric(
        lines,
        "askme_voice_vad_available",
        "Whether the VAD engine is available",
        "gauge",
        voice_status.get("vad_available"),
    )
    _append_metric(
        lines,
        "askme_voice_kws_available",
        "Whether the wake-word detector is available",
        "gauge",
        voice_status.get("kws_available"),
    )
    _append_metric(
        lines,
        "askme_voice_tts_busy",
        "Whether TTS is currently playing or queued",
        "gauge",
        voice_status.get("tts_busy"),
    )
    _append_metric(
        lines,
        "askme_voice_last_input_chars",
        "Character length of the most recent recognized voice input",
        "gauge",
        voice_status.get("last_input_chars"),
    )
    _append_metric(
        lines,
        "askme_voice_input_last_peak",
        "Most recent observed microphone peak amplitude",
        "gauge",
        voice_input.get("last_peak") if isinstance(voice_input, dict) else None,
    )
    _append_metric(
        lines,
        "askme_voice_input_peak_max_10s",
        "Maximum observed microphone peak amplitude over the recent window",
        "gauge",
        voice_input.get("peak_max_10s") if isinstance(voice_input, dict) else None,
    )
    _append_metric(
        lines,
        "askme_voice_input_peak_p95_10s",
        "P95 observed microphone peak amplitude over the recent window",
        "gauge",
        voice_input.get("peak_p95_10s") if isinstance(voice_input, dict) else None,
    )
    _append_metric(
        lines,
        "askme_voice_input_last_rms",
        "Most recent observed microphone RMS amplitude",
        "gauge",
        voice_input.get("last_rms") if isinstance(voice_input, dict) else None,
    )
    _append_metric(
        lines,
        "askme_voice_input_rms_p95_10s",
        "P95 observed microphone RMS amplitude over the recent window",
        "gauge",
        voice_input.get("rms_p95_10s") if isinstance(voice_input, dict) else None,
    )
    _append_metric(
        lines,
        "askme_voice_input_asr_timeouts",
        "Count of ASR listen timeouts observed by the audio agent",
        "counter",
        voice_input.get("asr_timeouts") if isinstance(voice_input, dict) else None,
    )
    _append_metric(
        lines,
        "askme_voice_input_sample_count_10s",
        "Number of microphone frames in the recent input diagnostics window",
        "gauge",
        voice_input.get("sample_count_10s") if isinstance(voice_input, dict) else None,
    )

    _append_metric(
        lines,
        "askme_ota_bridge_enabled",
        "Whether OTA bridge reporting is enabled",
        "gauge",
        ota_status.get("enabled"),
    )
    _append_metric(
        lines,
        "askme_ota_bridge_registered",
        "Whether the OTA bridge currently has valid registration",
        "gauge",
        ota_status.get("registered"),
    )
    _append_metric(
        lines,
        "askme_ota_bridge_info",
        "Static OTA bridge metadata",
        "gauge",
        1,
        labels={
            "channel": ota_status.get("channel", ""),
            "device_id": ota_status.get("device_id", ""),
            "product": ota_status.get("product", ""),
            "state": ota_status.get("state", ""),
        },
    )

    return "".join(lines)


def _json_snapshot_response(provider: HealthProvider, endpoint_name: str) -> JSONResponse:
    payload = _snapshot_payload(provider, endpoint_name)
    if isinstance(payload, JSONResponse):
        return payload
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


def _snapshot_payload(
    provider: Callable[[], dict[str, Any]],
    endpoint_name: str,
) -> dict[str, Any] | JSONResponse:
    try:
        return provider()
    except Exception as exc:
        logger.error("Askme %s endpoint failed: %s", endpoint_name, exc, exc_info=True)
        return JSONResponse(
            {"status": "error", "error": str(exc)},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


def _accepted_keyword_args(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return {}
    try:
        parameters = signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _clean_secret(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _is_remote_bind_host(host: str) -> bool:
    cleaned = host.strip().lower()
    if cleaned in _REMOTE_BIND_HOSTS:
        return True
    if cleaned in {"localhost", "127.0.0.1", "::1", "[::1]"}:
        return False
    try:
        address = ipaddress.ip_address(cleaned.strip("[]"))
    except ValueError:
        return True
    return not address.is_loopback


def _append_metric(
    lines: list[str],
    name: str,
    help_text: str,
    metric_type: str,
    value: Any,
    *,
    labels: dict[str, Any] | None = None,
) -> None:
    lines.append(f"# HELP {name} {help_text}\n")
    lines.append(f"# TYPE {name} {metric_type}\n")
    lines.append(f"{name}{_format_labels(labels)} {_format_metric_value(value)}\n")


def _format_labels(labels: dict[str, Any] | None) -> str:
    if not labels:
        return ""

    parts = [
        f'{key}="{_escape_label_value(value)}"'
        for key, value in sorted(labels.items())
    ]
    return "{" + ",".join(parts) + "}"


def _escape_label_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NaN"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if not math.isfinite(numeric):
        return "NaN"
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _disabled_ota_status() -> dict[str, Any]:
    return {
        "enabled": False,
        "state": "disabled",
        "registered": False,
        "device_id": None,
        "channel": "",
        "product": "",
    }


_DASHBOARD_STATIC_DIR = Path(__file__).parent / "static"
_DASHBOARD_ASSET_DIR = _DASHBOARD_STATIC_DIR / "dashboard"
_DASHBOARD_PAGES = frozenset(
    {
        "conversation",
        "field",
        "knowledge",
        "voice",
        "delivery",
    }
)
_DASHBOARD_HTML = (_DASHBOARD_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

build_health_app = create_health_app
HealthServer = AskmeHealthServer
AskmeHealthHTTPServer = AskmeHealthServer
