"""Embedded HTTP health endpoints for the askme runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.composition import ApiRouteDependencies, register_api_routes
from askme.api.services import field_runtime_callback_security as _callback_security
from askme.api.services import prometheus_metrics as _prometheus_metrics
from askme.api.services.conversation_service import (
    ConversationService,
    authorized_runtime_context_from_body,
)
from askme.api.services.field_route_roots import (
    field_operations_path_roots as _resolve_field_operations_path_roots,
)
from askme.api.services.field_runtime_callback_security import (
    field_runtime_callback_delivery_body as _field_runtime_callback_delivery_body,
)
from askme.api.services.field_runtime_callback_security import (
    field_runtime_callback_trust as _field_runtime_callback_trust,
)
from askme.api.services.field_runtime_plan import (
    build_field_runtime_plan_from_event as _field_runtime_plan_from_event,
)
from askme.api.services.field_runtime_plan import (
    field_runtime_delivery_status as _field_runtime_delivery_status,
)
from askme.api.services.http_helpers import accepted_keyword_args as _accepted_keyword_args
from askme.api.services.http_helpers import clean_secret as _clean_secret
from askme.api.services.http_helpers import is_remote_bind_host as _is_remote_bind_host
from askme.api.services.http_helpers import json_snapshot_response as _json_snapshot_response
from askme.api.services.http_helpers import maybe_await as _maybe_await
from askme.api.services.http_helpers import public_error_payload as _public_error_payload
from askme.api.services.http_helpers import snapshot_payload as _snapshot_payload
from askme.api.services.http_runtime_config import (
    api_documentation_urls as _api_documentation_urls,
)
from askme.api.services.http_runtime_config import (
    conversation_runtime_settings as _conversation_runtime_settings,
)
from askme.api.services.monitor_service import MonitorService
from askme.api.services.prometheus_metrics import render_prometheus_metrics
from askme.config import get_config, project_root
from askme.governance import OperatorDirectory
from askme.robot.dog.runtime_health import RuntimeHealthSnapshot, merge_voice_pipeline_status

logger = logging.getLogger(__name__)

_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_DEGRADED_OTA_STATES = {"auth_error", "degraded"}
_UTC = timezone.utc  # noqa: UP017 - Sunrise runs Python 3.10, where datetime.UTC is unavailable.
_PUBLIC_HTTP_PATHS = frozenset(("/health", "/healthz", "/metrics", "/metrics/prometheus"))
_PROTECTED_HTTP_PREFIXES = ("/api/", "/dashboard/")
_PROTECTED_HTTP_PATHS = frozenset(("/dashboard", "/trace"))
_FIELD_EVIDENCE_ROOT_NAMES = ("artifacts", "output", "data")
_append_metric = _prometheus_metrics.append_metric
_escape_label_value = _prometheus_metrics.escape_label_value
_format_labels = _prometheus_metrics.format_labels
_format_metric_value = _prometheus_metrics.format_metric_value
sign_field_runtime_callback_payload = _callback_security.sign_field_runtime_callback_payload

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


def _field_operations_path_roots(app_config: dict[str, Any]) -> dict[str, Path]:
    return _resolve_field_operations_path_roots(app_config, project_root=project_root())


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


def _skill_growth_candidate_prompt(candidate: dict[str, Any]) -> str:
    examples = candidate.get("examples") if isinstance(candidate.get("examples"), list) else []
    reasons = candidate.get("reasons") if isinstance(candidate.get("reasons"), list) else []
    example_text = "\n".join(f"- {item}" for item in examples[:5]) or "- 无"
    reason_text = ", ".join(str(item) for item in reasons[:5]) or "unknown"
    return (
        "你是园区机器狗的受控技能草稿。这个技能来自在线增长候选池，"
        "必须保持低风险、可审计、可拒绝。\n\n"
        "处理目标：根据用户输入判断是否能完成该候选需求；如果缺少地图、知识、"
        "传感器或权限，就明确说明需要人工配置或转交管理员，不要编造结果。\n\n"
        f"候选摘要：{candidate.get('summary') or ''}\n"
        f"候选原因：{reason_text}\n"
        f"证据数量：{candidate.get('evidence_count') or 0}\n"
        f"历史表达：\n{example_text}\n\n"
        "用户输入：{{user_input}}\n\n"
        "请用一句到三句话回复，先说明能否处理，再说明下一步。"
    )


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
            _now_rec.strftime("%Y-%m-%dT%H:%M:%S.") + f"{_now_rec.microsecond // 1000:03d}Z"
        )

    degraded_reasons: list[str] = []
    if not merged_voice_status.get("pipeline_ok", True):
        degraded_reasons.append("voice_pipeline")
    if ota_status and ota_status.get("enabled") and ota_status.get("state") in _DEGRADED_OTA_STATES:
        degraded_reasons.append("ota_bridge")

    # ISO 8601 UTC timestamp for this snapshot — lets OTA Agent detect stale payloads.
    now_utc = datetime.now(_UTC)
    snapshot_at = now_utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now_utc.microsecond // 1000:03d}Z"

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
        from askme.robot.dog.runtime_health import get_service_summary

        snapshot["services"] = get_service_summary()
    except Exception:
        logger.exception("[Health] Runtime service summary fetch failed")

    return snapshot


ChatHandler = Callable[..., Any]  # async def handler(text: str, *, speak: bool = False)


VisionSnapshotHandler = Callable[[], Any]  # async () -> dict | None
VisionAnalyzeHandler = Callable[[str], Any]  # async (image_b64: str) -> str

# async (image_bytes, label, description, width, height) -> dict
ArchiveSnapshotHandler = Callable[[bytes, str, str, int, int], Any]
ArchiveListHandler = Callable[[], Any]  # async () -> list[dict]
ArchiveGetHandler = Callable[[str], Any]  # async (capture_id) -> dict | None
ArchiveDeleteHandler = Callable[[str], Any]  # async (capture_id) -> bool


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
        field_runtime_callback_secret or os.getenv("ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET")
    )
    resolved_runtime_callback_max_age_s = max(1.0, float(field_runtime_callback_max_age_s))
    app_config = get_config()
    field_path_roots = _field_operations_path_roots(app_config)
    conversation_settings = _conversation_runtime_settings(app_config)
    if field_operations_handler is None:
        from askme.pipeline.field.field_operations import FieldOperationsService

        field_operations_handler = FieldOperationsService.from_env()
    if space_handler is None:
        from askme.space import ParkSpaceService

        space_handler = ParkSpaceService.from_config(app_config)

    api_documentation_urls = _api_documentation_urls(app_config)
    app = FastAPI(
        title="askme-health",
        docs_url=api_documentation_urls["docs_url"],
        redoc_url=api_documentation_urls["redoc_url"],
        openapi_url=api_documentation_urls["openapi_url"],
    )
    _CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}
    _MISSION_JSON_HEADERS = {"Cache-Control": "no-store", **_CORS_HEADERS}
    _CORS_ALLOW_HEADERS = (
        "Content-Type, Authorization, X-Askme-Api-Key, X-Askme-Operator-Id, X-Operator-Id"
    )
    _operator_directory = OperatorDirectory(app_config)

    def _json_error(message: str, *, status_code: int) -> JSONResponse:
        return JSONResponse(
            _public_error_payload(message, message=message),
            status_code=status_code,
            headers=_CORS_HEADERS,
        )

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

    def _blueprint_catalog_payload() -> dict[str, Any]:
        from askme.blueprints import catalog_payload

        return catalog_payload(config=app_config)

    def _request_has_control_auth(request: Request) -> bool:
        if not resolved_control_api_key:
            return False
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            supplied = bearer[7:].strip()
            if secrets.compare_digest(supplied, resolved_control_api_key):
                return True
        supplied_key = request.headers.get("x-askme-api-key", "").strip()
        return bool(supplied_key and secrets.compare_digest(supplied_key, resolved_control_api_key))

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
                _public_error_payload(
                    "control API authentication required",
                    message="Control API authentication is required.",
                    next_action="Send a Bearer token or X-Askme-Api-Key header.",
                ),
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
        body: Any = None
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030"):
            try:
                body = json.loads(raw.decode(encoding))
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_error = exc
        else:
            raise ValueError(f"JSON object body could not be decoded: {last_error}")
        if not isinstance(body, dict):
            raise ValueError("JSON object body required")
        return body

    def _operator_action_kwargs(body: dict[str, Any]) -> dict[str, Any]:
        decision = body.get("operator_auth")
        permission = (
            str(decision.get("permission") or "").strip() if isinstance(decision, dict) else ""
        )
        context = authorized_runtime_context_from_body(
            body,
            conversation_session_id=str(body.get("conversation_session_id") or "").strip(),
            permission=permission or None,
        )
        operator_context = None
        if context is not None:
            operator_context = {
                "operator_id": context.operator_id,
                "roles": list(context.operator_roles),
                "authenticated": context.operator_authenticated,
                "source": context.operator_source,
                "permission": context.permission,
            }
            if context.conversation_session_id:
                operator_context["conversation_session_id"] = context.conversation_session_id

        return {
            "operator_id": context.operator_id
            if context is not None
            else str(body.get("operator_id") or ""),
            "reason": str(body.get("reason") or ""),
            "risk_acknowledgement": bool(
                body.get("risk_acknowledgement") or body.get("risk_ack") or body.get("acknowledged")
            ),
            "operator_context": operator_context,
        }

    def _field_manual_trigger_body(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        payload.setdefault("operator_id", _operator_id_from_request(request, body))
        payload.setdefault("trigger_source", "operator_manual")
        payload.setdefault("admission_path", "field_events_manual")
        return payload

    def _operator_id_from_request(request: Request, body: dict[str, Any]) -> str:
        identity = _operator_directory.resolve_context(headers=request.headers, body=body)
        return str(identity.operator_id or "dashboard.operator").strip()

    def _require_permission(
        request: Request,
        body: dict[str, Any],
        permission: str,
    ) -> JSONResponse | None:
        if permission.startswith("runtime:") and permission != "runtime:read":
            explicit_operator_id = str(
                body.get("operator_id")
                or request.headers.get("x-askme-operator-id")
                or request.headers.get("x-operator-id")
                or request.headers.get("x-askme-iam-operator-id")
                or ""
            ).strip()
            if not explicit_operator_id:
                return _mission_json(
                    {
                        "ok": False,
                        "error": "operator authorization provenance unavailable",
                        "reason": "runtime_operator_context_required",
                        "message": "Runtime mutations require an explicit operator identity.",
                        "next_action": "Attach an operator identity before requesting runtime control.",
                    },
                    status_code=403,
                )

        decision = _operator_directory.authorize(
            None,
            permission,
            headers=request.headers,
            body=body,
        )
        if decision.get("allowed"):
            body["operator_id"] = decision.get("operator", {}).get(
                "operator_id"
            ) or _operator_id_from_request(request, body)
            body["operator_auth"] = decision
            return None
        return _mission_json(
            {
                "ok": False,
                "error": "operator not authorized",
                "reason": decision.get("reason") or "operator_missing_permission",
                "operator_auth": decision,
                "message": "The current operator is not allowed to perform this action.",
                "next_action": "Switch to an operator with the required permission or request approval.",
            },
            status_code=403,
        )

    def _operator_directory_payload() -> dict[str, Any]:
        return _operator_directory.payload()

    def _identity_readiness_payload() -> dict[str, Any]:
        return _operator_directory.identity_gateway_readiness()

    def _current_operator_payload(operator_id: str | None, headers: Any = None) -> dict[str, Any]:
        return _operator_directory.current_operator_payload(operator_id, headers=headers)

    def _operator_authorization_payload(
        operator_id: str | None,
        permission: str,
        headers: Any = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _operator_directory.authorize(operator_id, permission, headers=headers, body=body)

    def _looks_like_device_ingest_without_scenario(body: dict[str, Any]) -> bool:
        if body.get("scenario_id"):
            return False
        source = str(body.get("source") or "").strip().lower()
        if source in {"camera", "sensor", "robot", "mqtt", "ros", "hikvision"}:
            return True
        return any(
            key in body
            for key in ("device_id", "camera_id", "sensor", "robot", "detections", "predictions")
        )

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
            logger.exception("[Health] Voice delivery record failed")
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
            logger.exception("[Health] Runtime delivery record failed")
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
            directive.get("resolved_profile") or directive.get("requested_profile") or ""
        ).strip()
        if profile_id:
            try:
                delivery["profile"] = await _dispatch_voice(
                    "set_voice_profile_payload",
                    {"profile_id": profile_id},
                )
            except Exception as exc:
                logger.exception("[Health] Voice profile set failed")
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
            logger.exception("[Health] Voice playback failed")
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
            logger.exception("[Health] Runtime plan submission failed")
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
                    handoff.get("handoff_id") or runtime_result.get("handoff_id") or ""
                ),
                "current_state": str(run.get("current_state") or runtime_result.get("state") or ""),
                "reason": str(runtime_result.get("reason") or ""),
            }
        )
        result["runtime_delivery"] = delivery
        return await _record_field_runtime_delivery(result, delivery)

    conversation_service = ConversationService(
        chat_handler=chat_handler,
        memory_handler=memory_handler,
        logger=logger,
        chat_timeout_s=conversation_settings.chat_timeout_s,
        chat_max_concurrency=conversation_settings.chat_max_concurrency,
        chat_slow_threshold_ms=conversation_settings.chat_slow_threshold_ms,
        chat_diagnostics_history_limit=conversation_settings.chat_diagnostics_history_limit,
        capabilities_provider=capabilities_provider,
        space_dispatch=lambda method_name, body: _dispatch_space(method_name, body),
    )
    monitor_service = MonitorService(
        config_provider=get_config,
        project_root=project_root(),
        conversation_provider=conversation_provider,
        logger=logger,
    )

    def _metrics_provider_with_conversation() -> dict[str, Any]:
        snapshot = resolved_metrics_provider()
        if not isinstance(snapshot, dict):
            return snapshot
        payload = dict(snapshot)
        payload["conversation_runtime"] = conversation_service.metrics_snapshot()["chat"]
        return payload

    register_api_routes(
        app,
        ApiRouteDependencies(
            health_provider=resolved_health_provider,
            metrics_provider=_metrics_provider_with_conversation,
            render_prometheus_metrics=render_prometheus_metrics,
            json_snapshot_response=_json_snapshot_response,
            snapshot_payload=_snapshot_payload,
            prometheus_content_type=_PROMETHEUS_CONTENT_TYPE,
            governance_payload=_operator_directory_payload,
            identity_readiness_payload=_identity_readiness_payload,
            current_operator_payload=_current_operator_payload,
            authorization_payload=_operator_authorization_payload,
            mission_json=_mission_json,
            cors_options_response=_cors_options_response,
            dispatch_memory=lambda method_name, body: _dispatch_memory(method_name, body),
            logger=logger,
            authorize=_require_permission,
            dispatch_cognition=_dispatch_cognition,
            json_error=_json_error,
            cors_headers=_CORS_HEADERS,
            dispatch_runtime=_dispatch_runtime,
            optional_json_body=_optional_json_body,
            operator_action_kwargs=_operator_action_kwargs,
            dispatch_voice=_dispatch_voice,
            dispatch_space=lambda method_name, body: _dispatch_space(method_name, body),
            dispatch_field_operations=_dispatch_field_operations,
            field_manual_trigger_body=_field_manual_trigger_body,
            looks_like_device_ingest_without_scenario=_looks_like_device_ingest_without_scenario,
            dispatch_field_voice_directive=_dispatch_field_voice_directive,
            dispatch_field_runtime_policy=_dispatch_field_runtime_policy,
            runtime_callback_trust=_field_runtime_callback_trust,
            runtime_callback_delivery_body=_field_runtime_callback_delivery_body,
            runtime_callback_secret=resolved_runtime_callback_secret,
            runtime_callback_max_age_s=resolved_runtime_callback_max_age_s,
            field_path_roots=field_path_roots,
            config_provider=get_config,
            dashboard_html=_DASHBOARD_HTML,
            dashboard_asset_dir=_DASHBOARD_ASSET_DIR,
            dashboard_pages=_DASHBOARD_PAGES,
            capabilities_provider=capabilities_provider,
            blueprints_provider=_blueprint_catalog_payload,
            operator_id_from_request=_operator_id_from_request,
            conversation_service=conversation_service,
            runtime_available=runtime_handler is not None,
            runtime_voice_turn_timeout_s=conversation_settings.runtime_voice_turn_timeout_s,
            monitor_service=monitor_service,
            dispatch_mission=_dispatch_mission,
            request_has_control_auth=_request_has_control_auth,
            skill_growth_candidate_prompt=_skill_growth_candidate_prompt,
            vision_snapshot_handler=vision_snapshot_handler,
            vision_analyze_handler=vision_analyze_handler,
            archive_snapshot_handler=archive_snapshot_handler,
            archive_list_handler=archive_list_handler,
            archive_get_handler=archive_get_handler,
            archive_delete_handler=archive_delete_handler,
        ),
    )

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
            cfg.get("allow_unsafe_remote", False) or cfg.get("allow_unsafe_control_api", False)
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
        from askme.pipeline.field.field_operations import FieldOperationsService

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

    async def _dispatch_chat(
        self,
        text: str,
        *,
        speak: bool = False,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ) -> Any:
        if self._chat_handler is None:
            return "[chat handler not configured]"
        kwargs = _accepted_keyword_args(
            self._chat_handler,
            {
                "speak": speak,
                "conversation_session_id": conversation_session_id,
                "planning_session_id": planning_session_id,
                "runtime_policy": runtime_policy,
                "runtime_bridge_mode": runtime_policy,
            },
        )
        return await _maybe_await(self._chat_handler(text, **kwargs))

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

    async def health_payload(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._dispatch_memory_handler("health_payload", payload or {})

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

    async def system_control_payload(self) -> dict[str, Any]:
        return await self._dispatch_voice_handler("system_control_payload")

    async def switch_system_component_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._dispatch_voice_handler(
            "switch_system_component_payload",
            payload,
        )

    async def update_prompt_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_voice_handler("update_prompt_payload", payload)

    async def speak_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch_voice_handler("speak_payload", payload)

    async def synthesize_speech_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._dispatch_voice_handler("synthesize_speech_payload", payload)

    async def speech_playback_audio_payload(
        self,
        playback_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch_voice_handler(
            "speech_playback_audio_payload",
            playback_id,
            payload or {},
        )

    async def speech_playback_status_payload(
        self,
        playback_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch_voice_handler(
            "speech_playback_status_payload",
            playback_id,
            payload or {},
        )

    async def cancel_speech_playback_payload(
        self,
        playback_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._dispatch_voice_handler(
            "cancel_speech_playback_payload",
            playback_id,
            payload,
        )

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
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "pause_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )

    async def runtime_resume_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "resume_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )

    async def runtime_cancel_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "cancel_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )

    async def runtime_advance_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch_runtime_handler(
            "advance_payload",
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
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
        return await asyncio.to_thread(archive.save, image_bytes, label, description, width, height)

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
        from askme.providers import capture_snapshot_payload

        return await capture_snapshot_payload(vb)

    async def _dispatch_analyze(self, image_b64: str) -> str:
        """Run VLM on a base64 image and return a Chinese description."""
        vb = self._vision_bridge
        if vb is None:
            return "视觉模块未配置"
        from askme.providers import analyze_image_base64

        return await analyze_image_base64(vb, image_b64)

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
                raise RuntimeError(f"Askme health server exited before binding {self.url}")
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


def _disabled_ota_status() -> dict[str, Any]:
    return {
        "enabled": False,
        "state": "disabled",
        "registered": False,
        "device_id": None,
        "channel": "",
        "product": "",
    }


from askme.api.services.dashboard_pages import dashboard_page_slugs

_DASHBOARD_STATIC_DIR = Path(__file__).parent / "static"
_DASHBOARD_ASSET_DIR = _DASHBOARD_STATIC_DIR / "dashboard"
_DASHBOARD_PAGES = frozenset(dashboard_page_slugs())
_DASHBOARD_HTML = (_DASHBOARD_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

build_health_app = create_health_app
HealthServer = AskmeHealthServer
AskmeHealthHTTPServer = AskmeHealthServer

__all__ = [
    "ArchiveDeleteHandler",
    "ArchiveGetHandler",
    "ArchiveListHandler",
    "ArchiveSnapshotHandler",
    "AskmeHealthHTTPServer",
    "AskmeHealthServer",
    "CapabilitiesProvider",
    "ChatHandler",
    "CognitionHandler",
    "FieldOperationsHandler",
    "HealthProvider",
    "HealthServer",
    "HealthSnapshotProvider",
    "MemoryHandler",
    "MetricsProvider",
    "MissionHandler",
    "RuntimeHandler",
    "SpaceHandler",
    "VisionAnalyzeHandler",
    "VisionSnapshotHandler",
    "VoiceHandler",
    "_append_metric",
    "_escape_label_value",
    "_format_labels",
    "_format_metric_value",
    "build_health_app",
    "build_health_snapshot",
    "create_health_app",
    "render_prometheus_metrics",
    "sign_field_runtime_callback_payload",
]
