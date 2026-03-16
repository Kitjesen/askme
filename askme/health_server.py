"""Embedded HTTP health endpoints for the askme runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
import uvicorn

from askme.runtime_health import RuntimeHealthSnapshot, merge_voice_pipeline_status

logger = logging.getLogger(__name__)

_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_DEGRADED_OTA_STATES = {"auth_error", "degraded"}

HealthProvider = Callable[[], dict[str, Any]]
MetricsProvider = Callable[[], dict[str, Any]]


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
        _now_rec = datetime.now(timezone.utc)
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
    now_utc = datetime.now(timezone.utc)
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
        from askme.runtime_health import get_service_summary
        snapshot["services"] = get_service_summary()
    except Exception:
        pass

    return snapshot


ChatHandler = Callable[[str], Any]  # async def handler(text: str) -> str


def create_health_app(
    provider: HealthProvider | None = None,
    *,
    health_provider: HealthProvider | None = None,
    metrics_provider: MetricsProvider | None = None,
    chat_handler: ChatHandler | None = None,
    conversation_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> FastAPI:
    """Create the HTTP app used for readiness and telemetry probes."""
    resolved_health_provider = health_provider or provider
    if resolved_health_provider is None:
        raise ValueError("health_provider is required")
    resolved_metrics_provider = metrics_provider or resolved_health_provider

    app = FastAPI(
        title="askme-health",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health", tags=["System"])
    async def health() -> JSONResponse:
        return _json_snapshot_response(resolved_health_provider, "health")

    @app.get(
        "/metrics",
        include_in_schema=False,
        tags=["System"],
        response_model=None,
    )
    async def metrics() -> Response:
        payload = _snapshot_payload(resolved_metrics_provider, "metrics")
        if isinstance(payload, JSONResponse):
            return payload
        return PlainTextResponse(
            content=render_prometheus_metrics(payload),
            media_type=_PROMETHEUS_CONTENT_TYPE,
            headers={"Cache-Control": "no-store"},
        )

    @app.get(
        "/metrics/prometheus",
        include_in_schema=False,
        tags=["System"],
        response_model=None,
    )
    async def metrics_prometheus() -> Response:
        payload = _snapshot_payload(resolved_metrics_provider, "metrics")
        if isinstance(payload, JSONResponse):
            return PlainTextResponse(
                content=render_prometheus_metrics({"status": "error"}),
                media_type=_PROMETHEUS_CONTENT_TYPE,
                status_code=payload.status_code,
                headers={"Cache-Control": "no-store"},
            )
        return PlainTextResponse(
            content=render_prometheus_metrics(payload),
            media_type=_PROMETHEUS_CONTENT_TYPE,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/trace", tags=["System"])
    async def trace() -> JSONResponse:
        """Return recent pipeline timing traces for diagnostics."""
        try:
            from askme.pipeline.trace import get_tracer
            tracer = get_tracer()
            return JSONResponse(
                {
                    "summary": tracer.get_summary(),
                    "recent": tracer.get_history(limit=10),
                },
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
                headers={"Cache-Control": "no-store"},
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
            text = body.get("text", "").strip()
            if not text:
                return JSONResponse(
                    {"error": "empty text"},
                    status_code=400,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            reply = await chat_handler(text)
            return JSONResponse(
                {"reply": reply, "text": text},
                headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
            )
        except Exception as exc:
            logger.error("Chat endpoint failed: %s", exc)
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    @app.get("/dashboard", tags=["Monitor"])
    async def dashboard() -> Response:
        """Serve a simple web dashboard for testing and monitoring."""
        return Response(content=_DASHBOARD_HTML, media_type="text/html")

    @app.options("/api/chat", include_in_schema=False)
    async def chat_cors() -> Response:
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
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
                with open(history_file, "r", encoding="utf-8") as fh:
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
        self.host = str(cfg.get("host", "0.0.0.0")).strip() or "0.0.0.0"
        self._access_log = bool(cfg.get("access_log", False))
        self._log_level = str(cfg.get("log_level", "warning")).strip().lower() or "warning"

        raw_port = cfg.get("port", 8765)
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = 8765
        self.port = min(max(port, 1024), 65535)
        self._startup_timeout_s = max(0.1, float(cfg.get("startup_timeout", 5.0)))
        self._shutdown_timeout_s = max(0.1, float(cfg.get("shutdown_timeout", 5.0)))

        self._chat_handler: ChatHandler | None = None

        resolved_health_provider = health_provider or snapshot_provider or provider
        if resolved_health_provider is None:
            raise ValueError("health_provider is required")
        resolved_metrics_provider = metrics_provider or resolved_health_provider

        self._conversation_provider: Callable[[], list[dict[str, Any]]] | None = None

        self._app = create_health_app(
            health_provider=resolved_health_provider,
            metrics_provider=resolved_metrics_provider,
            chat_handler=self._dispatch_chat,
            conversation_provider=self._get_conversation,
        )

    def _get_conversation(self) -> list[dict[str, Any]]:
        if self._conversation_provider is None:
            return []
        return self._conversation_provider()

    async def _dispatch_chat(self, text: str) -> str:
        if self._chat_handler is None:
            return "[chat handler not configured]"
        return await self._chat_handler(text)

    def set_chat_handler(self, handler: ChatHandler) -> None:
        """Wire the chat handler after construction (avoids circular deps)."""
        self._chat_handler = handler

    def set_conversation_provider(self, provider: Callable[[], list[dict[str, Any]]]) -> None:
        """Wire conversation history provider for /api/live endpoint."""
        self._conversation_provider = provider
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._started_event = asyncio.Event()
        self._bound_port: int | None = None

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
        except asyncio.TimeoutError:
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


_DASHBOARD_HTML = (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")

build_health_app = create_health_app
HealthServer = AskmeHealthServer
AskmeHealthHTTPServer = AskmeHealthServer
