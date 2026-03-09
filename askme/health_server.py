"""Embedded HTTP health endpoints for the askme runtime."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn

from askme.runtime_health import RuntimeHealthSnapshot

logger = logging.getLogger(__name__)

_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class HealthSnapshotProvider:
    """Compatibility wrapper that builds runtime health snapshots on demand."""

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
    ) -> None:
        self._snapshotter = RuntimeHealthSnapshot(
            app_name=app_name,
            app_version=app_version,
            brain_config={"model": default_model},
            voice_mode=voice_mode,
            robot_mode=robot_mode,
            metrics_provider=metrics.snapshot,
            active_skill_names_provider=lambda: [
                skill.name for skill in skill_manager.get_enabled()
            ],
            voice_status_provider=voice_status_provider,
            ota_status_provider=(
                ota_status_provider
                if ota_status_provider is not None
                else lambda: {
                    "enabled": False,
                    "state": "disabled",
                    "registered": False,
                    "device_id": None,
                    "channel": "",
                    "product": "",
                }
            ),
        )

    def health_snapshot(self) -> dict[str, Any]:
        return self._snapshotter.health_snapshot()

    def metrics_snapshot(self) -> dict[str, Any]:
        return self._snapshotter.metrics_snapshot()


def build_health_app(
    *,
    health_provider: Callable[[], dict[str, Any]],
    metrics_provider: Callable[[], dict[str, Any]],
) -> FastAPI:
    """Create the HTTP app used for readiness and telemetry probes."""
    app = FastAPI(
        title="askme-health",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health", tags=["System"])
    async def health() -> dict[str, Any]:
        return health_provider()

    @app.get("/metrics", include_in_schema=False, tags=["System"])
    async def metrics(request: Request) -> Any:
        snapshot = metrics_provider()
        accept = request.headers.get("accept", "")
        if "text/plain" in accept:
            return PlainTextResponse(
                content=render_prometheus_metrics(snapshot),
                media_type=_PROMETHEUS_CONTENT_TYPE,
            )
        return snapshot

    return app


create_health_app = build_health_app


class AskmeHealthServer:
    """Run the embedded FastAPI health server inside the current event loop."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        health_provider: Callable[[], dict[str, Any]],
        metrics_provider: Callable[[], dict[str, Any]],
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.host = str(cfg.get("host", "0.0.0.0")).strip() or "0.0.0.0"
        self._access_log = bool(cfg.get("access_log", False))
        self._log_level = str(cfg.get("log_level", "warning")).strip().lower() or "warning"
        self._startup_timeout_s = max(0.1, float(cfg.get("startup_timeout", 5.0)))
        self._shutdown_timeout_s = max(0.1, float(cfg.get("shutdown_timeout", 5.0)))

        raw_port = cfg.get("port", 8765)
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = 8765
        self.port = min(max(port, 1), 65535)

        self._app = build_health_app(
            health_provider=health_provider,
            metrics_provider=metrics_provider,
        )
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> asyncio.Task[None] | None:
        if not self.enabled:
            return None
        if self._task is not None and not self._task.done():
            return self._task

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
        self._task = asyncio.create_task(
            self._server.serve(),
            name="askme-health-server",
        )
        await self._wait_until_started(timeout_s=self._startup_timeout_s)
        logger.info("Askme health server listening on %s", self.url)
        return self._task

    async def stop(self) -> None:
        task = self._task
        server = self._server
        if task is None or server is None:
            return

        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=self._shutdown_timeout_s)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._task = None
            self._server = None

    async def _wait_until_started(self, *, timeout_s: float) -> None:
        assert self._task is not None
        assert self._server is not None

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if self._server.started:
                return
            if self._task.done():
                exc = self._task.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError(
                    f"Askme health server exited before binding {self.url}"
                )
            await asyncio.sleep(0.05)

        raise RuntimeError(
            f"Askme health server did not start within {timeout_s:.1f}s"
        )


AskmeHealthHTTPServer = AskmeHealthServer
HealthServer = AskmeHealthServer


def render_prometheus_metrics(snapshot: dict[str, Any]) -> str:
    """Render the runtime snapshot as Prometheus text exposition."""
    lines: list[str] = []
    voice_status = snapshot.get("voice_pipeline_status", {})
    active_skills = snapshot.get("active_skills", [])
    ota_status = snapshot.get("ota_bridge_status", {})

    _append_metric(lines, "askme_up", "Whether the askme process is running", "gauge", 1)
    _append_metric(
        lines,
        "askme_service_info",
        "Static askme service metadata",
        "gauge",
        1,
        labels={
            "service": snapshot.get("service_name", "askme"),
            "version": snapshot.get("service_version", "unknown"),
        },
    )
    _append_metric(
        lines,
        "askme_model_info",
        "Active LLM model metadata",
        "gauge",
        1,
        labels={"model_name": snapshot.get("model_name", "unknown")},
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
    _append_metric(
        lines,
        "askme_active_skills",
        "Number of currently enabled skills",
        "gauge",
        snapshot.get("active_skill_count", 0),
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

    return "".join(lines)


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
