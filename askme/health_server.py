"""Embedded HTTP health endpoints for the askme runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from askme.robot.runtime_health import RuntimeHealthSnapshot, merge_voice_pipeline_status

logger = logging.getLogger(__name__)

_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_DEGRADED_OTA_STATES = {"auth_error", "degraded"}

HealthProvider = Callable[[], dict[str, Any]]
MetricsProvider = Callable[[], dict[str, Any]]
CapabilitiesProvider = Callable[[], dict[str, Any]]


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
        _now_rec = datetime.now(UTC)
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
    now_utc = datetime.now(UTC)
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


ChatHandler = Callable[[str], Any]  # async def handler(text: str) -> str


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

    _CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}

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
            "Access-Control-Allow-Headers": "Content-Type",
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
            "Access-Control-Allow-Headers": "Content-Type",
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
            "Access-Control-Allow-Headers": "Content-Type",
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
            "Access-Control-Allow-Headers": "Content-Type",
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
        self._vision_bridge: Any | None = None
        self._image_archive: Any | None = None
        self._capabilities_provider: CapabilitiesProvider | None = None

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

    def set_capabilities_provider(self, provider: CapabilitiesProvider) -> None:
        """Wire the capabilities provider after construction."""
        self._capabilities_provider = provider

    def set_conversation_provider(self, provider: Callable[[], list[dict[str, Any]]]) -> None:
        """Wire conversation history provider for /api/live endpoint."""
        self._conversation_provider = provider

    def _get_capabilities(self) -> dict[str, Any]:
        if self._capabilities_provider is None:
            return {}
        return self._capabilities_provider()

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
                "timestamp": datetime.now(UTC).isoformat(),
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
        except TimeoutError:
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
