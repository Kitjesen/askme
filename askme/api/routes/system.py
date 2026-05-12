"""System-level FastAPI routes.

This module is the first step in moving the HTTP surface out of the legacy
``askme.health_server`` monolith. It intentionally receives helpers and
providers from the app factory so the existing behavior can stay stable while
routes move into product-oriented modules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, Response

HealthProvider = Callable[[], dict[str, Any]]
MetricsProvider = Callable[[], dict[str, Any]]
RenderPrometheusMetrics = Callable[[dict[str, Any]], str]
JsonSnapshotResponse = Callable[[HealthProvider, str], JSONResponse]
SnapshotPayload = Callable[[MetricsProvider, str], dict[str, Any] | JSONResponse]


def register_system_routes(
    app: FastAPI,
    *,
    health_provider: HealthProvider,
    metrics_provider: MetricsProvider,
    render_prometheus_metrics: RenderPrometheusMetrics,
    json_snapshot_response: JsonSnapshotResponse,
    snapshot_payload: SnapshotPayload,
    prometheus_content_type: str,
) -> None:
    """Register base health, metrics, and trace routes on ``app``."""

    @app.get("/health", tags=["System"])
    async def health() -> JSONResponse:
        return json_snapshot_response(health_provider, "health")

    @app.get("/healthz", include_in_schema=False, tags=["System"])
    async def healthz() -> JSONResponse:
        return json_snapshot_response(health_provider, "healthz")

    @app.get(
        "/metrics",
        include_in_schema=False,
        tags=["System"],
        response_model=None,
    )
    async def metrics() -> Response:
        payload = snapshot_payload(metrics_provider, "metrics")
        if isinstance(payload, JSONResponse):
            return payload
        return PlainTextResponse(
            content=render_prometheus_metrics(payload),
            media_type=prometheus_content_type,
            headers={"Cache-Control": "no-store"},
        )

    @app.get(
        "/metrics/prometheus",
        include_in_schema=False,
        tags=["System"],
        response_model=None,
    )
    async def metrics_prometheus() -> Response:
        payload = snapshot_payload(metrics_provider, "metrics")
        if isinstance(payload, JSONResponse):
            return PlainTextResponse(
                content=render_prometheus_metrics({"status": "error"}),
                media_type=prometheus_content_type,
                status_code=payload.status_code,
                headers={"Cache-Control": "no-store"},
            )
        return PlainTextResponse(
            content=render_prometheus_metrics(payload),
            media_type=prometheus_content_type,
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
