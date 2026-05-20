"""Platform health and monitoring route registration."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from askme.api.routes.monitor import register_monitor_routes
from askme.api.routes.system import register_system_routes


def register_platform_routes(app: FastAPI, deps: Any) -> None:
    """Register health, metrics, and operational monitor routes."""

    register_system_routes(
        app,
        health_provider=deps.health_provider,
        metrics_provider=deps.metrics_provider,
        render_prometheus_metrics=deps.render_prometheus_metrics,
        json_snapshot_response=deps.json_snapshot_response,
        snapshot_payload=deps.snapshot_payload,
        prometheus_content_type=deps.prometheus_content_type,
    )
    register_monitor_routes(
        app,
        monitor_service=deps.monitor_service,
        logger=deps.logger,
    )


__all__ = ["register_platform_routes"]
