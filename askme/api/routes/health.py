"""Trace injection middleware and Kubernetes health probe endpoints.

This module serves two responsibilities:

1. **Trace middleware** -- injects ``trace_id`` into every HTTP request for
   end-to-end correlation in structured logs.

2. **K8s health probes** -- three standardised endpoints (``/healthz``,
   ``/ready``, ``/health``) that return JSON with a ``status`` field,
   compatible with the existing askme health document schema.

Usage from an app factory::

    from askme.api.routes.health import (
        register_trace_middleware,
        register_health_routes,
    )
    from askme.api.services.health_service import HealthService

    app = FastAPI(...)
    register_trace_middleware(app)

    svc = HealthService()
    svc.register("llm", check_fn)
    register_health_routes(app, svc)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.services.health_service import HealthService
from askme.telemetry.tracing import TraceContext, get_trace

# ── Trace middleware ────────────────────────────────────────────────────


def register_trace_middleware(app: FastAPI) -> None:
    """Register the trace_id injection middleware on ``app``.

    The middleware applies to every HTTP route.  It should be registered
    early in the app factory so all downstream handlers and loggers see
    the trace context.
    """

    @app.middleware("http")
    async def _trace_injection_middleware(
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        # Inherit trace_id from request header, or generate a new one.
        trace_id = request.headers.get("X-Trace-Id", "")
        if not trace_id:
            from askme.telemetry.logging import generate_trace_id

            trace_id = generate_trace_id()

        # Run the request pipeline inside a TraceContext so every log
        # record emitted during the request carries this trace_id.
        with TraceContext(trace_id):
            response = await call_next(request)

        # Attach trace_id to the response for end-to-end correlation.
        if isinstance(response, JSONResponse):
            response.headers["X-Trace-Id"] = get_trace()
        elif hasattr(response, "headers"):
            response.headers["X-Trace-Id"] = get_trace()

        return response


# ── K8s health probe endpoints ──────────────────────────────────────────


def _healthz_handler(hs: HealthService) -> dict[str, Any]:
    """Kubernetes liveness probe -- must complete in <10ms."""
    return hs.liveness()


def register_health_routes(
    app: FastAPI,
    health_service: HealthService,
    *,
    routes: tuple[str, ...] = ("healthz", "ready", "health"),
) -> None:
    """Register K8s health probe endpoints on *app*.

    Args:
        app: The FastAPI application to register routes on.
        health_service: A configured :class:`HealthService` instance.
        routes:
            Subset of endpoints to register.  Defaults to all three.
            Pass ``("ready",)`` when ``/healthz`` and ``/health`` are already
            registered by another module (e.g. ``system.py``) to avoid
            duplicate-route conflicts.

    Endpoints
    ---------
    ``GET /healthz``
        Liveness probe.  Returns ``{"alive": true, "status": "ok"}`` in <10ms
        with ``Cache-Control: no-store``.  Response time is the critical metric;
        no component checks run.

    ``GET /ready``
        Readiness probe.  Runs every registered component check and returns an
        aggregated readiness document including per-component status.

    ``GET /health``
        Detailed health document.  Runs every registered component check and
        returns the aggregated result including per-component status and
        latency, plus uptime and snapshot timestamp.
    """

    tags = ["K8s Health"]

    if "healthz" in routes:

        @app.get("/healthz", tags=tags, include_in_schema=False)
        async def healthz() -> JSONResponse:
            return JSONResponse(
                _healthz_handler(health_service),
                headers={"Cache-Control": "no-store"},
            )

    if "ready" in routes:

        @app.get("/ready", tags=tags, include_in_schema=True)
        async def ready() -> JSONResponse:
            payload = await health_service.check_all()
            ready_doc: dict[str, Any] = {
                "ready": all(
                    comp.get("status") == "healthy"
                    for comp in payload.get("components", {}).values()
                ),
                "status": payload["status"],
                "uptime_s": payload["uptime_s"],
                "components": payload["components"],
            }
            return JSONResponse(
                ready_doc,
                headers={"Cache-Control": "no-store"},
            )

    if "health" in routes:

        @app.get("/health", tags=tags, include_in_schema=True)
        async def health() -> JSONResponse:
            payload = await health_service.check_all()
            now_utc = datetime.now(UTC)
            snapshot_at = (
                now_utc.strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{now_utc.microsecond // 1000:03d}Z"
            )
            return JSONResponse(
                {
                    "status": payload["status"],
                    "uptime_s": payload["uptime_s"],
                    "components": payload["components"],
                    "snapshot_at": snapshot_at,
                },
                headers={"Cache-Control": "no-store"},
            )


__all__ = [
    "register_health_routes",
    "register_trace_middleware",
]
