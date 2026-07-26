"""Enterprise health check service with component-level status."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentHealth:
    """Health status of a single registered component."""

    status: str  # "healthy", "degraded", "unhealthy"
    name: str
    latency_ms: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)


class HealthService:
    """Aggregated health checking for Kubernetes probes and diagnostics.

    Usage::

        svc = HealthService()
        svc.register("llm", check_llm_health)
        svc.register("memory", check_memory_health)
        result = await svc.check_all()
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._checks: dict[str, Callable] = {}

    def register(self, name: str, check_fn: Callable) -> None:
        """Register a health check function for *name*.

        The check function should return a dict with at minimum a ``status`` key
        (``"healthy"``, ``"degraded"``, or ``"unhealthy"``).  It may be sync or
        async.
        """
        self._checks[name] = check_fn

    def uptime_seconds(self) -> float:
        """Seconds since this service was instantiated."""
        return time.time() - self._start_time

    async def check_all(self) -> dict[str, Any]:
        """Run every registered check and return an aggregated health document."""
        results: dict[str, Any] = {}
        overall = "healthy"
        for name, check_fn in self._checks.items():
            try:
                t0 = time.perf_counter()
                status = await check_fn() if asyncio.iscoroutinefunction(check_fn) else check_fn()
                latency = (time.perf_counter() - t0) * 1000
                results[name] = {"status": "healthy", "latency_ms": latency, **(status or {})}
                component_status = str(results[name].get("status") or "healthy")
                if component_status != "healthy":
                    overall = (
                        "unhealthy"
                        if component_status in {"unhealthy", "error"}
                        else "degraded"
                        if overall == "healthy"
                        else overall
                    )
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
                overall = "unhealthy"
        return {
            "status": overall,
            "uptime_s": self.uptime_seconds(),
            "components": results,
        }

    def readiness(self) -> dict[str, Any]:
        """Kubernetes readiness probe -- is the service ready to accept traffic?"""
        return {"ready": True, "uptime_s": self.uptime_seconds()}

    def liveness(self) -> dict[str, Any]:
        """Kubernetes liveness probe -- is the process alive?"""
        return {"alive": True, "status": "ok"}


__all__ = [
    "ComponentHealth",
    "HealthService",
]
