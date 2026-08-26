"""Prometheus-compatible metrics collectors.

This module provides a lightweight in-memory metrics store that can be
extended with ``prometheus_client`` later. The interface is kept stable
so callers do not need to change when the backend is swapped.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

# In-memory metric counters and bounded histogram samples.
# This stable interface can move to prometheus_client without changing callers.
METRICS: dict[str, Any] = {
    "askme_requests_total": 0,
    "askme_errors_total": 0,
    "askme_llm_latency_ms": [],
    "askme_tts_latency_ms": [],
}
_LOCK = Lock()
_MAX_SAMPLES_PER_METRIC = 512


def record_metric(name: str, value: float) -> None:
    """Record a bounded in-memory histogram sample."""
    with _LOCK:
        samples = METRICS.setdefault(str(name), [])
        if not isinstance(samples, list):
            return
        samples.append(float(value))
        overflow = len(samples) - _MAX_SAMPLES_PER_METRIC
        if overflow > 0:
            del samples[:overflow]


def increment_counter(name: str, delta: int = 1) -> None:
    """Increment an in-memory counter metric."""
    with _LOCK:
        current = METRICS.setdefault(str(name), 0)
        if isinstance(current, int):
            METRICS[str(name)] = current + int(delta)


def snapshot_metrics() -> dict[str, Any]:
    """Return a detached copy of all current metric values."""
    with _LOCK:
        return {
            name: list(value) if isinstance(value, list) else value
            for name, value in METRICS.items()
        }


def reset_metrics() -> None:
    """Reset all known in-memory metrics while preserving their types."""
    with _LOCK:
        for name, value in METRICS.items():
            if isinstance(value, list):
                value.clear()
            elif isinstance(value, int):
                METRICS[name] = 0


__all__ = [
    "METRICS",
    "increment_counter",
    "record_metric",
    "reset_metrics",
    "snapshot_metrics",
]
