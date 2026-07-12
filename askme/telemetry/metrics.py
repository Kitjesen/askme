"""Prometheus-compatible metrics collectors.

This module provides a lightweight in-memory metrics store that can be
extended with ``prometheus_client`` later.  The interface is kept stable
so callers do not need to change when the backend is swapped.
"""

from __future__ import annotations

from typing import Any

# In-memory metric counters and histograms.
# Extended with prometheus_client in a future iteration.
METRICS: dict[str, Any] = {
    "askme_requests_total": 0,
    "askme_errors_total": 0,
    "askme_llm_latency_ms": [],
    "askme_tts_latency_ms": [],
}


def record_metric(name: str, value: float) -> None:
    """Record a metric sample.

    Currently a no-op placeholder.  When ``prometheus_client`` is added,
    this will dispatch to the appropriate Counter / Histogram.
    """
    _ = name
    _ = value


def increment_counter(name: str, delta: int = 1) -> None:
    """Increment an in-memory counter metric."""
    if name in METRICS and isinstance(METRICS[name], int):
        METRICS[name] += delta


def snapshot_metrics() -> dict[str, Any]:
    """Return a copy of all current metric values."""
    return dict(METRICS)


def reset_metrics() -> None:
    """Reset all in-memory metrics to their initial state."""
    METRICS["askme_requests_total"] = 0
    METRICS["askme_errors_total"] = 0
    METRICS["askme_llm_latency_ms"].clear()
    METRICS["askme_tts_latency_ms"].clear()


__all__ = [
    "METRICS",
    "increment_counter",
    "record_metric",
    "reset_metrics",
    "snapshot_metrics",
]
