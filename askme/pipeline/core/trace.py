"""Pipeline tracing — lightweight timing instrumentation.

Records timing spans for each stage of the voice pipeline
(ASR → intent → LLM → TTS) and exposes them via the health server.

Usage::

    tracer = get_tracer()

    trace = tracer.start_trace("voice_turn")
    with tracer.span("asr"):
        text = await asr_engine.transcribe(audio)
    with tracer.span("llm", model="minimax"):
        response = await llm.chat(text)
    tracer.finish_trace()
    # => "asr: 230ms → llm: 890ms → tts: 340ms = 1460ms total"
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TRACE_HISTORY = 50
_MIN_DURATION_SECONDS = 1e-9


def _monotonic_after(start: float) -> float:
    """Return a monotonic timestamp that is strictly after ``start``.

    On coarse Windows timer ticks a start/end pair can land on the same
    ``time.monotonic()`` value for extremely fast traces.  Trace completion is
    a lifecycle fact, so expose it as a positive duration instead of a
    zero-length completed trace.
    """

    current = time.monotonic()
    if current <= start:
        return start + _MIN_DURATION_SECONDS
    return current


@dataclass
class Span:
    """A single timed segment of the pipeline."""

    name: str
    start: float = 0.0
    end: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end <= 0:
            return 0.0
        return (self.end - self.start) * 1000

    @property
    def is_complete(self) -> bool:
        return self.end > 0


@dataclass
class Trace:
    """A complete pipeline trace for one turn (ASR → LLM → TTS)."""

    id: str = ""
    name: str = ""
    start: float = 0.0
    end: float = 0.0
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        if self.end <= 0:
            return 0.0
        return (self.end - self.start) * 1000

    def summary(self) -> str:
        """Human-readable timing summary.

        Example: ``asr: 230ms → llm: 890ms → tts: 340ms = 1460ms total``
        """
        parts = []
        for span in self.spans:
            if span.is_complete:
                parts.append(f"{span.name}: {span.duration_ms:.0f}ms")
        chain = " → ".join(parts) if parts else "no spans"
        total = f" = {self.total_ms:.0f}ms total" if self.end > 0 else ""
        return f"{chain}{total}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON API response."""
        return {
            "id": self.id,
            "name": self.name,
            "total_ms": round(self.total_ms, 1),
            "summary": self.summary(),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    "metadata": s.metadata,
                }
                for s in self.spans
                if s.is_complete
            ],
            "metadata": self.metadata,
        }


class PipelineTracer:
    """Lightweight pipeline timing instrumentation.

    Thread-safe. One tracer per pipeline; each voice turn starts a new trace.
    Completed traces are kept in a bounded history for the ``/trace`` endpoint.
    """

    def __init__(self, max_history: int = _MAX_TRACE_HISTORY) -> None:
        self._lock = threading.Lock()
        self._current: ContextVar[Trace | None] = ContextVar(
            f"pipeline_trace_{id(self)}",
            default=None,
        )
        self._history: deque[Trace] = deque(maxlen=max_history)
        self._trace_counter = 0

    def start_trace(self, name: str = "voice_turn") -> Trace:
        """Begin a new trace for one pipeline turn."""
        current = self._current.get()
        with self._lock:
            if current is not None and current.end <= 0:
                current.end = _monotonic_after(current.start)
                self._history.append(current)
            self._trace_counter += 1
            trace = Trace(
                id=f"t{self._trace_counter:05d}",
                name=name,
                start=time.monotonic(),
            )
        self._current.set(trace)
        return trace

    def finish_trace(self) -> Trace | None:
        """Complete the current trace and add it to history."""
        trace = self._current.get()
        if trace is None:
            return None
        with self._lock:
            if trace.end <= 0:
                trace.end = _monotonic_after(trace.start)
                self._history.append(trace)
        self._current.set(None)
        logger.info("[Trace] %s", trace.summary())
        return trace

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Generator[Span, None, None]:
        """Context manager for timing a pipeline stage.

        Usage::

            with tracer.span("llm", model="minimax") as s:
                result = await llm.chat(messages)
                s.metadata["tokens"] = len(result)
        """
        s = Span(name=name, start=time.monotonic(), metadata=metadata)
        try:
            yield s
        finally:
            s.end = _monotonic_after(s.start)
            trace = self._current.get()
            with self._lock:
                if trace is not None and trace.end <= 0:
                    trace.spans.append(s)

    def record_span(self, name: str, duration_ms: float, **metadata: Any) -> None:
        """Record a pre-measured span (for stages timed externally)."""
        now = time.monotonic()
        s = Span(
            name=name,
            start=now - duration_ms / 1000,
            end=now,
            metadata=metadata,
        )
        trace = self._current.get()
        with self._lock:
            if trace is not None and trace.end <= 0:
                trace.spans.append(s)

    @property
    def current_trace(self) -> Trace | None:
        return self._current.get()

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent completed traces as dicts (for ``/trace`` API)."""
        with self._lock:
            traces = list(self._history)
        traces.reverse()
        return [t.to_dict() for t in traces[:limit]]

    def get_summary(self) -> dict[str, Any]:
        """Aggregate stats across recent traces."""
        with self._lock:
            traces = list(self._history)
        if not traces:
            return {"count": 0}

        totals = [t.total_ms for t in traces if t.total_ms > 0]
        if not totals:
            return {"count": len(traces), "avg_ms": 0}

        stage_times: dict[str, list[float]] = {}
        for t in traces:
            for s in t.spans:
                if s.is_complete:
                    stage_times.setdefault(s.name, []).append(s.duration_ms)

        stage_avg = {
            name: round(sum(times) / len(times), 1)
            for name, times in stage_times.items()
        }

        sorted_totals = sorted(totals)
        n = len(sorted_totals)
        return {
            "count": len(traces),
            "avg_total_ms": round(sum(totals) / n, 1),
            "p50_total_ms": round(sorted_totals[n // 2], 1),
            "p95_total_ms": round(
                sorted_totals[min(int(n * 0.95), n - 1)], 1
            ),
            "stage_avg_ms": stage_avg,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_tracer: PipelineTracer | None = None


def get_tracer() -> PipelineTracer:
    """Get or create the global pipeline tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = PipelineTracer()
    return _global_tracer
