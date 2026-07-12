"""Request tracing with context variables."""

from __future__ import annotations

import contextvars
import uuid

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def set_trace(trace_id: str) -> None:
    """Set the current trace_id in context."""
    trace_id_var.set(trace_id)


def get_trace() -> str:
    """Return the current trace_id from context."""
    return trace_id_var.get()


class TraceContext:
    """Context manager that sets and restores a trace_id.

    Usage::

        with TraceContext("abc123"):
            logger.info("hello")  # includes trace_id=abc123
    """

    def __init__(self, trace_id: str = "") -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self._token: contextvars.Token[str] | None = None

    def __enter__(self) -> TraceContext:
        self._token = trace_id_var.set(self.trace_id)
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            trace_id_var.reset(self._token)


__all__ = [
    "TraceContext",
    "get_trace",
    "set_trace",
    "trace_id_var",
]
