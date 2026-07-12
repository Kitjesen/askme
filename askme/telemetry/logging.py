"""Enterprise structured logging with trace correlation."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

from askme.telemetry.tracing import get_trace


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON with trace_id correlation.

    Example output::

        {"timestamp":"2026-06-01T10:30:00.123+00:00","level":"INFO",\
         "logger":"askme.api","message":"request started",\
         "trace_id":"a1b2c3d4e5f6g7h8","service":"askme","version":"4.1.0"}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace() or getattr(record, "trace_id", ""),
            "service": "askme",
            "version": os.environ.get("ASKME_VERSION", "4.1.0"),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with JSON formatting to stderr.

    Call once at process startup to replace default logging with
    structured JSON output including trace_id on every record.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def generate_trace_id() -> str:
    """Return a short hex trace identifier."""
    import uuid

    return uuid.uuid4().hex[:16]


__all__ = [
    "JsonFormatter",
    "generate_trace_id",
    "setup_structured_logging",
]
