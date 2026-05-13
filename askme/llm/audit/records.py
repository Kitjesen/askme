"""Auditable metadata for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMCallAuditRecord:
    """Non-secret record suitable for runtime traces and customer evidence."""

    purpose: str
    provider: str
    model: str
    session_id: str | None = None
    operator_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    latency_ms: float | None = None
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def redact_llm_messages(messages: list[dict[str, Any]], *, max_chars: int = 160) -> list[dict[str, Any]]:
    """Return a short, non-secret preview of chat messages for diagnostics."""

    redacted: list[dict[str, Any]] = []
    for item in messages:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        redacted.append({"role": role, "content_preview": content})
    return redacted
