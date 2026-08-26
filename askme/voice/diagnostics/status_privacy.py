"""Privacy-safe serialization for transcript-bearing voice status payloads."""

from __future__ import annotations

from typing import Any

_TEXT_FIELDS = frozenset(
    {
        "answer",
        "content",
        "final_text",
        "input_text",
        "partial_text",
        "prompt",
        "query",
        "question",
        "raw_text",
        "reply",
        "text",
        "transcript",
        "user_input",
        "user_text",
        "utterance",
    }
)
_TEXT_FIELD_SUFFIXES = (
    "_content",
    "_prompt",
    "_query",
    "_reply",
    "_text",
    "_transcript",
    "_utterance",
)


def sanitize_voice_status(
    status: dict[str, Any],
    *,
    include_transcripts: bool = False,
) -> dict[str, Any]:
    """Return a copy of *status* with transcript text replaced by metadata."""
    sanitized: dict[str, Any] = {}
    for key, value in status.items():
        if _is_private_text_field(key) and (
            isinstance(value, str) or value is None
        ):
            text = value if isinstance(value, str) else ""
            if include_transcripts is True:
                sanitized[key] = text
            sanitized[f"{key}_present"] = bool(text)
            sanitized[f"{key}_chars"] = len(text)
            continue
        sanitized[key] = _sanitize_value(
            value,
            include_transcripts=include_transcripts,
        )
    return sanitized


def _is_private_text_field(key: object) -> bool:
    normalized = str(key or "").strip().lower()
    return normalized in _TEXT_FIELDS or normalized.endswith(_TEXT_FIELD_SUFFIXES)


def _sanitize_value(value: Any, *, include_transcripts: bool) -> Any:
    if isinstance(value, dict):
        return sanitize_voice_status(
            value,
            include_transcripts=include_transcripts,
        )
    if isinstance(value, list):
        return [_sanitize_value(item, include_transcripts=include_transcripts) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_value(item, include_transcripts=include_transcripts) for item in value
        )
    return value
