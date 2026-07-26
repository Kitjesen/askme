"""Deterministic, read-only import for legacy conversation JSON files."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from askme.conversation.models import LegacyMigrationResult

if TYPE_CHECKING:
    from askme.conversation.ledger import VoiceTurnLedger

_FALLBACK_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


def migrate_legacy_history(
    history_path: str | Path,
    ledger: VoiceTurnLedger,
) -> LegacyMigrationResult:
    """Import legacy rolling history without mutating or replacing its file.

    Missing IDs and timestamps are derived from content, session identity, and
    sequence so repeated imports produce the same entity and event IDs.
    """

    source = Path(history_path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    sessions = _legacy_sessions(raw)
    imported_thread_ids: list[str] = []
    turn_count = 0
    message_count = 0

    for session_index, (legacy_session_id, messages) in enumerate(sessions.items()):
        thread_id = legacy_session_id or str(
            uuid5(NAMESPACE_URL, "askme:legacy-thread:__default__")
        )
        exchanges = _pair_exchanges(messages)
        message_count += sum(
            1
            for message in messages
            if message.get("role") in {"user", "assistant"} and message.get("content")
        )
        first_timestamp = _first_timestamp(messages)
        thread_at = first_timestamp or (_FALLBACK_EPOCH + timedelta(days=session_index))
        ledger.resolve_thread(
            conversation_thread_id=thread_id,
            channel="voice",
            metadata={
                "legacy_import": True,
                "legacy_session_id": legacy_session_id or None,
            },
            event_id=_stable_id("thread-event", thread_id),
            at=thread_at,
        )
        imported_thread_ids.append(thread_id)

        exchange_occurrences: dict[str, int] = {}
        for sequence, (user_message, assistant_message) in enumerate(exchanges, start=1):
            user_text = _content(user_message)
            assistant_text = _content(assistant_message)
            inferred = _message_datetime(user_message) is None and _message_datetime(
                assistant_message
            ) is None
            fallback = thread_at + timedelta(seconds=sequence * 2)
            turn_at = (
                _message_datetime(user_message)
                or _message_datetime(assistant_message)
                or fallback
            )
            commit_at = _message_datetime(assistant_message) or (turn_at + timedelta(seconds=1))
            exchange_fingerprint = _stable_id(
                "legacy-exchange",
                thread_id,
                _message_identity(user_message),
                _message_identity(assistant_message),
            )
            occurrence = exchange_occurrences.get(exchange_fingerprint, 0) + 1
            exchange_occurrences[exchange_fingerprint] = occurrence
            turn_id = _stable_id("turn", exchange_fingerprint, str(occurrence))
            metadata = {
                "legacy_import": True,
                "legacy_sequence": sequence,
                "legacy_timestamp_inferred": inferred,
            }
            ledger.start_turn(
                thread_id,
                turn_id=turn_id,
                source="legacy",
                user_text=user_text,
                metadata=metadata,
                event_id=_stable_id("turn-start-event", turn_id),
                at=turn_at,
            )
            ledger.commit_turn(
                turn_id,
                assistant_text=assistant_text,
                heard_text=assistant_text,
                metadata=metadata,
                event_id=_stable_id("turn-commit-event", turn_id),
                at=max(turn_at, commit_at),
            )
            turn_count += 1

    return LegacyMigrationResult(
        thread_count=len(imported_thread_ids),
        turn_count=turn_count,
        message_count=message_count,
        thread_ids=tuple(imported_thread_ids),
    )


def _legacy_sessions(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, list):
        return {"": _message_dicts(raw)}
    if isinstance(raw, dict) and isinstance(raw.get("sessions"), dict):
        return {
            str(session_id): _message_dicts(messages)
            for session_id, messages in raw["sessions"].items()
            if isinstance(messages, list)
        }
    return {}


def _message_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    return [message for message in messages if isinstance(message, dict)]


def _pair_exchanges(
    messages: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    exchanges: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    pending_user: dict[str, Any] | None = None
    for message in messages:
        role = message.get("role")
        if role == "user" and message.get("content"):
            if pending_user is not None:
                exchanges.append((pending_user, None))
            pending_user = message
        elif role == "assistant" and message.get("content"):
            exchanges.append((pending_user, message))
            pending_user = None
    if pending_user is not None:
        exchanges.append((pending_user, None))
    return exchanges


def _content(message: dict[str, Any] | None) -> str:
    if message is None:
        return ""
    return str(message.get("content") or "")


def _message_identity(message: dict[str, Any] | None) -> str:
    if message is None:
        return ""
    explicit_id = (
        message.get("message_id")
        or message.get("id")
        or message.get("turn_id")
    )
    if explicit_id is not None and str(explicit_id).strip():
        return f"id:{str(explicit_id).strip()}"
    raw_timestamp = (
        message.get("created_at")
        or message.get("timestamp")
        or message.get("at")
        or ""
    )
    return _stable_id(
        "message",
        str(message.get("role") or ""),
        str(raw_timestamp),
        _content(message),
    )


def _first_timestamp(messages: list[dict[str, Any]]) -> datetime | None:
    for message in messages:
        timestamp = _message_datetime(message)
        if timestamp is not None:
            return timestamp
    return None


def _message_datetime(message: dict[str, Any] | None) -> datetime | None:
    if message is None:
        return None
    raw = message.get("created_at") or message.get("timestamp") or message.get("at")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=UTC)
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (OverflowError, TypeError, ValueError):
        return None


def _stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "askme:" + "\x1f".join(parts)))
