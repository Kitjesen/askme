"""Compatibility identity helpers for the Conversation Core boundary."""

from __future__ import annotations

from askme.conversation.models import ConflictingThreadAliases


def canonical_thread_id(
    *,
    thread_id: str | None = None,
    conversation_thread_id: str | None = None,
    conversation_session_id: str | None = None,
    conversation_id: str | None = None,
    chat_session_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """Resolve legacy aliases into one canonical logical conversation ID.

    Empty aliases are ignored. Multiple non-empty aliases are accepted only
    when they agree, preventing a request from accidentally crossing threads.
    """

    aliases = {
        "thread_id": thread_id,
        "conversation_thread_id": conversation_thread_id,
        "conversation_session_id": conversation_session_id,
        "conversation_id": conversation_id,
        "chat_session_id": chat_session_id,
        "session_id": session_id,
    }
    normalized = {
        name: str(value).strip()
        for name, value in aliases.items()
        if value is not None and str(value).strip()
    }
    unique = set(normalized.values())
    if len(unique) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in normalized.items())
        raise ConflictingThreadAliases(f"thread aliases disagree: {detail}")
    return next(iter(unique), None)
