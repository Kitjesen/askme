"""Process-local conversation context projection for gateway compatibility."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(slots=True)
class ConversationTurn:
    """One projected user/assistant exchange in the gateway context cache."""

    user_text: str = ""
    assistant_text: str = ""
    intent: str | None = None
    gate_decision: str | None = None
    skill_name: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    handoff_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    turn_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.created_at = _as_utc(self.created_at)
        self.tool_calls = deepcopy(self.tool_calls)
        self.metadata = deepcopy(self.metadata)
        self.sequence = max(0, int(self.sequence or 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "intent": self.intent,
            "gate_decision": self.gate_decision,
            "skill_name": self.skill_name,
            "tool_calls": deepcopy(self.tool_calls),
            "handoff_id": self.handoff_id,
            "metadata": deepcopy(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class ConversationSession:
    """Mutable process-local context projection for one gateway conversation."""

    channel: str
    operator_id: str | None = None
    robot_id: str | None = None
    site_id: str | None = None
    status: str = "active"
    active_planning_session_id: str | None = None
    current_task_id: str | None = None
    handoff_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid4()))
    summary: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    last_activity_at: datetime = field(default_factory=_utc_now)
    closed_at: datetime | None = None
    close_reason: str | None = None

    def __post_init__(self) -> None:
        self.created_at = _as_utc(self.created_at)
        self.updated_at = _as_utc(self.updated_at)
        self.last_activity_at = _as_utc(self.last_activity_at)
        if self.closed_at is not None:
            self.closed_at = _as_utc(self.closed_at)
        self.metadata = deepcopy(self.metadata)

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Read model for a conversation session.

    Snapshot values are deep-copied from the store so callers can inspect nested
    metadata/tool call structures without mutating live session state.
    """

    session_id: str
    channel: str
    operator_id: str | None
    robot_id: str | None
    site_id: str | None
    status: str
    active_planning_session_id: str | None
    current_task_id: str | None
    handoff_id: str | None
    metadata: dict[str, Any]
    summary: str
    turns: tuple[ConversationTurn, ...]
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    closed_at: datetime | None
    close_reason: str | None


class InMemorySessionStore:
    """Thread-safe process-local store for gateway context projections."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> ConversationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: ConversationSession) -> ConversationSession:
        with self._lock:
            self._sessions[session.session_id] = session
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list(self) -> list[ConversationSession]:
        with self._lock:
            return list(self._sessions.values())

    def find_active(
        self,
        *,
        channel: str,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
    ) -> ConversationSession | None:
        with self._lock:
            for session in self._sessions.values():
                if (
                    session.status == "active"
                    and session.channel == channel
                    and session.operator_id == operator_id
                    and session.robot_id == robot_id
                    and session.site_id == site_id
                ):
                    return session
            return None


class ConversationSessionManager:
    """Maintain the voice gateway's process-local compatibility projection.

    Conversation Core owns authoritative thread and turn lifecycle. This manager
    only caches delivered turns for bridge context and compatibility diagnostics.
    """

    def __init__(
        self,
        store: InMemorySessionStore | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store or InMemorySessionStore()
        self._clock = clock or _utc_now
        self._lock = RLock()

    @property
    def store(self) -> InMemorySessionStore:
        return self._store

    def get_or_create(
        self,
        *,
        channel: str,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        status: str = "active",
        active_planning_session_id: str | None = None,
        current_task_id: str | None = None,
        handoff_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ConversationSession:
        now = self._now()
        requested_session_id = session_id or None
        with self._lock:
            if requested_session_id is not None:
                existing = self._store.get(requested_session_id)
                if existing is not None:
                    if existing.status != "active":
                        raise ValueError(
                            f"Conversation session is not active: {requested_session_id}"
                        )
                    _validate_session_identity(
                        existing,
                        channel=channel,
                        operator_id=operator_id,
                        robot_id=robot_id,
                        site_id=site_id,
                    )
                    existing.last_activity_at = now
                    existing.updated_at = now
                    if metadata:
                        existing.metadata.update(deepcopy(metadata))
                    if active_planning_session_id is not None:
                        existing.active_planning_session_id = active_planning_session_id
                    if current_task_id is not None:
                        existing.current_task_id = current_task_id
                    if handoff_id is not None:
                        existing.handoff_id = handoff_id
                    return existing

            if requested_session_id is None and any(
                _has_identity(value) for value in (operator_id, robot_id, site_id)
            ):
                existing = self._store.find_active(
                    channel=channel,
                    operator_id=operator_id,
                    robot_id=robot_id,
                    site_id=site_id,
                )
                if existing is not None:
                    existing.last_activity_at = now
                    existing.updated_at = now
                    if metadata:
                        existing.metadata.update(deepcopy(metadata))
                    if active_planning_session_id is not None:
                        existing.active_planning_session_id = active_planning_session_id
                    if current_task_id is not None:
                        existing.current_task_id = current_task_id
                    if handoff_id is not None:
                        existing.handoff_id = handoff_id
                    return existing

            session = ConversationSession(
                session_id=requested_session_id or str(uuid4()),
                channel=channel,
                operator_id=operator_id,
                robot_id=robot_id,
                site_id=site_id,
                status=status,
                active_planning_session_id=active_planning_session_id,
                current_task_id=current_task_id,
                handoff_id=handoff_id,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
                last_activity_at=now,
            )
            return self._store.save(session)

    def append_turn(
        self,
        session_id: str,
        turn: ConversationTurn | None = None,
        *,
        user_text: str = "",
        assistant_text: str = "",
        intent: str | None = None,
        gate_decision: str | None = None,
        skill_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        handoff_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> ConversationTurn:
        now = self._now()
        with self._lock:
            session = self._require_session(session_id)
            if session.status != "active":
                raise ValueError(f"Conversation session is not active: {session_id}")
            new_turn = turn or ConversationTurn(
                user_text=user_text,
                assistant_text=assistant_text,
                intent=intent,
                gate_decision=gate_decision,
                skill_name=skill_name,
                tool_calls=tool_calls or [],
                handoff_id=handoff_id,
                metadata=metadata or {},
                created_at=now,
            )
            if new_turn.sequence <= 0:
                new_turn.sequence = len(session.turns) + 1
            session.turns.append(new_turn)
            session.last_activity_at = now
            session.updated_at = now
            if handoff_id:
                session.handoff_id = handoff_id
            session.summary = summary if summary is not None else self._summarize_latest(new_turn)
            self._store.save(session)
            return new_turn

    def update_summary(self, session_id: str, summary: str) -> ConversationSession:
        now = self._now()
        with self._lock:
            session = self._require_session(session_id)
            session.summary = summary
            session.updated_at = now
            self._store.save(session)
            return session

    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        active_planning_session_id: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        current_task_id: str | None = None,
        handoff_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> ConversationSession:
        now = self._now()
        with self._lock:
            session = self._require_session(session_id)
            if status is not None:
                session.status = status
            if active_planning_session_id is not None:
                session.active_planning_session_id = active_planning_session_id
            if operator_id is not None:
                session.operator_id = operator_id
            if robot_id is not None:
                session.robot_id = robot_id
            if site_id is not None:
                session.site_id = site_id
            if current_task_id is not None:
                session.current_task_id = current_task_id
            if handoff_id is not None:
                session.handoff_id = handoff_id
            if metadata:
                session.metadata.update(deepcopy(metadata))
            if summary is not None:
                session.summary = summary
            session.updated_at = now
            self._store.save(session)
            return session

    def close_session(
        self,
        session_id: str,
        *,
        status: str = "closed",
        reason: str | None = None,
    ) -> ConversationSession | None:
        now = self._now()
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            session.status = status
            session.updated_at = now
            session.closed_at = now
            session.close_reason = reason or status
            self._store.save(session)
            return session

    def expire_idle_sessions(
        self,
        idle_after: timedelta,
        *,
        now: datetime | None = None,
        status: str = "expired",
        reason: str | None = None,
    ) -> list[str]:
        current = _as_utc(now) if now is not None else self._now()
        cutoff = current - idle_after
        expired: list[str] = []
        with self._lock:
            for session in self._store.list():
                if session.status != "active":
                    continue
                if session.last_activity_at <= cutoff:
                    session.status = status
                    session.updated_at = current
                    session.closed_at = current
                    session.close_reason = reason or status
                    self._store.save(session)
                    expired.append(session.session_id)
        return expired

    def snapshot(self, session_id: str) -> SessionSnapshot | None:
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            return SessionSnapshot(
                session_id=session.session_id,
                channel=session.channel,
                operator_id=session.operator_id,
                robot_id=session.robot_id,
                site_id=session.site_id,
                status=session.status,
                active_planning_session_id=session.active_planning_session_id,
                current_task_id=session.current_task_id,
                handoff_id=session.handoff_id,
                metadata=deepcopy(session.metadata),
                summary=session.summary,
                turns=tuple(deepcopy(session.turns)),
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
                closed_at=session.closed_at,
                close_reason=session.close_reason,
            )

    def context_payload(
        self,
        session_id: str,
        *,
        recent_turn_limit: int = 6,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """Return a compact, prompt-safe view of one conversation session."""

        snapshot = self.snapshot(session_id)
        if snapshot is None:
            return {}
        turn_limit = max(0, int(recent_turn_limit))
        recent = list(snapshot.turns[-turn_limit:]) if turn_limit else []
        char_limit = None if max_chars is None else max(0, int(max_chars))
        remaining = char_limit
        lines: list[str] = []
        turns: list[dict[str, Any]] = []

        for turn in recent:
            turn_payload = turn.to_dict()
            line_parts = []
            if turn.user_text:
                line_parts.append(f"User: {turn.user_text}")
            if turn.assistant_text:
                line_parts.append(f"Assistant: {turn.assistant_text}")
            line = " | ".join(line_parts)
            if remaining is not None:
                separator_cost = 1 if lines else 0
                available = remaining - separator_cost
                if available <= 0:
                    break
                if len(line) > available:
                    line = line[:available]
                    if turn_payload.get("assistant_text"):
                        turn_payload["assistant_text"] = str(turn_payload["assistant_text"])[
                            :available
                        ]
                    elif turn_payload.get("user_text"):
                        turn_payload["user_text"] = str(turn_payload["user_text"])[:available]
                remaining -= separator_cost + len(line)
            turns.append(turn_payload)
            if line:
                lines.append(line)

        return {
            "session_id": snapshot.session_id,
            "channel": snapshot.channel,
            "status": snapshot.status,
            "operator_id": snapshot.operator_id,
            "robot_id": snapshot.robot_id,
            "site_id": snapshot.site_id,
            "active_planning_session_id": snapshot.active_planning_session_id,
            "current_task_id": snapshot.current_task_id,
            "handoff_id": snapshot.handoff_id,
            "summary": snapshot.summary,
            "turn_count": len(snapshot.turns),
            "recent_turns": turns,
            "text": "\n".join(lines),
            "created_at": snapshot.created_at.isoformat(),
            "updated_at": snapshot.updated_at.isoformat(),
            "last_activity_at": snapshot.last_activity_at.isoformat(),
            "closed_at": snapshot.closed_at.isoformat() if snapshot.closed_at else None,
            "close_reason": snapshot.close_reason,
        }

    def _now(self) -> datetime:
        return _as_utc(self._clock())

    def _require_session(self, session_id: str) -> ConversationSession:
        session = self._store.get(session_id)
        if session is None:
            raise KeyError(f"Unknown conversation session: {session_id}")
        return session

    @staticmethod
    def _summarize_latest(turn: ConversationTurn) -> str:
        parts = []
        if turn.user_text:
            parts.append(f"User: {turn.user_text}")
        if turn.assistant_text:
            parts.append(f"Assistant: {turn.assistant_text}")
        return " | ".join(parts)


def _has_identity(value: str | None) -> bool:
    return bool(str(value or "").strip())


def _validate_session_identity(
    session: ConversationSession,
    *,
    channel: str,
    operator_id: str | None,
    robot_id: str | None,
    site_id: str | None,
) -> None:
    """Reject attempts to reuse an explicit session under a different identity."""

    if session.channel != channel:
        raise ValueError(
            f"Conversation session identity conflict for {session.session_id}: channel"
        )
    for field_name, supplied in (
        ("operator_id", operator_id),
        ("robot_id", robot_id),
        ("site_id", site_id),
    ):
        if _has_identity(supplied) and getattr(session, field_name) != supplied:
            raise ValueError(
                f"Conversation session identity conflict for {session.session_id}: {field_name}"
            )
