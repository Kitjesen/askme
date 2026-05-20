"""Short-lived working memory for task-aware robot dialog."""
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkingMemoryItem:
    """One short-lived memory item for the current operator session."""

    kind: str
    content: str
    salience: float = 0.5
    task_id: str = ""
    tags: tuple[str, ...] = ()
    session_id: str = ""
    turn_id: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        return self.expires_at is not None and (now if now is not None else time.time()) > self.expires_at

    def to_dict(self, *, now: float | None = None) -> dict[str, Any]:
        current = now if now is not None else time.time()
        return {
            "kind": self.kind,
            "content": self.content,
            "salience": self.salience,
            "task_id": self.task_id,
            "tags": list(self.tags),
            "session_id": self.session_id,
            "conversation_session_id": self.session_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "age_s": round(max(0.0, current - self.created_at), 3),
            "expires_at": self.expires_at,
            "expired": self.is_expired(current),
            "metadata": dict(self.metadata),
        }


class WorkingMemory:
    """Privacy-conscious in-memory task context.

    This layer is deliberately ephemeral by default.  Persistent memory belongs
    behind a policy-controlled memory service, not this planning scratchpad.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_items: int = 80,
        retention_seconds: float = 1800.0,
        persist_enabled: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.persist_enabled = bool(persist_enabled)
        self.retention_seconds = max(1.0, float(retention_seconds))
        self._items: deque[WorkingMemoryItem] = deque(maxlen=max(1, int(max_items)))
        self._focus: dict[str, Any] = {}
        self._session_focus: dict[str, dict[str, Any]] = {}

    def record(
        self,
        kind: str,
        content: str,
        *,
        salience: float = 0.5,
        task_id: str = "",
        session_id: str = "",
        conversation_session_id: str = "",
        turn_id: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_s: float | None = None,
    ) -> WorkingMemoryItem | None:
        if not self.enabled:
            return None
        text = str(content or "").strip()
        if not text:
            return None
        now = time.time()
        expires_at = now + (self.retention_seconds if ttl_s is None else max(1.0, float(ttl_s)))
        item = WorkingMemoryItem(
            kind=str(kind or "note"),
            content=text,
            salience=min(max(float(salience), 0.0), 1.0),
            task_id=str(task_id or ""),
            tags=tuple(str(tag) for tag in tags or () if str(tag).strip()),
            session_id=str(conversation_session_id or session_id or ""),
            turn_id=str(turn_id or ""),
            created_at=now,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )
        self._items.append(item)
        self.prune()
        return item

    def record_turn(
        self,
        user_text: str,
        *,
        assistant_text: str = "",
        task_id: str = "",
        session_id: str = "",
        conversation_session_id: str = "",
        turn_id: str = "",
        observations: list[str] | None = None,
    ) -> None:
        conversation_session = str(conversation_session_id or session_id or "")
        self.record(
            "operator_utterance",
            user_text,
            salience=0.7,
            task_id=task_id,
            conversation_session_id=conversation_session,
            turn_id=turn_id,
            tags=("dialog", "operator"),
        )
        if assistant_text:
            self.record(
                "assistant_reply",
                assistant_text,
                salience=0.5,
                task_id=task_id,
                conversation_session_id=conversation_session,
                turn_id=turn_id,
                tags=("dialog", "assistant"),
            )
        for observation in observations or []:
            self.record(
                "observation",
                observation,
                salience=0.6,
                task_id=task_id,
                conversation_session_id=conversation_session,
                turn_id=turn_id,
                tags=("observation",),
            )

    def set_focus(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        session = str(
            kwargs.get("conversation_session_id") or kwargs.get("session_id") or ""
        ).strip()
        target = (
            self._session_focus.setdefault(session, {})
            if session
            else self._focus
        )
        for key, value in kwargs.items():
            if value is None:
                target.pop(str(key), None)
            else:
                target[str(key)] = deepcopy(value)
        if not session:
            return
        if not target:
            self._session_focus.pop(session, None)
            if str(self._focus.get("conversation_session_id") or "") == session:
                self._focus.pop("conversation_session_id", None)

    def prune(self) -> None:
        now = time.time()
        retained = [item for item in self._items if not item.is_expired(now)]
        self._items.clear()
        self._items.extend(retained)

    def select_context(
        self,
        *,
        max_items: int | None = None,
        max_chars: int | None = None,
        kinds: list[str] | tuple[str, ...] | set[str] | None = None,
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        conversation_session_id: str | None = None,
        include_focus: bool = True,
    ) -> dict[str, Any]:
        """Return a ranked, budgeted context slice for prompt construction."""

        self.prune()
        allowed_kinds = {str(kind) for kind in kinds or ()}
        required_tags = {str(tag) for tag in tags or ()}
        task_filter = None if task_id is None else str(task_id or "")
        session_value = conversation_session_id if conversation_session_id is not None else session_id
        session_filter = None if session_value is None else str(session_value or "")
        item_limit = None if max_items is None else max(0, int(max_items))
        char_limit = None if max_chars is None else max(0, int(max_chars))
        now = time.time()

        ranked = sorted(
            (
                item
                for item in self._items
                if (not allowed_kinds or item.kind in allowed_kinds)
                and (not required_tags or required_tags.issubset(set(item.tags)))
                and (task_filter is None or item.task_id == task_filter)
                and (session_filter is None or item.session_id == session_filter)
            ),
            key=lambda item: (item.salience, item.created_at),
            reverse=True,
        )
        if item_limit is not None:
            ranked = ranked[:item_limit]

        selected: list[dict[str, Any]] = []
        lines: list[str] = []
        remaining = char_limit

        focus_payload = self._focus_for(session_filter) if include_focus else {}
        if focus_payload:
            focus = ", ".join(f"{key}={value}" for key, value in sorted(focus_payload.items()))
            focus_line = f"focus: {focus}"
            if remaining is None:
                lines.append(focus_line)
            elif remaining > 0:
                lines.append(focus_line[:remaining])
                remaining -= len(lines[-1])

        for item in ranked:
            item_dict = item.to_dict(now=now)
            prefix = f"{item.kind}: "
            line = f"{prefix}{item.content}"
            if remaining is not None:
                separator_cost = 1 if lines else 0
                available = remaining - separator_cost
                if available <= 0:
                    break
                if len(line) > available:
                    content_available = max(0, available - len(prefix))
                    item_dict["content"] = item.content[:content_available]
                    line = f"{prefix}{item_dict['content']}"[:available]
                remaining -= separator_cost + len(line)
            selected.append(item_dict)
            lines.append(line)

        return {
            "focus": focus_payload,
            "item_count": len(selected),
            "items": selected,
            "text": "\n".join(lines),
        }

    def promote_candidates(
        self,
        *,
        limit: int = 5,
        min_salience: float = 0.75,
        session_id: str | None = None,
        conversation_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.prune()
        session_value = conversation_session_id if conversation_session_id is not None else session_id
        session_filter = None if session_value is None else str(session_value or "")
        threshold = min(max(float(min_salience), 0.0), 1.0)
        ranked = sorted(
            (
                item
                for item in self._items
                if item.salience >= threshold
                and (session_filter is None or item.session_id == session_filter)
            ),
            key=lambda item: (item.salience, item.created_at),
            reverse=True,
        )
        now = time.time()
        return [item.to_dict(now=now) for item in ranked[: max(0, int(limit))]]

    def clear_session(
        self,
        session_id: str = "",
        *,
        conversation_session_id: str = "",
    ) -> int:
        session = str(conversation_session_id or session_id or "")
        before = len(self._items)
        retained = [item for item in self._items if item.session_id != session]
        self._items.clear()
        self._items.extend(retained)
        if session:
            self._session_focus.pop(session, None)
            if str(self._focus.get("conversation_session_id") or "") == session:
                self._focus.clear()
        return before - len(retained)

    def snapshot(self) -> dict[str, Any]:
        self.prune()
        now = time.time()
        items = [item.to_dict(now=now) for item in self._items]
        focus = dict(self._focus)
        if not focus and len(self._session_focus) == 1:
            session, session_focus = next(iter(self._session_focus.items()))
            if {"last_plan_id", "planning_session_id"} & set(session_focus):
                focus["conversation_session_id"] = session
        return {
            "enabled": self.enabled,
            "persist_enabled": self.persist_enabled,
            "item_count": len(items),
            "retention_seconds": self.retention_seconds,
            "focus": focus,
            "session_focus": deepcopy(self._session_focus),
            "items": items,
        }

    def summary(
        self,
        *,
        limit: int = 6,
        session_id: str | None = None,
        conversation_session_id: str | None = None,
    ) -> str:
        session_value = conversation_session_id if conversation_session_id is not None else session_id
        context = self.select_context(
            max_items=max(1, int(limit)),
            session_id=session_value,
            include_focus=True,
        )
        return str(context.get("text") or "")

    def _focus_for(self, session_id: str | None) -> dict[str, Any]:
        if session_id is not None:
            return deepcopy(self._session_focus.get(str(session_id or ""), {}))
        return deepcopy(self._focus)
