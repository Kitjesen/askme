"""Short-lived working memory for task-aware robot dialog."""

from __future__ import annotations

import time
from collections import deque
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

    def record(
        self,
        kind: str,
        content: str,
        *,
        salience: float = 0.5,
        task_id: str = "",
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
        observations: list[str] | None = None,
    ) -> None:
        self.record(
            "operator_utterance",
            user_text,
            salience=0.7,
            task_id=task_id,
            tags=("dialog", "operator"),
        )
        if assistant_text:
            self.record(
                "assistant_reply",
                assistant_text,
                salience=0.5,
                task_id=task_id,
                tags=("dialog", "assistant"),
            )
        for observation in observations or []:
            self.record(
                "observation",
                observation,
                salience=0.6,
                task_id=task_id,
                tags=("observation",),
            )

    def set_focus(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        for key, value in kwargs.items():
            if value is None:
                self._focus.pop(key, None)
            else:
                self._focus[str(key)] = value

    def prune(self) -> None:
        now = time.time()
        retained = [item for item in self._items if not item.is_expired(now)]
        self._items.clear()
        self._items.extend(retained)

    def snapshot(self) -> dict[str, Any]:
        self.prune()
        now = time.time()
        items = [item.to_dict(now=now) for item in self._items]
        return {
            "enabled": self.enabled,
            "persist_enabled": self.persist_enabled,
            "item_count": len(items),
            "retention_seconds": self.retention_seconds,
            "focus": dict(self._focus),
            "items": items,
        }

    def summary(self, *, limit: int = 6) -> str:
        self.prune()
        ranked = sorted(
            self._items,
            key=lambda item: (item.salience, item.created_at),
            reverse=True,
        )
        snippets = [f"{item.kind}: {item.content}" for item in ranked[: max(1, limit)]]
        if self._focus:
            focus = ", ".join(f"{key}={value}" for key, value in sorted(self._focus.items()))
            snippets.insert(0, f"focus: {focus}")
        return "\n".join(snippets)
