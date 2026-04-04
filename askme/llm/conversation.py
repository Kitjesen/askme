"""
Conversation history manager.

Maintains a rolling window of user/assistant messages, persisted to a
JSON file on disk so history survives restarts.

Usage::

    from askme.brain import ConversationManager

    conv = ConversationManager()
    conv.add_user_message("你好")
    messages = conv.get_messages(system_prompt="你是一个助手。")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from askme.config import get_config, project_root
from askme.robot.ota_bridge import OTABridgeMetrics

if TYPE_CHECKING:
    from askme.memory.session import SessionMemory

logger = logging.getLogger(__name__)

# Sliding window compression constants
COMPRESS_THRESHOLD = 30   # Start compressing when history exceeds this
KEEP_RECENT = 10          # Keep this many recent messages verbatim
HARD_LIMIT = 80           # Absolute max messages — truncate even if compression failed
SUMMARY_TAG = "[对话摘要]"

COMPRESS_PROMPT = """\
请把以下对话历史压缩为一段简洁的摘要（3-5句话）。
保留：关键事实、用户偏好、决策结论、重要上下文。
丢弃：寒暄、重复内容、无意义的确认。

对话：
{conversation}"""


class ConversationManager:
    """Rolling conversation history with JSON persistence and sliding window compression."""

    def __init__(
        self,
        *,
        history_file: str | Path | None = None,
        max_history: int | None = None,
        session_memory: SessionMemory | None = None,
        metrics: OTABridgeMetrics | None = None,
        config: dict | None = None,
    ) -> None:
        cfg = (config if config is not None else get_config()).get("conversation", {})

        # Resolve history file path relative to project root
        raw_path = history_file or cfg.get("history_file", "data/conversation_history.json")
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = project_root() / resolved
        self._history_file: Path = resolved

        self.max_history: int = max_history or cfg.get("max_history", 40)
        self._session_memory = session_memory
        self._metrics = metrics
        self.history: list[dict[str, Any]] = []
        self._compress_backoff_until: float = 0.0  # back off after compression failure
        self._save_scheduled: bool = False  # debounce: coalesce per-turn saves
        # Lock prevents concurrent compress calls from clobbering each other's history.
        self._compress_lock: asyncio.Lock = asyncio.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        """Append a user message and persist."""
        self.history.append({"role": "user", "content": content})
        if self._metrics is not None:
            self._metrics.record_conversation_turn()
        self._trim()
        self._save()

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message and persist."""
        self.history.append({"role": "assistant", "content": content})
        self._trim()
        self._save()

    def add_tool_exchange(
        self,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> None:
        """Record one assistant tool-call turn and its results.

        Tool messages are ephemeral — ``_trim()`` strips them on the next
        regular turn, keeping only user/assistant text in long-term history.
        No ``_save()`` here: tool exchanges are transient within a single turn.
        """
        if not tool_calls:
            return
        self.history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })
        for tr in tool_results:
            self.history.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["content"],
            })

    def get_messages(self, system_prompt: str) -> list[dict[str, Any]]:
        """Return the full message list with system prompt prepended."""
        return [{"role": "system", "content": system_prompt}] + self.history

    def clear(self) -> None:
        """Wipe all history and persist the empty state."""
        self.history = []
        self._save()

    async def maybe_compress(self, llm: Any) -> None:
        """Compress old messages into a summary to extend effective context window.

        Uses a sliding window approach:
          - If history exceeds COMPRESS_THRESHOLD, take the oldest messages
          - Summarize them (including any existing summary) into a single message
          - Keep the most recent KEEP_RECENT messages verbatim

        If HARD_LIMIT is reached (compression keeps failing), the oldest messages
        are dropped without LLM summarization to prevent unbounded memory growth.

        Called by BrainPipeline at the start of each turn. A lock ensures that
        concurrent invocations (background task vs. new turn) never clobber each
        other's in-flight history rebuild.
        """
        # Hard limit: drop oldest messages immediately if compression keeps failing.
        regular = [
            m for m in self.history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if len(regular) > HARD_LIMIT:
            dropped_count = len(regular) - HARD_LIMIT
            logger.warning(
                "[Conversation] Hard limit %d exceeded (%d messages) — "
                "dropping %d oldest messages without compression",
                HARD_LIMIT, len(regular), dropped_count,
            )
            self.history = regular[-HARD_LIMIT:]
            self._save()

        if time.monotonic() < self._compress_backoff_until:
            return

        async with self._compress_lock:
            await self._compress_locked(llm)

    async def _compress_locked(self, llm: Any) -> None:
        """Compression body — must be called while holding _compress_lock."""
        # Skip compression while a tool exchange is in flight — compressing
        # mid-exchange would drop the tool messages and corrupt the API context.
        if any(m.get("role") == "assistant" and m.get("tool_calls") for m in self.history):
            return

        regular = [
            m for m in self.history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if len(regular) <= COMPRESS_THRESHOLD:
            return

        # Separate existing summary (if any) from conversation messages
        existing_summary = ""
        start_idx = 0
        if regular and str(regular[0].get("content", "")).startswith(SUMMARY_TAG):
            existing_summary = regular[0]["content"]
            start_idx = 1

        # Messages to compress: everything except the last KEEP_RECENT
        to_compress = regular[start_idx:-KEEP_RECENT]
        if len(to_compress) < 4:
            return

        # Build text for summarization
        lines = []
        if existing_summary:
            lines.append(existing_summary)
        for m in to_compress:
            role = "用户" if m["role"] == "user" else "助手"
            content = m.get("content", "")
            # Truncate very long messages for summary
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")

        try:
            summary = await asyncio.wait_for(
                llm.chat([
                    {"role": "system", "content": "你是一个对话压缩助手。"},
                    {"role": "user", "content": COMPRESS_PROMPT.format(
                        conversation="\n".join(lines)
                    )},
                ]),
                timeout=10.0,
            )
            summary = summary.strip()
            if not summary:
                return

            # Rebuild history: summary + recent messages that arrived *after*
            # the snapshot.  Messages appended during the LLM await are merged
            # back in to prevent data loss.
            summary_msg = {"role": "assistant", "content": f"{SUMMARY_TAG} {summary}"}
            # Re-read current history so we include any messages that arrived
            # while the LLM was running.  Take KEEP_RECENT from the live state.
            current_regular = [
                m for m in self.history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            recent = current_regular[-KEEP_RECENT:]
            new_count = max(0, len(current_regular) - len(regular))
            self.history = [summary_msg] + recent
            self._save()
            logger.info(
                "[Conversation] Compressed %d messages into summary (%d new arrivals preserved)",
                len(to_compress), new_count,
            )
        except Exception as exc:
            logger.warning("[Conversation] Compression failed: %s", exc)
            self._compress_backoff_until = time.monotonic() + 60.0  # retry after 60s

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Keep only the most recent ``max_history`` user/assistant messages.

        Tool-call and tool-result messages are stripped during trimming to
        prevent unbounded growth from multi-turn tool interactions.
        Trimmed messages are sent to SessionMemory for summarization.
        """
        clean = [
            m for m in self.history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if len(clean) > self.max_history:
            dropped = clean[:-self.max_history]
            clean = clean[-self.max_history:]
            # Summarize dropped messages in background (fire-and-forget)
            if dropped and self._session_memory:
                try:
                    task = asyncio.get_running_loop().create_task(
                        self._session_memory.summarize_and_save(dropped)
                    )
                    task.add_done_callback(
                        lambda t: t.exception() and logger.warning(
                            "[Conversation] Session summary failed: %s", t.exception()
                        )
                    )
                except RuntimeError:
                    # No event loop running (e.g. during tests)
                    logger.debug("[Conversation] No event loop for session summary.")
        self.history = clean

    def _save(self) -> None:
        """Schedule a deferred disk write (debounced).

        Two calls within the same event-loop turn (e.g. add_user + add_assistant)
        collapse into one write, reducing SD-card I/O on sunrise by ~50%.
        Falls back to synchronous write when no event loop is running (tests).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._save_sync()
            return
        if not self._save_scheduled:
            self._save_scheduled = True
            def _done(_: object) -> None:
                self._save_scheduled = False
            t = loop.create_task(asyncio.to_thread(self._save_sync))
            t.add_done_callback(_done)

    def _save_sync(self) -> None:
        """Persist current history to disk (synchronous, runs in thread)."""
        try:
            os.makedirs(self._history_file.parent, exist_ok=True)
            with open(self._history_file, "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            # Persistence is best-effort; never crash the conversation.
            logger.error("[Conversation] Failed to save history: %s", exc)

    def _load(self) -> None:
        """Load history from disk if the file exists."""
        try:
            if self._history_file.exists():
                with open(self._history_file, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                self.history = self._strip_orphan_tool_messages(raw)
        except Exception:
            self.history = []

    @staticmethod
    def _strip_orphan_tool_messages(history: list[dict]) -> list[dict]:
        """Remove tool-call messages that lack a proper context on load.

        Guards against rare crash windows where a tool exchange was partially
        persisted to disk without the paired assistant or follow-up message.
        """
        clean: list[dict] = []
        for msg in history:
            role = msg.get("role")
            if role == "tool":
                # Keep if preceded by an assistant tool_calls msg OR another tool result
                # (multiple tool results from the same assistant tool_calls are all valid)
                prev = clean[-1] if clean else None
                preceded_by_assistant = (
                    prev is not None
                    and prev.get("role") == "assistant"
                    and prev.get("tool_calls")
                )
                preceded_by_tool = prev is not None and prev.get("role") == "tool"
                if preceded_by_assistant or preceded_by_tool:
                    clean.append(msg)
                # else drop orphan tool message
            elif role == "assistant" and not msg.get("content") and not msg.get("tool_calls"):
                pass  # drop degenerate empty assistant message
            else:
                clean.append(msg)
        return clean
