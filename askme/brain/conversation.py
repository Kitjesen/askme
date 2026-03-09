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
from pathlib import Path
from typing import Any, TYPE_CHECKING

from askme.config import get_config, project_root
from askme.ota_bridge import OTABridgeMetrics

if TYPE_CHECKING:
    from askme.brain.session_memory import SessionMemory

logger = logging.getLogger(__name__)

# Sliding window compression constants
COMPRESS_THRESHOLD = 30   # Start compressing when history exceeds this
KEEP_RECENT = 10          # Keep this many recent messages verbatim
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
    ) -> None:
        cfg = get_config().get("conversation", {})

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

        Called by BrainPipeline at the start of each turn.
        """
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

            # Rebuild history: summary + recent messages
            summary_msg = {"role": "assistant", "content": f"{SUMMARY_TAG} {summary}"}
            recent = regular[-KEEP_RECENT:]
            self.history = [summary_msg] + recent
            self._save()
            logger.info("[Conversation] Compressed %d messages into summary", len(to_compress))
        except Exception as exc:
            logger.warning("[Conversation] Compression failed: %s", exc)

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
                    asyncio.get_running_loop().create_task(
                        self._session_memory.summarize_and_save(dropped)
                    )
                except RuntimeError:
                    # No event loop running (e.g. during tests)
                    logger.debug("[Conversation] No event loop for session summary.")
        self.history = clean

    def _save(self) -> None:
        """Persist current history to disk."""
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
                    self.history = json.load(fh)
        except Exception:
            self.history = []
