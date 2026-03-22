"""
Session memory — Layer 2 of the three-layer memory architecture.

Layer 1: Rolling JSON conversation (ConversationManager) — short-term, 20 messages
Layer 2: Session summaries as .md files (this module) — medium-term, days/weeks
Layer 3: MemU vector DB (MemoryBridge) — long-term, permanent facts/preferences

When conversation history trims old messages, they're summarized and saved here.
Recent session summaries are injected into the system prompt so the assistant
maintains continuity across conversations.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from askme.config import get_config, project_root

if TYPE_CHECKING:
    from askme.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Maximum number of recent session files to include in system prompt
MAX_RECENT_SESSIONS = 5
# Maximum total characters from session summaries in system prompt
MAX_SUMMARY_CHARS = 1500

SUMMARIZE_PROMPT = """\
请用中文简洁概括以下对话内容（2-4句话）。
重点提取：用户提到的事实、偏好、需求、关键决策。
忽略寒暄和无意义的对话。如果对话没有有意义的内容，回复"无重要内容"。

对话：
{conversation}"""


class SessionMemory:
    """Manages session summary .md files for medium-term memory."""

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        cfg = get_config()
        data_dir = cfg.get("app", {}).get("data_dir", "data")
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = project_root() / resolved
        self._sessions_dir: Path = resolved / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        # Cache for get_recent_summaries() — avoids glob on every LLM turn
        self._summary_cache: str = ""
        self._summary_cache_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def summarize_and_save(self, messages: list[dict[str, Any]]) -> None:
        """Summarize trimmed messages and save as a session .md file.

        Called by ConversationManager when messages are dropped during trim.
        """
        if not messages or not self._llm:
            return

        # Format messages for summarization
        conversation_text = self._format_messages(messages)
        if not conversation_text.strip():
            return

        try:
            summary = await asyncio.wait_for(
                self._llm.chat([
                    {"role": "system", "content": "你是一个对话摘要助手。"},
                    {"role": "user", "content": SUMMARIZE_PROMPT.format(
                        conversation=conversation_text
                    )},
                ]),
                timeout=10.0,
            )
            summary = summary.strip()
            if summary and summary != "无重要内容":
                self._save_summary(summary)
                logger.info("[SessionMemory] Saved session summary: %s", summary[:60])
        except Exception as exc:
            logger.warning("[SessionMemory] Summarization failed: %s", exc)

    def get_recent_summaries(self) -> str:
        """Load recent session summaries for system prompt injection (cached 5s).

        Returns a formatted string of recent session summaries, or empty string.
        """
        import time as _time
        now = _time.monotonic()
        if (now - self._summary_cache_time) < 5.0:
            return self._summary_cache
        session_files = sorted(self._sessions_dir.glob("*.md"), reverse=True)
        if not session_files:
            self._summary_cache = ""
            self._summary_cache_time = now
            return ""

        summaries = []
        total_chars = 0
        for f in session_files[:MAX_RECENT_SESSIONS]:
            try:
                content = f.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                # Extract date from filename (YYYY-MM-DD_HHMMSS.md)
                date_str = f.stem.replace("_", " ", 1)
                entry = f"[{date_str}] {content}"
                if total_chars + len(entry) > MAX_SUMMARY_CHARS:
                    break
                summaries.append(entry)
                total_chars += len(entry)
            except Exception:
                continue

        if not summaries:
            result = ""
        else:
            result = "历史会话摘要:\n" + "\n".join(summaries)
        self._summary_cache = result
        self._summary_cache_time = now
        return result

    def save_direct(self, summary: str) -> None:
        """Save a summary directly without LLM summarization."""
        if summary.strip():
            self._save_summary(summary.strip())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_summary(self, summary: str) -> None:
        """Write summary to a timestamped .md file."""
        # Use microsecond precision to avoid silent overwrite when two trim
        # tasks fire within the same second (e.g. rapid consecutive turns).
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        filepath = self._sessions_dir / f"{timestamp}.md"
        filepath.write_text(summary, encoding="utf-8")
        self._summary_cache_time = 0.0  # invalidate so next call re-reads

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        """Format message list into readable conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                lines.append(f"用户: {content}")
            elif role == "assistant" and content:
                lines.append(f"助手: {content}")
        return "\n".join(lines)
