"""
Intent router for askme.

Processes user input through a priority pipeline:
  1. Emergency stop detection (hardcoded, zero-latency, no LLM)
  2. Voice trigger matching (keyword → skill, no LLM)
  3. LLM-based intent recognition with tool calling

This ensures safety-critical commands are always handled instantly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IntentType(Enum):
    ESTOP = "estop"
    VOICE_TRIGGER = "voice_trigger"
    COMMAND = "command"          # /clear, /quit, /history etc.
    QUICK_REPLY = "quick_reply"  # simple greetings — skip LLM, instant response
    GENERAL = "general"         # fallback → LLM

# Instant replies for simple greetings — no LLM needed, <0.5s response
_QUICK_REPLIES: dict[str, str] = {
    "你好": "你好，有什么需要帮忙的？",
    "谢谢": "不客气。",
    "谢谢你": "不客气，随时叫我。",
    "再见": "再见，有事随时叫我。",
    "拜拜": "拜拜。",
    "在吗": "在的，有什么事？",
    "你在吗": "在的，说吧。",
    "嗯": "嗯，我在听。",
    "好的": "好的。",
}


@dataclass
class Intent:
    type: IntentType
    skill_name: str | None = None
    command: str | None = None
    raw_text: str = ""


class IntentRouter:
    """Route user input to the correct handler with safety-first priority."""

    # Built-in commands (handled before any AI processing)
    BUILTIN_COMMANDS = {
        "/quit", "/exit", "exit", "quit",
        "/clear", "/history", "/help", "/skills",
    }

    def __init__(
        self,
        safety_checker: Any | None = None,
        voice_triggers: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            safety_checker: A SafetyChecker instance with is_estop_command().
            voice_triggers: Mapping of trigger phrase → skill name.
        """
        self._safety = safety_checker
        self._voice_triggers: dict[str, str] = voice_triggers or {}
        self._sorted_triggers: list[tuple[str, str]] = self._build_sorted_triggers()

    def _build_sorted_triggers(self) -> list[tuple[str, str]]:
        """Return triggers sorted longest-first (cached; rebuild on update)."""
        return sorted(
            self._voice_triggers.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

    def update_voice_triggers(self, triggers: dict[str, str]) -> None:
        """Replace the voice trigger map (called after skill reload)."""
        self._voice_triggers = triggers
        self._sorted_triggers = self._build_sorted_triggers()

    def route(self, text: str) -> Intent:
        """Determine the intent for a given user input.

        Priority order:
          1. Emergency stop keywords → IntentType.ESTOP
          2. Built-in commands (/quit, /clear, etc.) → IntentType.COMMAND
          3. Voice trigger match → IntentType.VOICE_TRIGGER
          4. Everything else → IntentType.GENERAL (sent to LLM)
        """
        stripped = text.strip()

        # 1. Emergency stop — HIGHEST PRIORITY, zero delay
        if self._safety and self._safety.is_estop_command(stripped):
            logger.critical("E-STOP detected in text: %s", stripped)
            return Intent(
                type=IntentType.ESTOP,
                raw_text=stripped,
            )

        # 2. Quick replies — simple greetings, skip LLM entirely
        quick = _QUICK_REPLIES.get(stripped)
        if quick:
            logger.info("Quick reply: '%s' → '%s'", stripped, quick)
            return Intent(
                type=IntentType.QUICK_REPLY,
                raw_text=stripped,
                skill_name=quick,  # reuse skill_name field to carry the reply text
            )

        # 3. Built-in commands
            return Intent(
                type=IntentType.COMMAND,
                command=stripped.lower(),
                raw_text=stripped,
            )

        # 3. Voice trigger matching (substring match)
        matched_skill = self._match_voice_trigger(stripped)
        if matched_skill:
            logger.info("Voice trigger matched: '%s' → skill '%s'", stripped, matched_skill)
            return Intent(
                type=IntentType.VOICE_TRIGGER,
                skill_name=matched_skill,
                raw_text=stripped,
            )

        # 4. General → LLM handles it
        return Intent(
            type=IntentType.GENERAL,
            raw_text=stripped,
        )

    # Minimum trigger length to avoid false positives from short substrings
    # (e.g. "去" matching "进去", "出去", "回去")
    MIN_TRIGGER_LENGTH = 2

    # Chinese negation words that immediately precede a trigger phrase.
    # Ordered longest-first so "不要" matches before "不".
    _NEGATION_PREFIXES: tuple[str, ...] = (
        "不要", "不用", "不能", "不想", "不让", "没有", "别再",
        "不", "别", "没",
    )

    # Chinese question-ending particles. When any of these appear at the end
    # of the utterance (within 2 characters of the end), or immediately after
    # the trigger phrase, the input is treated as a question, not a command.
    _QUESTION_SUFFIXES: tuple[str, ...] = ("吗", "么", "呢", "嘛")

    def _is_negated(self, text: str, trigger_pos: int) -> bool:
        """Return True if the trigger at trigger_pos is preceded by a negation word."""
        prefix = text[max(0, trigger_pos - 4) : trigger_pos]
        return any(prefix.endswith(neg) for neg in self._NEGATION_PREFIXES)

    def _is_question_context(self, text: str) -> bool:
        """Return True if the utterance is a question, not a command.

        Checks:
        1. Text ends with a Chinese question particle (吗, 么, 呢, 嘛).
        2. Text ends with a Chinese/ASCII question mark (？ ?).

        Deliberately conservative: only triggers on unambiguous question
        endings, not on internal question words which might be in commands.
        """
        stripped = text.rstrip()
        if not stripped:
            return False
        # Ends with ASCII/fullwidth question mark
        if stripped.endswith("?") or stripped.endswith("？"):
            return True
        # Ends with question particle
        return any(stripped.endswith(q) for q in self._QUESTION_SUFFIXES)

    def _match_voice_trigger(self, text: str) -> str | None:
        """Find the best matching voice trigger, skipping negated and question occurrences.

        Uses substring matching: if any trigger phrase appears in the text,
        it's considered a match. Longer triggers are checked first to avoid
        false positives from short substrings.

        Triggers shorter than MIN_TRIGGER_LENGTH are skipped to prevent
        single-character Chinese triggers from matching common suffixes.

        A trigger is skipped if it is immediately preceded by a Chinese
        negation word (不, 别, 不要, etc.) to prevent "不要停" from firing
        the stop_speaking skill.

        A trigger is also skipped if the whole utterance ends with a question
        particle (吗, 么, 呢, 嘛) or a question mark (? ？), because the user
        is asking about the feature rather than invoking it — e.g. "导航会失
        败吗" should not fire the navigate skill.
        """
        if not self._voice_triggers:
            return None

        text_lower = text.lower()

        for trigger_phrase, skill_name in self._sorted_triggers:
            if len(trigger_phrase) < self.MIN_TRIGGER_LENGTH:
                continue
            pos = text_lower.find(trigger_phrase.lower())
            if pos >= 0 and not self._is_negated(text_lower, pos) and not self._is_question_context(text_lower):
                return skill_name

        return None
