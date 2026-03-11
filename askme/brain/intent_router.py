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
    GENERAL = "general"         # fallback → LLM


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

    def update_voice_triggers(self, triggers: dict[str, str]) -> None:
        """Replace the voice trigger map (called after skill reload)."""
        self._voice_triggers = triggers

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

        # 2. Built-in commands
        if stripped.lower() in self.BUILTIN_COMMANDS:
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

    def _match_voice_trigger(self, text: str) -> str | None:
        """Find the best matching voice trigger in the text.

        Uses substring matching: if any trigger phrase appears in the text,
        it's considered a match. Longer triggers are checked first to avoid
        false positives from short substrings.

        Triggers shorter than MIN_TRIGGER_LENGTH are skipped to prevent
        single-character Chinese triggers from matching common suffixes.
        """
        if not self._voice_triggers:
            return None

        text_lower = text.lower()

        # Sort triggers by length (longest first) for greedy matching
        sorted_triggers = sorted(
            self._voice_triggers.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

        for trigger_phrase, skill_name in sorted_triggers:
            if len(trigger_phrase) < self.MIN_TRIGGER_LENGTH:
                continue
            if trigger_phrase.lower() in text_lower:
                return skill_name

        return None
