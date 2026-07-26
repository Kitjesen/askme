"""
Intent router for askme.

Processes user input through a priority pipeline:
  1. Emergency stop detection (hardcoded, zero-latency, no LLM)
  2. Quick replies and built-in commands (no LLM)
  3. Voice trigger matching (keyword → skill, no LLM)
  4. General fallback for downstream LLM/tool handling

This ensures safety-critical commands are always handled instantly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from askme.robot_interaction.routing.fast_voice_intents import (
    FastVoiceIntentKind,
    match_fast_voice_intent,
    normalize_fast_voice_text,
)
from askme.robot_interaction.routing_policy import DEFAULT_ROUTING_POLICY, RoutingPolicy
from askme.robot_interaction.scenario_intents import classify_scenario_intent

logger = logging.getLogger(__name__)


class IntentType(Enum):
    ESTOP = "estop"
    VOICE_TRIGGER = "voice_trigger"
    COMMAND = "command"          # /clear, /quit, /history etc.
    QUICK_REPLY = "quick_reply"  # simple greetings — skip LLM, instant response
    GENERAL = "general"         # fallback → LLM

# Backward-compatible aliases for legacy imports/tests that reached into this
# module. New code should prefer RoutingPolicy.
_ESTOP_KEYWORDS = DEFAULT_ROUTING_POLICY.estop_keywords
_QUICK_REPLIES = DEFAULT_ROUTING_POLICY.quick_replies


@dataclass
class Intent:
    type: IntentType
    skill_name: str | None = None
    command: str | None = None
    raw_text: str = ""
    reply_text: str | None = None
    trigger_phrase: str | None = None
    reason: str | None = None
    scenario_id: str | None = None
    confidence: float | None = None
    route_evidence: dict[str, Any] | None = None
    cached_audio_key: str | None = None
    preface_text: str | None = None
    preface_audio_key: str | None = None
    fast_path: bool = False


class IntentRouter:
    """Route user input to the correct handler with safety-first priority."""

    # Camera questions must reach TurnExecutor so it can capture the current
    # LingTu frame and ask the configured vision model.  Keep this check ahead
    # of legacy voice skills such as environment_report.
    _VISUAL_QUERY_MARKERS = (
        "看见",
        "看到",
        "看看周围",
        "看一下环境",
        "描述环境",
        "前面有什么",
        "周围有什么",
        "周围看到",
        "摄像头",
        "相机",
        "图像",
        "画面",
        "图片",
        "眼前有什么",
    )

    BUILTIN_COMMANDS = DEFAULT_ROUTING_POLICY.builtin_commands
    MIN_TRIGGER_LENGTH = DEFAULT_ROUTING_POLICY.min_trigger_length
    _NEGATION_PREFIXES = DEFAULT_ROUTING_POLICY.negation_prefixes
    _QUESTION_SUFFIXES = DEFAULT_ROUTING_POLICY.question_suffixes
    _QUESTION_SAFE_SKILLS = DEFAULT_ROUTING_POLICY.question_safe_skills

    def __init__(
        self,
        safety_checker: Any | None = None,
        voice_triggers: dict[str, str] | None = None,
        policy: RoutingPolicy | None = None,
    ) -> None:
        """
        Args:
            safety_checker: A SafetyChecker instance with is_estop_command().
            voice_triggers: Mapping of trigger phrase → skill name.
            policy: Deterministic routing policy values.
        """
        self._safety = safety_checker
        self._policy = policy or DEFAULT_ROUTING_POLICY
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
          2. Quick replies → IntentType.QUICK_REPLY
          3. Built-in commands (/quit, /clear, etc.) → IntentType.COMMAND
          4. Voice trigger match → IntentType.VOICE_TRIGGER
          5. Everything else → IntentType.GENERAL (sent downstream)
        """
        stripped = text.strip()

        estop_reason = self._estop_reason(stripped)
        if estop_reason:
            logger.critical("E-STOP detected in text: %s", stripped)
            return Intent(
                type=IntentType.ESTOP,
                raw_text=stripped,
                reason=estop_reason,
            )

        # 2. Quick replies — simple greetings, skip LLM entirely
        fast_intent = match_fast_voice_intent(
            stripped,
            quick_replies=self._policy.quick_replies,
            estop_keywords=self._policy.estop_keywords,
        )
        if (
            fast_intent is not None
            and fast_intent.kind is FastVoiceIntentKind.READ_ONLY_SKILL
        ):
            logger.info(
                "Read-only fast voice intent: '%s' -> skill '%s'",
                stripped,
                fast_intent.skill_name,
            )
            return Intent(
                type=IntentType.VOICE_TRIGGER,
                skill_name=fast_intent.skill_name,
                raw_text=stripped,
                trigger_phrase=stripped,
                reason="read_only_fast_path",
                preface_text=fast_intent.preface_text,
                preface_audio_key=fast_intent.cache_key,
                fast_path=True,
            )

        quick = self._policy.quick_replies.get(stripped)
        if (
            quick is None
            and fast_intent is not None
            and fast_intent.kind is FastVoiceIntentKind.QUICK_REPLY
        ):
            quick = fast_intent.reply_text
        if quick:
            logger.info("Quick reply: '%s' → '%s'", stripped, quick)
            return Intent(
                type=IntentType.QUICK_REPLY,
                raw_text=stripped,
                reply_text=quick,
                # Compatibility: older loops read the quick reply from skill_name.
                skill_name=quick,
                reason="quick_reply",
                cached_audio_key=(
                    fast_intent.cache_key if fast_intent is not None else None
                ),
                fast_path=fast_intent is not None,
            )

        # 3. Built-in commands
        command = stripped.lower()
        if command in self._policy.builtin_commands:
            return Intent(
                type=IntentType.COMMAND,
                command=command,
                raw_text=stripped,
                reason="builtin_command",
            )

        # Route visual questions to the normal pipeline.  This intentionally
        # precedes voice-trigger matching so "看看周围" does not invoke the
        # legacy environment_report skill and its long, non-visual response.
        if any(marker in stripped for marker in self._VISUAL_QUERY_MARKERS):
            logger.info("Visual query routed to camera pipeline: '%s'", stripped)
            return Intent(
                type=IntentType.GENERAL,
                raw_text=stripped,
                reason="visual_query",
            )

        # 4. Voice trigger matching (substring match)
        trigger_match = self._match_voice_trigger(stripped)
        if trigger_match:
            matched_skill, trigger_phrase = trigger_match
            logger.info(
                "Voice trigger matched: '%s' → skill '%s'",
                stripped,
                matched_skill,
            )
            return Intent(
                type=IntentType.VOICE_TRIGGER,
                skill_name=matched_skill,
                raw_text=stripped,
                trigger_phrase=trigger_phrase,
                reason="voice_trigger",
            )

        # 5. Product scenario intent matching before LLM fallback.
        scenario_match = self._match_scenario_intent(stripped)
        if scenario_match is not None:
            logger.info(
                "Scenario intent matched: %s -> skill '%s'",
                scenario_match.rule_id,
                scenario_match.skill_name,
            )
            return Intent(
                type=IntentType.VOICE_TRIGGER,
                skill_name=scenario_match.skill_name,
                raw_text=stripped,
                trigger_phrase=",".join(scenario_match.matched_terms),
                reason="scenario_intent",
                scenario_id=scenario_match.scenario_id,
                confidence=scenario_match.confidence,
                route_evidence={
                    "rule_id": scenario_match.rule_id,
                    "matched_terms": list(scenario_match.matched_terms),
                    "risk_level": scenario_match.risk_level,
                    "evidence": scenario_match.evidence,
                },
            )

        return Intent(
            type=IntentType.GENERAL,
            raw_text=stripped,
            reason="empty_input" if not stripped else "general_fallback",
        )

    def _estop_reason(self, text: str) -> str | None:
        """Return the ESTOP source when text is a hard-stop command."""
        normalized = normalize_fast_voice_text(text)
        if normalized and any(
            normalize_fast_voice_text(keyword) == normalized
            for keyword in self._policy.estop_keywords
        ):
            return "estop_keyword"

        checker = self._safety
        if checker is None:
            return None
        try:
            if checker.is_estop_command(text):
                return "safety_checker"
        except Exception:
            logger.exception("SafetyChecker failed during ESTOP routing")
        return None

    def _match_scenario_intent(self, text: str):
        """Return a deterministic product-scenario route when one is safe enough."""

        decision = classify_scenario_intent(
            text,
            available_skills=set(self._voice_triggers.values()),
        )
        if decision is None:
            return None
        if (
            decision.risk_level != "visitor_service"
            and self._question_context_blocks_skill(text, decision.skill_name)
        ):
            return None
        return decision

    def _is_negated(self, text: str, trigger_pos: int) -> bool:
        """Return True if the trigger at trigger_pos is preceded by a negation word."""
        prefix = text[max(0, trigger_pos - 4) : trigger_pos]
        return any(prefix.endswith(neg) for neg in self._policy.negation_prefixes)

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
        return any(stripped.endswith(q) for q in self._policy.question_suffixes)

    def _question_context_blocks_skill(self, text: str, skill_name: str) -> bool:
        """Return whether question punctuation should suppress this skill."""
        return (
            self._is_question_context(text)
            and skill_name not in self._policy.question_safe_skills
        )

    def _match_voice_trigger(self, text: str) -> tuple[str, str] | None:
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
            if len(trigger_phrase) < self._policy.min_trigger_length:
                continue
            pos = text_lower.find(trigger_phrase.lower())
            if (
                pos >= 0
                and not self._is_negated(text_lower, pos)
                and not self._question_context_blocks_skill(text_lower, skill_name)
            ):
                return skill_name, trigger_phrase

        return None
